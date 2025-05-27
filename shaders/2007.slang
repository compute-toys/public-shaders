import std;
import shadertoy;

#define LOD_OFFSET 1.0
#define PI 3.14159265

#define texel(a, p) texelFetch(a, ivec2(p), 0)
#define ch0 0
#define ch1 1
#define ch2 2
#define ch3 3

#define ITERS 4u
#define SLOWDOWN 1

#define ITERS_PER_FRAME 8

#define POINTS_SQRT 512
#define POINTS_COUNT (POINTS_SQRT * POINTS_SQRT)

// Workgroup size 
static const int WG_X = 16;
static const int WG_Y = 16;
static const int WORKGROUPS_X = POINTS_SQRT / WG_X;
static const int WORKGROUPS_Y = POINTS_SQRT / WG_Y;

//point data
[StorageBuffer(POINTS_COUNT)]
RWStructuredBuffer<vec2> point_pos;

[StorageBuffer(POINTS_COUNT)]
RWStructuredBuffer<vec2> point_grid_pos;

//sorted grid positions
[StorageBuffer(POINTS_COUNT)]
RWStructuredBuffer<uint> grid_pos;

[StorageBuffer(POINTS_COUNT)]
RWStructuredBuffer<vec2> sorted_point_pos;

uvec2 i2xy(uint id)
{
    return uvec2(id % POINTS_SQRT, id / POINTS_SQRT);
}

uint xy2i(uvec2 xy)
{
    return xy.x + xy.y*POINTS_SQRT;
}

uint pcg(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

uint mod2(uint idx, uint mask)
{
    return idx & mask;
}

uint xor_swap(uint idx, uint n, uint seed, uint group)
{
    uint mask = group - 1u;
    uint offset = mod2(seed, mask); //offset of the chunks
    uint idx0 = idx + offset;
    uint gidx = mod2(idx0, ~mask) - offset;
    uint xor_val = pcg(seed * 277803737u + gidx);
    uint xor_idx = mod2(idx0 ^ xor_val, mask);
    uint final = xor_idx + gidx;
    if(final >= n) return idx; //invalid range
    return final;
}

uint shuffle(uint idx, uint n, uint seed, uint group)
{
    uint group_size = 1u<<group;
    for(uint i = 0u; i < ITERS; i++)
    {
        idx = xor_swap(idx, n, seed + i, group_size);
    }
    return idx;
}

uint unshuffle(uint idx, uint n, uint seed, uint group)
{
    uint group_size = 1u<<group;
    for (uint i = ITERS; i-- > 0u; )
    {
        idx = xor_swap(idx, n, seed + i, group_size);
    }
    return idx;
}

void flip(inout uint idx, inout uvec2 n2)
{
    uint y = idx / n2.x;
    uvec2 pix = uvec2(idx - y*n2.x, y);
    idx = pix.y + pix.x * n2.y;  
    n2 = n2.yx;
}

uint shuffle2d(uint idx, uvec2 n2, uint seed, uint group)
{
    uint n = n2.x * n2.y;
    idx = shuffle(idx, n, seed, group);
    flip(idx, n2);
    idx = shuffle(idx, n, seed, group);
    flip(idx, n2);
    return idx;
}

uint unshuffle2d(uint idx, uvec2 n2, uint seed, uint group)
{
    uint n = n2.x * n2.y;
    flip(idx, n2);
    idx = unshuffle(idx, n, seed, group);
    flip(idx, n2);
    idx = unshuffle(idx, n, seed, group);
    return idx;
}

float randIO(inout uint seed)
{
    seed = pcg(seed);
    return float(seed) * (1.0/float(0xffffffffu));
}

vec2 rand2IO(inout uint seed)
{
    return vec2(randIO(seed),randIO(seed));
}

vec4 rand4IO(inout uint seed)
{
    return vec4(randIO(seed),randIO(seed),randIO(seed),randIO(seed));
}

float rand(uint seed)
{
    return randIO(seed);
}

vec2 rand2(uint seed)
{
    return rand2IO(seed);
}

vec4 rand4(uint seed)
{
    return rand4IO(seed);
}

float sqr(float x)
{
    return x*x;
}

[WorkgroupCount(WORKGROUPS_X, WORKGROUPS_Y, 1)]
[shader("compute")] 
[numthreads(WG_X, WG_Y, 1)]
void PointDispatch(uint3 id : SV_DispatchThreadID) { 
    vec2 p = vec2(id.xy);
    float time = 1.5 + custom.Speed*iTime;
    uvec2 pu = uvec2(p);
    uint pixID = pu.x + 4000u*pu.y;
    vec4 U;

    float r = rand(pixID * 4923534u);
    vec2 s0 = ((p + 0.*rand2(pixID))/POINTS_SQRT - 0.5) * vec2(3.0, 3.0);

    for (float n = .1; n < 0.5; n*=2.0)
    {
        s0 -= abs(sin(10.0*length(s0) + time + 2.0*s0 + 4.0*vec2(-0.3,0.5)*s0.yx + s0.yx * s0 * n * 5.))*custom.Str / n; 
    }

    for (int i = 0; i < 16; i++)
    {
        s0 += 0.05*s0 / (dot(s0, s0)+1.0);
    }

    vec2 s = s0;
    U = vec4(0.15*s.xy+vec2(0.33, 0.33), 1e-8, 1e-8); //comment do do colors of image instead of points
    
    point_pos[xy2i(id.xy)] = U.xy;

    if(iFrame < 16) {
        grid_pos[xy2i(id.xy)] = xy2i(id.xy);
    }
}

#define SORT_BLOCK 2
#define SORT_ELEMS (SORT_BLOCK*SORT_BLOCK)

struct BLOCK {
    vec2 pos[SORT_ELEMS];
    vec2 point_pos[SORT_ELEMS];
    uint point_ids[SORT_ELEMS];
};


float scoreAB(vec2 grid_pos, vec2 space_pos)
{
    return dot(grid_pos, space_pos); //maximizing alignment of point space with grid space
}

float bestPermutation(in vec2 pos[4], in vec2 point_pos[4], out ivec4 bestPerm)
{
    float best = -1e38;
    ivec4 bestP = ivec4(0,1,2,3);

    for(int a = 0; a < 4; ++a)
    for(int b = 0; b < 4; ++b) if(b != a)
    for(int c = 0; c < 4; ++c) if(c != a && c != b)
    for(int d = 0; d < 4; ++d) if(d != a && d != b && d != c)
    {
        ivec4 p = ivec4(a, b, c, d);

        float s =
            scoreAB(pos[0], point_pos[p.x]) +
            scoreAB(pos[1], point_pos[p.y]) +
            scoreAB(pos[2], point_pos[p.z]) +
            scoreAB(pos[3], point_pos[p.w]);

        if(s > best)
        {
            best  = s;
            bestP = p;
        }
    }

    bestPerm = bestP;
    return best;
}

[DispatchCount(ITERS_PER_FRAME)]
[WorkgroupCount(WORKGROUPS_X, WORKGROUPS_Y, 1)]
[shader("compute")] 
[numthreads(WG_X/2, WG_Y/2, 1)]
void SortStep(uint3 id : SV_DispatchThreadID) { 
    uint seed = uint(iFrame)*ITERS_PER_FRAME + dispatch.id;
    uint group = (dispatch.id < 3)?3u:1u;
    //Load random block of values
    BLOCK data;   
    for(int i = 0; i < SORT_BLOCK; i++)
    for(int j = 0; j < SORT_BLOCK; j++)
    {
        uvec2 pix2 = id.xy*SORT_BLOCK + uvec2(i,j);
        uint shuffled = shuffle2d(xy2i(pix2), POINTS_SQRT, seed, group);
        uint ij = uint(i + j * SORT_BLOCK);
        uint pid = grid_pos[shuffled];
        vec2 val = point_pos[pid];
        data.pos[ij] = i2xy(shuffled);
        data.point_ids[ij] = pid;
        data.point_pos[ij] = val;
    }
    
    //Find best permutation
    ivec4 bestPerm;
    float best = bestPermutation(data.pos, data.point_pos, bestPerm);

    //Store values
    for(uint i = 0u; i < SORT_ELEMS; i++) {
        grid_pos[xy2i(uvec2(data.pos[i]))] = data.point_ids[bestPerm[i]];
    }
}

[WorkgroupCount(WORKGROUPS_X, WORKGROUPS_Y, 1)]
[shader("compute")] 
[numthreads(WG_X, WG_Y, 1)]
void ReorderPoints(uint3 id : SV_DispatchThreadID) {  
    uint point = grid_pos[xy2i(id.xy)];
    pass_out[int3(id.xy, 2)] = point_pos[point].xyxy;
}

vec3 searchClosestGrid(vec3 oldBest, vec2 target, vec2 window, int radius)
{
    vec3 newBest = oldBest;
    for(int i = -radius; i <= radius; i++)
    for(int j = -radius; j <= radius; j++)
    {
        vec2 loc0 = clamp(oldBest.xy + window * vec2(i,j) / float(radius), vec2(0.0), POINTS_SQRT - 1.0);
        vec2 val = texel(ch2, loc0).xy;
        float dist = distance(target, val);
        if(dist < newBest.z) {
            newBest = vec3(loc0, dist);
        }
    }
    return newBest;
}

#define MAX_ITERS 16
vec2 findClosestPoint(vec2 target)
{
    vec2 window = vec2(POINTS_SQRT * 0.5);
    vec3 best = vec3(window, 1e10);
    for(int i = 0; i < MAX_ITERS; i++)
    {
        int radius = 4;
        vec3 newBest = searchClosestGrid(best, target, window, radius);
        best = newBest;
        if(window.x <= float(8)) break;
        window = max(vec2(8), window*0.4);
    }
    return best.xy;
}

float avgDist(vec2 pos, int radius)
{
    vec2 sum = vec2(0.0);
    vec2 center = texel(ch2, pos).xy;
    for(int i = -radius; i <= radius; i++)
    for(int j = -radius; j <= radius; j++)
    {
        vec2 dx = vec2(i,j);
        vec2 loc0 = pos + dx;
        dx /= float(radius);
        vec2 val = texel(ch2, loc0).xy;
        float weight = exp(-2.0*dot(dx,dx));
        sum += weight * vec2(distance(center, val), 1.0);
    }
    return sum.x / sum.y;
}

vec4 softGrad(vec2 pos, int radius)
{
    vec4 grad = vec4(0.0);
    float sum = 0.0;
    for(int i = -radius; i <= radius; i++)
    for(int j = -radius; j <= radius; j++)
    {
        vec2 dx = vec2(i,j);
        vec2 loc0 = pos + dx;
        dx /= float(radius);
        vec2 val = texel(ch2, loc0).xy;
        float weight = exp(- 2.0*(dot(dx,dx)));
        sum += weight;
        grad.xy += dx * val.x * weight;
        grad.zw += dx * val.y * weight;
    }
    return grad / sum;
}

void mainImage( out vec4 col, in vec2 p )
{    
    p = floor(p);
    
    if(keyDown(80)){
        float line = iResolution.x*0.0;
        if(iMouse.z > 0.0) line = iMouse.x;
        vec2 val = texel(ch2, p).xy;
        vec4 color = vec4(sin(100.0*val.xy) * 0.5 + 0.5, 0.0, 0.0);
        vec4 grad = vec4(0.25) / (iResolution.x * length(softGrad(p, 1)));
        grad = 0.1/ grad;
        color = (p.x < line) ? color : grad;
        col = color;
        return;
    }
    
    float scale = 1.0;
    vec2 mouseuv = 1.*(iMouse.xy / iResolution.x - vec2(0.0, 0.0));
    vec2 uv = 1.*(p.xy / iResolution.x - vec2(0.0, 0.0));
    
    if(iMouse.z > 0.0)
    {
        uv = 0.25*(p.xy / iResolution.x - 0.5) + mouseuv;
    }
    
    vec2 closest = findClosestPoint(uv);
        
    //you can estimate the density of the points
    //by computing the inverse gradient of their sorted positions, very cool
    //float d0 = avgDist(closest, 8);
    //float density = 4.0 / (iResolution.x * d0);
    
    col = vec4(0.0);
    for(int i = -8; i <= 8; i++)
    for(int j = -8; j <= 8; j++)
    {
        vec2 loc0 = closest.xy + vec2(i,j);
        vec2 val = texel(ch2, loc0).xy;
        float dist = distance(uv, val);
        col.xyz += exp(-sqr(1.25*dist*iResolution.x)) * vec3(closest / iResolution.xy, 1.0);
    }

    //col.xyz *= density;
    
    col.xyz = tanh(pow(0.5*abs(col.xyz), vec3(1.0)));
    col.w = 1.0;

}


#define ComputeFragColor(bfunc, id) \
    if(any(id.xy >= SCREEN_SIZE)) return; \
    vec4 fragColor; \
    bfunc(fragColor, vec2(id.xy) + 0.5);

[shader("compute")] [numthreads(WG_X, WG_Y, 1)]
void Image(uint3 id : SV_DispatchThreadID) {
    ComputeFragColor(mainImage, id)
    screen[ivec2(id.x, SCREEN_SIZE.y - 1 - id.y)] = pow(fragColor, 2.2);
}