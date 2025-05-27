import std;
import shadertoy;

#define LOD_OFFSET 1.0
#define PI 3.14159265

#define texel(a, p) texelFetch(a, ivec2(p), 0)
#define ch0 0
#define ch1 1
#define ch2 2
#define ch3 3

#define REST_DENSITY 15.0

#define ITERS 2u
#define SLOWDOWN 1

#define ITERS_PER_FRAME 8

#define BITONIC_STEPS 2

static const int POINTS_SQRT = 256;
static const int SORT_WG_SIZE = 128;
static const int POINTS_COUNT = POINTS_SQRT * POINTS_SQRT;

// Workgroup size 
static const int WG_X = 16;
static const int WG_Y = 16;
static const int WORKGROUPS_X = POINTS_SQRT / WG_X;
static const int WORKGROUPS_Y = POINTS_SQRT / WG_Y;

//point data
[StorageBuffer(POINTS_COUNT)]
RWStructuredBuffer<vec2> point_pos0;
[StorageBuffer(POINTS_COUNT)]
RWStructuredBuffer<vec2> point_vel0;
[StorageBuffer(POINTS_COUNT)]
RWStructuredBuffer<vec2> point_pos1;
[StorageBuffer(POINTS_COUNT)]
RWStructuredBuffer<vec2> point_vel1;

[StorageBuffer(POINTS_COUNT)]
RWStructuredBuffer<float> point_dens;

//sorted grid positions
[StorageBuffer(POINTS_COUNT)]
RWStructuredBuffer<uint> grid_pos;

bool inGrid(uvec2 p) {
    return all(p >= 0) && all(p < POINTS_SQRT);
}

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

float EOS(float rho)
{
    return (rho / REST_DENSITY) - 1.0;
}

[WorkgroupCount(WORKGROUPS_X, WORKGROUPS_Y, 1)]
[shader("compute")] 
[numthreads(WG_X, WG_Y, 1)]
void PointDispatch(uint3 id : SV_DispatchThreadID) { 
    vec2 p = vec2(id.xy);
    float time = 1.5 + custom.Speed*iTime;
    uint pixID = xy2i(id.xy);

    if(iFrame < 8) {
        grid_pos[pixID] = pixID;
        float r = rand(pixID * 4923534u);
        vec2 s0 = ((p + 0.*rand2(pixID))/POINTS_SQRT - 0.5) * vec2(3.0, 3.0);
        point_pos0[pixID] = s0.xy;
        point_vel0[pixID] = 0.0005*vec2(s0.y, -s0.x);
        return;
    }

    vec2 pos = point_pos1[pixID];
    vec2 vel = point_vel1[pixID];
    float dens = point_dens[pixID];

    float dist = length(pos);
    vec2 force = -1e-6*normalize(pos+1e-6);

    for(int i = -8; i <= 8; i++)
    for(int j = -8; j <= 8; j++)
    {
        if(i == 0 && j == 0) continue;
        vec2 loc = id.xy + vec2(i,j);
        if(!inGrid(loc)) continue;
        uint pixID0 = xy2i(loc);
        vec2 pos0 = point_pos1[pixID0];
        vec2 vel0 = point_vel1[pixID0];
        float dens0 = point_dens[pixID0];


        float P = 0.5 * (EOS(dens0) + EOS(dens));
        vec2 dx = pos - pos0;
        vec2 dv = vel - vel0;
        float dist = length(dx);
        float weight = exp(-sqr(dist/0.01));
        force += -dot(dv, dx) * dx * 0.5 * weight / max(1e-3, dist * dist);
        force += 0.01* dx * P * weight;
        force += 0.1 * dx  * exp(-sqr(dist/0.003));
    }

    force = clamp(force, -0.1, 0.1);
    vel += custom.Speed * force;
    pos += custom.Speed * vel;

    vel = clamp(vel, -0.1, 0.1);
    point_pos0[pixID] = pos;
    point_vel0[pixID] = vel;
}

static const uint BLOCKSIZE = POINTS_SQRT;

groupshared uint2 temp[BLOCKSIZE];
groupshared uint particleIDs[BLOCKSIZE];

void LocalSort(uint GI, uint j, uint k)
{
    //bitonic sort magic
    uint mask = (k==j) ? (2*k - 1) : k;
    uint e = GI%k + 2*k*(GI/k);

    //get local pair of elements
    uint2 id = uint2(e, e ^ mask); 
    uint2 A = temp[id.x];
    uint2 B = temp[id.y];

    //swap if unsorted
    if(A.x > B.x)
    {
        temp[id.x] = B;
        temp[id.y] = A;
    }
}

static const uint sign_bit = ~0x7FFFFFFF;

uint map_float_to_uint(float x) {
    uint ux = asuint(x);
    uint mask = ((ux >> 31) == 1u) ? ~0u : sign_bit;
    return ux ^ mask;
}

float map_uint_to_float(uint x) {
    uint mask = ((x >> 31) == 1u) ? ~0u : sign_bit;
    return asfloat(x ^ mask);
}

uint map_int_to_uint(int x) {
    return asuint(x) ^ sign_bit;
}

int map_uint_to_int(uint x) {
    return asint(x ^ sign_bit);
}

uint linearIndex(uint2 id) {
    return id.x + id.y * POINTS_SQRT;
}

uint getAxisIndex(uint id, uint group, uint axis) {
    uint2 idx;
    idx[axis] = id;
    idx[1 - axis] = group;
    return linearIndex(idx);
}



[DispatchCount(BITONIC_STEPS)]
[WorkgroupCount(POINTS_SQRT, 1, 1)]
[shader("compute")]
[numthreads(SORT_WG_SIZE, 1, 1)]
void SortAxes(uint3 gtid : SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    int group = gid.x;
    uint index = gtid.x;
    uint axis = dispatch.id%2;

    uint M = POINTS_SQRT / SORT_WG_SIZE;

    for (uint i = 0u; i < M; i++) {
        uint rowIndex = index + i * SORT_WG_SIZE;
        uint pid = grid_pos[getAxisIndex(rowIndex, group, axis)];
        particleIDs[rowIndex] = pid;
        float sortvalue = point_pos0[pid][axis];
        temp[rowIndex] = uint2(map_float_to_uint(sortvalue), rowIndex);
    }

    GroupMemoryBarrierWithGroupSync();
    
    for (uint j = 1; j < BLOCKSIZE; j <<= 1)
    for (uint k = j; k > 0; k>>= 1)
    {
        for (uint i = 0u; i < M/2; i++) {
            LocalSort(index + i * SORT_WG_SIZE, j, k);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    //save block into global memory
    for (uint i = 0u; i < M; i++) {
        uint rowIndex = index + i * SORT_WG_SIZE;
        grid_pos[getAxisIndex(rowIndex, group, axis)] = particleIDs[temp[rowIndex].y];
    }
}

#define SORT_BLOCK 2
#define SORT_ELEMS (SORT_BLOCK*SORT_BLOCK)

struct BLOCK {
    vec2 pos[SORT_ELEMS];
    vec2 point_pos[SORT_ELEMS];
    uint point_ids[SORT_ELEMS];
};

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

uvec2 i2pix(inout uint idx, inout uvec2 n2)
{
    uint y = idx / n2.x;
    uvec2 pix = uvec2(idx - y*n2.x, y);
    return pix;
}

void flip(inout uint idx, inout uvec2 n2)
{
    uvec2 pix = i2pix(idx, n2);
    idx = pix.y + pix.x * n2.y;  
    n2 = n2.yx;
}

uint shuffle2d(uint idx, uvec2 n2, uint seed, uint group)
{
    uint n = n2.x * n2.y;
    uint idx0 = idx;
    uvec2 pix0 = i2pix(idx0, n2);
    idx = shuffle(idx, n, seed, group);
    flip(idx, n2);
    idx = shuffle(idx, n, seed, group);
    flip(idx, n2);
    uvec2 pix1 = i2pix(idx, n2);
    if(distance(vec2(pix0), vec2(pix0)) > 32.0 * float(group)) idx = idx0;
    return idx;
}

uint unshuffle2d(uint idx, uvec2 n2, uint seed, uint group)
{
    uint n = n2.x * n2.y;
    uint idx0 = idx;
    uvec2 pix1 = i2pix(idx0, n2);
    flip(idx, n2);
    idx = unshuffle(idx, n, seed, group);
    flip(idx, n2);
    idx = unshuffle(idx, n, seed, group);
    uvec2 pix0 = i2pix(idx, n2);
    if(distance(vec2(pix0), vec2(pix0)) > 32.0 * float(group)) idx = idx0;
    return idx;
}

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
[WorkgroupCount(WORKGROUPS_X/2, WORKGROUPS_Y/2, 1)]
[shader("compute")] 
[numthreads(WG_X, WG_Y, 1)]
void SortStep(uint3 id : SV_DispatchThreadID) { 
    uint seed = uint(iFrame)*ITERS_PER_FRAME + dispatch.id;
    uint group = 1u;
    //Load random block of values
    BLOCK data;   
    for(int i = 0; i < SORT_BLOCK; i++)
    for(int j = 0; j < SORT_BLOCK; j++)
    {
        uvec2 pix2 = id.xy*SORT_BLOCK + uvec2(i,j);
        uint shuffled = shuffle2d(xy2i(pix2), POINTS_SQRT, seed, group);
        uint ij = uint(i + j * SORT_BLOCK);
        uint pid = grid_pos[shuffled];
        vec2 val = point_pos0[pid];
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
    vec2 pos = point_pos0[point];
    vec2 vel = point_vel0[point];
    pass_out[int3(id.xy, 2)] = pos.xyxy;
    point_pos1[xy2i(id.xy)] = pos;
    point_vel1[xy2i(id.xy)] = vel;
    grid_pos[xy2i(id.xy)] = xy2i(id.xy);
}

[WorkgroupCount(WORKGROUPS_X, WORKGROUPS_Y, 1)]
[shader("compute")] 
[numthreads(WG_X, WG_Y, 1)]
void ComputeDensity(uint3 id : SV_DispatchThreadID) { 
    vec2 p = vec2(id.xy);
    float time = 1.5 + custom.Speed*iTime;
    uint pixID = xy2i(id.xy);

    vec2 pos = point_pos1[pixID];
    float density = 0.0;
    for(int i = -8; i <= 8; i++)
    for(int j = -8; j <= 8; j++)
    {
        vec2 loc = id.xy + vec2(i,j);
        if(!inGrid(loc)) continue;
        uint pixID0 = xy2i(loc);
        vec2 pos0 = point_pos1[pixID0];
        vec2 dx = pos - pos0;
        float dist = length(dx);
        float weight = exp(-sqr(dist/0.01));
        density += weight;
    }

    point_dens[pixID] = density;
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
        if(window.x <= float(4)) break;
        window = max(vec2(4), window*0.5);
    }
    return best.xy;
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
        if(!inGrid(loc0)) continue;
        dx /= float(radius);
        vec2 val = texel(ch2, loc0).xy;
        float weight = exp(- 2.0*(dot(dx,dx)));
        sum += weight;
        grad.xy += dx * val.x * weight;
        grad.zw += dx * val.y * weight;
    }
    return grad / sum;
}

float minc(vec2 x)
{
    return min(x.x, x.y);
}

void mainImage( out vec4 col, in vec2 p )
{    
    p = floor(p);
    
    if(keyDown(80)){
        float line = iResolution.x*1.;
        if(iMouse.z > 0.0) line = iMouse.x;
        vec2 gpos = POINTS_SQRT*p/iResolution.y;
        vec2 val = texel(ch2, gpos).xy;
        vec4 color = vec4(val.xy, 0.0, 0.0);
        vec4 grad = vec4(1.25) / (POINTS_SQRT * length(softGrad(gpos, 1)));
        grad = 0.5/ grad;
        color = (p.x < line) ? color : grad;
        col = color;
        return;
    }
    
    float scale = 1.0;
    vec2 mouseuv =  4.5*(iMouse.xy / iResolution.x  - vec2(0.5, 0.25));
    vec2 uv = 4.5*(p.xy / iResolution.x - vec2(0.5, 0.25));
    
    if(iMouse.z > 0.0)
    {
        uv = 1.0*(p.xy / iResolution.x - 0.5) + mouseuv;
    }
    
    vec2 closest = findClosestPoint(uv);
        
    //you can estimate the density of the points
    //by computing the inverse gradient of their sorted positions, very cool
    //float d0 = avgDist(closest, 8);
    //float density = 4.0 / (iResolution.x * d0);
    
    col = vec4(0.0);
    for(int i = -4; i <= 4; i++)
    for(int j = -4; j <= 4; j++)
    {
        vec2 loc0 = closest.xy + vec2(i,j);
        if(!inGrid(loc0)) continue;
        vec2 val = texel(ch2, loc0).xy;
        float dens = 0.01*point_dens[xy2i(loc0)];
        float dist = distance(uv, val);
        float dist2b = minc(min(loc0, POINTS_SQRT - 1.0 - loc0));
        vec2 norm =floor(16.0*loc0/POINTS_SQRT)/16.0;
        col.xyz += exp(-sqr(0.25*dist*iResolution.x)) * vec3(sin(5.0*norm)*0.5 + 0.5, 1.0);
    }

    //col.xyz *= density;
    
    col.xyz = tanh(pow(abs(col.xyz), vec3(1.0)));
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