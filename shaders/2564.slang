import std;

static const uint NUM_POINTS = 1024 * 16 * 16;
static const float3 CAMERA_CENTER = float3(0,0,25);

static const float PI = radians(180);

uint murmur3(uint h) {
	h ^= h >> 16;
	h *= 0x85ebca6b;
	h ^= h >> 13;
	h *= 0xc2b2ae35;
	h ^= h >> 16;
    return h;
}

struct RNG {
    uint state = 0xDEADBEEF;

    static uint xorshift32(uint x) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        return x;
    }

    [mutating]
    void seed(uint s) {
        state = s ^ 0xA533C1F1;
    }

    [mutating]
    uint next_raw() {
        state = xorshift32(state);
        return state;
    }

    [mutating]
    float next() {
        return next_raw() / float(0xFFFFFFFF);
    }   

    [mutating]
    vector<float, N> next<int N>() {
        vector<float, N> result;
        for (var i = 0; i < N; i++)
            result[i] = next();
        return result;
    }
};

struct Point {
    float3 loc; // Note: cannot use name pos as otherwise it will cause mouse.pos to be assigned to mouse.pos_1 instead of mouse.pos_0 -> breaks camera
    float3 vel;
    uint id;

    __init(uint hash) {
        RNG rng;
        rng.seed(hash);
        loc = rng.next<3>() * 2 - 1;
        loc.z = 0;
        vel = rng.next<3>();
        id = hash;
    }

    [mutating]
    void tick(float dt) {
        // Lorentz update to velocity
        // x' = -σx + σy
        // y' =  Rx − y − xz
        // z' = -Bz + xy

        float sigma = 10;
        float B = 8 / 3;
        float R = 28;
        vel = float3(
            (-loc.x + loc.y) * sigma,
            R * loc.x - loc.y - loc.x * loc.z,
            -B * loc.z + loc.x * loc.y
        );

        // Update position, explicit euler
        loc += vel * dt;
    }

    float3 color() {
        RNG rng;
        rng.seed(id);
        return rng.next<3>();
    }
};
[StorageBuffer(NUM_POINTS)]
RWStructuredBuffer<Point> points;


// Create a matrix for a symmetric perspective projection
// fov in deg, aspect=w/h, near/far = near/far plane distances
// Yoinked from https://github.com/shader-slang/slangpy/blob/main/src/sgl/math/matrix_math.h
float4x4 perspective(float fov_x, float aspect, float near, float far) {
    float tan_half_fovx = tan(radians(fov_x * 0.5));

    float4x4 m = 0;
    m[0][0] = 1.0 / (tan_half_fovx);
    m[1][1] = aspect / (tan_half_fovx);
    m[2][2] = far / (near - far);
    m[3][2] = -1.0;
    m[2][3] = -(far * near) / (far - near);
    return m;
}

float4x4 lookat(float3 target, float3 from, float3 up) {
    float3 fwd = normalize(target - from);
    float3 right = normalize(cross(fwd, up));
    float3 up = normalize(cross(right, fwd));
    float3x3 rot = float3x3(right, up, fwd);
    float3 t = mul(rot, -from);
    return float4x4(
        float4(rot[0], t.x),
        float4(rot[1], t.y),
        float4(rot[2], t.z),
        float4(0, 0, 0, 1)
    );
}

float4x4 getCamera() {
    uint width, height;
    screen.GetDimensions(width, height);
    float2 dim = float2(width, height);
    
    float fov_x = custom.cam_fov_x;
    float r = custom.cam_r;
    float ang_h = mouse.pos.x / dim.x * 2 * PI;
    float ang_v = (dim.y-mouse.pos.y) / dim.y * PI;

    float3 pos = float3(sin(ang_v)*cos(ang_h), sin(ang_v)*sin(ang_h), cos(ang_v)) * r;
    float4x4 view = lookat(CAMERA_CENTER, CAMERA_CENTER + pos, float3(0,0,1));
    float4x4 proj = perspective(fov_x, dim.x/dim.y, 0.001, 1000.0);
    return mul(proj, view);
}


[shader("compute")]
[numthreads(16, 16, 1)]
[DispatchOnce]
void init(uint3 id: SV_DispatchThreadID) {
    uint width, height;
    screen.GetDimensions(width, height);
    uint idx = id.x + id.y * width + id.z * width*height; 

    if (id.x >= width || id.y >= height || idx > NUM_POINTS)
        return;

    points[idx] = Point(murmur3(idx));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void update(uint3 id: SV_DispatchThreadID) {
    uint width, height;
    screen.GetDimensions(width, height);
    uint idx = id.x + id.y * width + id.z * width*height; 

    if (id.x >= width || id.y >= height || idx > NUM_POINTS)
        return;

    Point p = points[idx];
    p.tick(time.delta * exp2(-10*(1-custom.sim_speed)));
    points[idx] = p;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void clear(uint3 id : SV_DispatchThreadID)
{
    uint2 dim;
    screen.GetDimensions(dim.x, dim.y);
    if (any(id.xy >= dim.xy))
        return;
    screen[id.xy] = float4(0.002);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main_image(uint3 id : SV_DispatchThreadID)
{
    // Viewport resolution (in pixels)
    uint2 dim;
    screen.GetDimensions(dim.x, dim.y);

    uint idx = id.x + id.y * dim.x + id.z * dim.x*dim.y; 

    // Prevent overdraw for workgroups on the edge of the viewport
    if (any(id.xy >= dim.xy) || idx > NUM_POINTS)
        return;

    // Splat the point to the image
    Point p = points[idx];
    float4 ndc = mul(getCamera(), float4(p.loc, 1));
    if (ndc.w > 0)
        return;
    ndc /= ndc.w;
    uint2 px = uint2(int2(round((ndc.xy*.5 +.5) * dim)));
    if (any(px >= dim.xy))
        return;

    float3 col = p.color();
    screen[px] = float4(col, 1);
}
