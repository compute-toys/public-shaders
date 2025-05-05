import std;

static const int SIZE = 1024;
static const int ELEMENTS = SIZE*SIZE;
static const int RADIX2 = 10;
static const int FFT_WG_SIZE = 128;
static const int IMG_WG_SIZE = 16;
static const int WG_COUNT_IMAGE = SIZE / IMG_WG_SIZE;
static const int AXIS_COUNT = 2;

static const float PI = 3.14159265f;
static const float TWO_PI = 2.0f*PI;

[StorageBuffer(ELEMENTS)]
RWStructuredBuffer<float2> image;

uint linearIndex(uint2 id) {
    return id.x + id.y * SIZE;
}

uint getAxisIndex(uint id, uint group, uint axis) {
    uint2 idx;
    idx[axis] = id;
    idx[1 - axis] = group;
    return linearIndex(idx);
}

float2 expi(float angle) {
    return float2(cos(angle), sin(angle));
}

float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

uint fftshift(uint index) {
    return (index + SIZE / 2) % SIZE;
}

uint ifftshift(uint index) {
    return (index + (SIZE + 1) / 2) % SIZE;
}

groupshared float2 temp[SIZE];

void radix2(uint span, uint index, bool inverse)
{
    //compute pair of indices of elements 
    //to perform the radix2 butterfly to
    //every iteration we operate on groups of N * span elements, n our radix
    uint group_size = span << 1;
    uint group_half_mask = span - 1;
    //get the index of this thread relative to group
    uint group_offset = index & group_half_mask;
    //get the index offset of the group this thread is in times two
    uint group_index = (index - group_offset) << 1;
    //first element is group + offset in first group half
    uint k1 = group_index + group_offset;
    //second element is group + offset in second group half
    uint k2 = k1 + span;

    float d = inverse ? 1.0 : -1.0;
    float angle = TWO_PI * d * float(group_offset) / float(group_size);

    //radix2 butterfly
    float2 v1 = temp[k1];
    float2 v2 = cmul(expi(angle), temp[k2]);
    temp[k1] = v1 + v2;
    temp[k2] = v1 - v2;
}

void fft(uint index, uint group, uint axis, bool inverse) {
    //number of elements to load per workgroup thread
    uint M = SIZE / FFT_WG_SIZE;
    
    //load elements from input buffer and store them at bit reversed indices
    for (uint i = 0u; i < M; i++) {
        uint rowIndex = index + i * FFT_WG_SIZE;
        uint idx = reversebits(rowIndex) >> (32u - RADIX2);
        temp[idx] = image[getAxisIndex(rowIndex, group, axis)];
    }

    //wait for data be loaded
    GroupMemoryBarrierWithGroupSync();

    //in-place FFT loop
    for (uint span = 1u; span < SIZE; span*=2u)
    {
        for (uint j = 0u; j < M >> 1; j++) {
            let rowIndex = index + j * FFT_WG_SIZE;
            radix2(span, rowIndex, inverse);
        }
        //wait for all warps to complete work
        GroupMemoryBarrierWithGroupSync();
    }

    //store the result back into input buffer
    for (uint i = 0u; i < M; i++) {
        uint rowIndex = index + i * FFT_WG_SIZE;
        uint idx = getAxisIndex(rowIndex, group, axis);
        image[idx] = temp[rowIndex] / (inverse ? SIZE : 1.0);
    }
}

[WorkgroupCount(WG_COUNT_IMAGE, WG_COUNT_IMAGE, 1)]
[shader("compute")]
[numthreads(IMG_WG_SIZE, IMG_WG_SIZE, 1)]
void set_image(uint3 id : SV_DispatchThreadID) {
    float4 color = channel0[id.xy];
    image[linearIndex(id.xy)] = color.xy;
}

[DispatchCount(2)]
[WorkgroupCount(SIZE, 1, 1)]
[shader("compute")]
[numthreads(FFT_WG_SIZE, 1, 1)]
void fft_kernel(uint3 gtid : SV_GroupThreadID, uint3 gid: SV_GroupID) {
    fft(gtid.x, gid.x, dispatch.id, false);
}

[shader("compute")]
[numthreads(IMG_WG_SIZE, IMG_WG_SIZE, 1)]
void main_image(uint3 id : SV_DispatchThreadID)
{
    // Viewport resolution (in pixels)
    uint width, height;
    screen.GetDimensions(width, height);
        
    // Prevent overdraw for workgroups on the edge of the viewport
    int2 pos = int2(id.xy) - (int2(width, height) - (SIZE)) /2;

    if (any(pos >= SIZE) || any(pos < 0)) {
        return;
    }

    uint2 posu = uint2(pos);
    // Get image 
    uint idx = linearIndex(uint2(fftshift(posu.x), fftshift(posu.y)));
    float2 value = image[idx];
    float len = length(value)*0.005;
    float3 col = float3(len, len, len);

    // Output to screen (linear colour space)
    screen[id.xy] = float4(col, 1.0);
}

