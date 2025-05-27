import std;

static const int SIZE = 1024;
static const int ELEMENTS = SIZE*SIZE;
static const int SORT_WG_SIZE = 256;
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

[WorkgroupCount(WG_COUNT_IMAGE, WG_COUNT_IMAGE, 1)]
[shader("compute")]
[numthreads(IMG_WG_SIZE, IMG_WG_SIZE, 1)]
void set_image(uint3 id : SV_DispatchThreadID) {
    float4 color = channel0[id.xy];
    image[linearIndex(id.xy)] = color.xy;
}

static const uint BLOCKSIZE = SIZE;

groupshared uint2 temp[BLOCKSIZE];
groupshared float2 data[BLOCKSIZE];

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
    
[DispatchCount(2)]
[WorkgroupCount(SIZE, 1, 1)]
[shader("compute")]
[numthreads(SORT_WG_SIZE, 1, 1)]
void block_sort(uint3 gtid : SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    int group = gid.x;
    uint index = gtid.x;
    uint axis = dispatch.id%2;

    uint M = SIZE / SORT_WG_SIZE;

    for (uint i = 0u; i < M; i++) {
        uint rowIndex = index + i * SORT_WG_SIZE;
        float2 value = image[getAxisIndex(rowIndex, group, axis)];
        data[rowIndex] = value;
        float sortvalue = value[axis];
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
        image[getAxisIndex(rowIndex, group, axis)] = data[temp[rowIndex].y];
    }
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
    uint idx = linearIndex(posu);
    float2 value = image[idx];
    float len = length(value);
    float3 col = float3(value.xy, 0.0);

    // Output to screen (linear colour space)
    screen[id.xy] = float4(col, 1.0);
}

