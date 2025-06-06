import std;
import shadertoy;

static float WIND_SPEED_X = custom.WIND_SPEED_X;
static float WIND_SPEED_Y = custom.WIND_SPEED_Y;
static float WIND_FETCH_X = custom.WIND_FETCH_X;
static float WIND_FETCH_Y = custom.WIND_FETCH_Y;

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

struct Rng {
	uint state;
}
static Rng rng_obj;
public void init_rng(uint2 pixel, uint width, uint frame_num) {
	// Seed the PRNG using the scalar index of the pixel and the current frame count.
	uint seed = (pixel.x + pixel.y * width) ^ jenkins_hash(frame_num);
	rng_obj.state = jenkins_hash(seed);
}
uint jenkins_hash(uint i) {
	var x = i;
	x += x << 10u;
	x ^= x >> 6u;
	x += x << 3u;
	x ^= x >> 11u;
	x += x << 15u;
	return x;
}
uint xorshift32() {
	uint x = rng_obj.state;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	rng_obj.state = x;
	return x;
}
public float rand_f32() {
	return asfloat(0x3f800000u | (xorshift32() >> 9u)) - 1.;
}
public float2 random_unit_vector() {
	return normalize(float2(rand_f32() * 2. - 1., rand_f32() * 2. - 1.));
}

//-------------------- JONSWAP -------------------
const static float g = 9.81;
const static float e = 2.71828;

public float2 jonswap(float f, float speed, float fetch) {
	let f_peak = peak(speed, fetch);
	let alpha = intensity(f, f_peak, speed);

	let beta = 5.0 / 4.0;
	let exponent = -beta * powr(f_peak, 4) / powr(f, 4);

	let gamma = 3.3;
    let power = ((alpha * g * g) / powr(f, 5)) * powr(e, exponent) * powr(gamma, jonswap_exponent(f, f_peak));
    let amplitude = sqrt(power/2.0);

	return amplitude * random_unit_vector() * 2000000.0;
}

float peak(float speed, float fetch) {
	return 2.84 * powr(g, 0.7) * powr(fetch, -0.3) * pow(speed, -0.4);
}

// A constant that relates to the wind speed and fetch length.
// Typical values in the northern north sea are in the range of 0.0081 to 0.01 (https://www.codecogs.com/library/engineering/fluid_mechanics/waves/spectra/jonswap.php)
float intensity(float f, float f_peak, float wind_speed) {
	return 0.033 * pow((f_peak * wind_speed) / g, 2.0 / 3.0);
    // return 0.01;
}

float jonswap_exponent(float f, float f_peak) {
	let s = sigma(f, f_peak);

	let num = (f - f_peak) * (f - f_peak);
	let den = 2.0 * f_peak * f_peak * s * s;

	return powr(e, -(num / den));
}

float sigma(float f, float f_peak) {
	if (f <= f_peak) {
		return 0.07;
	} else {
		return 0.09;
	}
}
//--------------------------------------------------

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

void radix2(uint span, uint index, bool inverse) {
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
    init_rng(id.xy, SIZE, (uint)(time.elapsed*1000.0));
        // Viewport resolution (in pixels)
    // uint width, height;
    // screen.GetDimensions(width, height);
        
    // Prevent overdraw for workgroups on the edge of the viewport
    // int2 pos = int2(id.xy) - (int2(width, height) - (SIZE)) /2;

    // if (any(pos >= SIZE) || any(pos < 0)) return;

    int2 distance_from_center = int2(id.xy) - int2(SIZE/2);
    float distance = sqrt(distance_from_center.x*distance_from_center.x + distance_from_center.y*distance_from_center.y);

    // float2 j = jonswap(distance, WIND_SPEED_X, WIND_FETCH_X);
    // let j_y = jonswap(id.y/float(SIZE/2), WIND_SPEED_Y, WIND_FETCH_Y);

    // float4 color = channel0[id.xy];
    // image[linearIndex(id.xy)] = color.xy;
    // image[linearIndex(id.xy)] = float2(j_x, j_y);
    image[linearIndex(id.xy)] = jonswap(distance, WIND_SPEED_X, WIND_FETCH_X);
    // image[linearIndex(id.xy)] = float2(1.0, 0.6);
}

[DispatchCount(2)]
[WorkgroupCount(SIZE, 1, 1)]
[shader("compute")]
[numthreads(FFT_WG_SIZE, 1, 1)]
void fft_kernel(uint3 gtid : SV_GroupThreadID, uint3 gid: SV_GroupID) {
    // fft(gtid.x, gid.x, dispatch.id%2, true);
    // fft(gtid.x, gid.x, dispatch.id%2, ((dispatch.id/2)%2) == 0);
}

[shader("compute")]
[numthreads(IMG_WG_SIZE, IMG_WG_SIZE, 1)]
void main_image(uint3 id : SV_DispatchThreadID)
{
    // Viewport resolution (in pixels)
    uint width, height;
    screen.GetDimensions(width, height);
        
    // Prevent overdraw for workgroups on the edge of the viewport
    // int2 pos = int2(id.xy) - (int2(width, height) - (SIZE)) /2;

    // if (any(pos >= SIZE) || any(pos < 0)) {
    //     return;
    // }

    // uint2 posu = uint2(pos);
    // Get image 
    // uint idx = linearIndex(uint2(fftshift(posu.x), fftshift(posu.y)));
    uint idx = linearIndex(uint2(ifftshift(id.x), ifftshift(id.y)));
    float2 value = image[idx];
    // float2 center = image[3];
    // float len = length(value)/length(center);
    // float3 col = float3(len, len, len);

    // Output to screen (linear colour space)
    // screen[id.xy] = float4(col, 1.0);
    screen[id.xy] = float4(value.xy,0.0, 1.0);

    // let j_x = jonswap(id.x/float(width/2), WIND_SPEED_X, WIND_FETCH_X);
    // let j_y = jonswap(id.y/float(height/2), WIND_SPEED_Y, WIND_FETCH_Y);

    // screen[id.xy] = float4(j_x, j_y, 0.6, 1.0);
    // screen[id.xy] = float4(id.x / float(width), id.y / float(height),0.6,1.0);
}

