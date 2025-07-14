import std;
import shadertoy;

static float WIND_SPEED = custom.WIND_SPEED;
static float WIND_ANGLE = custom.WIND_ANGLE;
static float LEEWARD_FETCH = custom.LEEWARD_FETCH;
static float DEPTH = custom.DEPTH;
static float LENGTH_SCALE = custom.LENGTH_SCALE;

static const int SIZE = 1024;
static const int ELEMENTS = SIZE*SIZE;
static const int RADIX2 = 10;
static const int FFT_WG_SIZE = 128;
static const int IMG_WG_SIZE = 16;
static const int WG_COUNT_IMAGE = SIZE / IMG_WG_SIZE;
static const int AXIS_COUNT = 2;

static const float g = 9.81f;
static const float PI = 3.14159265f;
static const float TWO_PI = 2.0f*PI;

[StorageBuffer(ELEMENTS)]
RWStructuredBuffer<float2> image;

[StorageBuffer(ELEMENTS)]
RWStructuredBuffer<float2> tangent;
[StorageBuffer(ELEMENTS)]
RWStructuredBuffer<float2> bitangent;

// ---------------------------- RNG ----------------------------
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
public float2 random_unit_vector2() {
	return normalize(float2(rand_f32() * 2. - 1., rand_f32() * 2. - 1.));
}

// ---------------------------- Base JONSWAP ----------------------------
float jonswap(float dispersion, float dispersion_peak, float speed, float fetch) {
	if (dispersion == 0.0) {
		return 0.0;
	}

	let alpha = intensity(speed, fetch);

	let beta = 5.0 / 4.0;
	let exponent = -beta * powr(dispersion_peak, 4) / powr(dispersion, 4);

	let gamma = 3.3;

	return ((alpha * g * g) / powr(dispersion, 5)) * exp(exponent) * powr(gamma, jonswap_exponent(dispersion, dispersion_peak));
}

float peak_dispersion(float wind_speed, float fetch) {
	return 22.0 * powr(g * g / (wind_speed * fetch), 0.333333);
}

// A constant that relates to the wind speed and fetch length.
// Typical values in the northern north sea are in the range of 0.0081 to 0.01 (https://www.codecogs.com/library/engineering/fluid_mechanics/waves/spectra/jonswap.php)
float intensity(float wind_speed, float fetch) {
	return 0.076 * powr(wind_speed * wind_speed / (fetch * g), 0.22);
}

float jonswap_exponent(float dispersion, float dispersion_peak) {
	let s = sigma(dispersion, dispersion_peak);

	let num = (dispersion - dispersion_peak) * (dispersion - dispersion_peak);
	let den = 2.0 * dispersion_peak * dispersion_peak * s * s;

	return exp(-(num / den));
}

float sigma(float dispersion, float dispersion_peak) {
	if (dispersion <= dispersion_peak) {
		return 0.07;
	} else {
		return 0.09;
	}
}

// ---------------------------- Kitaigordskii Depth Correction ----------------------------
float depth_correction(float dispersion, float depth) {
	let omega_h = dispersion * sqrt(depth / g);
	if (omega_h <= 1.0) {
		return 0.5 * omega_h * omega_h;
	} else if (omega_h <= 2.0) {
		return 1.0 - 0.5 * (2 - omega_h) * (2 - omega_h);
	} else {
		return 1.0;
	}
}

// ---------------------------- Donelan-Banner Directional Spreading ----------------------------
float directional_spreading(float dispersion, float dispersion_peak, float relative_wind_angle) {
	let ratio = dispersion / dispersion_peak;
	float beta = 0.0;
	if (ratio < 0.56) {
		beta = 2.61 * powr(0.56, 1.3);	// TODO: Verify this is how we want to deal with values outside the equation
	} else if (ratio <= 0.95) {
		beta = 2.61 * powr(ratio, 1.3);
	} else if (ratio <= 1.6) {
		beta = 2.28 * powr(ratio, -1.3);
	} else {
		beta = powr(10.0, -0.4 + 0.8393 * exp(-0.567 * powr(log(ratio), 2)));
	}
	return beta * pow(cosh(beta * relative_wind_angle), -2) / (2.0 * tanh(beta * PI));
}

// ---------------------------- Final Wave Spectrum ----------------------------
float wave_spectrum(float2 wave_vector, float wave_vector_mag, float dispersion, float wind_speed, float fetch, float wind_angle, float depth, float2 delta_k) {
	// let relative_wind_angle = (atan(wave_vector.y/ wave_vector.x) - wind_angle) % PI;
	var relative_wind_angle = (atan2(wave_vector.y, wave_vector.x) - wind_angle);
	if (relative_wind_angle > PI) {
		relative_wind_angle -= 2*PI;
	} else if (relative_wind_angle < -PI) {
		relative_wind_angle += 2*PI;
	}

	let dispersion_peak = peak_dispersion(wind_speed, fetch);

	let S_j = jonswap(dispersion, dispersion_peak, wind_speed, fetch);
	let D = directional_spreading(dispersion, dispersion_peak, relative_wind_angle);
	// let D = 1;

	float dw_dk_k = 0.0;

	if (wave_vector_mag != 0.0) {
		dw_dk_k = g / (2 * sqrt(g * wave_vector_mag)) / wave_vector_mag;
	}

	return 2 * S_j * depth_correction(dispersion, depth) * D * dw_dk_k * delta_k.x * delta_k.y;
}

// ---------------------------- Fourier Spectrum ----------------------------
float2 h_0_tilde(float2 wave_vector, float wave_vector_mag, float dispersion, float wind_speed, float fetch, float wind_angle, float depth, float2 delta_k) {
	let common = sqrt(wave_spectrum(wave_vector, wave_vector_mag, dispersion, wind_speed, fetch, wind_angle, depth, delta_k) / 2.0);
	return common * random_unit_vector2();
}

float2 fourier_spectrum(float2 wave_vector, float wind_speed, float fetch, float wind_angle, float depth, float2 delta_k, float t) {
	let wave_vector_mag = length(wave_vector);
	let dispersion = sqrt(g * wave_vector_mag);

	var base = h_0_tilde(wave_vector, wave_vector_mag, dispersion, wind_speed, fetch, wind_angle, depth, delta_k);
	var conj = h_0_tilde(-wave_vector, wave_vector_mag, dispersion, wind_speed, fetch, wind_angle, depth, delta_k);
	conj.y = -conj.y;

	var phase = dispersion * t;

	base = cmul(base, float2(sin(phase), cos(phase)));
	conj = cmul(conj, float2(sin(-phase), cos(-phase)));

	return base + conj;
}

// ---------------------------- FFT ----------------------------

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

void fft(uint index, uint group, uint axis, bool inverse, RWStructuredBuffer<float2> img) {
    //number of elements to load per workgroup thread
    uint M = SIZE / FFT_WG_SIZE;
    
    //load elements from input buffer and store them at bit reversed indices
    for (uint i = 0u; i < M; i++) {
        uint rowIndex = index + i * FFT_WG_SIZE;
        uint idx = reversebits(rowIndex) >> (32u - RADIX2);
        temp[idx] = img[getAxisIndex(rowIndex, group, axis)];
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
        img[idx] = temp[rowIndex] / (inverse ? SIZE : 1.0);
    }
}

[WorkgroupCount(WG_COUNT_IMAGE, WG_COUNT_IMAGE, 1)]
[shader("compute")]
[numthreads(IMG_WG_SIZE, IMG_WG_SIZE, 1)]
void set_image(uint3 id : SV_DispatchThreadID) {
	if (id.x > SIZE || id.y > SIZE) {
		return;
	}
    // Generate a seed that's constant between all subsequent frames
    init_rng(id.xy, SIZE, 0);

	// Wave vector is a value [-pi, pi] mapped to the axes of the texture
    float2 wave_vector = float2(id.xy) / float2(SIZE) * 2 * PI - PI;
	wave_vector /= LENGTH_SCALE; // Length scale of ocean patch 
    float2 delta_k = float2(2*PI/float(SIZE));

	float2 h = fourier_spectrum(wave_vector,  WIND_SPEED,  LEEWARD_FETCH,  WIND_ANGLE,  DEPTH, delta_k,  time.elapsed);

	let ih = float2(-h.y, h.x);

    tangent[linearIndex(id.xy)] = ih * wave_vector.x;
	bitangent[linearIndex(id.xy)] = ih * wave_vector.y;

    // image[linearIndex(id.xy)] = h.y / length(wave_vector) * wave_vector;
    image[linearIndex(id.xy)] = ih * 1000;
}

[DispatchCount(2)]
[WorkgroupCount(SIZE, 1, 1)]
[shader("compute")]
[numthreads(FFT_WG_SIZE, 1, 1)]
void fft_kernel(uint3 gtid : SV_GroupThreadID, uint3 gid: SV_GroupID) {
    fft(gtid.x, gid.x, dispatch.id%2, true, image);
    fft(gtid.x, gid.x, dispatch.id%2, true, tangent);
    fft(gtid.x, gid.x, dispatch.id%2, true, bitangent);
}

[shader("compute")]
[numthreads(IMG_WG_SIZE, IMG_WG_SIZE, 1)]
void main_image(uint3 id : SV_DispatchThreadID)
{
    // Viewport resolution (in pixels)
    uint width, height;
    screen.GetDimensions(width, height);

    // uint idx = linearIndex(id.xy);
    // float2 value = image[idx];

    uint idx = linearIndex(uint2(ifftshift(id.x), ifftshift(id.y)));
    float2 value = image[idx] * 1000;
    // float2 value = image[idx];

	let tang = tangent[idx].x*SIZE*SIZE*100;
	let bitang = tangent[idx].x*SIZE*SIZE*100;

	let normal = normalize(float3(-tang, 1, bitang));

 	// if (id.x > SIZE || id.y > SIZE) {
	// 	value = float2(0.0);
	// }

    screen[id.xy] = float4(value.x, 0.0, value.x, 1.0);
    // screen[id.xy] = float4(normal, 1.0);
    // screen[id.xy] = float4(tang, 0.0, 0.0, 1.0);
    // screen[id.xy] = float4(0.0, 0.01, value.x, 1.0);
}
 