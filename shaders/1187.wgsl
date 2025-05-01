// 2D FFT using workgroup memory. Implements the Cooley-Tukey FFT algorithm with
// radix-2 and radix-4.

// The size of the image to FFT: a power of 2 between 256 and 2048
#define N 1024
// Must be manually set to N/256
#define N_256 4
// The number of channels in the image to FFT: 1, 2 or 3
const N_CHANNELS = 3;
// The radix to use for the FFT: 2 or 4
const RADIX = 4;

#storage image array<array<array<vec2f, N>, N>, N_CHANNELS>

// Initialize the image to something interesting
#workgroup_count initialize_image N_256 N 1
@compute @workgroup_size(256)
fn initialize_image(@builtin(global_invocation_id) id: vec3u) {
    let uv = (vec2f(id.xy) + 0.5) / N;
    var color = textureSampleLevel(channel0, bilinear, uv, 0).rgb * 0.5;

    if N_CHANNELS == 1 {
        color.r = dot(vec3(0.2126, 0.7152, 0.0722), color);
    }

    for (var i = 0; i < N_CHANNELS; i++) {
        image[i][id.y][id.x] = vec2(color[i], 0.0);
    }
}

const LOG2_N = firstLeadingBit(u32(N));
const LOG4_N = LOG2_N / 2;
const PI = 3.1415927;

fn sinc(x: f32) -> f32 {
    return select(sin(x) / x, 1.0, abs(x) < 4e-4);
}

fn mul(x: vec2f, y: vec2f) -> vec2f {
    return vec2(x.x * y.x - x.y * y.y, x.x * y.y + x.y * y.x);
}

fn cis(x: f32) -> vec2f {
    return vec2(cos(x), sin(x));
}

fn set_component(v: vec2u, i: u32, x: u32) -> vec2u {
    var u = v;
    u[i] = x;
    return u;
}

fn reverse_digits_base_2(x: u32, n: u32) -> u32 {
    return reverseBits(x) >> (32 - n);
}

fn reverse_digits_base_4(x: u32, n: u32) -> u32 {
    var v = x;
    var y = 0u;
    
    for (var i = 0u; i < n; i++) {
        y = (y << 2) | (v & 3u);
        v >>= 2;
    }
    
    return y;
}

const WORKGROUP_SIZE = min(N / RADIX, 256);
var<workgroup> X: array<vec2f, N>;

// Perform an FFT over one axis of the image
fn fft(local_index: u32, workgroup_index: u32, axis: u32, inverse: bool) {
    for (var ch = 0; ch < N_CHANNELS; ch++) {
        // Copy a row/column of values from the storage buffer into workgroup
        // memory, doing a bit-reversal permutation in the process
        for (var i = 0u; i < N / WORKGROUP_SIZE; i++) {
            let j = local_index + i * WORKGROUP_SIZE;
            let p = set_component(vec2(workgroup_index), axis, j);
            var k: u32;

            if RADIX == 2 {
                k = reverse_digits_base_2(j, LOG2_N);
            } else {
                k = reverse_digits_base_4(j >> (LOG2_N & 1), LOG4_N);
                k |= (j & (LOG2_N & 1)) << (LOG2_N - 1);
            }

            X[k] = image[ch][p.y][p.x];
        }

        workgroupBarrier();
        let d = select(-1.0, 1.0, inverse);

        // Do the radix-4 passes if chosen
        for (var p = 0u; RADIX == 4 && p < LOG4_N; p++) {
            let s = 1u << (2 * p);

            for (var i = 0u; i < N / WORKGROUP_SIZE / 4; i++) {
                let j = local_index + i * WORKGROUP_SIZE;
                let k = j & (s - 1);
                let t = d * 2.0 * PI / f32(s * 4) * f32(k);
                let k0 = ((j >> (2 * p)) << (2 * p + 2)) + k;
                let k1 = k0 + 1 * s;
                let k2 = k0 + 2 * s;
                let k3 = k0 + 3 * s;
                let x0 = X[k0];
                let x1 = mul(cis(t), X[k1]);
                let x2 = mul(cis(t * 2.0), X[k2]);
                let x3 = mul(cis(t * 3.0), X[k3]);
                X[k0] = x0 + x1 + x2 + x3;
                X[k1] = x0 + d * vec2(-x1.y, x1.x) - x2 + d * vec2(x3.y, -x3.x);
                X[k2] = x0 - x1 + x2 - x3;
                X[k3] = x0 + d * vec2(x1.y, -x1.x) - x2 + d * vec2(-x3.y, x3.x);
            }

            workgroupBarrier();
        }

        // Do the radix-2 passes if chosen or if there is a leftover pass
        // required from doing the radix-4 passes (i.e. if N isn't a power of 4)
        for (var p = select(0, 2 * LOG4_N, RADIX == 4); p < LOG2_N; p++) {
            let s = 1u << p;

            for (var i = 0u; i < N / WORKGROUP_SIZE / 2; i++) {
                let j = local_index + i * WORKGROUP_SIZE;
                let k = j & (s - 1);
                let k0 = ((j >> p) << (p + 1)) + k;
                let k1 = k0 + s;
                let x0 = X[k0];
                let x1 = mul(cis(d * 2.0 * PI / f32(s * 2) * f32(k)), X[k1]);
                X[k0] = x0 + x1;
                X[k1] = x0 - x1;
            }

            workgroupBarrier();
        }

        // Copy the FFT'd workgroup memory back into the storage buffer
        for (var i = 0u; i < N / WORKGROUP_SIZE; i++) {
            let j = local_index + i * WORKGROUP_SIZE;
            let p = set_component(vec2(workgroup_index), axis, j);
            image[ch][p.y][p.x] = X[j] / select(1.0, N, inverse);
        }

        storageBarrier();
    }
}

// The number of times to perform the FFT+IFFT, only meant for timing purposes
#define DISPATCH_COUNT 1

// Convert image to frequency domain
#dispatch_count fft_horizontal DISPATCH_COUNT
#workgroup_count fft_horizontal N 1 1
@compute @workgroup_size(WORKGROUP_SIZE)
fn fft_horizontal(
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) workgroup_id: vec3u
) {
    fft(local_index, workgroup_id.x, 0, false);
}

#dispatch_count fft_vertical DISPATCH_COUNT
#workgroup_count fft_vertical N 1 1
@compute @workgroup_size(WORKGROUP_SIZE)
fn fft_vertical(
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) workgroup_id: vec3u
) {
    fft(local_index, workgroup_id.x, 1, false);
}

// Modify the image in the frequency domain
#workgroup_count modify_frequencies N_256 N 1
@compute @workgroup_size(256)
fn modify_frequencies(@builtin(global_invocation_id) id: vec3u) {
    let t = 0.5 - 0.5 * cos(time.elapsed * 3.0);
    let f2 = vec2f(vec2i(id.xy + N / 2) % N - N / 2) * t;
    let f = length(f2);
    var scale = 1.0;

    // Circular bokeh
    scale = sinc(f);

    // Gaussian blur
    // scale = exp(-0.2 * f * f);

    // Box blur
    // scale = sinc(f2.x) * sinc(f2.y);

    for (var i = 0; i < N_CHANNELS; i++) {
        image[i][id.y][id.x] *= scale;
    }
}

// Convert image back to spatial domain
#dispatch_count ifft_horizontal DISPATCH_COUNT
#workgroup_count ifft_horizontal N 1 1
@compute @workgroup_size(WORKGROUP_SIZE)
fn ifft_horizontal(
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) workgroup_id: vec3u
) {
    fft(local_index, workgroup_id.x, 0, true);
}

#dispatch_count ifft_vertical DISPATCH_COUNT
#workgroup_count ifft_vertical N 1 1
@compute @workgroup_size(WORKGROUP_SIZE)
fn ifft_vertical(
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) workgroup_id: vec3u
) {
    fft(local_index, workgroup_id.x, 1, true);
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let dimensions = textureDimensions(screen);

    if any(id.xy >= dimensions) {
        return;
    }

    var p = vec2i(id.xy) - vec2i(dimensions) / 2 + N / 2;
    // Wrap image
    // p = select(p % N, p % N + N, p < vec2(0));

    if any((p < vec2(0)) | (p >= vec2(N))) {
        textureStore(screen, id.xy, vec4(0.0, 0.0, 0.0, 1.0));
        return;
    }
    
    var color = vec3(0.0);

    for (var i = 0; i < N_CHANNELS; i++) {
        let data = image[i][p.y][p.x];
        color[i] = length(data);
    }

    if N_CHANNELS == 1 {
        color = color.rrr;
    }

    textureStore(screen, id.xy, vec4(color, 1.0));
}