//check out https://compute.toys/view/1187 too

#define SIZE 512
#define IT_NUM 8 //floor(log2(N))-1
#define WG_SIZE 256
#define WG_COUNT_IMAGE 32 //SIZE / 16
#define PI 3.14159265
#define AXIS_COUNT 2
#define ELEMENT_COUNT SIZE*SIZE
#storage sim array<vec2f, ELEMENT_COUNT>

fn reverseLowestBits(num: u32, bits: u32) -> u32 {
    let reversed = reverseBits(num);
    let shifted_reversed = reversed >> (32u - bits);
    let upper_bits = (num >> bits) << bits;
    return shifted_reversed + upper_bits;
}

fn getIndexPair(i: u32, it: u32) -> vec2<u32> {
    let k1 = reverseLowestBits(2u * i, it + 1u);
    let k2 = k1 + (1u << it);
    return vec2u(k1, k2);
}

fn linearIndex(id: vec2u) -> u32
{
    return id.x + id.y*SIZE;
}

fn getAxisIndex(id: u32, group: u32, axis: u32) -> u32
{
    var idx = vec2u();
    idx[axis] = id;
    idx[1-axis] = group; 
    return linearIndex(idx);
}

fn expi(angle: float) -> vec2f {
    return vec2f(cos(angle), sin(angle));
}

fn cmul(a: vec2f, b: vec2f) -> vec2f {
    return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn unityRoot(k: u32, N: u32, inverse: bool) -> vec2f {
    let d = select(-1.0, 1.0, inverse);
    return expi(2.0 * d * PI * float(k) / float(N));
}

fn sinc(x: f32) -> f32 {
    return select(sin(x) / x, 1.0 - x * x / 6.0, abs(x) < 1e-4);
}

fn fftshift(index: u32) -> u32 
{
    return (index + SIZE / 2) % SIZE;
}

fn ifftshift(index: u32) -> u32 
{
    return (index + (SIZE + 1) / 2) % SIZE;
}

var<workgroup> TEMP: array<vec2f, SIZE>;

fn fft(index: u32, group: u32, axis: u32, inverse: bool) {
    for (var iteration = 0u; iteration <= u32(IT_NUM); iteration++)
    {
        var ids = getIndexPair(index, select(iteration, IT_NUM, iteration == 0u));
        let v1 = select(TEMP[ids.x], sim[getAxisIndex(ids.x, group, axis)], iteration == 0u);
        let v2 = select(TEMP[ids.y], sim[getAxisIndex(ids.y, group, axis)], iteration == 0u);

        ids = getIndexPair(index, iteration);
        let rootIndex = (ids.x & ((1u << iteration) - 1u)) << (u32(IT_NUM) - iteration);
        let V = cmul(unityRoot(rootIndex, SIZE, inverse), v2);
        TEMP[ids.x] = v1 + V;
        TEMP[ids.y] = v1 - V;

        workgroupBarrier();
    }

    let M = SIZE / WG_SIZE;
    for (var i = 0u; i < u32(M); i++) {
        let rowIndex = index + i * WG_SIZE;
        let idx = getAxisIndex(rowIndex, group, axis);
        sim[idx] = TEMP[rowIndex] / select(1.0, SIZE, inverse);
    }

    storageBarrier();
}

//Set image to sim array
#workgroup_count set_image WG_COUNT_IMAGE WG_COUNT_IMAGE 1
@compute @workgroup_size(16, 16)
fn set_image(@builtin(global_invocation_id) id: vec3u) {
    let uv = (vec2f(id.xy) + 0.5) / SIZE;
    var color = textureSampleLevel(channel0, bilinear, uv, 0).rgb * 0.5;
    sim[linearIndex(id.xy)] = vec2f(color.x, 0);
}

//Convert image to frequency domain
#dispatch_count fft_kernel 2
#workgroup_count fft_kernel SIZE 1 1
@compute @workgroup_size(WG_SIZE)
fn fft_kernel(
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) workgroup_id: vec3u
) {
    fft(local_index, workgroup_id.x, dispatch.id, false);
}

// Convolve the image with a kernel in frequency domain
#workgroup_count convolution WG_COUNT_IMAGE WG_COUNT_IMAGE 1
@compute @workgroup_size(16, 16)
fn convolution(@builtin(global_invocation_id) id: vec3u) {
    let freq2 = vec2i(vec2u(ifftshift(id.x), ifftshift(id.y))) - SIZE/2;
    let freq = length(vec2f(freq2));
    let t = 0.5 - 0.5 * cos(time.elapsed * 3.0);

    // Circular bokeh
    let scale = sinc(freq*t);
    
    sim[linearIndex(id.xy)] *= scale;
}

//Convert image back to spatial domain
#dispatch_count ifft_kernel 2
#workgroup_count ifft_kernel SIZE 1 1
@compute @workgroup_size(WG_SIZE)
fn ifft_kernel(
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) workgroup_id: vec3u
) {
    fft(local_index, workgroup_id.x, dispatch.id, true);
}

//display the image
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= SIZE || id.y >= SIZE) {
        return;
    }

    // Get image 
    let idx = linearIndex(id.xy);
    let value = sim[idx];
    let len = length(value);
    let col = vec3f(len, len, len);

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}
