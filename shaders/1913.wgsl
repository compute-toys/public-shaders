const MAX_RADIX = 4u;

// #define SIZE 512
// const RADIX_COUNT = 5u;                             
// const RADICES = array<u32, RADIX_COUNT>(4, 4, 4, 4, 2);
// #define WG_COUNT_IMAGE 32 //SIZE / 16

#define SIZE 384
const RADIX_COUNT = 5u;                             
const RADICES = array<u32, RADIX_COUNT>(4, 4, 4, 3, 2);
#define WG_COUNT_IMAGE 24 //SIZE / 16

#define WG_SIZE 32
#define PI 3.14159265
#define TWO_PI (2.0 * PI)
#define AXIS_COUNT 2
#define ELEMENT_COUNT SIZE*SIZE
#storage sim array<vec2f, ELEMENT_COUNT>

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
    let d = select(1.0, -1.0, inverse);
    return expi(- 2.0 * PI * d * float(k) / float(N));
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

fn mixed_radix_reverse(n_in: u32) -> u32 {
    var n = n_in;
    var q = u32(SIZE);
    var out = 0u;

    for (var i: u32 = 0u; i < RADIX_COUNT; i++) {
        let radix = RADICES[RADIX_COUNT - 1 - i];
        q   = q / radix;
        let n1 = n / radix;
        out = out + (n - n1 * radix) * q;
        n = n1;
    }

    return out;
}

fn radix2(span: uint, index: uint, inverse: bool)
{
    //compute pair of indices of elements 
    //to perform the radix2 butterfly to
    //every iteration we operate on groups of N * span elements, n our radix
    let group_size = span << 1;
    let group_half_mask = span - 1;
    //get the index of this thread relative to group
    let group_offset = index & group_half_mask;
    //get the index offset of the group this thread is in times two
    let group_index = (index - group_offset) << 1;
    //first element is group + offset in first group half
    let k1 = group_index + group_offset;
    //second element is group + offset in second group half
    let k2 = k1 + span;

    //radix2 butterfly
    let v1 = TEMP[k1];
    let v2 = TEMP[k2];
    //the unity root of this element pair in this group
    let uroot = unityRoot(group_offset, group_size, inverse);
    let V = cmul(uroot, v2);
    TEMP[k1] = v1 + V;
    TEMP[k2] = v1 - V;
}

fn radixN(span: uint, N: uint, index: uint, inverse: bool)
{
    //compute DFT buttlerfly of sets of N elements
    let group_size = span * N;
    let group_index =  (index / span)  * span;
    let group_offset = index - group_index;
    let first_element = group_index * N + group_offset;
    
    let d = select(-1.0, 1.0, inverse);
    let angle = TWO_PI * d * float(group_offset) / float(group_size);

    var values: array<vec2f, MAX_RADIX>;
    for (var i = 0u; i < N; i++) {
        values[i] = cmul(TEMP[first_element + span*i], expi(angle * float(i)));
    }

    for (var i = 0u; i < N; i++) {
        var result = vec2f(0.0);
        for (var j = 0u; j < N; j++) {
            result += cmul(values[j], expi(TWO_PI * d * float(i) * float(j) / float(N)));
        }
        TEMP[first_element + span*i] = result;
    }
}

fn fft(index: u32, group: u32, axis: u32, inverse: bool) {
    //number of elements to load per workgroup thread
    let M = SIZE / WG_SIZE;
    
    //load elements from input buffer and store them at bit reversed indices
    for (var i = 0u; i < u32(M); i++) {
        let rowIndex = index + i * WG_SIZE;
        let idx = mixed_radix_reverse(rowIndex);
        TEMP[idx] = sim[getAxisIndex(rowIndex, group, axis)];
    }

    //wait for data be loaded
    workgroupBarrier();

    //in-place FFT loop
    var span = 1u;
    for (var i = 0u; i < RADIX_COUNT; i++)
    {
        let radix = RADICES[i];
        for (var j = 0u; j < u32(M)/radix; j++) {
            let rowIndex = index + j * WG_SIZE;
            radixN(span, radix, rowIndex, inverse);
        }
        span*=radix;
        //wait for all warps to complete work
        workgroupBarrier();
    }

    //store the result back into input buffer
    for (var i = 0u; i < u32(M); i++) {
        let rowIndex = index + i * WG_SIZE;
        let idx = getAxisIndex(rowIndex, group, axis);
        sim[idx] = TEMP[rowIndex] / select(1.0, SIZE, inverse);
    }

    //make sure the data is written
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

//display the image
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport


    let pos = vec2i(id.xy) - (vec2i(screen_size) - (SIZE)) /2;

    if (any(pos >= vec2i(SIZE)) || any(pos < vec2i(0))) {
        return;
    }

    let posu = vec2u(pos);
    // Get image 
    let idx = linearIndex(vec2u(fftshift(posu.x), fftshift(posu.y)));
    let value = sim[idx];
    let len = length(value)*0.05;
    let col = vec3f(len, len, len);

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}
