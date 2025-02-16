//check out https://compute.toys/view/1187 too

#define SIZE 512
#define IT_NUM 8 //floor(log2(N))-1
#define WG_SIZE 256
#define WG_COUNT_IMAGE 32 //SIZE / 16
#define PI 3.14159265
#define AXIS_COUNT 2
#define ELEMENT_COUNT SIZE*SIZE
#define MASS 1000.0
#define H_PLANK 1.0
#define TIME_STEP 0.1
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

fn PotentialAbsorb(pos: vec2u) -> vec2f {
    let d = length(vec2f(pos)- SIZE/2.0);
    let V = 0.0*(1.0 - smoothstep(95, 100, d));
    let A = 1.0;//*smoothstep(40, 95, d);
    return vec2f(V,A);
}

#workgroup_count real_space WG_COUNT_IMAGE WG_COUNT_IMAGE 1
@compute @workgroup_size(16, 16)
fn real_space(@builtin(global_invocation_id) id: vec3u) {
    let index = linearIndex(id.xy);
    if(time.frame < 5)
    {
        let uv = (vec2f(id.xy) + 0.5) / SIZE;
        var color = textureSampleLevel(channel0, bilinear, uv, 0).rgb * 0.5;
        sim[index] = vec2f(color.x, 0);
    } else
    {
        //apply potential energy time evolution operator
        var field = sim[index];
        var VA = PotentialAbsorb(id.xy);
        #define SUBSTEPS 8
        for(var i = 0; i < SUBSTEPS; i++)
        {
            let dt = TIME_STEP  / float(SUBSTEPS);
            let potentialUpdate = expi(-dt * VA.x / (2*H_PLANK)) * VA.y;
            field = cmul(potentialUpdate, field);
            field += 10.0*dt*field*(1 - 0.002*length(field));
        }
        sim[index] = field;
    }
}

//Convert field to momentum space
#dispatch_count fft_kernel AXIS_COUNT
#workgroup_count fft_kernel SIZE 1 1
@compute @workgroup_size(WG_SIZE)
fn fft_kernel(
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) workgroup_id: vec3u
) {
    fft(local_index, workgroup_id.x, dispatch.id, false);
}

#workgroup_count momentum_space WG_COUNT_IMAGE WG_COUNT_IMAGE 1
@compute @workgroup_size(16, 16)
fn momentum_space(@builtin(global_invocation_id) id: vec3u) {
    let freq2 = vec2i(vec2u(ifftshift(id.x), ifftshift(id.y))) - SIZE/2;
    let freq = length(vec2f(freq2));
    //apply kinetic energy time evolution operator
    let kineticUpdate = expi(-TIME_STEP*H_PLANK*freq*freq/(2.0*MASS));
    let index = linearIndex(id.xy);
    sim[index] = cmul(kineticUpdate, sim[index]);
}

//Convert image back to real space
#dispatch_count ifft_kernel AXIS_COUNT
#workgroup_count ifft_kernel SIZE 1 1
@compute @workgroup_size(WG_SIZE)
fn ifft_kernel(
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) workgroup_id: vec3u
) {
    fft(local_index, workgroup_id.x, dispatch.id, true);
}

fn hsv2rgb_smooth(c: vec3f) -> vec3f {
    let m = (c.x * 6.0 + vec3f(0.0, 4.0, 2.0)) % 6.0;
    var rgb = clamp(
        abs(m - 3.0) - 1.0,
        vec3f(0.0),
        vec3f(1.0)
    );
    rgb = rgb * rgb * (3.0 - 2.0 * rgb);
    return c.z * mix(vec3f(1.0), rgb, c.y);
}

//display the image
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Get the full screen (output texture) size
    let screenSize = textureDimensions(screen);

    // Avoid out-of-range writes
    if (id.x >= screenSize.x || id.y >= screenSize.y) {
        return;
    }

    // Convert to float for aspect-ratio math
    let screenWidth  = f32(screenSize.x);
    let screenHeight = f32(screenSize.y);

    // We'll treat 'SIZE' as both width and height for the sim array
    let simWidth  = f32(SIZE);
    let simHeight = f32(SIZE);

    // Calculate aspect ratios
    let screenAspect = screenWidth / screenHeight;
    let simAspect    = simWidth / simHeight;

    // Determine scale so that sim fits the screen dimension without cropping
    var scale: f32;
    if (screenAspect > simAspect) {
        // Screen is "wider" than sim -> match heights
        scale = screenHeight / simHeight;
    } else {
        // Screen is "taller" or equal -> match widths
        scale = screenWidth / simWidth;
    }

    // Calculate the scaled width/height
    let scaledW = simWidth  * scale;
    let scaledH = simHeight * scale;

    // Offset to center the image
    // (We place the scaled sim so it's centered in the screen)
    let offsetX = (screenWidth  - scaledW) * 0.5;
    let offsetY = (screenHeight - scaledH) * 0.5;

    // Check if this pixel (id.xy) is within the scaled/letterboxed region
    let fx = f32(id.x);
    let fy = f32(id.y);

    // If we're outside the region, draw black (or skip)
    if (fx < offsetX || fx >= (offsetX + scaledW) ||
        fy < offsetY || fy >= (offsetY + scaledH)) 
    {
        // Example: paint black
        textureStore(screen, id.xy, vec4f(0.0, 0.0, 0.0, 1.0));
        return;
    }

    // Convert this screen pixel back into sim coordinates
    let simX = (fx - offsetX) / scale; // in [0, simWidth)
    let simY = (fy - offsetY) / scale; // in [0, simHeight)

    // Use nearest-neighbor integer coordinates
    // (u32() floors the float value)
    let ix = u32(simX);
    let iy = u32(simY);

    // Stay safe if we had rounding issues (optional clamp)
    if (ix >= SIZE || iy >= SIZE) {
        textureStore(screen, id.xy, vec4f(0.0, 0.0, 0.0, 1.0));
        return;
    }

    // Sample from your simulation buffer
    let idx = linearIndex(vec2u(ix, iy));
    let value = sim[idx]; 

    let angle = atan2(value.y, value.x);
    let len = length(value) * 0.001;
    let col = hsv2rgb_smooth(vec3f((angle + PI) / (2.0 * PI), 0.4, len));

    // Write the color to the screen
    textureStore(screen, id.xy, vec4f(col, 1.0));
}