#define SIZE 512
#define IT_NUM 8 //floor(log2(N))-1
#define WG_SIZE 256
#define WG_COUNT_IMAGE 32 //SIZE / 16
#define PI 3.14159265
#define AXIS_COUNT 2
#define ELEMENT_COUNT SIZE*SIZE

#define DISPATCHES_PER_STEP 6 //2*AXIS_COUNT + 2
#dispatch_count mega_kernel 48 //DISPATCHES_PER_STEP * STEP_COUNT

#define MASS 1000.0
#define H_PLANK 1.0
#define TIME_STEP custom.TimeStep
#define SUBSTEPS 1
#define LOW_PASS_FREQ custom.LowPass
#define PRESSURE custom.Pressure
#define NOISE_ABSORB custom.NoiseAbsorb

#storage sim array<vec2f, ELEMENT_COUNT>

fn SelfInteractionPotential(field: vec2f) -> float {
    let len = length(field);
    let sV = len * (1.0 - len) * (1.0 - len);
    return PRESSURE*sV;
}

fn PotentialAbsorb(field: vec2f, pos: vec2u) -> vec2f {
    let posf = vec2f(pos);
    let d = length(posf- SIZE/2.0);
    let mPos = vec2f(mouse.pos);
    let simPos = screenToSim(mPos);
    var V = 5.0*gaussian(posf - simPos, 50.0) * select(0.0, 1.0, mouse.click == 1);
    var A = 0.0;
    V += SelfInteractionPotential(field);
    //remove background noise
    A += NOISE_ABSORB / (dot(field,field)*1000.0 + 1.0);
    return vec2f(V,A);
}

fn screenToSim(pos: vec2f) -> vec2f {
    let screenSize = vec2f(textureDimensions(screen));
    let minScreenSize = min(screenSize.x, screenSize.y);
    let simSize = vec2f(SIZE);
    let screenCenterPos = (pos - 0.5 * screenSize) / minScreenSize + 0.5;
    let simPos = screenCenterPos * simSize;
    return simPos;
}

fn simToScreen(pos: vec2f) -> vec2f {
    let screenSize = vec2f(textureDimensions(screen));
    let minScreenSize = min(screenSize.x, screenSize.y);
    let simSize = vec2f(SIZE);
    let screenCenterPos = pos / simSize;
    let screenPos = minScreenSize * (screenCenterPos - 0.5) + 0.5 * screenSize;
    return screenPos;
}

fn linearIndex(id: vec2u) -> u32
{
    return id.x + id.y*SIZE;
}

fn indexTo2D(index: u32) -> vec2u
{
    return vec2u(index % SIZE, index / SIZE);
}

fn getAxisIndex(id: u32, group: u32, axis: u32) -> u32
{
    var idx = vec2u();
    idx[axis] = id;
    idx[1-axis] = group; 
    return linearIndex(idx);
}

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

fn gaussian(dx: vec2f, sigma: float) -> float {
    return exp(-dot(dx, dx) / (2.0 * sigma * sigma));
}

fn real_space_update(index: u32)
{
    let id = indexTo2D(index);
    if(time.frame < 5) 
    {
        let uv = (vec2f(id.xy) + 0.5) / SIZE;
        let len = length(vec2f(1.6,1.0)*(uv-vec2f(0.5,0.5)));
        var color = textureSampleLevel(channel0, bilinear, uv, 0).rgb * 0.5;
        sim[index] = 1.02*vec2f(1.0 - smoothstep(0.52,0.55,len), 0);
    } 
    else
    {
        //apply potential energy time evolution operator
        var field = sim[index];
        for(var i = 0; i < SUBSTEPS; i++)
        {
            let dt = TIME_STEP  / float(SUBSTEPS);
            let VA = PotentialAbsorb(field, id.xy);
            let potentialUpdate = expi(-dt * VA.x / (2*H_PLANK)) * exp(-dt * VA.y / (2*H_PLANK));
            field = cmul(potentialUpdate, field);
        }
        sim[index] = field;
    }
}

fn momentum_space_update(index: u32)
{
    let id = indexTo2D(index);
    let freq2 = vec2i(vec2u(ifftshift(id.x), ifftshift(id.y))) - SIZE/2;
    let freq = length(vec2f(freq2));
    //apply kinetic energy time evolution operator
    let kineticUpdate = expi(-TIME_STEP*H_PLANK*freq*freq/(2.0*MASS));
    sim[index] = cmul(kineticUpdate, sim[index]) * (1.0 - smoothstep(LOW_PASS_FREQ-1.0,LOW_PASS_FREQ,freq));
}

#workgroup_count mega_kernel SIZE 1 1
@compute @workgroup_size(WG_SIZE)
fn mega_kernel(
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) workgroup_id: vec3u
) { 
    let linearIndex = local_index + workgroup_id.x * SIZE;
    let currentStep = dispatch.id / DISPATCHES_PER_STEP;
    var currentDispatch = dispatch.id % DISPATCHES_PER_STEP;

    if(currentDispatch == 0) { real_space_update(linearIndex); real_space_update(linearIndex+WG_SIZE); return; }
    currentDispatch-=1;
    if(currentDispatch < AXIS_COUNT) { fft(local_index, workgroup_id.x, currentDispatch, false); return; }
    currentDispatch-=AXIS_COUNT;
    if(currentDispatch == 0) { momentum_space_update(linearIndex); momentum_space_update(linearIndex+WG_SIZE); return; }
    currentDispatch-=1;
    if(currentDispatch < AXIS_COUNT) { fft(local_index, workgroup_id.x, currentDispatch, true); return; }
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
    let pos = vec2f(id.xy);

    let simPos = screenToSim(pos);

    //if out of bounds, do nothing
    if (any(simPos < vec2f(0.0)) || any(simPos >= vec2f(SIZE))) {
        return;
    }

    let idx = linearIndex(vec2u(simPos));
    let value = sim[idx]; 

    let angle = atan2(value.y, value.x);
    let len = dot(value,value);
    let absval = smoothstep(0.0, 0.91, len);
    let col = hsv2rgb_smooth(vec3f((angle + PI) / (2.0 * PI), 0.8, absval));

    // Write the color to the screen
    textureStore(screen, id.xy, vec4f(col, 1.0));
}