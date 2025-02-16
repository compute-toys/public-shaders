#define SIZE 128
#define IT_NUM 6 //floor(log2(N))-1
#define WG_SIZE 64
#define PI 3.14159265
#define INVPI 0.31830988
#define TWO_PI (2.0*PI)
#define FOV 1.0
#define AXIS_COUNT 3
#define GROUP_COUNT 8192 //SIZE*SIZE
#define ELEMENT_COUNT SIZE*SIZE*SIZE

#define DISPATCHES_PER_STEP 8 //2*AXIS_COUNT + 2
#dispatch_count mega_kernel 32 //DISPATCHES_PER_STEP * STEP_COUNT

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

fn PotentialAbsorb(field: vec2f, pos: vec3u) -> vec2f {
    let posf = vec3f(pos);
    let d = length(posf- SIZE/2.0);
    var V = 0.0;
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

fn linearIndex(id: vec3u) -> u32
{
    return id.x + id.y*SIZE + id.z*SIZE*SIZE;
}

fn linearIndexRepeat(id: vec3i) -> u32
{
    let repeat = id - SIZE * (id / SIZE);
    return linearIndex(vec3u(repeat));
}

fn indexTo3D(index: u32) -> vec3u
{
    return vec3u(index % SIZE, (index / SIZE) % SIZE, index / (SIZE * SIZE));
}

fn indexTo2D(index: u32) -> vec2u
{
    return vec2u(index % SIZE, index / SIZE);
}

fn getAxisIndex(id: u32, group: u32, axis: u32) -> u32
{
    let groupGrid = indexTo2D(group);
    if(axis == 0) { return linearIndex(vec3u(id, groupGrid)); }
    if(axis == 1) { return linearIndex(vec3u(groupGrid.x, id, groupGrid.y)); }
    return linearIndex(vec3u(groupGrid, id));
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

fn gaussian(dx: vec3f, sigma: float) -> float {
    return exp(-dot(dx, dx) / (2.0 * sigma * sigma));
}

fn real_space_update(index: u32)
{
    let id = indexTo3D(index);
    if(time.frame < 5) 
    {
        let uv = (vec3f(id) + 0.5) / SIZE;
        let len = length(vec3f(1.6,1.0,1.0)*(uv-vec3f(0.45)));
        sim[index] = 1.5*vec2f(1.0 - smoothstep(0.35,0.4,len), 0);
    } 
    else
    {
        //apply potential energy time evolution operator
        var field = sim[index];
        for(var i = 0; i < SUBSTEPS; i++)
        {
            let dt = TIME_STEP  / float(SUBSTEPS);
            let VA = PotentialAbsorb(field, id);
            let potentialUpdate = expi(-dt * VA.x / (2*H_PLANK)) * exp(-dt * VA.y / (2*H_PLANK));
            field = cmul(potentialUpdate, field);
        }
        sim[index] = field;
    }
}

fn momentum_space_update(index: u32)
{
    let id = indexTo3D(index);
    let freq2 = vec3i(vec3u(ifftshift(id.x), ifftshift(id.y), ifftshift(id.z))) - SIZE/2;
    let freq = length(vec3f(freq2));
    //apply kinetic energy time evolution operator
    let kineticUpdate = expi(-TIME_STEP*H_PLANK*freq*freq/(2.0*MASS));
    sim[index] = cmul(kineticUpdate, sim[index]) * (1.0 - smoothstep(LOW_PASS_FREQ-1.0,LOW_PASS_FREQ,freq));
}

fn mega_kernel_base(local_index: u32, workgroup_id: vec3u) { 
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


#workgroup_count mega_kernel GROUP_COUNT 1 1
@compute @workgroup_size(WG_SIZE)
fn mega_kernel(
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) workgroup_id: vec3u
) { 
   mega_kernel_base(local_index, 2*workgroup_id);
   mega_kernel_base(local_index, 2*workgroup_id+1);
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

//trilinear interpolation
fn sampleSim(pos: vec3f) -> vec2f {
    let posi = vec3i(pos);
    let posf = fract(pos);

    let c00 = mix(sim[linearIndexRepeat(posi)], sim[linearIndexRepeat(posi + vec3i(1, 0, 0))], posf.x);
    let c10 = mix(sim[linearIndexRepeat(posi + vec3i(0, 1, 0))], sim[linearIndexRepeat(posi + vec3i(1, 1, 0))], posf.x);
    let c01 = mix(sim[linearIndexRepeat(posi + vec3i(0, 0, 1))], sim[linearIndexRepeat(posi + vec3i(1, 0, 1))], posf.x);
    let c11 = mix(sim[linearIndexRepeat(posi + vec3i(0, 1, 1))], sim[linearIndexRepeat(posi + vec3i(1, 1, 1))], posf.x);

    let c0 = mix(c00, c10, posf.y);
    let c1 = mix(c01, c11, posf.y);

    return mix(c0, c1, posf.z);
}

fn density(value: vec2f) -> float {
    return smoothstep(0.95, 1.0, dot(value, value));
}
fn sampleDensity(pos: vec3f) -> float {
    return density(sampleSim(pos));
}

fn normal(pos: vec3f) -> vec3f {
    let eps = vec3f(0.5, 0.0, 0.0);
    let grad = vec3f(
        sampleDensity(pos + eps.xyy) - sampleDensity(pos - eps.xyy),
        sampleDensity(pos + eps.yxy) - sampleDensity(pos - eps.yxy),
        sampleDensity(pos + eps.yyx) - sampleDensity(pos - eps.yyx)
    );
    return normalize(grad);
}

struct Camera 
{
  pos: float3,
  cam: float3x3,
  fov: float,
  size: float2
}

struct Ray
{
    ro: float3,
    rd: float3,
}

var<private> camera : Camera;

fn GetCameraMatrix(ang: float2) -> float3x3
{
    let x_dir = float3(cos(ang.x)*sin(ang.y), cos(ang.y), sin(ang.x)*sin(ang.y));
    let y_dir = normalize(cross(x_dir, float3(0.0,1.0,0.0)));
    let z_dir = normalize(cross(y_dir, x_dir));
    return float3x3(-x_dir, y_dir, z_dir);
}

fn SetCamera()
{
    let screen_size = int2(textureDimensions(screen));
    let screen_size_f = float2(screen_size);
    let ang = float2(mouse.pos.xy)*float2(-TWO_PI, PI)/screen_size_f + float2(0.4, 0.4);

    camera.fov = FOV;
    camera.cam = GetCameraMatrix(ang); 
    camera.pos = - (camera.cam*float3(custom.Radius+0.5,0.0,0.0));
    camera.size = screen_size_f;
}

fn GetRay(pos: float2) -> Ray
{
    let p = (pos - camera.size/2.0) / camera.size.y;
    let rd = normalize(camera.cam*float3(1.0, p));
    return Ray(camera.pos, rd);
}

fn unitCubeToSim(pos: vec3f) -> vec3f {
    return (pos + 0.5) * vec3f(SIZE);
}

#define ISO_VALUE 0.9

fn intersectCube(pos: vec3f, dir: vec3f) -> vec2f {
    let tmin = -pos / dir;
    let tmax = (vec3f(SIZE) - pos) / dir;
    let t0 = min(tmin, tmax);
    let t1 = max(tmin, tmax);
    let tminf = max(max(t0.x, t0.y), t0.z);
    let tmaxf = min(min(t1.x, t1.y), t1.z);
    return vec2f(tminf, tmaxf);
}

fn rayMarch(ray: Ray, dither: float) -> vec4f {
    var rho = 0.0;
    var col = vec3f(0.0);
    let step = 1.0;
    var t = dither * step;
    for (var i = 0u; i < 256u; i++) {
        let pos = ray.ro + t * ray.rd;
        t += step;
        let value = sampleSim(pos);
        let d = 50.0*density(value)/SIZE;
        rho += d * step;
        let opacity = exp(-rho);
        let angle = atan2(value.y, value.x);
        let local_col = hsv2rgb_smooth(vec3f((angle + PI) / (2.0 * PI), 0.8, d));
        col += step * local_col * opacity;
        if (rho > 30.0) {
            break;
        }
    }
    return vec4f(col, rho);
}

fn skyTexture(rd: vec3f) -> vec3f
{
    return textureSampleLevel(channel0, bilinear, INVPI * vec2f(.5 * atan2(rd.z, rd.x), asin(rd.y)) + .5, 0.).rgb;
}

fn noiseTexture(pos: vec2f) -> float
{
    return textureSampleLevel(channel1, nearest, pos.xy/1024.0, 0.).r;
}

//display the image
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let pos = vec2f(id.xy);

    // let simPos = screenToSim(pos);

    // //if out of bounds, do nothing
    // if (any(simPos < vec2f(0.0)) || any(simPos >= vec2f(SIZE))) {
    //     return;
    // }


    // let slice = screenToSim(vec2f(mouse.pos)).x;
    // let idx = linearIndex(vec3u(vec2u(simPos), u32(slice)));
    // let value = sim[idx]; 

    // let angle = atan2(value.y, value.x);
    // let len = dot(value,value);
    // let absval = smoothstep(0.3, 1.0, len);
    // let col = hsv2rgb_smooth(vec3f((angle + PI) / (2.0 * PI), 0.8, absval));
    
    SetCamera();
    let dither = noiseTexture(pos);
    var ray = GetRay(pos);
    ray.ro = unitCubeToSim(ray.ro);
    var col = rayMarch(ray, dither);


    //sample background
    let sky = skyTexture(ray.rd);
    col = col + vec4f(sky * exp(-vec3f(1.0,0.5,0.2)*col.a),col.a);

    // Write the color to the screen
    textureStore(screen, id.xy, col);
}