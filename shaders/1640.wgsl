#define SIZE 128
#define IT_NUM 6 //floor(log2(N))-1
#define WG_SIZE 64
#define PI 3.14159265
#define INVPI 0.31830988
#define TWO_PI (2.0*PI)
#define FOV 1.0
#define AXIS_COUNT 3
#define GROUP_COUNT 8192 //SIZE*SIZE/2
#define ELEMENT_COUNT SIZE*SIZE*SIZE

#define DISPATCHES_PER_STEP 8 //2*AXIS_COUNT + 2
#define STEP_COUNT 2
#define TOTAL_DISPATCH_COUNT 16 //DISPATCHES_PER_STEP * STEP_COUNT
#dispatch_count mega_kernel TOTAL_DISPATCH_COUNT 

#define CUR_INDEX (time.frame * STEP_COUNT + dispatch.id/DISPATCHES_PER_STEP)

#define MASS 1000.0
#define H_PLANK 1.0
#define TIME_STEP custom.TimeStep
#define SUBSTEPS 1
#define LOW_PASS_FREQ custom.LowPass
#define PRESSURE custom.Pressure
#define NOISE_ABSORB custom.NoiseAbsorb
#define GRAVITY -0.005

#define ISO_VALUE custom.Isovalue
#define ABSORPTION (custom.Absorption*vec3(0.4, 0.1, 0.05))
#define IOR custom.IOR

#define TERMINAL_ROWS 30
#define TERMINAL_COLS 96

#define INITIALIZED_OK 100

#storage sim array<vec2f, ELEMENT_COUNT>
#storage gstate GlobalState

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

struct GlobalState
{
    initialized: uint,

    pos: float3,
    dposdt: float3,

    rot: float4,
    drotdt: float3,

    mouse: float2,
    dmousedt: float2,

    prevpos: float3,
    prevrot: float4,
    prevmouse: float2,

    camera: float3x3,
    fov: float,

    terminal_grid: array<array<uint,TERMINAL_COLS>,TERMINAL_ROWS>,

    simulation_on: int
}

var<private> camera : Camera;

fn matrixCompMult(a: float3x3, b: float3x3) -> float3x3 
{
    return float3x3(a[0] * b[0], a[1] * b[1], a[2] * b[2]);
}

fn outerProduct(a: float3, b: float3) -> float3x3 
{
    return float3x3(a[0] * b, a[1] * b, a[2] * b);
}

// Return quaternion from axis and angle
fn aa2q(axis: float3, ang: float) -> float4 
{
    let g: float2 = float2(sin(ang), cos(ang)) * 0.5;
    return normalize(float4(axis * g.x, g.y));
}

// Return AxisAngle of NORMALIZED quaternion input
fn q2aa(q: float4) -> float4 
{
    return float4(q.xyz / sqrt(1.0 - q.w * q.w), acos(q.w) * 2.0);
}

// Return q2, rotated by q1, order matters (is non commutative) : (aka quaternion multiplication == AxisAngleRotation)
fn qq2q(q1: float4, q2: float4) -> float4
{
    return float4(q1.xyz * q2.w + q2.xyz * q1.w + cross(q1.xyz, q2.xyz), (q1.w * q2.w) - dot(q1.xyz, q2.xyz));
}

fn float3x3f(value: f32) -> float3x3
{
    return float3x3(
        float3(value, 0.0, 0.0),
        float3(0.0, value, 0.0),
        float3(0.0, 0.0, value)
    );
}

fn float3x3d(value: f32) -> float3x3
{
    return float3x3(
        float3(value, value, value),
        float3(value, value, value),
        float3(value, value, value),
    );
}

fn quaternion_to_matrix(quat: float4) -> float3x3 {
    var m: float3x3 = float3x3(
        float3(0.0, 0.0, 0.0),
        float3(0.0, 0.0, 0.0),
        float3(0.0, 0.0, 0.0)
    );

    let x = quat.x; let y = quat.y; let z = quat.z; let w = quat.w;
    let x2 = x + x;  let y2 = y + y;  let z2 = z + z;
    let xx = x * x2; let xy = x * y2; let xz = x * z2;
    let yy = y * y2; let yz = y * z2; let zz = z * z2;
    let wx = w * x2; let wy = w * y2; let wz = w * z2;

    m[0] = float3(1.0 - (yy + zz), xy - wz, xz + wy);
    m[1] = float3(xy + wz, 1.0 - (xx + zz), yz - wx);
    m[2] = float3(xz - wy, yz + wx, 1.0 - (xx + yy));

    return m;
}

#include <string>

var<private> terminal_cursor: uint2;


fn terminal_write_char(ascii: uint)
{
    if (ascii == 0) { // NULL
    } else if (ascii == 0x0a) { // '\n'
        terminal_cursor.x = 0u;
        terminal_cursor.y += 1u;
    } else {
        gstate.terminal_grid[terminal_cursor.y][terminal_cursor.x] = ascii;
        terminal_cursor.x += 1u;
    }
}

fn terminal_write(s: String) {
    for (var i = 0u; i < s.len; i++) {
        let ascii = s.chars[i];
        terminal_write_char(ascii);
    }
}

fn terminal_writei_auto(x: float)
{   
    if(x<0.0)
    {
        terminal_write_char(0x2d); //minus sign
    }

    let d = max(int(ceil(log(abs(x))/log(10.0))), 1);
    var d0 = pow(10, float(d-1));
    for(var i = 0; i < d; i++)
    {
        let digit = uint(abs(x)/d0); 
        terminal_write_char(0x30 + (digit % 10));
        d0 /= 10.0;
    }
}

fn terminal_writei(x: float, d: int)
{   
    var d0 = pow(10, float(d-1));
    for(var i = 0; i < d; i++)
    {
        let digit = uint(abs(x)/d0); 
        terminal_write_char(0x30 + (digit % 10));
        d0 /= 10.0;
    }
}

fn terminal_writef(x: float, fraction_digits: int)
{
    let m = pow(10.0, float(fraction_digits));
    terminal_writei_auto(x);
    terminal_write(".");
    terminal_writei(fract(abs(x)) * m, fraction_digits);
}

fn terminal_write2f(x: float2, fd: int)
{
    terminal_write("(");
    terminal_writef(x.x,fd);
    terminal_write(", ");
    terminal_writef(x.y,fd);
    terminal_write(")");
}

fn terminal_write3f(x: float3, fd: int)
{
    terminal_write("(");
    terminal_writef(x.x,fd);
    terminal_write(", ");
    terminal_writef(x.y,fd);
    terminal_write(", ");
    terminal_writef(x.z,fd);
    terminal_write(")");
}

fn terminal_write4f(x: float4, fd: int)
{
    terminal_write("(");
    terminal_writef(x.x,fd);
    terminal_write(", ");
    terminal_writef(x.y,fd);
    terminal_write(", ");
    terminal_writef(x.z,fd);
    terminal_write(", ");
    terminal_writef(x.w,fd);
    terminal_write(")");
}

fn terminal_writevec2(name: String, x: float2)
{
    terminal_write(name);
    terminal_write(": ");
    terminal_write2f(x,3);
    terminal_write("\n");
}

fn terminal_writevec3(name: String, x: float3)
{
    terminal_write(name);
    terminal_write(": ");
    terminal_write3f(x,3);
    terminal_write("\n");
}

fn terminal_writevec4(name: String, x: float4)
{
    terminal_write(name);
    terminal_write(": ");
    terminal_write4f(x,3);
    terminal_write("\n");
}

fn terminal_clear() {
    for (var i = 0; i < TERMINAL_ROWS; i += 1) {
        for (var j = 0; j < TERMINAL_COLS; j += 1) {
            gstate.terminal_grid[i][j] = 0;
        }
    }
}


#workgroup_count UpdateGlobalState 1 1 1
@compute @workgroup_size(1)
fn UpdateGlobalState() 
{
    let dt: f32 = 0.001;
    let speed: f32 = 1000.0;
    let mouse_sens: f32 = 3.0;
    let roll_speed: f32 = 3.0;

    if(gstate.initialized != INITIALIZED_OK)
    {
        gstate.pos = float3(0.0, 0.0, 0.0);
        gstate.dposdt = float3(0.0);
        gstate.rot = float4(0.0, 0.0, 0.0, 1.0);
        gstate.drotdt = float3(0.0);
        gstate.mouse = float2(0.0);
        gstate.dmousedt = float2(0.0);
        gstate.prevpos = float3(0.0);
        gstate.prevrot = float4(0.0, 0.0, 0.0, 1.0);
        gstate.prevmouse = float2(0.0);
        gstate.camera = float3x3f(0.0);
        gstate.fov = FOV;
        gstate.simulation_on = 1;

        gstate.initialized = INITIALIZED_OK;
    }

    // Get velocities
    var cv: float3 = gstate.dposdt;
    let cav: float3 = gstate.drotdt;

    // Update position
    if(keyDown(87u)) { cv += speed * dt * gstate.camera * float3(0.0,0.0,1.0); } // W
    if(keyDown(83u)) { cv += speed * dt * gstate.camera * float3(0.0,0.0,-1.0); } // S
    if(keyDown(65u)) { cv += speed * dt * gstate.camera * float3(-1.0,0.0,0.0); } // A
    if(keyDown(68u)) { cv += speed * dt * gstate.camera * float3(1.0,0.0,0.0); } // D

    gstate.pos += dt * cv;
    cv += -cv * tanh(100.0 * dt);

    // Update camera orientation
    let dmouse: float2 = dt * mouse_sens * (float2(mouse.pos) - gstate.prevmouse);

    if(length(dmouse) < 0.1)
    {
        // Rotate around y axis
        gstate.rot = qq2q(gstate.rot, aa2q(gstate.camera * float3(0,1,0), -dmouse.x));
        // Rotate around x axis
        gstate.rot = qq2q(gstate.rot, aa2q(gstate.camera * float3(1,0,0), dmouse.y));
    }

   // Roll camera
    if(keyDown(81u)) { gstate.rot = qq2q(gstate.rot, aa2q(gstate.camera * float3(0,0,1), roll_speed * dt)); } // Q
    if(keyDown(69u)) { gstate.rot = qq2q(gstate.rot, aa2q(gstate.camera * float3(0,0,1), -roll_speed * dt)); } // E

    // Turn on and off simulation (M key)
    if(keyDown(77u)) { gstate.simulation_on = 1 - gstate.simulation_on; }

    gstate.rot = normalize(gstate.rot);
    gstate.dposdt = cv;

    // Update previous states
    gstate.prevpos = gstate.pos;
    gstate.prevrot = gstate.rot;
    gstate.prevmouse = float2(mouse.pos);

    // Update camera orientation
    gstate.camera = quaternion_to_matrix(gstate.rot);

    terminal_clear();
    terminal_writevec3("Camera position", gstate.pos);
    terminal_writevec3("Camera velocity", gstate.dposdt);
    terminal_writevec3("Camera x", gstate.camera * float3(1,0,0));
    terminal_writevec3("Camera y", gstate.camera * float3(0,1,0));
    terminal_writevec3("Camera z", gstate.camera * float3(0,0,1));
}




fn cubicWellPotential(pos: vec3f, T: f32, K: f32) -> f32 {
    let d_x = min(pos.x, SIZE - pos.x);
    let d_y = min(pos.y, SIZE - pos.y);
    let d_z = min(pos.z, SIZE - pos.z);
    let d = min(d_x, min(d_y, d_z));
    return max(0.0, K * (1.0 - (d / T)));
}

fn SelfInteractionPotential(field: vec2f) -> float {
    let len = length(field);
    let sV = len * (1.0 - len) * (1.0 - len);
    return PRESSURE*sV;
}

fn PotentialAbsorb(field: vec2f, pos: vec3u) -> vec2f {
    let posf = vec3f(pos);
    let d = length(posf- SIZE/2.0);
    var V = cubicWellPotential(posf, 8.0, 8.0) + posf.y * GRAVITY;
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
    let repeat = id;// - SIZE * (id / SIZE);
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

fn blobIC(pos: vec3f, bpos:vec3f, vel:vec3f, rad: float) -> vec2f
{
    let thickness = 5.0;
    let dx = pos - bpos;
    let dist = length(dx);
    let density = 1.0 - smoothstep(rad, rad + thickness, dist);
    return 1.0 * density * expi(-dot(vel, dx));
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
    if(gstate.simulation_on == 0) { return; }

    let id = indexTo3D(index);
    let posf = vec3f(id);
    if(time.frame < 5) 
    {
        let uv = (vec3f(id) + 0.5) / SIZE;
        let len = length(vec3f(1.6,1.0,1.0)*(uv-vec3f(0.45)));
        var InitCond = blobIC(posf, vec3f(0.5, 0.5, 0.5)*SIZE, vec3f(0.0, 0.0, 0.0), 40.0);
        sim[index] = InitCond;
    } 
    else
    {
        //apply potential energy time evolution operator
        var field = sim[index];
        

        if(CUR_INDEX%1000 == 0)
        {
           field += blobIC(posf, vec3f(0.5, 0.4, 0.5)*SIZE, vec3f(0.0, 0.0, 0.0), 24.0);
        }

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
    if(gstate.simulation_on == 0) { return; }

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
    return dot(value, value);
}
fn sampleDensity(pos: vec3f) -> float {
    return density(sampleSim(pos));
}

fn normal(pos: vec3f) -> vec3f {
    let eps = vec3f(1.0, 0.0, 0.0);
    let grad = vec3f(
        sampleDensity(pos + eps.xyy) - sampleDensity(pos - eps.xyy),
        sampleDensity(pos + eps.yxy) - sampleDensity(pos - eps.yxy),
        sampleDensity(pos + eps.yyx) - sampleDensity(pos - eps.yyx)
    );
    return -normalize(grad);
}


fn GetCameraMatrix(ang: float2) -> float3x3
{
    let x_dir = float3(cos(ang.x)*sin(ang.y), cos(ang.y), sin(ang.x)*sin(ang.y));
    let y_dir = normalize(cross(x_dir, float3(0.0,1.0,0.0)));
    let z_dir = normalize(cross(y_dir, x_dir));
    return float3x3(-x_dir, y_dir, z_dir);
}

fn SetCamera()
{
    camera.fov = gstate.fov;
    camera.cam[0] = gstate.camera * float3(1,0,0);
    camera.cam[1] = gstate.camera * float3(0,1,0);
    camera.cam[2] = gstate.camera * float3(0,0,1);
    camera.pos = gstate.pos;
    camera.size = float2(textureDimensions(screen));
}

fn GetRay(pos: float2) -> Ray
{
    let p = (pos - camera.size/2.0) / camera.size.y;
    let rd = normalize(camera.cam*float3(p,1.0));
    return Ray(camera.pos, rd);
}

fn unitCubeToSim(pos: vec3f) -> vec3f {
    return (pos + 0.5) * vec3f(SIZE);
}



fn intersectCube(pos: vec3f, dir: vec3f) -> vec2f {
    let bmin = vec3f(0.0);
    let bmax = vec3f(SIZE);
    let t1 = (bmin - pos) / dir;
    let t2 = (bmax - pos) / dir;
    let tmin = min(t1, t2);
    let tmax = max(t1, t2);
    let tminmax = min(tmax.x, min(tmax.y, tmax.z));
    let tmaxmin = max(tmin.x, max(tmin.y, tmin.z));
    return vec2f(tmaxmin, tminmax);
}

fn skyTexture(rd: vec3f) -> vec3f
{
    return textureSampleLevel(channel0, bilinear, INVPI * vec2f(.5 * atan2(rd.z, rd.x), asin(rd.y)) + .5, 0.).rgb;
}

fn noiseTexture(pos: vec2f) -> float
{
    return textureSampleLevel(channel1, nearest, pos.xy/1024.0, 0.).r;
}

//reflection
fn fresnel(h: vec3f, n: vec3f, n1: float, n2: float) -> float {
    let f0 = 0.05;
    return f0 + (1.0 - f0) * pow(dot(h, n), 5.0);
}

fn rayMarch(ray: Ray, dither: float) -> vec4f {
    var step = 1.0;
    var t = 0.001+dither * step;
    var tt = 0.0;
    var prev_d = 0.0;
    let value = sampleSim(ray.ro);
    var d = density(value);
    var inside = d > ISO_VALUE;
    var cray = ray;
    var incoming = vec3f(0.0);
    var absorption = vec3f(0.0);
    var bounces = 0.0;
    for (var i = 0u; i < 1024u; i++) {
        let pos = cray.ro + t * cray.rd;
        t += step;
        //if out of bounds, stop
        if (any(pos < vec3f(0.0)) || any(pos >= vec3f(SIZE))) {
            tt+=t;
            incoming += skyTexture(cray.rd) * exp(-absorption);
            break;
        }
        let value = sampleSim(pos);
        var d = density(value);
        d = select(d, 2.0*ISO_VALUE - d, inside);
        step = abs(1.0 - d) * 0.5;
        if (d > ISO_VALUE) {
            //find intersection point assuming linear density interpolation
            let t0 = t - step;
            let t1 = t;
            let tmid = t0 + (ISO_VALUE - prev_d) * step / (d - prev_d);
            let ipos = cray.ro + tmid * cray.rd;
            let normal = normal(ipos);

            //refraction ray direction
            let n = select(normal, -normal, inside);
            let n1 = select(1.0, IOR, inside);
            let n2 = select(IOR, 1.0, inside);
            let refDir = refract(cray.rd, n, n1 / n2);

            let reflPos = cray.ro + (t0 - 1.0) * cray.rd;
            let reflDir = reflect(cray.rd, n);

            if(inside) {
                absorption += ABSORPTION * tmid;
            } 

            //full internal reflection
            if(length(refDir) < 0.5)
            {
                cray.ro = reflPos;
                cray.rd = reflDir;
                //still inside
            }
            else
            {
                let h = normalize(cray.rd + n);
                let fres = fresnel(h, n, n1, n2);
                let relf_amount = fres;
                let refr_amount = 1.0 - fres;

                //just sample the sky
                incoming += skyTexture(reflDir) * exp(-absorption) * relf_amount;
                absorption -= log(1e-6+refr_amount);

                cray.ro = pos;
                cray.rd = refDir;
                inside = !inside;
            }

            tt += t;
            t = 0.0;
            bounces+=1.0;

            if(bounces > 6.0) {break; }
        }
        prev_d = d;
    }
    //return vec4f(vec3f(bounces/5.0), 1.0);
    return vec4f(incoming, 1.0);
}

fn traceRay(ray: Ray, dither: float) -> vec4f {
    //box intersection
    let t = intersectCube(ray.ro, ray.rd);
    let tmin = max(t.x, 1.0);
    let tmax = t.y;
    if (tmin > tmax) {
        return vec4f(skyTexture(ray.rd), 1.0);
    }
    let newRay = Ray(ray.ro + tmin * ray.rd, ray.rd);
    return rayMarch(newRay, dither);
}


fn terminal_render(pos: uint2) -> float4 {
    let screen_size = uint2(textureDimensions(screen));
    let aspect = float(screen_size.y) / float(screen_size.x) * float(TERMINAL_COLS) / float(TERMINAL_ROWS);
    let texel = float(TERMINAL_ROWS) / float(screen_size.y);
    var uv = float2(pos) * float2(aspect, 1.) * texel;
    let ascii = gstate.terminal_grid[int(uv.y)][int(uv.x)];

    if (0x20 < ascii && ascii < 0x80) { // printable character
        uv = fract(uv);
        uv.x = (uv.x - .5) / aspect + .5; // aspect ratio correction
        uv += float2(uint2(ascii % 16u, ascii / 16u)); // character lookup
        let sdf = textureSampleLevel(channel1, trilinear, uv / 16., 0.).a;

        var col = float4(0);
        col = mix(col, float4(0,0,0,1), smoothstep(.525 + texel, .525 - texel, sdf));
        col = mix(col, float4(1,1,1,1), smoothstep(.490 + texel, .490 - texel, sdf));
        return col;
    }
    return float4(0);
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
    let dither = 0.0*noiseTexture(pos);
    var ray = GetRay(pos);
    ray.ro = unitCubeToSim(ray.ro);
    var col = traceRay(ray, dither);

    let text = terminal_render(id.xy);
    col = mix(col, text, text.a);

    // Write the color to the screen
    textureStore(screen, id.xy, col);
}