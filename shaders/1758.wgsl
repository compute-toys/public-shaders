

#storage atomic_storage array<atomic<i32>>

//Check Uniforms
//Mode 0 - additive blending (atomicAdd)
//Mode 1 - closest sample (atomicMax)

#define group_size 128
#define group_count 64
#define N (group_size * group_count)

#define FOV 1.8
const PI = 3.14159265;
const TWO_PI = 6.28318530718;

const DEPTH_MIN = 0.05;
const DEPTH_MAX = 5.0;
const DEPTH_BITS = 16u;

var<private> state : uint4;

fn pcg4d(a: uint4) -> uint4
{
	var v = a * 1664525u + 1013904223u;
    v.x += v.y*v.w; v.y += v.z*v.x; v.z += v.x*v.y; v.w += v.y*v.z;
    v = v ^  ( v >> uint4(16u) );
    v.x += v.y*v.w; v.y += v.z*v.x; v.z += v.x*v.y; v.w += v.y*v.z;
    return v;
}

fn rand4() -> float4
{ 
    state = pcg4d(state);
    return float4(state)/float(0xffffffffu); 
}

fn nrand4(sigma: float, mean: float4) -> float4
{
    let Z = rand4();
    return mean + sigma * sqrt(-2.0 * log(Z.xxyy)) * 
           float4(cos(TWO_PI * Z.z),sin(TWO_PI * Z.z),cos(TWO_PI * Z.w),sin(TWO_PI * Z.w));
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

//project to clip space
fn Project(cam: Camera, p: float3) -> float3
{
    let td = distance(cam.pos, p);
    let dir = (p - cam.pos)/td;
    let screen = dir*cam.cam;
    return float3(screen.xy*cam.size.y/(cam.fov*screen.z) + 0.5*cam.size,screen.z*td);
}

struct Ellipsoid {
    c: vec3f, // center
    r: vec3f, // radii
    q: vec4f, // quaternion
};

struct ProjEllipsoidResult {
    center : vec2f,
    axis1  : vec2f,
    axis2  : vec2f,
    size   : vec2f,
    depth : f32,
    projectionFailed : bool,
};

fn parametricEllipse(proj: ProjEllipsoidResult, t: f32) -> vec2f {
    let angle = t;
    let cosA  = cos(angle);
    let sinA  = sin(angle);

    // x = center + (axis1 * (size.x * cos(t))) + (axis2 * (size.y * sin(t)))
    let point = proj.center
               + proj.axis1 * (proj.size.x * cosA)
               + proj.axis2 * (proj.size.y * sinA);

    return point;
}

// A safe divide that mimics your Python safe_divide
fn safe_divide(a: f32, b: f32) -> f32 {
    let epsilon = 1e-2;
    let abs_b = abs(b);
    if abs_b < epsilon {
        // emulate: a / (epsilon * sign(b))
        return a / (epsilon * sign(b));
    }
    return a / b;
}

fn safe_sqrt(x: f32) -> f32 { return sqrt(max(x, 0.0)); }


fn axis_angle_to_quaternion(axis_angle: vec3f) -> vec4f {
    let angle = length(axis_angle);
    let safeAngle = select(angle, 1e-6, angle < 1e-6);
    let axis = axis_angle / safeAngle;
    let half = 0.5 * angle;
    let s    = sin(half);
    let c    = cos(half);
    return vec4f(c, axis.x * s, axis.y * s, axis.z * s);
}

fn qrotate(v: vec3f, q: vec4f) -> vec3f {
    let t = 2.0 * cross(q.yzw, v);
    return v + q.x * t + cross(q.yzw, t);
}

fn conj(q:vec4f) -> vec4f {
    return vec4f(-q.x, q.yzw);
}

fn vecToUnitSpace(v: vec3f, e: Ellipsoid) -> vec3f {
    return qrotate(v, conj(e.q)) / e.r;
}

fn vecToWorldSpace(v: vec3f, e: Ellipsoid) -> vec3f {
    return qrotate(v * e.r, e.q);
}

fn posToUnitSpace(p: vec3f, e: Ellipsoid) -> vec3f {
    return vecToUnitSpace(p - e.c, e);
}

fn posToWorldSpace(p: vec3f, e: Ellipsoid) -> vec3f {
    return vecToWorldSpace(p, e) + e.c;
}

// Extract ellipse from conic parameters (a, b, c, d, e, f)
fn extractEllipse(a: f32, b: f32, c: f32, d: f32, e: f32, f0: f32) -> ProjEllipsoidResult {

    var result = ProjEllipsoidResult();
    result.projectionFailed = true;

    let delta = c*c - 4.0*a*b;
    if (delta >= 0.0) {
        return result;
    }

    // Compute center (h, k)
    let h = safe_divide(2.0*b*d - c*e, delta);
    let k = safe_divide(2.0*a* e - c*d, delta);

    // Plug (h,k) back into the conic to find F'
    let F_prime = a*h*h + b*k*k + c*h*k + d*h + e*k + f0;

    let diff_ba = b - a;
    let sum_ba  = b + a;
    let J       = sqrt(diff_ba * diff_ba + c*c);
    let lambda1 = 0.5 * (sum_ba + J);
    let lambda2 = 0.5 * (sum_ba - J);

    // Angles for the ellipse axes
    let r  = safe_divide(diff_ba, c);
    let ca = 0.5 * sign(c) / sqrt(1.0 + r*r);
    let ch = sqrt(0.5 + ca) * sqrt(0.5);
    let sh = sqrt(0.5 - ca) * sqrt(0.5) * sign(diff_ba);

    let cos_theta = ch - sh;
    let sin_theta = ch + sh;

    let denom1 = -safe_divide(F_prime, lambda1);
    let denom2 = -safe_divide(F_prime, lambda2);

    if (denom1 <= 0.0 || denom2 <= 0.0) {
        return result;
    }

    let a1 = sqrt(denom1);
    let a2 = sqrt(denom2);

    result.center           = vec2f(h, k);
    result.axis1            = vec2f(cos_theta, sin_theta);
    result.axis2            = vec2f(-sin_theta, cos_theta);
    result.size             = vec2f(a1, a2);
    result.projectionFailed = false;

    return result;
}

// Project an ellipsoid into screen-space (building conic parameters, then extracting ellipse).
fn projEllipsoid(ellips: Ellipsoid, cam: Camera) -> ProjEllipsoidResult {
    // 1) Clamp radii
    let rad_max    = max(ellips.r.x, max(ellips.r.y, ellips.r.z));
    let rad_clamped = clamp(
        ellips.r,
        vec3f(rad_max * 0.01, rad_max * 0.01, rad_max * 0.01),
        vec3f(rad_max,       rad_max,       rad_max)
    );
    let e0 = Ellipsoid(ellips.c, rad_clamped, ellips.q);

    let v0 = vecToUnitSpace(cam.cam[0], e0);
    let v1 = vecToUnitSpace(cam.cam[1], e0);
    let v2 = vecToUnitSpace(cam.cam[2], e0);
    let ro = posToUnitSpace(cam.pos, e0);
    let focal_length = 1.0 / cam.fov;

    let s  = dot(ro, ro) - 1.0;
    let vv0 = ro * dot(v0, ro) - v0 * s;
    let vv1 = ro * dot(v1, ro) - v1 * s;
    let vv2 = ro * dot(v2, ro) - v2 * s;

    let a = dot(v0, vv0);
    let b = dot(v1, vv1);
    let c = 2.0 * dot(v0, vv1);  
    let d = 2.0 * focal_length * dot(v0, vv2);
    let e = 2.0 * focal_length * dot(v1, vv2);
    let f = focal_length * focal_length * dot(v2, vv2);

    var ellipse = extractEllipse(a, b, c, d, e, f);
    let min_res = min(cam.size.x, cam.size.y);
    let i = ellipse.center.x * min_res + 0.5 * cam.size.x;
    let j = ellipse.center.y * min_res + 0.5 * cam.size.y;
    ellipse.center = vec2f(i, j);
    ellipse.size = ellipse.size * min_res;
    ellipse.depth = dot(cam.cam[2], e0.c - cam.pos);

    return ellipse;
}

@compute @workgroup_size(16, 16)
fn Clear(@builtin(global_invocation_id) id: uint3) {
    let screen_size = int2(textureDimensions(screen));
    let idx0 = int(id.x) + int(screen_size.x * int(id.y));

    atomicStore(&atomic_storage[idx0*4+0], 0);
    atomicStore(&atomic_storage[idx0*4+1], 0);
    atomicStore(&atomic_storage[idx0*4+2], 0);
    atomicStore(&atomic_storage[idx0*4+3], 0);
}

fn Pack(a: uint, b: uint) -> int
{
    return int(a + (b << (31u - DEPTH_BITS)));
}

fn Unpack(a: int) -> float
{
    let mask = (1 << (DEPTH_BITS - 1u)) - 1;
    return float(a & mask)/256.0;
}

fn ClosestPoint(color: float3, depth: float, index: int)
{
    let inverseDepth = 1.0/depth;
    let scaledDepth = (inverseDepth - 1.0/DEPTH_MAX)/(1.0/DEPTH_MIN - 1.0/DEPTH_MAX);
    
    if(scaledDepth > 1.0 || scaledDepth < 0.0) 
    {
        return;
    }

    let uintDepth = uint(scaledDepth*float((1u << DEPTH_BITS) - 1u));
    let uintColor = uint3(color * 256.0);
    
    atomicMax(&atomic_storage[index*4+0], Pack(uintColor.x, uintDepth));
    atomicMax(&atomic_storage[index*4+1], Pack(uintColor.y, uintDepth));
    atomicMax(&atomic_storage[index*4+2], Pack(uintColor.z, uintDepth));
}

fn AdditiveBlend(color: float3, depth: float, index: int)
{
    let scaledColor = 256.0 * color/depth;

    atomicAdd(&atomic_storage[index*4+0], int(scaledColor.x));
    atomicAdd(&atomic_storage[index*4+1], int(scaledColor.y));
    atomicAdd(&atomic_storage[index*4+2], int(scaledColor.z));
}

fn RasterizePixel(color: float3, depth: float, pos: int2)
{
    let screen_size = int2(camera.size);
    if(pos.x < 0 || pos.x >= screen_size.x || 
        pos.y < 0 || pos.y >= screen_size.y || depth < 0.0)
    {
        return;
    }

    let idx = pos.x + screen_size.x * pos.y;
    
    if(custom.Mode < 0.5)
    {
        AdditiveBlend(color, depth, idx);
    }
    else
    {
        ClosestPoint(color, depth, idx);
    }
}

fn RasterizeEllipsoid(color: float3, ellipsoid: Ellipsoid)
{
    let projection = projEllipsoid(ellipsoid, camera);

    if(projection.projectionFailed) { return; }

    for(var phi = 0.0; phi < TWO_PI; phi += 0.01)
    {
        let p = parametricEllipse(projection, phi);
        RasterizePixel(color, projection.depth, int2(p));
    }
}

#workgroup_count Rasterize group_count 1 1
@compute @workgroup_size(group_size)
fn Rasterize(@builtin(global_invocation_id) id: uint3) {

    SetCamera();

    //RNG state
    state = uint4(id.x, id.y, id.z, 0u*time.frame);

    let rand = nrand4(1.0, float4(0.0));
    var pos = 0.2*rand.xyz;
    let col = float3(0.5 + 0.5*sin(10.0*pos));

    let sec = 5.0+0.5*time.elapsed;
    //move points along sines
    pos += sin(float3(2.0,1.0,1.5)*sec)*0.1*sin(30.0*0.3*pos);
    pos += sin(float3(2.0,1.0,1.5)*sec)*0.02*sin(30.0*0.2*pos.zxy);
    let rot = nrand4(1.0, float4(0.0)).xyz;
    let q = axis_angle_to_quaternion(rot);
    let e = Ellipsoid(pos, custom.R*vec3f(custom.A,custom.B,custom.C), q);
    RasterizeEllipsoid(col, e);
}

fn Sample(pos: int2) -> float3
{
    let screen_size = int2(textureDimensions(screen));
    let idx = pos.x + screen_size.x * pos.y;

    var color: float3;
    if(custom.Mode < 0.5)
    {
        let x = float(atomicLoad(&atomic_storage[idx*4+0]))/(256.0);
        let y = float(atomicLoad(&atomic_storage[idx*4+1]))/(256.0);
        let z = float(atomicLoad(&atomic_storage[idx*4+2]))/(256.0);
        
        color = tanh(0.025*float3(x,y,z));
    }
    else
    {
        let x = Unpack(atomicLoad(&atomic_storage[idx*4+0]));
        let y = Unpack(atomicLoad(&atomic_storage[idx*4+1]));
        let z = Unpack(atomicLoad(&atomic_storage[idx*4+2]));
        
        color = float3(x,y,z);
    }

    return abs(color);
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) 
{
    let screen_size = uint2(textureDimensions(screen));

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = float2(float(id.x) + .5, float(id.y) + .5);

  
    let color = Sample(int2(id.xy));

    // Output to screen (linear colour space)
    textureStore(screen, int2(id.xy), float4(color, 1.));
}

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

#define INITIALIZED_OK 100

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
}



