#storage atomic_storage array<atomic<i32>>


#define INITIALIZED_OK 100
const MaxSamples = 32.0;
const FOV = 0.8;
const PI = 3.14159265;
const TWO_PI = 6.28318530718;
const STEP = 0.01;
const LARGENUM = 1e10;
const ATOMIC_SCALE = 64.0;
const BULB_POWER_DELTA_MAX = 1.0;
const BULB_MAX_POWER = 8.0;

struct Camera 
{
  pos: float3,
  cam: float3x3,
  fov: float,
  size: float2
}

#storage gstate GlobalState

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
}

var<private> camera : Camera;
var<private> state : uint4;

fn SetCamera()
{
    camera.fov = gstate.fov;
    camera.cam[0] = gstate.camera * float3(1,0,0);
    camera.cam[1] = gstate.camera * float3(0,1,0);
    camera.cam[2] = gstate.camera * float3(0,0,1);
    camera.pos = gstate.pos;
    camera.size = float2(textureDimensions(screen));
}

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


#workgroup_count UpdateGlobalState 1 1 1
@compute @workgroup_size(1)
fn UpdateGlobalState() 
{
    let dt: f32 = 0.001;
    let speed: f32 = 1000.0;
    let mouse_sens: f32 = 1.0;
    let roll_speed: f32 = 3.0;

    if(gstate.initialized != INITIALIZED_OK)
    {
        gstate.pos = float3(0.0,0.0, -2.0);
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
    cv += -cv * tanh(300.0 * dt);

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

    gstate.rot = normalize(gstate.rot);
    gstate.dposdt = cv;

    // Update previous states
    gstate.prevpos = gstate.pos;
    gstate.prevrot = gstate.rot;
    gstate.prevmouse = float2(mouse.pos);

    // Update camera orientation
    gstate.camera = quaternion_to_matrix(gstate.rot);
}

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

fn udir(rng: float2) -> float3
{
    let r = float2(2.*PI*rng.x, acos(2.*rng.y - 1.0));
    let c = cos(r);
    let s = sin(r);
    return float3(c.x*s.y, s.x*s.y, c.y);
}

fn disk(rng: float2) -> float2
{
    return float2(sin(TWO_PI*rng.x), cos(TWO_PI*rng.x))*sqrt(rng.y);
}

fn Rotate(t: float) -> float2x2 
{
    return float2x2(
        cos(t), sin(t), 
      - sin(t), cos(t), 
    );
}

fn RotXY(x: float3, t: float) -> float3
{
    return float3(Rotate(t)*x.xy, x.z);
} 

fn GetCameraMatrix(ang: float2) -> float3x3
{
    let x_dir = float3(cos(ang.x)*sin(ang.y), cos(ang.y), sin(ang.x)*sin(ang.y));
    let y_dir = normalize(cross(x_dir, float3(0.0,1.0,0.0)));
    let z_dir = normalize(cross(y_dir, x_dir));
    return float3x3(-x_dir, y_dir, z_dir);
}


//project to clip space
fn Project(cam: Camera, p: float3) -> float3
{
    let td = distance(cam.pos, p);
    let dir = (p - cam.pos)/td;
    let screen = dir*cam.cam;
    return float3(screen.xy*cam.size.y/(cam.fov*screen.z) + 0.5*cam.size,screen.z*td);
}

const max_iterations = 256;
const color_thresholds = float4(255.0, 130.0, 80.0, 255.0);


fn AdditiveBlend(color: float3, projected: float2)
{
    let screen_size = int2(camera.size);
    let screen_coord = int2(projected.xy);
    let index = screen_coord.x + screen_size.x * screen_coord.y;
    let scaledColor = int3(floor(ATOMIC_SCALE*color + rand4().xyz));

    if(scaledColor.x>0)
    {
        atomicAdd(&atomic_storage[index*4+0], scaledColor.x);
    }
       
    if(scaledColor.y>0)
    {
        atomicAdd(&atomic_storage[index*4+1], scaledColor.y);
    }

    if(scaledColor.z>0)
    {
        atomicAdd(&atomic_storage[index*4+2], scaledColor.z);
    }
}

fn RasterizePoint(pos: float3, color: float3)
{
    let screen_size = int2(camera.size);
    let projectedPos = Project(camera, pos);
    let screenCoord = int2(projectedPos.xy);

    //outside of our view
    if(screenCoord.x < 0 || screenCoord.x >= screen_size.x || 
        screenCoord.y < 0 || screenCoord.y >= screen_size.y || projectedPos.z < 0.0)
    {
        return;
    }

    let depth = projectedPos.z;
    let new_color = color * exp(-0.0*depth)/(depth*depth+1e-8);

    var rad = clamp(0.1/depth, 1.0, 8.0);
    var radi = round(rad);
    rad = max(rad*0.15, 0.33);
    if(depth < 0.2) {radi += 1.0;}

    let screenCoordf = float2(screenCoord);
    
    for(var x = -radi; x <= radi; x+=1.0)
    {
        for(var y = -radi; y <= radi; y+=1.0)
        {
            let dx = vec2(x,y);
            let projPos = screenCoordf + dx;
            let dist = distance(projPos, projectedPos.xy)/rad;
            let w = exp(-dist)/(rad*rad);
            AdditiveBlend(new_color*w, projPos);
        }
    }

}


fn saturate(x: f32) -> f32 {
    return min(1.0, max(0.0, x));
}

fn saturate_vec3(x: vec3<f32>) -> vec3<f32> {
    return min(vec3<f32>(1.0, 1.0, 1.0), max(vec3<f32>(0.0, 0.0, 0.0), x));
}

fn bump3y(x: vec3<f32>, yoffset: vec3<f32>) -> vec3<f32> {
    var y: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0) - x * x;
    y = saturate_vec3(y - yoffset);
    return y;
}

fn spectral_zucconi(w: f32) -> vec3<f32> {
    let x: f32 = saturate((w - 400.0) / 300.0);

    let cs: vec3<f32> = vec3<f32>(3.54541723, 2.86670055, 2.29421995);
    let xs: vec3<f32> = vec3<f32>(0.69548916, 0.49416934, 0.28269708);
    let ys: vec3<f32> = vec3<f32>(0.02320775, 0.15936245, 0.53520021);

    return bump3y(cs * (x - xs), ys);
}

fn hue(v: float) -> float3 {
    return .6 + .6 * cos(6.3 * v + float3(0.,23.,21.));
}

fn buddhabrot(iters: i32) -> vec4<f32> {
    var z: vec3<f32> = vec3<f32>(0.01, 0.01, 0.01);
    var c: vec3<f32> = (rand4().xyz - 0.5) * 1.2;

    var i: i32 = 0;
    let BULB_POWER = custom.BulbPower * BULB_MAX_POWER + 5.0*(0.5*sin(0.25*custom.AnimationSpeed*time.elapsed)+0.5);
    let BULB_POWER_DELTA = (custom.PowerDelta - 0.5) * BULB_POWER_DELTA_MAX;
    loop {
        if (i >= iters) { break; }

        let r: f32 = length(z);
        let b: f32 = (BULB_POWER + BULB_POWER_DELTA) * acos(z.y / r);
        let a: f32 =(BULB_POWER + BULB_POWER_DELTA) * atan2(z.x, z.z);
        z = c + pow(r, (BULB_POWER + BULB_POWER_DELTA)) * vec3<f32>(sin(b) * sin(a), cos(b), sin(b) * cos(a));

        if (length(z) > 4.0) {
            break;
        }
        i = i + 1;
    }

    if (i >= iters) {
        return vec4<f32>(1e5, 1e5, 1e5, 1e5);
    }

    z = vec3<f32>(0.01, 0.01, 0.01);

    for (var j: i32 = 0; j <= 64; j = j + 1) {
        let r: f32 = length(z);
        let b: f32 = BULB_POWER * acos(z.y / r);
        let a: f32 = BULB_POWER * atan2(z.x, z.z);
        z = c + pow(r, BULB_POWER) * vec3<f32>(sin(b) * sin(a), cos(b), sin(b) * cos(a));

        var color = spectral_zucconi(460 + 4.0*float(j)) + 0.025*float3(1.0,1.0,1.0);
        color = pow(color, vec3(1.0));
        RasterizePoint(z, color/(custom.Samples*MaxSamples + 1.0));
    }
  

    return vec4<f32>(z, f32(i));
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

@compute @workgroup_size(16, 16)
fn Rasterize(@builtin(global_invocation_id) id: uint3) 
{
    SetCamera();

    //RNG state
    state = uint4(id.x, id.y, id.z, uint(custom.NoiseAnimation)*time.frame);

    for(var i: i32 = 0; i < int(custom.Samples*MaxSamples + 1.0); i++)
    {
        var bud = buddhabrot(128);
    }
}

fn Sample(pos: int2) -> float3
{
    let screen_size = int2(textureDimensions(screen));
    let idx = pos.x + screen_size.x * pos.y;

    var color: float3;

    let x = float(atomicLoad(&atomic_storage[idx*4+0]));
    let y = float(atomicLoad(&atomic_storage[idx*4+1]));
    let z = float(atomicLoad(&atomic_storage[idx*4+2]));
    
    color = pow(float3(x,y,z)/ATOMIC_SCALE, vec3(1.15));

    return abs(color);
}

@compute @workgroup_size(16, 16)
fn FinalPass(@builtin(global_invocation_id) id: uint3) 
{
    let screen_size = uint2(textureDimensions(screen));

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = float2(float(id.x) + .5, float(id.y) + .5);


    let oldColor = textureLoad(pass_in, int2(id.xy), 0, 0);

    var color = float4(Sample(int2(id.xy)), 1.0);



    color += oldColor * custom.Accumulation;
    

    let exposed = 1.0 - exp(-0.05*custom.Exposure*color.xyz/color.w);
    
    // Output to buffer
    textureStore(pass_out, int2(id.xy), 0, color);


    textureStore(screen, int2(id.xy), float4(exposed, 1.));
}
