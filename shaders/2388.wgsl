#storage atomic_storage array<atomic<i32>>

//Check Uniforms
//Mode 0 - additive blending (atomicAdd)
//Mode 1 - closest sample (atomicMax)

const MaxSamples = 64.0;
const FOV = 0.8;
const PI = 3.14159265;
const INVPI = 0.31830988;
const TWO_PI = 6.28318530718;

const DEPTH_MIN = 0.2;
const DEPTH_MAX = 100.0;
const DEPTH_BITS = 16u;

const remap_distance = 0.5;

fn GELU(x: f32) -> f32 {
    return x * 0.5 * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
}

fn ELU(x: f32) -> f32 {
    return select(x, exp(x) - 1.0, x < 0.0);
}

fn viewDirToCircle(d: vec2<f32>, X: vec2<f32>) -> vec2<f32> {
    let dn = normalize(d);                          
    let s = dot(X, dn);
    let t = -s + sqrt(s * s + 1.0 - dot(X, X));     
    return X + t * dn;                               
}

fn transformRay(rd: vec2<f32>, rad0: f32) -> vec2<f32> {
    let s = clamp(1.0 - remap_distance / rad0, 0.0, 1.0);
    let X = vec2<f32>(0.0, s);                 
    return viewDirToCircle(rd, -X); 
}

fn transformRayInv(rd: vec2<f32>, rad0: f32) -> vec2<f32> {        
    let s = clamp(1.0 - remap_distance / rad0, 0.0, 1.0);
    return normalize(rd + vec2(0.0, s));
}


const EPS: f32 = 1e-08;

const H: u32 = 6u;
const IN: u32 = 8u;

const W1: array<f32, 48u> = array<f32, 48u>(
        -0.0295961108, -0.271011263, 0.925183773, 0.357251674, -0.59920156, 0.4135966, -0.000869847834, -0.0300796963,
        -0.403331101, 0.00253559323, -0.760167539, -0.751619637, 0.418163449, 0.668142736, -0.00143730547, -1.86708772,
        -0.178387448, -0.363470256, 1.77829075, -0.0530340336, 0.505375385, -1.18786359, -0.00529918261, 0.205751494,
        0.23840034, -0.421429962, 0.844540179, 0.873498917, -2.2546804, 0.740431786, -0.00495470781, -0.109265603,
        -0.0849070773, 0.171677575, 0.35548082, -0.654942095, -1.14423823, 0.115525961, -0.0121639883, 0.695323765,
        0.429000229, -0.440778226, -0.0709175244, 1.19453621, 0.30376035, -0.862229764, 0.00147002423, 1.09598851
    );
const B1: array<f32, 6u> = array<f32, 6u>(
        -0.902096391, 0.627739787, -1.76775062, -0.775565803, 0.0883439407, 0.604211688
    );
const WH: array<f32, 36u> = array<f32, 36u>(
        0.435583889, 0.146988556, -0.942280352, 0.0990441218, -0.207154393, 0.833539963, 0.141709343, 0.183765933,
        0.313979238, 0.23510699, 0.401597559, 0.478932291, 0.0538431332, -0.70171541, 0.0494740792, 0.0410975516,
        -0.433611065, -0.242934465, 0.511239469, 0.0724557787, 0.695890129, -1.07301652, -0.427882522, -0.206279144,
        0.851955771, 0.204983696, -0.160287008, -1.04253268, 0.0465813503, -0.111925952, 0.752238691, -0.361777425,
        -1.37334847, 0.087751314, 0.612253666, -0.239087746
    );
const BH: array<f32, 6u> = array<f32, 6u>(
        0.0892086476, 0.210533544, 1.3448205, -0.226439908, 0.84676522, -0.0111006377
    );
const W2: array<f32, 12u> = array<f32, 12u>(
        -0.00844737329, 0.00540382974, 0.00213702396, -0.00399108417, -0.00703751296, 0.00653843954, -1.30415678, 0.632811308,
        0.292917639, -0.435869128, -1.15112591, 0.855104864
    );
const B2: array<f32, 2u> = array<f32, 2u>(
        -0.000851286575, 0.617058814
    );
const K: array<f32, 15u> = array<f32, 15u>(
        -1.04302323, -0.750994861, 1.10156524, 0.938264966, 0.595364869, -0.276644737, -1.00197685, 0.5183025,
        -0.532480419, 1.03988421, 0.355645597, 1.11154306, 0.80976516, 0.0688146502, 0.222828895
    );


fn remap_polar_nn(angle: f32, rad: f32, rad0: f32) -> vec2<f32> {
    let phi = angle / PI;
    let u  = 1.0 / max(rad,  EPS);
    let u0 = 1.0 / max(rad0, EPS);

    if(rad < 0.5) { return vec2<f32>(1e10); }

    let s2 = clamp(u0*(K[11] + u0*(K[0] + K[1]*u0 + K[2]*u0*u0)), 0.0, 1.0);
    let shadow_angle = asin(sqrt(s2));
    let u1 = u0 * (K[3] + K[4]*u + K[5]*u*u) * (1.0 + K[6]*u0 + K[7]*u0*u0) * (1.0 + K[8]*abs(phi) + K[9]*phi*phi);
    let w1 = 1.0 + pow(abs(phi), 1.0 / (K[10]*u1));
    let ang_step = (PI - shadow_angle) * phi * pow(w1, -K[10]*u1);
    var X = vec2<f32>(sin(ang_step), cos(ang_step) - rad0 / rad);
    X = transformRayInv(normalize(X), rad0);

    var F: array<f32, IN>;
    F[0] = u; F[1] = tanh(K[12]*rad); F[2] = u0; F[3] = tanh(K[13]*rad0);
    F[4] = abs(phi); F[5] = tanh(K[14]*phi*phi);
    F[6] = X.x; F[7] = X.y;

    var h: array<f32, H>;
    for (var i: u32 = 0u; i < H; i++) {
        var s: f32 = B1[i];
        for (var j: u32 = 0u; j < IN; j++) {
            s = s + W1[i*IN + j] * F[j];
        }
        h[i] = s;
    }

    for (var i: u32 = 0u; i < H; i++) {
        h[i] += GELU(h[i]);
    }

    var t: array<f32, H>;
    for (var i: u32 = 0u; i < H; i++) {
        var s: f32 = BH[i];
        for (var j: u32 = 0u; j < H; j++) {
            s = s + WH[i*H + j] * h[j];
        }
        t[i] = s;
    }
    for (var i: u32 = 0u; i < H; i++) {
        h[i] += GELU(t[i]);
    }

    // X = normalize( X + fc2(h) )
    var o0: f32 = B2[0];
    var o1: f32 = B2[1];
    for (var j: u32 = 0u; j < H; j++) {
        o0 = o0 + W2[0u*H + j] * h[j];
        o1 = o1 + W2[1u*H + j] * h[j];
    }
    return normalize(vec2<f32>(X.x + o0, X.y + o1));
}

fn remap_2d(xo: float3, e1: float3, e0: float3, rad0: float, wind: float) -> float2
{
    let angle = atan2(dot(xo, e1), dot(xo, e0));
    let rad = length(xo);
    return transformRay(remap_polar_nn(angle + sign(angle) * wind * PI, rad, rad0), rad0);         
}

fn remap_in_OCX_plane(C: vec3<f32>, X: vec3<f32>, wind: f32) -> vec3<f32> {
    let O = vec3<f32>(0.0);
    let rad0 = length(C - O);

    let e0 = (C - O) / rad0;
    var n = normalize(cross(e0, X - C));             
    let e1 = normalize(cross(n, e0)); 

    let eps = 0.05;
    let xo = X - O;
    let s = remap_2d(xo, e1, e0, rad0, wind);
    let s1 = remap_2d(xo + e1*eps, e1, e0, rad0, wind);
    let effective_dist = eps / distance(s, s1); 
    return C + effective_dist * (e0 * s.y + e1 * s.x); 
}

struct Camera 
{
  pos: float3,
  cam: float3x3,
  fov: float,
  size: float2
}

var<private> camera : Camera;
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

fn GetCameraMatrix(ang: float2) -> float3x3
{
    let x_dir = float3(cos(ang.x)*sin(ang.y), cos(ang.y), sin(ang.x)*sin(ang.y));
    let y_dir = normalize(cross(x_dir, float3(0.0,1.0,0.0)));
    let z_dir = normalize(cross(y_dir, x_dir));
    return float3x3(-x_dir, y_dir, z_dir);
}

fn SetCamera(ang: float2, fov: float)
{
    camera.fov = fov;
    camera.cam = GetCameraMatrix(ang); 
    camera.pos = - (camera.cam*float3(3.0*custom.Radius+0.1,0.0,0.0));
    camera.size = float2(textureDimensions(screen));
}

//project to clip space
fn Project(cam: Camera, p: float3) -> float3
{
    let td = distance(cam.pos, p);
    let dir = (p - cam.pos)/td;
    let screen = dir*cam.cam;
    return float3(screen.yz*cam.size.y/(cam.fov*screen.x) + 0.5*cam.size,screen.x*td);
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
    let scaledColor = 256.0 * color / depth;

    atomicAdd(&atomic_storage[index*4+0], int(scaledColor.x));
    atomicAdd(&atomic_storage[index*4+1], int(scaledColor.y));
    atomicAdd(&atomic_storage[index*4+2], int(scaledColor.z));
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

    let idx = screenCoord.x + screen_size.x * screenCoord.y;
    
    if(custom.Mode < 0.5)
    {
        AdditiveBlend(color, projectedPos.z, idx);
    }
    else
    {
        ClosestPoint(color, projectedPos.z, idx);
    }
}

fn RasterizePointBHwind(pos: float3, color: float3, wind: float)
{
    let scale = 13.0;
    //if(scale*length(pos) < 2.0) {return;}
    let transformed_pos = remap_in_OCX_plane(scale*camera.pos, scale*pos, wind)/scale;

    RasterizePoint(transformed_pos, color);
}

fn RasterizePointBH(pos: float3, color: float3)
{
    RasterizePointBHwind(pos, color, 0.0);
    //TODO figure out brightness and validity
    RasterizePointBHwind(pos, color, 1.0);
    //RasterizePointBHwind(pos, color, 2.0);
}

fn skyTexture(rd: vec3f) -> vec3f
{
    return textureSampleLevel(channel0, bilinear, INVPI * vec2f(.5 * atan2(rd.z, rd.x), asin(rd.y)) + .5, 0.).rgb;
}

@compute @workgroup_size(16, 16)
fn Rasterize(@builtin(global_invocation_id) id: uint3) {
    // Viewport resolution (in pixels)
    let screen_size = int2(textureDimensions(screen));
    let screen_size_f = float2(screen_size);
    
    let ang = float2(mouse.pos.xy)*float2(-TWO_PI, PI)/screen_size_f + float2(0.4, 0.4);
    
    SetCamera(ang, FOV);

    //RNG state
    state = uint4(id.x, id.y, id.z, 0u*time.frame);

    for(var i: i32 = 0; i < int(custom.Samples*MaxSamples + 1.0); i++)
    {
        let rand = rand4();
        var pos = float3(0.0);
        var col = pos;
        let nrand = nrand4(1.0, float4(0.0));
        if(rand.w > 0.5) {
            pos = 10.0*normalize(nrand.xyz);
            col = 1000.0 * skyTexture(normalize(pos));
        } else {
            pos = 8.0*custom.Rad*(rand.xyz-0.5)*float3(0.1,0.1,1.0);
            col = float3(0.5 + 0.5*sin(10.0*pos));

            let sec = 5.0+custom.Speed*time.elapsed;
            //move points along sines
        //  pos += sin(float3(2.0,1.0,1.5)*sec)*0.1*sin(30.0*custom.Sine1*pos);
        //  pos += sin(float3(2.0,1.0,1.5)*sec)*0.02*sin(30.0*custom.Sine2*pos.zxy);
        }


        RasterizePointBH(pos, col);
    }
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
        
        color = tanh(0.1*float3(x,y,z)/(custom.Samples*MaxSamples + 1.0));
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

//to remove canvas aliasing
fn SampleBlur(pos: int2) -> float3
{
    let avg = Sample(pos+int2(1,0))+Sample(pos+int2(-1,0))+
              Sample(pos+int2(0,1))+Sample(pos+int2(0,-1));
    return mix(Sample(pos), 0.25*avg, custom.Blur);
}

fn hsv2rgb(hsv: vec3<f32>) -> vec3<f32> {
    // h in [0,1), s,v in [0,1]
    let h = fract(hsv.x);
    let s = hsv.y;
    let v = hsv.z;
    let p = vec3<f32>(h, h + 2.0 / 3.0, h + 1.0 / 3.0);
    let k = clamp(abs(fract(p) * 6.0 - 3.0) - 1.0, vec3<f32>(0.0), vec3<f32>(1.0));
    return v * ((1.0 - s) + s * k); // v * mix(1, k, s)
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) 
{
    let screen_size = uint2(textureDimensions(screen));

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = float2(float(id.x) + .5, float(id.y) + .5);

    let color = SampleBlur(int2(id.xy));

    let phi = 2.0*PI*(float(id.x) / float(screen_size.x) - 0.5);
    let u = 2.0 * (1.0 - float(id.y) / float(screen_size.y));

    let u1 = 2.0 * (1.0 - float(mouse.pos.y) / float(screen_size.y)); 

    let dir = remap_polar_nn(phi, 1.0 / u, 1.0 / u1);
    let ang = atan2(dir.y, dir.x)/(2.0 * PI) + 0.5;

    let col = hsv2rgb(vec3(ang, 1.0, fract(64*ang)));

    // Output to screen (linear colour space)
    textureStore(screen, int2(id.xy), float4(color, 1.));
}
