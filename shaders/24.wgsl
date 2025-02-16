#storage atomic_storage array<atomic<i32>>

const MaxSamples = 32.0;
const FOV = 0.8;
const PI = 3.14159265;
const TWO_PI = 6.28318530718;
const STEP = 0.01;
const LARGENUM = 1e10;


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

fn SetCamera()
{
    let screen_size = int2(textureDimensions(screen));
    let screen_size_f = float2(screen_size);
    let ang = float2(mouse.pos.xy)*float2(-TWO_PI, PI)/screen_size_f + float2(0.4, 0.4);

    camera.fov = FOV;
    camera.cam = GetCameraMatrix(ang); 
    camera.pos = - (camera.cam*float3(5.0*custom.Radius+0.5,0.0,0.0));
    camera.size = screen_size_f;
}

//project to clip space
fn Project(cam: Camera, p: float3) -> float3
{
    let td = distance(cam.pos, p);
    let dir = (p - cam.pos)/td;
    let screen = dir*cam.cam;
    return float3(screen.yz*cam.size.y/(cam.fov*screen.x) + 0.5*cam.size,screen.x*td);
}

const max_iterations = 256;
const color_thresholds = float4(255.0, 130.0, 80.0, 255.0);

fn buddhabrot(iters: i32) -> vec4<f32> {
    var z: vec3<f32> = vec3<f32>(0.01, 0.01, 0.01);
    var c: vec3<f32> = (rand4().xyz - 0.5) * 1.5;

    var i: i32 = 0;
    loop {
        if (i >= iters) { break; }

        let r: f32 = length(z);
        let b: f32 = 2.0 * acos(z.y / r);
        let a: f32 = 2.0 * atan2(z.x, z.z);
        z = c + pow(r, 2.0) * vec3<f32>(sin(b) * sin(a), cos(b), sin(b) * cos(a));

        if (length(z) > 4.0) {
            break;
        }
        i = i + 1;
    }

    if (i >= iters) {
        return vec4<f32>(1e5, 1e5, 1e5, 1e5);
    }

    let maxj: i32 = i32(rand4().x * 1000.0) % i;

    z = vec3<f32>(0.01, 0.01, 0.01);

    for (var j: i32 = 0; j <= maxj; j = j + 1) {
        let r: f32 = length(z);
        let b: f32 = 3.0 * acos(z.y / r);
        let a: f32 = 3.0 * atan2(z.x, z.z);
        z = c + pow(r, 3.0) * vec3<f32>(sin(b) * sin(a), cos(b), sin(b) * cos(a));
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

fn AdditiveBlend(color: float3, depth: float, index: int)
{
    let scaledColor = int3(256.0 * color/(depth*depth + 0.2));

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


    let idx = screenCoord.x + screen_size.x * screenCoord.y;
    AdditiveBlend(color, projectedPos.z, idx);
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

@compute @workgroup_size(16, 16)
fn Rasterize(@builtin(global_invocation_id) id: uint3) 
{
    SetCamera();

    //RNG state
    state = uint4(id.x, id.y, id.z, uint(custom.NoiseAnimation)*time.frame);

    for(var i: i32 = 0; i < int(custom.Samples*MaxSamples + 1.0); i++)
    {
        var bud = buddhabrot(128);
        if(bud.w < 16.0) {continue;}
        bud.w = 300.0 + 3.0*bud.w;
        RasterizePoint(bud.xyz, spectral_zucconi(bud.w)/(custom.Samples*MaxSamples + 1.0));
    }
}

fn Sample(pos: int2) -> float3
{
    let screen_size = int2(textureDimensions(screen));
    let idx = pos.x + screen_size.x * pos.y;

    var color: float3;

    let x = float(atomicLoad(&atomic_storage[idx*4+0]))/(256.0);
    let y = float(atomicLoad(&atomic_storage[idx*4+1]))/(256.0);
    let z = float(atomicLoad(&atomic_storage[idx*4+2]))/(256.0);
    
    color = tanh(32.0*float3(x,y,z));

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


    if(mouse.click != 1)
    {
       color += oldColor * custom.Accumulation;
    }
    
    // Output to buffer
    textureStore(pass_out, int2(id.xy), 0, color);


    textureStore(screen, int2(id.xy), float4(color.xyz/color.w, 1.));
}
