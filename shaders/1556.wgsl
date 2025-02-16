#storage atomic_storage array<atomic<i32>>

const MaxSamples = 256.0;
const FOV = 0.6;
const PI = 3.14159265;
const TWO_PI = 6.28318530718;
const STEP = 0.01;
const LARGENUM = 1e10;
const ATOMIC_SCALE = 1024.0;

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
    camera.pos = - (camera.cam*float3(8.0*custom.Radius+0.5,0.0,0.0));
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


fn AdditiveBlend(color: float3, depth: float, index: int)
{
    let scaledColor = int3(floor(ATOMIC_SCALE*color/(depth*depth + 0.2) + rand4().xyz));

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
    let screenCoord = int2(projectedPos.xy+0.5*rand4().xy);

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

fn hue(v: float) -> float3 {
    return .6 + .6 * cos(6.3 * v + float3(0.,23.,21.));
}
 
fn rot(v: float) -> float2 {
    return float2(cos(v), sin(v));
}

fn force(v: float3, t: float) -> float3 {
    let sc = custom.P2*(rot(3.215*time.elapsed)+0.15*rot(15.515*time.elapsed).yx);
    let sc2 = 7.0*rot(2.0*time.elapsed) + 5.0*rot(0.25*3.14519*2.0*time.elapsed);
    let sc3 = 5.0*rot(1.3547*time.elapsed) + 5.0*rot(2.7547*time.elapsed);
    let sc4 = custom.P3*rot(0.4*time.elapsed + t);
    var f = sin(custom.P1*vec3(sc3.x + 1.0-t, sc2.x + sc3.y + 2.0+t, sc2.y + 3.0+2.0*t) * v);
    f += custom.P4*sin(vec3(3.0,sc4.x,sc4.y)*(v + vec3(-0.4, 0.7,0.1)+ 0.02*f));
    f = -sc.x*sin(8.0*vec3(1.5, 2.0, 1.6) * f.zxy) - 0.3*sin(3.0*vec3(1.5, 2.0, 1.6) * f.yzx);
    f += sc.y*sin(8.0*vec3(1.5, 2.0, 1.6) * f.yzx);
    f += sc.x*sin(8.0*vec3(1.5, 2.0, 1.6) * f.yzx);
    f -= sc.y*sin(8.0*vec3(0.5, 0.2, 2.6) * f.yzx);
   
    
    return f; 
}

fn point_gen() {
    var r4 = rand4();

    let p0 = vec3(-1.0, 0.0, 0.0);
    let p1 = vec3(1.0, 0.0, 0.0);

    var p = mix(p0, p1, r4.x);
    let center = r4.x*(1.0-r4.x);
    let sc = custom.P4*rot(0.5*time.elapsed);
    let delta = 0.25*vec3(0.1+sc.x,0.2,0.4+sc.y)*sin(50.0*vec3(1.1+sc.y,1.0,1.3)*r4.x) - 2.0*vec3(0.0,1.0,0.0);
    p += 2.0*center * delta;
    p += vec3(0.0,0.75,0.0);

    let t = 2.0*r4.y - 1.0;
    var color = mix(vec3(1.0,0.4,0.1), vec3(0.10, 0.40, 1.0), r4.y);
    //color = mix(color, vec3(0.0, 2.0, 0.05), 2.0*r4.y*(1.0 - r4.y));
    color *= mix(1.0 - t*t, t*t, custom.P5);
    let dt = custom.DT;
    for(var i = 0; i < 16; i++)
    {
        let time = float(i)*t*dt;
        p += dt*force(p, time) * t * center;
    }

    RasterizePoint(p.xyz, 32.0*color);
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

    let max_iters = int(custom.Samples);
    for(var i = 0; i < max_iters; i++)
    {
        point_gen();
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
    
    color = float3(x,y,z)/ATOMIC_SCALE;

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

    let exposed = 0.1*custom.Exposure*color.xyz/(color.w*custom.Samples);
    
    // Output to buffer
    textureStore(pass_out, int2(id.xy), 0, color);


    textureStore(screen, int2(id.xy), float4(exposed, 1.));
}
