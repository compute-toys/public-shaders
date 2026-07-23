//Port of "Branchless Voxel Raycasting" with some modifications 
//All credit for the original code goes to fb39ca4's shader on
//Shadertoy: https://www.shadertoy.com/view/4dX3zl
const Size = 1.; //Adjust this for denser voxels
const MaxIterations = 128u * u32(ceil(Size));

struct DDAResult{
    Normal : vec3<f32>,
    Distance : f32,
    HitVoxel : bool
};

struct Ray {
    Pos: vec3<f32>,
    Dir: vec3<f32>
};

// https://compute.toys/view/15
fn Hash33(p: float3) -> float3 {
    var p3 = fract(p * float3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yxz+33.33);
    return fract((p3.xxy + p3.yxx)*p3.zyx);
}

// https://compute.toys/view/16
/* discontinuous pseudorandom uniformly distributed in [-0.5, +0.5]^3 */
fn random3(c: float3) -> float3
{
    var j = 4096.0*sin(dot(c,vec3(17.0, 59.4, 15.0)));
    var r = float3(0.);
    r.z = fract(512.0*j);
    j *= .125;
    r.x = fract(512.0*j);
    j *= .125;
    r.y = fract(512.0*j);
    return r - 0.5;
}

/* skew constants for 3d simplex functions */
const F3 = 0.3333333;
const G3 = 0.1666667;

/* 3d simplex noise */
fn simplex3d(p: float3) -> float
{
    /* 1. find current tetrahedron T and it's four vertices */
    /* s, s+i1, s+i2, s+1.0 - absolute skewed (integer) coordinates of T vertices */
    /* x, x1, x2, x3 - unskewed coordinates of p relative to each of T vertices*/

    /* calculate s and x */
    let s = floor(p + dot(p, vec3(F3)));
    let x = p - s + dot(s, vec3(G3));

    /* calculate i1 and i2 */
    let e = step(vec3(0.0), x - x.yzx);
    let i1 = e*(1.0 - e.zxy);
    let i2 = 1.0 - e.zxy*(1.0 - e);

    /* x1, x2, x3 */
    let x1 = x - i1 + G3;
    let x2 = x - i2 + 2.0*G3;
    let x3 = x - 1.0 + 3.0*G3;

    /* 2. find four surflets and store them in d */
    var w = float4(0.);
    var d = float4(0.);

    /* calculate surflet weights */
    w.x = dot(x, x);
    w.y = dot(x1, x1);
    w.z = dot(x2, x2);
    w.w = dot(x3, x3);

    /* w fades from 0.6 at the center of the surflet to 0.0 at the margin */
    w = max(0.6 - w, float4(0.0));

    /* calculate surflet components */
    d.x = dot(random3(s), x);
    d.y = dot(random3(s + i1), x1);
    d.z = dot(random3(s + i2), x2);
    d.w = dot(random3(s + 1.0), x3);

    /* multiply d by w^4 */
    w *= w;
    w *= w;
    d *= w;

    /* 3. return the sum of the four surflets */
    return dot(d, vec4(52.0));
}

fn SDSphere(p : vec3<f32>, d : f32) -> f32{
    return length(p) - d;
}

fn SDBox(p : vec3<f32>, b : vec3<f32>) -> f32{
    let d = abs(p) - b;
    return min(max(d.x, max(d.y, d.z)), 0.) + length(max(d, vec3<f32>(0.)));
}

fn GetVoxel(c : vec3<f32>) -> bool{
    // let p : vec3<f32> = c / Size + vec3<f32>(.5);
    // var d : f32 = max(-SDSphere(p, 12.), SDBox(p, vec3<f32>(10.)));
    // d = min(d, p.y + 8.);
    // d = min(d, SDSphere(p - float3(0., -2., 0.), 5.));
    // d = min(d, SDSphere(p - float3(20., 8., 0.), 6.));
    // return d < 0.;
    if(length(((c.xz + 200.) % 50.) - float2(25.)) < 3.5) {
        return true;
    }
    if((c.y == 25. || abs(c.z) == 30.) && c.x % 200. > 100.) {
        return true;
    }
    let p = (c + 0.5) / 20.;
    return simplex3d(p) < -p.y * 2. - 1.;
    //return p.y < -0.2;
}

fn Rotate2D(v : vec2<f32>, a : f32) -> vec2<f32>{
    let SinA : f32 = sin(a);
    let CosA : f32 = cos(a);
    return vec2<f32>(v.x * CosA - v.y * SinA, v.y * CosA + v.x * SinA);
}

fn DDA(RayPosition : vec3<f32>, RayDirection : vec3<f32>) -> DDAResult{
    let DeltaDistance = abs(vec3(length(RayDirection)) / RayDirection);
    let RayStep = sign(RayDirection);

    var MapPosition = floor(RayPosition);
    var SideDistance = (sign(RayDirection) * (MapPosition - RayPosition) + (sign(RayDirection) * .5) + .5) * DeltaDistance;
    var Normal = vec3<f32>(0.);

    for(var i : u32 = 0u; i < MaxIterations; i++){
        if(GetVoxel(MapPosition)){
            return DDAResult(Normal, length(Normal * (SideDistance - DeltaDistance)) / length(RayDirection), true);
        }
        Normal = step(SideDistance, min(SideDistance.yxy, SideDistance.zzx));
        SideDistance = fma(Normal, DeltaDistance, SideDistance);
        MapPosition = fma(Normal, RayStep, MapPosition);
    }
    return DDAResult(vec3<f32>(0.), 0., false);
}

fn GetCameraPos() -> vec3<f32> {
    var RayPosition = vec3<f32>(0., 0., -10.);
    //var RayPosition = float3(0., 10., -16. - 10. * cos(time.elapsed / 6.));
    //RayPosition.y = 0.;
    return RayPosition;
}

fn GetPrimaryRay(FragCoord: vec2<f32>) -> Ray {
    let Resolution = float2(textureDimensions(screen).xy);
    let FragOff = Hash33(float3(FragCoord.x + 100., FragCoord.y - 100., time.elapsed + 20.)).xy - 0.5;
    let UV = 2. * (FragCoord + FragOff * 0.5) / Resolution - 1.;

    let CameraDirection = vec3<f32>(0., 0., .8);
    let CameraPlaneU = vec3<f32>(1., 0., 0.);
    let AspectRatio = f32(Resolution.y) / f32(Resolution.x);
    let CameraPlaneV = vec3<f32>(0., AspectRatio, 0.);
    
    var RayDirection = CameraDirection + UV.x * CameraPlaneU + UV.y * CameraPlaneV;
    var RayPosition = GetCameraPos();


    let DirectionRotation = Rotate2D(RayDirection.xz, time.elapsed / 4.);
    let PositionRotation = Rotate2D(RayPosition.xz, time.elapsed / 4.);

    RayDirection = vec3<f32>(DirectionRotation.x, RayDirection.y, DirectionRotation.y);
    RayPosition = vec3<f32>(PositionRotation.x, RayPosition.y, PositionRotation.y) * Size;
    return Ray(RayPosition + float3(time.elapsed * 10., 0., 0.), RayDirection);
}

fn GetPixel(HitPosition: vec3<f32>) -> vec2<f32> {
    let Resolution = vec2<f32>(textureDimensions(screen).xy);
    let AspectRatio = Resolution.y / Resolution.x;

    // 1. Calculate the ray direction from the camera to the hit point
    let RayDirection = HitPosition - GetCameraPos();

    // 2. Project the direction onto the camera plane at depth Z = 0.8
    // (This cancels out any distance/length scaling in the ray)
    let Scale = 0.8 / RayDirection.z;
    let UV = vec2<f32>(
        RayDirection.x * Scale,
        (-RayDirection.y * Scale) / AspectRatio
    );

    // 3. Map UV [-1, 1] back to FragCoord [0, Resolution]
    return (UV + 1.0) * 0.5 * Resolution;
}

fn PointIsGlowing(c: float3) -> bool {
    let p : vec3<f32> = c - 0.5;
    return length(((p.xz + 200.) % 50.) - float2(25.)) < 4.5;
    //return false;
}

fn GetPointFluence(p: float3) -> float3 {
    if(PointIsGlowing(p)) {
        return float3(1.);
    } else {
        if(simplex3d(round(p) / 40.) < 0.) {
            return float3(0.2, 0.9, 0.2);
        } else {
            return float3(0.5, 0.3, 0.1);
        }
        
    }
}

fn GetPointLight(p: float3, norm: float3, Rng: float3) -> float3 {
    var Light = float3(0.);
    if(PointIsGlowing(p)) {
        Light += float3(sin(p.x / 25.) * 0.5 + 0.5, cos(p.x / 25.) * 0.5 + 0.5, sin(p.z / 15.));
    }
    let SunDir = normalize(float3(1., 2., 0.5) + (Rng - 0.5) * 0.1);
    var SunResult = DDA(p, SunDir);
    if(!SunResult.HitVoxel) {
        Light += dot(-norm, SunDir) * float3(1., 0.5, 0.25);
    }
    return Light;
}

fn CollectLight(Point: float3, RawHash: float3, Normal: float3) -> float3 {
    var Rng = RawHash;
    var Fluence = GetPointFluence(Point);
    var Light = GetPointLight(Point, Normal, Rng) * Fluence;
    Rng = Hash33(Rng);
    var RayPos = Point;
    var RayDir = normalize((Rng - 0.5) - Normal * 0.5);
    for(var i = 0; i < 3; i++) {
        var Result = DDA(RayPos, RayDir);
        if(Result.HitVoxel) {
            RayPos = RayPos + RayDir * Result.Distance * 0.9999;
            Rng = Hash33(Rng);
            let BounceNormal = Result.Normal * sign(RayDir);
            RayDir = normalize((Rng - 0.5) - BounceNormal * 0.5);
            Light += Fluence * GetPointLight(RayPos, BounceNormal, Rng);
            Rng = Hash33(Rng);
            Fluence *= GetPointFluence(RayPos);
        } else {
            Light += max(0., RayDir.y) * float3(0., 0.5, 0.75) * Fluence;
            break;
        }
    }
    //return floor(Light + Rng);
    return Light;
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) Pixel: uint3) {
    let Resolution = textureDimensions(screen).xy;
    if (Pixel.x >= Resolution.x || Pixel.y >= Resolution.y){
        return;
    }
    let FragCoord = float2(float(Pixel.x) + .5, float(Resolution.y - Pixel.y) - .5);
    let Ray = GetPrimaryRay(FragCoord);

    let Primary = DDA(Ray.Pos, Ray.Dir);
    let HitPos = Ray.Pos + Ray.Dir * Primary.Distance * .9999;
    let Normal = Primary.Normal * sign(Ray.Dir);
    var Alignment = sqrt(dot(Ray.Dir, Normal));
    let Bias = (pow(textureLoad(channel0, Pixel.xy % 1024, 0).r, 1/2.2) - 0.5) * 1.;
    let Quant = pow(0.5, ceil(1. * log2(Primary.Distance / Alignment) - Bias * 1.)) * float(Resolution.x) / custom.QuantizationScaling * 1.;
    let QuantPos = floor(Quant * HitPos) / Quant;
    let RawHash = Hash33(QuantPos * 100.);
    var Color = float3(0.);
    if(Primary.HitVoxel){
        var Light = CollectLight(HitPos, RawHash, Normal);
        var TaxiDist = (abs(Ray.Dir.x) + abs(Ray.Dir.y) + abs(Ray.Dir.z)) * Primary.Distance;
        var Fog = pow(min(1., TaxiDist / 126.), 4);
        Color = Light * (1. - Fog) + Fog * float3(0., 0.5, 0.75);
    } else {
        Color = float3(0., 0.5, 0.75);
    }
    //Color = vec3(HitPos);
    //var diff = Primary.Distance * length(Ray.Dir) - distance(Ray.Pos, Prev.xyz);
    //Color = float3(sin(diff * 10));
    textureStore(screen, Pixel.xy, vec4<f32>(Color, 1.));
    //var lum = max(Color.r, max(Color.g, Color.b));
    //textureStore(pass_out, Pixel.xy, 0, float4(Prev.x, Prev.y, Prev.z, lum));
}
