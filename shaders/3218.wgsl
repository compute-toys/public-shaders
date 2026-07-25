//Port of "Branchless Voxel Raycasting" with some modifications 
//All credit for the original code goes to fb39ca4's shader on
//Shadertoy: https://www.shadertoy.com/view/4dX3zl
const Size = 1.; //Adjust this for denser voxels
const MaxIterations = 128u * u32(ceil(Size));
const SunDir = normalize(float3(1., 1., 0.5));

struct DDAResult{
    Normal : vec3<f32>,
    Distance : f32,
    HitVoxel : bool
};

struct Ray {
    Pos: vec3<f32>,
    Dir: vec3<f32>
};


var<private> CamPos = float3(0., 0., 0.);
var<private> CamDir = 0.;


fn IGN(p: float3) -> float {
    var o = p.xy + p.z * 5.588238;
    return (52.9829189*(0.06711056*o.x+0.00583715*o.y)%1)%1;
}

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
    // if((c.y == 10. || abs(c.z) == 15.) && c.x % 400. > 200.) {
    //     return true;
    // }
    let p = (c + 0.5) / 20.;
    let noise = simplex3d(p);
    return noise < -p.y * 2. - 1.1 && noise < 0.;
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

// fn GetCameraPos() -> vec3<f32> {
//     var RayPosition = vec3<f32>(0., 0., -10.);
//     //var RayPosition = float3(0., 10., -16. - 10. * cos(time.elapsed / 6.));
//     //RayPosition.y = 0.;
//     return RayPosition;
// }

fn GetPrimaryRay(FragCoord: vec2<f32>) -> Ray {
    let Resolution = float2(textureDimensions(screen).xy);
    //let FragOff = Hash33(float3(FragCoord.x + 100., FragCoord.y - 100., time.elapsed + 20.)).xy - 0.5;
    let FragOff = float2(0.);
    let UV = 2. * (FragCoord + FragOff * 0.5) / Resolution - 1.;

    let CameraDirection = vec3<f32>(0., 0., .8);
    let CameraPlaneU = vec3<f32>(1., 0., 0.);
    let AspectRatio = f32(Resolution.y) / f32(Resolution.x);
    let CameraPlaneV = vec3<f32>(0., AspectRatio, 0.);
    
    var RayDirection = CameraDirection + UV.x * CameraPlaneU + UV.y * CameraPlaneV;
    var RayPosition = CamPos;


    let DirectionRotation = Rotate2D(RayDirection.xz, CamDir);
    //let PositionRotation = Rotate2D(RayPosition.xz, CamDir);

    RayDirection = vec3<f32>(DirectionRotation.x, RayDirection.y, DirectionRotation.y);
    //RayPosition = vec3<f32>(PositionRotation.x, RayPosition.y, PositionRotation.y) * Size;
    return Ray(RayPosition, RayDirection);
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

fn GetPointLight(p: float3, norm: float3) -> float3 {
    var Light = float3(0.);
    if(PointIsGlowing(p)) {
        Light += float3(sin(p.x / 25.) * 0.5 + 0.5, cos(p.x / 25.) * 0.5 + 0.5, sin(p.z / 15.));
    }
    
    if(PointInSun(p)) {
        Light += dot(-norm, SunDir) * float3(1., 0.75, 0.5);
    }
    return Light;
}

fn PointInSun(p: float3) -> bool {
    var SunResult = DDA(p, SunDir);
    return !SunResult.HitVoxel;
}

fn CollectLight(Point: float3, RawHash: float3, Normal: float3) -> float3 {
    var Rng = RawHash;
    var Fluence = GetPointFluence(Point);
    var RayDir = normalize((Rng - 0.5) - Normal * 0.5);
    var Light = GetPointLight(Point, Normal) * Fluence;
    var RayPos = Point;
    for(var i = 0; i < 3; i++) {
        var Result = DDA(RayPos, RayDir);
        if(Result.HitVoxel) {
            RayPos = RayPos + RayDir * Result.Distance * 0.9999;
            Rng = Hash33(Rng);
            let BounceNormal = Result.Normal * sign(RayDir);
            RayDir = normalize((Rng - 0.5) - BounceNormal * 0.5);
            Light += Fluence * GetPointLight(RayPos, BounceNormal);
            Fluence *= GetPointFluence(RayPos);
        } else {
            Light += max(0., RayDir.y) * float3(0., 0.25, 0.5) * Fluence;
            break;
        }
    }
    return Light;
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) Pixel: uint3) {
    let Resolution = textureDimensions(screen).xy;
    if (Pixel.x >= Resolution.x || Pixel.y >= Resolution.y){
        return;
    }
    var CamData = textureLoad(pass_in, Pixel.xy, 0, 0);
    let rot_speed = 3. * time.delta;
    if(keyDown(37)) {
        CamData.w += rot_speed;
    }
    if(keyDown(39)) {
        CamData.w -= rot_speed;
    }
    let speed = 10. * time.delta;
    let fwd = Rotate2D(float2(0., 1.), CamData.w) * speed;
    let rig = Rotate2D(float2(1., 0.), CamData.w) * speed;
    var hp = CamData.xz;
    if(keyDown(65)) {
        hp -= rig;
    }
    if(keyDown(68)) {
        hp += rig;
    }
    if(keyDown(87)) {
        hp += fwd;
    }
    if(keyDown(83)) {
        hp -= fwd;
    }
    CamData.x = hp.x;
    CamData.z = hp.y;
    if(keyDown(69)) {
        CamData.y += speed;
    }
    if(keyDown(81)) {
        CamData.y -= speed;
    }
    CamPos = CamData.xyz;
    CamDir = CamData.w;
    textureStore(pass_out, Pixel.xy, 0, CamData);
    let FragCoord = float2(float(Pixel.x) + .5, float(Resolution.y - Pixel.y) - .5);
    let Ray = GetPrimaryRay(FragCoord);

    let Primary = DDA(Ray.Pos, Ray.Dir);
    let HitPos = Ray.Pos + Ray.Dir * Primary.Distance * .9999;
    let Normal = Primary.Normal * sign(Ray.Dir);
    var Alignment = sqrt(dot(Ray.Dir, Normal));
    let Bias = (pow(textureLoad(channel0, Pixel.xy % 1024, 0).r, 1/2.2) - 0.5) * 1.;
    var Quant = pow(0.5, ceil(1. * log2(Primary.Distance / Alignment) - Bias * 1.)) * float(Resolution.x) / custom.QuantizationScaling;
    Quant = clamp(Quant, 0., 256.);
    let QuantPos = abs(floor(Quant * HitPos)) / Quant * 2.;
    // let RawHash = Hash33(QuantPos * 100.);
    let RawHash = float3(
        simplex3d(QuantPos),
        simplex3d(QuantPos + float3(0., 10., 0.)),
        simplex3d(QuantPos + float3(0., 20., 0.)),
    ) * 0.5 + 0.5;
    var Color = float3(0.);
    if(Primary.HitVoxel){
        var Light = CollectLight(HitPos, RawHash, Normal);
        Color = Light;

        var absorb = pow(0.995, Primary.Distance);
        var add = 0.001 * Primary.Distance;
        for(var j = 4; j >= 0; j--) {
            var off = (float(j) + Bias + 0.5) / 5.;
            var SamplePos = Ray.Pos + Ray.Dir * (Primary.Distance * off);
            Color *= absorb;
            if(PointInSun(SamplePos)) {
                Color += add * float3(1., 0.75, 0.5);
            }
        }

        var TaxiDist = (abs(Ray.Dir.x) + abs(Ray.Dir.y) + abs(Ray.Dir.z)) * Primary.Distance / 126.;
        var Fog = min(1., max(0., TaxiDist * 10. - 9.));
        Color = Color * (1. - Fog) + Fog * float3(0.0, 0.5, 0.75);
    } else {
        Color = float3(0., 0.5, 0.75);
    }
    //Color = vec3(HitPos);
    //var diff = Primary.Distance * length(Ray.Dir) - distance(Ray.Pos, Prev.xyz);
    //Color = float3(sin(diff * 10));
    var fx = float(Pixel.x);
    var fy = float(Pixel.y);
    var v = IGN(float3(fx, fy, 0.));
    //Color = pow(mortonCurve3D(fx / 928.), float3(2.));
    textureStore(screen, Pixel.xy, vec4<f32>(Color, 1.));
    //var lum = max(Color.r, max(Color.g, Color.b));
    //textureStore(pass_out, Pixel.xy, 0, float4(Prev.x, Prev.y, Prev.z, lum));
}
