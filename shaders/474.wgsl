//Port of "Branchless Voxel Raycasting" with some modifications 
//All credit for the original code goes to fb39ca4's shader on
//Shadertoy: https://www.shadertoy.com/view/4dX3zl
const Size = 1.; //Adjust this for denser voxels
const MaxIterations = 64u * u32(ceil(Size));

struct DDAResult{
    Normal : vec3<f32>,
    Distance : f32,
    HitVoxel : bool
};

fn SDSphere(p : vec3<f32>, d : f32) -> f32{
    return length(p) - d;
}

fn SDBox(p : vec3<f32>, b : vec3<f32>) -> f32{
    let d = abs(p) - b;
    return min(max(d.x, max(d.y, d.z)), 0.) + length(max(d, vec3<f32>(0.)));
}

fn GetVoxel(c : vec3<f32>) -> bool{
    let p : vec3<f32> = c / Size + vec3<f32>(.5);
    let d : f32 = max(-SDSphere(p, 7.5), SDBox(p, vec3<f32>(6.)));
    return d < 0.;
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

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) Pixel: uint3) {
    let Resolution = textureDimensions(screen).xy;
    let AspectRatio = f32(Resolution.y) / f32(Resolution.x);

    if (Pixel.x >= Resolution.x || Pixel.y >= Resolution.y){
        return;
    }
    
    let FragCoord = float2(float(Pixel.x) + .5, float(Resolution.y - Pixel.y) - .5);

    let UV = 2. * FragCoord / float2(Resolution) - 1.;

    let CameraDirection = vec3<f32>(0., 0., .8);
    let CameraPlaneU = vec3<f32>(1., 0., 0.);
    let CameraPlaneV = vec3<f32>(0., AspectRatio, 0.);
    
    var RayDirection = CameraDirection + UV.x * CameraPlaneU + UV.y * CameraPlaneV;
    var RayPosition = vec3<f32>(0., 2. * sin(time.elapsed * 2.7), -12.);

    let DirectionRotation = Rotate2D(RayDirection.xz, time.elapsed);
    let PositionRotation = Rotate2D(RayPosition.xz, time.elapsed);

    RayDirection = vec3<f32>(DirectionRotation.x, RayDirection.y, DirectionRotation.y);
    RayPosition = vec3<f32>(PositionRotation.x, RayPosition.y, PositionRotation.y) * Size;

    let Primary = DDA(RayPosition, RayDirection);
    let Shadow = DDA(RayPosition + RayDirection * Primary.Distance * .9999, normalize(vec3<f32>(cos(time.elapsed / 10.), 1., sin(time.elapsed / 10.))));

    var Colour = vec3<f32>(length(Primary.Normal * vec3<f32>(.75, 1., .5)));
    if(Shadow.HitVoxel){
        Colour *= .5;
    }
    if(!Primary.HitVoxel){
        Colour = vec3(.7, .9, .9);
    }
    textureStore(screen, Pixel.xy, vec4<f32>(Colour, 1.));
}
