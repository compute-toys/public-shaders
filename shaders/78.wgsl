//Port of "Branchless Voxel Raycasting" with some modifications 
//All credit for the original code goes to fb39ca4's shader on
//Shadertoy: https://www.shadertoy.com/view/4dX3zl
const Size = 10.; //Adjust this for denser voxels
const MaxIterations = 64u * u32(ceil(Size));


// This can be changed to index into a buffer, 3d texture, etc.
// In that case, dividing by `Size` would not be necessary, this is just for tracing
// SDFs in higher resolution as shown in this demo.
fn GetVoxel(c : vec3<f32>) -> bool{
    let p = c / Size + vec3<f32>(.5);
    let d = min(max(-SDSphere(p, 7.5), SDBox(p, vec3<f32>(6.))), -SDSphere(p, 25.));
    return d < 0.;
}


@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) Pixel: uint3) {
    let Resolution = textureDimensions(screen).xy;
    let AspectRatio = f32(Resolution.y) / f32(Resolution.x);

    if (Pixel.x >= Resolution.x || Pixel.y >= Resolution.y){
        return;
    }
    
    // Get pixel coordinate in floating point vector, the +/- .5 makes it trace
    // at the centre of the pixel instead of the bottom left corner
    let FragCoord = vec2<f32>(f32(Pixel.x) + .5, f32(Resolution.y - Pixel.y) - .5);
    
    // Same as pixel coordinate, but mapped from -1 .. 1 in each axis
    let UV = 2. * FragCoord / vec2<f32>(Resolution) - 1.;

    // Get 3d ray direction vector
    let CameraDirection = vec3<f32>(0., 0., .8);
    let CameraPlaneU = vec3<f32>(1., 0., 0.);
    let CameraPlaneV = vec3<f32>(0., AspectRatio, 0.);
    
    var RayDirection = CameraDirection + UV.x * CameraPlaneU + UV.y * CameraPlaneV;

    // Initial position of ray
    var RayPosition = vec3<f32>(0., 2. * sin(time.elapsed * 2.7), -12.);


    let DirectionRotation = Rotate2D(RayDirection.xz, time.elapsed);
    let PositionRotation = Rotate2D(RayPosition.xz, time.elapsed);

    // Rotate camera direction
    RayDirection = vec3<f32>(DirectionRotation.x, RayDirection.y, DirectionRotation.y);
    
    // Rotate camera position
    RayPosition = vec3<f32>(PositionRotation.x, RayPosition.y, PositionRotation.y) * Size;


    // DDA logic
    let DeltaDistance = abs(vec3(length(RayDirection)) / RayDirection);
    let RayStep = sign(RayDirection);
    var MapPosition = floor(RayPosition);
    var SideDistance = (sign(RayDirection) * (MapPosition - RayPosition) + (sign(RayDirection) * .5) + .5) * DeltaDistance;
    var Normal = vec3<f32>(0.);

    for(var i : u32 = 0u; i < MaxIterations; i++){
        if(GetVoxel(MapPosition)){
            break;
        }
        Normal = step(SideDistance, min(SideDistance.yxy, SideDistance.zzx));
        SideDistance = fma(Normal, DeltaDistance, SideDistance);
        MapPosition = fma(Normal, RayStep, MapPosition);
    }

    let Colour = vec3<f32>(length(Normal * vec3<f32>(.75, 1., .5)));
    textureStore(screen, Pixel.xy, vec4<f32>(Colour, 1.));
}



fn SDSphere(p : vec3<f32>, d : f32) -> f32{
    return length(p) - d;
}

fn SDBox(p : vec3<f32>, b : vec3<f32>) -> f32{
    let d = abs(p) - b;
    return min(max(d.x, max(d.y, d.z)), 0.) + length(max(d, vec3<f32>(0.)));
}


fn Rotate2D(v : vec2<f32>, a : f32) -> vec2<f32>{
    let SinA = sin(a);
    let CosA = cos(a);
    return vec2<f32>(v.x * CosA - v.y * SinA, v.y * CosA + v.x * SinA);
}