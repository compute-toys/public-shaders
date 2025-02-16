#storage Memory MemoryStruct

const FOV = 1.54;

//https://github.com/toji/gl-matrix/blob/master/src/mat4.js
fn Projection(FOV : f32, Aspect : f32, Near : f32, Far : f32) -> mat4x4<f32>{
    let f = 1. / tan(FOV / 2.);
    let nf = 1. / (Near - Far);

    return mat4x4<f32>(
        vec4<f32>(f / Aspect, 0., 0., 0.),
        vec4<f32>(0., f, 0., 0.),
        vec4<f32>(0., 0., (Far + Near) * nf, -1.),
        vec4<f32>(0., 0., 2. * Far * Near * nf, 0.)
    );
}

fn RotateX(m : mat4x4<f32>, a : f32) -> mat4x4<f32>{
    let s = sin(a);
    let c = cos(a);
    return mat4x4<f32>(
        m[0],
        m[1] * c + m[2] * s,
        m[2] * c - m[1] * s,
        m[3]
    );
}
fn RotateY(m : mat4x4<f32>, a : f32) -> mat4x4<f32>{
    let s = sin(a);
    let c = cos(a);
    return mat4x4<f32>(
        m[0] * c - m[2] * s,
        m[1],
        m[2] * c + m[0] * s,
        m[3]
    );
}

fn Translate(m : mat4x4<f32>, c : vec3<f32>) -> mat4x4<f32>{
    return mat4x4<f32>(
        m[0],
        m[1],
        m[2],
        vec4<f32>(
            m[0].x * c.x + m[1].x * c.y + m[2].x * c.z + m[3].x,
            m[0].y * c.x + m[1].y * c.y + m[2].y * c.z + m[3].y,
            m[0].z * c.x + m[1].z * c.y + m[2].z * c.z + m[3].z,
            m[0].w * c.x + m[1].w * c.y + m[2].w * c.z + m[3].w
        )
    );
}

//https://www.shadertoy.com/view/lslXDf
fn Barycentric(a : vec2<f32>, b : vec2<f32>, c : vec2<f32>, p : vec2<f32>) -> vec3<f32>{
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;
    let inv_denom = 1. / (v0.x * v1.y - v1.x * v0.y);
    let v = (v2.x * v1.y - v1.x * v2.y) * inv_denom;
    let w = (v0.x * v2.y - v2.x * v0.y) * inv_denom;
    let u = 1. - v - w;
    return abs(vec3<f32>(u,v,w));
}

const IdentityMatrix = mat4x4<f32>(1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.);

struct MemoryStruct{
    ModelViewProjection : mat4x4<f32>,
    Vertices : array<vec4<f32>, 3>,
    Points : array<vec2<f32>, 3>,
    AspectRatio : f32,
    Position : vec3<f32>,
    Rotation : vec2<f32>,
    IsBehindCamera : u32
}

#workgroup_count Setup 1 1 1
@compute @workgroup_size(1)
fn Setup(){
    HandleControls();

    Memory.AspectRatio = f32(textureDimensions(screen).x) / f32(textureDimensions(screen).y);
    var Rotation = Memory.Rotation; //vec2<f32>(0., 0.);
    var Position = Memory.Position * .05; //vec3<f32>(0., 0., sin(time.elapsed) + 3.);
    
    //Rotation.x = sin(time.elapsed);
    
    let ProjectionMatrix = Projection(FOV, Memory.AspectRatio, 1., 2.);
    let ModelViewMatrix = Translate(RotateY(RotateX(IdentityMatrix, Rotation.y), Rotation.x), -Position); //mat4x4<f32>(1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.);
    Memory.ModelViewProjection = ProjectionMatrix * ModelViewMatrix;
    
    Memory.Vertices[0] = vec4<f32>(1.5, .2, 2.5, 1.);
    Memory.Vertices[1] = vec4<f32>(1., 1., 1., 1.);
    Memory.Vertices[2] = vec4<f32>(2.6, -1.2, .7, 1.);
    
    Memory.IsBehindCamera = 0u; //I won't render the triangle if one or more vertices is behind the camera

    for(var i = 0; i < 3; i++){
        let Projected = Memory.ModelViewProjection * Memory.Vertices[i];
        var Point = Projected.xy / Projected.w;
        if(Projected.w <= 0.){//Vertex is behind camera
            Memory.IsBehindCamera++;
        }
        Memory.Points[i] = Point;
    }
}


@compute @workgroup_size(16, 16)
fn MainImage(@builtin(global_invocation_id) Pixel: uint3) {
    let Resolution = textureDimensions(screen).xy;

    if (Pixel.x >= Resolution.x || Pixel.y >= Resolution.y){
        return;
    }
    
    let FragCoord = vec2<f32>(float(Pixel.x) + .5, float(Resolution.y - Pixel.y) - .5);

    let UV = 2. * FragCoord / vec2<f32>(Resolution) - 1.;
    
    var Colour : vec3<f32>;
    
    let b = Barycentric(Memory.Points[0], Memory.Points[1], Memory.Points[2], UV);
    if(Memory.IsBehindCamera == 0u && dot(b, vec3<f32>(1.)) - 1. <= 1e-5){
        Colour = b;
    } else{
        Colour = .5 + .5 * cos(time.elapsed + UV.xyx + float3(0.,2.,4.));
    }

    Colour = pow(Colour, vec3<f32>(2.2));
    textureStore(screen, Pixel.xy, vec4<f32>(Colour, 1.));
}

fn HandleControls(){
    var MousePos = vec2<f32>(mouse.pos);
    if(MousePos.x == 0. && MousePos.y == 0.){
        MousePos = vec2<f32>(400., 250.);
    }
    MousePos.y = MousePos.y - f32(textureDimensions(screen).y) / 2.;
    Memory.Rotation = vec2<f32>(
        f32(MousePos.x) / 200.,
        clamp(f32(MousePos.y) / 200., -1.570796, 1.570796)
    );

    let Movement = vec3<f32>(vec3<i32>(vec3<u32>(
        (KeyDown(39) | KeyDown(68)) - (KeyDown(37) | KeyDown(65)),
        KeyDown(32) - KeyDown(16),
        (KeyDown(38) | KeyDown(87)) - (KeyDown(40) | KeyDown(83))
    )));
    
    let RotationF = normalize(GetRayDirection(vec2<f32>(0.), Memory.Rotation).xz);
    let RotationS = normalize(GetRayDirection(vec2<f32>(0.), vec2<f32>(Memory.Rotation.x + 1.57, Memory.Rotation.y)).xz);

    Memory.Position.y += Movement.y;
    Memory.Position.x += Movement.z * RotationF.x + Movement.x * RotationS.x;
    Memory.Position.z += Movement.z * RotationF.y + Movement.x * RotationS.y;
}

fn KeyDown(Code : u32) -> u32{
    return (_keyboard[Code >> 7u][(Code >> 5u) & 3u] >> (Code & 31u)) & 1u;
}

fn GetRayDirection(UV : vec2<f32>, Rotation : vec2<f32>) -> vec3<f32>{
    let Resolution = textureDimensions(screen).xy;
    
    var RayDirection = vec3<f32>(-UV.x * Memory.AspectRatio, UV.y, 1. / tan(FOV / 2.));
    RayDirection *= RotationMatrixX(Rotation.y);
    RayDirection *= RotationMatrixY(3.14159 - Rotation.x);
    return RayDirection;
}

fn RotationMatrixX(a : f32) -> mat3x3<f32>{
    let c = cos(a);
    let s = sin(a);
    return mat3x3<f32>(
        1., 0., 0.,
        0., c ,-s ,
        0., s , c
    );
}

fn RotationMatrixY(a : f32) -> mat3x3<f32>{
    let c = cos(a);
    let s = sin(a);
    return mat3x3<f32>(
        c,  0., s ,
        0., 1., 0.,
        -s, 0., c
    );
}