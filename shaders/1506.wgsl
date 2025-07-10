// For nicer controls, paste the following code below into the browser console.
// It enables a pointer lock, which means that mouse movement isn't restricted,
// and it stops the page from scrolling when you spress space to fly up.
// To activate, click on the canvas. To deactivate, press alt or escape.
/*
void function(){
  //If this stops working, please feel free to DM me on Discord at @57a.
  //Last update: 2025-07-09.
  const MouseSensitivity = 0.2; //Set your sensitivity here
  const Canvas = document.querySelector("#editor-canvas");
  let x = 100000;
  let y = Canvas.height >> 1;
  window.addEventListener("keydown", function(Event){
    if(document.pointerLockElement !== Canvas) return;
    if(Event.code === "Space") Event.preventDefault();
  });
  Canvas.addEventListener("mousedown", function(){
    if(document.pointerLockElement !== Canvas){
        Canvas.onmousedown(new MouseEvent("mousedown", {"button": 0, "buttons": 1, "clientX": x, "clientY": y}));
        Canvas.requestPointerLock({"unadjustedMovement": true});
        Canvas.focus();
    }
  });
  document.addEventListener("pointerlockchange", function(){
    if(document.pointerLockElement !== Canvas){
        Canvas.onmouseup(new MouseEvent("mouseup", {"button": 0, "buttons": 1, "clientX": x, "clientY": y}));
    }
  });
  Canvas.addEventListener("mousemove", function(Event){
    if(document.pointerLockElement !== Canvas) return;
    Canvas.onmouseleave(new MouseEvent("mouseleave"));
    x = x + Event.movementX * MouseSensitivity;
    y = Math.max(Math.min(y + Event.movementY * MouseSensitivity, Canvas.height), 0.);
    Canvas.onmousedown(new MouseEvent("mousedown", {"button": 0, "buttons": 1, "clientX": x, "clientY": y}));
  });
}();
*/
#storage Memory MemoryStruct

#storage WorldData array<f32,33554432>





const HORIZONTAL_SCALE : f32 = 60000.0;
const VERTICAL_SCALE : f32 = 160000.0;

const EROSION_TILES_I : i32 = 8;
const EROSION_TILES_F : f32 = 8.;
const EROSION_OCTAVES : i32 = 8;
const EROSION_GAIN : f32 = 0.5;
const EROSION_LACUNARITY_F : f32 = 2.;
const EROSION_LACUNARITY_I : i32 = 2;
const EROSION_SLOPE_STRENGTH : f32 = 3.0;
const EROSION_BRANCH_STRENGTH : f32 = 2.0;
const EROSION_STRENGTH : f32 = 0.0125;
const EROSION_SIZE : i32 = 5120;

const HEIGHT_TILES_F : f32 = 3.0;
const HEIGHT_TILES_I : i32 = 3;
const HEIGHT_OCTAVES : i32 = 3;
const HEIGHT_AMP : f32 = 0.25;
const HEIGHT_GAIN : f32 = 0.1;
const HEIGHT_LACUNARITY_F : f32 = 2.;
const HEIGHT_LACUNARITY_I : i32 = 2;

fn Hash(ix : i32, iy : i32) -> vec2<f32>{
    var x = u32((0x51571302 * iy) ^ ix);
    x = x ^ (x >> 15);
    x = x * 0x21f0aaad;
    x = x ^ (x >> 15);
    x = x * 0xd35a2d97;
    x = x ^ (x >> 15);
    return vec2<f32>(
        bitcast<f32>(0x40000000 | (x & 0x007fffff)) - 3.,
        bitcast<f32>(0x40000000 | (x >> 9)) - 3.
    );
}

fn Hash_05(ix : i32, iy : i32) -> vec2<f32>{
    var x = u32((0x51571302 * iy) ^ ix);
    x = x ^ (x >> 15);
    x = x * 0x21f0aaad;
    x = x ^ (x >> 15);
    x = x * 0xd35a2d97;
    x = x ^ (x >> 15);
    return vec2<f32>(
        bitcast<f32>(0x3f800000 | (x & 0x007fffff)) - 1.5,
        bitcast<f32>(0x3f800000 | (x >> 9)) - 1.5
    );
}

fn DerivativeNoise(ix : i32, iy : i32, fx : f32, fy : f32) -> vec3<f32>{
    let ux = (((fx * fx) * fx) * ((fx * ((fx * 6.) - 15.)) + 10.));
    let uy = (((fy * fy) * fy) * ((fy * ((fy * 6.) - 15.)) + 10.));
    let dux = (((30. * fx) * fx) * ((fx * (fx - 2.)) + 1.));
    let duy = (((30. * fy) * fy) * ((fy * (fy - 2.)) + 1.));
    let ga = Hash(ix, iy);
    let gb = Hash(ix + 1, iy);
    let gc = Hash(ix, iy + 1);
    let gd = Hash(ix + 1, iy + 1);

    let va = ((ga.x * fx) + (ga.y * fy));
    let vb = ((gb.x * (fx - 1.)) + (gb.y * fy));
    let vc = ((gc.x * fx) + (gc.y * (fy - 1.)));
    let vd = ((gd.x * (fx - 1.)) + (gd.y * (fy - 1.)));

    let h = (((va + (ux * (vb - va))) + (uy * (vc - va))) + ((ux * uy) * (((va - vb) - vc) + vd)));
    let dx = ((((ga.x + (ux * (gb.x - ga.x))) + (uy * (gc.x - ga.x))) + ((ux * uy) * (((ga.x - gb.x) - gc.x) + gd.x))) + (dux * (((uy * (((va - vb) - vc) + vd)) + vb) - va)));
    let dy = ((((ga.y + (ux * (gb.y - ga.y))) + (uy * (gc.y - ga.y))) + ((ux * uy) * (((ga.y - gb.y) - gc.y) + gd.y))) + (duy * (((ux * (((va - vb) - vc) + vd)) + vc) - va)));
    
    return vec3<f32>(h, dx, dy);
}

fn Erosion(ipx : i32, ipy : i32, fpx : f32, fpy : f32, _dirx : f32, _diry : f32) -> vec3<f32>{
    var dirx = _dirx;
    var diry = -_diry;

    let a = sqrt(((dirx * dirx) + (diry * diry)));

    //I added this:
    dirx = (((dirx / (a + 0.05)) * 0.2) + (dirx * 0.8));
    diry = (((diry / (a + 0.05)) * 0.2) + (diry * 0.8));

    var wt = 0.;
    var vax = 0.;
    var vay = 0.;
    var vaz = 0.;
    
    for(var i = -2; i < 2; i++){
      let ox = f32(i);
      for(var j = -2; j < 2; j++){
        let oy = f32(j);
        let h = Hash_05((ipx - i), (ipy - j));
        let ppx = ((fpx + ox) - h.x);
        let ppy = ((fpy + oy) - h.y);
        let d = ((ppx * ppx) + (ppy * ppy));
        
        let Temp2 = (d + 1.3);
        let w = max(((2.197 - d) / ((Temp2 * Temp2) * Temp2)), 0.);
        wt = (wt + w);

        let mag = ((ppx * dirx) + (ppy * diry));

        vax += cos(mag * 6.28318) * w;
        vay += (-sin(mag * 6.28318) * (ppx + dirx)) * w;
        vaz += (-sin(mag * 6.28318) * (ppy + diry)) * w;
      }
    }
    return vec3<f32>(vax / wt, vay / wt, vaz / wt);
}


fn Heightmap(X : i32, Z : i32) -> f32{
    let ix = X >> 16;
    let iz = Z >> 16;
    let fx = f32(X & 65535) / 65536.;
    let fz = f32(Z & 65535) / 65536.;

    var nff = 1.;
    var nfi = 1;
    var na = HEIGHT_AMP;

    var nx = 0.;
    var ny = 0.;
    var nz = 0.;

    for(var i = 0; i < HEIGHT_OCTAVES; i++){
        let Result = DerivativeNoise(
            ((ix * nfi) + i32(floor((fx * nff)))),
            ((iz * nfi) + i32(floor((fz * nff)))),
            fract((fx * nff)),
            fract((fz * nff))
        );
        nx = ((Result.x * na) + nx);
        ny = (((Result.y * na) * nff) + ny);
        nz = (((Result.z * na) * nff) + nz);

        na *= HEIGHT_GAIN;
        nff *= HEIGHT_LACUNARITY_F;
        nfi *= HEIGHT_LACUNARITY_I;
    }
    return nx * .5;
}

fn GetErosion(X : i32, Z : i32) -> f32{
    let ix = X >> 16;
    let iz = Z >> 16;
    let fx = f32(X & 65535) / 65536.;
    let fz = f32(Z & 65535) / 65536.;

    var xM = (Heightmap((X - EROSION_SIZE), Z) * VERTICAL_SCALE);
    var xP = (Heightmap((X + EROSION_SIZE), Z) * VERTICAL_SCALE);
    var zM = (Heightmap(X, (Z - EROSION_SIZE)) * VERTICAL_SCALE);
    var zP = (Heightmap(X, (Z + EROSION_SIZE)) * VERTICAL_SCALE);


    var dirx = ((zP - zM) * -0.00009765625);
    var dirz = ((xP - xM) * -0.00009765625);

    var a = 0.5;
    var ff = 1.;
    var fi = 1;

    var hx = 0.;
    var hy = 0.;
    var hz = 0.;

    //a = f32x4.mul(Smoothstep(f32x4.sub(v128.const<[f32x4, WATER_HEIGHT]>(), v128.const<[f32x4, 0.13]>()), f32x4.add(v128.const<[f32x4, WATER_HEIGHT]>(), v128.const<[f32x4, 0.27]>()), nx), a);

    for(var i = 0; i < EROSION_OCTAVES; i++){
        let Result = Erosion(
            (((ix * fi) * EROSION_TILES_I) + i32(floor(((fx * ff) * EROSION_TILES_F)))),
            (((iz * fi) * EROSION_TILES_I) + i32(floor(((fz * ff) * EROSION_TILES_F)))),
            fract(((fx * ff) * EROSION_TILES_F)),
            fract(((fz * ff) * EROSION_TILES_F)),
            (dirx + (hz * EROSION_BRANCH_STRENGTH)),
            (dirz + (hy * EROSION_BRANCH_STRENGTH))
        );
        var Temp = (Result.x * a);
        hx = (Temp + hx);

        hy = (((Result.y * a) * ff) + hy);
        hz = (((Result.z * a) * ff) + hz);

        a = (EROSION_GAIN * a);
        ff = (EROSION_LACUNARITY_F * ff);
        fi = (EROSION_LACUNARITY_I * fi);
    }

    return ((hx - .5) * EROSION_STRENGTH);
    //return 
    //  i32x4.trunc_sat_f32x4_s(f32x4.mul(f32x4.mul(dirx, hx), v128.const<[f32x4, 30000.0]>())),
    //  i32x4.trunc_sat_f32x4_s(f32x4.mul(f32x4.mul(dirz, hx), v128.const<[f32x4, 30000.0]>()));
}

fn GetHeight(X : i32, Z : i32) -> f32{
    var Height = Heightmap(X, Z);

    let ErosionStrength = GetErosion(X, Z);
    Height = (Height + ErosionStrength);

    return Height * VERTICAL_SCALE;
}










































const MaxIterations = 13000u;
const EquirectangularSize = 2.0;

fn KeyDown(Code : u32) -> u32{
    return (_keyboard[Code >> 7u][(Code >> 5u) & 3u] >> (Code & 31u)) & 1u;
}

struct MemoryStruct{
    Position : vec3<f32>,
    LastMousePos : vec2<u32>,
    LastMousePos2 : vec2<u32>,
    MousePos : vec2<i32>,
    MinMaxY : array<AtomicMinMax, 32768>,
    EquirectangularImage : array<vec2<f32>>
}

struct AtomicMinMax{
    Min : atomic<u32>,
    Max : atomic<u32>
}


#workgroup_count GenerateWorld 32 2048 1
@compute @workgroup_size(64)
fn GenerateWorld(@builtin(global_invocation_id) ID : vec3<u32>){
    /*if(KeyDown(82) == 0){
        return;
    }*/
    WorldData[(ID.y << 12) | ID.x] = textureLoad(channel0, ID.xy, 0).r * -64.;//GetHeight(i32(ID.x) << 7, i32(ID.y) << 7) * .02;
}

#workgroup_count ControlsPass 1 1 1
@compute @workgroup_size(1)
fn ControlsPass(@builtin(global_invocation_id) ID: vec3<u32>){
    if(any(ID != vec3(0u))) {
        return;
    }
    let Movement = vec3<f32>(vec3<i32>(vec3<u32>(
        (KeyDown(39) | KeyDown(68)) - (KeyDown(37) | KeyDown(65)),
        KeyDown(16) - KeyDown(32),
        (KeyDown(38) | KeyDown(87)) - (KeyDown(40) | KeyDown(83))
    ))) * 1.;
    
    let LastMousePos = Memory.LastMousePos2;
    
    var MousePos = vec2<i32>(mouse.pos);
    MousePos.y = MousePos.y;
    Memory.MousePos = MousePos;
    
    let RotationF = normalize(GetRayDirection(vec2<f32>(0.)).xz);
    let RotationS = normalize(GetRayDirection(vec2<f32>(1024., 0.)).xz);

    Memory.Position.y += Movement.y;
    Memory.Position.x += Movement.z * RotationF.x + Movement.x * RotationS.x;
    Memory.Position.z += Movement.z * RotationF.y + Movement.x * RotationS.y;
}

#workgroup_count Clear 1024 1 1
@compute @workgroup_size(32)
fn Clear(@builtin(global_invocation_id) ID : vec3<u32>){
    atomicStore(&Memory.MinMaxY[ID.x].Min, 4294967295u);
    atomicStore(&Memory.MinMaxY[ID.x].Max, 0u);
}

@compute @workgroup_size(16, 16)
fn SetMinMaxY(@builtin(global_invocation_id) Pixel : vec3<u32>){
    let Resolution = textureDimensions(screen).xy;

    if (Pixel.x >= Resolution.x || Pixel.y >= Resolution.y){
        return;
    }
    
    let FragCoord = vec2<f32>(float(Pixel.x) + .5, float(Resolution.y - Pixel.y) - .5);

    let UV = 2. * FragCoord / vec2<f32>(Resolution) - 1.;

    let RayDirection = normalize(GetRayDirection(UV));

    
    let PI = 3.14159;
    let TextureCoordinate = vec2<f32>(atan2(RayDirection.z, RayDirection.x) + PI, acos(-RayDirection.y)) / vec2(2. * PI, PI);

    let EquirectangularPixel = vec2<u32>(TextureCoordinate * vec2<f32>(vec2<u32>(vec2<f32>(textureDimensions(screen).xy) * EquirectangularSize)));
    
    atomicMin(&Memory.MinMaxY[EquirectangularPixel.x].Min, EquirectangularPixel.y);
    atomicMax(&Memory.MinMaxY[EquirectangularPixel.x].Max, EquirectangularPixel.y);
}


fn Rotate2D(v : vec2<f32>, a : f32) -> vec2<f32>{
    let SinA : f32 = sin(a);
    let CosA : f32 = cos(a);
    return vec2<f32>(v.x * CosA - v.y * SinA, v.y * CosA + v.x * SinA);
}

fn GetRayDirection(UV : vec2<f32>) -> vec3<f32>{
    let Resolution = textureDimensions(screen).xy;
    let AspectRatio = f32(Resolution.y) / f32(Resolution.x);

    let RotationX = f32(-Memory.MousePos.x) / 100.;
    let RotationY = f32(Memory.MousePos.y) / 100.;
    let MatrixX = mat2x2<f32>(cos(RotationX),sin(RotationX),-sin(RotationX),cos(RotationX));
    let MatrixY = mat2x2<f32>(cos(RotationY),sin(RotationY),-sin(RotationY),cos(RotationY));

    var RayDirection = (vec3<f32>(UV.x / AspectRatio, UV.y, 1.));

    let Y = vec2<f32>(cos(RotationY) * RayDirection.y + sin(RotationY) * RayDirection.z, cos(RotationY) * RayDirection.z - sin(RotationY) * RayDirection.y);//RayDirection.yz * MatrixY;
    RayDirection = vec3<f32>(RayDirection.x, Y.xy);

    let X = vec2<f32>(cos(RotationX) * RayDirection.x + sin(RotationX) * RayDirection.z, cos(RotationX) * RayDirection.z - sin(RotationX) * RayDirection.x);//RayDirection.xz * MatrixX;
    RayDirection = vec3<f32>(X.x, RayDirection.y, X.y);

    return (RayDirection);
}

fn EquirectangularStore(Pixel : vec2<u32>, Value : vec2<f32>, Resolution : vec2<u32>){
    let Index = Pixel.x * Resolution.y + Pixel.y;
    Memory.EquirectangularImage[Index] = Value;
}

fn EquirectangularSample(Coordinate : vec2<f32>, Resolution : vec2<u32>) -> vec2<f32>{
    let Pixel = vec2<u32>(Coordinate * vec2<f32>(Resolution));
    return Memory.EquirectangularImage[Pixel.x * Resolution.y + Pixel.y];
}

fn Tan(x : f32) -> f32{
    let S = x * x;
    return x * (2.471688400562703 - 0.189759681063053 * S) / (2.4674011002723397 - S);
    //return x * fma(x * -0.189759681063053, x, 2.471688400562703) / fma(x, -x, 2.4674011002723397);
}

#workgroup_count Equirectangular 2048 1 1
@compute @workgroup_size(32)
fn Equirectangular(@builtin(global_invocation_id) Pixel: uint3) {
    let Resolution = vec2<u32>(vec2<f32>(textureDimensions(screen).xy) * EquirectangularSize);

    if (Pixel.x >= Resolution.x){
        return;
    }

    let AspectRatio = f32(Resolution.y) / f32(Resolution.x);
    
    let RotationX = f32(Memory.MousePos.x) / 100.;
    let RotationY = f32(Memory.MousePos.y) / 100.;

    let RayPositionX = Memory.Position.x;
    let RayPositionY = Memory.Position.y;
    let RayPositionZ = Memory.Position.z;

    let CosRotationX = cos(RotationX);
    let SinRotationX = sin(RotationX);

    let RayDirectionYMin = RotationY - 1.;
    let RayDirectionYRange = 2.;
    let RayDirectionYIncrement = RayDirectionYRange / f32(Resolution.y);

    let x = Pixel.x;

    let a = (f32(x) / f32(Resolution.x) - .5) * 3.14159 * 2.;
    let RayDirectionX = cos(a);
    let RayDirectionZ = sin(a);

    let RayStepX = i32(sign(RayDirectionX));
    let RayStepZ = i32(sign(RayDirectionZ));

    let StartMapPositionX = i32(floor(RayPositionX));
    let StartMapPositionZ = i32(floor(RayPositionZ));

    var MapPositionX = StartMapPositionX;
    var MapPositionZ = StartMapPositionZ;

    let DeltaDistanceX = abs(1. / RayDirectionX);
    let DeltaDistanceZ = abs(1. / RayDirectionZ);

    let OriginalSideDistanceX = (sign(RayDirectionX) * (floor(RayPositionX) - RayPositionX) + sign(RayDirectionX) * .5 + .5) * DeltaDistanceX;
    let OriginalSideDistanceZ = (sign(RayDirectionZ) * (floor(RayPositionZ) - RayPositionZ) + sign(RayDirectionZ) * .5 + .5) * DeltaDistanceZ;
    var SideDistanceX = (sign(RayDirectionX) * (floor(RayPositionX) - RayPositionX) + sign(RayDirectionX) * .5 + .5) * DeltaDistanceX;
    var SideDistanceZ = (sign(RayDirectionZ) * (floor(RayPositionZ) - RayPositionZ) + sign(RayDirectionZ) * .5 + .5) * DeltaDistanceZ;

    var Distance = 0.;
    var PreviousHeight = 1.e37;
    var SideIsX = false;
    var FinishedOnTop = false;


    let MinY = atomicLoad(&Memory.MinMaxY[Pixel.x].Min) - 1;
    let MaxY = atomicLoad(&Memory.MinMaxY[Pixel.x].Max);
    if(MinY > MaxY){
        return;
    }

    let InvResolutionY = 3.14159265359 / f32(Resolution.y);
    let Minus = -1.57079632679;

    var y = MaxY;

    var RayDirectionY = Tan(fma(f32(y), InvResolutionY, Minus));

    for(var i = 0u; i <= (MaxIterations >> 1); i++){
        var Height = textureLoad(channel0, vec2<i32>(MapPositionX & 4095, MapPositionZ & 4095), 0).r * -64.;//WorldData[(((MapPositionZ) & 4095) << 12) | ((MapPositionX) & 4095)];//sin(f32(MapPositionZ + MapPositionX) / 10.) * 50.;
        var MinHeight = min(Height, PreviousHeight);
        loop{
            if(y == MinY || MinHeight >= fma(RayDirectionY, Distance, RayPositionY)){
                break;
            }
            let C = PreviousHeight >= fma(RayDirectionY, Distance, RayPositionY);
            let A = select(PreviousHeight - RayPositionY, Distance, C);
            let B = select(A / RayDirectionY, RayDirectionY * Distance, C);
            let Distance3D = sqrt(A * A + B * B);
            EquirectangularStore(vec2<u32>(x, y), vec2<f32>(Distance3D, select(PreviousHeight, Height, C)), Resolution);
            y--;
            //RayDirectionY -= RayDirectionYIncrement;
            RayDirectionY = Tan(fma(f32(y), InvResolutionY, Minus));
        }
        PreviousHeight = Height;//sin(f32(MapPositionZ + MapPositionX) / 10.) * 50.;//WorldData[(((MapPositionZ) & 4095) << 12) | ((MapPositionX) & 4095)];//GetHeight(i32(MapPositionX) << 7, i32(MapPositionZ) << 7) * .02;

        SideIsX = SideDistanceX < SideDistanceZ;
        Distance = select(SideDistanceZ, SideDistanceX, SideIsX);/* select(
            OriginalSideDistanceZ + abs(DeltaDistanceZ * f32(StartMapPositionZ - MapPositionZ)),
            OriginalSideDistanceX + abs(DeltaDistanceX * f32(StartMapPositionX - MapPositionX)),
            SideIsX
        );*/
        if(SideIsX){
          SideDistanceX += DeltaDistanceX;
          MapPositionX += RayStepX;
        } else{
          SideDistanceZ += DeltaDistanceZ;
          MapPositionZ += RayStepZ;
        }

        
        Height = textureLoad(channel0, vec2<i32>(MapPositionX & 4095, MapPositionZ & 4095), 0).r * -64.;//WorldData[(((MapPositionZ) & 4095) << 12) | ((MapPositionX) & 4095)];//sin(f32(MapPositionZ + MapPositionX) / 10.) * 50.;
        MinHeight = min(Height, PreviousHeight);
        loop{
            if(y == MinY || MinHeight >= fma(RayDirectionY, Distance, RayPositionY)){
                break;
            }
            let C = PreviousHeight >= fma(RayDirectionY, Distance, RayPositionY);
            let A = select(PreviousHeight - RayPositionY, Distance, C);
            let B = select(A / RayDirectionY, RayDirectionY * Distance, C);
            let Distance3D = sqrt(A * A + B * B);
            EquirectangularStore(vec2<u32>(x, y), vec2<f32>(Distance3D, select(PreviousHeight, Height, C)), Resolution);
            y--;
            //RayDirectionY -= RayDirectionYIncrement;
            RayDirectionY = Tan(fma(f32(y), InvResolutionY, Minus));
        }
        PreviousHeight = Height;//sin(f32(MapPositionZ + MapPositionX) / 10.) * 50.;//WorldData[(((MapPositionZ) & 4095) << 12) | ((MapPositionX) & 4095)];//GetHeight(i32(MapPositionX) << 7, i32(MapPositionZ) << 7) * .02;

        SideIsX = SideDistanceX < SideDistanceZ;
        Distance = select(SideDistanceZ, SideDistanceX, SideIsX);/*select(
            OriginalSideDistanceZ + abs(DeltaDistanceZ * f32(StartMapPositionZ - MapPositionZ)),
            OriginalSideDistanceX + abs(DeltaDistanceX * f32(StartMapPositionX - MapPositionX)),
            SideIsX
        );*/
        if(SideIsX){
          SideDistanceX += DeltaDistanceX;
          MapPositionX += RayStepX;
        } else{
          SideDistanceZ += DeltaDistanceZ;
          MapPositionZ += RayStepZ;
        }
    }

    while(y != MinY){
        EquirectangularStore(vec2<u32>(x, y), vec2<f32>(1.e37, 0.), Resolution);
        y--;
    }

    /*for(var i = MinY; i < MaxY; i++){
        EquirectangularStore(vec2<u32>(x, i), vec4<f32>(vec3<f32>(.3, .0, 1.), 1.), Resolution);
    }*/
}

struct _SampleAtFragCoord{
    Position : vec3<f32>,
    Height : f32
}

fn SampleAtFragCoord(FragCoord : vec2<f32>) -> _SampleAtFragCoord{
    let Resolution = textureDimensions(screen).xy;
    let UV = 2. * FragCoord / vec2<f32>(Resolution) - 1.;

    let RayDirection = normalize(GetRayDirection(UV));

    
    let PI = 3.14159;
    let TextureCoordinate = vec2<f32>(atan2(RayDirection.z, RayDirection.x) + PI, acos(-RayDirection.y)) / vec2(2. * PI, PI);

    let Value = EquirectangularSample(TextureCoordinate, vec2<u32>(vec2<f32>(Resolution) * EquirectangularSize));

    return _SampleAtFragCoord(RayDirection * Value.x, Value.y);
}

@compute @workgroup_size(16, 16)
fn MainImage(@builtin(global_invocation_id) Pixel: uint3){
    let Resolution = textureDimensions(screen).xy;

    if (Pixel.x >= Resolution.x || Pixel.y >= Resolution.y){
        return;
    }

    // View equirectangular
    //textureStore(screen, Pixel.xy, unpack4x8unorm(Memory.EquirectangularImage[Pixel.y * Resolution.x + Pixel.x]));
    //return;
    
    let FragCoord = vec2<f32>(float(Pixel.x) + .5, float(Resolution.y - Pixel.y) - .5);

    let A = SampleAtFragCoord(FragCoord);
    if(length(A.Position) > 1e11){
        textureStore(screen, Pixel.xy, vec4<f32>(.3, .6, .9, 1.));
        return;
    }
    var B = SampleAtFragCoord(FragCoord - vec2<f32>(1., 0.));
    if(length(B.Position) > 1e11){
        B = SampleAtFragCoord(FragCoord + vec2<f32>(1., 0.));
    }
    var C = SampleAtFragCoord(FragCoord - vec2<f32>(0., 1.));
    if(length(C.Position) > 1e11){
        C = SampleAtFragCoord(FragCoord + vec2<f32>(0., 1.));
    }

    let Colour = vec3<f32>(length(normalize(cross(B.Position - A.Position, C.Position - A.Position)) * vec3<f32>(.55, .95, .75)));
    
    
    textureStore(screen, Pixel.xy, vec4<f32>(Colour * (-A.Height / 64.), 1.));
    
}