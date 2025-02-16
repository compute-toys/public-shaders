//just found that the code that translates data using a velocity field
//is not needed so that code is removed in this post
//all those years trying to find perfect algorithm that translates data
//using velocity were in vain because that algorithm is not needed in the first place

#define rez 192u    //simulationCube side length
#define rZ 4f       //cube side length that reads  to simulationCube
#define PI 3.141592653589793f

#storage D array<vec3f,rez*rez*rez>;
#storage C array<f32,rez*rez*rez>;
var<workgroup> D2: array<vec3f,rez>;
var<workgroup> C2: array<f32,rez>;
fn hash(x2: u32) -> u32
{
    var x = x2;
    x ^= x >> 17;  x *= 0xed5ad4bbu;
    x ^= x >> 11;  x *= 0xac4c1b51u;
    x ^= x >> 15;  x *= 0x31848babu;
    x ^= x >> 14;  return x;
}
fn rnd(a: u32) -> f32
{
    var h   = hash(a);
    var msk = (1u << 23u) - 1u;
    return f32(h & msk) / f32(msk);
}
fn wave(v:f32) -> f32
{
    var l = v+f32(v==0f)*.000001f;
    var t   = 8f*custom.a;   //how expanded is the wave
    var wid = 4f*custom.b;     //wave width
    var c   = 1f/wid;
    return 1f/l*(-(t - l)*exp(-pow(t - l,2f)/c/c)
                 +(t + l)*exp(-pow(t + l,2f)/c/c));
}
fn wave3(v:vec3f) -> vec3f
{
    var l = v+vec3f(v==vec3f(0))*.000001f;
    var t   = 8f*custom.a;   //how expanded is the wave
    var wid = 4f*custom.b;     //wave width
    var c   = 1f/wid;
    return 1f/l*(-(t - l)*exp(-pow(t - l,vec3f(2))/c/c)
                 +(t + l)*exp(-pow(t + l,vec3f(2))/c/c));
}
fn erf(x:f32) -> f32
{
    //return sign(x) * sqrt(1f - exp(-1.239192f * x * x));
    return sign(x) * sqrt(1f - exp2(-1.787776f * x * x)); // likely faster version by @spalmer
}
//convert 1D 3D coordinates
fn c13(a: u32) -> vec3u{return vec3u(a%rez, (a/rez)%rez, a/(rez*rez));}
fn c31(a: vec3u) -> u32{return a.x + a.y*rez + a.z*rez*rez;}
fn mod1(a: f32 , b: f32) -> f32 {return fract(a/b)*b;}
fn mod3(a: vec3f, b: f32) -> vec3f{return fract(a/b)*b;}
fn loadLineD(p: float3, xyz: float3)
{
    var m = mod3(p+xyz, f32(rez));
    var r = u32(p.x) + u32(m.y)*rez + u32(m.z)*rez*rez;
    var w = u32(p.x);
    workgroupBarrier();
    D2[w] = D[r];
    workgroupBarrier();
}
fn loadLineC(p: vec3f, xyz: vec3f)
{
    var m = mod3(p+xyz, f32(rez));
    var r = u32(p.x) + u32(m.y)*rez + u32(m.z)*rez*rez;
    var w = u32(p.x);
    workgroupBarrier();
    C2[w] = C[r];
    workgroupBarrier();
}
#workgroup_count cnv 1 rez rez
@compute @workgroup_size(rez,1,1)
fn cnv(@builtin(global_invocation_id) id3: vec3u)
{
    var id = c31(id3);
    var d  = D[id];        //voxel of simulationCube
    var p  = vec3f(id3)+.5f;
    var v  = 0f;
    var et = 0f;
    for(var z=-rZ; z<=rZ; z+=1f){
    for(var y=-rZ; y<=rZ; y+=1f){  loadLineD(p, vec3f(0,y,z));
    for(var x=-rZ; x<=rZ; x+=1f){
        var xyz  = vec3f(x,y,z);
        var l2   = dot(xyz,xyz);
        if(l2<.1f || l2>rZ*rZ+.1f){continue;}
        var i2 = u32(mod1(p.x+x , f32(rez)));
        var v0 = D2[i2];   //neighbor vector
        var l  = sqrt(l2);
        var xyzl = xyz/l;
        v0 = v0-d;
        var e = wave(l);  et += abs(e);
        v += e*length(v0-xyzl*dot(v0,xyzl));//rotations
    }}}
    C[id] = v/et;
}
#workgroup_count fun 1 rez rez
@compute @workgroup_size(rez,1,1)
fn fun(@builtin(global_invocation_id) id3: vec3u)
{
    var id = c31(id3);
    var d  = D[id];        //voxel of simulationCube
    var p  = vec3f(id3)+.5f;
    var v  = vec3f(0);
    var et = 0f;
    for(var z=-rZ; z<=rZ; z+=1f){
    for(var y=-rZ; y<=rZ; y+=1f){  loadLineC(p, vec3f(0,y,z));
    for(var x=-rZ; x<=rZ; x+=1f){
        var xyz  = vec3f(x,y,z);
        var l2   = dot(xyz,xyz);
        if(l2<.1f || l2>rZ*rZ+.1f){continue;}
        var i2 = u32(mod1(p.x+x , f32(rez)));
        var c  = C2[i2];
        var l  = sqrt(l2);
        var xyzl = xyz/l;
        var e = wave(l);  et += abs(e);
        v += e*xyzl*c;
    }}}
    var u = (p/f32(rez)-.5f)*f32(time.frame)*.01f;
    v = v/et*custom.c*33f +
        u/exp(dot(u,u))*3f;

    D[id] += (v-D[id])*.4f;
}
@compute @workgroup_size(8,8,1)
fn main_image(@builtin(global_invocation_id) id: vec3u)
{
    var screen_size = textureDimensions(screen);
    if(id.x >= screen_size.x || id.y >= screen_size.y){ return; }
    var fragCoord   = float2(id.xy) + .5f;
    var iResolution = float2(screen_size);
    var clk = f32(mouse.click);
    var iTime = time.elapsed;

    var u = (2f*fragCoord        -iResolution)/iResolution.y;
    var m = (2f*float2(mouse.pos)-iResolution)/iResolution.y*2f;
    var camPos = (1f-clk)*vec3f(cos(sin(iTime*.01f)*3.f),
                                    cos(iTime*.23f)*.3f,
                                sin(cos(iTime*.07f)*3.f))
                 + clk*vec3f(cos(m.x),m.y,sin(m.x));
        camPos = normalize(camPos)*3f;
    var camDir = -normalize(camPos);
    
    var mtx0 = normalize(vec3f(camDir.z,0.f,-camDir.x));
    var mtx = mat3x3<f32>(mtx0, cross(camDir,mtx0), camDir);
    var ray = mtx*normalize(vec3f(u,2.f));  //direction of ray from camera
    var ray2= 1.f/ray;
    var ray3= step(vec3f(0),ray)*2.f-1.f;
    
    var x3 = f32(rez);
    var x3d= 2.f/x3;
    var x4 = rez*2u;  //max voxels the ray will transverse inside simulationCube
    
        var inc  = step(camPos,vec3f( 1))*     //inside cube
                      step(vec3f(-1),camPos);
        var tMin = (vec3f(-1)-camPos)*ray2;
        var tMax = (vec3f( 1)-camPos)*ray2;    ray2 = abs(ray2);
        var t1 = min(tMin, tMax);
        var t2 = max(tMin, tMax);
        var tN = max(max(t1.x, t1.y), t1.z);    //length of ray between camera and simulationCube
        var tF = min(min(t2.x, t2.y), t2.z);
            tF = f32(tF>tN);
              
    var p = camPos+ray*tN*(1.f-inc.x*inc.y*inc.z)*1.001f;   //collision position of ray on simulationCube 
        p = (p*.5f+.5f)*x3;     //transform to voxels coordinates, range 0 to x3
    var lig = vec3f(1);        //light created by camera going to voxels
    var rif = vec3f(0);        //reflected light by voxels going to camera
    for(var i=0u; i<x4; i++)    //ray will transverse simulationCube's voxels
    {
        var o = abs(p*x3d-1f);
        if(o.x>=1f||o.y>=1f||o.z>=1f){break;}            //ray got out of simulationCube
        var g = (1f-fract(p*ray3))*ray2;
        var l = min(min(g.x,g.y),g.z);                      //length to transverse one voxel
        var r = c31(vec3u(p));
        var t = D[r];       //voxel value at "r"
        var t2 = dot(t,t);
        var n = lig*max(1f-t2*l,0f);                       //light after some energy absorved by voxel                          
        rif += (lig-n)*(t+.5f)*t2;    //light emited by voxel
        lig  = n;
        p += ray*l*1.001f;                                  //make ray transverse one voxel
    }
    textureStore(screen, int2(id.xy), vec4f(rif*tF,1));
}
