//3D version of https://compute.toys/view/2519
//with smaller convolution and less steps per frame
#define ZS 192      //size simulation space
#define ZT 3        //size simulation time
#define ZF 4        //size simulation steps per frame
#define ZC 4f       //size convolution
#define PI 3.141592653589793f
#storage D array<f32,ZS*ZS*ZS*ZT>;
var<workgroup> D2: array<f32,ZS*ZT>;
fn hash(x2: u32) -> u32
{
    var x = x2;
    x ^= x >> 17;  x *= 0xed5ad4bbu;
    x ^= x >> 11;  x *= 0xac4c1b51u;
    x ^= x >> 15;  x *= 0x31848babu;
    x ^= x >> 14;  return x;
}
fn rnd(a: u32) -> f32//uses 1 seeds from "a"
{
    var h   = hash(a);
    var msk = (1u << 23u) - 1u;
    return f32(h & msk) / f32(msk);
}
//convert 1D 3D coordinates
fn mod1(a: f32  , b: f32) -> f32  {return fract(a/b)*b;}
fn mod3(a: vec3f, b: f32) -> vec3f{return fract(a/b)*b;}
fn loadLine(p: vec3f, xyz: vec3f, fr: i32)
{
    var m = mod3(p+xyz, f32(ZS));
    var r = i32(p.x) + i32(m.y)*ZS + i32(m.z)*ZS*ZS + fr;
    var w = i32(p.x);
    workgroupBarrier();
    D2[w] = D[r];
    workgroupBarrier();
}
#dispatch_once ini
#workgroup_count ini 1 ZS ZS
@compute @workgroup_size(ZS,1,1)
fn ini(@builtin(global_invocation_id) id3: vec3u)
{
    var tim = (i32(time.frame)+ZT)*ZF + i32(dispatch.id);
    var fr1 = ((tim-0) % ZT)*ZS*ZS*ZS;
    var fr2 = ((tim-1) % ZT)*ZS*ZS*ZS;
    var fr3 = ((tim-2) % ZT)*ZS*ZS*ZS;
    var id1 = dot(vec3i(id3),vec3i(1,ZS,ZS*ZS));
    var p   = vec3f(id3)+.5f;
    var u = p/f32(ZS)-.5f;
    var d = rnd(u32(id1))/exp(2222f*dot(u,u))
                      +1f/exp(333f*dot(u,u));
    D[id1+fr1] = d;
    D[id1+fr2] = d;
}
#dispatch_count comb1 ZF
#workgroup_count comb1 1 ZS ZS
@compute @workgroup_size(ZS,1,1)
fn comb1(@builtin(global_invocation_id) id3: vec3u)
{
    var tim = (i32(time.frame)+ZT)*ZF + i32(dispatch.id);
    var fr1 = ((tim-0) % ZT)*ZS*ZS*ZS;
    var fr2 = ((tim-1) % ZT)*ZS*ZS*ZS;
    var fr3 = ((tim-2) % ZT)*ZS*ZS*ZS;
    var id1 = dot(vec3i(id3),vec3i(1,ZS,ZS*ZS));
    var d1  = D[id1+fr2];
    var d12 = D[id1+fr3];
    var p   = vec3f(id3)+.5f;
    var v1  = 0f;
    var et1 = 0f;
    for(var z=-ZC; z<=ZC; z+=1f){
    for(var y=-ZC; y<=ZC; y+=1f){  loadLine(p, float3(0,y,z), fr2);
    for(var x=-ZC; x<=ZC; x+=1f){
        var xyz  = vec3f(x,y,z);
        var l2   = dot(xyz,xyz);
        if(l2>ZC*ZC+.1f){continue;}
        var r3 = mod3(p+xyz,f32(ZS));
        //var r1 = dot(vec3i(r3),vec3i(1,ZS,ZS*ZS));
        var r1 = i32(mod1(p.x+x , f32(ZS)));
        //var d2 = D[r1+fr2];
        var d2 = D2[r1];
        var l  = sqrt(l2);
        var e1 = cos(l*custom.a)/exp(l2*custom.b);    et1+=abs(e1);
        v1 += (f32(d2>=0f)*2f-1f)*pow(abs(d2),1.2f)*e1;
    }}}
    if(et1!=0f){et1 = 1f/et1;}
    var d = 1f*v1*et1 - d12;
    //mouse
    {
        var res = vec2f(textureDimensions(screen));
        var m   = vec2f(mouse.pos);
            m   = (2f*m-res.xy)/res.y;
        var camPos = cos(time.elapsed*vec3f(-23,-9,27)*.02f+vec3f(11,2,22));
        var camDir = -normalize(camPos);
        var sd = normalize(vec3f(camDir.z,0f,-camDir.x));
        var up = normalize(cross(camDir,sd));
        var m3 = fract((sd*m.x+up*m.y)*.5f+.5f)-vec3f(id3)/f32(ZS);
            m3*= 33f;
        d += .3f/exp(dot(m3,m3))*f32(mouse.click!=0);
    }
    if(keyDown(65)){d=0f;}
    D[id1+fr1] = d;
}
@compute @workgroup_size(8,8,1)
fn main_image(@builtin(global_invocation_id) id: vec3u)
{
    var screen_size = textureDimensions(screen);
    if(id.x >= screen_size.x || id.y >= screen_size.y){ return; }
    var fragCoord   = vec2f(id.xy) + .5f;
    var iResolution = vec2f(screen_size);
    var tim = (i32(time.frame)+ZT)*ZF;
    var fr1 = ((tim-0) % ZT)*ZS*ZS*ZS;
    var fr2 = ((tim-1) % ZT)*ZS*ZS*ZS;
    var u = .5f*(2f*fragCoord        -iResolution)/iResolution.y;
    var m = (2f*vec2f(mouse.pos)-iResolution)/iResolution.y*2f;
    var camPos = cos(time.elapsed*vec3f(-23,-9,27)*.02f+vec3f(11,2,22));
    //if(mouse.click!=0){camPos = vec3f(cos(m.x),m.y,sin(m.x));}
        camPos = normalize(camPos)*3f;
    var camDir = -normalize(camPos);
    var sd = normalize(vec3f(camDir.z,0f,-camDir.x));
    var up = cross(camDir,sd);
    var ray = camDir + u.x*sd + u.y*up;  //direction of ray from camera
    var ray2= 1f/ray;
    var ray3= step(vec3f(0),ray)*2f-1f;
    var x3 = f32(ZS);
    var x3d= 2f/x3;
    var x4 = ZS*2;  //max voxels the ray will transverse inside simulationCube
    
        var inc  = step(camPos,vec3f( 1))*     //inside cube
                      step(vec3f(-1),camPos);
        var tMin = (vec3f(-1)-camPos)*ray2;
        var tMax = (vec3f( 1)-camPos)*ray2;    ray2 = abs(ray2);
        var t1 = min(tMin, tMax);
        var t2 = max(tMin, tMax);
        var tN = max(max(t1.x, t1.y), t1.z);    //length of ray between camera and simulationCube
        var tF = min(min(t2.x, t2.y), t2.z);
            tF = f32(tF>tN);
              
    var p = camPos+ray*tN*(1f-inc.x*inc.y*inc.z)*1.001f;   //collision position of ray on simulationCube 
        p = (p*.5f+.5f)*x3;     //transform to voxels coordinates, range 0 to x3
    var lig = vec3f(1);        //light created by camera going to voxels
    var rif = vec3f(0);        //reflected light by voxels going to camera
    for(var i=0; i<x4; i++)    //ray will transverse simulationCube's voxels
    {
        var o = abs(p*x3d-1f);
        if(o.x>=1f||o.y>=1f||o.z>=1f){break;}            //ray got out of simulationCube
        var g = (1f-fract(p*ray3))*ray2;
        var l = min(min(g.x,g.y),g.z);                      //length to transverse one voxel
        var r = dot(vec3i(p),vec3i(1,ZS,ZS*ZS));
        var t1 = D[r+fr1];
        var t2 = D[r+fr2]-t1;
        var tl = sqrt(t1*t1 + t2*t2);
        var n = lig*max(1f-tl*l,0f);                       //light after some energy absorved by voxel                          
        rif += (lig-n)*tl;    //light emited by voxel
        lig  = n;
        p += ray*l*1.001f;                                  //make ray transverse one voxel
    }
    textureStore(screen, vec2i(id.xy), vec4f(rif*tF,1));
}
