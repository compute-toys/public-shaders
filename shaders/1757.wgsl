
#define ZS 160      //size simulation space
#define ZT 2        //size simulation time
#define ZC 2f       //size convolution
#define ZG 1f       //size gaussian
#define PI 3.141592653589793f
#storage D array<vec3f,ZS*ZS*ZS*ZT>;
#storage C array<  f32,ZS*ZS*ZS*2>;
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
fn gaus(a: u32) -> f32//uses 2 seeds from "a"
{
    var r1 = 1f-rnd(a+0u);
    var r2 =    rnd(a+1u);
    return sqrt(-2f*log(r1))*cos(2f*PI*r2);
}
//convert 1D 3D coordinates
fn mod1(a: f32  , b: f32) -> f32  {return fract(a/b)*b;}
fn mod3(a: vec3f, b: f32) -> vec3f{return fract(a/b)*b;}
#workgroup_count cnv1 1 ZS ZS
@compute @workgroup_size(ZS,1,1)
fn cnv1(@builtin(global_invocation_id) id3: vec3u)
{
    var tim = i32(time.frame)+ZT;
    var fr1 = ((tim-0) % ZT)*ZS*ZS*ZS;
    var fr2 = ((tim-1) % ZT)*ZS*ZS*ZS;
    var id1 = dot(vec3i(id3),vec3i(1,ZS,ZS*ZS));
    var do2 = D[id1+fr2];
    var p  = vec3f(id3)+.5f;
    var v1 = 0f;
    var v3 = vec3f(0);
    var t3 = vec3f(0);
    for(var z=-ZC; z<=ZC; z+=1f){
    for(var y=-ZC; y<=ZC; y+=1f){
    for(var x=-ZC; x<=ZC; x+=1f){
        var xyz  = vec3f(x,y,z);
        var l2   = dot(xyz,xyz);
        if(l2<.1f || l2>ZC*ZC+.1f){continue;}
        var r3 = mod3(p+xyz,f32(ZS));
        var r1 = dot(vec3i(r3),vec3i(1,ZS,ZS*ZS));
        var d2 = D[r1+fr2]-do2;
        var l  = sqrt(l2);
        var e  = 1f/exp(l2*ZG);
        var xyzl = xyz/l;
        var div1 = dot(d2,xyzl);
        var div3 = div1*xyzl;
        var crl3 = d2 - div3;
        var crl1 = length(crl3);
        v1 += div1*e;//*custom.a;
        //v1 += crl1*e;//*custom.b;
        //v3 += div3*e;//*custom.a;
        //v3 += crl3*e;//*custom.b;
        t3 += cross(xyzl,crl3)*e;
    }}}
    C[id1+ZS*ZS*ZS*0] = abs(mix(v1,length(t3),custom.a));
    //C[id1+ZS*ZS*ZS*0] = length(v3);
}
#workgroup_count cnv2 1 ZS ZS
@compute @workgroup_size(ZS,1,1)
fn cnv2(@builtin(global_invocation_id) id3: vec3u)
{
    var tim = i32(time.frame)+ZT;
    var fr1 = ((tim-0) % ZT)*ZS*ZS*ZS;
    var fr2 = ((tim-1) % ZT)*ZS*ZS*ZS;
    var id1 = dot(vec3i(id3),vec3i(1,ZS,ZS*ZS));
    var do2 = D[id1+fr2];
    var p  = vec3f(id3)+.5f;
    var v  = 0f;
    for(var z=-ZC; z<=ZC; z+=1f){
    for(var y=-ZC; y<=ZC; y+=1f){
    for(var x=-ZC; x<=ZC; x+=1f){
        var xyz  = vec3f(x,y,z);
        var l2   = dot(xyz,xyz);
        if(l2>ZC*ZC+.1f){continue;}
        var r3 = mod3(p+xyz,f32(ZS));
        var r1 = dot(vec3i(r3),vec3i(1,ZS,ZS*ZS));
        var c2 = C[r1+ZS*ZS*ZS*0];
        var e  = 1f/exp(l2*ZG);
        v  += c2*e;
    }}}
    if(v!=0f){v = 1f/v;}
    C[id1+ZS*ZS*ZS*1] = v;
}
#workgroup_count cnv3 1 ZS ZS
@compute @workgroup_size(ZS,1,1)
fn cnv3(@builtin(global_invocation_id) id3: vec3u)
{
    var tim = i32(time.frame)+ZT;
    var fr1 = ((tim-0) % ZT)*ZS*ZS*ZS;
    var fr2 = ((tim-1) % ZT)*ZS*ZS*ZS;
    var id1 = dot(vec3i(id3),vec3i(1,ZS,ZS*ZS));
    var do2 = D[id1+fr2];
    var co1 = C[id1+ZS*ZS*ZS*0];
    var co2 = C[id1+ZS*ZS*ZS*1];
    var p  = vec3f(id3)+.5f;
    var v  = vec3f(0);
    for(var z=-ZC; z<=ZC; z+=1f){
    for(var y=-ZC; y<=ZC; y+=1f){
    for(var x=-ZC; x<=ZC; x+=1f){
        var xyz  = vec3f(x,y,z);
        var l2   = dot(xyz,xyz);
        if(l2>ZC*ZC+.1f){continue;}
        var r3 = mod3(p+xyz,f32(ZS));
        var r1 = dot(vec3i(r3),vec3i(1,ZS,ZS*ZS));
        var d2 = D[r1+fr2];
        var c1 = C[r1+ZS*ZS*ZS*0];
        var c2 = C[r1+ZS*ZS*ZS*1];
        var l  = sqrt(l2);
        var e  = 1f/exp(l2*ZG);
        v += d2*co1*e*c2;
    }}}
    var d = v*1f;
    if(time.frame==0u)  //reset simulation        
    {
        var r = vec3f(
            rnd(u32(id1*2+ZS*ZS*ZS*0)),
            rnd(u32(id1*2+ZS*ZS*ZS*2)),
            rnd(u32(id1*2+ZS*ZS*ZS*4)))*2f-1f;
        var u = (p/f32(ZS)-.5f)*11f;
        d = r*1f;///exp(dot(u,u));
        //d = u/exp(dot(u,u));
    }
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
            m3*= 16f;
        d += m3*.03f/exp(dot(m3,m3))*f32(mouse.click!=0);
    }
    if(keyDown(65)){d=vec3f(0);}
    D[id1+fr1] = d;
    if(time.frame==0u){D[id1+fr2] = d;}
}
@compute @workgroup_size(8,8,1)
fn main_image(@builtin(global_invocation_id) id: vec3u)
{
    var screen_size = textureDimensions(screen);
    if(id.x >= screen_size.x || id.y >= screen_size.y){ return; }
    var fragCoord   = vec2f(id.xy) + .5f;
    var iResolution = vec2f(screen_size);
    var tim = i32(time.frame)+ZT;
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
        var t = D[r]*4f;       //voxel value at "r"
        var t2 = length(t);
        var n = lig*max(1f-t2*l,0f);                       //light after some energy absorved by voxel                          
        rif += (lig-n)*(t+.5f)*t2;    //light emited by voxel
        lig  = n;
        p += ray*l*1.001f;                                  //make ray transverse one voxel
    }
    textureStore(screen, vec2i(id.xy), vec4f(rif*tF,1));
}
