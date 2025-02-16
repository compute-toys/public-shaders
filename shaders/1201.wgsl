#define thrds 160u       //GPU threads per wrap
#define rez 160u        //simulationCube side length
#define wZ 1f            //cube side length that writes to simulationCube
#define rZ 4f            //cube side length that reads  to simulationCube
#define PI 3.141592653589793f

struct pO
{
    v: vec3f, //velocity
    m: f32 , //mass
};
#define N 4u
#storage D array<atomic<u32>,N*rez*rez*rez*2u>;
#storage C array<f32,rez*rez*rez>;
var<workgroup> D2: array<f32,rez*N>;
var<workgroup> C2: array<f32,rez>;

fn wave(l:f32) -> f32
{
    var t   = 8f*custom.a;   //how expanded is the wave
    var wid = 4f*custom.b;     //wave width
    var c   = 1f/wid;
    return 1f/l*(-(t - l)*exp(-pow(t - l,2f)/c/c)
                 +(t + l)*exp(-pow(t + l,2f)/c/c));
}
fn erf(x:f32) -> f32
{
    //return sign(x) * sqrt(1f - exp(-1.239192f * x * x));
    return sign(x) * sqrt(1f - exp2(-1.787776f * x * x)); // likely faster version by @spalmer
}
//convert 1D 3D coordinates
fn c13(a: u32) -> vec3u{return vec3u(a%rez, (a/rez)%rez, a/(rez*rez));}
fn c31(a: vec3u) -> u32{return a.x + a.y*rez + a.z*rez*rez;}
//read write to D
fn rD1(r: u32) -> f32
{
    var v = atomicLoad(&D[r]);
    return bitcast<f32>(v);
}
fn rD(r: u32) -> pO
{
    var v1 = atomicLoad(&D[r+0u*rez*rez*rez]);
    var v2 = atomicLoad(&D[r+1u*rez*rez*rez]);
    var v3 = atomicLoad(&D[r+2u*rez*rez*rez]);
    var v4 = atomicLoad(&D[r+3u*rez*rez*rez]);
    return pO(vec3f(bitcast<f32>(v1),
                    bitcast<f32>(v2),
                    bitcast<f32>(v3)),
              bitcast<f32>(v4));
}
fn wD1(w: u32, v: f32)
{
    var u = bitcast<u32>(v);
    atomicStore(&D[w], u);
}
fn atomicAddFloat(w: u32, v: f32)
{
    var old = atomicLoad(&D[w]);
    loop {
        var n = bitcast<u32>(bitcast<f32>(old) + v);
        var r = atomicCompareExchangeWeak(&D[w], old, n);
        if r.exchanged { break; }
        old = r.old_value;
    }
}
fn mod1(a: f32 , b: f32) -> f32 {return fract(a/b)*b;}
fn mod3(a: vec3f, b: f32) -> vec3f{return fract(a/b)*b;}
fn loadLineD(p: float3, xyz: float3, fr: uint)
{
    var m = mod3(p+xyz, f32(rez));
    var r = u32(p.x) + u32(m.y)*rez + u32(m.z)*rez*rez + fr;
    var w = u32(p.x);
    workgroupBarrier();
    for(var n=0u; n<N; n++)
    {
        var v = atomicLoad(&D[r]);
        D2[w] = bitcast<f32>(v);
        r+=rez*rez*rez;
        w+=rez;
    }
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
#workgroup_count clearD 1 rez rez
@compute @workgroup_size(thrds,1,1)
fn clearD(@builtin(global_invocation_id) id3: vec3u)
{
    var fw = ((time.frame+0u) & 1u)*rez*rez*rez*N;
    var fr = ((time.frame+1u) & 1u)*rez*rez*rez*N;
    var w = fw + id3.x + id3.y*rez + id3.z*rez*rez;
    for(var i=0u; i<N; i++)
    {
        atomicStore(&D[w], 0u);  w+=rez*rez*rez;
    }
    var id = c31(id3);
    var d = rD(fr+id);
    var m = d.m;
    if(d.m!=0.f){m = 1f/m;}
    //if(d.m==0f){d.m = 1f;}  d.m = 1f/d.m;
    d.v *= m;
    w = fr + id;
    wD1(w+0u*rez*rez*rez, d.v.x);
    wD1(w+1u*rez*rez*rez, d.v.y);
    wD1(w+2u*rez*rez*rez, d.v.z);
}
#workgroup_count cnv 1 rez rez
@compute @workgroup_size(thrds,1,1)
fn cnv(@builtin(global_invocation_id) id3: vec3u)
{
    var fw = ((time.frame+0u) & 1u)*rez*rez*rez*N;
    var fr = ((time.frame+1u) & 1u)*rez*rez*rez*N;
    var id = c31(id3);
    var d  = rD(fr+id);        //voxel of simulationCube
    var p  = vec3f(id3)+.5f;
    //velocity created this frame
    var v  = 0f;
    var e2 = 0f;
    for(var z=-rZ; z<=rZ; z+=1f){
    for(var y=-rZ; y<=rZ; y+=1f){  loadLineD(p, vec3f(0,y,z), fr);
    for(var x=-rZ; x<=rZ; x+=1f){
        var xyz  = vec3f(x,y,z);
        var l2   = dot(xyz,xyz);
        if(l2<.1f || l2>rZ*rZ+.1f){continue;}
        var i2 = u32(mod1(p.x+x , f32(rez)));
        var v2 = vec3f(D2[i2+0u*rez],   //neighbor velocity
                       D2[i2+1u*rez],
                       D2[i2+2u*rez]);
        var m2 = D2[i2+3u*rez];         //neighbor mass
        var l = sqrt(l2);
        var xyzl = xyz/l;
        v2 = v2-d.v;

        //var e = wave(l);  e2 += abs(e);
        //var u = xyz;
        //u = vec3(atan2(u.y,u.x),
        //         atan2(u.z,length(u.xy)),
        //         log(length(u)));    u = u+vec3f(1,1,0)*PI*.5f;
        //u = vec3(cos(u.y)*cos(u.x),
        //         cos(u.y)*sin(u.x),
        //         sin(u.y))*exp(u.z);
        //v += abs(e*dot(v2,u/l));

        //var e = wave(l);  //e2 += abs(e);
        var k1=vec2f(-y,x);  var h1=length(k1);
        var k2=vec2f(-z,y);  var h2=length(k2);
        var k3=vec2f(-z,x);  var h3=length(k3);
        var n1=wave(h1)/exp(z*z*.5f);  if(h1==0f){n1=0f;}  e2+=abs(n1);
        var n2=wave(h2)/exp(x*x*.5f);  if(h2==0f){n2=0f;}  e2+=abs(n2);
        var n3=wave(h3)/exp(y*y*.5f);  if(h3==0f){n3=0f;}  e2+=abs(n3);
        if(h1!=0f){h1=1f/h1;}
        if(h2!=0f){h2=1f/h2;}
        if(h3!=0f){h3=1f/h3;}
        v += abs(n1*dot(v2.xy,k1*h1))+
             abs(n2*dot(v2.yz,k2*h2))+
             abs(n3*dot(v2.xz,k3*h3)); //rotation
    }}}   e2 = 1f/e2;
    C[id] = v*e2;
}
#workgroup_count fun 1 rez rez
@compute @workgroup_size(thrds,1,1)
fn fun(@builtin(global_invocation_id) id3: vec3u)
{
    var fw = ((time.frame+0u) & 1u)*rez*rez*rez*N;
    var fr = ((time.frame+1u) & 1u)*rez*rez*rez*N;
    var id = c31(id3);
    var d  = rD(fr+id);        //voxel of simulationCube
    var p  = vec3f(id3)+.5f;
    //velocity created this frame
    var v  = vec3f(0);
    var e2 = 0f;
    for(var z=-rZ; z<=rZ; z+=1f){
    for(var y=-rZ; y<=rZ; y+=1f){  loadLineC(p, vec3f(0,y,z));
    for(var x=-rZ; x<=rZ; x+=1f){
        var xyz  = vec3f(x,y,z);
        var l2   = dot(xyz,xyz);
        if(l2<.1f || l2>rZ*rZ+.1f){continue;}
        var i2 = u32(mod1(p.x+x , f32(rez)));
        var c2 = C2[i2];
        var l  = sqrt(l2);
        var xyzl = xyz/l;
        var e = 1f/exp(l2*.3f);  e2 += e;
        v += xyzl*c2*e;
    }}}  e2 = 1f/e2;
    d.v += v*e2*custom.c*11f;
    //reset simulationCube's voxels values
    if(time.frame==0u)
    {
        var u = (p/f32(rez)-.5f)*8f;
        d.m = 1f;//.3f/exp(dot(u,u));
        d.v = -u/exp(dot(u,u));
    }
    //translate voxels data based on velocity
    var d2 = d.v+.5f;
    var p1 = -(fract(d2)-.5f);
        p += floor(d2);
    var s  = 0f;
    for(var z=-wZ; z<=wZ; z+=1f){
    for(var y=-wZ; y<=wZ; y+=1f){
    for(var x=-wZ; x<=wZ; x+=1f){
        var xyz = p1+vec3f(x,y,z);
        s += 1f/exp(dot(xyz,xyz));
    }}} s = 1f/s;
    for(var z=-wZ; z<=wZ; z+=1f){
    for(var y=-wZ; y<=wZ; y+=1f){
    for(var x=-wZ; x<=wZ; x+=1f){
        var xyz = p1+vec3f(x,y,z);
        var w = fw+c31(vec3u(mod3(p+xyz,f32(rez))));
        var m = d.m*s/exp(dot(xyz,xyz));
            m =     s/exp(dot(xyz,xyz));
        var v = d.v*m;
        atomicAddFloat(w+0u*rez*rez*rez, v.x);
        atomicAddFloat(w+1u*rez*rez*rez, v.y);
        atomicAddFloat(w+2u*rez*rez*rez, v.z);
        atomicAddFloat(w+3u*rez*rez*rez, m);
    }}}
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
    var fw = ((time.frame+0u) & 1u)*rez*rez*rez*N;

    var u = (2.f*fragCoord        -iResolution)/iResolution.y;
    var m = (2.f*float2(mouse.pos)-iResolution)/iResolution.y;
    var camPos = (1.f-clk)*vec3f(cos(sin(iTime*.01f)*3.f),
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
        var r = fw+c31(vec3u(p));
        var t = vec3f(rD1(r+0u*rez*rez*rez),
                      rD1(r+1u*rez*rez*rez),
                      rD1(r+2u*rez*rez*rez));       //voxel value at "r"
        var t2 = dot(t,t);
        var n = lig*max(1f-t2*l*.3f,0f);                       //light after some energy absorved by voxel                          
        rif += (lig-n)*(t+.5f)*t2;    //light emited by voxel
        lig  = n;
        p += ray*l*1.001f;                                  //make ray transverse one voxel
    }
    textureStore(screen, int2(id.xy), vec4f(rif*tF*2f,1));
}
