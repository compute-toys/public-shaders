#define thrds 128u       //GPU threads per wrap
#define rez 128u        //simulationCube side length
#define wZ 1.f            //cube side length that writes to simulationCube
#define rZ 4.f            //cube side length that reads  to simulationCube

struct pO
{
    v: float3, //velocity
    m: float , //mass
};
#define N 4u
#storage D array<atomic<u32>,N*rez*rez*rez*2u>;
var<workgroup> D2: array<float,rez*N>;

//convert 1D 3D coordinates
fn c13(a: uint) -> uint3{return uint3(a%rez, (a/rez)%rez, a/(rez*rez));}
fn c31(a: uint3) -> uint{return a.x + a.y*rez + a.z*rez*rez;}
//read write to D
fn rD1(r: uint) -> float
{
    var v = atomicLoad(&D[r]);
    return bitcast<f32>(v);
}
fn rD(r: uint) -> pO
{
    var v1 = atomicLoad(&D[r+0u*rez*rez*rez]);
    var v2 = atomicLoad(&D[r+1u*rez*rez*rez]);
    var v3 = atomicLoad(&D[r+2u*rez*rez*rez]);
    var v4 = atomicLoad(&D[r+3u*rez*rez*rez]);
    return pO(float3(bitcast<f32>(v1),
                     bitcast<f32>(v2),
                     bitcast<f32>(v3)),
              bitcast<f32>(v4));
}
fn wD1(w: uint, v: float)
{
    var u = bitcast<u32>(v);
    atomicStore(&D[w], u);
}
fn atomicAddFloat(w: uint, v: f32)
{
    var old = atomicLoad(&D[w]);
    loop {
        var n = bitcast<u32>(bitcast<f32>(old) + v);
        var r = atomicCompareExchangeWeak(&D[w], old, n);
        if r.exchanged { break; }
        old = r.old_value;
    }
}
fn mod1(a: float , b: float) -> float {return fract(a/b)*b;}
fn mod3(a: float3, b: float) -> float3{return fract(a/b)*b;}
fn loadLine(p: float3, xyz: float3, fr: uint)
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
#workgroup_count clearD 1 128 128
@compute @workgroup_size(thrds,1,1)
fn clearD(@builtin(global_invocation_id) id3: uint3)
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
    if(d.m!=0.f){m = 1.f/m;}
    d.v *= m;
    w = fr + id;
    wD1(w+0u*rez*rez*rez, d.v.x);
    wD1(w+1u*rez*rez*rez, d.v.y);
    wD1(w+2u*rez*rez*rez, d.v.z);
}
#workgroup_count fun 1 128 128
@compute @workgroup_size(thrds,1,1)
fn fun(@builtin(global_invocation_id) id3: uint3)
{
    var fw = ((time.frame+0u) & 1u)*rez*rez*rez*N;
    var fr = ((time.frame+1u) & 1u)*rez*rez*rez*N;
    var id = c31(id3);
    var d  = rD(fr+id);        //voxel of simulationCube
    var p  = float3(id3)+.5f;
    //velocity created this frame
    var v = float3(0);
    for(var z=-rZ; z<=rZ; z+=1.f){
    for(var y=-rZ; y<=rZ; y+=1.f){  loadLine(p, float3(0,y,z), fr);
    for(var x=-rZ; x<=rZ; x+=1.f){
        var xyz  = float3(x,y,z);
        var l    = dot(xyz,xyz);
        if(l<.1f || l>rZ*rZ+.1f){continue;}
        var i2 = u32(mod1(p.x+x , f32(rez)));
        var m  = D2[i2+3u*rez];   //neighbor mass
        var l2 = sqrt(l);
        v += xyz/l2*(d.m-m)*cos(l2*custom.A*5.f)/exp(l*.13f);
    }}} d.v += v;
    //reset simulationCube's voxels values
    if(time.frame==0u)
    {
        var u = (p/float(rez)-.5f)*6.f;
        d.v = float3(0,0,0);
        d.m = .3f/exp(dot(u,u));
    }
    //translate voxels data based on velocity
    var d2 = d.v+.5f;
    var p1 = -(fract(d2)-.5f);
        p += floor(d2);
    var s  = 0.f;
    for(var z=-wZ; z<=wZ; z+=1.f){
    for(var y=-wZ; y<=wZ; y+=1.f){
    for(var x=-wZ; x<=wZ; x+=1.f){
        var xyz = p1+float3(x,y,z);
        s += 1.f/exp(dot(xyz,xyz));
    }}} s = 1.f/s;
    for(var z=-wZ; z<=wZ; z+=1.f){
    for(var y=-wZ; y<=wZ; y+=1.f){
    for(var x=-wZ; x<=wZ; x+=1.f){
        var xyz = p1+float3(x,y,z);
        var w = fw+c31(uint3(mod3(p+xyz,float(rez))));
        var m = d.m*s/exp(dot(xyz,xyz));
        var v = d.v*m;
        atomicAddFloat(w+0u*rez*rez*rez, v.x);
        atomicAddFloat(w+1u*rez*rez*rez, v.y);
        atomicAddFloat(w+2u*rez*rez*rez, v.z);
        atomicAddFloat(w+3u*rez*rez*rez, m);
    }}}
}
@compute @workgroup_size(8,8,1)
fn main_image(@builtin(global_invocation_id) id: uint3)
{
    var screen_size = textureDimensions(screen);
    if(id.x >= screen_size.x || id.y >= screen_size.y){ return; }
    var fragCoord   = float2(id.xy) + .5f;
    var iResolution = float2(screen_size);
    var clk = float(mouse.click);
    var iTime = time.elapsed;
    var fw = ((time.frame+0u) & 1u)*rez*rez*rez*N;

    var u = (2.f*fragCoord        -iResolution)/iResolution.y;
    var m = (2.f*float2(mouse.pos)-iResolution)/iResolution.y;
    var camPos = (1.f-clk)*float3(cos(sin(iTime*.11f)*3.f),cos(iTime*.7f)*.3f,sin(cos(iTime*.17f)*3.f))*2.f
                 + clk*float3(cos(m.x),m.y,sin(m.x))*2.f;
    var camDir = -normalize(camPos);
    
    var mtx0 = normalize(float3(camDir.z,0.f,-camDir.x));
    var mtx = mat3x3<f32>(mtx0, cross(camDir,mtx0), camDir);
    var ray = mtx*normalize(float3(u,2.f));  //direction of ray from camera
    var ray2= 1.f/ray;
    var ray3= step(float3(0),ray)*2.f-1.f;
    
    var x3 = float(rez);
    var x3d= 2.f/x3;
    var x4 = rez*2u;  //max voxels the ray will transverse inside simulationCube
    
        var inc  = step(camPos,float3( 1))*     //inside cube
                      step(float3(-1),camPos);
        var tMin = (float3(-1)-camPos)*ray2;
        var tMax = (float3( 1)-camPos)*ray2;    ray2 = abs(ray2);
        var t1 = min(tMin, tMax);
        var t2 = max(tMin, tMax);
        var tN = max(max(t1.x, t1.y), t1.z);    //length of ray between camera and simulationCube
        var tF = min(min(t2.x, t2.y), t2.z);
            tF = float(tF>tN);
              
    var p = camPos+ray*tN*(1.f-inc.x*inc.y*inc.z)*1.001f;   //collision position of ray on simulationCube 
        p = (p*.5f+.5f)*x3;     //transform to voxels coordinates, range 0 to x3
    var lig = float4(1);        //light created by camera going to voxels
    var rif = float4(0);        //reflected light by voxels going to camera
    for(var i=0u; i<x4; i++)    //ray will transverse simulationCube's voxels
    {
        var g = (1.f-fract(p*ray3))*ray2;
        var l = min(min(g.x,g.y),g.z);                      //length to transverse one voxel
        var t = rD1(fw+c31(uint3(p))+3u*rez*rez*rez);       //voxel value at "p"
        var n = lig*max(1.f-t*l,0.f);                       //light after some energy absorved by voxel                          
        rif += (lig-n)*mix(float4(1,0,0,0),float4(1),t);    //light emited by voxel
        lig  = n;
        p += ray*l*1.001f;                                  //make ray transverse one voxel
        var o = abs(p*x3d-1.f);
        if(o.x>=1.f||o.y>=1.f||o.z>=1.f){break;}            //ray got out of simulationCube
    }
    textureStore(screen, int2(id.xy), rif*tF);
}
