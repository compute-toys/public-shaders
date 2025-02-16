#define thrds 128u       //GPU threads per wrap
#define rez 128u         //simulationCube side length
#define wZ 1.f           //cube side length that writes to simulationCube
#define rZ 3.f           //cube side length that reads  to simulationCube
#define N 4u
#storage D array<atomic<u32>,N*rez*rez*rez*2u>;
var<workgroup> D2: array<float,rez*N>;

//convert 1D 3D coordinates
fn c13(a: uint) -> uint3{return uint3(a%rez, (a/rez)%rez, a/(rez*rez));}
fn c31(a: uint3) -> uint{return a.x + a.y*rez + a.z*rez*rez;}
//read write to D
fn rD4(r: uint) -> float4
{
    var u = uint4(atomicLoad(&D[r+0u*rez*rez*rez]),
                  atomicLoad(&D[r+1u*rez*rez*rez]),
                  atomicLoad(&D[r+2u*rez*rez*rez]),
                  atomicLoad(&D[r+3u*rez*rez*rez]));
    return bitcast<vec4<f32>>(u);
}
fn wD4(w: uint, v: float4)
{
    var u = bitcast<vec4<u32>>(v);
    var w4 = w*N;
    atomicStore(&D[w4+0u*rez*rez*rez], u.x);
    atomicStore(&D[w4+1u*rez*rez*rez], u.y);
    atomicStore(&D[w4+2u*rez*rez*rez], u.z);
    atomicStore(&D[w4+3u*rez*rez*rez], u.w);
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
}
#workgroup_count fun 1 128 128
@compute @workgroup_size(thrds,1,1)
fn fun(@builtin(global_invocation_id) id3: uint3)
{
    var fw = ((time.frame+0u) & 1u)*rez*rez*rez*N;
    var fr = ((time.frame+1u) & 1u)*rez*rez*rez*N;
    var id = id3.x + id3.y*rez + id3.z*rez*rez;
    var d  = rD4(fr+id);        //voxel of simulationCube
    var p  = float3(id3)+.5f;
    //velocity created this frame
    {
        var v  = float3(0);
        var l1 = length(d.xyz);
        for(var z=-rZ; z<=rZ; z+=1.f){
        for(var y=-rZ; y<=rZ; y+=1.f){  loadLine(p, float3(0,y,z), fr);
        for(var x=-rZ; x<=rZ; x+=1.f){
            var xyz = float3(x,y,z);
            var l   = dot(xyz,xyz);
            if(l<.1f || l>rZ*rZ+.1f){continue;}
            var i2  = u32(mod1(p.x+x , f32(rez)));
            var d2  = float3(D2[i2+0u*rez],
                             D2[i2+1u*rez],
                             D2[i2+2u*rez]);  //neighbor voxel of simulationCube
            var t3 = length(d2)-l1;             //amount of velocity in neighbor voxel - my voxel
            v += xyz/sqrt(l)*t3/exp(l*.3f);     //compute velocity pointing to bigger neighbor velocity
        }}} d += float4(v * .01f,0);
    }
    if(time.frame==0u)          //reset simulationCube's voxels values
    {
        var u = (p/float(rez)-.5f)*7.f;
        d = float4(0.f,0.f,0.f,.1f)-float4(u.zxy,0)/exp(dot(u,u));
    }
    //translate voxels data based on velocity
    var d2 = d.xyz+.5f;
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
        var w3 = uint3(mod3(p+xyz,float(rez)));
        var w1 = fw+c31(w3);
        var ds = d*s/exp(dot(xyz,xyz));
        atomicAddFloat(w1+0u*rez*rez*rez, ds.x);
        atomicAddFloat(w1+1u*rez*rez*rez, ds.y);
        atomicAddFloat(w1+2u*rez*rez*rez, ds.z);
        atomicAddFloat(w1+3u*rez*rez*rez, ds.w);
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
        var l = min(min(g.x,g.y),g.z);              //length to transverse one voxel
        var t = abs(rD4(fw+c31(uint3(p))));   //voxel value at "p"
        var n = lig*max(1.f-dot(t,t)*l*.3f,0.f);    //light after some energy absorved by voxel                          
        rif += (lig-n)*t;                           //light emited by voxel
        lig  = n;
        p += ray*l*1.001f;                                 //make ray transverse one voxel
        var o = abs(p*x3d-1.f);
        if(o.x>=1.f||o.y>=1.f||o.z>=1.f){break;}    //ray got out of simulationCube
    }
    textureStore(screen, int2(id.xy), rif*tF);
}