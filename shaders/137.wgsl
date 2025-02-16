#define thrds 32u   //GPU threads per wrap
#define rez 192u    //simulationCube side length
#define Z 1.f       //cube side length that writes to simulationCube
#storage D array<float2,rez*rez*rez*2u>;
#storage N array<float2,rez*rez*rez>;
//convert 1D 3D coordinates
fn c13(a: uint) -> uint3{return uint3(a%rez, (a/rez)%rez, a/(rez*rez));}
fn c31(a: uint3) -> uint{return a.x + a.y*rez + a.z*rez*rez;}
fn mod3(a: float3, b: float) -> float3
{
    return fract(a/b)*b;
}
#workgroup_count fun1 6 192 192
@compute @workgroup_size(thrds,1,1)
fn fun1(@builtin(global_invocation_id) id: uint3)
{
    var fw = ((time.frame+0u) & 1u)*rez*rez*rez;
    var fr = ((time.frame+1u) & 1u)*rez*rez*rez;
    var w1 = id.x + id.y*rez + id.z*rez*rez;
    var p  = float3(id)+.5f;
    var s = float2(0);
    for(var z=-Z; z<=Z; z+=1.f){
    for(var y=-Z; y<=Z; y+=1.f){
    for(var x=-Z; x<=Z; x+=1.f){
        var r = uint3(mod3(p+float3(x,y,z),float(rez)));
        s += abs(D[fr+c31(r)]);
    }}}
    if(s.x!=0.f){s.x = 1.f/s.x;}
    if(s.y!=0.f){s.y = 1.f/s.y;}
    N[w1] = s;
}
#workgroup_count fun2 6 192 192
@compute @workgroup_size(thrds,1,1)
fn fun2(@builtin(global_invocation_id) id: uint3)
{
    var fw = ((time.frame+0u) & 1u)*rez*rez*rez;
    var fr = ((time.frame+1u) & 1u)*rez*rez*rez;
    var w1  = id.x + id.y*rez + id.z*rez*rez;
    var p  = float3(id)+.5f;
    var d  = D[fr+w1];        //voxel of simulationCube
    var da = abs(d);
    var t  = Z*2.f+1.f;
        t  = 1.f/(t*t*t);
    var b = 0.f;
    for(var z=-Z; z<=Z; z+=1.f){
    for(var y=-Z; y<=Z; y+=1.f){
    for(var x=-Z; x<=Z; x+=1.f){
        var r = c31(uint3(mod3(p+float3(x,y,z),float(rez))));
        var d2= D[fr+r];
        var s = N[r]*da;
            s = mix(s,float2(t),min(abs(d2),float2(1)));
        b += dot(s*d2,float2(2,-1));
    }}}
    var b2 = float2(b,d.x);
    if(time.frame==0u)          //reset simulationCube's voxels values
    {
        var u = p/float(rez)*2.f-1.f;
        b2 = float2(.01f/exp(dot(u,u)*5.f));
    }
    D[fw+w1] = b2;
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
    var fw = ((time.frame+0u) & 1u)*rez*rez*rez;

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
        var l = min(min(g.x,g.y),g.z);                  //length to transverse one voxel
        var t = D[fw+c31(uint3(p))];              //voxel value at "p"
        var t2= dot(t,float2(1,-1))*6.f;
        var t3= clamp(t2+.5f,0.f,1.f);
        var n = lig*max(1.f-abs(t2)*l*.3f,0.f);         //light after some energy absorved by voxel                          
        rif += (lig-n)*mix(float4(0,1,0,0),
                           float4(0,0,1,0),t3);         //light emited by voxel
        lig  = n;
        p += ray*l*1.001f;                                     //make ray transverse one voxel
        var o = abs(p*x3d-1.f);
        if(o.x>=1.f||o.y>=1.f||o.z>=1.f){break;}        //ray got out of simulationCube
    }
    textureStore(screen, int2(id.xy), rif*tF);
}
