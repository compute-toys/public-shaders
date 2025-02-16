#define rZ 128u     //simulationCube side length
#define cZ 1.f      //convolution side length
#define bZ 4u       //how many history bits, must be 2^
#storage D array<u32,rZ*rZ*rZ*bZ/32u*2u>;
fn hash(a: uint) -> uint
{
    var x = a;
    x ^= x >> 17;  x *= 0xed5ad4bbu;
    x ^= x >> 11;  x *= 0xac4c1b51u;
    x ^= x >> 15;  x *= 0x31848babu;
    x ^= x >> 14;  return x;
}
fn rD(r1: float3, f: uint) -> uint
{
    var u1 = uint3(r1);
    var u  = u1.x +
             u1.y *rZ +
             u1.z *rZ*rZ;
    var v = D[u*bZ/32u + rZ*rZ*rZ*bZ/32u*f];
    return v >> (u%(32u/bZ)*bZ);
}
fn mod3(a: float3, b: float) -> float3
{
    return fract(a/b)*b;
}
#workgroup_count fun 4096 1 1  //rZ*rZ*rZ*bZ/32u threads
@compute @workgroup_size(64,1,1)
fn fun(@builtin(global_invocation_id) id3: uint3)
{
    var fr = (time.frame + 0u) & 1u;
    var fw = (time.frame + 1u) & 1u;
    var id = id3.x;
    var d = D[id + rZ*rZ*rZ*bZ/32u*fr];
    var o = 0u;
    var i1 = (id+0u) * 32u/bZ;
    var i2 = (id+1u) * 32u/bZ;
    var mv = 0u;
    var hs = hash(id + time.frame * rZ*rZ*rZ*bZ/32u + 4784u);
    for(var i=i1; i<i2; i++)
    {
        var pu = uint3(i % rZ,
                       i / rZ % rZ,
                       i /(rZ * rZ));
        var p = float3(pu)+.5f;
            p = mod3(p,float(rZ));
        var s = 0u;
        for(var z=-cZ; z<=cZ; z+=1.f){
        for(var y=-cZ; y<=cZ; y+=1.f){
        for(var x=-cZ; x<=cZ; x+=1.f){
            var r = mod3(p+float3(x,y,z),float(rZ));
            var e = rD(r,fr);
            s += ((e>>0) & 1u) + 
                 ((e>>1) & 1u);
        }}}
        var xZ = 8u;
        var yZ = 8u;
        var a = 5u + 2u*xZ + 
                xZ*yZ *(7u + 7u*xZ) +
                xZ*yZ * xZ*yZ *((1u<<4)-1u);  a = a*2u+1u;
        //if(s>=32U){a = 1430607221U;}
        var msk = (~0u)>>(32u-bZ);
        var v = (d<<1u) | ((a>>(s&31u))&1u);
            v = v & msk;
        o = o | (v << mv);
        d = d >> bZ;
        
        var u = p/float(rZ)-.5f;
        if(dot(u,u)>.2f){hs = hs & (~(msk<<mv));}

        mv+=bZ;
    }
    if(time.frame==0u){o = hs;}
    D[id + rZ*rZ*rZ*bZ/32u*fw] = o;
}
@compute @workgroup_size(8,8,1)
fn main_image(@builtin(global_invocation_id) id: uint3)
{
    //if((time.frame%6u)!=0u){return;}
    var screen_size = textureDimensions(screen);
    if(id.x >= screen_size.x || id.y >= screen_size.y){ return; }
    var fragCoord   = float2(id.xy) + .5f;
    var iResolution = float2(screen_size);
    var clk = float(mouse.click);
    var iTime = time.elapsed;
    var fr = (time.frame + 0u) & 1u;
    var fw = (time.frame + 1u) & 1u;

    var u = (2.f*fragCoord        -iResolution)/iResolution.y;
    var m = (2.f*float2(mouse.pos)-iResolution)/iResolution.y;
    var camPos = (1.f-clk)*float3(cos(sin(iTime*.11f)*3.f),cos(iTime*.7f)*.3f,sin(cos(iTime*.17f)*3.f))*3.f
                 + clk*float3(cos(m.x),m.y,sin(m.x))*3.f;
    var camDir = -normalize(camPos);
    
    var mtx0 = normalize(float3(camDir.z,0.f,-camDir.x));
    var mtx = mat3x3<f32>(mtx0, cross(camDir,mtx0), camDir);
    var ray = mtx*normalize(float3(u,2.f));  //direction of ray from camera
    var ray2= 1.f/ray;
    var ray3= step(float3(0),ray)*2.f-1.f;
    
    var x3 = float(rZ);
    var x3d= 2.f/x3;
    var x4 = rZ*2u;  //max voxels the ray will transverse inside simulationCube
    
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

    var q = 3u;
    var k = (1u<<q)-1u;
    var i = time.frame % q;

    for(var z=0u; z<x4; z++)
    {
        var b = rD(p,fw);
            b = b & k;
            b = (b>>i) | (b<<(q-i));
            b = b & k;
        var g = (1.f-fract(p*ray3))*ray2;
        var l = min(min(g.x,g.y),g.z);          
        var n = lig*max(1.f-l*custom.A*float(b!=0u&&b!=4u),0.f);  
        rif += (lig-n)*float4(uint4(0u,
                                    (b>>1u)&1u,
                                    (b>>0u)&1u,0u)); 
        lig  = n;
        p += ray*l*1.001f;                                     
        var o = abs(p*x3d-1.f);
        if(o.x>=1.f||o.y>=1.f||o.z>=1.f){break;}     
    }
    textureStore(screen, int2(id.xy), rif*tF);
}
