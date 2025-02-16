#define rez 512u         //simulationCube side length
#define wZ 3.f           //cube side length that writes to simulationCube
#define rZ 9.f           //cube side length that reads  to simulationCube
#define N 3u
#define PI 3.14159265358979f
#storage D array<float2,rez*rez>;
#storage D2 array<float2,rez*rez>;
//convert 1D 3D coordinates
fn c12(a: uint) -> uint2{return uint2(a%rez, a/rez);}
fn c21(a: uint2) -> uint{return a.x + a.y*rez;}
fn hash(x2: uint) -> uint
{
    var x = x2;
    x ^= x >> 17;  x *= 0xed5ad4bbu;
    x ^= x >> 11;  x *= 0xac4c1b51u;
    x ^= x >> 15;  x *= 0x31848babu;
    x ^= x >> 14;  return x;
}
fn rnd(a: uint) -> float
{
    var h   = hash(a);
    var msk = (1u << 23u) - 1u;
    return float(h & msk) / float(msk);
}
fn mod2(a: float2, b: float) -> float2
{
    return fract(a/b)*b;
}
//fill 2Dvectors with noise
#workgroup_count fill 16 512 1
@compute @workgroup_size(32,1,1)
fn fill(@builtin(global_invocation_id) id3: uint3)
{
    var id = c21(id3.xy);
    var an = rnd(id+rez*rez*0u);
    var si = rnd(id+rez*rez*1u);
    var rt = rnd(id+rez*rez*2u);
        rt = (rt*2.f-1.f)*time.elapsed*2.f;
    D[id]  = si*cos(an-float2(0,.5)*PI+rt);
}
//get how much divergence there is in 2Dvectors
#workgroup_count comb1 16 512 1
@compute @workgroup_size(32,1,1)
fn comb1(@builtin(global_invocation_id) id3: uint3)
{
    var id = c21(id3.xy);
    var d  = D[id];
    var p  = float2(id3.xy)+.5f;
    var v  = float2(0);
    var e2 = 0.f;
    for(var y=-rZ; y<=rZ; y+=1.f){
    for(var x=-rZ; x<=rZ; x+=1.f){
        var xy = float2(x,y);
        var l2 = dot(xy,xy);
        if(l2<.1f || l2>rZ*rZ+.1f){continue;}
        var i2 = uint2(mod2(p+xy,float(rez)));
        var d2 = D[c21(i2)];
        var l  = sqrt(l2);
        var e  = 1.f/exp(l2*.0f);  e2+=e;
        var f  = cos((l-rZ*.5f)*2.f       *custom.A -float2(0,.5)*PI); //divergence
      //var f  = cos(atan2(x,-y)*floor(8.f*custom.A)-float2(0,.5)*PI); //rotation
        v += e*dot(d2-d,xy          /l)*f; //divergence*frequency
      //v += e*dot(d2-d,float2(-y,x)/l)*f; //rotation  *frequency
    }}
    D2[id] = v/e2;
}
//get 2Dvectors after filtered by divergence
#workgroup_count comb2 16 512 1
@compute @workgroup_size(32,1,1)
fn comb2(@builtin(global_invocation_id) id3: uint3)
{
    var id = c21(id3.xy);
    var d  = D[id];
    var p  = float2(id3.xy)+.5f;
    var v  = float2(0);
    var e2 = 0.f;
    for(var y=-rZ; y<=rZ; y+=1.f){
    for(var x=-rZ; x<=rZ; x+=1.f){
        var xy = float2(x,y);
        var l2 = dot(xy,xy);
        if(l2<.1f || l2>rZ*rZ+.1f){continue;}
        var i2 = uint2(mod2(p+xy,float(rez)));
        var d2 = D2[c21(i2)];
        var l  = sqrt(l2);
        var e  = 1.f/exp(l2*.0f);  e2+=e;
        var f  = d2*cos((l-rZ*.5f)*2.f       *custom.A -float2(0,.5)*PI);
      //var f  = d2*cos(atan2(-x,y)*floor(8.f*custom.A)-float2(0,.5)*PI);
        v += -xy         /l*e*(f.x+f.y);
      //v += float2(-y,x)/l*e*(f.x+f.y);
    }}
    D[id] = v/e2;
}
@compute @workgroup_size(8,8,1)
fn main_image(@builtin(global_invocation_id) id3: uint3)
{
    var screen_size = textureDimensions(screen);
    if(id3.x >= screen_size.x){return;}
    if(id3.y >= screen_size.y){return;}
    var cz = float2(screen_size);
    var u   = float2(id3.xy)+ .5f;
        u   = (2.f*u-cz.xy)/cz.y;
        u   = fract(u*.5f+.5f);
    var u2  = uint2(u*float(rez));
    var d  = D[c21(u2)];
    //var c  = float4(0,d,0);
    var c  = float4(length(d));
    //var c  = float4(atan(d.y,d.x));
    textureStore(screen, id3.xy, c*33.f);
}