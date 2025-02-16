#define PI 3.1415926535897932384626f
#define thrds 32u       //GPU threads per wrap
#define rez 512u         //simulationCube side length
#define wZ 3.f            //cube side length that writes to simulationCube
#define rZ 8.f            //cube side length that reads  to simulationCube
#define N 4u
#storage D array<atomic<u32>,N*rez*rez*2u>;
struct pO
{
    v: float2, //velocity
    m: float2, //mass
};
//convert 1D 3D coordinates
fn c12(a: uint) -> uint2{return uint2(a%rez, a/rez);}
fn c21(a: uint2) -> uint{return a.x + a.y*rez;}
//read write to D
fn rD1(r: uint) -> float
{
    var u = atomicLoad(&D[r]);
    return bitcast<f32>(u);
}
fn rD(r: uint) -> pO
{
    var r2 = r*N;
    var v1 = atomicLoad(&D[r2+0u]);
    var v2 = atomicLoad(&D[r2+1u]);
    var v3 = atomicLoad(&D[r2+2u]);
    var v4 = atomicLoad(&D[r2+3u]);
    return pO(float2(bitcast<f32>(v1),
                     bitcast<f32>(v2)),
              float2(bitcast<f32>(v3),
                     bitcast<f32>(v4)));
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
fn mod2(a: float2, b: float) -> float2
{
    return fract(a/b)*b;
}
fn cmul(a: float2, b: float2) -> float2
{
    return vec2( a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x );
}
fn cdiv(a: float2, b: float2) -> float2
{
    return vec2( dot(a,b), a.y*b.x - a.x*b.y ) / dot(b,b);
}
#workgroup_count clearD 16 512 1
@compute @workgroup_size(thrds,1,1)
fn clearD(@builtin(global_invocation_id) id3: uint3)
{
    var fw = ((time.frame+0u) & 1u)*rez*rez;
    var fr = ((time.frame+1u) & 1u)*rez*rez;
    var id = id3.x + id3.y*rez;
    var w = (fw + id)*N;
    for(var i=0u; i<N; i++){atomicStore(&D[w], 0u);  w++;}

    var d = rD(fr+id);
    var l = length(d.m); if(l!=0.f){l = 1.f/l;}
    d.v *= l;

    w = (fr + id)*N;
    wD1(w+0u, d.v.x);
    wD1(w+1u, d.v.y);
}
#workgroup_count fun 16 512 1
@compute @workgroup_size(thrds,1,1)
fn fun(@builtin(global_invocation_id) id3: uint3)
{
    var fw = ((time.frame+0u) & 1u)*rez*rez;
    var fr = ((time.frame+1u) & 1u)*rez*rez;
    var id = id3.x + id3.y*rez;
    var d  = rD(fr+id);        //voxel of simulationCube
    var p  = float2(id3.xy)+.5f;
    //velocity created this frame
    var v  = float2(0);
    for(var y=-rZ; y<=rZ; y+=1.f){
    for(var x=-rZ; x<=rZ; x+=1.f){
        var xy = float2(x,y);
        var l  = dot(xy,xy);
        if(l<.1f || l>rZ*rZ+.1f){continue;}
        var i2 = uint2(mod2(p+xy,float(rez)));
        var d2 = rD(fr+c21(i2));   //neighbor
        var l2 = sqrt(l);
        var wv = cos(l2*custom.A*5.f-float2(0,PI)*.5f);
        v += xy/l2*dot(cmul(d.m-d2.m,wv),float2(1,0))/exp(l*.08f);        
    }}  d.v += v*.2f;
    //reset simulationCube's voxels values
    if(time.frame==0u)          
    {
        d.v = float2(0);
        d.m = float2(0);
        var u = p*(1.f/float(rez))-.5f;
        var t = 8.f;
        for(var a=0.f; a<t; a+=1.f)
        {
            var g = cos(a/t*PI*2.f-float2(0,PI)*.5f);
            var m = (u+g*.3f)*11.f;
            d.m += g/exp(dot(m,m));
        }
    }
    //mouse
    {
        var res = float2(textureDimensions(screen));
        var m   = float2(mouse.pos);
            m   = (2.f*m-res.xy)/res.y;
            m   = fract(m*.5f+.5f)-float2(id3.xy)/float(rez);
            m  *= 22.f;
        d.v -= m*.5f/exp(dot(m,m))*float(mouse.click!=0);
    }
    //translate voxels data based on velocity
    var d2 = d.v+.5f;
    var p1 = -(fract(d2)-.5f);
        p += floor(d2);
    var s  = 0.f;
    for(var y=-wZ; y<=wZ; y+=1.f){
    for(var x=-wZ; x<=wZ; x+=1.f){
        var xy = float2(x,y);
        var l  = dot(xy,xy);
        if(l>wZ*wZ+.1f){continue;}
        xy += p1;
        s += 1.f/exp(dot(xy,xy));
    }}  s  = 1.f/s;
    for(var y=-wZ; y<=wZ; y+=1.f){
    for(var x=-wZ; x<=wZ; x+=1.f){
        var xy = float2(x,y);
        var l  = dot(xy,xy);
        if(l>wZ*wZ+.1f){continue;}
        xy += p1;
        var w = c21(uint2(mod2(p+xy,float(rez))));
            w = (w+fw)*N;
        var n = s/exp(dot(xy,xy));
        var m = d.m*n;
        var v = d.v*length(d.m)*n;
        atomicAddFloat(w+0u, v.x);
        atomicAddFloat(w+1u, v.y);
        atomicAddFloat(w+2u, m.x);
        atomicAddFloat(w+3u, m.y);
    }}
}
@compute @workgroup_size(thrds,1,1)
fn main_image(@builtin(global_invocation_id) id3: uint3)
{
    var screen_size = textureDimensions(screen);
    if (id3.x >= screen_size.x) { return; }
    var res = float2(screen_size);
    var u   = float2(id3.xy)+ .5f;
        u   = (2.f*u-res.xy)/res.y;
        u   = fract(u*.5f+.5f);
    var u2  = uint2(u*float(rez));

    var fw = ((time.frame+0u) & 1u)*rez*rez;
    var fr = ((time.frame+1u) & 1u)*rez*rez;
    var u1 = u2.x + u2.y*rez;
    var c  = float4(0,rD(fr+u1).m,0)+.5f;
    textureStore(screen, id3.xy, c);
}