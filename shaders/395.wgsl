#define rez 512u         //simulationCube side length
#define wZ 3.f           //cube side length that writes to simulationCube
#define rZ 4.f           //cube side length that reads  to simulationCube
#define N 3u
#define PI 3.14159265358979f
#storage C array<float2,rez*rez>;
#storage D array<atomic<u32>,N*rez*rez*2u>;
struct pO
{
    v: float2, //velocity
    m: float , //mass
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
    return pO(float2(bitcast<f32>(v1),
                     bitcast<f32>(v2)),
              bitcast<f32>(v3));
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
#workgroup_count clearD 16 512 1
@compute @workgroup_size(32,1,1)
fn clearD(@builtin(global_invocation_id) id3: uint3)
{
    var fw = ((time.frame+0u) & 1u)*rez*rez;
    var fr = ((time.frame+1u) & 1u)*rez*rez;
    var id = id3.x + id3.y*rez;
    var w = fw*N + id;
    for(var i=0u; i<N; i++){atomicStore(&D[w], 0u);  w+=rez*rez;}

    var d = rD(fr+id);
    if(d.m!=0.f){d.m = 1.f/d.m;}
    d.v *= d.m;
    w = (fr + id)*N;
    wD1(w+0u, d.v.x);
    wD1(w+1u, d.v.y);
}
#workgroup_count comb 16 512 1
@compute @workgroup_size(32,1,1)
fn comb(@builtin(global_invocation_id) id3: uint3)
{
    var fw = ((time.frame+0u) & 1u)*rez*rez;
    var fr = ((time.frame+1u) & 1u)*rez*rez;
    var id = id3.x + id3.y*rez;
    var d  = rD(fr+id);
    var p  = float2(id3.xy)+.5f;
    var v1 = 0.f;
    var v2 = 0.f;
    for(var y=-rZ; y<=rZ; y+=1.f){
    for(var x=-rZ; x<=rZ; x+=1.f){
        var xy = float2(x,y);
        var l2 = dot(xy,xy);
        if(l2<.1f || l2>rZ*rZ+.1f){continue;}
        var i2 = uint2(mod2(p+xy,float(rez)));
        var d2 = rD(fr+c21(i2));
        var l  = sqrt(l2);
        var xyl= xy/l;
        v1 += dot(d2.v-d.v,xyl                )/exp(l2*.3f);
        v2 += dot(d2.v-d.v,xyl.yx*float2(-1,1))/exp(l2*.3f);
    }}
    C[id] = float2(v1,v2);
}
#workgroup_count fun 16 512 1
@compute @workgroup_size(32,1,1)
fn fun(@builtin(global_invocation_id) id3: uint3)
{
    var fw = ((time.frame+0u) & 1u)*rez*rez;
    var fr = ((time.frame+1u) & 1u)*rez*rez;
    var id = id3.x + id3.y*rez;
    var d  = rD(fr+id);
    var p  = float2(id3.xy)+.5f;
    //velocity created this frame
    var c1 = C[id];
    var v1 = float2(0);
    var v2 = float2(0);
    for(var y=-rZ; y<=rZ; y+=1.f){
    for(var x=-rZ; x<=rZ; x+=1.f){
        var xy = float2(x,y);
        var l2 = dot(xy,xy);
        if(l2<.1f || l2>rZ*rZ+.1f){continue;}
        var i2 = uint2(mod2(p+xy,float(rez)));
        var d2 = rD(fr+c21(i2));
        var c2 = C[c21(i2)];
        var l  = sqrt(l2);
        var xyl= xy/l;
        v1 +=                            xyl             *abs(c2.x)/exp(l2*.3f);
        v2 += (float(c2.y>=0.f)*2.f-1.f)*xyl.yx*float2(-1,1)*(c2.x)/exp(l2*.3f);
    }}
    d.v += v1*(custom.A-.5f)*.5f+
           v2*(custom.B-.5f)*.5f;
    //reset simulation
    if(time.frame==0u)          
    {
        var u = (p/float(rez)-.5f)*8.f;
        d.v = u*1.f/exp(dot(u,u));
        //d.m = .5f/exp(dot(u,u));
    }
    //mouse
    {
        var res = float2(textureDimensions(screen));
        var m   = float2(mouse.pos);
            m   = (2.f*m-res.xy)/res.y;
            m   = fract(m*.5f+.5f)-float2(id3.xy)/float(rez);
            m  *= 22.f;
        d.v -= m*.1f/exp(dot(m,m))*float(mouse.click!=0);
        //d.v -= m.yx*float2(-1,1)*.1f/exp(dot(m,m))*float(mouse.click!=0);
        //d.v -= float2(1,0)*.2f/exp(dot(m,m))*float(mouse.click!=0);
    }
    //translate data based on velocity
    var d2 = d.v+.5f;
    var p1 = -(fract(d2)-.5f);
        p += floor(d2);
    var s  = 0.f;
    for(var y=-wZ; y<=wZ; y+=1.f){
    for(var x=-wZ; x<=wZ; x+=1.f){
        var xy = float2(x,y);
        var l2 = dot(xy,xy);
        if(l2>wZ*wZ+.1f){continue;}
        xy = p1+xy;
        s += 1.f/exp(dot(xy,xy));
    }}  s  = 1.f/s;
    for(var y=-wZ; y<=wZ; y+=1.f){
    for(var x=-wZ; x<=wZ; x+=1.f){
        var xy = float2(x,y);
        var l2 = dot(xy,xy);
        if(l2>wZ*wZ+.1f){continue;}
        xy = p1+xy;
        var i2 = uint2(mod2(p+xy,float(rez)));
        var w = (fw+c21(i2))*N;
        //var m = d.m*s/exp(dot(xy,xy));
        var m = s/exp(dot(xy,xy));
        var v = d.v*m;
        atomicAddFloat(w+0u, v.x);
        atomicAddFloat(w+1u, v.y);
        atomicAddFloat(w+2u, m);
    }}
}
@compute @workgroup_size(8,8,1)
fn main_image(@builtin(global_invocation_id) id3: uint3)
{
    var screen_size = textureDimensions(screen);
    if(id3.x >= screen_size.x ||
       id3.y >= screen_size.y){return;}
    var res = float2(screen_size);
    var u   = float2(id3.xy)+ .5f;
        u   = (2.f*u-res.xy)/res.y;
        u   = fract(u*.5f+.5f);
    var u2  = uint2(u*float(rez));

    var fw = ((time.frame+0u) & 1u)*rez*rez;
    var fr = ((time.frame+1u) & 1u)*rez*rez;
    var u1 = u2.x + u2.y*rez;
    var d  = rD(fr+u1);
    //var c  = float4(0,rD(fr+u1).v*.5f+.5f,0);
    //var c  = float4(rD(fr+u1).m);
    //var c  = float4(1.f-length(rD(fr+u1).v));
    var c  = sin(d.v.x*2.f+float4(3,1,4,4)+1.f)*.25f +
             sin(d.v.y*2.f+float4(2,1,3,4)+1.f)*.25f +.5f;
    textureStore(screen, id3.xy, c);
}