#define thrds 32u       //GPU threads per wrap
#define rez 512u         //simulationCube side length
#define wZ 3            //cube side length that writes to simulationCube
#define rZ 8            //cube side length that reads  to simulationCube

#storage D array<atomic<u32>,3u*rez*rez*2u>;

//convert 1D 3D coordinates
fn c12(a: uint) -> uint2{return uint2(a%rez, a/rez);}
fn c21(a: uint2) -> uint{return a.x + a.y*rez;}
//read write to D
fn rD1(r: uint) -> float
{
    var u = atomicLoad(&D[r]);
    return bitcast<f32>(u);
}
fn rD3(r: uint) -> float3
{
    var r3 = r*3u;
    var u = uint3(atomicLoad(&D[r3+0u]),
                  atomicLoad(&D[r3+1u]),
                  atomicLoad(&D[r3+2u]));
    return bitcast<vec3<f32>>(u);
}
fn wD3(w: uint, v: float3)
{
    var u = bitcast<vec3<u32>>(v);
    var w3 = w*3u;
    atomicStore(&D[w3+0u], u.x);
    atomicStore(&D[w3+1u], u.y);
    atomicStore(&D[w3+2u], u.z);
}
fn wD2(w: uint, v: float3)
{
    var u = bitcast<vec3<u32>>(v);
    let w3 = w*3u;
    atomicStore(&D[w3+0u], u.x);
    atomicStore(&D[w3+1u], u.y);
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
fn aD3(w: uint, v: float3)
{
    var w3 = w*3u;
    atomicAddFloat(w3+0u, v.x);
    atomicAddFloat(w3+1u, v.y);
    atomicAddFloat(w3+2u, v.z);
}
#workgroup_count clearD 16 512 1
@compute @workgroup_size(thrds,1,1)
fn clearD(@builtin(global_invocation_id) id3: uint3)
{
    var fw = ((time.frame+0u) & 1u)*rez*rez;
    var fr = ((time.frame+1u) & 1u)*rez*rez;
    var id = id3.x + id3.y*rez;
    var w = (fw + id)*3u;
    atomicStore(&D[w+0u], 0u);
    atomicStore(&D[w+1u], 0u);
    atomicStore(&D[w+2u], 0u);

    var d = rD3(fr+id);
    var e = d.z;  if(e!=0.f){e = 1.f/e;}
    d *= float3(e,e,1);
    wD2(fr+id, d);
}
#workgroup_count fun 16 512 1
@compute @workgroup_size(thrds,1,1)
fn fun(@builtin(global_invocation_id) id3: uint3)
{
    var fw = ((time.frame+0u) & 1u)*rez*rez;
    var fr = ((time.frame+1u) & 1u)*rez*rez;
    var id = id3.x + id3.y*rez;
    var d  = rD3(fr+id);        //voxel of simulationCube
    var p  = id3.xy;//c12(id);
    //velocity created this frame
    {
        var v  = float2(0);
        for(var x=-rZ; x<=rZ; x++){
        for(var y=-rZ; y<=rZ; y++){
            var xyi = int2(x,y);
            var xy  = float2(xyi);
            var l   = dot(xy,xy);
            if(l<.1f || l>rZ*rZ+.1f){continue;}
            var i2 = (p+uint2(xyi)+rez) % rez;
            var d2 = rD1((fr+c21(i2))*3u+2u);   //neighbor voxel of simulationCube
            var l2 = sqrt(l); 
            var f  = xy/l2*(d.z-d2)/exp(l*.08f);
            l2 = l2*custom.A*5.f;
            v += f*cos(l2);
            v += f*sin(l2);
        }}  d -= float3(v*.2f,0);
    }
    var p2 = float2(p)*(1.f/float(rez))-.5f;
    if(time.frame==0u)          //reset simulationCube's voxels values
    {
        var u1 = (p2+float2(.25,0))*8.f;
        var u2 = (p2-float2(.25,0))*8.f;
        d =  float3(0,0,.5)/exp(dot(u1,u1))
            -float3(0,0,.5)/exp(dot(u2,u2));
    }
    //mouse
    {
        var res = float2(textureDimensions(screen));
        var m   = float2(mouse.pos);
            m   = (2.f*m-res.xy)/res.y;
            m   = fract(m*.5f+.5f)-float2(id3.xy)/float(rez);
            m  *= 22.f;
        d -= float3(m*.2,0)/exp(dot(m,m))*float(mouse.click!=0);
    }
    //walls
    {
        var p3 = p2-float2(0,.3);
        d +=  float(p2.x<-.46f)*float3(1,0,0)
             +float(p2.y<-.46f)*float3(0,1,0)
             -float(p2.x> .46f)*float3(1,0,0)
             -float(p2.y> .46f)*float3(0,1,0)
             +float(p2.x< .04f && p2.x>.0f && p2.y<.3f)*float3(1,0,0)
             -float(p2.x>-.04f && p2.x<.0f && p2.y<.3f)*float3(1,0,0)
             +float(length(p3)<.04f)*float3(normalize(p3),0);
    }
    //translate voxels data based on velocity
    var d2 = d.xy+.5f;
    var p1 = -(fract(d2)-.5f);
        p += uint2(int2(floor(d2)));
    var s  = 0.f;
    for(var x=-wZ; x<=wZ; x++){
    for(var y=-wZ; y<=wZ; y++){
        var xyi = int2(x,y);
        var xy  = float2(xyi);
        var l   = dot(xy,xy);
        if(l>wZ*wZ+.1f){continue;}
        xy = p1+xy;
        s += 1.f/exp(dot(xy,xy));
    }}  s  = 1.f/s;
    for(var x=-wZ; x<=wZ; x++){
    for(var y=-wZ; y<=wZ; y++){
        var xyi = int2(x,y);
        var xy  = float2(xyi);
        var l   = dot(xy,xy);
        if(l>wZ*wZ+.1f){continue;}
        xy = p1+xy;
        var w = c21((p+uint2(xyi)+rez) % rez);
        var m = d.z*s/exp(dot(xy,xy));
        aD3(fw+w, d*float3(m,m,0)+float3(0,0,m));
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
    var v  = rD1((fr+u1)*3u+2u);
    var c  = float4(sqrt(abs(v))*(float(v>=0.f)*2.f-1.f)+.5f);
    textureStore(screen, id3.xy, c);
}