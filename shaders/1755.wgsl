#define TD 16       //gpu threads divisor
#define ZS 512      //size simulation space
#define ZT 3        //size simulation time
#define ZC 3f       //size convolution
#define ZG 1f      //size gaussian
#define PI 3.14159265358979f
#storage C array<vec2f,ZS*ZS*2>;
#storage D array<vec2f,ZS*ZS*ZT>;
fn hash(a: u32) -> u32
{
    var x = a;
    x ^= x >> 17;  x *= 0xed5ad4bbu;
    x ^= x >> 11;  x *= 0xac4c1b51u;
    x ^= x >> 15;  x *= 0x31848babu;
    x ^= x >> 14;  return x;
}
fn rnd2(a: u32) -> f32//uses 1 seeds from "a"
{
    var h   = hash(a);
    var msk = (1u << 23u) - 1u;
    return f32(h & msk) / f32(1u << 23u);
}
fn gaus(a: u32) -> f32//uses 2 seeds from "a"
{
    var r1 = 1f-rnd2(a+0u);
    var r2 =    rnd2(a+1u);
    return sqrt(-2f*log(r1))*cos(2f*PI*r2);
}
fn mod2(a: vec2f, b: f32) -> vec2f
{
    return fract(a/b)*b;
}
#workgroup_count combC TD ZS 1
@compute @workgroup_size(ZS/TD,1,1)
fn combC(@builtin(global_invocation_id) id3: vec3u)
{
    var tim = i32(time.frame)+ZT;
    var fr1 = ((tim-0) % ZT)*ZS*ZS;
    var fr2 = ((tim-1) % ZT)*ZS*ZS;
    var fr3 = ((tim-2) % ZT)*ZS*ZS;
    var id1 = dot(vec3i(id3),vec3i(1,ZS,0));
    var id2 = vec2f(id3.xy)+.5f;
    var do2 = D[id1+fr2];
    var v   = vec2f(0);
    var et  = 0f;
    for(var y=-ZC; y<=ZC; y+=1f){
    for(var x=-ZC; x<=ZC; x+=1f){
        var xy = vec2f(x,y);
        var l2 = dot(xy,xy);
        if(l2<.1f||l2>ZC*ZC+.1f){continue;}
        var r2 = mod2(id2+xy,f32(ZS));
        var r1 = dot(vec2i(r2),vec2i(1,ZS));
        var d2 = D[r1+fr2]-do2;
        var l  = sqrt(l2);
        var e  = 1f/exp(l2*ZG);    et+=e;
        v += dot(d2,xy/l               )*e*(1f-custom.a);
        v += dot(d2,xy.yx/l*vec2f(-1,1))*e*(   custom.a);
    }}
    C[id1+ZS*ZS*0] = v/et;
}
#workgroup_count comb1 TD ZS 1
@compute @workgroup_size(ZS/TD,1,1)
fn comb1(@builtin(global_invocation_id) id3: vec3u)
{
    var tim = i32(time.frame)+ZT;
    var fr1 = ((tim-0) % ZT)*ZS*ZS;
    var fr2 = ((tim-1) % ZT)*ZS*ZS;
    var fr3 = ((tim-2) % ZT)*ZS*ZS;
    var id1 = dot(vec3i(id3),vec3i(1,ZS,0));
    var id2 = vec2f(id3.xy)+.5f;
    var do2 = D[id1+fr2];
    var v   = vec2f(0);
    var et  = 0f;
    for(var y=-ZC; y<=ZC; y+=1f){
    for(var x=-ZC; x<=ZC; x+=1f){
        var xy = vec2f(x,y);
        var l2 = dot(xy,xy);
        if(l2>ZC*ZC+.1f){continue;}
        var r2 = mod2(id2+xy,f32(ZS));
        var r1 = dot(vec2i(r2),vec2i(1,ZS));
        var d2 = D[r1+fr2];
        var c2 = C[r1+ZS*ZS*0];
        var l  = sqrt(l2);
        var e  = 1f/exp(l2*ZG);
        et += length(c2)*e;
        //v += dot(p2-p1,xy                 )*e;
        //v += dot(p2-p1,xy.yx/l*vec2f(-1,1))*e;
    }}
    if(et!=0f){et = 1f/et;}
    C[id1+ZS*ZS*1] = vec2f(et);
}
#workgroup_count comb2 TD ZS 1
@compute @workgroup_size(ZS/TD,1,1)
fn comb2(@builtin(global_invocation_id) id3: vec3u)
{
    var tim = i32(time.frame)+ZT;
    var fr1 = ((tim-0) % ZT)*ZS*ZS;
    var fr2 = ((tim-1) % ZT)*ZS*ZS;
    var fr3 = ((tim-2) % ZT)*ZS*ZS;
    var id1 = dot(vec3i(id3),vec3i(1,ZS,0));
    var id2 = vec2f(id3.xy)+.5f;
    var do2 = D[id1+fr2];
    var do3 = D[id1+fr3];
    var co1 = C[id1+ZS*ZS*0];
    var co2 = C[id1+ZS*ZS*1];
    var v   = vec2f(0);
    for(var y=-ZC; y<=ZC; y+=1f){
    for(var x=-ZC; x<=ZC; x+=1f){
        var xy = vec2f(x,y);
        var l2 = dot(xy,xy);
        if(l2>ZC*ZC+.1f){continue;}
        var r2 = mod2(id2+xy,f32(ZS));
        var r1 = dot(vec2i(r2),vec2i(1,ZS));
        var d2 = D[r1+fr2];
        var c1 = C[r1+ZS*ZS*0];
        var c2 = C[r1+ZS*ZS*1];
        var l  = sqrt(l2);
        var e  = 1f/exp(l2*ZG);
        v += d2*length(co1)*c2*e;
    }}
    var d = v*1f;//-do3;
    if(time.frame==0u)  //reset simulation        
    {
        var r = vec2f(
            rnd2(u32(id1+ZS*ZS*0)),
            rnd2(u32(id1+ZS*ZS*1)))*2f-1f;
        var u = (id2/f32(ZS)-.5f)*22f;
        d = r*1f;///exp(dot(u,u));
    }
    //mouse
    {
        var res = vec2f(textureDimensions(screen));
        var m   = vec2f(mouse.pos);
            m   = (2f*m-res.xy)/res.y;
            m   = fract(m*.5f+.5f)-vec2f(id3.xy)/f32(ZS);
            m  *= 33f;
        d += m*.03f/exp(dot(m,m))*f32(mouse.click!=0);
    }
    if(keyDown(65)){d=vec2f(0);}
    D[id1+fr1] = d;
    if(time.frame==0u){D[id1+fr2] = d;}
    if(time.frame==0u){D[id1+fr3] = d;}
}
@compute @workgroup_size(8,8,1)
fn main_image(@builtin(global_invocation_id) id3: vec3u)
{
    var screen_size = textureDimensions(screen);
    if(id3.x >= screen_size.x){return;}
    if(id3.y >= screen_size.y){return;}
    var res = vec2f(screen_size);
    var u   = vec2f(id3.xy)+ .5f;
        u   = (2f*u-res.xy)/res.y;
        u   = fract(u*.5f+.5f)*f32(ZS);
    var tim = i32(time.frame)+ZT;
    var fr1 = ((tim-0) % ZT)*ZS*ZS;
    var fr2 = ((tim-1) % ZT)*ZS*ZS;
    var fr3 = ((tim-2) % ZT)*ZS*ZS;
    var id1 = dot(vec2i(u),vec2i(1,ZS));
    var do1 = D[id1+fr1];
    var do2 = D[id1+fr2];
    //var c  = vec4f(0,rD(fr+u1).v*.5f+.5f,0);
    var c  = vec4f(0,do1*111f*custom.b+.5,0);
    //var c  = vec4f(1f-length(rD(fr+u1).v));
    //var c  = sin(d.v.x*2f+vec4f(3,1,4,4)+1f)*.25f +
    //         sin(d.v.y*2f+vec4f(2,1,3,4)+1f)*.25f +.5f;
    textureStore(screen, id3.xy, c);
}