//based on https://compute.toys/view/2519
//explanation there
#define TD 16       //gpu threads divisor
#define ZS 512      //size simulation space
#define ZT 4        //size simulation time
#define ZF 22        //size simulation steps per frame
#define ZC 4f       //size convolution
#define PI 3.14159265358979f
#storage D array<f32,ZS*ZS*ZT>;
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
fn mod2(a: vec2f, b: f32) -> vec2f {return fract(a/b)*b;}
#dispatch_count comb1 ZF
#workgroup_count comb1 TD ZS 1
@compute @workgroup_size(ZS/TD,1,1)
fn comb1(@builtin(global_invocation_id) id3: vec3u)
{
    var tim = (i32(time.frame)+ZT)*ZF + i32(dispatch.id);
    var fr1 = ((tim-0) % ZT)*ZS*ZS;
    var fr2 = ((tim-1) % ZT)*ZS*ZS;
    var fr3 = ((tim-2) % ZT)*ZS*ZS;
    var id1 = dot(vec3i(id3),vec3i(1,ZS,0));
    var id2 = vec2f(id3.xy)+.5f;
    var d1  = D[id1+fr2];
    var d12 = D[id1+fr3];
    //var c1  = C[id1];
    var v1  = 0f;
    var v2  = 0f;
    var et1 = 0f;
    var et2 = 0f;
    for(var y=-ZC; y<=ZC; y+=1f){
    for(var x=-ZC; x<=ZC; x+=1f){
        var xy = vec2f(x,y);
        var l2 = dot(xy,xy);
        if(l2>ZC*ZC+.1f){continue;}
        var r2 = mod2(id2+xy,f32(ZS));
        var r1 = dot(vec2i(r2),vec2i(1,ZS));
        var d2 = D[r1+fr2];
        var d3 = D[r1+fr3];
        var l  = sqrt(l2);
        var e1 = cos(l*custom.a)/exp(l2*custom.b);    et1+=abs(e1);
        var e2 = cos(l*custom.a)/exp(l2*custom.c);    et2+=abs(e2);
        v1 += (f32(d2>=0f)*2f-1f)*pow(abs(d2),1.2f)*e1;
        v2 += (f32(d3>=0f)*2f-1f)*pow(abs(d3),1.0f)*e2;
    }}
    if(et1!=0f){et1 = 1f/et1;}
    if(et2!=0f){et2 = 1f/et2;}
    var d = 1f*v1*et1 - d12;
        d = 1.8f*v1*et1 - 1f*v2*et2;
    if(time.frame==0u)  //reset simulation        
    {
        var u = id2/f32(ZS)-.5f;
        d = rnd2(u32(id1))/exp(111f*dot(u,u))
                       +1f/exp(111f*dot(u,u));
    }
    //mouse
    {
        var res = vec2f(textureDimensions(screen));
        var m   = vec2f(mouse.pos);
            m   = (2f*m-res.xy)/res.y;
            m   = fract(m*.5f+.5f)-vec2f(id3.xy)/f32(ZS);
            m  *= 66f;
        d += .3f/exp(dot(m,m))*f32(mouse.click!=0);
    }
    if(keyDown(65)){d=0f;}
    D[id1+fr1] = d;
    if(time.frame==0u){D[id1+fr2] = d;}
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
    var tim = (i32(time.frame)+ZT)*ZF;
    var fr1 = ((tim-0) % ZT)*ZS*ZS;
    var fr2 = ((tim-1) % ZT)*ZS*ZS;
    var fr3 = ((tim-2) % ZT)*ZS*ZS;
    var id1 = dot(vec2i(u),vec2i(1,ZS));
    var d1  = D[id1+fr1];
    var d2  = D[id1+fr2];
    var d3  = D[id1+fr3];
    var c  = vec4f(d1+.5f);
        c  = vec4f(4f*sqrt(d1*d1 + d2*d2));
    textureStore(screen, id3.xy, c);
}