//based on the idea that a simulation with nonlinear springs
//creates particles so i took it and remixed with my previous convolutions
//that idea is the simplest,efficient,accurate way to simulate particles
//the space is made of 1 float per voxel 
//but previous frame is used to compute change between frames
//better persepective, space is made of 1 complexNumber per voxel = 2 floats
#define TD 16       //gpu threads divisor
#define ZS 512      //size simulation space
#define ZT 4        //size simulation time
#define ZF 16        //size simulation steps per frame
#define ZC 6f       //size convolution
#define PI 3.14159265358979f
#storage D array<f32  ,ZS*ZS*ZT>;
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
    var v1  = 0f;
    var et1 = 0f;
    for(var y=-ZC; y<=ZC; y+=1f){
    for(var x=-ZC; x<=ZC; x+=1f){
        var xy = vec2f(x,y);
        var l2 = dot(xy,xy);
        if(l2>ZC*ZC+.1f){continue;}
        var r2 = fract((id2+xy)/f32(ZS))*f32(ZS);
        var r1 = dot(vec2i(r2),vec2i(1,ZS));
        var d2 = D[r1+fr2];
        var l  = sqrt(l2);
        var e1 = cos(l*custom.a)/exp(l2*custom.b);    et1+=abs(e1);
        v1 += d2*min(pow(abs(d2),.1f),1f)*e1;
    }}
    if(et1!=0f){et1 = 1f/et1;}
    var d = 1f*v1*et1 - d12;
    if(time.frame==0u)  //reset simulation        
    {
        var u = (id2/f32(ZS)-.5f)*16f;
        d = 1f/exp(dot(u,u));
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
    var id1 = dot(vec2i(u),vec2i(1,ZS));
    var d1  = D[id1+fr1];
    var d2  = D[id1+fr2];
    var c  = vec4f(d1+.5f);
        c  = vec4f(sqrt(d1*d1 + d2*d2));
    textureStore(screen, id3.xy, c);
}