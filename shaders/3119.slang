//previous try still got slow as wave expanded https://compute.toys/view/3116
//and accuracy of path was bad
//this try is simpler and fixes every problem there was
//just expand the wave and clamp the wave

//findind the path of least effor on a 2Dheightmap
//use slider to see 2Deffortmap or 2Dheightmap
//going across mountain slows the wave down
//starting location is where wave starts expanding
//final    location is the white point
//click to change final location

import std;
static const float  pi   = 3.1415926535897932384626433832795;
static const int    thrd = 32;     //gpu threads
static const int    mapz = 512;
static const int    frmz = 2;               //simulation frames
static const float2 strt = float2( .4, .4); //location start
static const float2 ende = float2( .3,-.4); //location end
static const float  camz = .6f;//camera zoom
[StorageBuffer(mapz*mapz     )] RWStructuredBuffer<float > hmap;//height map
[StorageBuffer(mapz*mapz*frmz)] RWStructuredBuffer<float > leem;//wave   map
[StorageBuffer(mapz*mapz     )] RWStructuredBuffer<float2> noor;//normal map
[StorageBuffer(mapz*mapz     )] RWStructuredBuffer<float > leep;//least effort paths
[StorageBuffer(mapz          )] RWStructuredBuffer<float2> papo;//paths painters
void drawLine(inout float r, float2 u, float2 p1, float2 p2, float w)
{
    var a = normalize(p2-p1);
    u = u-(p1+p2)*.5f;
    u = float2(dot(a,u),dot(a.yx*float2(-1,1),u));
    var l = distance(p1,p2)*.5f;
    u.x = max(abs(u.x)-l,0.f);
    u /= w;
    r = max(1.f-length(u),r);
}
float terrainHeight(float2 u)
{
    var r = 0.f;
    var a = .1f;
    drawLine(r,u,float2(-.2,.1),float2(.8,.1),a);
    drawLine(r,u,float2(.0,.1),float2(.0,.5),a);
    drawLine(r,u,float2(.1,-.7),float2(.3,.1),a);
    drawLine(r,u,float2(.5,-.6),float2(.5,-.4),a);
    drawLine(r,u,float2(.5,-.3),float2(.5,.1),a);
    return r;
}
float angle(float2 a, float2 b)
{
    var al = length(a);   if(al==0.f){return 0.f;}
    var bl = length(b);   if(bl==0.f){return 0.f;}
    a /= al;
    b /= bl;
    var s = a.x * b.y - a.y * b.x;
    var c   = dot(a, b);
    return atan2(s, c);
}
float2 mod2(float2 a, float b)
{
    return fract(a/b)*b;
}
[DispatchOnce]
[shader("compute")]
[WorkgroupCount(mapz/thrd, mapz, 1)]
[numthreads(thrd, 1, 1)]
void ini(uint3 dtid : SV_DispatchThreadID)
{
    var i1 = dot(dtid,uint3(1,mapz,1));
    var i2 = float2(dtid.xy)+.5f;
    var uv = i2/float(mapz) *2.f-1.f;
    hmap[i1] = terrainHeight(uv);
    var j2  = (strt+1.f)*.5f*float(mapz);
    var j1  = dot(int2(j2),int2(1,mapz));
    var tim = int(time.frame);
    var fr1 = ((tim-0+frmz) % frmz)*mapz*mapz;
    var fr2 = ((tim-1+frmz) % frmz)*mapz*mapz;
    //var v   = 2.0f*float(j1==i1);
    var v   = 1.f/exp(dot(uv-strt,uv-strt)*11111.f);
    leem[fr1+i1] = v;
    leem[fr2+i1] = v;
    leep[    i1] = 0.f;
    noor[    i1] = float2(0);
}
[shader("compute")]
[WorkgroupCount(mapz/thrd, mapz, 1)]
[numthreads(thrd, 1, 1)]
void grow(uint3 dtid : SV_DispatchThreadID)
{
    var tim = int(time.frame);
    var fr1 = ((tim-0+frmz) % frmz)*mapz*mapz;
    var fr2 = ((tim-1+frmz) % frmz)*mapz*mapz;
    var i2 = float2(dtid.xy)+.5f;
    var uv = i2/float(mapz) *2.f-1.f;
    var i1 = dot(int2(i2),int2(1,mapz));
    var h0 = hmap[i1];
    var l0 = leem[i1];
    var s1  = 0.f;
    var et  = 0.f;
    var cz  = 6;
    for(var y = -cz; y <= cz; y++){
    for(var x = -cz; x <= cz; x++){
        var xy = float2(x,y);
        var l2 = dot(xy,xy);
        if(l2>cz*cz+.1f){continue;}
        var j2 = mod2(i2+xy,float(mapz));
            j2 = clamp(i2+xy,float2(.5),float2(mapz)-.5f);
        var j1 = dot(int2(j2),int2(1,mapz));
        var h1 = hmap[j1]*4.f+.8f;  h1*=h1;
        var v1 = leem[fr2+j1];
        var e  = 1.f/exp(l2*h1);  et += e;
        s1 += v1*e;
    }}
    leem[fr1+i1] = min(1.f, s1/et*2.f);
}
[shader("compute")]
[WorkgroupCount(mapz/thrd, mapz, 1)]
[numthreads(thrd, 1, 1)]
void norf(uint3 dtid : SV_DispatchThreadID)
{
    var tim = int(time.frame);
    var fr1 = ((tim-0+frmz) % frmz)*mapz*mapz;
    var fr2 = ((tim-1+frmz) % frmz)*mapz*mapz;
    var i2  = float2(dtid.xy)+.5f;
    var i1  = dot(int2(i2),int2(1,mapz));
    var v0  = leem[fr1+i1];
    var s1  = float2(0);
    var et  = 0.f;
    var cz  = 6;
    for(var y = -cz; y <= cz; y++){
    for(var x = -cz; x <= cz; x++){
        var xy = float2(x,y);
        var l2 = dot(xy,xy);
        if(l2<.1f || l2>cz*cz+.1f){continue;}
        var j2 = mod2(i2+xy,float(mapz));
            j2 = clamp(i2+xy,float2(.5),float2(mapz)-.5f);
        var j1 = dot(int2(j2),int2(1,mapz));
        var v1 = leem[fr1+j1]-v0;
        var e  = 1.f/exp(l2);  et += e;
        s1 += v1*normalize(xy)*e;
    }}
    noor[i1] += 1.f*s1/et;
}
[shader("compute")]
[WorkgroupCount(1, 1, 1)]
[numthreads(1, 1, 1)]
void fall(uint3 dtid : SV_DispatchThreadID)
{
    var tim = int(time.frame);
    var fr1 = ((tim-0+frmz) % frmz)*mapz*mapz;
    var fr2 = ((tim-1+frmz) % frmz)*mapz*mapz;
    var strt2 = (strt+1.f)*.5f*float(mapz);
    var ende2 = (ende+1.f)*.5f*float(mapz);
    var i2 = papo[0];
    var uv = i2/float(mapz) *2.f-1.f;
    var i1 = dot(int2(i2),int2(1,mapz));
    var dd = noor[i1]*4.f;
    i2 = clamp(i2+dd,float2(.5),float2(mapz)-.5f);
    if(time.frame==0u        ){i2 = ende2;}
    if(length(i2-strt2) < 1.f){i2 = ende2;}
    {
        uint width, height;
        screen.GetDimensions(width, height);
        var scr = float2(width, height);
        uv = mouse.pos + .5f;
        uv = camz*(2.f*uv-scr.xy)/scr.y;
        uv = fract(uv+float2(0,.5))*float(mapz);
        if(mouse.click != 0){i2 = uv;}
    }
    papo[0] = i2;
    i1 = dot(int2(i2),int2(1,mapz));
    leep[i1] = 1.f;
}
[shader("compute")]
[numthreads(16, 16, 1)]
void main_image(uint3 dtid : SV_DispatchThreadID)
{
    uint width, height;
    screen.GetDimensions(width, height);
    if (dtid.x >= width || dtid.y >= height){return;}
    var scc = float2(dtid.xy);
    var scr = float2(width, height);
    var u   = scc + .5f;
        u   = camz*(2.f*u-scr.xy)/scr.y;
    var uv  = fract(u+float2(0,.5))*float(mapz);
    var i1  = dot(int2(uv),int2(1,mapz));
    var tim = int(time.frame);
    var fr1 = ((tim-0+frmz) % frmz)*mapz*mapz;
    var fr2 = ((tim-1+frmz) % frmz)*mapz*mapz;
    var l   = leem[fr1+i1]*1.0f;
    var n   = noor[    i1]*.5f+.5f;
    var c   = float4(0);
    //var c   = toLinear(turbo(fract(l)));
    var q   = int(u.x < 0.f);
    if(q==0){c = float4(l);}
    if(q==1){c = float4(0,n,0) + leep[i1] + hmap[i1]*.5f;}
    screen[dtid.xy] = c;
}