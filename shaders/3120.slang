//previous try was not smooth https://compute.toys/view/2076
//i thought when dropin a marble in bottom-left map
//it will roll down to the least effort path
//but it do not, it creates 2 twin paths?
//that other path is very unexpected

//findind the path of least effor on a 2Dheightmap
//going across mountain slows the wave down
//click top left  to change strt location
//click top right to change ende location
//click bottom left to drop a marble
//bottom left  is the sum of the two top screen maps
//bottom right is the sum of the two top screen maps length

import std;
static const float  pi   = 3.1415926535897932384626433832795;
static const int    thrd = 32;     //gpu threads
static const int    mapz = 256;
static const int    frmz = 2;               //simulation frames
static const int    wrln = 2;               //worlds to sim at same time
[StorageBuffer(2                  )] RWStructuredBuffer<float2> lctn;//start-end locations
[StorageBuffer(mapz*mapz          )] RWStructuredBuffer<float > hmap;//height map
[StorageBuffer(mapz*mapz*frmz*wrln)] RWStructuredBuffer<float > leem;//wave   map
[StorageBuffer(mapz*mapz     *wrln)] RWStructuredBuffer<float2> noor;//normal map
[StorageBuffer(mapz*mapz          )] RWStructuredBuffer<float > leep;//least effort paths
[StorageBuffer(mapz               )] RWStructuredBuffer<float2> papo;//paths painters
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
[WorkgroupCount(1, 1, 1)]
[numthreads(1, 1, 1)]
void inil(uint3 dtid : SV_DispatchThreadID)
{
    lctn[0] = float2( .4, .4);
    lctn[1] = float2( .3,-.4);
}
[DispatchOnce]
[shader("compute")]
[WorkgroupCount(mapz/thrd, mapz, 1)]
[numthreads(thrd, 1, 1)]
void ini(uint3 dtid : SV_DispatchThreadID)
{
    var id1 = dot(dtid,uint3(1,mapz,1));
    var id2 = float2(dtid.xy)+.5f;
    var uv = id2/float(mapz) *2.f-1.f;
    hmap[id1] = terrainHeight(uv);
    var tim = int(time.frame);
    var fr1 = ((tim-0+frmz) % frmz)*mapz*mapz;
    var fr2 = ((tim-1+frmz) % frmz)*mapz*mapz;
    for(var i=0; i<wrln; i++)
    {
        var u = uv-lctn[i];
        var v = 1.f/exp(dot(u,u)*11111.f);
        leem[fr1+id1 + i*mapz*mapz*frmz] = v;
        leem[fr2+id1 + i*mapz*mapz*frmz] = v;
        noor[    id1 + i*mapz*mapz     ] = float2(0);
    }
    leep[id1] = 0.f;
}
[shader("compute")]
[WorkgroupCount(mapz/thrd, mapz, 1)]
[numthreads(thrd, 1, 1)]
void grow(uint3 dtid : SV_DispatchThreadID)
{
    var tim = int(time.frame);
    var fr1 = ((tim-0+frmz) % frmz)*mapz*mapz;
    var fr2 = ((tim-1+frmz) % frmz)*mapz*mapz;
    var id2 = float2(dtid.xy)+.5f;
    var uv  = id2/float(mapz) *2.f-1.f;
    var id1 = dot(int2(id2),int2(1,mapz));
    var h0  = hmap[id1];
    var l0  = leem[id1];
    float s1[] = {0.f,0.f};
    var et  = 0.f;
    var cz  = 6;
    for(var y = -cz; y <= cz; y++){
    for(var x = -cz; x <= cz; x++){
        var xy = float2(x,y);
        var l2 = dot(xy,xy);
        if(l2>cz*cz+.1f){continue;}
        var j2 = mod2(id2+xy,float(mapz));
            j2 = clamp(id2+xy,float2(.5),float2(mapz)-.5f);
        var j1 = dot(int2(j2),int2(1,mapz));
        var h1 = hmap[j1]*4.f+.8f;  h1*=h1;
        var e  = 1.f/exp(l2*h1);  et += e;
        for(var i=0; i<wrln; i++)
        {
            s1[i] += leem[fr2+j1 + i*mapz*mapz*frmz]*e;
        }
    }}
    for(var i=0; i<wrln; i++)
    {
        leem[fr1+id1 + i*mapz*mapz*frmz] = min(1.f, s1[i]/et*2.f);
    }
    if(mouse.click != 0)
    {
        uint width, height;
        screen.GetDimensions(width, height);
        var scr = float2(width, height);
        var um  = float2(mouse.pos) + .5f;
            um  = (2.f*um-scr.xy)/scr.y;
        var i   = 0;
        if(um.x<0.f && um.y<0.f){i = 0;}
        if(um.x>0.f && um.y<0.f){i = 1;}
        var u = fract(um)*2.f-1.f - uv;
        if(mouse.click != 0 && um.y<0.f)
        {
            leem[fr1+id1 + i*mapz*mapz*frmz] = 1.f/exp(dot(u,u)*11111.f);
            noor[    id1 + i*mapz*mapz] = float2(0);
        }
    }
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
    float v0[] = {0.f,0.f};
    for(var i=0; i<wrln; i++)
    {
        v0[i] = leem[fr1+i1 + i*mapz*mapz*frmz];
    }
    float2 s1[] = {float2(0),float2(0)};
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
        var e  = 1.f/exp(l2);  et += e;
        for(var i=0; i<wrln; i++)
        {
            var v1 = leem[fr1+j1 + i*mapz*mapz*frmz]-v0[i];
            s1[i] += v1*normalize(xy)*e;
        }
    }}
    for(var i=0; i<wrln; i++)
    {
        noor[i1 + i*mapz*mapz] += 1.f*s1[i]/et;
    }
}
[shader("compute")]
[WorkgroupCount(1, 1, 1)]
[numthreads(1, 1, 1)]
void fall(uint3 dtid : SV_DispatchThreadID)
{
    var tim = int(time.frame);
    var fr1 = ((tim-0+frmz) % frmz)*mapz*mapz;
    var fr2 = ((tim-1+frmz) % frmz)*mapz*mapz;
    var i2 = papo[0];
    var uv = i2/float(mapz) *2.f-1.f;
    var i1 = dot(int2(i2),int2(1,mapz));
    var dd = float2(0);
    for(var i=0; i<wrln; i++)
    {
        dd += noor[i1 + i*mapz*mapz]*2.f;
    }
    var ddl = length(dd);
    if(ddl!=0.f){ddl = 1.f/ddl;}
    //dd *= ddl;
    i2 = clamp(i2+dd,float2(.5),float2(mapz)-.5f);
    if(time.frame==0u){i2 = float2(mapz*.5f);}
    {
        uint width, height;
        screen.GetDimensions(width, height);
        var scr = float2(width, height);
        var um = mouse.pos + .5f;
            um = (2.f*um-scr.xy)/scr.y;
        var u = fract(um)*float(mapz);
        if(mouse.click != 0 && um.y>=0.f){i2 = u;}
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
    var uv  = scc + .5f;
        uv  = (2.f*uv-scr.xy)/scr.y;
    var i1  = dot(int2(fract(uv)*float(mapz)),int2(1,mapz));
    var tim = int(time.frame);
    var fr1 = ((tim-0+frmz) % frmz)*mapz*mapz;
    var fr2 = ((tim-1+frmz) % frmz)*mapz*mapz;
    var l   = leem[fr1+i1]*1.0f;
    var n1  = noor[i1 + 0*mapz*mapz];
    var n2  = noor[i1 + 1*mapz*mapz];
    var c   = float4(0);
    var q   = dot(int2(uv+1.f),int2(1,2));
    if(q==0){c = float4(0,n1*.5f+.5f,0);}
    if(q==1){c = float4(0,n2*.5f+.5f,0);}
    if(q==2){c = float4(0,(n1+n2)*.25f+.5f,0)+leep[i1];}
    if(q==3){c = float4(length(n1+n2)*.5f);}
    screen[dtid.xy] = c * float(uv.x>-1.f&&uv.x<1.f);
}