//findind the path of least effor on a 2Dheightmap
//use slider to see 2Deffortmap or 2Dheightmap
//going somewhere norupdown uses energy
//going somewhere uphill    uses more energy
//going somewhere downhill  uses same energy as norupdown
//click to change final location
//starting location is the dark  point
//final    location is the white point
import std;
static const int    thrd = 16;     //gpu threads
static const int    mapz = thrd*16;
static const float2 strt = float2( .4, .4); //location start
static const float2 ende = float2( .3,-.4); //location end
[StorageBuffer(mapz*mapz  )] RWStructuredBuffer<float > hmap;//height map
[StorageBuffer(mapz*mapz*2)] RWStructuredBuffer<float > leem;//least effort map
[StorageBuffer(mapz*mapz  )] RWStructuredBuffer<float > leep;//least effort paths
[StorageBuffer(mapz       )] RWStructuredBuffer<float2> papo;//paths painters
float4 toLinear(float4 sRGB)
{
	var cutoff = float4(sRGB >= float4(0.04045));
	var higher = pow((sRGB + 0.055f)/1.055f, float4(2.4));
	var lower  = sRGB/12.92f;
	return higher*cutoff + lower*(1.f-cutoff);
}
float4 turbo(float t) {
    var c0 = float4(0.1140890109226559, 0.06288340699912215, 0.2248337216805064,0);
    var c1 = float4(6.716419496985708, 3.182286745507602, 7.571581586103393,0);
    var c2 = float4(-66.09402360453038, -4.9279827041226, -10.09439367561635,0);
    var c3 = float4(228.7660791526501, 25.04986699771073, -91.54105330182436,0);
    var c4 = float4(-334.8351565777451, -69.31749712757485, 288.5858850615712,0);
    var c5 = float4(218.7637218434795, 67.52150567819112, -305.2045772184957,0);
    var c6 = float4(-52.88903478218835, -21.54527364654712, 110.5174647748972,0);
    return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));
}
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
float2 mod2(float2 a, float b)
{
    return fract(a/b)*b;
}
[DispatchOnce]
[shader("compute")]
[WorkgroupCount(mapz/thrd, mapz/thrd, 1)]
[numthreads(thrd, thrd, 1)]
void ini(uint3 dtid : SV_DispatchThreadID)
{
    var i1 = dot(dtid,uint3(1,mapz,1));
    var i2 = float2(dtid.xy)+.5f;
    var uv = i2/float(mapz) *2.f-1.f;
    hmap[i1] = terrainHeight(uv);
    var j2  = (strt+1.f)*.5f*float(mapz);
    var j1  = dot(int2(j2),int2(1,mapz));
    var tim = int(time.frame);
    var fr1 = ((tim+0) & 1)*mapz*mapz;
    var fr2 = ((tim+1) & 1)*mapz*mapz;
    var v   = 32.f* float(mapz) * float(j1!=i1);
    leem[fr1+i1] = v;
    leem[fr2+i1] = v;
}
[shader("compute")]
[WorkgroupCount(mapz/thrd, mapz/thrd, 1)]
[numthreads(thrd, thrd, 1)]
void grow(uint3 dtid : SV_DispatchThreadID)
{
    var tim = int(time.frame);
    var fr1 = ((tim+0) & 1)*mapz*mapz;
    var fr2 = ((tim+1) & 1)*mapz*mapz;
    var i2 = float2(dtid.xy)+.5f;
    var uv = i2/float(mapz) *2.f-1.f;
    var i1 = dot(int2(i2),int2(1,mapz));
    var h0 = hmap[i1];
    var l0 = leem[i1];
    var cz = 1;
    for(var y = -cz; y <= cz; y++){
    for(var x = -cz; x <= cz; x++){
        var xy = float2(x,y);
        var j2 = mod2(i2+xy,float(mapz));
            j2 = clamp(i2+xy,float2(.5),float2(mapz)-.5f);
        var j1 = dot(int2(j2),int2(1,mapz));
        var hm = max(0.f, h0-hmap[j1]);
        var ef = leem[fr1+j1] + length(xy) + hm*333.f;
        l0 = min(l0,ef);
    }}
    leem[fr2+i1] = l0;
    if(time.frame == 0u){leep[i1] = 0.f;}
    //if(mouse.click != 0){leep[i1] = 0.f;}
}
[shader("compute")]
[WorkgroupCount(1, 1, 1)]
[numthreads(1, 1, 1)]
void fall(uint3 dtid : SV_DispatchThreadID)
{
    var tim = int(time.frame);
    var fr1 = ((tim+0) & 1)*mapz*mapz;
    var fr2 = ((tim+1) & 1)*mapz*mapz;
    var strt2 = (strt+1.f)*.5f*float(mapz);
    var ende2 = (ende+1.f)*.5f*float(mapz);
    var i2 = papo[0];
    var uv = i2/float(mapz) *2.f-1.f;
    var i1 = dot(int2(i2),int2(1,mapz));
    var dd = float2(0);
    var l0 = leem[fr2+i1];
    var cz = 1;
    for(var y = -cz; y <= cz; y++){
    for(var x = -cz; x <= cz; x++){
        var xy = float2(x,y);
        var j2 = mod2(i2+xy,float(mapz));
            j2 = clamp(i2+xy,float2(.5),float2(mapz)-.5f);
        var j1 = dot(int2(j2),int2(1,mapz));
        var l1 = leem[fr2+j1];
        if(l1 < l0){dd = xy;  l0 = l1;}
    }}
    i2 = clamp(i2+dd,float2(.5),float2(mapz)-.5f);
    if(time.frame==0u        ){i2 = ende2;}
    if(length(i2-strt2) < 1.f){i2 = ende2;}
    {
        uint width, height;
        screen.GetDimensions(width, height);
        var scr = float2(width, height);
        uv = mouse.pos + .5f;
        uv = (2.f*uv-scr.xy)/scr.y;
        uv = fract(uv*.5f+.5f)*float(mapz);
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
    var uv  = scc + .5f;
        uv  = (2.f*uv-scr.xy)/scr.y;
        uv  = fract(uv*.5f+.5f)*float(mapz);
    var i1  = dot(int2(uv),int2(1,mapz));
    var tim = int(time.frame);
    var fr1 = ((tim+0) & 1)*mapz*mapz;
    var fr2 = ((tim+1) & 1)*mapz*mapz;
    var l   = leem[fr2+i1]/mapz*.4f;
    var c   = toLinear(turbo(fract(l)));
    screen[dtid.xy] = c + leep[i1];
    if(custom.a >= .5f) screen[dtid.xy] = float4(hmap[i1]);
}
