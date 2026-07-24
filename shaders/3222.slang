//n pendulum used to configure parameters and know how it behaves
//mouse moves it

import std;
static const float pi     = 3.1415926535897932384626433832795;
static const int   thrd   = 1<<5;     //gpu threads
static const int   pdv    = 3;      //pendulum divisions
static const float brsf   = 4.0f;    //bar force
static const float brsr   = 1.0f;    //bar friction
static const float brsl   = 8.0f;    //bar length
static const float brsw   = 1.0f;    //bar radius
static const float phyl   = 2;       //physics loops per frame
static const float grvt   =-.03;     //physics loops per frame
static const float dt     = .1f;     //time step
static const float camdis = 22.f;     //camera distance

struct Doto
{
    float2 p[pdv]; //position
    float2 v[pdv]; //velocity
    float2 f[pdv]; //force
};
[StorageBuffer(3*pdv*2*4)] RWStructuredBuffer<Doto> D;    //lines position

float2 mod2(float2 a, float b)
{
    return fract(a/b)*b;
}
uint hash(uint a)
{
    var x = a;
    x ^= x >> 17;  x *= 0xed5ad4bbu;
    x ^= x >> 11;  x *= 0xac4c1b51u;
    x ^= x >> 15;  x *= 0x31848babu;
    x ^= x >> 14;  return x;
}
float rnd(uint a)
{
    var h   = hash(a+0u);
    var msk = (1u << 23u) - 1u;
    return float(h & msk) / float(1u << 23u);
}
float2 barforce(float2 p1,float2 p2,float2 v1,float2 v2, float barl)
{
    var np = p1-p2;
    var l  = length(np);
    if(l!=0.f){np = np/l;}
    var f = float2(0);
    f += np*(barl-l) * brsf;        //force bar atraction
    f += dot(v2-v1,np)*np * brsr;   //force bar friction
    return f;
}
[shader("compute")]
[WorkgroupCount(1, 1, 1)]
[numthreads(1, 1, 1)]
void forward(uint3 id3 : SV_DispatchThreadID)
{
    var id1 = int(id3.x);
    if(time.frame==0)
    {
        for(var k=0; k<pdv; k++)
        {
            D[0].p[k] = float2(0,k*brsl);
            D[0].v[k] = float2(0);
        }
    }
    uint width, height;
    screen.GetDimensions(width, height);
    var scr = float2(width,height);
    var mus = (2.f*float2(mouse.pos)-scr)/scr.y * camdis;
    for(var i=0; i<phyl; i++)
    {
        for(var k=0; k<pdv; k++)
        {
            D[0].f[k] = float2(0,float(k!=0))*grvt;
        }
        if(true){
            var dr = (mus*float2(1,0)-D[0].p[0]) * brsf;
            var v1 = D[0].v[0] * brsr;
            D[0].f[0] += dr-v1;
        }
        for(var k=0; k<pdv-1; k++)//bar
        {
            var p1 = D[0].p[k+0];
            var v1 = D[0].v[k+0];
            var p2 = D[0].p[k+1];
            var v2 = D[0].v[k+1];
            var f = barforce(p1,p2, v1,v2, brsl);
            D[0].f[k+0] += f;
            D[0].f[k+1] -= f;
        }
        for(var k=0; k<pdv; k++) D[0].v[k] += D[0].f[k]*dt;
        for(var k=0; k<pdv; k++) D[0].p[k] += D[0].v[k];
    }
}
float sdCapsule(float3 p, float3 a, float3 b, float r)
{
    var pa = p - a;
    var ba = b - a;
    var h = clamp(dot(pa, ba) / dot(ba, ba), 0.f, 1.f);
    return length(pa - ba * h) - r;
}
float objects(float3 p)
{
    var l = 111111111.f;
    for(var k = 0; k < pdv-1; k++)//union of all pendulum bars
    {
        var p1 = float3(D[0].p[k+0], 0);
        var p2 = float3(D[0].p[k+1], 0);
        l = min(l, sdCapsule(p, p1, p2, brsw));
    }
    return l;
    //return length(p)-1.f;
}
[shader("compute")]
[numthreads(16, 16, 1)]
void draw(uint3 id : SV_DispatchThreadID)
{
    uint width, height;
    screen.GetDimensions(width, height);
    if (id.x >= width ) return;
    if (id.y >= height) return;
    var c = float2(width,height);
    var u = (2.f*float2(id.xy)-c) / c.y;
    var m = (2.f*float2(mouse.pos)-c)/c.y;
    var campos = cos(time.elapsed*float3(-23,-9,27)*.02f+float3(11,2,22));
        campos = cos(0.f*float3(-23,-9,27)*.02f+float3(11,2,22));
        campos = float3(0,0,-1);
    //if(mouse.click!=0){campos = float3(cos(m.x),m.y,sin(m.x));}
    campos *= camdis;
    var camdir = -normalize(campos);

    float3x3 mtx;
    mtx[0] = normalize(float3(camdir.z,0,-camdir.x));
    mtx[1] = cross(camdir,-mtx[0]);
    mtx[2] = camdir;
    var ray = mul(normalize(float3(u,1)),mtx);//direction of ray from camera

    var p = campos;
    for(var i = 0; i < 44; i++)
    {
        p += objects(p)*ray;
    }
    var s  = 1.f/64.f;
    var c1 = ray*.5f+.5f;
    var c2 = normalize(float3(
        objects(p+float3(s,0,0)) - objects(p-float3(s,0,0)),//color of surface
        objects(p+float3(0,s,0)) - objects(p-float3(0,s,0)),
        objects(p+float3(0,0,s)) - objects(p-float3(0,0,s))));
    if(objects(p)<.1) c1 = c2*.5f+.5f;
    screen[id.xy] = float4(c1,1);
}