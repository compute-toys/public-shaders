//using gradient descent to make pendulum go up and stay up
//using slang autodiff
//this is the most straight forward algorithm but its very slow or gets stuck
//previously backprop computed how to position pendulum end through time
//but now backprop computes how much force apply to pendulum end through time
//this algo is better but still cant keep the pendulum up after geting it up
//ok new update looks like it solves it
//mouse controls camera

import std;
static const float  pi     = 3.1415926535897932384626433832795;
static const int    thrd   = 1<<5;   //gpu threads
static const int    phyt   = 1<<8;   //physics loops total
static const int    phyl   = 1<<8;   //physics loops per frame
static const int    pdv    = 3;      //pendulum divisions
static const float  brsf   = .2f;    //bar force
static const float  brsr   = .1f;    //bar friction
static const float  brsl   = 8.0f;   //bar length
static const float  brsw   = 1.0f;   //bar radius
static const float  grvt   =-.003;   //gravity
static const float  camdis = 66.f;   //camera distance
static const float3 campdf = float3(0,0,1);   //camera pos default
static const float  erfz   = .05f;    //error force amount 

struct Doto
{
    float2 p[pdv*phyt]; //position
    float2 v[pdv*phyt]; //velocity
    float2 f[pdv]; //force
    float2 p2[pdv];     //position backpropagation
    float2 v2[pdv];     //velocity backpropagation
    float2 f2[pdv];     //force    backpropagation
    float2 pt[pdv];     //position target
    float2 vt[pdv];     //velocity target
    float  m[phyt];     //input movement pos
    float  m2[phyt];    //input movement pos change
    float  m3;          //sum
    int trgl;           //target step location
};
[StorageBuffer(1)] RWStructuredBuffer<Doto> D;    //lines position

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
    var h   = hash(a);
    var msk = (1u << 23u) - 1u;
    return float(h & msk) / float(1u << 23u);
}
[Differentiable]
float2 barforce(float2 p1,float2 p2,float2 v1,float2 v2, no_diff float barl)
{
    var np = p1-p2;
    var l  = length(np);
    if(l!=0.f){np = np/l;}
    var f = float2(0);
    f += np*(barl-l) * brsf;        //force bar atraction
    f += dot(v2-v1,np)*np * brsr;   //force bar friction
    return f;
}
[Differentiable]
float2 controlforce(float2 p,float2 v, float m)
{
    var yf = (float2(0,0)-p) * brsf * float2(0,1); //updown force
    var v1 = v * brsr; //friction
    return yf-v1+float2(m,0);
}
[shader("compute")]
[WorkgroupCount(1, 1, 1)]
[numthreads(1, 1, 1)]
void forward(uint3 id3 : SV_DispatchThreadID)
{
    var id1 = int(id3.x);
    var tim = int(time.frame);
    if(tim==0)
    {
        D[0].trgl = phyt-1;
        for(var k=0; k<pdv; k++)
        {
            D[0].p[k] = float2(0,-k*brsl);
            D[0].v[k] = float2(0);
        }
        for(var i=0; i<phyt; i++){
            D[0].m[i] = (rnd(uint(i))*2.f-1.f)*.1f;
        }
    }
    for(var i=1; i<phyt; i++)
    {
        for(var k=0; k<pdv; k++) D[0].f[k] = float2(0,float(k!=0))*grvt;
        var i1 = i-1;
        if(true){
            var p = D[0].p[0+i1*pdv];
            var v = D[0].v[0+i1*pdv];
            var m = D[0].m[i1];
            D[0].f[0] += controlforce(p, v, m);
        }
        for(var k=0; k<pdv-1; k++)//bar
        {
            var p1 = D[0].p[k+0+i1*pdv];
            var v1 = D[0].v[k+0+i1*pdv];
            var p2 = D[0].p[k+1+i1*pdv];
            var v2 = D[0].v[k+1+i1*pdv];
            var f = barforce(p1,p2, v1,v2, brsl);
            D[0].f[k+0] += f;
            D[0].f[k+1] -= f;
        }
        for(var k=0; k<pdv; k++) D[0].v[k+i*pdv] = D[0].v[k+i1*pdv]+D[0].f[k];
        for(var k=0; k<pdv; k++) D[0].p[k+i*pdv] = D[0].p[k+i1*pdv]+D[0].v[k+i*pdv];
    }
    if(tim==phyt/phyl-1)//set target
    {
        //for(var k=0; k<pdv; k++) D[0].pt[k] = D[0].p[k+(phyt-1)*pdv];
        //for(var k=0; k<pdv; k++) D[0].vt[k] = D[0].v[k+(phyt-1)*pdv];
        for(var k=0; k<pdv; k++) D[0].pt[k] = float2(0,brsl*float(pdv-1));
        for(var k=0; k<pdv; k++) D[0].vt[k] = float2(0,0);
    }
}
[shader("compute")]
[WorkgroupCount(1, 1, 1)]
[numthreads(1, 1, 1)]
void backward(uint3 id3 : SV_DispatchThreadID)
{
    var tim = int(time.frame);
    var id1 = int(id3.x);
    //var trgl = D[0].trgl;
    //var trgp = 0.f;
    //var trgv = 0.f;
    //for(var k=0; k<pdv; k++){trgp = length(D[0].pt[k]-D[0].p[k+trgl*pdv])*float(k==pdv-1);}
    //for(var k=0; k<pdv; k++){trgv = length(D[0].vt[k]-D[0].v[k+trgl*pdv])*float(k==pdv-1);}
    //if(trgp<.1f*brsl && trgv<.2f){
    //if(trgp<.1f*brsl){
    //    trgl--;
    //    D[0].trgl=trgl;
    //    //for(var i=0; i<phyt-1; i++) D[0].m[i] = D[0].m[i+1];
    //    //var mm = 1111111.f;
    //    //for(var i= 0; i<trgl-1; i++) mm = min(mm,D[0].m[i]);
    //    //for(var i=mm; i<trgl-1; i++) D[0].m[i] = D[0].m[i+1];
    //}
    for(var k=0; k<pdv; k++) D[0].p2[k] = float2(0);
    for(var k=0; k<pdv; k++) D[0].v2[k] = float2(0);
    for(var i=0; i<phyt; i++) D[0].m2[i] = 0.f;
    var nrmlz = 0.f;
    for(var i=phyt-1; i>=1; i--)
    {
        var w = float(i)/float(phyt-1);
        w = pow(w,4.f);
        var i1 = i-1;
        //if(i>=trgl){ //add error to backpropagate
            for(var k=pdv-1; k<pdv; k++){
                var er = (D[0].pt[k]-D[0].p[k+i*pdv]);//*float2(0,1);
                //var el = length(er);  if(el>1.f && el!=0.f){er=er/el;}
                D[0].p2[k] += er*w;
            }
            //for(var k=pdv-1; k<pdv; k++){
            //    var er = (D[0].vt[k]-D[0].v[k+i*pdv]);//*float2(0,1);
            //    //var el = length(er);  if(el>1.f && el!=0.f){er=er/el;}
            //    D[0].v2[k] += er*w*.2f;
            //}
        //}
        for(var k=0; k<pdv; k++){
            var p1 = D[0].p[k+min(phyt-1,i+1)*pdv];
            var p2 = D[0].p[k+max(     0,i-1)*pdv];
            var er = (p1+p2)*.5f - D[0].p[k+i*pdv];
            //D[0].p2[k] += er*.1f;
        }
        for(var k=0; k<pdv; k++) D[0].v2[k] += D[0].p2[k];
        for(var k=0; k<pdv; k++) D[0].f2[k]  = D[0].v2[k];
        for(var k=0; k<pdv-1; k++)//bar
        {
            var p1 = diffPair(D[0].p[k+0+i1*pdv]);
            var v1 = diffPair(D[0].v[k+0+i1*pdv]);
            var p2 = diffPair(D[0].p[k+1+i1*pdv]);
            var v2 = diffPair(D[0].v[k+1+i1*pdv]);
            var d = D[0].f2[k+0]-
                    D[0].f2[k+1];
            bwd_diff(barforce)(p1,p2, v1,v2, brsl, d);
            D[0].p2[k+0] += p1.d;
            D[0].v2[k+0] += v1.d;
            D[0].p2[k+1] += p2.d;
            D[0].v2[k+1] += v2.d;
        }
        if(true){//movecontrol
            var p = diffPair(D[0].p[0+i1*pdv]);
            var v = diffPair(D[0].v[0+i1*pdv]);
            var m = diffPair(D[0].m[i1]);
            var d = D[0].f2[0];
            bwd_diff(controlforce)(p, v, m, d);
            D[0].p2[0] += p.d;
            D[0].v2[0] += v.d;
            D[0].m2[i1] = m.d;
            nrmlz += abs(m.d);
        }
    }
    if(nrmlz != 0.f){nrmlz = 1.f/nrmlz;}
    D[0].m3 = nrmlz;
    //D[0].m3 = .05f/float(pdv*phyt);//nrmlz;
}
[shader("compute")]
[WorkgroupCount(1, 1, 1)]
[numthreads(1, 1, 1)]
void apply(uint3 id3 : SV_DispatchThreadID)
{
    var id1 = int(id3.x);
    var m   = D[0].m3 * erfz;
    for(int i=0; i<phyt; i++) D[0].m[i] += D[0].m2[i]*m;
    for(int i=0; i<phyt; i++) D[0].m[i] *= .999f;
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
    var i = int(time.frame) % phyt;
    var z = float(i)/float(phyt)*2.f-1.f;
        z *= custom.z*44.f;
    var l = 111111111.f;
    for(var k = 0; k < pdv-1; k++)//union of all pendulum bars
    {
        var p1 = float3(D[0].p[k+0+i*pdv], z);
        var p2 = float3(D[0].p[k+1+i*pdv], z);
        l = min(l, sdCapsule(p, p1, p2, brsw));
    }
    return l;
    //return length(p)-1.f;
}
[shader("compute")]
[numthreads(16, 16, 1)]
void clear(uint3 id3 : SV_DispatchThreadID)
{
    uint width, height;
    screen.GetDimensions(width, height);
    if (id3.x >= width ) return;
    if (id3.y >= height) return;
    var zoom = custom.a*.1f;
    var c = float2(width,height);
    var u = (2.f*float2(id3.xy)   -c)/c.y/zoom;
    var m = (2.f*float2(mouse.pos)-c)/c.y;
    var campos = cos(time.elapsed*float3(-23,-9,27)*.002f+float3(11,2,22));
        //campos = cos(0.f*float3(-23,-9,27)*.02f+float3(11,2,22));
        campos = campdf;
    if(mouse.click!=0){campos = float3(cos(m.x),m.y,sin(m.x));}
    campos *= camdis;
    var camdir = -normalize(campos);
    var sd = normalize(float3(camdir.z,0,-camdir.x));
    var up = -cross(camdir,sd);

    var p = campos + sd*u.x + up*u.y;
    for(var i = 0; i < 33; i++)
    {
        p += objects(p)*camdir;
    }
    var s  = 1.f/64.f;
    var c1 = float3(0);//camdir*.5f+.5f;
    var c2 = normalize(float3(
        objects(p+float3(s,0,0)) - objects(p-float3(s,0,0)),//color of surface
        objects(p+float3(0,s,0)) - objects(p-float3(0,s,0)),
        objects(p+float3(0,0,s)) - objects(p-float3(0,0,s))));
    if(objects(p)<.1) c1 = c2*.5f+.5f;
    screen[id3.xy] = float4(c1,1);
}
[shader("compute")]
[WorkgroupCount(phyt/thrd, 1, 1)]
[numthreads(thrd, 1, 1)]
void draw(uint3 id3 : SV_DispatchThreadID)
{
    var id1 = int(id3.x);
    uint width, height;
    screen.GetDimensions(width, height);
    var zoom = custom.a*.1f;
    var c = float2(width,height);
    var u = (2.f*float2(id3.xy)   -c)/c.y/zoom;
    var m = (2.f*float2(mouse.pos)-c)/c.y;
    var campos = cos(time.elapsed*float3(-23,-9,27)*.002f+float3(11,2,22));
        //campos = cos(0.f*float3(-23,-9,27)*.02f+float3(11,2,22));
        campos = campdf;
    if(mouse.click!=0){campos = float3(cos(m.x),m.y,sin(m.x));}
    campos *= camdis;
    var camdir = -normalize(campos);
    var sd = normalize(float3(camdir.z,0,-camdir.x));
    var up = -cross(camdir,sd);
    for(var k=0; k<pdv; k++)
    {
        var z  = float(id1)/float(phyt)*2.f-1.f;
            z *= custom.z*44.f;
        var p  = float3(D[0].p[k+id1*pdv],z);
        var a  = p - campos;
        var b  = int2(int((dot(a,sd)*zoom*.5f*c.y/c.x+.5f)*c.x),
                      int((dot(a,up)*zoom*.5f        +.5f)*c.y));
        var col = float(k)/float(pdv)+float4(0,1,2,3)/3.f;
            col = cos(col*2.f*pi)*.5f+.5f;
        if(b.x>=0 && b.x<int(width) && b.y>=0 && b.y<int(height)) screen[b] = col;
    }
}