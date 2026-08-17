//this generates the training data for a NN that will control a n-pendulum
//each pendulum starts with a random velocity, position, movements throught time
//then using gradient descent finds the shortest path to get to the final position, velocity
//in the least amount of time and force

import std;
static const float  pi     = 3.1415926535897932384626433832795;
static const int    thrdr  = 1<<5;      //gpu threads render
static const int    thrd   = 1<<5;      //gpu threads physcs
static const int    phyt   = 1<<7;      //physics loops total
static const int    btch   = 1<<8;     //batches
static const int    pdv    = 3;         //pendulum divisions
static const float  brsf   = .2;        //bar force
static const float  brsr   = .1;        //bar friction
static const float  brsl   = 8.0;       //bar length
static const float  brsw   = 1.0;       //bar radius
static const float  grvt   =-.003;      //gravity
static const float  camdis = 66.;       //camera distance
static const float3 campdf = float3(0,0,1);//camera pos default
static const float  erfz   = .02;       //error force amount
static const uint   rnds   = 1234567;   //random seed
static const uint   shwz   = 128;       //square size to show each batch
struct Doto
{
    float2 p[pdv*phyt*btch]; //position
    float2 v[pdv*phyt*btch]; //velocity
    float2 f[pdv*phyt*btch]; //force
    float  t[phyt*btch];     //time passed per step
    float  t2[phyt*btch];    //time backpro change
    float  ts[phyt*btch];    //time passed sum
    float2 p2[pdv*btch];     //position backpropagation
    float2 v2[pdv*btch];     //velocity backpropagation
    float2 f2[pdv*btch];     //force    backpropagation
    float2 pt[pdv*btch];     //position target
    float2 vt[pdv*btch];     //velocity target
    float  m[phyt*btch];     //input movement force
    float  m2[phyt*btch];    //input backpro change
    float  m3[btch];          //sum
    int    trg[btch];        //target step
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
float rnd(int ai)
{
    var a   = uint(ai); 
    var h   = hash(a+rnds);
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
[WorkgroupCount(btch/thrd, 1, 1)]
[numthreads(thrd, 1, 1)]
void forward(uint3 id3 : SV_DispatchThreadID)
{
    var id1 = int(id3.x);
    var tim = int(time.frame);
    if(tim==0)
    {
        var p = float2(0);
        for(var k=0; k<pdv; k++){
            D[0].p[id1+k*btch] = p;
            p += cos(rnd(id1+(k+0*pdv)*btch)*2.f*pi-float2(0,.5)*pi)*brsl;
        }
        for(var k=0; k< pdv; k++) D[0].v[id1+k*btch] = (rnd(id1+(k+1*pdv)*btch)*2.f-1.f)*.9f;
        for(var i=0; i<phyt; i++) D[0].m[id1+i*btch] = (rnd(id1+(i+2*pdv)*btch)*2.f-1.f)*.3f;
        for(var i=0; i<phyt; i++) D[0].t[id1+i*btch] = 1.f;
        D[0].ts[id1] = 0.f;
        D[0].trg[id1] = phyt-1;
    }
    var trg = D[0].trg[id1];
    var ts = 0.f;
    for(var i=1; i<=trg; i++)
    {
        var i1 = i-1;
        for(var k=0; k<pdv; k++) D[0].f[id1+(k+i*pdv)*btch] = float2(0,float(k!=0))*grvt;
        if(true){
            var p = D[0].p[id1+(0+i1*pdv)*btch];
            var v = D[0].v[id1+(0+i1*pdv)*btch];
            var m = D[0].m[id1+i1*btch];
            D[0].f[id1+(0+i*pdv)*btch] += controlforce(p, v, m);
        }
        for(var k=0; k<pdv-1; k++)//bar
        {
            var p1 = D[0].p[id1+(k+0+i1*pdv)*btch];
            var v1 = D[0].v[id1+(k+0+i1*pdv)*btch];
            var p2 = D[0].p[id1+(k+1+i1*pdv)*btch];
            var v2 = D[0].v[id1+(k+1+i1*pdv)*btch];
            var f = barforce(p1,p2, v1,v2, brsl);
            D[0].f[id1+(k+0+i*pdv)*btch] += f;
            D[0].f[id1+(k+1+i*pdv)*btch] -= f;
        }
        var stp = D[0].t[id1+i*btch];
        ts += stp;
        D[0].ts[id1+i*btch] = ts;
        for(var k=0; k<pdv; k++) D[0].v[id1+(k+i*pdv)*btch] = D[0].v[id1+(k+i1*pdv)*btch]+D[0].f[id1+(k+i*pdv)*btch]*stp;
        for(var k=0; k<pdv; k++) D[0].p[id1+(k+i*pdv)*btch] = D[0].p[id1+(k+i1*pdv)*btch]+D[0].v[id1+(k+i*pdv)*btch]*stp;
    }
    if(tim==0)//set target
    {
        for(var k=0; k<pdv; k++) D[0].pt[id1+k*btch] = D[0].p[id1+(k+trg*pdv)*btch];
        for(var k=0; k<pdv; k++) D[0].vt[id1+k*btch] = D[0].v[id1+(k+trg*pdv)*btch];
        //for(var k=0; k<pdv; k++) D[0].pt[id1+k*btch] = float2(0,brsl*float(k)*.9f);
        //for(var k=0; k<pdv; k++) D[0].vt[id1+k*btch] = float2(0,0);
    }
}
[shader("compute")]
[WorkgroupCount(btch/thrd, 1, 1)]
[numthreads(thrd, 1, 1)]
void backward(uint3 id3 : SV_DispatchThreadID)
{
    var tim = int(time.frame);
    var id1 = int(id3.x);
    var trg = D[0].trg[id1];
    for(var k=0; k<pdv; k++) D[0].p2[id1+k*btch] = float2(0);
    for(var k=0; k<pdv; k++) D[0].v2[id1+k*btch] = float2(0);
    for(var i=0; i<=trg; i++) D[0].m2[id1+i*btch] = 0.f;
    for(var i=0; i<=trg; i++) D[0].t2[id1+i*btch] = 0.f;
    var nrmlz = 0.f;
    for(var i=trg; i>=1; i--)
    {
        var dmp = 1.f/float(i);
        var i1 = i-1;
        if(i==trg){
            for(var k=pdv-1; k<pdv; k++){
                D[0].p2[id1+k*btch] += (D[0].pt[id1+k*btch]-D[0].p[id1+(k+i*pdv)*btch])*dmp;
                D[0].v2[id1+k*btch] += (D[0].vt[id1+k*btch]-D[0].v[id1+(k+i*pdv)*btch])*dmp*11.f;
            }
        }
        var stp = D[0].t[id1+i*btch];
        var t2  = 0.f;
        for(var k=0; k<pdv; k++) D[0].v2[id1+k*btch] += D[0].p2[id1+k*btch]*stp;
        for(var k=0; k<pdv; k++) D[0].f2[id1+k*btch]  = D[0].v2[id1+k*btch]*stp;
        for(var k=0; k<pdv; k++) t2 += dot(D[0].p2[id1+k*btch],D[0].v[id1+(k+i*pdv)*btch]);
        for(var k=0; k<pdv; k++) t2 += dot(D[0].v2[id1+k*btch],D[0].f[id1+(k+i*pdv)*btch]);
        D[0].t2[id1+i*btch] = t2;
        nrmlz += abs(t2);
        for(var k=0; k<pdv-1; k++)//bar
        {
            var p1 = diffPair(D[0].p[id1+(k+0+i1*pdv)*btch]);
            var v1 = diffPair(D[0].v[id1+(k+0+i1*pdv)*btch]);
            var p2 = diffPair(D[0].p[id1+(k+1+i1*pdv)*btch]);
            var v2 = diffPair(D[0].v[id1+(k+1+i1*pdv)*btch]);
            var d = D[0].f2[id1+(k+0)*btch]-
                    D[0].f2[id1+(k+1)*btch];
            bwd_diff(barforce)(p1,p2, v1,v2, brsl, d);
            D[0].p2[id1+(k+0)*btch] += p1.d;
            D[0].v2[id1+(k+0)*btch] += v1.d;
            D[0].p2[id1+(k+1)*btch] += p2.d;
            D[0].v2[id1+(k+1)*btch] += v2.d;
        }
        if(true){//movecontrol
            var p = diffPair(D[0].p[id1+(0+i1*pdv)*btch]);
            var v = diffPair(D[0].v[id1+(0+i1*pdv)*btch]);
            var m = diffPair(D[0].m[id1+(i1)*btch]);
            var d = D[0].f2[id1+0*btch];
            bwd_diff(controlforce)(p, v, m, d);
            D[0].p2[id1+0*btch] += p.d;
            D[0].v2[id1+0*btch] += v.d;
                var mx = .06f; 
                var m1 = D[0].m[id1+i1*btch];
                var mc = clamp(m1 + m.d, -mx, mx)-m1;
                //var mc = m.d*max(0.f,1.f-pow(abs(m1)/mx,8.f));
            D[0].m2[id1+i1*btch] = m.d;
            nrmlz += abs(m.d);
        }
    }
    if(nrmlz != 0.f){nrmlz = 1.f/nrmlz;}
    D[0].m3[id1] = nrmlz;
}
[shader("compute")]
[WorkgroupCount(btch/thrd, 1, 1)]
[numthreads(thrd, 1, 1)]
void apply(uint3 id3 : SV_DispatchThreadID)
{
    var id1 = int(id3.x);

    var trgi = D[0].trg[id1];
    var trgp = 0.f;
    var trgv = 0.f;
    for(var k=pdv-1; k<pdv; k++){trgp += length(D[0].pt[id1+k*btch]-D[0].p[id1+(k+trgi*pdv)*btch]);}
    for(var k=pdv-1; k<pdv; k++){trgp += length(D[0].vt[id1+k*btch]-D[0].v[id1+(k+trgi*pdv)*btch])*11.f;}
    //var th = trgp/float(pdv*2)/brsl*11.f;
    var th = trgp/2.f/brsl*11.f;
        th = 1.f-.001f/pow(2.f,th*th);

    var m   = D[0].m3[id1] * erfz;
    var a   = .07f;
    for(int i=0; i<=trgi; i++) D[0].m[id1+i*btch] += D[0].m2[id1+i*btch]*m;
    for(int i=0; i<=trgi; i++) D[0].t[id1+i*btch] += D[0].t2[id1+i*btch]*m;
    //for(int i=0; i<=trgi; i++) D[0].m[id1+i*btch] = clamp(D[0].m[id1+i*btch]+D[0].m2[id1+i*btch]*m,-a,a);
    for(int i=0; i<=trgi; i++) D[0].m[id1+i*btch] *= .999f;
    for(int i=0; i<=trgi; i++) D[0].t[id1+i*btch] *= th;

    if(D[0].ts[id1+trgi*btch]<float(trgi)*.8f)
    {
        for(int i=0; i<=trgi; i++) D[0].m2[id1+i*btch] = D[0].m[id1+i*btch];
        int k = 0;
        for (int x = 0; x<=trgi; x++)
        {
            // Before the first point
            if (x <= D[0].ts[id1+0*btch]){
                D[0].m[id1+x*btch] = D[0].m2[id1+0*btch];
                continue;
            }
            // Find the two old points surrounding x
            while (k < trgi-1 && x > D[0].ts[id1+(k+1)*btch]) k++;
            // After the last point
            if (x >= D[0].ts[id1+trgi*btch]){
                D[0].m[id1+x*btch] = 0.f;
                continue;
            }
            // Linear interpolation
            float x0 = D[0].ts[id1+(k+0)*btch];
            float x1 = D[0].ts[id1+(k+1)*btch];
            float y0 = D[0].m2[id1+(k+0)*btch];
            float y1 = D[0].m2[id1+(k+1)*btch];
            float t  = (x - x0) / (x1 - x0);
            D[0].m[id1+x*btch] = y0 + t * (y1 - y0);
        }
        for(int i=0; i<=trgi; i++) D[0].t[id1+i*btch] = 1.f;
        D[0].trg[id1] = int(D[0].ts[id1+trgi*btch]);
    }
}
float sdCapsule(float3 p, float3 a, float3 b, float r)
{
    var pa = p - a;
    var ba = b - a;
    var h = clamp(dot(pa, ba) / dot(ba, ba), 0.f, 1.f);
    return length(pa - ba * h) - r;
}
float objects(float3 p, float t, int trg, int id1)
{
    var i = int(time.frame) % trg;
    var z = t/float(phyt)*2.f-1.f;
        z *= custom.z*44.f;
    var l = 111111111.f;
    for(var k = 0; k < pdv-1; k++)//union of all pendulum bars
    {
        var p1 = float3(D[0].p[id1+(k+0+i*pdv)*btch], z);
        var p2 = float3(D[0].p[id1+(k+1+i*pdv)*btch], z);
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
    var id3i = int2(id3.xy);
    var id1  = dot(id3i/shwz,int2(1,int(width)/shwz));
    var zoom = custom.a*.1f;
    var cc= float(shwz);
    var c = float2(width,height);
    var u = (2.f*float2(id3i % shwz)-cc)/cc/zoom;
    var m = (2.f*float2(mouse.pos)  -c)/c.y;
    var campos = cos(time.elapsed*float3(-23,-9,27)*.002f+float3(11,2,22));
        //campos = cos(0.f*float3(-23,-9,27)*.02f+float3(11,2,22));
        campos = campdf;
    if(mouse.click!=0){campos = float3(cos(m.x),m.y,sin(m.x));}
    campos *= camdis;
    var camdir = -normalize(campos);
    var sd = normalize(float3(camdir.z,0,-camdir.x));
    var up = -cross(camdir,sd);
    var trg = D[0].trg[id1]+1;
    var t = D[0].ts[id1+(int(time.frame) % trg)*btch];

    var p = campos + sd*u.x + up*u.y;
    for(var i = 0; i < 33; i++)
    {
        p += objects(p,t,trg,id1)*camdir;
    }
    var s  = 1.f/64.f;
    var c1 = float3(0);//camdir*.5f+.5f;
    var c2 = normalize(float3(
        objects(p+float3(s,0,0),t,trg,id1) - objects(p-float3(s,0,0),t,trg,id1),//color of surface
        objects(p+float3(0,s,0),t,trg,id1) - objects(p-float3(0,s,0),t,trg,id1),
        objects(p+float3(0,0,s),t,trg,id1) - objects(p-float3(0,0,s),t,trg,id1)));
    if(objects(p,t,trg,id1)<.1) c1 = c2*.5f+.5f;
    screen[id3.xy] = float4(c1,1);
}
[shader("compute")]
[WorkgroupCount(phyt/thrdr, btch, 1)]
[numthreads(thrdr, 1, 1)]
void draw(uint3 id3 : SV_DispatchThreadID)
{
    var id3x = int(id3.x);
    var id1  = int(id3.y);
    var trg = D[0].trg[id1];
    if(id3x>trg) return;
    uint width, height;
    screen.GetDimensions(width, height);
    var ftx = int(width)/shwz;
    var mxy = int2(id1 % ftx, id1 / ftx) * shwz;
    var zoom = custom.a*.1f;
    var cc= float(shwz);
    var c = float2(width,height);
    var m = (2.f*float2(mouse.pos)-c)/c.y;
    var campos = cos(time.elapsed*float3(-23,-9,27)*.002f+float3(11,2,22));
        //campos = cos(0.f*float3(-23,-9,27)*.02f+float3(11,2,22));
        campos = campdf;
    if(mouse.click!=0){campos = float3(cos(m.x),m.y,sin(m.x));}
    campos *= camdis;
    var camdir = -normalize(campos);
    var sd = normalize(float3(camdir.z,0,-camdir.x));
    var up = -cross(camdir,sd);
    var ts = D[0].ts[id1+id3x*btch];
    for(var k=0; k<pdv; k++)
    {
        var z  = ts/float(phyt)*2.f-1.f;
            z *= custom.z*44.f;
        var p  = float3(D[0].p[id1+(k+id3x*pdv)*btch],z);
        var a  = p - campos;
        var b  = int2(int((dot(a,sd)*zoom*.5f+.5f)*cc),
                      int((dot(a,up)*zoom*.5f+.5f)*cc))+mxy;
        var col = float(k)/float(pdv)+float4(0,1,2,3)/3.f;
            col = cos(col*2.f*pi)*.5f+.5f;
        if(b.x>=0 && b.x<int(width) && b.y>=0 && b.y<int(height)) screen[b] = col;
    }
}