//generates the training data for NN
//NN trained on small paths will be able to deduce long paths
//this algorithm can be also used for other simulations like
//double pendulum, walking blob... etc
import std;
static const float pi   = 3.1415926535897932384626433832795;
static const int   thrd = 1<<5;     //gpu threads
static const int   lnt  = 1<<9;//total lines
static const int   lnd  = 64;//line sibdivisions
static const float stpp = .04;//step translation
static const float wllf = -.04;//wall force
static const float prna = .25;//path randomness amount
static const float erfz = .07;//error force amount 
[StorageBuffer(lnd*lnt)] RWStructuredBuffer<float2> lnp;    //lines position
[StorageBuffer(lnd*lnt)] RWStructuredBuffer<float2> lnm;    //lines movement
[StorageBuffer(lnd*lnt)] RWStructuredBuffer<float2> lnm2;   //lines movement change
[StorageBuffer(1  *lnt)] RWStructuredBuffer<float > lnm3;   //lines movement change length
[StorageBuffer(1  *lnt)] RWStructuredBuffer<float2> lng;    //lines position target
[StorageBuffer(1  *lnt)] RWStructuredBuffer<int   > lns;    //lines position source

struct MyDataPtr : IDifferentiablePtrType
{
    typealias Differential = MyDataPtr;
    RWStructuredBuffer<float2> buf;
};
float2 load(MyDataPtr p, uint i)
{
    return p.buf[i];
}
[BackwardDerivativeOf(load)]
void load_bwd(DifferentialPtrPair<MyDataPtr> p,uint i,float2 dOut)
{
    p.d.buf[i] += dOut;
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
[Differentiable]
float2 lineGrad(float2 u, float2 p1, float2 p2, float w)
{
    float2 a = normalize(p2 - p1);
    u -= (p1 + p2) * .5f;
    float2 q = float2(
        dot(a, u),
        dot(float2(-a.y, a.x), u)
    );
    float l = distance(p1, p2) * .5f;
    float2 d = q - float2(clamp(q.x, -l, l), 0.f);
    float ld = length(d);
    //if (ld < 1e-6f) return 0.f.xx;
    float h = saturate(1.f - ld / w);
    //float h = 1.f/exp(dot(d,d)/w/w*2.5f);
    //float h = w/(.01f+dot(d,d)*dot(d,d)*999.f)*.1f;
    float2 n = -d / ld;
    return h * (a * n.x + float2(-a.y, a.x) * n.y);
    //return (normalize(u-p1)/exp(dot(u-p1,u-p1)*11.)+normalize(u-p2)/exp(dot(u-p2,u-p2)*11.))*.5;
    //return (normalize(u-p1)/length(u-p1)+normalize(u-p2)/length(u-p2))*.02;
}
[Differentiable]
float2 terrainHeight2(float2 u)
{
    var r = float2(0);
    var a = .1f+sin(float(time.frame)*.2f)*.0f;
    r += lineGrad(u,float2(-.2,.1),float2(.8,.1),a);
    r += lineGrad(u,float2(.0,.1),float2(.0,.5),a);
    r += lineGrad(u,float2(.1,-.7),float2(.3,.1),a);
    r += lineGrad(u,float2(.5,-.6),float2(.5,-.4),a);
    r += lineGrad(u,float2(.5,-.3),float2(.5,.1),a);
    r += lineGrad(u,float2(-.6,-.4),float2(-.7,.0),a);
    r += lineGrad(u,float2(-.6,.6),float2(-.7, .3),a);
    return r;
}
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
float2 limit(float2 m)
{
    var ml = length(m);
    if(ml!=0.f){m = m/ml*min(1.f,ml);}
    return m;
}
[Differentiable]
float2 phyStep(float2 p, float2 m)
{
    p += m*stpp;                    //self move
    p += terrainHeight2(p)*wllf;   //world reaction
    return p;
}
[DispatchOnce]
[shader("compute")]
[WorkgroupCount(lnt/thrd, 1, 1)]
[numthreads(thrd, 1, 1)]
void ini(uint3 id3 : SV_DispatchThreadID)
{
    var id1 = int(id3.x);
    var w = 0*lnt+id1;
    var p = lnp[w];
    var r1 = rnd(uint(w*2+0));
    var r2 = rnd(uint(w*2+1));
    for(int i=0; i<lnd; i++)
    {
        lnm[w] = cos(r1*2.f*pi-float2(0,.5)*pi)*sqrt(r2);
        w += lnt;
        r1+=(rnd(uint(w*2+0))-.5f)*prna;
        r2 = rnd(uint(w*2+1));
    }
    //r1 = rnd(uint(w*2+0));
    //r2 = rnd(uint(w*2+1));
    //lnp[id1] = float2(r1,r2)*2.f-1.f;
    lnp[id1] = cos(r1*2.f*pi-float2(0,.5)*pi)*sqrt(r2);
    lnp[id1] += terrainHeight2(lnp[id1])*wllf;//avoid starting inside walls
    lnp[id1] += terrainHeight2(lnp[id1])*wllf;//avoid starting inside walls
}
[shader("compute")]
[WorkgroupCount(lnt/thrd, 1, 1)]
[numthreads(thrd, 1, 1)]
void forward(uint3 id3 : SV_DispatchThreadID)
{
    var id1 = int(id3.x);
    var w = 0*lnt+id1;
    var p = lnp[w];
    for(int i=0; i<lnd; i++)
    {
        lnp[w] = p;
        p = phyStep(lnp[w], lnm[w]);
        w += lnt;
    }
    if(time.frame==0) lng[id1] = lnp[(lnd-1)*lnt+id1];
    if(time.frame==0) lns[id1] = lnd-2;
}
[shader("compute")]
[WorkgroupCount(lnt/thrd, 1, 1)]
[numthreads(thrd, 1, 1)]
void backward(uint3 id3 : SV_DispatchThreadID)
{
    var id1 = int(id3.x);
    var tg  = lng[id1];
    var ti  = lns[id1];
    var d   = float2(0);
    var et  = 0.f;
    var ct  = 0.f;
    if(length(lnp[(ti+1)*lnt+id1]-tg)<stpp*.4f){ti--;  lns[id1]=ti;}
    //for(int i=0; i<lnd; i++)
    //{
    //    lnm2[i*lnt+id1] = float2(0);
    //}
    for(int i=lnd-2; i>ti; i--)
    {
        var w   = i*lnt+id1;
        var d_p = diffPair(lnp[w]);
        var d_m = diffPair(lnm[w]);
        var p2  = lnp[(i+1)*lnt+id1];
        var td  = tg-p2;
        var d2  = float2(0);
        d2     += tg-p2;
        et     += dot(d2,d2);
        d      += d2;
        bwd_diff(phyStep)(d_p, d_m, d);
        ct += dot(d_m.d,d_m.d);
        lnm2[w] = d_m.d;
        d       = d_p.d;
    }
    d = float2(0);
    for(int i=ti; i>=0; i--)
    {
        var w   = i*lnt+id1;
        var d_p = diffPair(lnp[w]);
        var d_m = diffPair(lnm[w]);
        var p0  = lnp[(i+0)*lnt+id1];
        var p1  = lnp[(i+1)*lnt+id1];
        var p2  = tg;
        if(i+2<lnd){p2 = lnp[(i+2)*lnt+id1];}
        var td  = tg-p1;
        var tl  = length(td);  if(tl!=0.f){td /= tl;}
        var d2  = float2(0);
            d2 += td*float(ti==i);
            d2 += (p2+p0)*.5f-p1;
        et     += dot(d2,d2);
        d      += d2;
        bwd_diff(phyStep)(d_p, d_m, d);
        ct     += dot(d_m.d,d_m.d);
        lnm2[w] = d_m.d;
        d       = d_p.d;
    }
    

    //lnm3[id1] = 1.0f;
    if(ct != 0.f){ct = sqrt(et/ct);}
    lnm3[id1] = ct;
}
[shader("compute")]
[WorkgroupCount(lnt/thrd, 1, 1)]
[numthreads(thrd, 1, 1)]
void apply(uint3 id3 : SV_DispatchThreadID)
{
    var id1 = int(id3.x);
    var m   = lnm3[id1]*erfz;
    for(int i=0; i<lnd; i++)
    {
        var w   = i*lnt+id1;
        lnm[w] = limit(lnm[w] + lnm2[w]*m);
    }
}
[shader("compute")]
[numthreads(16, 16, 1)]
void clear(uint3 id : SV_DispatchThreadID)
{
    uint width, height;
    screen.GetDimensions(width, height);
    if (id.x >= width ) return;
    if (id.y >= height) return;
    float2 u = (2.f*float2(id.xy)-float2(width,height)) / float(height);
    screen[id.xy] = float4(0,terrainHeight2(u)+.5f,0);
    //var d_p = diffPair(u);
    //bwd_diff(terrainHeight2)(d_p, float2(1,0));
    //screen[id.xy] = float4(0,d_p.d*.05f+.5f,0);
    //screen[id.xy] = float4(0);
}
[shader("compute")]
[WorkgroupCount(lnt/thrd, lnd-1, 1)]
[numthreads(thrd,1,1)]
void render(uint3 id3 : SV_DispatchThreadID)
{
    var id1 = int(dot(id3,uint3(1,lnt,1)));
    var p1 = lnp[id1+0*lnt];
    var p2 = lnp[id1+1*lnt];
    uint width, height;
    screen.GetDimensions(width, height);
    //draw ray
    {
        var G  = 1.f;
        var d  = p2-p1;
        var dl = length(d);
        if(dl!=0.f){d /= dl;}
        var d1 = 1.f/d;  
        if(d.x==0.f){d1.x=0.f;}
        if(d.y==0.f){d1.y=0.f;}
        var d2 = abs(d1);
        var d3 = step(float2(0),d)*2.f-1.f;
        var sf = float2(width,height);
        var sm = G*(2.f*sf-sf)/sf.y;
            sm*= .999f;
        var m1 = min(( sm-p1)* d1,
                     (-sm-p1)* d1);
        var m2 = min(( sm-p2)*-d1,
                     (-sm-p2)*-d1);
        p1 +=  d*max(0.f,max(m1.x,m1.y));
        p2 += -d*max(0.f,max(m2.x,m2.y));
        var pl = dot(p2-p1,d);
        var p  = (p1*sf.y/G + sf)/2.f;       //world coordinates to pixel coordinates
        var l0 =  pl*sf.y/G      /2.f;       //world size        to pixel size
        var lt = 0.f;
        if(all(abs(p1)<sm*1.001f) &&
           all(abs(p2)<sm*1.001f)){lt=l0;}
        while(lt > 0.f)
        {
            var g = (1.001f-fract(p*d3))*d2;
            var l = min(g.x,g.y);  if(l==0.f){lt=0.f;}
            var pd= float2((fract(p)==float2(0)) & (d<float2(0)));
            screen[uint2(p-pd)] = cos((float(id3.y)/float(lnd)+float4(0,1,2,0)/3.f)*pi*2.f)*.5f+.5f;
            p  += l*d;
            lt -= l;
        }
    }
}