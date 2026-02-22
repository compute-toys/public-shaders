//whats the performance of bruteforce pathtracing with physics on 2025
//on a common computer very meh
//click to control balls and light
//try bigger value in line 9 for ryT2
#define PI 3.1415926535897932384f
//rays
#define G 8f            //camera zoom
#define ryT  64u        //threads
#define ryT2 1024u       //try 512u 1024u 2048u 4096u 8888u 32768u
#define ryL 4u          //rays loop size
#define sWH (SCREEN_WIDTH*SCREEN_HEIGHT)
#define fti (1<<10)     //float to int, int to float, resolution
//balls
#define dt .1f      // time step
#define BZ 64       // total balls
struct Ray
{
    p: vec2f,       //ray start position
    d: vec2f,       //ray direction
    c: vec3f,       //ray color
};
struct Data
{
    ballsPos: array<vec2f,BZ>,
    ballsVel: array<vec2f,BZ>,
    rays:     array<Ray,ryT*ryT2>,
}
#storage C array<atomic<i32>,SCREEN_WIDTH*SCREEN_HEIGHT*3>
#storage D Data;
var<workgroup> Bp: array<vec2f,BZ>;
var<workgroup> Bv: array<vec2f,BZ>;
fn hash(a: u32) -> u32
{
    var x = a;
    x ^= x >> 17;  x *= 0xed5ad4bbu;
    x ^= x >> 11;  x *= 0xac4c1b51u;
    x ^= x >> 15;  x *= 0x31848babu;
    x ^= x >> 14;  return x;
}
fn rnd(a: u32) -> f32
{
    var h   = hash(a);
    var msk = (1u << 23u) - 1u;
    return f32(h & msk) / f32(1u << 23u);
}
fn getBallSize(id: i32) -> f32
{
    return .2f*pow(2f,f32(id)/f32(BZ)*3f);
}
fn getBallCol(id: i32) -> vec3f
{
    return cos(f32(id)*custom.ballcols+1f+vec3(2,1,3))*.3f+.7f;
}
#dispatch_once physcIni
#workgroup_count physcIni 1 1 1 
@compute @workgroup_size(BZ,1,1)
fn physcIni(@builtin(global_invocation_id) id3: vec3u)
{
    var id1 = id3.x;
    var p = vec2f(rnd(id1+u32(BZ)*0u + 2124135346u),
                  rnd(id1+u32(BZ)*1u + 2124135346u))*2f-1f;
    var v = vec2f(0);
    D.ballsPos[id1] = p*G;
    D.ballsVel[id1] = v;
}
#workgroup_count physc 1 1 1 
@compute @workgroup_size(BZ,1,1)
fn physc(@builtin(global_invocation_id) id3: vec3u)
{
    var res  = vec2f(SCREEN_WIDTH,SCREEN_HEIGHT);
    var res2 = res/res.y*G*.75f;
    var mus  = G*(2f*vec2f(mouse.pos)-res)/res.y;
    var id1 = i32(id3.x);
    var r   = id1;
    var vs  = vec2f(0);
    var bl1 = getBallSize(id1);
    Bp[id1] = D.ballsPos[id1];
    Bv[id1] = D.ballsVel[id1];
    workgroupBarrier();
    var b1p = Bp[id1];
    var b1v = Bv[id1];
    for(var j=0; j<BZ; j++)
    {
        var bl2 = getBallSize(j);
        var bl3 = bl1+bl2;
        var p3 = Bp[j]-b1p;
        var v3 = Bv[j]-b1v;
        var l  = length(p3);
        var f  = max(0f,bl3-l);
        var ld = 0f; if(l!=0f){ld=1f/l;}
        var d  = p3*ld;
        vs += -d*f;
        vs += d*dot(d,v3)*f32(f!=0f)*.1f;
    }
    var wf = -.01f;
    var w1 = max(vec2f(0),b1p-res2);
    var w2 = min(vec2f(0),b1p+res2);
    vs += (w1+w2)*wf;
    //vs += -b1p*.001f;
    var msv = mus-b1p;
    vs += f32(mouse.click!=0)*f32(length(msv)<bl1)*(msv-b1v);
    vs = b1v + vs*dt;
    vs *= custom.friction;
    D.ballsPos[id1] = vs+b1p;
    D.ballsVel[id1] = vs;
    //if(false)//draw
    //{
    //    var m    = (2f*vec2f(mouse.pos)-res)/res.y;
    //    var zoom = custom.a*.5f;
    //    var b = vec2i(b1p*res.y*zoom+.5f*res);
    //    textureStore(screen, b, vec4f(1));
    //}
}
fn rayMarch(rayIn:Ray, seed:u32) -> Ray
{
    var ray = rayIn;
    var p1  = ray.p;
    var lot = 99f;//ignored length
    var bsh = lot;//shortest intersection to ball
    var ibl = 0;//ball id intersected
    for(var i=0; i<BZ; i++)//ray ball intersect
    {
        var bp = Bp[i];
        var br = getBallSize(i);
        var db = bp-ray.p;
        var l  = dot(ray.d, db);
        var dh = ray.p+ray.d*l-bp;
        var h2 = br*br-dot(dh,dh);
        var h1 = sqrt(h2);
        var l1 = l-h1;          if(l1<.001f){l1=lot;}
        var l2 = l+h1;          if(l2<.001f){l2=lot;}
            l1 = min(l1,l2);    if(h2<.001f){l1=lot;}
        if(l1 < bsh){ibl = i;}
        bsh = min(bsh,l1);
    }
    ray.p += ray.d*bsh*1.001f;
    var bonce = bsh != lot;
    //draw ray
    {
        var p2 = ray.p;
        var d  = ray.d;
        var d1 = 1f/d;  
        if(d.x==0f){d1.x=0f;}
        if(d.y==0f){d1.y=0f;}
        var d2 = abs(d1);
        var d3 = step(vec2f(0),d)*2f-1f;
        var sf = vec2f(SCREEN_WIDTH,SCREEN_HEIGHT);
        var sm = G*(2f*sf-sf)/sf.y;
            sm*= .999f;
        var m1 = min(( sm-p1)* d1,
                     (-sm-p1)* d1);
        var m2 = min(( sm-p2)*-d1,
                     (-sm-p2)*-d1);
        p1 +=  d*max(0f,max(m1.x,m1.y));
        p2 += -d*max(0f,max(m2.x,m2.y));
        var pl = dot(p2-p1,d);
        var p  = (p1*sf.y/G + sf)/2f;       //world coordinates to pixel coordinates
        var l0 =  pl*sf.y/G      /2f;       //world size        to pixel size
        var lt = 0f;
        if(all(abs(p1)<sm*1.001f) &&
           all(abs(p2)<sm*1.001f)){lt=l0;}
        while(lt > 0f)
        {
            var g = (1f-fract(p*d3))*d2;
            var l = min(g.x,g.y);  if(l==0f){lt=0f;}
            var pd= vec2f((fract(p)==vec2f(0)) & (d<vec2f(0)));
            var w = dot(vec2u(p-pd),vec2u(1,SCREEN_WIDTH));
            var c = vec3i(min(l,lt)*ray.c*f32(fti));
            atomicAdd(&C[w+0u*sWH], c.x);
            atomicAdd(&C[w+1u*sWH], c.y);
            atomicAdd(&C[w+2u*sWH], c.z);
            p  += l*d;
            lt -= l;
        }
    }
    if(length(ray.p)>=lot*.5f || dot(ray.c,ray.c)<pow(.002f,2f))//ray new
    {
        var r0 = rnd(seed+0u);
        var r1 = rnd(seed+1u);
        var r2 = rnd(seed+2u);
        var r3 = rnd(seed+3u);
        var sf = vec2f(SCREEN_WIDTH,SCREEN_HEIGHT);
        var lp = G*.5f*cos(vec2f(.77,.55)*time.elapsed+vec2f(0,2.22));
        if(mouse.click!=0){lp = G*(2f*vec2f(mouse.pos)-sf)/sf.y;}
        var p  = cos(r0*PI*2f-vec2f(0,.5)*PI)*sqrt(r1)*.1f + lp;
            //p  = cos(r0*PI*2f-vec2f(0,.5)*PI)*sqrt(r1)*getBallSize(0) + Bp[0];
        var d  = cos(r2*PI*2f-vec2f(0,.5)*PI);
        ray.p  = p;
        ray.d  = d;
        ray.c  = vec3f(1);
        bonce = false;
    }
    if(bonce)//ray bounce
    {
        var r0  = rnd(seed+0u)-.5f;
        //var nrm = ray.p-Bp[ibl];
        //var nrl = length(nrm);
        //if(nrl!=0f){nrl=1f/nrl;}
        //nrm *= nrl;
        //var rfl = reflect(ray.d, nrm);
        var a1 = cos(6f*r0-vec2f(0,.5)*PI);
        ray.d  = ray.d*a1.x + ray.d.yx*vec2f(-1,1)*a1.y;
        ray.c *= getBallCol(ibl);
    }
    return ray;
}
#dispatch_once rayIni
#workgroup_count rayIni ryT2 1 1
@compute @workgroup_size(ryT,1,1)
fn rayIni(@builtin(global_invocation_id) id3: vec3u)
{
    D.rays[id3.x] = Ray();
}
#workgroup_count rayDo ryT2 1 1
@compute @workgroup_size(ryT,1,1)
fn rayDo(@builtin(global_invocation_id) id3: vec3u)
{
    var id1  = id3.x;
    var seed = (time.frame*ryT*ryT2 + id1)*4u*ryL+756352967u;
    var ray  = D.rays[id1];
    for(var w = id1 % ryT; w < BZ; w += ryT)
    {
        Bp[w] = D.ballsPos[w];
    }
    workgroupBarrier();
    for(var k = 0u; k < ryL; k++)
    {
        ray = rayMarch(ray, seed + k*6u);
    }
    D.rays[id1] = ray;
}
@compute @workgroup_size(16,16,1)
fn main_image(@builtin(global_invocation_id) id3: vec3u)
{
    var sc = vec2u(SCREEN_WIDTH,SCREEN_HEIGHT);
    if(id3.x >= sc.x) { return; }
    if(id3.y >= sc.y) { return; }
    var r = id3.y*sc.x + id3.x;
    var c = vec4i(atomicLoad(&C[r+0u*sWH]),
                  atomicLoad(&C[r+1u*sWH]),
                  atomicLoad(&C[r+2u*sWH]),0);
    var cc = i32(custom.accumul);
    atomicStore(&C[r+0u*sWH], c.x*cc/33);
    atomicStore(&C[r+1u*sWH], c.y*cc/33);
    atomicStore(&C[r+2u*sWH], c.z*cc/33);
    var d = vec4f(c)/f32(fti)*custom.bright/f32(ryT2);

    var sf = vec2f(SCREEN_WIDTH,SCREEN_HEIGHT);
    var p  = G*(2f*vec2f(id3.xy)-sf)/sf.y;
    //d += getCol(p);
    //d  = vec4f(getNrm(p),0,0);

    textureStore(screen, id3.xy, d);
}