//3D version of https://compute.toys/view/2714
#define PI 3.1415926535897932384f
//rays
#define G 32f            //camera zoom
#define ryT  32u        //threads
#define ryT2 16384u       //try 512u 1024u 2048u 4096u 8192u 16384u 32768u
#define ryL 22u          //rays loop size
#define sWH (SCREEN_WIDTH*SCREEN_HEIGHT)
#define scz G          //screen size in the 3D world
#define fti (1<<10)     //float to int, int to float, resolution
//balls
#define dt .1f      // time step
#define BZ 64       // total balls
struct Ray
{
    p: vec3f,       //ray start position
    d: vec3f,       //ray direction
    c: vec3f,       //ray color
};
struct Data
{
    ballsPos: array<vec3f,BZ>,
    ballsVel: array<vec3f,BZ>,
    rays:     array<Ray,ryT*ryT2>,
}
#storage C array<atomic<i32>,sWH*3>
#storage D Data
var<workgroup> Bp: array<vec3f,BZ>;
var<workgroup> Bv: array<vec3f,BZ>;
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
    return custom.balsiz*pow(2f,f32(id)/f32(BZ)*3f);
}
fn getBallCol(id: i32) -> vec3f
{
    return cos(f32(id)/f32(BZ)*4f*PI+vec3(0,1,2)*2f*PI/3f)*.2f+.8f;
}
fn getBallDifuse(id: i32) -> f32
{
    return .1f;//cos(f32(id)/f32(BZ)*22f)*1f+1f;
}
#dispatch_once physcIni
#workgroup_count physcIni 1 1 1 
@compute @workgroup_size(BZ,1,1)
fn physcIni(@builtin(global_invocation_id) id3: vec3u)
{
    var id1 = id3.x;
    var p = vec3f(
        rnd(id1+u32(BZ)*0u + 2124135346u),
        rnd(id1+u32(BZ)*1u + 2124135346u),
        rnd(id1+u32(BZ)*2u + 2124135346u))*2f-1f;
    var v = vec3f(0);
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
    var vs  = vec3f(0);
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
    vs += -b1p*.002f;
    //var msv = mus-b1p;
    //vs += f32(mouse.click!=0)*f32(length(msv)<bl1)*(msv-b1v);
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
fn rayMarch(rayIn:Ray, seed:u32, camPos:vec3f, camDir:vec3f, lp:vec3f) -> Ray
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
    ray.p += ray.d*bsh*.999f;
    var bonce = bsh != lot;
    //draw ray
    {
        var p2  = ray.p;
        var sr  = f32(SCREEN_WIDTH)/f32(SCREEN_HEIGHT);
        var sx  = cross(camDir,vec3f(0,1,0));
        var sy  = cross(camDir,sx);
        var sxl = length(sx);  if(sxl != 0f){sxl = 1f/sxl;}
        var syl = length(sy);  if(syl != 0f){syl = 1f/syl;}
        var l   = dot(p1 - camPos, camDir) / -dot(ray.d, camDir);
        var hit = p1 + ray.d * l;
        var hic = hit - camPos;
        var ap1 = custom.lenz;

        var pc = p1-camPos;
        var rx = dot(pc, sx*sxl)/(scz*sr);
        var ry = dot(pc, sy*syl)/(scz   );
        var cx = u32((rx*.5f+.5f) * f32(SCREEN_WIDTH ));
        var cy = u32((ry*.5f+.5f) * f32(SCREEN_HEIGHT));
        var w  = cx + cy*u32(SCREEN_WIDTH);
        var co = vec3i(ray.c*f32(fti));
        if(dot(camDir,ray.d)<0f && l>=0f && dot(hic,hic)<1f/ap1/ap1 && abs(rx)<1f && abs(ry)<1f)
        {
            atomicAdd(&C[w+0u*sWH], co.x);
            atomicAdd(&C[w+1u*sWH], co.y);
            atomicAdd(&C[w+2u*sWH], co.z);
        }
    }
    if(!bonce || dot(ray.c,ray.c)<pow(.002f,2f))//ray new
    {
        var r0 = rnd(seed+0u);
        var r1 = rnd(seed+1u);
        var r2 = rnd(seed+2u);
        var r3 = rnd(seed+3u);
        var r4 = rnd(seed+4u);
        var z     = r0 * 2f - 1f;
        var theta = r1 * PI*2f;
        var r     = sqrt(1f - z*z);
        ray.d  = vec3f(r*cos(theta), r*sin(theta), z);
            z     = r2 * 2f - 1f;
            theta = r3 * PI*2f;
            r     = sqrt(1f - z*z);
        ray.p  = vec3f(r*cos(theta), r*sin(theta), z)*pow(r4, 1f/3f)*1f + lp;
        ray.c  = vec3f(1);
    }
    if(bonce)//ray bounce
    {
        var rf = reflect(ray.d, normalize(Bp[ibl]-ray.p));  //ray.d
        var an = cos((rnd(seed+0u)-.5f)*getBallDifuse(ibl)-vec2f(0,.5)*PI);
        var xy = cos(rnd(seed+1u)*PI*2f-vec2f(0,.5)*PI)*an.y;
        var sx = cross(rf,vec3f(0,1,0));
        var sy = cross(rf,          sx);
        var sxl = length(sx);  if(sxl != 0f){sxl = 1f/sxl;}
        var syl = length(sy);  if(syl != 0f){syl = 1f/syl;}
        ray.d  = rf*an.x + sx*sxl*xy.x + sy*syl*xy.y;
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
    var lp  = G*.5f*cos(vec3f(.77,.55,.33)*time.elapsed+vec3f(0,2.22,7.77));
    var res = vec2f(SCREEN_WIDTH,SCREEN_HEIGHT);
    var m   = (2f*vec2f(mouse.pos)-res)/res.y;
    var camPos = vec3f(1,0,0);
    if(mouse.click!=0){camPos = vec3f(cos(m.x),m.y,sin(m.x));}
    camPos *= G;
    var camDir = -normalize(camPos);
    for(var k = 0u; k < ryL; k++)
    {
        ray = rayMarch(ray, seed + k*6u, camPos, camDir, lp);
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
    var d = vec4f(c)/f32(fti)*1111f*custom.bright/f32(ryT2);
    textureStore(screen, id3.xy, d);
}