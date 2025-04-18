#define PI  3.1415926535897932384f
#define bnc 99f          //ray max bounces
#define srf .01f        //distance considered surface
#define nrd (srf*.1f)   //normal calcualtion
#define ote 88.f        //outside fractal space
#define rml 70u         //raymarch loop
#define G 64f           //size of fractal
#define ryT  64u        //threads
#define ryT2 128u        //threads
#define ryL 2u          //rays loop size
#define sWH (SCREEN_WIDTH*SCREEN_HEIGHT)
#define fti (1<<10)     //float to int, int to float, resolution

struct rayST
{
    p: vec2f,       //ray start position
    d: vec2f,       //ray direction
    c: vec3f,       //ray color
    l: f32,         //ray in SDF
    b: f32,         //ray bounces
    t: f32,         //ray collision
};
#storage D array<atomic<i32>,SCREEN_WIDTH*SCREEN_HEIGHT*3>
#storage R array<rayST,ryT*ryT2+1u>

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
fn toru(p2:vec2f, a:f32, ra:f32, rb:f32) -> f32
{
    var sc = cos(a-vec2f(.5,.0)*PI);
    var p = p2;
    p.x = abs(p.x);
    var k = 0f;
    if(sc.y*p.x>sc.x*p.y){k = dot(p.xy,sc);}
    else                 {k = length(p.xy);}
    return sqrt( dot(p,p) + ra*ra - 2f*ra*k ) - rb;
}
fn getSDF(p2: vec2f) -> f32
{
    var n = cos(custom.A*6f-vec2f(0,.5)*PI);
    var m = mat2x2f(n,vec2f(-1,1)*n.yx)*2f;
    var a = 99999f;
    var s = 1f;
    var p = p2;
    for(var i=0; i<3; i++)
    {
        p*=m; s*=.5f;
        p = abs(p)-G*1.2f;
        var to = toru(p, 2f, 4f*G/9f, 1f*G/9f)*s;
        a = min(a,abs(to));
    }
    return a;
}
fn getCol(p2: vec2f) -> vec4f
{
    var n = cos(custom.A*6f-vec2f(0,.5)*PI);
    var m = mat2x2f(n,vec2f(-1,1)*n.yx)*2f;
    var a = vec4f(0);
    var s = 1f;
    var p = p2;
    for(var i=0; i<3; i++)
    {
        p*=m; s*=.5f;
        p = abs(p)-G*1.2f;
        var c = cos(f32(i)*1.2f+3f+vec4f(1,2,3,4))
                *vec4f(.25,.25,.25,0)+vec4f(.75,.75,.75,.01);
        var to = toru(p, 2f, 4f*G/9f, 1f*G/9f)*s;
        var l = 1f-4f*abs(to);
            l = max(l,0f);
        a += l*c;
    }
    return a;
}
fn getNrm(pos: vec2f) -> vec2f
{
    return normalize(vec2f(getSDF(pos+nrd*vec2f(1,0)) - getSDF(pos-nrd*vec2f(1,0)),
                           getSDF(pos+nrd*vec2f(0,1)) - getSDF(pos-nrd*vec2f(0,1))));
}
fn rayNew(seed:u32) -> rayST
{
    var r0 = rnd(seed+0u);
    var r1 = rnd(seed+1u);
    var r2 = rnd(seed+2u);
    var sf = vec2f(SCREEN_WIDTH,SCREEN_HEIGHT);
    var p  = cos(r0*PI*2f-vec2f(0,.5)*PI)*sqrt(r1)*2f +
             G*(2f*vec2f(mouse.pos)-sf)/sf.y;
    var d  = cos(r2*PI*2f-vec2f(0,.5)*PI);
    var l  = getSDF(p);
    return rayST(
        p,              //ray start position
        d,              //ray direction
        vec3f(1),       //ray color
        l,              //ray in SDF
        0f,             //ray bounces
        0f,             //ray collision
    );
}
fn rayMarch(rayIn:rayST) -> rayST
{
    var ray = rayIn;
    var p1 = ray.p;
    for(var i=0u; i<rml && ray.t==0f && ray.l<ote; i++)
    {
        ray.p += ray.l*ray.d;
        var l = getSDF(ray.p);
        ray.t = f32(l<srf && l<ray.l);
        ray.l = l;
    }
    //draw ray
    var p2 = ray.p - f32(ray.l < srf)*ray.d*srf;
    var d  = ray.d;  if(dot(p2-p1,d)<0f){d=-d;}
    var d1 = 1f/d;  
    if(d.x==0f){d1.x=0f;}
    if(d.y==0f){d1.y=0f;}
    var d2 = abs(d1);
    var d3 = step(vec2f(0),d)*2f-1f;
    var sf = vec2f(SCREEN_WIDTH,SCREEN_HEIGHT);
    var sm = G*(2f*sf-sf)/sf.y;    sm*=.99f;
    var m1 = min(( sm-p1)* d1,
                 (-sm-p1)* d1);
    var m2 = min(( sm-p2)*-d1,
                 (-sm-p2)*-d1);
    p1 +=  d*max(0f,max(m1.x,m1.y));
    p2 += -d*max(0f,max(m2.x,m2.y));
    if(dot(p2-p1,d)<0f){d=-d;}
    var p  = (p1 *sf.y/G + sf)/2f;    //world coordinates to pixel coordinates
    var l0 = length(p2-p1)*sf.y/G/2f; //world size        to pixel size
        l0 = min(l0,length(sf)*.9f);
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
        atomicAdd(&D[w+0u*sWH], c.x);
        atomicAdd(&D[w+1u*sWH], c.y);
        atomicAdd(&D[w+2u*sWH], c.z);
        p  += l*d;
        lt -= l;
    }
    return ray;
}
fn refra(e1:vec2f, e2:vec2f, e3:f32) -> vec2f
{
    let k = 1.0 - e3 * e3 * (1.0 - dot(e2, e1) * dot(e2, e1));
    return e3 * e1 - (e3 * dot(e2, e1) + sqrt(k)) * e2;
}
fn rayBounce(rayIn:rayST, seed:u32) -> rayST
{
    var ray = rayIn;
    ray.p -= ray.d*srf;
    var nrm = getNrm(ray.p);
    var col = getCol(ray.p);
    var rfl = reflect(ray.d, nrm);
        rfl = refra(ray.d, nrm, .6f);
    var r1 = rnd(seed+0u)-.5f;
    var a1 = cos(col.w*r1-vec2f(0,.5)*PI);
    //rfl = rfl*a1.x + rfl.yx*vec2f(-1,1)*a1.y;
    ray.d = rfl;
    ray.c *= col.xyz;
    ray.p += nrm*getSDF(ray.p)*-2f*f32(dot(rfl,nrm)<0f);
    ray.l = getSDF(ray.p);
    ray.t = 0f;
    ray.b += 1f;
    return ray;
}
#dispatch_once rayIni
#workgroup_count rayIni ryT2 1 1
@compute @workgroup_size(ryT,1,1)
fn rayIni(@builtin(global_invocation_id) id3: vec3u)
{
    var id1  = id3.x;
    var seed = (time.frame*ryT*ryT2 + id1)*6u*ryL+756352967u;
    R[id1] = rayNew(seed);
}
#workgroup_count rayDo ryT2 1 1
@compute @workgroup_size(ryT,1,1)
fn rayDo(@builtin(global_invocation_id) id3: vec3u)
{
    var sc = textureDimensions(screen);
    var id1  = id3.x;
    var seed = (time.frame*ryT*ryT2 + id1)*6u*ryL;
    var ray  = R[id1];
    for(var k = 0u; k < ryL; k++)
    {
        var b = ray.b>=bnc || ray.l>=ote;
        if(b          ){ray = rayNew(seed);}        seed +=4u;
        if(ray.t != 0f){ray = rayBounce(ray,seed);} seed +=2u;
        ray = rayMarch(ray);
    }
    R[id1] = ray;
    //use last structure on buffer to save
    //fast aproximate of total rays fired
    if(id3.x==0u){R[ryT*ryT2].b += 1f;}
}
@compute @workgroup_size(16,16,1)
fn main_image(@builtin(global_invocation_id) id3: vec3u)
{
    var sc = textureDimensions(screen);
    if(id3.x >= sc.x) { return; }
    if(id3.y >= sc.y) { return; }
    var r = id3.y*sc.x + id3.x;
    if(mouse.click!=0){
        if(id3.x==0u && id3.y==0u){R[ryT*ryT2].b = 1f;}
        atomicStore(&D[r+0u*sWH], 0);
        atomicStore(&D[r+1u*sWH], 0);
        atomicStore(&D[r+2u*sWH], 0);
    }
    var c = vec4i(atomicLoad(&D[r+0u*sWH]),
                  atomicLoad(&D[r+1u*sWH]),
                  atomicLoad(&D[r+2u*sWH]),0);
    var d = vec4f(c)/f32(fti)/R[ryT*ryT2].b*custom.C;

    var sf = vec2f(SCREEN_WIDTH,SCREEN_HEIGHT);
    var p  = G*(2f*vec2f(id3.xy)-sf)/sf.y;
    //d += getCol(p);

    textureStore(screen, id3.xy, d);
}