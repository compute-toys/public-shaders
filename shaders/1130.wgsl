#define PI  3.1415926535897932384f
#define srf .01f        //distance considered surface
#define nrd (srf*.1f)   //normal calcualtion
#define ote 88.f        //outside fractal space
#define rml 66u         //raymarch loop
#define G 64f           //size of fractal
#define ryT  64u        //threads
#define ryT2 4096u        //threads
#define ryL 16u          //rays loop size
#define sWH (SCREEN_WIDTH*SCREEN_HEIGHT)
#define fti (1<<10)     //float to int, int to float, resolution

struct rayST
{
    p: vec2f,       //ray start position
    d: vec2f,       //ray direction
    cf: f32,        //ray color frequency
    ci: f32,        //ray color intensity
    l: f32,         //ray in SDF
    l2: f32,        //ray in SDF side
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
    //return length(p2) - ra;
}
fn getSDF(p2: vec2f) -> f32
{
    var n = cos(time.elapsed*.05f-vec2f(0,.5)*PI);
    var m = mat2x2f(n,vec2f(-1,1)*n.yx)*2f;
    var a = 99999f;
    var s = 1f;
    var p = p2;
    for(var i=0; i<3; i++)
    {
        p*=m; s*=.5f;
        p = abs(p)-G*1.2f;
        var to = toru(p, 2f, 4f*G/9f, 1f*G/9f)*s;
        a = min(a,to);
    }
    return a;
}
fn getCol(p2: vec2f) -> vec4f
{
    var n = cos(time.elapsed*.05f-vec2f(0,.5)*PI);
    var m = mat2x2f(n,vec2f(-1,1)*n.yx)*2f;
    var a = vec4f(0);
    var s = 1f;
    var p = p2;
    for(var i=0; i<3; i++)
    {
        p*=m; s*=.5f;
        p = abs(p)-G*1.2f;
        var c = cos(f32(i)*1.2f+3f+vec4f(1,2,3,4))
                *vec4f(.5,.5,.5,0)+vec4f(.5,.5,.5,.0);
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
    var r3 = rnd(seed+3u);
    var sf = vec2f(SCREEN_WIDTH,SCREEN_HEIGHT);
    var lp = G*.5f*cos(vec2f(.77,.55)*time.elapsed+vec2f(0,2.22));
    if(mouse.click!=0){lp = G*(2f*vec2f(mouse.pos)-sf)/sf.y;}
    var p  = cos(r0*PI*2f-vec2f(0,.5)*PI)*sqrt(r1)*2f + lp;
    var d  = cos(r2*PI*2f-vec2f(0,.5)*PI);
    var l  = getSDF(p);
    return rayST(
        p,                  //ray start position
        d,                  //ray direction
        r3,                 //ray color frequency
        1f,                 //ray color intensity
        l,                  //ray in SDF
        f32(l>=0f)*2f-1f,   //ray in SDF side
        0f,                 //ray collision
    );
}
fn rayMarch(rayIn:rayST) -> rayST
{
    var ray = rayIn;
    var p1 = ray.p;
    
    for(var i=0u; i<rml && ray.t==0f && ray.l<ote; i++)//raymarch
    {
        ray.p += abs(ray.l)*ray.d;
        var l = getSDF(ray.p)*ray.l2;
        ray.t = f32(l<srf && l<ray.l);
        ray.l = l;
    }
    if(ray.l<srf*.5f)//jump sdf edge
    {
        ray.p += ray.d*srf;
        var l = getSDF(ray.p)*ray.l2;
        ray.t = f32(l<srf && l<ray.l);
        ray.l = l;
    }
    //draw ray
    {
        var cr= cos(ray.cf*5f+1.7f+vec3f(1,2,3));//frequency to color
            cr= max(cr,vec3f(0)) * ray.ci;
        var p2 = ray.p;
        var d  = ray.d;
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
            var c = vec3i(min(l,lt)*cr*f32(fti));
            atomicAdd(&D[w+0u*sWH], c.x);
            atomicAdd(&D[w+1u*sWH], c.y);
            atomicAdd(&D[w+2u*sWH], c.z);
            p  += l*d;
            lt -= l;
        }
    }
    return ray;
}
fn rayBounce(rayIn:rayST, seed:u32) -> rayST
{
    var r0 = rnd(seed+0u)-.1f;
    var r1 = rnd(seed+1u)-.5f;
    var ray = rayIn;
    var nrm = getNrm(ray.p)*ray.l2;
    var col = getCol(ray.p);
    var rfl = reflect(ray.d, nrm);
              var idx = .5f+ray.cf*.5f;
    if(ray.l2<0f){idx = 1f/idx;}
    var k  = 1f - idx * idx * (1f - dot(nrm, ray.d) * dot(nrm, ray.d));
    var ra = idx * ray.d - (idx * dot(nrm, ray.d) + sqrt(k)) * nrm;
    r0 = f32(r0 >= 0f && k >= 0f);
    if(r0!=0f){rfl = ra;}
    if(r0!=0f){ray.l  *= -1f;}
    if(r0!=0f){ray.l2 *= -1f;}
    if(r0!=0f){nrm    *= -1f;}
    var a1 = cos(col.w*r1-vec2f(0,.5)*PI);
    rfl = rfl*a1.x + rfl.yx*vec2f(-1,1)*a1.y;
    rfl -= 2f*nrm*min(dot(rfl,nrm),0f);
    //var dd = dot(rfl,rfl);
    //if(dd==0f){rfl=vec2f(1,0);}
    //if(dd!=dd){rfl=vec2f(1,0);}
    ray.d  = rfl;
    ray.ci*= .9f;
    ray.t  = 0f;
    if(ray.l<srf*.5f)//jump sdf edge
    {
        ray.p += ray.d*srf;
        var l = getSDF(ray.p)*ray.l2;
        ray.t = f32(l<srf && l<ray.l);
        ray.l = l;
    }
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
        if(ray.l>=ote){ray = rayNew(seed);}       seed +=4u;
        if(ray.t!=0f){ray = rayBounce(ray,seed);} seed +=2u;
        ray = rayMarch(ray);
    }
    R[id1] = ray;
}
@compute @workgroup_size(16,16,1)
fn main_image(@builtin(global_invocation_id) id3: vec3u)
{
    var sc = textureDimensions(screen);
    if(id3.x >= sc.x) { return; }
    if(id3.y >= sc.y) { return; }
    var r = id3.y*sc.x + id3.x;
    var c = vec4i(atomicLoad(&D[r+0u*sWH]),
                  atomicLoad(&D[r+1u*sWH]),
                  atomicLoad(&D[r+2u*sWH]),0);
    atomicStore(&D[r+0u*sWH], c.x*1/2);
    atomicStore(&D[r+1u*sWH], c.y*1/2);
    atomicStore(&D[r+2u*sWH], c.z*1/2);
    var d = vec4f(c)/f32(fti)*custom.C*.001f;

    var sf = vec2f(SCREEN_WIDTH,SCREEN_HEIGHT);
    var p  = G*(2f*vec2f(id3.xy)-sf)/sf.y;
    //d += getCol(p);
    //d  = vec4f(getNrm(p),0,0);

    textureStore(screen, id3.xy, d);
}