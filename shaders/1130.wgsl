#define PI  3.1415926535897932384f
#define srf .01f        //distance considered surface
#define nrd (srf*.1f)   //normal calcualtion
#define ote 88.f        //outside fractal space
#define rml 32u         //raymarch loop
#define G 64f           //size of fractal
#define ryT  32u        //threads
#define ryT2 32768u        //threads
#define ryL 1u          //rays loop size
#define sWH (SCREEN_WIDTH*SCREEN_HEIGHT)
#define fti (1<<10)     //float to int, int to float, resolution

struct rayST
{
    p: vec2f,       //ray start position
    d: vec2f,       //ray direction
    cf: f32,        //ray color frequency
    ci: f32,        //ray color intensity
};
#storage D array<atomic<i32>,SCREEN_WIDTH*SCREEN_HEIGHT*3>
#storage R array<rayST,ryT*ryT2>

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
    var n = cos(time.elapsed*.0f+1.1f-vec2f(0,.5)*PI);
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
fn rayMarch(rayIn:rayST, seed:u32) -> rayST
{
    var ray = rayIn;
    var p1  = ray.p;
    var rl  = abs(getSDF(p1));
    for(var i=0u; i<rml && rl<srf; i++)//jump sdf edge
    {
        ray.p += ray.d*srf;
        rl = abs(getSDF(ray.p));
    }
    var lstl  = 0f;
    var bonce = false;
    for(var i=0u; i<rml && rl>=srf && rl<ote; i++)//raymarch
    {
        ray.p += rl*ray.d;
        lstl  = getSDF(ray.p);
        var l = abs(lstl);
        bonce = l<srf && l<rl;
        rl = l;
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
        if(all(abs(p1)<sm*1.0001f) &&
           all(abs(p2)<sm*1.0001f)){lt=l0;}
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
    if(rl>=ote || ray.ci<.002f)//ray new
    {
        var r0 = rnd(seed+0u);
        var r1 = rnd(seed+1u);
        var r2 = rnd(seed+2u);
        var r3 = rnd(seed+3u);
        var sf = vec2f(SCREEN_WIDTH,SCREEN_HEIGHT);
        var lp = G*.5f*cos(vec2f(.77,.55)*.2f*time.elapsed+vec2f(0,2.22));
        if(mouse.click!=0){lp = G*(2f*vec2f(mouse.pos)-sf)/sf.y;}
        var p  = cos(r0*PI*2f-vec2f(0,.5)*PI)*sqrt(r1)*2f + lp;
        var d  = cos(r2*PI*2f-vec2f(0,.5)*PI);
        ray.p  = p;
        ray.d  = d;
        ray.cf = r3;
        ray.ci = 1f;
        bonce = false;
    }
    if(bonce)//ray bounce
    {
        var r0 = rnd(seed+4u)-.1f;
        var r1 = rnd(seed+5u)-.5f;
        var nrm = getNrm(ray.p); if(lstl<0f){nrm = -nrm;}
        var col = vec4f(.0);//getCol(ray.p);
        var rfl = reflect(ray.d, nrm);
                var idx = .5f+ray.cf*.5f;
        if(lstl<0f){idx = 1f/idx;}
        var k  = 1f - idx * idx * (1f - dot(nrm, ray.d) * dot(nrm, ray.d));
        var ra = idx * ray.d - (idx * dot(nrm, ray.d) + sqrt(k)) * nrm;
        r0 = f32(r0 >= 0f && k >= 0f);
        if(r0!=0f){rfl = ra;}
        if(r0!=0f){nrm    *= -1f;}
        var a1 = cos(col.w*r1-vec2f(0,.5)*PI);
        rfl = rfl*a1.x + rfl.yx*vec2f(-1,1)*a1.y;
        rfl -= 2f*nrm*min(dot(rfl,nrm),0f);
        ray.d  = rfl;
        ray.ci*= .9f;
    }
    return ray;
}
#dispatch_once rayIni
#workgroup_count rayIni ryT2 1 1
@compute @workgroup_size(ryT,1,1)
fn rayIni(@builtin(global_invocation_id) id3: vec3u)
{
    R[id3.x] = rayST();
}
#workgroup_count rayDo ryT2 1 1
@compute @workgroup_size(ryT,1,1)
fn rayDo(@builtin(global_invocation_id) id3: vec3u)
{
    var sc = textureDimensions(screen);
    var id1  = id3.x;
    var seed = (time.frame*ryT*ryT2 + id1)*6u*ryL+756352967u;
    var ray  = R[id1];
    for(var k = 0u; k < ryL; k++)
    {
        ray = rayMarch(ray,seed + k*6u);
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
    var d = vec4f(c)/f32(fti)*custom.a*.002f;

    var sf = vec2f(SCREEN_WIDTH,SCREEN_HEIGHT);
    var p  = G*(2f*vec2f(id3.xy)-sf)/sf.y;
    //d += getCol(p);
    //d  = vec4f(getNrm(p),0,0);

    textureStore(screen, id3.xy, d);
}