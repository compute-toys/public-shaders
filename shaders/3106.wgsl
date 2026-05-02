//previous try was not assigning a weight to every ray
//depending on ray cone angle whe it hits the sky
#define PI  3.1415926535897932384f
#define ryl 3           //ray loops
#define srf .01f        //distance considered surface
#define nrd .001f        //normal calcualtion
#define ote 111f        //outside fractal space
#define rml 33          //raymarch loop
#define G   66f          //size of fractal
#define sWH (SCREEN_WIDTH*SCREEN_HEIGHT)
#storage D1 array<rayST1,sWH>
#storage D2 data2
struct rayST1
{
    p: vec3f,   //ray start position
    d: vec3f,   //ray direction
    c: vec3f,   //ray color
};
struct rayST2
{
    x: vec4f,   //screen color
    l: f32,     //ray.p in SDF
    t: f32,     //ray length traveled
    a: f32,     //ray min cone angle
    o: f32,     //ray ini cone angle
};
struct rayST
{
    p: vec3f,   //ray start position
    d: vec3f,   //ray direction
    c: vec3f,   //ray color
    x: vec4f,   //.xyz pixel color .w pixel importance
    l: f32,     //ray.p in SDF
    t: f32,     //ray length traveled
    a: f32,     //ray min cone angle
    o: f32,     //ray ini cone angle
};
struct data2
{
    ray: array<rayST2,sWH>,
    camDir: vec3f,
    camPos: vec3f,
    mouPre: vec3f,
    xclear: f32,  //clear screen
};
fn getray(id1: u32) -> rayST
{
    var r1 = D1[id1];
    var r2 = D2.ray[id1];
    return rayST(
        r1.p,
        r1.d,
        r1.c,
        r2.x,
        r2.l,
        r2.t,
        r2.a,
        r2.o
    );
}
fn setray(id1: u32, ray: rayST)
{
    D1[id1].p = ray.p;
    D1[id1].d = ray.d;
    D1[id1].c = ray.c;
    D2.ray[id1].x = ray.x;
    D2.ray[id1].l = ray.l;
    D2.ray[id1].t = ray.t;
    D2.ray[id1].a = ray.a;
    D2.ray[id1].o = ray.o;
}
fn bkgr(d: vec3f, lvl:f32) -> vec3f
{
    var uv = vec2(atan2(d.z, d.x), asin(-d.y));
        uv = uv * vec2(1f/(2f*PI), 1f/PI) + .5f;
    return textureSampleLevel(channel0, bilinear, uv, lvl).xyz;
}
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
fn getSDF(p2: vec3f) -> f32
{
    var p = p2;
    var rt = cos(custom.fractala*PI*2f-vec2f(0,.5)*PI);
    var st = cos(custom.fractalb*PI*2f-vec2f(0,.5)*PI);
    var m  = mat3x3f(rt.x,0,rt.y, 0,1,0, -rt.y,0,rt.x)*
             mat3x3f(st.x,st.y,0, -st.y,st.x,0, 0,0,1)*1.8f;
    var a = 2048.f;
    var s = 1.f;
    for(var i=0; i<9; i++)
    {
        p.x = abs(p.x)-.55f*G;
        p.z = abs(p.z)-1.1f*G;
        p *= m;
        s *= 1.8f;
        a = min(a,(length(p+vec3f(0,1.1,0)*G)-1.f*G)/s);
    }
    return a;
    //return length(p2)-11f;
}
fn getCol(pos: vec3f) -> vec4f
{
    var p  = pos;
    var px = dot(p,vec3f(2f/G));
    var rt = cos(custom.fractala*PI*2f-vec2f(0,.5)*PI);
    var st = cos(custom.fractalb*PI*2f-vec2f(0,.5)*PI);
    var m  = mat3x3f(rt.x,0,rt.y, 0,1,0, -rt.y,0,rt.x)*
             mat3x3f(st.x,st.y,0, -st.y,st.x,0, 0,0,1)*1.8f;
    var s  = 1f;
    var co = vec4f(0);
    for(var i=.5f; i<9f; i+=1f)
    {
        p.x = abs(p.x)-.55f*G;
        p.z = abs(p.z)-1.1f*G;
        p *= m;
        s *= 1.8f;
        co += (cos(i*1f+2f-vec4f(1,2,3,4))*vec4f(.2,.2,.2,1)+vec4f(.4,.4,.4,1))*
               max(0f,1f-4f*abs((length(p+vec3f(0,1.1,0)*G)-1f*G)/s));
    }
    return clamp(co,vec4f(0),vec4f(1,1,1,PI*.5*.99));
    //return vec4f(.5,.5,.5,.3);
}
fn getNrm(pos: vec3f) -> vec3f
{
    return normalize(vec3f(getSDF(pos+nrd*vec3f(1,0,0)) - getSDF(pos-nrd*vec3f(1,0,0)),
                           getSDF(pos+nrd*vec3f(0,1,0)) - getSDF(pos-nrd*vec3f(0,1,0)),
                           getSDF(pos+nrd*vec3f(0,0,1)) - getSDF(pos-nrd*vec3f(0,0,1))));
}
fn rayMarch(rayIn:rayST, seed:u32, angmin:f32, id2f:vec2f, camPos:vec3f, camMtx:mat3x3f) -> rayST
{
    var ray = rayIn;
    var p1  = ray.p;
    var bonce = false;
    for(var i=0u; i<rml && !bonce && ray.l<ote; i++)//raymarch
    {
        var l = getSDF(ray.p);
        var l2 = abs(l);
        if(abs(l) < srf){l2 = srf*.5f;}
        ray.p += l2*ray.d;
        ray.t += l2;
        ray.a = min(ray.a, l2/ray.t);
        bonce = abs(ray.l) >= srf && abs(l) < srf;
        ray.l = l;
    }
    if(ray.l>=ote)//paint screen
    {
        var an1 = atan(ray.a);
        var an2 = atan(ray.o);
        var lvl = max(0f,floor(log2(an1*angmin)-1.5f));
        var col = bkgr(ray.d, lvl);
        var imp = (1f-cos(an1))/(1f-cos(an2));  if(an2==0f){imp = .00001f;}
        ray.x += vec4f(col*ray.c, 1f)*imp;
    }
    if(ray.l>=ote || dot(ray.c,ray.c)<.002f*.002f)//ray new
    {
        var rs = vec2f(SCREEN_WIDTH,SCREEN_HEIGHT);
        var uv = id2f+vec2f(rnd(seed+0u),
                            rnd(seed+1u));
            uv = vec2f(1,-1)*(2f*uv-rs.xy)/rs.y;
        ray.p = camPos;
        ray.d = camMtx*normalize(vec3f(uv,1));
        ray.c = vec3f(1);
        ray.l = getSDF(ray.p);
        ray.t = 0f;
        ray.a = 1f/angmin;
        ray.o = ray.a;
        bonce = false;
    }
    if(bonce)//ray bounce
    {
        var nrm = getNrm(ray.p);
        var col = getCol(ray.p);
        var rdy = dot(ray.d, nrm);
        var rdx = ray.d - rdy*nrm;
        var rdxl= length(rdx);  if(rdxl != 0f){rdxl = 1f/rdxl;}
        var ax  = max(col.w , PI*.5f-acos(abs(rdy)));
        var nd2 = cos(ax-vec2f(0,.5)*PI);
        var rfl = nd2.x*rdx*rdxl + nd2.y*nrm;
        var rfx = cross(rfl,nrm);
        var rfy = cross(rfl,rfx);
        var rfxl= length(rfx);  if(rfxl != 0f){rfxl = 1f/rfxl;}
        var rfyl= length(rfy);  if(rfyl != 0f){rfyl = 1f/rfyl;}
        var r1  = rnd(seed+2u)*col.w;
        var r2  = rnd(seed+3u);
        var a1  = cos(r1*.5f*PI-vec2f(0,.5)*PI);
        var a2  = cos(r2* 2f*PI-vec2f(0,.5)*PI)*a1.y;
        ray.d = a2.x*rfx*rfxl + a2.y*rfy*rfyl + a1.x*rfl;
        ray.c *= col.xyz;
        ray.t = 0f;
        ray.a = tan(col.w);
        ray.o = ray.a;
    }
    return ray;
}
#workgroup_count camera 1 1 1
@compute @workgroup_size(1,1,1)
fn camera(@builtin(global_invocation_id) id: uint3)
{
    var keyW = f32(keyDown(87u));
    var keyS = f32(keyDown(83u));
    var keyA = f32(keyDown(65u));
    var keyD = f32(keyDown(68u));
    var camDir = D2.camDir;  if(time.frame==0u){camDir = vec3f(0,0,-1);}
    var camPos = D2.camPos;  if(time.frame==0u){camPos = vec3f(0,0,190);}
    var mouPre = D2.mouPre;
    var mouNow = vec3f(vec2f(mouse.pos),f32(mouse.click));
    var mouDif = mouNow-mouPre;
    let rs    = vec2f(SCREEN_WIDTH,SCREEN_HEIGHT);
    //mod camDir data
    {
        var d = camDir;
        var m = mouPre.z*mouDif*-4.f/rs.y;
        var y = asin(d.y) + m.y;
        if(-1.56f > y){y =-1.56f;};
        if( 1.56f < y){y = 1.56f;};
        var a = cos(m.x-vec2f(0,PI*.5f));
        var n = cos(y)*normalize(vec2f(d.x*a.x - d.z*a.y,
                                       d.x*a.y + d.z*a.x));
        if(any(m.xy!=vec2f(0))){d = vec3f(n.x,sin(y),n.y);}
        D2.camDir = d;
    }
    //mod camPos data
    {
        var s = getSDF(camPos)*.05f; //*time.elapsed;
        D2.camPos = camPos + (keyW-keyS)*s*camDir +
                             (keyD-keyA)*s*normalize(vec3f(camDir.z,0,-camDir.x));
    }
    //mod mouse data
    {
        D2.mouPre = vec3f(
            mouNow.xy,
            f32(mouNow.z!=0.f && (any(mouDif.xy!=vec2f(0)) || mouPre.z!=0.f))
        );
    }
    D2.xclear = f32((keyW+keyS+keyA+keyD + f32(mouse.click!=0)) == 0f);
}
@compute @workgroup_size(8,8,1)
fn main_image(@builtin(global_invocation_id) id3: uint3)
{
    if(id3.x >= SCREEN_WIDTH ){return;}
    if(id3.y >= SCREEN_HEIGHT){return;}
    var id1  = id3.x + id3.y*SCREEN_WIDTH;
    var id2f = vec2f(id3.xy);
    var seep = 4u; //seeds per pixel
    var seed = (time.frame*sWH + id1)*u32(ryl*seep);
    var camPos = D2.camPos;
    var camDir = D2.camDir;
    var mtx0   = normalize(vec3f(camDir.z,0,-camDir.x));
    var camMtx = mat3x3f(mtx0, cross(camDir,mtx0), camDir);
    var imgr   = vec2f(textureDimensions(channel0));
    var angmin = imgr.x/PI; //background image minimun angle, yes PI not 2PI
    var ray    = getray(id1);
    if(time.frame==0u || D2.xclear != 1f){ray = rayST();}
    for(var k = 0u; k < ryl; k++)
    {
        ray = rayMarch(ray, seed + k*seep, angmin, id2f, camPos, camMtx);
    }
    textureStore(screen, id3.xy, ray.x/ray.x.w*custom.bright);
    setray(id1, ray);
}