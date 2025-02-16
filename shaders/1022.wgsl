#define PI  3.1415926535897932384f
#define rys 4           //rays to fire per frame
#define bnc 4           //light bounces
#define srf .01f         //distance considered surface
#define nrd .01f        //normal calcualtion
#define ote 66.f       //outside fractal space
#define rml 77          //raymarch loop
#define G 64.f         //size of fractal
#define spl custom.A         //original sphere radius
#storage D array<vec4f,SCREEN_HEIGHT*SCREEN_WIDTH>
fn bkgr(d: vec3f) -> vec3f
{
    var uv = vec2(atan2(d.z, d.x), asin(-d.y));
        uv = uv * vec2(1.f / (2.f*PI), 1.f / PI) + .5f;
    return textureSampleLevel(channel0, bilinear, uv, 0).xyz;
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
fn gaus(a: u32) -> f32
{
    var r1 = 1.f-rnd(a   );
    var r2 =     rnd(a+1u);
    return sqrt(-2.f*log(r1))*cos(2.f*PI*r2);
}
fn pntSphr(i:f32, iz:f32) -> vec3f
{
    var a = i*1.61803398875f*2f*PI;
    var b = (i+.5f)/iz*2f-1f;
    var c = sqrt(1f - b*b);
    return vec3f(cos(a)*c, sin(a)*c, b);
}
fn objec(p2: vec3f) -> f32
{
    var rt = cos(.25*PI*2.-vec2f(0,.5)*PI);
    var st = sin(.42*PI*2.-vec2f(0,.5)*PI);
    var rt3 = mat3x3<f32>( rt.x, rt.y, 0,
                          -rt.y, rt.x, 0,
                              0,    0, 1);
    var st3 = mat3x3<f32>( 1, 0, 0,
                           0,  st.x, st.y,
                           0, -st.y, st.x);
    var m2 = mat3x3<f32>(.98058,.19611,0,
                         .19611,-.98058,0,
                         0,0,1);
    var p = p2;
    var q = p;
    var a = 999999.f;
    var s = 1.f;
    for(var i=0; i<9; i++)
    {
        p = abs(p);
        p = rt3*p;
        p+=min(dot(p,vec3f(1,-1,0)),0.)*vec3f(-1, 1,0);
        p+=min(dot(p,vec3f(1,0,-1)),0.)*vec3f(-1, 0,1);
        p+=min(dot(p,vec3f(0,1,-1)),0.)*vec3f( 0,-1,1);
        p = st3*p;
        p *= 1.92f;
        s *= 1.92f;
        p += vec3f(-3.44,-.69,-1.14)*G;
        a = min(a,(length(max(abs(p+vec3f(0,.1,1.3)*G)-vec3f(8,0,0)*G,vec3f(0)))-.1f*G)/s);
    }
    a = min(a,(length(p)-2.5f*G)/s);
    return a;
}
fn getCol(pos: vec3f) -> vec4f
{
    var rt = cos(.25*PI*2.-vec2f(0,.5)*PI);
    var st = sin(.42*PI*2.-vec2f(0,.5)*PI);
    var rt3 = mat3x3<f32>( rt.x, rt.y, 0,
                          -rt.y, rt.x, 0,
                              0,    0, 1);
    var st3 = mat3x3<f32>( 1, 0, 0,
                           0,  st.x, st.y,
                           0, -st.y, st.x);
    var m2 = mat3x3<f32>(.98058,.19611,0,
                         .19611,-.98058,0,
                         0,0,1);
    var p   = pos;
    var px  = dot(p,vec3f(2.f/G));
    var s   = 1.f;
    var col = vec4f(0);
    for(var i=0; i<9; i++)
    {
        p = abs(p);
        p = rt3*p;
        p+=min(dot(p,vec3f(1,-1,0)),0.)*vec3f(-1, 1,0);
        p+=min(dot(p,vec3f(1,0,-1)),0.)*vec3f(-1, 0,1);
        p+=min(dot(p,vec3f(0,1,-1)),0.)*vec3f( 0,-1,1);
        p = st3*p;
        p *= 1.92f;
        s *= 1.92f;
        p += vec3f(-3.44,-.69,-1.14)*G;
        var mc = cos(f32(i)*20.f+vec4f(1,2,3,6))*vec4f(.2,.2,.2,.0)
                +vec4f(.6,.6,.6,pow((1f-f32(i)/8f)*.13f,2f));
        var a = (length(max(abs(p+vec3f(0,.1,1.3)*G)-vec3f(8,0,0)*G,vec3f(0)))-.1f*G)/s;
        col += max(1f-a*2f,0f)*222f*mc;
    }
    return clamp(col*vec4f(vec3f(1./G*.2),1),vec4f(0),vec4f(1));
}
fn getNrm(pos: vec3f) -> vec3f
{
    return normalize(vec3f(objec(pos+nrd*vec3f(1,0,0)) - objec(pos-nrd*vec3f(1,0,0)),
                           objec(pos+nrd*vec3f(0,1,0)) - objec(pos-nrd*vec3f(0,1,0)),
                           objec(pos+nrd*vec3f(0,0,1)) - objec(pos-nrd*vec3f(0,0,1))));
}
fn rayBnc(inc:vec3f, nrm:vec3f, met:f32, seed:u32) -> vec3f
{
    var v1 = cos(rnd(seed+0u)*2f*PI    -vec2f(0,.5)*PI);
    var v2 = cos(rnd(seed+1u)*1f*PI*met-vec2f(0,.5)*PI);
    var rf = reflect(inc,nrm);
    var x  = normalize(nrm-dot(rf,nrm)*rf);
    var y  = cross(x,rf);
    var xy = x*v1.x + y*v1.y;
    rf = rf*v2.x + xy*v2.y;
    rf = rf - 2f*nrm*min(dot(rf,nrm),0f);
    return rf;
}
#storage camo array<vec3f,4>;
#workgroup_count camera 1 1 1
@compute @workgroup_size(1,1,1)
fn camera(@builtin(global_invocation_id) id: uint3)
{
    var keyW = f32(keyDown(87u));
    var keyS = f32(keyDown(83u));
    var keyA = f32(keyDown(65u));
    var keyD = f32(keyDown(68u));
    var camDir = camo[0];  if(time.frame==0u){camDir = vec3f(0,0,1);}
    var camPos = camo[1];  if(time.frame==0u){camPos = vec3f(0,44,-88);}
    var mouPre = camo[2];
    var mouNow = vec3f(vec2f(mouse.pos),f32(mouse.click));
    var mouDif = mouNow-mouPre;
    let rs    = vec2f(textureDimensions(screen));
    //mod camDir data
    {
        var cd = camDir;
        var m = mouPre.z*mouDif*-4.f/rs.y;
        var y = asin(camDir.y) + m.y;
        if(-1.56f > y){y =-1.56f;};
        if( 1.56f < y){y = 1.56f;};
        var a = cos(m.x-vec2f(0,PI*.5f));
        var n = cos(y)*normalize(vec2f(cd.x*a.x - cd.z*a.y,
                                        cd.x*a.y + cd.z*a.x));
        if(any(m.xy!=vec2f(0))){cd = vec3f(n.x,sin(y),n.y);}
        camo[0] = cd;
    }
    //mod camPos data
    {
        var s = objec(camPos)*.04f; //*time.elapsed;
        camo[1] = camPos + (keyW-keyS)*s*camDir +
                           (keyD-keyA)*s*normalize(vec3f(camDir.z,0,-camDir.x));
    }
    //mod mouse data
    {
        camo[2] = vec3f(mouNow.x,mouNow.y,f32(mouNow.z!=0.f && (any(mouDif.xy!=vec2f(0)) || mouPre.z!=0.f)));
    }
    camo[3].x = f32((keyW+keyS+keyA+keyD + f32(mouse.click!=0)) == 0f);
}
@compute @workgroup_size(8,8,1)
fn main_image(@builtin(global_invocation_id) id: uint3)
{
    var screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x) { return; }
    if (id.y >= screen_size.y) { return; }
    var id1 = id.y*SCREEN_WIDTH + id.x;
    var seed = (time.frame*SCREEN_WIDTH*SCREEN_HEIGHT + id1)*u32(rys*(2+bnc*2));
    var rs = vec2f(screen_size);
    var camDir = camo[0];
    var camPos = camo[1];
    var mtx0   = normalize(vec3f(camDir.z,0,-camDir.x));
    var mtx    = mat3x3<f32>(mtx0, cross(camDir,mtx0), camDir);
    var rstImg = camo[3].x;
    var c = vec3f(0);

    for(var k = 0; k < rys; k++)
    {
        var r2 = vec2f(rnd(seed+0u),
                    rnd(seed+1u));  seed+=2u;
        var uv = vec2f(id.xy)+r2;
            uv = vec2f(1,-1)*(2.f*uv-rs.xy)/rs.y;
        var ray = mtx*normalize(vec3f(uv,1)); //direction of ray from camera
        var p = camPos;
        var m = vec3f(1);
        var o  = 0f;
        for(var i = 0; i < bnc; i++)
        {
            var l2 = 0f;
            for(var j=0; j<rml; j++)
            {
                var l = objec(p);
                o = f32(l<srf)-f32(l>ote);
                if(o!=0f){break;}
                p += l*ray;
                l2+= l;
            }   if(o<0f){break;}
            var col = getCol(p);
            m *= col.xyz;
            ray = rayBnc(ray, getNrm(p), col.w, seed);  seed+=2u;
            p+=ray*.2f;
        }
        c += m*bkgr(ray)*f32(o<0f);
    }
    var d = D[id1]*rstImg + vec4f(c,f32(rys))*(1f/512f);
    D[id1] = d;
    textureStore(screen, id.xy, d/d.w*custom.C);
}