//speedier path tracing algorthm only works if
//  SDF world
//  light bounce reflections into uniformly sample cone
//  light source is infinitly far away (background image)
//basically a ray shot from camera bounces on SDF
//until it escapes into infinity
//then check this last bounce uniformly sample cone angle
//this angle has to be the biggest possible so cone doesnt collide with SDF
//this angle determines how blured is the background image
//sample the image using mipmap level for blurness
//thats it
#define PI  3.1415926535897932384f
#define rys 2           //rays to fire per frame
#define bnc 5           //light bounces
#define srf .01f        //distance considered surface
#define nrd .01f        //normal calcualtion
#define ote 66.f        //outside fractal space
#define rml 77          //raymarch loop
#define G 128.f          //size of fractal
#define esc .1f         //help ray escape from SDF surface
#define bkX 1024        //X resolution of background img
#storage D array<vec4f,SCREEN_WIDTH*SCREEN_HEIGHT>
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
    var p = p2;
    var rt = cos(-1.6f*.5f*.3f-vec2f(0,.5)*PI);
    var st = cos(-1.6f*.7f*.3f-vec2f(0,.5)*PI);
    var a = 2048.f;
    var s = 1.f;
    for(var i=0; i<9; i++)
    {
        p.x = abs(p.x)-.55f*G;
        p.z = abs(p.z)-.55f*G;
        p = vec3f(p.x*rt.x - p.z*rt.y, p.y,
                  p.x*rt.y + p.z*rt.x );
        p = vec3f(p.x*st.x - p.y*st.y,
                  p.x*st.y + p.y*st.x, p.z );
        p *= 1.8f;
        s *= 1.8f;
        rt*=vec2f(-1,1);
        st*=vec2f(1,-1);
        a = min(a,(length(p+vec3f(0,1.1,0)*G)-1.f*G)/s);
    }
    return a;
}
fn getCol(pos: vec3f) -> vec4f
{
    var p  = pos;
    var px = dot(p,vec3f(2f/G));
    var rt = cos(-1.6f*.5f*.3f-vec2f(0,.5)*PI);
    var st = cos(-1.6f*.7f*.3f-vec2f(0,.5)*PI);
    var s  = 1f;
    var co = vec4f(0);
    for(var i=.5f; i<9f; i+=1f)
    {
        p.x = abs(p.x)-.55f*G;
        p.z = abs(p.z)-.55f*G;
        p = vec3f(p.x*rt.x - p.z*rt.y, p.y,
                  p.x*rt.y + p.z*rt.x );
        p = vec3f(p.x*st.x - p.y*st.y,
                  p.x*st.y + p.y*st.x, p.z );
        p *= 1.8f;
        s *= 1.8f;
        rt*=vec2f(-1,1);
        st*=vec2f(1,-1);
        co += (cos(i*5f+3f-vec4f(1,2,3,4))*vec4f(.25,.25,.25,2)+vec4f(.3,.3,.3,2))*
               max(0f,.5f-3f*abs((length(p+vec3f(0,1.1,0)*G)-1f*G)/s));
    }
    return clamp(co,vec4f(0),vec4f(1,1,1,PI));
    //return vec4f(.4,.4,.4,1.2);
}
fn getNrm(pos: vec3f) -> vec3f
{
    return normalize(vec3f(objec(pos+nrd*vec3f(1,0,0)) - objec(pos-nrd*vec3f(1,0,0)),
                           objec(pos+nrd*vec3f(0,1,0)) - objec(pos-nrd*vec3f(0,1,0)),
                           objec(pos+nrd*vec3f(0,0,1)) - objec(pos-nrd*vec3f(0,0,1))));
}
fn vecRnd(v:vec3f, ang:f32, seed:u32) -> vec3f
{
    var r1 = rnd(seed+0u);
    var r2 = rnd(seed+1u);
    var a0 = acos(r1*(.5f-.5f*cos(ang))*-2f+1f);
    var a1 = cos(r2*2f*PI-vec2f(0,.5)*PI);
    var a2 = cos(a0      -vec2f(0,.5)*PI);
    var x  = normalize(vec3f(1)-dot(v,vec3f(1))*v);
    var y  = cross(x,v);
    var xy = x*a1.x + y*a1.y;
    return v*a2.x + xy*a2.y;
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
        var s = objec(camPos)*.05f; //*time.elapsed;
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
    var seed = (time.frame*SCREEN_WIDTH*SCREEN_HEIGHT + id1)*u32(rys*(2+bnc*2+2));
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
        var l  = 0f;
        var l2 = .001f;
        var ma = 0f;//minimun angle
        for(var i = 0; i < bnc; i++)
        {
            for(var j=0; j<rml; j++)
            {
                l = objec(p);
                ma = min(ma,asin((l-srf)/l2));
                if(l<srf || l>ote){break;}
                p += l*ray;
                l2+= l;
            }   if(l>ote){break;}
            var nrm = getNrm(p);
            var col = getCol(p);
            m *= col.xyz;
            p += ray*(l-srf);
            {
                ray = reflect(ray, nrm);

                var r1 = rnd(seed);    seed+=1u;
                var r2 = rnd(seed);    seed+=1u;
                var a0 = acos(r1*(.5f-.5f*cos(col.w))*-2f+1f);
                var a1 = cos(r2*2f*PI-vec2f(0,.5)*PI);
                var a2 = cos(a0      -vec2f(0,.5)*PI);
                var x  = normalize(vec3f(1)-dot(ray,vec3f(1))*ray);
                var y  = cross(x,ray);
                var xy = x*a1.x + y*a1.y;
                ray = ray*a2.x + xy*a2.y;

                ray = ray - 2f*nrm*min(dot(ray,nrm),0f);
                ma = col.w-a0;
            }
            p += ray*esc;
            l2 = esc;
        }
        const angMin = PI*2f/f32(bkX);//background image minimun angle
        var lv = max(0f,log2(ma/angMin));
        c += m*bkgr(ray,lv)*f32(l>ote);
    }
    var d = D[id1]*rstImg + vec4f(c,f32(rys))*(1f/512f);
    D[id1] = d;
    textureStore(screen, id.xy, d/d.w*custom.C);
}