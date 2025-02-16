//speedier path tracing than try3
//uses same algo as try3 but also
//uses alias method to sample from background image O(1)
//click [reset] after channel0 loads or changes
//then wait 3sec for the alias method to complete
#define PI  3.1415926535897932384f
#define rys 2           //rays to fire per frame
#define bnc 5           //light bounces
#define srf .01f        //distance considered surface
#define nrd .01f        //normal calcualtion
#define ote 90.f        //outside fractal space
#define rml 80          //raymarch loop
#define G 64.f          //size of fractal
#define esc .125f         //help ray escape from SDF surface
#define bkT 256         //threads
#define sWH (SCREEN_WIDTH*SCREEN_HEIGHT)
#define Z1 sWH
#define Z2 (1<<21)      //bigger than channel0
#storage D array<vec4f,Z1+Z2+1>
struct rayST
{
    p: vec3f,   //ray start position
    d: vec3f,   //ray direction
    c: vec3f,   //ray color
    w: f32,     //ray color weight of probability
    l: f32,     //ray end in SDF
    a: f32,     //ray cone angle
    b: f32,     //ray bounces
};
fn bkgr(d: vec3f, lvl:f32) -> vec3f
{
    var uv = vec2(atan2(d.z, d.x), asin(-d.y));
        uv = uv * vec2(1f/(2f*PI), 1f/PI) + .5f;
    return textureSampleLevel(channel0, bilinear, uv, lvl).xyz;
}
fn bkgr2(uv: vec2f) -> vec3f
{
    return textureSampleLevel(channel0, nearest, uv, 0f).xyz;
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
fn objec(p2: vec3f) -> f32
{
    var p = p2;
    var rt = cos(-1.6f*.5f*.5f-vec2f(0,.5)*PI);
    var st = cos(-1.6f*.8f*.5f-vec2f(0,.5)*PI);
    var a = 2048.f;
    var s = 1.f;
    for(var i=0; i<9; i++)
    {
        p.x = abs(p.x)-.55f*G;
        p.z = abs(p.z)-1.1f*G;
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
    var rt = cos(-1.6f*.5f*.5f-vec2f(0,.5)*PI);
    var st = cos(-1.6f*.8f*.5f-vec2f(0,.5)*PI);
    var s  = 1f;
    var co = vec4f(0);
    for(var i=.5f; i<9f; i+=1f)
    {
        p.x = abs(p.x)-.55f*G;
        p.z = abs(p.z)-1.1f*G;
        p = vec3f(p.x*rt.x - p.z*rt.y, p.y,
                  p.x*rt.y + p.z*rt.x );
        p = vec3f(p.x*st.x - p.y*st.y,
                  p.x*st.y + p.y*st.x, p.z );
        p *= 1.8f;
        s *= 1.8f;
        rt*=vec2f(-1,1);
        st*=vec2f(1,-1);
        co += (cos(i*5f+3f-vec4f(1,2,3,4))*vec4f(.3,.3,.3,2)+vec4f(.5,.5,.5,2))*
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
fn rayNew(id2f:vec2f,camPos:vec3f,mtx:mat3x3f,seed:u32) -> rayST
{
    var rs = vec2f(SCREEN_WIDTH,SCREEN_HEIGHT);
    var r2 = vec2f(rnd(seed+0u),
                   rnd(seed+1u));
    var uv = id2f+r2;
        uv = vec2f(1,-1)*(2f*uv-rs.xy)/rs.y;
    return rayST(
        camPos,     //ray start position
        mtx*normalize(vec3f(uv,1)),   //ray direction
        vec3f(1),   //ray color
        1f,         //ray color weight of probability
        0f,         //ray end in SDF
        0f,         //ray cone angle
        0f,         //ray bounces
    );
}
fn rayMarch(rayIn:rayST) -> rayST
{
    var ray = rayIn;
    ray.p += ray.d*esc;
    var l2 = esc;
    for(var i=0; i<rml; i++)
    {
        ray.l = objec(ray.p);
        ray.a = min(ray.a,asin((ray.l-srf)/l2));
        if(ray.l<srf || ray.l>ote){break;}
        ray.p += ray.l*ray.d;
        l2 += ray.l;
    }
    return ray;
}
fn rayBounce(rayIn:rayST, seed:u32) -> rayST
{
    var ray = rayIn;
    ray.p += ray.d*(ray.l-srf);
    var nrm = getNrm(ray.p);
    var col = getCol(ray.p);
    var rfl = reflect(ray.d, nrm);
    var r1 = rnd(seed+0u);
    var r2 = rnd(seed+1u);
    var a0 = acos(r1*(.5f-.5f*cos(col.w))*-2f+1f);
    var a1 = cos(r2*2f*PI-vec2f(0,.5)*PI);
    var a2 = cos(a0      -vec2f(0,.5)*PI);
    var x  = normalize(vec3f(1)-dot(rfl,vec3f(1))*rfl);
    var y  = cross(x,rfl);
    var xy = x*a1.x + y*a1.y;
    rfl = rfl*a2.x + xy*a2.y;
    ray.d = rfl - 2f*nrm*min(dot(rfl,nrm),0f);
    ray.a = col.w-a0;
    ray.c *= col.xyz;
    ray.b += 1f;
    return ray;
}
fn rayBounce2(rayIn:rayST, seed:u32, res:vec2f, s2:vec3f) -> rayST
{
    var ray = rayIn;
    ray.p += ray.d*(ray.l-srf);
    var nrm = getNrm(ray.p);
    var col = getCol(ray.p);
    var rfl = reflect(ray.d, nrm);
    var r1 = rnd(seed+0u);
    var r2 = rnd(seed+1u);
    var r3 = rnd(seed+2u);
    var r4 = rnd(seed+3u);
    var d2 = D[Z1+u32(r1*f32(Z2))];
    var         pn = d2.z;
    if(r2>d2.y){pn = d2.w;}
    pn = (pn+r3)*res.x;
    var bku = vec2f(fract(pn),(floor(pn)+r4)*res.y);  //background uv 2D
    var bka = (bku-.5f)*vec2(2f*PI, PI);
    var nxz = cos(bka.x-vec2f(0,.5)*PI)*cos(bka.y);
    ray.d = vec3f(nxz.x,-sin(bka.y),nxz.y); //background direction 3D
    ray.a =  col.w-acos(dot(rfl,ray.d));
    ray.c *= col.xyz;
    ray.b += 1f;
    ray.w = f32(ray.a>=0f)/dot(bkgr2(bku),s2);
    return ray;
}
var<workgroup> S1: array<f32,bkT>;
var<workgroup> S2: array<f32,bkT>;
#dispatch_once iniD
#workgroup_count iniD 1 1 1
@compute @workgroup_size(bkT,1,1)
fn iniD(@builtin(global_invocation_id) id3: uint3)
{
    //compute sums of all pixels
    var res = textureDimensions(channel0);
    var pxt = res.x*res.y;
    var id  = id3.x;
    var rd  = id;
    var s1  = 0f;
    var s2  = 0f;
    while(rd < pxt)
    {
        var x = rd % res.x;
        var y = rd / res.x;
        var u = (vec2f(f32(x),f32(y))+.5f)/vec2f(res);
        var tx= textureSampleLevel(channel0, nearest, u, 0f);
            //tx= tx*vec4f(1,1,1,0);
        var t = dot(tx,vec4f(1,1,1,0));//length(tx);
        var ar = sin((f32(y)+.5f)/f32(res.y)*PI);//area of pixel in sphere
        D[Z1 + rd] = vec4f(t*ar,t*ar,f32(rd),0);
        s1 += t*ar;
        s2 += t;
        rd += bkT;
    }
    for(var i = bkT/2u; i > 0u; i /= 2u)
    {
        S1[id] = s1;
        S2[id] = s2;
        workgroupBarrier();
        var r = (id+i) % bkT;
        s1 += S1[r];
        s2 += S2[r];
        workgroupBarrier();
    }
    s1 = f32(pxt)/s1;
    s2 = f32(pxt)/s2;
    //normalize
    var m = vec4f(s1,s1,1,1);
    rd = id;
    while(rd < pxt)
    {
        D[Z1 + rd] *= m;
        rd += bkT;
    }
    if(id==0u){D[Z1+Z2+0] = vec4(0,0,0,s2);}
}
#workgroup_count iniD2 1 1 1
@compute @workgroup_size(1,1,1)
fn iniD2(@builtin(global_invocation_id) id3: uint3)
{
    //build table
    var res = textureDimensions(channel0);
    var pxt = res.x*res.y;
    var d0 = D[Z1+Z2+0];
    var rB = u32(d0.x);//pos of value bigger than 1
    var rL = u32(d0.y);//pos of value lower than 1
    var lop = 0;//just compute by parts, instead of all at once
    while(lop<(1<<13) && rB < pxt && rL < pxt)
    {
        lop++;
        var dB = D[Z1 + rB];
        var dL = D[Z1 + rL];
        if(dB.x>1f && dL.x<1f)
        {
            var sb  = dB.x-(1f-dL.x);
            var dB2 = vec4f(sb,  sb,dB.z,   0);
            var dL2 = vec4f(1f,dL.y,dL.z,dB.z);
            D[Z1 + rB] = dB2;
            D[Z1 + rL] = dL2;
        }
        dB = D[Z1 + rB];
        dL = D[Z1 + rL];
        if((rB > rL && dB.x==1f)|| 
        (rL > rB && dL.x==1f))
        {
            D[Z1 + rB] = dL;
            D[Z1 + rL] = dB;
        }
        dB = D[Z1 + rB];
        dL = D[Z1 + rL];
        if(dB.x<=1f && !(rB > rL && dB.x==1f)){rB+=1u;}
        if(dL.x>=1f && !(rL > rB && dL.x==1f)){rL+=1u;}
    }
    D[Z1+Z2+0] = vec4f(f32(rB),f32(rL),d0.z,d0.w);
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
    var camDir = camo[0];  if(time.frame==0u){camDir = vec3f(-1,0,0);}
    var camPos = camo[1];  if(time.frame==0u){camPos = vec3f(0,22,-88);}
    var mouPre = camo[2];
    var mouNow = vec3f(vec2f(mouse.pos),f32(mouse.click));
    var mouDif = mouNow-mouPre;
    let rs    = vec2f(SCREEN_WIDTH,SCREEN_HEIGHT);
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
    if(id.x >= SCREEN_WIDTH ){return;}
    if(id.y >= SCREEN_HEIGHT){return;}
    var res = 1f/vec2f(textureDimensions(channel0));
    var s2  = vec3f(D[Z1+Z2+0].w);
    var id1 = id.y*SCREEN_WIDTH + id.x;
    var id2f= vec2f(id.xy);
    var seed = (time.frame*sWH + id1)*u32(2*rys*(2+bnc*4));
    var camDir = camo[0];
    var camPos = camo[1];
    var mtx0   = normalize(vec3f(camDir.z,0,-camDir.x));
    var mtx    = mat3x3f(mtx0, cross(camDir,mtx0), camDir);
    var rstImg = camo[3].x;
    var angMin = 1f/(PI*2f*res.x);//background image minimun angle
    var c = vec4f(0);
    for(var k = 0; k < rys; k++)//loop using alias method
    {
        var ray = rayNew(id2f,camPos,mtx,seed);  seed+=2u;
        while(ray.b < bnc)
        {
            ray = rayMarch(ray);
            if(ray.l>ote){break;}
            if(ray.b!=0f){ray = rayBounce(ray,seed);}
            else         {ray = rayBounce2(ray,seed,res,s2);}  seed+=4u;
        }
        var lv = max(0f,floor(log2(ray.a*angMin)));
        var fc = f32(ray.b<bnc)*ray.c*bkgr(ray.d,lv);
        c += vec4f(fc,1)*ray.w*(1f/512f);
    }
    for(var k = 0; k < rys; k++)//loop not using alias method
    {
        var ray = rayNew(id2f,camPos,mtx,seed);  seed+=2u;
        while(ray.b < bnc)
        {
            ray = rayMarch(ray);
            if(ray.l>ote){break;}
            ray = rayBounce(ray,seed); seed+=2u;
        }
        var lv = max(0f,floor(log2(ray.a*angMin)));
        var fc = f32(ray.b<bnc)*ray.c*bkgr(ray.d,lv);
        c += vec4f(fc,1)*(1f/512f);
    }
    c += D[id1];
    D[id1] = c*rstImg;
    textureStore(screen, id.xy, c/c.w*custom.C);
}