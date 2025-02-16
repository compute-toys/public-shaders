#define PI  3.1415926535897932384f
#define thrds 32        //GPU threads per wrap
#define bnc 8           //light bounces
#define pex 4           //how many points to explore
#define lex 2           //how many lights to explore
#define srf (1.f/256.f)         //distance considered surface
#define ote 333.f       //outside fractal space
#define rml 50          //raymarch loop
#define G 64.f         //size of fractal

#storage D array<array<float4,SCREEN_HEIGHT>,SCREEN_WIDTH>

fn bkgr(d: float3) -> float3
{
    var uv = vec2(atan2(d.z, d.x), asin(-d.y));
        uv = uv * vec2(1.f / (2.f*PI), 1.f / PI) + .5f;
    return textureSampleLevel(channel0, bilinear, uv, 0).xyz;
}

fn hash(a: uint) -> uint
{
    var x = a;
    x ^= x >> 17;  x *= 0xed5ad4bbu;
    x ^= x >> 11;  x *= 0xac4c1b51u;
    x ^= x >> 15;  x *= 0x31848babu;
    x ^= x >> 14;  return x;
}
fn rnd(a: uint) -> float
{
    var h   = hash(a);
    var msk = (1u << 23u) - 1u;
    return float(h & msk) / float(1u << 23u);
}
fn gaus(a: uint) -> float
{
    var r1 = 1.f-rnd(a   );
    var r2 =     rnd(a+1u);
    return sqrt(-2.f*log(r1))*cos(2.f*PI*r2);
}
fn objec(p2: float3) -> float
{
    var rt = cos(.25*PI*2.-float2(0,.5)*PI);
    var st = sin(.42*PI*2.-float2(0,.5)*PI);
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
        p+=min(dot(p,float3(1,-1,0)),0.)*float3(-1, 1,0);
        p+=min(dot(p,float3(1,0,-1)),0.)*float3(-1, 0,1);
        p+=min(dot(p,float3(0,1,-1)),0.)*float3( 0,-1,1);
        p = st3*p;
        p *= 1.92f;
        s *= 1.92f;
        p += float3(-3.44,-.69,-1.14)*G;
        if(i==7){a = min(a,(length(p)-1.*G)/s);}
        a = min(a,(length(max(abs(p+float3(0,.1,1.3)*G)-float3(8,0,0)*G,float3(0)))-.1f*G)/s);
    }
    a = min(a,(length(p)-2.5f*G)/s);
    return a;
}
fn getCol(pos: float3) -> float3
{
    var rt = cos(.25*PI*2.-float2(0,.5)*PI);
    var st = sin(.42*PI*2.-float2(0,.5)*PI);
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
    var px  = dot(p,float3(2.f/G));
    var s   = 1.f;
    var col = float3(0);
    for(var i = 0; i < 9; i++)
    {
        p = abs(p);
        p = rt3*p;
        p+=min(dot(p,float3(1,-1,0)),0.)*float3(-1, 1,0);
        p+=min(dot(p,float3(1,0,-1)),0.)*float3(-1, 0,1);
        p+=min(dot(p,float3(0,1,-1)),0.)*float3( 0,-1,1);
        p = st3*p;
        p *= 1.92f;
        s *= 1.92f;
        p += float3(-3.44,-.69,-1.14)*G;
        col += (cos(float(i)*16.f*.7015f+float3(1,2,3)*2.f)*.5f+.5f)*
               min(length(max(abs(p+float3(0,.1,1.3)*G)-float3(8,0,0)*G,float3(0)))-.8f*G,G*.9f);
    }
    return col/G*.25f*.6f;
}
fn getNrm(pos: float3) -> float3
{
    var l = objec(pos);
    return normalize(float3(objec(pos+srf*float3(1,0,0)) - l,
                            objec(pos+srf*float3(0,1,0)) - l,
                            objec(pos+srf*float3(0,0,1)) - l));
    //return normalize(float3(objec(pos+srf*float3(1,0,0)) - objec(pos-srf*float3(1,0,0)),
    //                        objec(pos+srf*float3(0,1,0)) - objec(pos-srf*float3(0,1,0)),
    //                        objec(pos+srf*float3(0,0,1)) - objec(pos-srf*float3(0,0,1))));
}
#storage camo array<float3,3>;
#workgroup_count camera 1 1 1
@compute @workgroup_size(1,1,1)
fn camera(@builtin(global_invocation_id) id: uint3)
{
    var keyW = float(keyDown(87u));
    var keyS = float(keyDown(83u));
    var keyA = float(keyDown(65u));
    var keyD = float(keyDown(68u));
    var camDir = camo[0];  if(time.frame==0u){camDir = float3(0,0,1);}
    var camPos = camo[1];  if(time.frame==0u){camPos = float3(0,44,-88);}
    var mouPre = camo[2];
    var mouNow = float3(float2(mouse.pos),float(mouse.click));
    var mouDif = mouNow-mouPre;
    let res    = float2(textureDimensions(screen));
    //mod camDir data
    {
        var cd = camDir;
        var m = mouPre.z*mouDif*-4.f/res.y;
        var y = asin(camDir.y) + m.y;
        if(-1.56f > y){y =-1.56f;};
        if( 1.56f < y){y = 1.56f;};
        var a = cos(m.x-float2(0,PI*.5f));
        var n = cos(y)*normalize(float2(cd.x*a.x - cd.z*a.y,
                                        cd.x*a.y + cd.z*a.x));
        if(any(m.xy!=float2(0))){cd = float3(n.x,sin(y),n.y);}
        camo[0] = cd;
    }
    //mod camPos data
    {
        var s = objec(camPos)*.04f; //*time.elapsed;
        camo[1] = camPos + (keyW-keyS)*s*camDir +
                           (keyD-keyA)*s*normalize(float3(camDir.z,0,-camDir.x));
    }
    //mod mouse data
    {
        camo[2] = float3(mouNow.x,mouNow.y,float(mouNow.z!=0.f && (any(mouDif.xy!=float2(0)) || mouPre.z!=0.f)));
    }
}
struct pS
{
    p: float3, //position
    n: float3, //normal at p
    c: float3, //color at p
};
struct pL
{
    l1: float3, //energy loss from this to that
    l2: float3, //energy loss from that to this
};
var<workgroup> pnts1: array<pS    ,thrds*pex>;
var<workgroup> pnts2: array<pL    ,thrds*pex*(pex-1)/2>;
var<workgroup> pnts3: array<float3,thrds*pex* 2>;
@compute @workgroup_size(thrds,1,1)
fn main_image(@builtin(global_invocation_id) id: uint3, @builtin(local_invocation_index) id2: uint)
{
    var screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x) { return; }
    var res = float2(screen_size);
    var uv  = float2(id.xy)+ .5f;
        uv  = float2(1,-1)*(2.f*uv.xy-res.xy)/res.y;
    var camDir = camo[0];
    var camPos = camo[1];
    var mtx0 = normalize(float3(camDir.z,0,-camDir.x));
    var mtx  = mat3x3<f32>(mtx0, cross(camDir,mtx0), camDir);
    var ray0 = mtx*normalize(float3(uv,1)); //direction of ray from camera
    var p0 = camPos;
    var l0 = 0.f;  for(var i = 0; i < rml; i++){l0 = objec(p0);  p0 += l0*ray0;}
    var n0 = getNrm(p0);
    var c0 = getCol(p0)*float(l0<srf);  //(b>>r)&1u
    var w1 = id2*uint(pex);
    var w2 = id2*uint(pex*(pex-1)/2);
    var w3 = id2*uint(pex*2);
    pnts1[w1].p = p0;
    pnts1[w1].n = n0;
    pnts1[w1].c = c0;  w1+=1u;
    var ids  = id.y*SCREEN_WIDTH + id.x;
    var seed = (time.frame*uint(SCREEN_WIDTH*SCREEN_HEIGHT) + ids) * uint(3*2*pex*(lex+1));
    for(var i = 1u; i < uint(pex); i++)
    {
        var ray = normalize(float3(gaus(seed+2u*0u),
                                   gaus(seed+2u*1u),
                                   gaus(seed+2u*2u)));  seed+=2u*3u;
        var d = dot(ray,n0);
        ray = ray - d*n0 + abs(d)*n0;
        var p = p0+ray;
        var l = 0.f;  for(var j = 0; j < rml; j++){l = objec(p);  p += l*ray;}
        var n = getNrm(p);
        var c = getCol(p);
        pnts1[w1].p = p;
        pnts1[w1].n = n;
        pnts1[w1].c = c;  w1+=1u;
        var m = float(l<srf);
        var l1 = m*c0*abs(d);             if(any(l1!=l1)){l1 = float3(0);}
        var l2 = m*c *abs(dot(ray,n));    if(any(l2!=l2)){l2 = float3(0);}
        pnts2[w2].l1 = l1;
        pnts2[w2].l2 = l2;  w2+=1u;
    }
    w1 = id2*uint(pex);
    for(var i =   1u; i < uint(pex); i++){var pi = pnts1[w1+i].p; var ni = pnts1[w1+i].n; var ci = pnts1[w1+i].c;
    for(var j = i+1u; j < uint(pex); j++){var pj = pnts1[w1+j].p; var nj = pnts1[w1+j].n; var cj = pnts1[w1+j].c;
        var ray = normalize(pj-pi);
        var p   = pi+ray;
        var l = 0.f;  for(var k = 0; k < rml; k++){l = objec(p);  p += l*ray;}
        p = p-pj;
        var m = float(dot(p,p)<srf*srf);
        var l1 = m*ci*abs(dot(ray,ni));   if(any(l1!=l1)){l1 = float3(0);}
        var l2 = m*cj*abs(dot(ray,nj));   if(any(l2!=l2)){l2 = float3(0);}
        pnts2[w2].l1 = l1;
        pnts2[w2].l2 = l2;  w2+=1u;
    }}
    w1 = id2*uint(pex);
    for(var i = 0u; i < uint(pex); i++){
        var pi  = pnts1[w1].p;
        var ni  = pnts1[w1].n;
        var ci  = pnts1[w1].c;  w1+=1u;
    for(var j = 0u; j < uint(lex); j++){
        var ray = normalize(float3(gaus(seed+2u*0u),
                                   gaus(seed+2u*1u),
                                   gaus(seed+2u*2u)));  seed+=2u*3u;
        var d = dot(ray,ni);
        ray = ray - d*ni + abs(d)*ni;
        var p = pi+ray;
        var l = 0.f;  for(var k = 0; k < rml; k++){l = objec(p);  p += l*ray;}
        var m = float(dot(p,p)>ote*ote);
        var c = ci*m*abs(d)*bkgr(ray);  if(any(c!=c)){c = float3(0);}
        pnts3[w3+i] = c;
    }}
    var finCol = pnts3[w3];
    for(var k = 0u; k < uint(bnc); k++)
    {
        var rd = w3 + ((k+0u)&1u)*uint(pex);
        var wt = w3 + ((k+1u)&1u)*uint(pex);
        for(var i = 0u; i < uint(pex); i++){pnts3[wt+i] = float3(0);}
        w2 = id2*uint(pex*(pex-1)/2);
        for(var i =   0u; i < uint(pex); i++){
        for(var j = i+1u; j < uint(pex); j++){
            pnts3[wt+i] += pnts3[rd+j]*pnts2[w2].l1;
            pnts3[wt+j] += pnts3[rd+i]*pnts2[w2].l2;
            w2+=1u;
        }}
        finCol += pnts3[wt];
    }
    {
        var k = uint(bnc);
        var rd = w3 + ((k+0u)&1u)*uint(pex);
        var wt = w3 + ((k+1u)&1u)*uint(pex);
        for(var i = 0u; i < uint(pex); i++){pnts3[wt+i] = float3(0);}
        w2 = id2*uint(pex*(pex-1)/2);
        for(var i =   0u; i < uint(1  ); i++){
        for(var j = i+1u; j < uint(pex); j++){
            pnts3[wt+i] += pnts3[rd+j]*pnts2[w2].l1;
            w2+=1u;
        }}
        finCol += pnts3[wt];
    }   //finCol*=.5f;
    var bk = bkgr(ray0);
    if(dot(p0,p0)>ote*ote || l0!=l0){finCol = bk;}

    var c = D[id.x][id.y];
    if(mouse.click!=0){c = float4(0);}
    c += float4(finCol,1);
    D[id.x][id.y] = c;
    textureStore(screen, id.xy, c/c.w);
}
