#define PI 3.14159265358979323846f
#define T 128       // threads
#define Z 1024      // bodies
#define ZL (1<<14)  // loop each body
#define ZT 8        // Z/T
#define FtI 2048f   // trans betwen float-int
#define SC (SCREEN_WIDTH*SCREEN_HEIGHT)
#storage S array<atomic<i32>,SC*3>
fn xyzTtor(u:vec3f) -> vec3f
{
    var v = vec2f(log(length(u.xz)),u.y);
    return vec3f(atan2(u.z,u.x),atan2(v.y,v.x),log(length(v)));
}
fn torTxyz(u:vec3f) -> vec3f
{
    var w = cos(u.y-vec2f(0,.5)*PI)*exp(u.z);
    var v = cos(u.x-vec2f(0,.5)*PI)*exp(w.x);
    return vec3f(v.x,w.y,v.y);
}
fn hash(a: u32) -> u32
{
    var x = a;
    x ^= x >> 17;  x *= 0xed5ad4bbu;
    x ^= x >> 11;  x *= 0xac4c1b51u;
    x ^= x >> 15;  x *= 0x31848babu;
    x ^= x >> 14;  return x;
}
fn rnd2(a: u32) -> f32
{
    var h   = hash(a);
    var msk = (1u << 23u) - 1u;
    return f32(h & msk) / f32(1u << 23u);
}
fn gaus(a: u32) -> f32//remember this uses 2 seeds from "x"
{
    var r1 = 1.f-rnd2(a+0u);
    var r2 =     rnd2(a+1u);
    return sqrt(-2.f*log(r1))*cos(2.f*PI*r2);
}
fn nrm4(a: vec4f) -> vec4f
{
    var l = length(a);
    var r = a;
    if(l!=0f){r = a/l;}
    return r;
}
@compute @workgroup_size(8, 8)
fn clear(@builtin(global_invocation_id) id: vec3u)
{
    if(id.x >= SCREEN_WIDTH ){ return; }
    if(id.y >= SCREEN_HEIGHT){ return; }
    var a = id.x+id.y*SCREEN_WIDTH;
    var s = vec4i(atomicLoad(&S[a+0*SC]),
                  atomicLoad(&S[a+1*SC]),
                  atomicLoad(&S[a+2*SC]),0);
    atomicStore(&S[a+0*SC], 0);
    atomicStore(&S[a+1*SC], 0);
    atomicStore(&S[a+2*SC], 0);
    textureStore(screen, id.xy, vec4f(s)/FtI);
}
#workgroup_count fun ZT 1 1 
@compute @workgroup_size(T,1,1)
fn fun(@builtin(global_invocation_id) id3: vec3u)
{
    var id1 = i32(id3.x);
    var p = vec4f(gaus(u32((id1+Z*0)*2 + 1124135346)),
                  gaus(u32((id1+Z*1)*2 + 1124135346)),
                  gaus(u32((id1+Z*2)*2 + 1124135346)),
                  gaus(u32((id1+Z*3)*2 + 1124135346)));
    //var<private> b : array<vec4f,2> = array<vec4f>(0, 1);
    var gld = 1.618033988f;
    var gsu = vec4f(1,1,1,1)/(2f-1f/gld);
    var bz = 5;
    var baba = array(
        //vec4f(0,0,0.6124,0),
        //vec4f(-0.2887,-0.5,-0.2041,0),
        //vec4f(-0.2887,0.5,-0.2041,0),
        //vec4f(0.5774,0,-0.2041,0),

        //vec4f(1,0,0,0),
        //vec4f(0,1,0,0),
        //vec4f(0,0,1,0),
        //-vec4f(1,0,0,0),
        //-vec4f(0,1,0,0),
        //-vec4f(0,0,1,0),

        vec4f(2,0,0,0)-gsu,
        vec4f(0,2,0,0)-gsu,
        vec4f(0,0,2,0)-gsu,
        vec4f(0,0,0,2)-gsu,
        vec4f(gld,gld,gld,gld)-gsu,
    );
    var res = vec2f(SCREEN_WIDTH,SCREEN_HEIGHT);
    var bes = dot(res,vec2f(1))/f32(512*512);
    var m = (2f*vec2f(mouse.pos)-res)/res.y;
    var camPos = cos(time.elapsed*vec4f(23,27,17,-13)*.01f+vec4f(1,8,3,14));
    if(mouse.click!=0){camPos = vec4f(cos(m.x),m.y,sin(m.x),0)*2f;}
    var camDir = -normalize(camPos);
    var sd = cos(time.elapsed*vec4f(13,-23,7,-17)*.01f+vec4f(15,5,8,22));
        sd = normalize(sd-camDir*dot(camDir,sd));
    var up = cos(time.elapsed*vec4f(-11,-31,-33,-27)*.01f-vec4f(22,15,81,12));
        up =          (up-camDir*dot(camDir,up));
        up = normalize(up-    sd*dot(    sd,up));
    var s  = u32(id1*ZL*3) + 2456242745u;
    for(var i=0; i<ZL; i++)
    {
        var lm = 1f;
        //var pl = length(p);  if(pl>lm){p = p/pl*fract(pl);}
        var r1 = i32(rnd2(s) * f32(bz-0));    s++;
        var r2 = i32(rnd2(s) * f32(bz-1));    s++;
        var r3 = i32(rnd2(s) * f32(bz-2));    s++;
        if(r2>=r1){r2+=1;}
        var m1 = min(r1,r2);
        var m2 = max(r1,r2);
        if(r3>=m1){r3+=1;}
        if(r3>=m2){r3+=1;}
        var v1 =        p-baba[r1];
        var v2 = baba[r2]-baba[r1];
        var v3 = baba[r3]-baba[r1];
        var c1 = nrm4(v2);
        var c2 = v3;
            c2 = c2-c1*dot(c1,c2);
            c2 = nrm4(c2);
        var c3 = v1;
            c3 = c3-c1*dot(c1,c3);
            c3 = c3-c2*dot(c2,c3);
            c3 = nrm4(c3);
        var tm = custom.a;
            //tm = cos(time.elapsed*.5f)*2.5f-1.4f;
            //tm = 1.2f-time.elapsed*.5f;
        var u = vec3f(dot(v1,c1),
                      dot(v1,c2),
                      dot(v1,c3));
        var cc = abs(nrm4(vec4f(u,0)));
        u = xyzTtor(u);    u *= vec3f(1,-1,1)*tm;
        u = torTxyz(u);    u = u+custom.b;
        p = baba[r1] + c1*u.x  + c2*u.y + c3*u.z;
        //draw
        var p2l = length(p);
        var plg = log(p2l);
        if(p2l!=0f){p2l = 1f/p2l;}
        var p2  = p*p2l*plg; 
        var a = p - (dot(p-camPos,camDir)*camDir + camPos);
        var b = vec2f(dot(a,sd)*res.y*.5f*custom.c+.5f*res.x,
                      dot(a,up)*res.y*.5f*custom.c+.5f*res.y);
        var col = vec4f(cc*FtI*bes);
        var z = 0f;
        for(var y=-z; y<=z; y+=1f){
        for(var x=-z; x<=z; x+=1f){
            var xy = vec2f(x,y);     if(dot(xy,xy)>z*z+1f){continue;}
            var b2 = b+xy;
            var b3 = xy-(fract(b)-.5f);
            var e = 1f/exp(dot(b3,b3));
            var c = col*e;
            if(b2.x<0 || b2.x>=SCREEN_WIDTH ){continue;}
            if(b2.y<0 || b2.y>=SCREEN_HEIGHT){continue;}
            var w = i32(b2.x)+i32(b2.y)*SCREEN_WIDTH;
            atomicAdd(&S[w+0*SC], i32(c.x));
            atomicAdd(&S[w+1*SC], i32(c.y));
            atomicAdd(&S[w+2*SC], i32(c.z));
        }}
    }
}