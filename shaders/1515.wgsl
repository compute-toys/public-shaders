#define PI 3.14159265358979323846f
#define T 128       // threads
#define Z 1024      // bodies
#define ZL (1<<14)        // loops per bodie
#define ZT 8        // Z/T
#define SC (SCREEN_WIDTH*SCREEN_HEIGHT)
#storage P array<vec4f,Z>;
#storage S array<atomic<i32>,SC*3>
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
#dispatch_once ini
#workgroup_count ini ZT 1 1 
@compute @workgroup_size(T,1,1)
fn ini(@builtin(global_invocation_id) id3: vec3u)
{
    var id1 = id3.x;
    var p = vec4f(gaus((id1+u32(Z)*0u)*2u + 1124135346u),
                  gaus((id1+u32(Z)*1u)*2u + 1124135346u),
                  gaus((id1+u32(Z)*2u)*2u + 1124135346u),
                  gaus((id1+u32(Z)*3u)*2u + 1124135346u));
    p*=vec4f(1,1,1,1);
    P[id1] = p;
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
    textureStore(screen, id.xy, vec4f(s)/f32(Z));
}
#workgroup_count fun ZT 1 1 
@compute @workgroup_size(T,1,1)
fn fun(@builtin(global_invocation_id) id3: vec3u)
{
    var id1 = i32(id3.x);
    var p = P[id1];
    //var<private> b : array<vec4f,2> = array<vec4f>(0, 1);
    var gld = 1.618033988f;
    var gsu = vec4f(1,1,1,1)/(2f-1f/gld);
    var bz = 4;
    var baba = array(
        vec4f(0,0,0.6124,0),
        vec4f(-0.2887,-0.5,-0.2041,0),
        vec4f(-0.2887,0.5,-0.2041,0),
        vec4f(0.5774,0,-0.2041,0),

        //vec4f(1,0,0,0),
        //vec4f(0,1,0,0),
        //vec4f(0,0,1,0),
        //-vec4f(1,0,0,0),
        //-vec4f(0,1,0,0),
        //-vec4f(0,0,1,0),

        //vec4f(2,0,0,0)-gsu,
        //vec4f(0,2,0,0)-gsu,
        //vec4f(0,0,2,0)-gsu,
        //vec4f(0,0,0,2)-gsu,
        //vec4f(gld,gld,gld,gld)-gsu,
    );
    var res = vec2f(SCREEN_WIDTH,SCREEN_HEIGHT);
    var m = (2.f*vec2f(mouse.pos)-res)/res.y;
    var camPos = vec3f(cos(m.x),m.y,sin(m.x))*2.f;
    var camDir = -normalize(camPos);
    var sd = normalize(vec3f(camDir.z,0.f,-camDir.x));
    var up = normalize(cross(camDir,sd));
    var s  = u32(id1*ZL*3) + 2456242745u;
    for(var i=0; i<ZL; i++)
    {
        var r1 = i32(rnd2(s) * f32(bz-0));    s++;
        var r2 = i32(rnd2(s) * f32(bz-1));    s++;
        if(r2>=r1){r2+=1;}
        var l1 = length(p-baba[r1]);
        var v1 =        p-baba[r1];  v1 = nrm4(v1);
        var v2 = baba[r2]-baba[r1];
            v2 = v2-v1*dot(v1,v2);   v2 = nrm4(v2);
        var rt = cos(custom.a*PI*2f-vec2f(0,.5)*PI)*custom.b*l1;
            rt = cos(time.elapsed*.2f-vec2f(0,.5)*PI)*l1*custom.b;
        var p2 = baba[r1] + v1*rt.x + v2*rt.y;
        var cc = abs(p2-p);
        p = p2;
        //draw
        var a = p.xyz - (dot(p.xyz-camPos,camDir)*camDir + camPos);
        var b = vec2f(dot(a,sd)*res.y*.5f*custom.c+.5f*res.x,
                      dot(a,up)*res.y*.5f*custom.c+.5f*res.y);
        var col = vec4f(cc*f32(Z)*.01f);
        var z = 0f;
        for(var y=-z; y<=z; y+=1f){
        for(var x=-z; x<=z; x+=1f){
            var xy = vec2f(x,y);     if(dot(xy,xy)>z*z+1f){continue;}
            var b2 = b+xy;
            var b3 = xy-(fract(b)-.5f);
            var e = 1f/exp(dot(b3,b3));
            var c = col*e;
            var w = i32(b2.x)+i32(b2.y)*SCREEN_WIDTH;
                w = clamp(w,0,SC);
            atomicAdd(&S[w+0*SC], i32(c.x));
            atomicAdd(&S[w+1*SC], i32(c.y));
            atomicAdd(&S[w+2*SC], i32(c.z));
        }}
    }
    P[id1] = p;
}