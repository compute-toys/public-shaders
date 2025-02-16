#define PI 3.14159265358979323846f
#define dt .000001f    // time step
#define T 128       // threads
#define Z 131072     // bodies
#define ZT 1024      // Z/T
#storage P array<vec3f,Z>;
#storage V array<vec3f,Z>;
var<workgroup> P2: array<vec3f,T>;
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
#dispatch_once ini
#workgroup_count ini ZT 1 1 
@compute @workgroup_size(T,1,1)
fn ini(@builtin(global_invocation_id) id3: vec3u)
{
    var id1 = id3.x;
    var p = vec3f(gaus((id1+u32(Z)*0u)*2u + 1124135346u),
                  gaus((id1+u32(Z)*1u)*2u + 1124135346u)*.05f,
                  gaus((id1+u32(Z)*2u)*2u + 1124135346u))*16f;
    var v = vec3f(-p.z,p.y,p.x);
    v = v*pow(dot(v,v),-.25f)*.05f;
    P[id1] = p;
    V[id1] = v;
}
@compute @workgroup_size(8, 8)
fn clear(@builtin(global_invocation_id) id: vec3u)
{
    let screen_size = textureDimensions(screen);
    if(id.x >= screen_size.x){ return; }
    if(id.y >= screen_size.y){ return; }
    textureStore(screen, id.xy, vec4f(0));
}
#workgroup_count fun ZT 1 1 
@compute @workgroup_size(T,1,1)
fn fun(@builtin(global_invocation_id) id3: vec3u)
{
    var id1 = i32(id3.x);
    var id2 = id1 % T;
    var r = id1;
    var p = vec3f(0);
    var v = vec3f(0);
    var s = 0f;
    for(var i=0; i<ZT; i++)
    {
        workgroupBarrier();
        P2[id2] = P[r];
        workgroupBarrier();
        if(i==0){p = P2[id2];}
        for(var j=0; j<T; j++)
        {
            var d = P2[j]-p;
            var l = dot(d,d);
            var f = 1f/(l+.0001f);
            if(l==0f){f=0f;}
            d *= f;
            v += d;
            s += length(d);
        }
        r = (r+T) % Z;
    }
    v = V[id1] + v*dt;
    V[id1] = v;
    P[id1] = v+p;
    //draw
    var res = vec2f(textureDimensions(screen));
    var m = (2f*vec2f(mouse.pos)-res)/res.y;
    var camPos = vec3f(cos(m.x),m.y,sin(m.x))*2f;
    //camPos = vec3f(.1,1,.1);
    var camDir = -normalize(camPos);
    var sd = normalize(vec3f(camDir.z,0f,-camDir.x));
    var up = normalize(cross(camDir,sd));
    var a = p - (dot(p-camPos,camDir)*camDir + camPos);
    var zoom = custom.a*.015f;
    var b = vec2i(i32(dot(a,sd)*res.y*zoom+.5f*res.x),
                  i32(dot(a,up)*res.y*zoom+.5f*res.y));
    textureStore(screen, b, cos(s*.0015f+vec4f(1,2,3,4))*.4f+.5f);
}