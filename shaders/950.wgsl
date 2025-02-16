//using gradient descent
//to make all points equally distant from all points
//somehow it creates a sphere, but some points are inside it

#define PI 3.14159265358979323846f
#define dt .00000000f    // time step
#define T 128       // threads
#define Z (1<<14)     // bodies
#define ZT 128      // Z/T
#storage P array<f32,Z*3>;
#storage V array<f32,Z*3>;
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
    var p = vec3f(rnd2((id1+u32(Z)*0u)*2u + 1124135346u),
                  rnd2((id1+u32(Z)*1u)*2u + 1124135346u),
                  rnd2((id1+u32(Z)*2u)*2u + 1124135346u));
    var l = rnd2((id1+u32(Z)*3u)*2u + 1124135346u);
    p=p*2.f-1.f;
    //p = p*l/length(p);
    P[id1+Z*0] = p.x;
    P[id1+Z*1] = p.y;
    P[id1+Z*2] = p.z;
    V[id1+Z*0] = 0.f;
    V[id1+Z*1] = 0.f;
    V[id1+Z*2] = 0.f;
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
    for(var i=0; i<ZT; i++)
    {
        workgroupBarrier();
        P2[id2] = vec3f(P[r+Z*0],
                        P[r+Z*1],
                        P[r+Z*2]);
        workgroupBarrier();
        if(i==0){p = P2[id2];}
        for(var j=0; j<T; j++)
        {
            var d = P2[j]-p;
            var l = length(d);
            var nrm = 1.f/l;
            if(l==0.f){nrm=0.f;}
            v -= d*nrm*(1.f-l);
        }
        r = (r+T) % Z;
    }
    v = v*(1.f/float(Z));
    var p2 = p+v;
    P[id1+Z*0] = p2.x;
    P[id1+Z*1] = p2.y;
    P[id1+Z*2] = p2.z;
    //draw
    var res = vec2f(textureDimensions(screen));
    var m = (2.f*vec2f(mouse.pos)-res)/res.y;
    var camPos = vec3f(cos(m.x),m.y,sin(m.x))*2.f;
    //camPos = vec3f(.1,1,.1);
    var camDir = -normalize(camPos);
    var sd = normalize(vec3f(camDir.z,0.f,-camDir.x));
    var up = normalize(cross(camDir,sd));
    var a = p - (dot(p-camPos,camDir)*camDir + camPos);
    var b = vec2i(i32(dot(a,sd)*res.y*.5f+.5f*res.x),
                  i32(dot(a,up)*res.y*.5f+.5f*res.y));
    textureStore(screen, b, vec4f(1));
}