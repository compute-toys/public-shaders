//using gradient descent
//to make all pair of points distant by f(x,y)
//x and y is the id of each point
//f(x,y) = cos(abs(x-y)*time)+1;

//WARNING code not doing what description says
//the fix it to change (i*T+j) to (r/T*T+j)
//fixing it makes simulation boring

#define PI 3.14159265358979323846f
#define T 256       // threads
#define Z (1<<15)     // bodies
#define ZT 128      // Z/T
#storage P array<f32,Z*4>;
#storage cam array<vec3f,2>;
var<workgroup> P2: array<vec4f,T>;
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
    var p = vec4f(gaus((id1+u32(Z)*0u)*2u + 1124135346u),
                  gaus((id1+u32(Z)*1u)*2u + 1124135346u),
                  gaus((id1+u32(Z)*2u)*2u + 1124135346u),
                  gaus((id1+u32(Z)*3u)*2u + 1124135346u));
    p*=vec4f(1,1,1,0);
    P[id1+Z*0] = p.x;
    P[id1+Z*1] = p.y;
    P[id1+Z*2] = p.z;
    P[id1+Z*3] = p.w;
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
    var ps= vec4f(0);
    var p = vec4f(0);
    var v = vec4f(0);
    var tim =  f32(time.frame)*.0000001f;
    for(var i=0; i<ZT; i++)
    {
        workgroupBarrier();
        P2[id2] = vec4f(P[r+Z*0],
                        P[r+Z*1],
                        P[r+Z*2],
                        P[r+Z*3]);
        workgroupBarrier();
        if(i==0){p = P2[id2];}
        for(var j=0; j<T; j++)
        {
            ps += P2[j];
            var d = P2[j]-p;
            var l = length(d);
            var nrm = 1.f/l;  if(l==0.f){nrm=0.f;}
            var n = cos(abs(f32(id1)-f32(i*T+j))*tim)+1.f;
            v -= d*nrm*(n-l);
        }
        r = (r+T) % Z;
    }
    v *= 1.f/f32(Z);
    var p2 = p+v;//-ps/f32(Z);
    P[id1+Z*0] = p2.x;
    P[id1+Z*1] = p2.y;
    P[id1+Z*2] = p2.z;
    P[id1+Z*3] = p2.w;
    //draw
    var pc = vec4f(P[Z*0],
                   P[Z*1],
                   P[Z*2],
                   P[Z*3]);
    var res = vec2f(textureDimensions(screen));
    var m = (2.f*vec2f(mouse.pos)-res)/res.y;
    var camPos = vec3f(11,0,0);
    var camDir = normalize(cam[0]);
    var sd = normalize(vec3f(camDir.z,0.f,-camDir.x));
    var up = normalize(cross(camDir,sd));
    var a = p.xyz - (dot(p.xyz-camPos,camDir)*camDir + camPos);
    var b = vec2i(i32(dot(a,sd)*res.y*.35f+.5f*res.x),
                  i32(dot(a,up)*res.y*.35f+.5f*res.y));
    textureStore(screen, b, normalize(v)*.5f+.5f);

    if(id1==0){cam[1] = ps.xyz/f32(Z)-camPos;}
}
#workgroup_count camera 1 1 1 
@compute @workgroup_size(1,1,1)
fn camera(@builtin(global_invocation_id) id3: vec3u)
{
    cam[0] += (cam[1]-cam[0])*.004f;
}