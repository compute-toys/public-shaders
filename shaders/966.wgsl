//using gradient descent
//to make all pair of points distant by f(x,y)
//x and y is the id of each point
//f(x,y) = z*z-abs(x-y)*(z-abs(x-y))      z=totalPoints-1

//WARNING code not doing what description says
//the fix is to change (i*T+j) to (r/T*T+j)
//fixing it makes simulation boring

#define PI 3.14159265358979323846f
#define T 256       // threads
#define Z (1<<15)     // bodies
#define ZT 128      // Z/T
#storage P array<f32,Z*4>;
#storage S array<atomic<i32>,SCREEN_WIDTH*SCREEN_HEIGHT*3>
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
fn gaus(a: u32) -> f32//remember this uses 2 seeds from "a"
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
    var a = (id.x+id.y*screen_size.x)*3u;
    var s = vec4i(atomicLoad(&S[a+0u]),
                  atomicLoad(&S[a+1u]),
                  atomicLoad(&S[a+2u]),0);
    atomicStore(&S[a+0u], 0);
    atomicStore(&S[a+1u], 0);
    atomicStore(&S[a+2u], 0);
    textureStore(screen, id.xy, vec4f(s)/f32(Z));
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
    var v2= 0.f;
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
            var n1 = f32(id1);
            var n2 = f32(i*T+j);
            var z1 = f32(Z-1);
            var n = (z1*z1-(abs(n1-n2)*(z1-abs(n1-n2))))*.000000002f;
            v -= d*nrm*(n-l);
            v2+= pow(n-l,2.f);
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
    var rex = i32(textureDimensions(screen).x);
    var res = vec2f(textureDimensions(screen));
    var m = (2.f*vec2f(mouse.pos)-res)/res.y;
    var camPos = ps.xyz/f32(Z)+vec3f(cos(m.x),m.y,sin(m.x))*2.f;
    var camDir = -normalize(vec3f(cos(m.x),m.y,sin(m.x)));
    var sd = normalize(vec3f(camDir.z,0.f,-camDir.x));
    var up = normalize(cross(camDir,sd));
    var a = p.xyz - (dot(p.xyz-camPos,camDir)*camDir + camPos);
    var b = vec2f(dot(a,sd)*res.y*.35f+.5f*res.x,
                  dot(a,up)*res.y*.35f+.5f*res.y);
    var col = pow(cos(sqrt(v2)*3.f+vec4f(4,1,2,4))*.5f+.5f,vec4f(2))*f32(Z)*.5f;
    
    var z = 1.f;
    for(var y=-z; y<=z; y+=1.f){
    for(var x=-z; x<=z; x+=1.f){
        var xy = vec2f(x,y);     if(dot(xy,xy)>z*z+1.f){continue;}
        var b2 = b+xy;
        var b3 = xy-(fract(b)-.5f);
        var e = 1.f/exp(dot(b3,b3));
        var c = col*e;
        var w = (i32(b2.x)+i32(b2.y)*rex)*3;
            w = clamp(w,0,SCREEN_WIDTH*SCREEN_HEIGHT*3);
        atomicAdd(&S[w+0], i32(c.x));
        atomicAdd(&S[w+1], i32(c.y));
        atomicAdd(&S[w+2], i32(c.z));
    }}
}