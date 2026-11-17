//click to find closest point to mouse
//the code creates tree structure that divides by two the data at each level
//then the structure can be used to get the closest point to mouse
//in O(log2(x)), x = total points = lz

//i need this code to clean the training data generated to train the NN
//the training data is filled with paths
//each path has a start-end positions and total time of travel
//"cleaning" means to group paths with similar start-end positions
//then for each group only leave the path with the minimum time

#define lz  11111
#define em  -99999.f //empty id
#define lz2 lz*3
//#define lz2 3*(2<<(ceil(log2(lz))-1))
#define thr  32
#calcdefine thr2 lz/thr+1
#define pi 3.1415926535897932384626433832795f

struct Data
{
    a: array<vec2f,lz>,     //random positions
    b: array<vec2f,lz>,
    f: array<f32,lz2>,
    i: array<i32,lz2>,
    j: array<i32,lz2>,
    o: array<i32,lz>,
    lst: array<i32,lz>,
    p: array<vec3i,lz>,     //offset-start, offset-end, mem-pos
    d: array<f32,lz>,       //divisor position for every level
}
#storage D Data

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
struct MaxResult {
    rf: f32,
    ri: i32,
    rj: i32,
};
fn getMax(f1:f32,f2:f32, i1:i32,i2:i32, j1:i32,j2:i32) -> MaxResult {
    var mx = max(f1, f2);
    var rf = mx;
    var ri = select(i1, i2, mx != f1);
    var rj = select(j1, j2, mx != f1);
    return MaxResult(rf, ri, rj);
}

#dispatch_once ini
#workgroup_count ini thr2 1 1 
@compute @workgroup_size(thr,1,1)
fn ini(@builtin(global_invocation_id) id3: vec3u)
{
    var id1 = i32(id3.x);  if(id1 >= lz){return;}
    D.a[id1] = vec2f(
        rnd(id3.x*2u+0u),
        rnd(id3.x*2u+1u)
    )*2.f-1.f;
    D.p[id1] = vec3i(0,lz,0);
}
#workgroup_count tree0 thr2 1 1 
@compute @workgroup_size(thr,1,1)
fn tree0(@builtin(global_invocation_id) id3: vec3u)
{
    var id1 = i32(id3.x);  if(id1 >= lz){return;}
    var pp  = D.p[id1];
    var zz  = pp.y-pp.x;  if(zz<2){return;}
    var fz  = zz / 2;
    var hz  = zz - fz + 1;
    var o1  = lz2/(1<<time.frame) * pp.z;
    var o2  = id1-pp.x;
    if(o2 >= hz){return;}
    var a = D.a[id1];
    D.f[o1+o2] = select(a.x,a.y,(time.frame % 2) != 0);
    D.i[o1+o2] = o2;
    D.j[o1+o2] = o2;
}
#workgroup_count tree1 thr2 1 1 
@compute @workgroup_size(thr,1,1)
fn tree1(@builtin(global_invocation_id) id3: vec3u)
{
    var id1 = i32(id3.x);  if(id1 >= lz){return;}
    var pp  = D.p[id1];
    var zz  = pp.y-pp.x;  if(zz<2){return;}
    var fz  = zz / 2;
    var hz  = zz - fz + 1;
    if(id1 != pp.x){return;}
    var o1  = lz2/(1<<time.frame) * pp.z;

    var w = hz;
    var r = 0;
    for(var j=hz; j!=1; j=j/2+(j&1)){
        var j1 = j >> 1;
        for(var i=0; i<j1; i++){
            var i2 = r+i*2;
            var m  = getMax(
                D.f[o1+i2+0], D.f[o1+i2+1],
                D.i[o1+i2+0], D.i[o1+i2+1],
                D.j[o1+i2+0], D.j[o1+i2+1]
            );
            D.f[o1+w+i] = m.rf;
            D.i[o1+w+i] = m.ri;
            D.j[o1+w+i] = m.rj;
        }
        if((j & 1) != 0){
            var i2 = r+(j-1);
            D.f[o1+w+j1] = D.f[o1+i2];
            D.i[o1+w+j1] = D.i[o1+i2];
            D.j[o1+w+j1] = D.j[o1+i2];
        }
        w+=j/2+(j&1);
        r+=j;
    }
    D.lst[id1] = w-1;
}
#workgroup_count tree2 thr2 1 1 
@compute @workgroup_size(thr,1,1)
fn tree2(@builtin(global_invocation_id) id3: vec3u)
{
    var id1 = i32(id3.x);  if(id1 >= lz){return;}
    var pp = D.p[id1];
    var zz  = pp.y-pp.x;  if(zz<2){return;}
    var fz  = zz / 2;
    var hz  = zz - fz + 1;
    if(id1 != pp.x){return;}
    var o1 = lz2/(1<<time.frame) * pp.z;

    var lst = D.lst[id1];
    var i = 0;
    while(true)
    {
        D.o[id1+i] = id1+D.j[o1+lst];  i++;
        if(i == fz){break;}
        var b = D.i[o1+lst];
        var a = D.a[id1+hz+i-1];
        D.f[o1+b] = select(a.x,a.y,(time.frame % 2) != 0);
        D.j[o1+b] = hz+i-1;
        var w = hz;
        var r = 0;
        for(var j=hz; j!=1; j=j/2+(j&1)){
            var b1 = !(b == j-1 && (j&1) != 0);
            b = b >> 1;
            var i2 = r + (b<<1);
            var m: MaxResult;
            if(b1){m = getMax(
                D.f[o1+i2+0], D.f[o1+i2+1],
                D.i[o1+i2+0], D.i[o1+i2+1],
                D.j[o1+i2+0], D.j[o1+i2+1]
            );}
            else{m = MaxResult(D.f[o1+i2+0], D.i[o1+i2+0], D.j[o1+i2+0]);}
            D.f[o1+w+b] = m.rf;
            D.i[o1+w+b] = m.ri;
            D.j[o1+w+b] = m.rj;
            w+=j/2+(j&1);
            r+=j;
        }
    }
}
#workgroup_count result1 thr2 1 1 
@compute @workgroup_size(thr,1,1)
fn result1(@builtin(global_invocation_id) id3: vec3u)
{
    var id1 = i32(id3.x);  if(id1 >= lz){return;}
    var pp  = D.p[id1];
    var zz  = pp.y-pp.x;  if(zz<2){return;}
    var fz  = zz / 2;
    var o2  = id1-pp.x;
    if(o2 >= fz){return;}

    var w = id1 + zz - fz;
    var o = D.o[id1];
    D.b[w] = vec2f(D.a[o]);
    D.a[o] = vec2f(em);
}
#workgroup_count result2 thr2 1 1 
@compute @workgroup_size(thr,1,1)
fn result2(@builtin(global_invocation_id) id3: vec3u)
{
    var id1 = i32(id3.x);  if(id1 >= lz){return;}
    var pp = D.p[id1];
    var zz  = pp.y-pp.x;  if(zz<2){return;}
    var fz  = zz / 2;
    if(id1 != pp.x){return;}

    var w = id1;
    for(var i=0; i<zz; i++)
    {
        var a = D.a[id1+i];
        if(a.x == em){continue;}
        D.b[w] = a;
        w++;
    }
    var r = pp.y - fz;
    var m = 0.f;
    var t = (time.frame % 2) == 0;
    if(t){m = D.b[r].x; for(var i=1; i<fz; i++){m = min(m,D.b[r+i].x);}}
    else {m = D.b[r].y; for(var i=1; i<fz; i++){m = min(m,D.b[r+i].y);}}
    D.d[pp.y - fz] = m;
}
#workgroup_count fin thr2 1 1 
@compute @workgroup_size(thr,1,1)
fn fin(@builtin(global_invocation_id) id3: vec3u)
{
    var id1 = i32(id3.x);  if(id1 >= lz){return;}
    D.a[id1] = D.b[id1];
}
#workgroup_count offsets thr2 1 1 
@compute @workgroup_size(thr,1,1)
fn offsets(@builtin(global_invocation_id) id3: vec3u)
{
    var id1 = i32(id3.x);  if(id1 >= lz){return;}
    var p = D.p[id1];
    var m = p.y-p.x;
        m = p.x+m-m/2;
    var a = vec3i(p.x,m, p.z*2+0);
    var b = vec3i(m,p.y, p.z*2+1);
    D.p[id1] = select(a,b, id1 >= m);
}
@compute @workgroup_size(16, 16)
fn clear(@builtin(global_invocation_id) id3: vec3u)
{
    var scz = textureDimensions(screen);
    if(id3.x >= scz.x){ return; }
    if(id3.y >= scz.y){ return; }
    textureStore(screen, id3.xy, vec4f(0));
}
#workgroup_count render thr2 1 1 
@compute @workgroup_size(thr,1,1)
fn render(@builtin(global_invocation_id) id3: vec3u)
{
    var id1 = i32(id3.x);  if(id1 >= lz){return;}
    var scz = textureDimensions(screen);
    var scf = vec2f(scz);
    var zom = custom.a*.5f;
    var mus = D.a[id1];
    var col = vec4f(0,0,1,0);
    var a   = vec2f(0);
    if(false){
        var lvl = 0.f;
        var lvs = 1.f;
        var p   = vec2i(0,lz);
        for(var i=0; true; i++){
            var zz = p.y-p.x;    if(zz==1){a = D.a[p.x]; break;}
            var r  = p.x + zz - zz/2;
            var sg = select(mus.x,mus.y,(i % 2) != 0) - D.d[r];
            p      = select(vec2i(p.x,r), vec2i(r,p.y), sg >= 0.f);
            if(lvs < 2.f){
                lvl = lvl*2.f + f32(sg >= 0.f);
                lvs*= 2.f;
            }
        }
        lvl /= lvs;
        col = cos(lvl*pi*2f+vec4f(1,2,3,4))*.5f+.5f;
    }
    var b = mus*scf.y*zom + scf*.5f;
    if(b.x < 0.f || b.x >= scf.x){return;}
    if(b.y < 0.f || b.y >= scf.y){return;}
    textureStore(screen, vec2i(b), vec4f(col));
}
#workgroup_count find 1 1 1 
@compute @workgroup_size(1,1,1)
fn find(@builtin(global_invocation_id) id3: vec3u)
{
    var zom = custom.a*.5f;
    var scz = textureDimensions(screen);
    var scf = vec2f(scz);
    var mus = (vec2f(mouse.pos)-scf*.5f)/scf.y/zom;
    var p   = vec2i(0,lz);
    var a   = vec2f(0);
    for(var i=0; true; i++){
        var zz = p.y-p.x;    if(zz==1){a = D.a[p.x]; break;}
        var r  = p.x + zz - zz/2;
        var sg = select(mus.x,mus.y,(i % 2) != 0) - D.d[r];
        p      = select(vec2i(p.x,r), vec2i(r,p.y), sg >= 0.f);
    }
    var b   = a*scf.y*zom + scf*.5f;
    if(b.x < 0.f || b.x >= scf.x){return;}
    if(b.y < 0.f || b.y >= scf.y){return;}
    textureStore(screen, vec2i(b), vec4f(0,1,0,0));
}