//wait for channel0 to load then click reset

#define PI  3.1415926535897932384f
#define bkT 256u         //threads for building table
#define bkT2 32u         //threads for rendering
#define Z1 (0u)          //no use
#define Z2 (1<<22)       //sufficienttly big to fit channel0
#storage D array<vec4f,Z1+Z2+1>
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
            tx= tx*vec4f(1,1,1,0);
        var t = length(tx);
        var ar = (-cos(f32(y+1u)/f32(res.y)*PI))-
                 (-cos(f32(y+0u)/f32(res.y)*PI));//area of pixel in sphere
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
    D[Z1+Z2+0] = vec4(0);
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
    D[Z1+Z2+0] = vec4f(f32(rB),f32(rL),0,0);
}
#workgroup_count main_image 1 1 1
@compute @workgroup_size(bkT2,1,1)
fn main_image(@builtin(global_invocation_id) id3: vec3u)
{
    var resSC = vec2f(textureDimensions(screen));
    var resCH = textureDimensions(channel0);
    var pxt = resCH.x*resCH.y;
    var id   = id3.x;
    var seed = (time.frame*bkT2 + id)*2u;
    var r1 = rnd(seed);    seed+=1u;
    var r2 = rnd(seed);    seed+=1u;
    var d2 = D[Z1+u32(r1*f32(pxt))];
    var         pn = d2.z;
    if(r2>d2.y){pn = d2.w;}
    pn = (pn+.5f)/f32(resCH.x);
    var bku = vec2f(fract(pn),(floor(pn)+.5f)/f32(resCH.y));

    var d0 = D[Z1+Z2+0];
    var rB = u32(d0.x);//pos of value bigger than 1
    var rL = u32(d0.y);//pos of value lower than 1
    var bo = rB < pxt && rL < pxt;
    var col = vec4f(1);
    if(bo && id==0u){bku = vec2(d0.x/f32(pxt),0.5f/f32(resSC.y));  col = vec4f(0,1,0,0);}
    if(bo && id==1u){bku = vec2(d0.y/f32(pxt),1.5f/f32(resSC.y));  col = vec4f(0,0,1,0);}
    if(bo && id> 1u){return;}
    textureStore(screen, vec2i(bku*resSC), col);
}
