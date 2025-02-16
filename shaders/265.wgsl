
#storage hist_atomic array<atomic<u32>>
#storage prev_z array<float>
#define ITER 10000
#define BAILOUT 2e7
#define DIVERGENCE 1e6
#define POWER 2.

struct seed_t {
    v: uint3
};

const k = 1103515245u;
fn get_next_rnd(x: ptr<function, seed_t>) -> float3 {

    (*x).v = (((*x).v >> uint3(8u)) ^ (*x).v.yzx) * k;
    (*x).v = (((*x).v >> uint3(8u)) ^ (*x).v.yzx) * k; 
    (*x).v = (((*x).v >> uint3(8u)) ^ (*x).v.yzx) * k;

    return vec3<float>((*x).v) * (1. / float(0xffffffffu));
}

fn cln(a: float2) -> float2 {
    let ql = length(a);
    return float2(log(ql), atan2(a.y, a.x));
}

fn cmul(a: float2, b: float2) -> float2 {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn cexp(a: float2) -> float2 {
    let ea = exp(a.x);
    let vl = a.y;
    return ea * float2(cos(vl), sin(vl));
}

fn cpow(a: float2, b: float2) -> float2 {
    return cexp(cmul(cln(a), b));
}

fn rotate(a: float2, b: float) -> float2 {
    let s = sin(b); let c = cos(b);
    let m = mat3x3(
        c,-s,0.,s,c,0.,0.,0.,1.);

    return (float3(a, 0.) * m).xy;
}


fn func(xtemp: float2, ctemp: float2) -> float2 {
    let x = xtemp;
    let c = ctemp;
    return cpow(x, float2(POWER, 0.)) + c;
    //return float2(x.x * x.x - x.y * x.y, 2. * x.x * x.y) + c;
}

fn dot2(x: float2) -> float {
    return x.x * x.x + x.y * x.y;
}

fn cabs(x: float2) -> float {
    return sqrt(dot2(x));
}

const center = float2(0.0, 0.0) * float2(1., -1.);
const zoom = 3. / 1.;
fn get_starting_point(uv: float2, aspect: float) -> float2 {
    return ((((uv - .5) * 3.)) / float2(aspect, 1.));
}

fn get_uv(c: float2, aspect: float) -> float2 {
    let uv = ((c.xy * float2(aspect, 1.)) - center) / zoom + .5;

    return uv;
}

@compute @workgroup_size(16,16)
fn splat(@builtin(global_invocation_id) id: uint3) {
    let screen_size = textureDimensions(screen);
    let aspect = float(screen_size.y) / float(screen_size.x);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Normalised pixel coordinates (from 0 to 1)
    let uv = float2(id.xy) / (float2(screen_size));

    var seed = seed_t(uint3(id.xy, time.frame));
    var c = get_starting_point(uv + get_next_rnd(&seed).xy / float2(screen_size), aspect);
    let cjulia = float2(-0.5251993);

    let escapelimit1 = int(float(ITER) * 0.02);
    let escapelimit2 = int(float(ITER) * 0.4);
    let escapelimit3 = int(float(ITER) * .9);

    let seedmod = cabs(c);
    var lowbound = 0.;
    var sumterm = 0.;
    var trisum = 0.;

    var z = c;
    var i = 0;

    for(i = 0; i < ITER; i++) {
        z = func(z, c);
        if (dot(z,z) > BAILOUT) { break; }

        var _mod = cabs(z);
        var dmod = cabs(z - c);

        lowbound = abs(dmod - seedmod);

        sumterm = ((_mod - lowbound) / ((dmod - lowbound) + seedmod));
        trisum += sumterm;
    }

    var colind = 0u;
    let zmod = log(cabs(z));
    let it = float(i);
    if(zmod > 0. && i > 1) {
        let exponent = POWER;
        let logbase = 1. / log(exponent);
        let loglimit = log(log(DIVERGENCE));

        let a = (trisum - sumterm) / (it - 1.);
        let b = trisum / it;

        let upperbound = log(zmod);
        let t = 1. + (loglimit - upperbound) * logbase;

        let index = a + (b - a) * t;
        colind = uint(index * 256.);
    }

    
    let idx = id.y * screen_size.x + id.x;
    z = c;
    var j: uint;
    for(j = 0; j < ITER; j++) {
        if(dot(z,z) > BAILOUT) { break; }
        z = func(z, c);

        let zuv = uint2(get_uv(z, aspect) * float2(screen_size));
        let zid = zuv.x + zuv.y * screen_size.x;

        if (i < escapelimit1) {
            atomicAdd(&hist_atomic[zid*4+2], colind);
        } else if (i < escapelimit2) {
            atomicAdd(&hist_atomic[zid*4+1], colind);
        } else if (i < escapelimit3) {
            atomicAdd(&hist_atomic[zid*4+0], colind);
        }
    }

    prev_z[idx * 2] = z.x;
    prev_z[idx * 2 + 1] = z.y;
}

fn coloring(c: float2) -> float {
    var sum = 0.;
    var sum2 = 0.;
    let ac = cabs(c);
    let il = 1. / log(2.);
    let lp = log(log(BAILOUT) / 2.);
    var az2 = 0.;
    var lowbound = 0.;
    var f = 0.;
    var first = true;

    var z = float2(0.);
    var i = 0;

    for(; i < ITER; i++) {
        if (dot2(z) > BAILOUT) { break; }
        z = func(z, c);

        sum2 = sum;
        if(!first){
        az2 = cabs(z - c);
        lowbound = abs(az2 - ac);
        sum += ((cabs(z) - lowbound) / (az2 + ac - lowbound));
        } else { first = false; }
    }
    
    let iter = float(i);
    sum /= iter;
    sum2 /= iter - 1;
    f = il * lp - il * log(log(cabs(z)));
    return sum2 + (sum - sum2) * (f + 1.);
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    
    let aspect = float(screen_size.y) / float(screen_size.x);
    let uv = float2(id.xy) / (float2(screen_size));

    let idx = (id.y) * screen_size.x + (id.x);
    
    let x = float(atomicLoad(& hist_atomic[idx*4+0])) / 256.;
    let y = float(atomicLoad(&hist_atomic[idx*4+1])) / 256.;
    let z = float(atomicLoad(&hist_atomic[idx*4+2])) / 256.;
    let val = 0.12 * float(ITER) * float3(x, y + x, z + y + x) / float(ITER * (time.frame + 1));

    //let colindex = coloring(get_starting_point(uv, aspect));
    var col = float3(val);

    // Convert from gamma-encoded to linear colour space
    col = pow(col, float3(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, float4(col, 1.));
}
