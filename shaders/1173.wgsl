// Code needs to be heavily cleaned up and explained.
// But can play around with the sliders i guess.

alias f = float;alias iv4 = vec4<i32>;alias iv3 = vec3<i32>;alias iv2 = vec2<i32>;alias uv4 = vec4<u32>;alias uv3 = vec3<u32>;alias uv2 = vec2<u32>;alias v4 = vec4<f32>;alias v3 = vec3<f32>;alias v2 = vec2<f32>;alias m2 = mat2x2<f32>;alias m3 = mat3x3<f32>;alias m4 = mat4x4<f32>;
#define global var<private>
var<workgroup> texCoords: array<array<float2, 16>, 16>;
#define F time.frame
#define T (time.elapsed) 
#define pi acos(-1.)
#define tau (acos(-1.)*2.)
#define INIT Ru = uint2(textureDimensions(screen)); R = v2(Ru); U = float2(float(id.x) + .5, float(Ru.y - id.y) - .5); seed = hash_u(id.x + 125124 + uint(custom.SEED*50) + hash_u(Ru.x*id.y*200u)*20u + hash_u(id.y)*250u + hash_u(id.x*21832)); seed = hash_u(seed); seed = hash_u(seed); seed = hash_u(seed); seed = hash_u(seed);

#define QUANT 500


#define K_CNT int(custom.K_COUNT+1)
#define PART_CNT uint(1 + custom.PARTICLES_COUNT)
#define ITERS (-1 + custom.K_ITERS) 

struct Particle{
    pos: vec3f,
    col: vec3f
}
struct K{
    pos: vec3f,
    col: vec3f,
    accum: array<atomic<int>,4>,
    min_col: array<atomic<uint>,3>,
    max_col: array<atomic<uint>,3>,
    accum_cnt: atomic<uint>
}
struct ParticleData{
    particles: array<Particle,10000>,
    k: array<K, 10000>,
    // cnt: atomic<uint>,
    iter: uint
}
#storage buff ParticleData


struct ScreenData{
    r: array<atomic<uint>,2073600>,
    g: array<atomic<uint>,2073600>,
    b: array<atomic<uint>,2073600>,
    last_click: int,
    last_last_click: int,
    last_mouse: float2,
    current_mouse: float2,
    delta_mouse: float2,
    mouse_input: float3,
}
#storage screendata ScreenData

fn sample_input_image(uv: vec2f) -> float4 {
    var col = textureSampleLevel(channel0,bilinear,uv,0);
    col = 1.-exp(-col*1.5);
    col = pow(col,vec4f(1.4));
    return col;
}


fn to_space(c: vec3f) -> vec3f{
    if(custom.DIST_HEUR < 1){
        // return srgb2oklab(c);
        return rgb2oklab(c);
// rgb2oklab(
// oklab2rgb
    } else if(custom.DIST_HEUR < 2){
        return srgb_to_oklch(c);
    } else {
        return c;
    }
}

fn from_space(c: vec3f) -> vec3f{
    if(custom.DIST_HEUR < 1){
        // return oklab2srgb(c);
        return oklab2rgb(c);
    } else if(custom.DIST_HEUR < 2){
        return oklch_to_srgb(c);
    } else {
        return c;
    }
}

fn colour_dist(a: vec3f, b: vec3f) -> float{
    if(custom.DIST_HEUR < 1.0 ){
        var aa = a;
        var bb = b;
        var v = abs(aa - bb);
        return length(v*vec3f(0.8,0.5,1.0));
    } else if(custom.DIST_HEUR < 2.0 ){
        // var aa = srgb_to_oklch(a);
        // var bb = srgb_to_oklch(b);
        // var aa = oklch_to_srgb(a);
        // var bb = oklch_to_srgb(b);
        var aa = a;
        var bb = b;
        var v = abs(aa - bb);
        if(true){
            var h_a = bb.z;
            var h_b = bb.z + tau;
            // h_b = h_b % tau;
            var dist_a = abs(aa.z - h_a);
            var dist_b = abs(aa.z - h_b);
            // var hlen 
            if(dist_a < dist_b){
                // col.z = mix(col.z, h_a, alpha);
                v.z = abs(h_a - aa.z);
            } else {
                v.z = abs(h_b - aa.z);
                // col.z = mix(col.z, h_b, alpha);
            }
            // return abs(v.z);
            // return length(v*vec3f(0.5,0.02,1./tau));
            // return length(v*vec3f(1.0,4.0,1./tau));
            return length(v*vec3f(0.4,1.0,0.01/tau));
        } else {
            return length(v*vec3f(1.,1,1));
        }
    } else {
        return length(a-b);
    }
}


fn luma(col: vec3f) -> float {
    return dot(col, vec3f(0.2126729, 0.7151522, 0.0721750));
}



#workgroup_count particles_init 100 1 1 
@compute @workgroup_size(256, 1)
fn particles_init(@builtin(global_invocation_id) id: uint3) {
    //let cnt = &buff.cnt;
    INIT

    if(id.x < PART_CNT){
        let part_id = id.x;
        
        var part: Particle;

        part.col = sample_input_image(hash_v2()).rgb;
        // part.col = max(part.col,vec3f(0));
        // part.col = min(part.col,vec3f(1));
        part.col = to_space(part.col);
        buff.particles[part_id] = part;
    }
    
    

    if(id.x < uint(K_CNT)){
        let k = &buff.k[id.x];
        seed -= 15u + id.x*5;
        // part.col = clamp(part.col,vec3f(0),vec3f(1);
        var col: vec3f;
        if(custom.PALETTE < 1.0){
            col = pow(hash_v3(),vec3f(3.));
        } else if(custom.PALETTE < 2.0) {
            col = sample_input_image(hash_v2()).rgb;

            col = srgb_to_oklch( col );
            // col.z += 5.5;
            col.z += hash_f()*15.;
            // col.z -= 2.;
            // col.x *= 0.;
            col.z = col.z%tau;
            col = oklch2rgb(col);
            col = pow(col,vec3f(.5));

            col = max(col,vec3f(0));
            col = min(col,vec3f(1));
        } else if(custom.PALETTE < 3.0) {
            let palAppleII = array<vec3f, 16>(
                vec3f(217, 60, 240)/255.,
                vec3f(64, 53, 120)/255.,
                vec3f(108, 41, 64)/255.,
                vec3f(0, 0, 0)/255. + 0.001,
            
                vec3f(236, 168, 191)/255.,
                vec3f(128, 128, 128)/255.,
                vec3f(217, 104, 15)/255.,
                vec3f(64, 75, 7)/255.,
            
                vec3f(191, 180, 248)/255.,
                vec3f(38, 151, 240)/255.,
                vec3f(128, 128, 128)/255.,
                vec3f(19, 87, 64)/255.,
            
                vec3f(255, 255, 255)/255.,
                vec3f(147, 214, 191)/255.,
                vec3f(191, 202, 135)/255.,
                vec3f(38, 195, 15)/255.
            );
            
            col = palAppleII[id.x%16];
            // col = min(col,vec3f(1.0));
            col = pow(col,vec3f(0.4));
            
        } else {
            // col = oklch_to_srgb( vec3f(0.3,0.1,hash_f()*tau + 0.01) );
            // col = oklch2rgb(vec3f(0.5,1.0,hash_f()*tau*0.01 + 0.3));

            col = sample_input_image(hash_v2()).rgb;
            col = rgb2oklch(col);
            col = oklch2rgb(vec3f(0.5 +col.x*0.0 - hash_f()*0.4,hash_f()*0.1,hash_f()*tau*1.0));
            if(luma(col) > 0.02){
                // col = vec3f(0.5);
            }
            // col = oklch2rgb(vec3f(0.5,1.0,hash_f()*tau*0.01 + 0.3));
            // col = srgb_to_oklch( col );
            // col.x = 0.5;
            // col.y = 2.;
            // col = oklch_to_srgb( col );
            

            col = clamp(col, vec3f(0), vec3f(1));
            // oklch_to_srgb( vec3f);
        }

        col = to_space(col);

        buff.k[id.x].pos = col;
        buff.k[id.x].col = col;
        // if()
        // buff.k[id.x].pos = rgb_to_oklch(buff.k[id.x].pos);

        atomicStore(&(*k).accum_cnt,0);
        atomicStore(&(*k).accum[0],0);
        atomicStore(&(*k).accum[1],0);
        atomicStore(&(*k).accum[2],0);
        atomicStore(&(*k).accum[3],0);
    }
}


#workgroup_count k_draw 20 1 1 
@compute @workgroup_size(256, 1)
fn k_draw(@builtin(global_invocation_id) id: uint3) {
    INIT
    let part_id = id.x;
    
    if(id.x < PART_CNT){
        var part: Particle = buff.particles[part_id];
        draw_part(part.col, vec3f(0,0,0.05f));
    }

    if(id.x < uint(K_CNT)){
        draw_part(buff.k[id.x].pos, vec3f(1,0.4,0));
    }
}

fn do_particles_iter(id: uint3){
    let part_id = id.x;
    if(part_id >= PART_CNT){
        return;
    }
    
    var part: Particle = buff.particles[part_id];
    var min_dist = 1000000.0;
    var k_idx = 0;
    for(var i = 0; i < K_CNT; i++){
        let k = &buff.k[i];

        // let dist_to_k = length(part.col - k.pos);
        let dist_to_k = colour_dist(part.col, k.pos);
        if(dist_to_k < min_dist){
            k_idx = i;
            min_dist = dist_to_k;
        }
    }
    let k = &buff.k[k_idx];
    #define IS_OKLCH (custom.DIST_HEUR > 1 && custom.DIST_HEUR < 2)
    
    if(custom.DIST_HEUR > 1 && custom.DIST_HEUR < 2){
    }
    
    atomicAdd(&(*k).accum[0], int(part.col[0]*QUANT));
    atomicAdd(&(*k).accum[1], int(part.col[1]*QUANT));
    if(IS_OKLCH){
        atomicAdd(&(*k).accum[2], int(cos(part.col[2])*QUANT));
        atomicAdd(&(*k).accum[3], int(sin(part.col[2])*QUANT));
    } else {
        atomicAdd(&(*k).accum[2], int(part.col[2]*QUANT));
    }
    atomicAdd(&(*k).accum_cnt, uint(1));
}

fn do_k_iter(id: uint3){
    INIT
    if(id.x >= uint(K_CNT)){
        return;
    }
    let k_id = id.x;
    let k = &buff.k[k_id];

    let acc_cnt = float(atomicLoad(&(*k).accum_cnt));
    let r = float(atomicLoad(&(*k).accum[0]))/QUANT;
    let g = float(atomicLoad(&(*k).accum[1]))/QUANT;
    let b = float(atomicLoad(&(*k).accum[2]))/QUANT;
    let w = float(atomicLoad(&(*k).accum[3]))/QUANT;


    var avg_col = vec3f(r,g,b)/acc_cnt;

    if(IS_OKLCH){
        avg_col.z = atan2(w,b);
        // avg_col.z = atan(b/w);
    }
    let prev_pos = (*(k)).pos;
    var next_pos = mix(prev_pos,avg_col,custom.CRAWL);
    if(acc_cnt > 0.01){
        // (*(k)).pos = avg_col;
        (*(k)).pos = next_pos;
        // let pos = avg_col;
    
        let iters = 50;
        for(var i =0; i < iters; i++){
            draw_point(mix(prev_pos, next_pos, float(i)/float(iters)), vec3f(0.6));
        }
    
        
        draw_part((*(k)).pos, vec3f(0.2,1.,0)*0.05);
    }

    
    atomicStore(&(*k).accum_cnt,0);
    atomicStore(&(*k).accum[0],0);
    atomicStore(&(*k).accum[1],0);
    atomicStore(&(*k).accum[2],0);
    atomicStore(&(*k).accum[3],0);
}

#workgroup_count particles_iter 100 1 1 
@compute @workgroup_size(256, 1)
fn particles_iter(@builtin(global_invocation_id) id: uint3){
    if(0 > ITERS){return;}
    do_particles_iter(id);
}
#workgroup_count k_iter 100 1 1
@compute @workgroup_size(256, 1)
fn k_iter(@builtin(global_invocation_id) id: uint3){
    if(0 > ITERS){return;}
    do_k_iter(id);
}

#workgroup_count particles_iter_2 100 1 1 
@compute @workgroup_size(256, 1)
fn particles_iter_2(@builtin(global_invocation_id) id: uint3){
    if(1 > ITERS){return;}
    do_particles_iter(id);
}

#workgroup_count k_iter_2 100 1 1
@compute @workgroup_size(256, 1)
fn k_iter_2(@builtin(global_invocation_id) id: uint3){
    if(1 > ITERS){return;}
    do_k_iter(id);
}

#workgroup_count particles_iter_3 100 1 1 
@compute @workgroup_size(256, 1)
fn particles_iter_3(@builtin(global_invocation_id) id: uint3){
    if(2 > ITERS){return;}
    do_particles_iter(id);
}

#workgroup_count k_iter_3 100 1 1
@compute @workgroup_size(256, 1)
fn k_iter_3(@builtin(global_invocation_id) id: uint3){
    if(2 > ITERS){return;}
    do_k_iter(id);
}

#workgroup_count particles_iter_4 100 1 1 
@compute @workgroup_size(256, 1)
fn particles_iter_4(@builtin(global_invocation_id) id: uint3){
    if(3 > ITERS){return;}
    do_particles_iter(id);
}

#workgroup_count k_iter_4 100 1 1
@compute @workgroup_size(256, 1)
fn k_iter_4(@builtin(global_invocation_id) id: uint3){
    if(3 > ITERS){return;}
    do_k_iter(id);
}

#workgroup_count particles_iter_5 100 1 1 
@compute @workgroup_size(256, 1)
fn particles_iter_5(@builtin(global_invocation_id) id: uint3){
    if(4 > ITERS){return;}
    do_particles_iter(id);
}

#workgroup_count k_iter_5 100 1 1
@compute @workgroup_size(256, 1)
fn k_iter_5(@builtin(global_invocation_id) id: uint3){
    if(4 > ITERS){return;}
    do_k_iter(id);
}

#workgroup_count particles_iter_6 100 1 1 
@compute @workgroup_size(256, 1)
fn particles_iter_6(@builtin(global_invocation_id) id: uint3){
    if(5 > ITERS){return;}
    do_particles_iter(id);
}

#workgroup_count k_iter_6 100 1 1
@compute @workgroup_size(256, 1)
fn k_iter_6(@builtin(global_invocation_id) id: uint3){
    if(5 > ITERS){return;}
    do_k_iter(id);
}


#workgroup_count particles_iter_7 100 1 1 
@compute @workgroup_size(256, 1)
fn particles_iter_7(@builtin(global_invocation_id) id: uint3){
    if(6 > ITERS){return;}
    do_particles_iter(id);
}

#workgroup_count k_iter_7 100 1 1
@compute @workgroup_size(256, 1)
fn k_iter_7(@builtin(global_invocation_id) id: uint3){
    if(6 > ITERS){return;}
    do_k_iter(id);
}

#workgroup_count particles_iter_8 100 1 1 
@compute @workgroup_size(256, 1)
fn particles_iter_8(@builtin(global_invocation_id) id: uint3){
    if(7 > ITERS){return;}
    do_particles_iter(id);
}

#workgroup_count k_iter_8 100 1 1
@compute @workgroup_size(256, 1)
fn k_iter_8(@builtin(global_invocation_id) id: uint3){
    if(7 > ITERS){return;}
    do_k_iter(id);
}





@compute @workgroup_size(16, 16)
fn draw_k_means_image(@builtin(global_invocation_id) id: vec3u) {
    INIT
    if (id.x >= Ru.x || id.y >= Ru.y) { return; } 

    var col = vec3f(0);

    var uv = (U)/R;

    let idx = uv_to_pixel_idx(uv);
    col.r += float(atomicLoad(&screendata.r[idx]));
    col.g += float(atomicLoad(&screendata.g[idx]));
    col.b += float(atomicLoad(&screendata.b[idx]));

    col /= QUANT;
    var closest_k = 100000.0;
    var k_idx = 0;
    for(var i = 0; i < K_CNT; i++){
        let k = &buff.k[i];
        // let dist_to_k = length(uv - k.pos.xy);
        let dist_to_k = colour_dist(
            vec3f(uv,0.), 
            from_space(vec3f(k.pos.xy,0.f))
        );
        if(dist_to_k < closest_k){
            k_idx = i;
            closest_k = dist_to_k;
        }
    }
    seed = 12513123u + uint(k_idx)*10;

    let k = &buff.k[k_idx]; 

    // k.pos.xyz;

    // col += pow(hash_v3(),vec3f(2.))*0.4;
    if(length(col)<0.00000000000001){
        col = k.pos.xyz*1.0;
        col = from_space(col);
        // if(custom.DIST_HEUR < 1.0){
        //     col = oklch_to_srgb(col);
        // }
    }
    passStore(0, vec2i(id.xy), vec4f(col,1));
}


@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    INIT
    if (id.x >= Ru.x || id.y >= Ru.y) { return; } 

    if(id.x == 0 && id.y == 0){
        // last_mouse = mouse.pos:
        screendata.last_mouse = screendata.current_mouse;
        screendata.current_mouse = float2(mouse.pos)/R.xy;
        screendata.delta_mouse = screendata.current_mouse - screendata.last_mouse;
        if(mouse.click > 0 && screendata.last_last_click > 0){
            screendata.mouse_input.x += screendata.delta_mouse.x;
            screendata.mouse_input.y += screendata.delta_mouse.y;

        }
        screendata.last_last_click = screendata.last_click;
        screendata.last_click = mouse.click;
        // if()

        // last_mouse: float2,
        // current_mouse: float2,
        // mouse_input: float3,

    }
    
    var col = vec3f(0);

    var uv = (U)/R;

    let idx = uv_to_pixel_idx(uv);

    if(uv.x < 0.5){
        uv.y = 1.-uv.y;
        uv.x *= 2.;
        col = passSampleLevelBilinearRepeat(0, uv, 0).rgb;
    } else {
        uv.x -= 0.5;
        uv.y = 1. - uv.y;


        col = sample_input_image(uv).rgb;
        var dith_amt = custom.DITH_AMT*10./float(K_CNT);
        // col += ;
        let dith_sc = 2u;
        var dith = (textureLoad(channel1,vec2i(id.xy/dith_sc) % 8,0).x*2. - 1.) * dith_amt;

        var closest_k = 100000.0;
        var k_idx = 0;
        var k_col = vec3f(0);
        for(var i = 0; i < K_CNT; i++){
            let k = &buff.k[i];
    
            // let dist_to_k = length(col - k.pos);
            let dist_to_k = colour_dist(
                to_space(col) + dith, 
                k.pos
            );
            if(dist_to_k < closest_k){
                k_idx = i;
                closest_k = dist_to_k;
                // k_col = k.pos;
                k_col = k.col;
            }
        }
        col = k_col;
            // col = oklch_to_srgb(col);
        col = from_space(col);
    }

    col = srgb_to_oklch( col );
    // col.z += 15.;
    // col.x = 0.5;
    col = oklch_to_srgb( col );
    // fn oklch_to_srgb( c: vec3f ) -> vec3f { return oklab2srgb(lch2lab(c)); }


    atomicStore(&screendata.r[idx],0);    
    atomicStore(&screendata.g[idx],0);    
    atomicStore(&screendata.b[idx],0);    
    textureStore(screen, id.xy, vec4f(col, 1.));
}








global R: v2; global Ru: uv2; global U: v2; global seed: uint; global muv: v2;
fn rot(a: float)-> m2{ return m2(cos(a), -sin(a),sin(a), cos(a));}
fn rotX(a: float) -> m3{
    let r = rot(a); return m3(1.,0.,0.,0.,r[0][0],r[0][1],0.,r[1][0],r[1][1]);
}
fn rotY(a: float) -> m3{
    let r = rot(a); return m3(r[0][0],0.,r[0][1],0.,1.,0.,r[1][0],0.,r[1][1]);
}
fn rotZ(a: float) -> m3{
    let r = rot(a); return m3(r[0][0],r[0][1],0.,r[1][0],r[1][1],0.,0.,0.,1.);
}
fn emod (a: f32, b: f32) -> f32 {var m = a % b;if (m < 0.0) {if (b < 0.0) {m -= b;} else {m += b;}}return m;}
fn gmod(a: v3, b: v3)->v3{return v3(emod(a.x,b.x),emod(a.y,b.y),emod(a.z,b.z));}
fn gmod_v2(a: v2, b: v2)->v2{return v2(emod(a.x,b.x),emod(a.y,b.y));}
fn hash_u(_a: uint) -> uint{ var a = _a; a ^= a >> 16;a *= 0x7feb352du;a ^= a >> 15;a *= 0x846ca68bu;a ^= a >> 16;return a; }
fn hash_f() -> float{ var s = hash_u(seed); seed = s;return ( float( s ) / float( 0xffffffffu ) ); }
fn hash_v2() -> v2{ return v2(hash_f(), hash_f()); }
fn hash_v3() -> v3{ return v3(hash_f(), hash_f(), hash_f()); }
fn hash_v4() -> v4{ return v4(hash_f(), hash_f(), hash_f(), hash_f()); }



global m1 = mat3x3<f32>(
    0.4122214708, 0.5363325363, 0.0514459929,
    0.2119034982, 0.6806995451, 0.1073969566,
    0.0883024619, 0.2817188376, 0.6299787005
);

global inverse_m1 = mat3x3<f32>(
    4.0767416621, -3.3077115913, 0.2309699292,
    -1.2684380046, 2.6097574011, -0.3413193965,
    -0.0041960863, -0.7034186147, 1.7076147010
);

global mm2 = mat3x3<f32>(
    0.2104542553, 0.7936177850, -0.0040720468,
    1.9779984951, -2.4285922050, 0.4505937099,
    0.0259040371, 0.7827717662, -0.8086757660
);

global inverse_m2 = mat3x3<f32>(
    1.0, 0.3963377774, 0.2158037573,
    1.0, -0.1055613458, -0.0638541728,
    1.0, -0.0894841775, -1.2914855480
);

fn cbrt(x: f32) -> f32 {
    var y = sign(x) * bitcast<f32>((bitcast<u32>(abs(x)) / 3u) + 0x2a514067u);

    for (var i = 0; i < 1; i = i + 1) {
        y = (2.0 * y + x / (y * y)) * 0.333333333;
    }

    for (var i = 0; i < 1; i = i + 1) {
        let y3 = y * y * y;
        y = y * (y3 + 2.0 * x) / (2.0 * y3 + x);
    }
    
    return y;
}

fn cbrt_vec3(xyz: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(cbrt(xyz.x), cbrt(xyz.y), cbrt(xyz.z));
}

fn rgb2oklab(rgb: vec3<f32>) -> vec3<f32> {
    return cbrt_vec3(rgb * m1) * mm2;
}

fn oklab2rgb(oklab: vec3<f32>) -> vec3<f32> {
    // todo: max
    return pow(abs(oklab * inverse_m2), vec3<f32>(3.0, 3.0, 3.0)) * inverse_m1;
}

fn oklab2oklch(oklab: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        oklab.x,
        sqrt(oklab.y * oklab.y + oklab.z * oklab.z),
        atan2(oklab.z, oklab.y) / tau
    );
}

fn oklch2oklab(oklch: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        oklch.x,
        oklch.y * cos(oklch.z * tau),
        oklch.y * sin(oklch.z * tau)
    );
}


fn rgb2oklch(rgb: vec3<f32>) -> vec3<f32> {
    return oklab2oklch(rgb2oklab(rgb));
}

fn oklch2rgb(oklch: vec3<f32>) -> vec3<f32> {
    return oklab2rgb(oklch2oklab(oklch));
}




fn mul3( m: mat3x3f, v: vec3f ) -> vec3f{
    return vec3f(dot(v,m[0]),dot(v,m[1]),dot(v,m[2]));
}

fn mul3b( v: vec3f, m: mat3x3f ) -> vec3f {return mul3(m,v);}

fn srgb2oklab(c: vec3f) -> vec3f {
    
    var m1 = mat3x3f(
        0.4122214708,0.5363325363,0.0514459929,
        0.2119034982,0.6806995451,0.1073969566,
        0.0883024619,0.2817188376,0.6299787005
    );
    
    var lms: vec3f = mul3(m1,c);
    
    lms = pow(max(lms,vec3f(0.)),vec3f(1./3.));

    var m2 = mat3x3f(
        0.2104542553,0.7936177850,-0.0040720468,
        1.9779984951,-2.4285922050,0.4505937099,
        0.0259040371,0.7827717662,-0.8086757660
    );
    
    return mul3(m2,lms);
}

fn oklab2srgb(c: vec3f) -> vec3f
{
    var m1 = mat3x3f(
        1.0000000000,0.3963377774,0.2158037573,
        1.0000000000,-0.1055613458,-0.0638541728,
        1.0000000000,-0.0894841775,-1.2914855480
    );

    var lms: vec3f = mul3(m1,c);
    
    lms = lms * lms * lms;
  
    var m2 = mat3x3f(
        4.0767416621,-3.3077115913,0.2309699292,
        -1.2684380046,2.6097574011,-0.3413193965,
        -0.0041960863,-0.7034186147,1.7076147010
    );
    return mul3(m2,lms);
}

fn lab2lch( c: vec3f ) -> vec3f{return vec3f(c.x,sqrt((c.y*c.y) + (c.z * c.z)),atan2(c.z,c.y));}

fn lch2lab( c: vec3f ) -> vec3f{return vec3f(c.x,c.y*cos(c.z),c.y*sin(c.z));}

fn srgb_to_oklch( c: vec3f ) -> vec3f { 
    return rgb2oklch(c);
    return lab2lch(srgb2oklab(c)); 
}
fn oklch_to_srgb( c: vec3f ) -> vec3f { 
    return oklch2rgb(c);
    return oklab2srgb(lch2lab(c)); 
}


fn uv_to_pixel_idx(k: v2)->uint{
    var uv = (k.xy);
    let cc = uv2(uv.xy*R.xy);
    let idx = cc.x - Ru.x * cc.y + Ru.x*Ru.y - 1;
    return idx;
}

fn sample_disk() -> v2{
    let r = hash_v2();
    return v2(sin(r.x*tau),cos(r.x*tau))*sqrt(r.y);
}
fn pmod(p:v3, amt: float)->v3{ return (v3(p + amt*0.5)%amt) - amt*0.5 ;}


fn proj_p(in_pos: vec3f) -> vec3f{
    var pos = in_pos;
    pos -= 0.5;
    // pos *= 0.5;
    // pos.y += 0.5;
    pos = rotY(-screendata.mouse_input.x*10) * pos;

    pos = rotX(screendata.mouse_input.y*4.5) * pos;
    


    // pos -= screendata.cam_pos;
    pos += float3(0,0,1)*1.6;

    pos.x /= pos.z;
    pos.y /= pos.z;
    pos.x += 0.5;
    pos.y += 0.5;
    if(pos.z < 0.){
        pos = -float3(1000);
    }
    return pos;
}

fn draw_point(pos: v3, col: vec3f){
    for(var i = 0; i < 1; i++){
        // var k = pos.xy;
        var k = proj_p(pos);

        if(k.z > 0.){
            var uv = k.xy;
            let idx = uv_to_pixel_idx(uv);
            if (
                uv.x > 0. && uv.x < 1. 
                && uv.y > 0. && uv.y < 1. 
                && idx < uint(Ru.x*Ru.y)
                ){
                atomicAdd(&screendata.r[idx],uint(col.r*QUANT));
                atomicAdd(&screendata.g[idx],uint(col.g*QUANT));
                atomicAdd(&screendata.b[idx],uint(col.b*QUANT));
            }
        }
        // k += sample_disk()*0.002;
    }

}

fn draw_part(pos: vec3f, col: vec3f){
    for(var i = 0; i < 40; i++){
        var k = proj_p(pos);

        if(k.z > 0.){
            var uv = k.xy;
            let idx = uv_to_pixel_idx(uv);
            if (
                uv.x > 0. && uv.x < 1. 
                && uv.y > 0. && uv.y < 1. 
                && idx < uint(Ru.x*Ru.y)
                ){
                atomicAdd(&screendata.r[idx],uint(col.r*QUANT));
                atomicAdd(&screendata.g[idx],uint(col.g*QUANT));
                atomicAdd(&screendata.b[idx],uint(col.b*QUANT));
            }
        }
        // var k = pos;
        // k.x += sample_disk().x*0.002;
        // k.y += sample_disk().y*0.002;
        
        // var uv = k.xy;
        // let idx = uv_to_pixel_idx(k);
        // if (
        //     uv.x > 0. && uv.x < 1. 
        //     && uv.y > 0. && uv.y < 1. 
        //     && idx < uint(Ru.x*Ru.y)
        //     ){
        //     atomicAdd(&screendata.r[idx],uint(col.r*QUANT));
        //     atomicAdd(&screendata.g[idx],uint(col.g*QUANT));
        //     atomicAdd(&screendata.b[idx],uint(col.b*QUANT));
        // }
    }
}




