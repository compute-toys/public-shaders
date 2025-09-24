const BLACK: f32 = 4. / 256.;
const WHITE: f32 = 1.0;
const DELTA: f32 = 1.0 / 120.0;
const GRAVITY: vec2f = vec2f(0.0, 980 * DELTA * DELTA);

const N: u32 = 16;

struct Data {
    position: vec2f,
    position_o: vec2f,
    imassive: f32,
}

// when frame is odd: read from DATA0, write to DATA1
#storage DATA array<Data, N>

#dispatch_once init
#workgroup_count init 1 1 1
@compute @workgroup_size(N, 1, 1)
fn init(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if i >= N { return; }

    let center = 0.5 * vec2f(textureDimensions(screen));
    let offset = 32. * f32(i32(i) - i32(N) / 2);

    let data = Data(
        center + offset,
        center + offset,
        1.,
    );
    DATA[i] = data;
}

#workgroup_count update 1 1 1
@compute @workgroup_size(1, 1, 1)
fn update() {
    // apply mouse control
    var data = DATA[0];
    data.position = mix(data.position, vec2f(mouse.pos), custom.easing);
    data.position_o = data.position;
    DATA[0] = data;

    data = DATA[1];
    data.position = vec2f(0., -1.) * custom.dist + DATA[0].position;
    data.position_o = data.position;
    DATA[1] = data;

    // apply force
    for (var i = 1u; i < N; i++) {
        data = DATA[i];

        let temp = data.position;
        data.position += data.position - data.position_o + GRAVITY;
        data.position_o = temp;

        DATA[i] = data;
    }
    
    for (var j = 0u; j < u32(custom.itime); j++) {
        // solve distance constraint
        for (var i = 1u; i < N; i++) {
            var cur = DATA[i];
            var prv = DATA[i - 1];

            let c = length(prv.position - cur.position) - custom.dist;
            let n = normalize(prv.position - cur.position);
            let w = cur.imassive + prv.imassive;

            let dlambda = c / w;

            let dpprv = -prv.imassive * dlambda * n;
            let dpcur = cur.imassive * dlambda * n;

            prv.position += dpprv;
            cur.position += dpcur;

            DATA[i] = cur;
            DATA[i - 1] = prv;
        }

        // solve angle constraint
        for (var i = 0u; i < N - 2; i++) {
            var _1 = DATA[i];
            var _2 = DATA[i + 1];
            var _3 = DATA[i + 2]; 

            let r1 = _2.position - _1.position;
            let r2 = _3.position - _2.position;
            
            // 使用 epsilon 来避免后续的潜在问题
            let epsilon = 1e-6; 
            let l1_sq = dot(r1, r1);
            let l2_sq = dot(r2, r2);

            // 如果长度太小，跳过这个约束
            if (l1_sq < epsilon || l2_sq < epsilon) { continue; }
            
            let l1 = sqrt(l1_sq);
            let l2 = sqrt(l2_sq);

            // --- GRADIENT CALCULATION ---
            let d = l1 * l2;
            let r1_dot_r2 = dot(r1, r2);

            let g1 = (r1_dot_r2 / l1_sq * r1 - r2) / d;
            let g3 = (r1 - r1_dot_r2 / l2_sq * r2) / d;
            let g2 = -(g1 + g3);
            
            // --- SOLVER ---
            let a = r1_dot_r2 / d - cos(0.0);
            
            // --- CRITICAL FIX HERE ---
            let S = dot(g1, g1) * _1.imassive + dot(g2, g2) * _2.imassive + dot(g3, g3) * _3.imassive;
            if (abs(S) < epsilon) { continue; } // 如果分母为零，跳过

            let dl = custom.alpha * -a / S;

            _1.position += dl * _1.imassive * g1;
            _2.position += dl * _2.imassive * g2;
            _3.position += dl * _3.imassive * g3;

            DATA[i] = _1;
            DATA[i + 1] = _2;
            DATA[i + 2] = _3;
        }
    }
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let frag_coord = vec2f(id.xy);
    let center = 0.5 *  vec2f(screen_size);
    let bsize = vec2f(200.);

    var sdf = 1000000.0;
    for (var i = 0u; i < N; i++) {
        let p = DATA[i].position;
        sdf = op_smin(sdf, sdf_circle(frag_coord - p, custom.radius / (0.1 * f32(i) + 1.)), custom.k);
    }
    // for (var i = 0u; i < N; i++) {
    //     let p = DATA[i].position;
    //     sdf = min(sdf, sdf_circle(frag_coord - p, custom.radius));
    // }

    let shape = linearstep(1., -1., sdf);

    var col = vec3f(mix(BLACK, WHITE, shape));

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn sdf_circle(p: vec2f, r: f32) -> f32 {
    return length(p) - r;
}

fn sdf_rect(p: vec2f, b: vec2f) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0);
}

fn op_smin(a: f32, b: f32, k: f32) -> f32 {
    let v = exp2(-a / k) + exp2(-b / k);
    return -k * log2(v);
}

fn op_shell(d: f32, t: f32) -> f32 {
	return abs(d) - t;
}

fn op_smax(a: f32, b:  f32, k: f32) -> f32 {
    return - op_smin(-a, -b, k);
}

fn sdf_segment(p: vec2f, a: vec2f, b: vec2f, r: f32) -> f32 {
    let pa = p - a;
    let ba = b - a;

    let h = saturate(dot(pa, ba) / dot(ba, ba));

    return length(pa - ba * h) - r;
}

fn linearstep(e0: f32, e1: f32, x: f32) -> f32 {
    return saturate((x - e0) / (e1 - e0));
}

fn op_sub(d1: f32, d2: f32) -> f32 {
	return max(d1, -d2);
}

fn cmul(a: vec2f, b: vec2f) -> vec2f {
    return vec2f(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x,
    );
}