const BLACK: f32 = 4. / 256.;
const WHITE: f32 = 1.0;
const DELTA: f32 = 1.0 / 120.0;
const GRAVITY: vec2f = vec2f(0.0, 980. * DELTA * DELTA);

const N: u32 = 2;

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
        select(1.0, 1.0, i != 0),
    );
    DATA[i] = data;
}

#workgroup_count update 1 1 1
@compute @workgroup_size(1, 1, 1)
fn update() {
    // apply mouse control
    var data_0 = DATA[0];
    data_0.position = mix(data_0.position, vec2f(mouse.pos), custom.easing);
    data_0.position_o = data_0.position;
    DATA[0] = data_0;

    // apply force
    for (var i = 1u; i < N; i++) {
        var data = DATA[i];

        let temp = data.position;
        data.position += data.position - data.position_o + GRAVITY;
        data.position_o = temp;

        DATA[i] = data;
    }

    var data = DATA[1];
    let to_position = vec2f(0., -1.) * custom.dist + DATA[0].position;
    data.position = mix(data.position, to_position, custom.stiffness * DELTA);
    DATA[1] = data;
    
    // solve constraint
    for (var j = 0u; j < u32(custom.itime); j++) {
        for (var i = 1u; i < N; i++) {
            var cur = DATA[i];
            var prv = DATA[i - 1];

            let c = length(prv.position - cur.position) - custom.dist;
            let n = normalize(prv.position - cur.position);
            let w = cur.imassive + prv.imassive;

            let dlambda = 0.80 * (custom.alpha * c / w);

            let dpprv = -prv.imassive * dlambda * n;
            let dpcur = cur.imassive * dlambda * n;

            prv.position += dpprv;
            cur.position += dpcur;

            DATA[i] = cur;
            DATA[i - 1] = prv;
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
        sdf = op_smin(sdf, sdf_circle(frag_coord - p, custom.radius / (0.75 * f32(i) + 1.)), custom.k);
    }
    let rect = sdf_rect(frag_coord - center, bsize);
    sdf = op_smax(sdf, rect, 0.16 * custom.radius);
    // for (var i = 0u; i < N; i++) {
    //     let p = DATA[i].position;
    //     sdf = min(sdf, sdf_circle(frag_coord - p, custom.radius));
    // }

    let shape = linearstep(1., -1., sdf);
    let shape2 = linearstep(1., -1., op_shell(rect, 4.0));

    var col = vec3f(mix(BLACK, WHITE, shape)) + mix(vec3f(BLACK), vec3f(0.1, 0.9, 0.1), shape2);

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