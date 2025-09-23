const BLACK: f32 = 4. / 256.;
const WHITE: f32 = 1.0;
const DELTA: f32 = 1.0 / 120.0;
const GRAVITY: vec2f = vec2f(0.0, 400 * DELTA * DELTA);

const N: u32 = 16;

struct Data {
    position: vec2f,
    position_o: vec2f,
    imassive: f32,
}

// when frame is odd: read from DATA0, write to DATA1
#storage DATA array<Data, N>
#storage LAMBDA array<f32, N - 1>

#dispatch_once init
#workgroup_count init 1 1 1
@compute @workgroup_size(N, 1, 1)
fn init(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if i >= N { return; }

    let center = 0.5 * vec2f(textureDimensions(screen));

    let data = Data(
        center,
        center,
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
    DATA[0] = data;

    // apply force
    for (var i = 1u; i < N; i++) {
        data = DATA[i];

        let temp = data.position;
        data.position += (data.position - data.position_o) * 0.99 + GRAVITY;
        data.position_o = temp;

        DATA[i] = data;
    }
    
    let aalpha = custom.aalpha;
    for (var i = 0u; i < N - 1; i++) {
        LAMBDA[i] = 0.;
    }
    // solve constraint
    for (var j = 0u; j < 16; j++) {
        for (var i = 0u; i < N - 1u; i++) {
            var cur = DATA[i];
            var nxt = DATA[i + 1];

            let c = length(cur.position - nxt.position) - custom.dist;
            let lambda = LAMBDA[i];
            let w = cur.imassive + nxt.imassive;
            let n = normalize(cur.position - nxt.position);

            let dlambda = - (c + aalpha * lambda) / (w + aalpha);
            let dpcur = cur.imassive * dlambda * n;
            let dpnxt = -nxt.imassive * dlambda * n;

            cur.position += dpcur;
            nxt.position += dpnxt;

            LAMBDA[i] = lambda + dlambda;
            DATA[i] = cur;
            DATA[i + 1] = nxt;
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

    var sdf = 1000000.0;
    for (var i = 0u; i < N; i++) {
        let p = DATA[i].position;
        sdf = min(sdf, sdf_circle(frag_coord - p, custom.radius));
    }
    for (var i = 0u; i < N - 1; i++) {
        let pa = DATA[i].position;
        let pb = DATA[i + 1].position;
        sdf = min(sdf, sdf_segment(frag_coord, pa, pb, 0.2 * custom.radius));
    }

    let shape = linearstep(1., -1., sdf);

    var col = vec3f(mix(BLACK, WHITE, shape));

    if id.y < 16 {
        let i = id.x / 16;
        let lambda = LAMBDA[i];
        col = vec3f(abs(lambda) / 4.);
    }

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn sdf_circle(p: vec2f, r: f32) -> f32 {
    return length(p) - r;
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