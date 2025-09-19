const BLACK: f32 = 4. / 256.;
const WHITE: f32 = 1.0;

const N: u32 = 4;

// when frame is odd: read from DATA0, write to DATA1
#storage DATA array<vec2f, N>

#workgroup_count update 1 1 1
@compute @workgroup_size(1, 1, 1)
fn update() {
    DATA[0] = mix(DATA[0], vec2f(mouse.pos), custom.easing);

    for (var i = 1u; i < N; i++) {
        let prv = DATA[i-1];
        let cur = DATA[i];

        DATA[i] = prv + normalize(cur - prv) * custom.dist;
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
        let p = DATA[i];
        sdf = min(sdf, sdf_circle(frag_coord - p, custom.radius));
    }
    for (var i = 0u; i < N - 1; i++) {
        let pa = DATA[i];
        let pb = DATA[i + 1];
        sdf = min(sdf, sdf_segment(frag_coord, pa, pb, 0.2 * custom.radius));
    }


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