const GSIZE: f32 = 128.0;

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let frag_coord = vec2f(id.xy);
    let cell_coord = modulo2(frag_coord, vec2f(GSIZE)) - 0.5 * GSIZE;
    let cell_numbr = floor(frag_coord / GSIZE);

    let R = 0.4 * GSIZE;

    let dir = vec2f(cos(time.elapsed), sin(time.elapsed));
    let str = cell_numbr.x;
    let brt = clamp(0.12 * (cell_numbr.y + 1.), 0.0, 1.0);
    let ishadow = inner_shadow(cell_coord, R, brt, vec2f(8, -4), str * dir);

    var col = vec3f(ishadow);

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn inner_shadow(
    p: vec2f,
    r: f32,
    brightness: f32,
    blur: vec2f,
    offset: vec2f,
) -> f32 {
    let mask = linearstep(1., -1., sdf_pentagram(p, r));
    let ishadow = brightness * smoothstep(blur.x, blur.y, sdf_pentagram(p + offset, r));

    return mix(1., ishadow, mask);
}

fn sdf_pentagram(p: vec2f, r: f32) -> f32 {
    let v1 = vec2f( 0.809016994, -0.587785252 ); // cos(π/5), -sin(π/5)
    let v2 = vec2f(-0.809016994, -0.587785252 );
    let v3 = vec2f( 0.309016994, -0.951056516 ); // sin(π/10), -cos(π/10)
    let k1z = 0.726542528; // tan(π/5)

    var q = vec2f(abs(p.x), p.y);
    q -= 2.0 * max(dot(v1, q), 0.0) * v1;
    q -= 2.0 * max(dot(v2, q), 0.0) * v2;
    q = vec2f(abs(q.x), q.y - r);

    let proj = clamp(dot(q, v3), 0.0, k1z * r);
    let d = length(q - v3 * proj);
    let s = sign(q.y * v3.x - q.x * v3.y);

    return d * s;
}

fn modulo(a: f32, b: f32) -> f32 {
    return a - b * floor(a / b);
}

fn modulo2(a: vec2f, b: vec2f) -> vec2f {
    return a - b * floor(a / b);
}

fn linearstep(e0: f32, e1: f32, x: f32) -> f32 {
    return clamp((x - e0) / (e1 - e0), 0.0, 1.0);
}