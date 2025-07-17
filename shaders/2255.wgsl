const R: f32 = 128.0;
const PI: f32 = 3.1415926;

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let frag_coord = vec2f(id.xy);
    let size = vec2f(f32(screen_size.x) / 2., f32(screen_size.y));
    let cell_coord = frag_coord % size - 0.5 * size;

    let t = time.elapsed;
    let l = vec2f(cos(t), sin(t));
    let n = n_circle(cell_coord);

    var sdf = sdf_circle(cell_coord, R);
    let shape = linearstep(1., -1., sdf);
    sdf = op_shell(sdf, custom.thickness);
    let edge = linearstep(1., -1., sdf);

    var rim = 0.0;
    if frag_coord.x / size.x < 1. {
        rim = pow(abs(dot(n, l)), custom.strength);
    } else {
        sdf = op_shell(sdf_line(cell_coord, l.x, l.y, 0.), custom.width);
        rim = pow(smoothstep(0, -custom.width, sdf), custom.strength);
    }

    var col = vec3f(edge * rim + 0.04 * shape);
    if mouse.click == 1 {
        col = vec3f(rim + 0.04 * shape);
    }

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn sdf_line(p: vec2f, a: f32, b: f32, c: f32) -> f32 {
    return (a * p.x + b * p.y + c) / length(vec2f(a, b));
}

fn sdf_circle(p: vec2f, r: f32) -> f32 {
    return length(p) - r;
}

fn n_circle(p: vec2f) -> vec2f {
    return normalize(p);
}

fn op_shell(d: f32, t: f32) -> f32 {
    return abs(d) - t;
}

fn linearstep(e0: f32, e1: f32, x: f32) -> f32 {
    return clamp((x - e0) / (e1 - e0), 0.0, 1.0);
}