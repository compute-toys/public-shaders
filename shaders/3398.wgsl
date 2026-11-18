fn sd_circle(p: vec2f, r: f32) -> f32 {
    return length(p) - r;
}

// iq smooth-min
fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(.5 + .5 * (b - a) / k, 0., 1.);
    return mix(a, b, h) - k * h * (1. - h);
}

fn rot(p: vec2f, a: f32) -> vec2f {
    let c = cos(a);
    let s = sin(a);
    return mat2x2f(c, s, -s, c) * p;
}

// Thisis a poor man Squircle = intersection of 4 circles centred at (+-s, +-s).
fn sd_squircle(p: vec2f, s: f32, a_diag: f32, k: f32) -> f32 {
    let Rc = sqrt(2. * (a_diag * a_diag + s * s));
    var d = sd_circle(p - vec2f( s,  s), Rc);
    d = smin(d, sd_circle(p - vec2f(-s,  s), Rc), k);
    d = smin(d, sd_circle(p - vec2f( s, -s), Rc), k);
    d = smin(d, sd_circle(p - vec2f(-s, -s), Rc), k);
    return d;
}

// iq 2D rounded box
fn sd_rounded_box(p: vec2f, b: vec2f, r: f32) -> f32 {
    let q = abs(p) - b + r;
    return min(max(q.x, q.y), 0.) + length(max(q, vec2f(0.))) - r;
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);
    let uv = (2. * fragCoord - vec2f(screen_size)) / f32(screen_size.y);
    let aspect = f32(screen_size.x) / f32(screen_size.y);

    let left = uv.x < 0.;
    let top  = uv.y > 0.;
    let cell_c = vec2f(select(aspect * .5, -aspect * .5, left),
                       select(-.5, .5, top));
    let p = (uv - cell_c) * 1.3;

    // the three pure shapes (TL squircle evaluated in two orientations)
    let rot45deg= rot(p, -0.7853982);
    let d_tl     = sd_squircle(rot45deg, 0.5, 0.69, 0.);
    let d_tl_rot = sd_squircle(rot45deg, 0.5, 0.69, 0.); 
    let d_sq = sd_rounded_box(p, vec2f(.6), .15);
    let d_ci = sd_circle(p, .6);

    var d: f32;
    if left && top {
        // TL: poor-mans squircle (4 huge circles)
        d = d_tl;                                          
    } else if !left && top {
        // TR: mix of iq sq + circle
        d = mix(d_sq, d_ci, custom.my_f1);                 
    } else if left && !top {
        // BL: mix of iq sq + TL squircle  
        d = mix(d_sq, d_tl_rot, custom.my_f2);             
    } else {
        // BR: mix of TL squircle + circle
        d = mix(d_tl, d_ci, custom.my_f3);                 
    }

    let yellow = vec3f(0.9882, 0.8784, 0.0235);
    let col = mix(yellow, vec3f(0.),
                  smoothstep(-1.5 / f32(screen_size.y), 1.5 / f32(screen_size.y), d));
    textureStore(screen, id.xy, vec4f(col, 1.));
}