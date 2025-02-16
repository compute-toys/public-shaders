// Port of 
// Smooth Mouse Drawing - fad
// https://www.shadertoy.com/view/dldXR7

// A recreation of https://lazybrush.dulnan.net/

const MAX_FLOAT = bitcast<f32>(0x7f7fffffu);
const PI = 3.1415927;
const TAU = 6.2831853;

struct Mousef {
    pos: vec2f,
    click: int,
}

struct Global {
    _prev_time: Time,
    delta_time: f32,
    mouseA: Mousef,
    mouseB: Mousef,
    mouseC: Mousef,
    sdf_buffer: array<f32>
}

#storage g Global

#workgroup_count update_lazy_mouse 1 1 1
@compute @workgroup_size(1)
fn update_lazy_mouse() {
    g.delta_time = time.elapsed - select(0.0, g._prev_time.elapsed, time.frame > 0);
    g._prev_time = time;
    g.mouseA = g.mouseB;
    g.mouseB = g.mouseC;
    g.mouseC = Mousef(vec2f(mouse.pos) + 0.5, mouse.click);
    let dist = distance(g.mouseB.pos, g.mouseC.pos);

    if time.frame > 0 && g.mouseB.click == 1 && dist > 0.0 {
        let dir = (g.mouseC.pos - g.mouseB.pos) / dist;
        let len = max(dist - custom.lazy_radius, 0.0);
        let iTimeDelta = 1.0 / 120.0;
        let ease = 1.0 - pow(custom.lazy_friction, g.delta_time * 10.0);
        g.mouseC.pos = g.mouseB.pos + dir * len * ease;
    }
}

// solve_quadratic(), solve_cubic(), solve() and sd_bezier() are from
// Quadratic Bezier SDF With L2 - Envy24
// https://www.shadertoy.com/view/7sGyWd
// with modification. Thank you! I tried a lot of different sd_bezier()
// implementations from across Shadertoy (including trying to make it
// myself) and all of them had bugs and incorrect edge case handling
// except this one.

fn solve_quadratic(a: f32, b: f32, c: f32, roots: ptr<function, vec2f>) -> u32 {
    // Return the number of real roots to the equation
    // a*x^2 + b*x + c = 0 where a != 0 and populate roots.
    let discriminant = b * b - 4.0 * a * c;

    if (discriminant < 0.0) {
        return 0;
    }

    if (discriminant == 0.0) {
        (*roots)[0] = -b / (2.0 * a);
        return 1;
    }

    let SQRT = sqrt(discriminant);
    (*roots)[0] = (-b + SQRT) / (2.0 * a);
    (*roots)[1] = (-b - SQRT) / (2.0 * a);
    return 2;
}

fn solve_cubic(a: f32, b: f32, c: f32, d: f32, roots: ptr<function, vec3f>) -> u32 {
    // Return the number of real roots to the equation
    // a*x^3 + b*x^2 + c*x + d = 0 where a != 0 and populate roots.
    let A = b / a;
    let B = c / a;
    let C = d / a;
    let Q = (A * A - 3.0 * B) / 9.0;
    let R = (2.0 * A * A * A - 9.0 * A * B + 27.0 * C) / 54.0;
    let S = Q * Q * Q - R * R;
    let sQ = sqrt(abs(Q));
    *roots = vec3f(-A / 3.0);

    if (S > 0.0) {
        *roots += -2.0 * sQ * cos(acos(R / (sQ * abs(Q))) / 3.0 + vec3(TAU, 0.0, -TAU) / 3.0);
        return 3;
    }
    
    if (Q == 0.0) {
        (*roots)[0] += -pow(C - A * A * A / 27.0, 1.0 / 3.0);
        return 1;
    }
    
    if (S < 0.0) {
        let u = abs(R / (sQ * Q));
        let v = select(sinh(asinh(u) / 3.0), cosh(acosh(u) / 3.0), Q > 0.0);
        (*roots)[0] += -2.0 * sign(R) * sQ * v;
        return 1;
    }
    
    *roots += vec3f(-2.0, 1.0, 0.0) * sign(R) * sQ;
    return 2;
}

fn solve(a: f32, b: f32, c: f32, d: f32, roots: ptr<function, vec3f>) -> u32 {
    // Return the number of real roots to the equation
    // a*x^3 + b*x^2 + c*x + d = 0 and populate roots.
    if (a == 0.0) {
        if (b == 0.0) {
            if (c == 0.0) {
                return 0;
            }
            
            (*roots)[0] = -d/c;
            return 1;
        }
        
        var r: vec2f;
        let num = solve_quadratic(b, c, d, &r);
        *roots = vec3f(r, 0.0);
        return num;
    }
    
    return solve_cubic(a, b, c, d, roots);
}

fn sd_bezier(p: vec2f, a: vec2f, b: vec2f, c: vec2f) -> f32 {
    let A = a - 2.0 * b + c;
    let B = 2.0 * (b - a);
    let C = a - p;
    var T: vec3f;
    let num = solve(
        2.0 * dot(A, A),
        3.0 * dot(A, B),
        2.0 * dot(A, C) + dot(B, B),
        dot(B, C),
        &T
    );
    T = clamp(T, vec3f(0.0), vec3f(1.0));
    var best = 1e30;
    
    for (var i = 0u; i < num; i++) {
        let t = T[i];
        let u = 1.0 - t;
        let d = u * u * a + 2.0 * t * u * b + t * t * c - p;
        best = min(best, dot(d, d));
    }
    
    return sqrt(best);
}

fn sd_segment(p: vec2f, a: vec2f, b: vec2f) -> f32 {
    let ap = p - a;
    let ab = b - a;
    return distance(p, a + ab * clamp(dot(ap, ab) / dot(ab, ab), 0.0, 1.0));
}

@compute @workgroup_size(16, 16)
fn update_sdf(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);

    if any(id.xy >= screen_size) {
        return;
    }

    let sd = &g.sdf_buffer[id.x + screen_size.x * id.y];

    if time.frame == 0 {
        *sd = MAX_FLOAT;
    }

    if mouse.click != 1 || keyDown(65u) {
        return;
    }

    let p = vec2f(id.xy) + 0.5;
    let A = time.frame > 1 && g.mouseA.click == 1;
    let B = time.frame > 0 && g.mouseB.click == 1;
    let C = g.mouseC.click == 1;
    let a = g.mouseA.pos;
    let b = g.mouseB.pos;
    let c = g.mouseC.pos;
    var d: f32;
    
    if !B && C {
        d = distance(p, c);
    } else if !A && B && C {
        d = sd_segment(p, b, mix(b, c, 0.5));
    } else if A && B && C {
        d = sd_bezier(p, mix(a, b, 0.5), b, mix(b, c, 0.5));
    } else if A && B && !C {
        d = sd_segment(p, mix(a, b, 0.5), b);
    }

    d -= custom.brush_radius;
    *sd = min(*sd, d);
}

fn blend_over(front: vec4f, back: vec4f) -> vec4f {
    let a = front.a + back.a * (1.0 - front.a);
    return select(
        vec4f(0.0),
        vec4f((front.rgb * front.a + back.rgb * back.a * (1.0 - front.a)) / a , a),
        a > 0.0
    );
}

fn blend_into(dst: ptr<function, vec4f>, src: vec4f) {
    *dst = blend_over(src, *dst);
}

fn read_buffer(P: vec2i) -> f32 {
    let R = vec2i(textureDimensions(screen));
    let Q = clamp(P, vec2i(0), R - 1);
    return g.sdf_buffer[Q.x + R.x * Q.y];
}

fn sd_drawing(p: vec2f) -> f32 {
    let R = vec2i(textureDimensions(screen));
    let pc = clamp(p, vec2f(0.5), vec2f(R) - 0.5);
    let q = round(pc) - 1.0;
    let w = pc - q - 0.5;
    let P = vec2i(q);
    let s00 = read_buffer(P + vec2(0, 0));
    let s10 = read_buffer(P + vec2(1, 0));
    let s01 = read_buffer(P + vec2(0, 1));
    let s11 = read_buffer(P + vec2(1, 1));
    return mix(mix(s00, s10, w.x), mix(s01, s11, w.x), w.y);
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    let screen_size = textureDimensions(screen);

    if any(id.xy >= screen_size) {
        return;
    }
    
    var p = vec2f(id.xy) + 0.5;
    var color = vec4f(1.0);
    var scale = 1.0;

    if keyDown(65u) {
        scale = 3.0;
        let m = vec2f(mouse.pos) + 0.5;
        p = (p - m) / scale + m;
    }

    let sd = sd_drawing(p) * scale;
    blend_into(&color, vec4f(0.0, 0.0, 0.0, clamp(0.5 - sd, 0.0, 1.0)));
    let spacing = f32(screen_size.y) * 0.02 * scale;
    let thickness = max(f32(screen_size.y) * 0.002, 2.0) * scale;
    let opacity = clamp(
        0.5 + 0.5 * thickness - 
        abs((sd - (spacing - thickness) * 0.5) % spacing - spacing * 0.5), 
        0.0, 1.0
    ) * 0.5 * exp(-sd / f32(screen_size.y) * 8.0);
    blend_into(&color, vec4f(0.0, 0.0, 0.0, opacity));
    textureStore(screen, id.xy, color);
}