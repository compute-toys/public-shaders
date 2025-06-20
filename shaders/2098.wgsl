const PI: f32 = 3.1415926;
const COUNT: vec2u = vec2u(4, 3);

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);
    
    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let frag_coord = vec2f(id.xy) + vec2f(0.5);
    let cell_size = vec2f(screen_size / COUNT);

    var cell_coord = repeat_center(frag_coord, cell_size);
    let r = fract(0.5 * time.elapsed);
    let t = 2.0 * PI * smoothstep(0., 1., r);
    let cell_numbr = nrepeat_corner(frag_coord, cell_size);

    let hsize = cell_size.y * 0.33;
    var sdf = 99.;
    if cell_numbr.y == 0 {
        if cell_numbr.x == 0 {
            sdf = sdf_hspace(cell_coord, 0., 1., 0.);
        } else if cell_numbr.x == 1 {
            sdf = sdf_circle(cell_coord, hsize);
        } else if cell_numbr.x == 2 {
            sdf = sdf_etirangle(cell_coord, hsize);
        } else {
            sdf = sdf_box(cell_coord, vec2f(hsize));
        }
    } else if cell_numbr.y == 1 {
        if cell_numbr.x == 0 {
            sdf = sdf_pentagon(cell_coord, hsize);
        } else if cell_numbr.x == 1 {
            sdf = sdf_hexagon(cell_coord, hsize);
        } else if cell_numbr.x == 2 {
            sdf = sdf_pentagram(cell_coord, hsize);
        } else {
            sdf = sdf_arc(cell_coord, vec2f(cos(-PI / 6.), sin(-PI / 6.)), hsize, 12.);
        }
    } else {
        if cell_numbr.x == 0 {
            sdf = sdf_moon(cell_coord, 0.4 * hsize, hsize, 0.75 * hsize);
        } else if cell_numbr.x == 1 {
            sdf = sdf_egg(cell_coord, hsize, 0.7 * hsize);
        } else if cell_numbr.x == 2 {
            sdf = sdf_flower(cell_coord, hsize, 5);
        } else {
            sdf = sdf_heart(cell_coord, hsize);
        }
    }

    let s = u32(modulo(time.elapsed, 4.) / 2.);
    let a = -fract(time.elapsed) * 16.;
    sdf = select(op_onion(sdf, 8., 4., a), op_lnion(sdf, 8., 4., a), s == 0);
    let shape = linearstep(-1., 1., sdf);

    let grid = grid(repeat_corner(frag_coord, cell_size), cell_numbr);

    var col = vec3f(grid * shape);
    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn sdf_hspace(p: vec2f, a: f32, b: f32, c: f32) -> f32 {
    return (a * p.x + b * p.y + c) / length(vec2f(a, b));
}

fn sdf_line(p: vec2f, a: f32, b: f32, c: f32, t: f32) -> f32 {
    return abs(sdf_hspace(p, a, b, c)) - t;
}

fn sdf_circle(p: vec2f, r: f32) -> f32 {
    return length(p) - r;
}

fn sdf_etirangle(p: vec2f, r: f32) -> f32 {
    const k = 1.7320508; // sqrt(3.0)
    var q = vec2f(abs(p.x) - r, p.y + r / k);
    if (q.x + k * q.y > 0.0) {
        q = 0.5 * vec2f(q.x - k * q.y, -k * q.x - q.y);
    }
    q.x = q.x - clamp(q.x, -2.0 * r, 0.0);
    return -length(q) * sign(q.y);
}

fn sdf_box(p: vec2f, b: vec2f) -> f32 {
    let d = abs(p) - b;
    return length(max(d, vec2f(0.0))) + min(max(d.x, d.y), 0.0);
}

fn sdf_pentagon(p: vec2f, r: f32) -> f32 {
    let k = vec3f(0.809016994, 0.587785252, 0.726542528); // 预计算常量
    var q = vec2f(abs(p.x), p.y);

    q -= 2.0 * min(dot(vec2f(-k.x, k.y), q), 0.0) * vec2f(-k.x, k.y);
    q -= 2.0 * min(dot(vec2f( k.x, k.y), q), 0.0) * vec2f( k.x, k.y);

    q -= vec2f(clamp(q.x, -r * k.z, r * k.z), r);

    return length(q) * sign(q.y);
}

fn sdf_hexagon(p: vec2f, r: f32) -> f32 {
    let k = vec3f(-0.866025404, 0.5, 0.577350269); // -√3/2, 1/2, 1/√3
    var q = abs(p);

    q -= 2.0 * min(dot(k.xy, q), 0.0) * k.xy;
    q -= vec2f(clamp(q.x, -k.z * r, k.z * r), r);

    return length(q) * sign(q.y);
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

fn sdf_arc(p: vec2f, sc: vec2f, ra: f32, rb: f32) -> f32 {
    let q = vec2f(abs(p.x), p.y);
    let c = sc.y * q.x > sc.x * q.y;
    let d = select(abs(length(q) - ra), length(q - sc * ra), c);
    return d - rb;
}

fn sdf_moon(p: vec2f, d: f32, ra: f32, rb: f32) -> f32 {
    var p_local = p;
    p_local.y = abs(p_local.y);
    let a = (ra * ra - rb * rb + d * d) / (2.0 * d);
    let b = sqrt(max(ra * ra - a * a, 0.0));
    
    if (d * (p_local.x * b - p_local.y * a) > d * d * max(b - p_local.y, 0.0)) {
        return length(p_local - vec2f(a, b));
    }
    return max(length(p_local) - ra, -length(p_local - vec2f(d, 0.0)) + rb);
}

fn sdf_egg(p: vec2f, ra: f32, rb: f32) -> f32 {
    let k = sqrt(3.0);
    var px = abs(p.x);
    let r = ra - rb;

    if (p.y < 0.0) {
        return length(vec2f(px, p.y)) - r - rb;
    } else if (k * (px + r) < p.y) {
        return length(vec2f(px, p.y - k * r)) - rb;
    } else {
        return length(vec2f(px + r, p.y)) - 2.0 * r - rb;
    }
}

// Pseudo-SDF: This function does not return a true signed distance.
// It produces a visual approximation of a flower shape for shading/thresholding,
// but the values are not proportional to Euclidean distance from the shape boundary.
fn sdf_flower(p: vec2f, r: f32, n: u32) -> f32 {
    let a = atan2(p.y, p.x);
    return length(p) - (0.5 * abs(cos(0.5 * f32(n) * a)) + 0.5) * r;
}

fn dot2(v: vec2<f32>) -> f32 {
    return dot(v, v);
}

fn signum(x: f32) -> f32 {
    return select(-1.0, 1.0, x >= 0.0);
}

fn sdf_heart(p: vec2f, r: f32) -> f32 {
    var x = abs(p.x);
    var y = r - p.y;
    var rr = r * 2.0;

    if (x + y > rr) {
        let dx = x - rr * 0.25;
        let dy = y - rr * 0.75;
        return sqrt(dot2(vec2<f32>(dx, dy))) - (rr * 0.3536);
    } else {
        let m = 0.5 * max(x + y, 0.0);
        let d1 = dot2(vec2<f32>(x, y - 1.0));
        let d2 = dot2(vec2<f32>(x - m, y - m));
        return sqrt(min(d1, d2)) * signum(x - y);
    }
}

fn repeat_corner(p: vec2f, s: vec2f) -> vec2f {
    return modulo2(p, s);
}

fn repeat_center(p: vec2f, s: vec2f) -> vec2f {
    return repeat_corner(p, s) - 0.5 * s;
}

fn nrepeat_corner(p: vec2f, s: vec2f) -> vec2i {
    return vec2i(p / s);
}

fn nrepeat_center(p: vec2f, s: vec2f) -> vec2i {
    return nrepeat_corner(p, s);
}

fn modulo(a: f32, b: f32) -> f32 {
    return a - b * floor(a / b);
}

fn modulo2(a: vec2f, b: vec2f) -> vec2f {
    return a - b * floor(a / b);
}

fn linearstep(e0: f32, e1: f32, x: f32) -> f32 {
    return clamp((x - e0) / (e1 - e0), 0., 1.);
}

fn grid(cell_coord: vec2f, cell_numbr: vec2i) -> f32 {
    let o = select(vec2f(0.), vec2f(1.), cell_numbr == vec2i(0));
    // hori
    var sdf = sdf_line(cell_coord, 0., 1., o.y, 1.);
    // vert
    sdf = min(sdf, sdf_line(cell_coord, 1., 0., o.x, 1.));

    return step(0.0, sdf);
}

fn op_onion(d: f32, l: f32, t: f32, o: f32) -> f32 {
    return abs(modulo(d + o, 2. * l) - l) - t;
}

fn op_lnion(d: f32, l: f32, t: f32, o: f32) -> f32 {
    let onion = op_onion(d, l, t, o);
    return max(onion, d);
}