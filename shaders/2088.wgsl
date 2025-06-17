const PI: f32 = 3.1415926;
const COUNT: vec2u = vec2u(4, 3);

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let cell_size = screen_size / COUNT;
    let frag_coord = vec2f(id.xy) + vec2f(0.5);

    let cell_id = nrepeat_center(frag_coord, vec2f(cell_size));
    let cell_numbr = cell_id.y * i32(COUNT.x) + cell_id.x;
    let cell_coord = repeat_center(frag_coord, vec2f(cell_size));
    
    let R = 0.5 * 0.75 * f32(cell_size.y);
    let sdf = sdf_circle(cell_coord, R);
    let cmask = linearstep(1., -1., sdf);

    let t = atan2(cell_coord.y, cell_coord.x);
    let r = length(cell_coord);
    let ppt = pingpong(t, PI);

    let tms = floor(time.elapsed);
    let tm2 = modulo(tms, 2.);
    let tm3 = modulo(tms, 3.);
    var ratio = r / R;
    if cell_numbr == 0 {
        ratio = tm2;
    } else if cell_numbr == 1 {
        ratio = mix(1. - ratio, ratio, tm2);
    } else if cell_numbr == 2 {
        if tm3 == 0 {
            ratio = i_ease_sine(ratio);
        } else if tm3 == 1 {
            ratio = o_ease_sine(ratio);
        } else {
            ratio = io_ease_sine(ratio);
        }
    } else if cell_numbr == 3 {
        if tm3 == 0 {
            ratio = i_ease_quad(ratio);
        } else if tm3 == 1 {
            ratio = o_ease_quad(ratio);
        } else {
            ratio = io_ease_quad(ratio);
        }
    } else if cell_numbr == 4 {
        if tm3 == 0 {
            ratio = i_ease_cubic(ratio);
        } else if tm3 == 1 {
            ratio = o_ease_cubic(ratio);
        } else {
            ratio = io_ease_cubic(ratio);
        }
    } else if cell_numbr == 5 {
        if tm3 == 0 {
            ratio = i_ease_quart(ratio);
        } else if tm3 == 1 {
            ratio = o_ease_quart(ratio);
        } else {
            ratio = io_ease_quart(ratio);
        }
    } else if cell_numbr == 6 {
        if tm3 == 0 {
            ratio = i_ease_quint(ratio);
        } else if tm3 == 1 {
            ratio = o_ease_quint(ratio);
        } else {
            ratio = io_ease_quint(ratio);
        }
    } else if cell_numbr == 7 {
        if tm3 == 0 {
            ratio = i_ease_expo(ratio);
        } else if tm3 == 1 {
            ratio = o_ease_expo(ratio);
        } else {
            ratio = io_ease_expo(ratio);
        }
    } else if cell_numbr == 8 {
        if tm3 == 0 {
            ratio = i_ease_circ(ratio);
        } else if tm3 == 1 {
            ratio = o_ease_circ(ratio);
        } else {
            ratio = io_ease_circ(ratio);
        }
    } else if cell_numbr == 9 {
        if tm3 == 0 {
            ratio = i_ease_back(ratio);
        } else if tm3 == 1 {
            ratio = o_ease_back(ratio);
        } else {
            ratio = io_ease_back(ratio);
        }
    } else if cell_numbr == 10 {
        if tm3 == 0 {
            ratio = i_ease_elastic(ratio);
        } else if tm3 == 1 {
            ratio = o_ease_elastic(ratio);
        } else {
            ratio = io_ease_elastic(ratio);
        }
    } else {
        if tm3 == 0 {
            ratio = i_ease_bounce(ratio);
        } else if tm3 == 1 {
            ratio = o_ease_bounce(ratio);
        } else {
            ratio = io_ease_bounce(ratio);
        }
    }

    // let cell_uv = cell_coord / f32(cell_size.y);
    let samp_uv = vec2f(0.5 * r / R, fract(time.elapsed) * 0.5);

    var col = textureSampleLevel(channel0, bilinear_repeat, samp_uv, 0).rrr;

    let rad = mix(custom.st, custom.ed, ratio);
    let pmask = linearstep(rad, 0., ppt);

    col *= pmask * cmask;

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn modulo(a: f32, b: f32) -> f32 {
    return a - b * floor(a / b);
}

fn modulo2(a: vec2f, b: vec2f) -> vec2f {
    return a - b * floor(a / b);
}

fn pingpong(x: f32, period: f32) -> f32 {
    return abs(modulo(x + 0.5 * period, period) - 0.5 * period);
}

fn rot2d_mat(t: f32) -> mat2x2f {
    let s = sin(t);
    let c = cos(t);

    return mat2x2f(c, s, -s, c);
}

fn repeat_corner(p: vec2f, s: vec2f) -> vec2f {
    return modulo2(p, s);
}

fn repeat_center(p: vec2f, s: vec2f) -> vec2f {
    return modulo2(p, s) - 0.5 * s;
}

fn nrepeat_corner(p: vec2f, s: vec2f) -> vec2i {
    return vec2i(p / s);
}

fn nrepeat_center(p: vec2f, s: vec2f) -> vec2i {
    return nrepeat_corner(p, s);
}

fn sdf_circle(p: vec2f, r: f32) -> f32 {
    return length(p) - r;
}

fn linearstep(e0: f32, e1: f32, x: f32) -> f32 {
    return clamp((x - e0) / (e1 - e0), 0., 1.);
}

fn i_ease_sine(x: f32) -> f32 {
    return sin(0.5 * PI * (x - 1)) + 1;
}

fn o_ease_sine(x: f32) -> f32 {
    return sin(0.5 * PI * x);
}

fn io_ease_sine(x: f32) -> f32 {
    return -0.5 * (cos(PI * x) - 1);
}

fn i_ease_quad(x: f32) -> f32 {
    return x * x;
}

fn o_ease_quad(x: f32) -> f32 {
    return 1 - (1 - x) * (1 - x);
}

fn io_ease_quad(x: f32) -> f32 {
    return select(
        1 - 2 * pow(1 - x, 2),
         2 * x * x,
        x < 0.5,
    );
}

fn i_ease_cubic(x: f32) -> f32 {
    return x * x * x;
}

 fn o_ease_cubic(x: f32) -> f32 {
    let t = 1 - x;
    return 1 - t * t * t;
 }

 fn io_ease_cubic(x: f32) -> f32 {
    return select(
        1 - 4 * pow(1 - x, 3),
        4 * x * x * x,
        x < 0.5,
    );
 }

 fn i_ease_quart(x: f32) -> f32 {
    return pow(x, 4);
 }

 fn o_ease_quart(x: f32) -> f32 { 
    return 1 - pow(1 - x, 4);
}

fn io_ease_quart(x: f32) -> f32 {
    return select(
        1 - 8 * pow(1 - x, 4),
        8 * pow(x, 4),
        x < 0.5,
    );
}

fn i_ease_quint(x: f32) -> f32 {
    return pow(x, 5);
}

fn o_ease_quint(x: f32) -> f32 {
    return 1 - pow(1 - x, 5);
}

fn io_ease_quint(x: f32) -> f32 {
    return select(
        1 - 16 * pow(1 - x, 5),
        16 * pow(x, 5),
        x < 0.5,
    );
}

fn i_ease_expo(x: f32) -> f32 {
    return select(
        pow(2, 10 * (x - 1)),
        x,
        x == 0
    );
}

fn o_ease_expo(x: f32) -> f32 {
    return select(
        1 - pow(2, -10 * x),
        x,
        x == 1,
    );
}

fn io_ease_expo(x: f32) -> f32 {
    return select(
        select(
            -0.5 * pow(2, 10 - (x * 20)) + 1,
            0.5 * pow(2, (20 * x) - 10),
            x < 0.5
        ),
        x,
        x == 0 || x == 1,
    );
}

fn i_ease_circ(x: f32) -> f32 {
    return 1 - sqrt(1 - x * x);
}

fn o_ease_circ(x: f32) -> f32 {
    let t = 1 - x;
    return sqrt(1 - t * t);
}

fn io_ease_circ(x: f32) -> f32 {
    return select(
        0.5 + 0.5 * sqrt(1 - pow(2 - 2 * x, 2)),
        0.5 - 0.5 * sqrt(1 - pow(2 * x, 2)),
        x < 0.5,
    );
}

const C1: f32 = 1.70158;
const C2: f32 = C1 * 1.525;
const C3: f32 = 1 + C1;

fn i_ease_back(x: f32) -> f32 {
    return C3 * pow(x, 3) - C1 * pow(x, 2);
}

fn o_ease_back(x: f32) -> f32 {
    let t = x - 1;
    return 1 + C3 * t * t * t + C1 * t * t;
}

fn io_ease_back(x: f32) -> f32 {
    let t0 = 2 * x - 2;
    let t1 = 2 * x;
    
    return select(
        0.5 * t0 * t0 * ((C2 + 1) * t0 + C2) + 1,
        0.5 * t1 * t1 * ((C2 + 1) * t1 - C2),
        x < 0.5,
    );
}

const C4: f32 = 0.6667 * PI;
const C5: f32 = 0.4444 * PI;

fn i_ease_elastic(x: f32) -> f32 {
    return  select(
        step(0., x),
        -pow(2, 10 * x - 10) * sin((10 * x - 10.75) * C4),
        x >= 0 && x <= 1,
    );
}

fn o_ease_elastic(x: f32) -> f32 {
    return  select(
        step(0., x),
        pow(2, -10 * x) * sin((10 * x - 0.75) * C4) + 1,
        x >= 0 && x <= 1,
    );
}

fn io_ease_elastic(x: f32) -> f32 {
    return  select(
        step(0., x),
        select(
         0.5 * pow(2, -20 * x + 10) * sin((20 * x - 11.125) * C5) + 1,
        -0.5 * pow(2, 20 * x - 10) * sin((20 * x - 11.125) * C5),
        x < 0.5,
        ),
        x >= 0 && x <= 1,
    );
}

fn i_ease_bounce(x: f32) -> f32 {
    return 1 - o_ease_bounce(1 - x);
}

fn o_ease_bounce(x: f32) -> f32 {
    let a = 4.0 / 11.0;
    let b = 8.0 / 11.0;
    let c = 9.0 / 10.0;

    let ca = 4356.0 / 361.0;
    let cb = 35442.0 / 1805.0;
    let cc = 16061.0 / 1805.0;

    let t2 = x * x;

    return select(
      select(
        select(
          10.8 * x * x - 20.52 * x + 10.72,
          ca * t2 - cb * x + cc,
          x < c
        ),
        9.075 * t2 - 9.9 * x + 3.4,
        x < b
      ),
      7.5625 * t2,
      x < a
    );
}

fn io_ease_bounce(x: f32) -> f32 {
  return select(
    0.5 * o_ease_bounce(2 * x - 1) + 0.5,
    0.5 * (1.0 - o_ease_bounce(1 - x * 2)),
    x < 0.5
  );
}