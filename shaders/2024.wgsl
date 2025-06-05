const PI: f32 = radians(180.);
const SIZE: vec2u = vec2u(150, 100);
const COUNT: vec2u = vec2u(6, 5);

const COLOR_0: vec3f =vec3f(0.18);
const COLOR_1: vec3f = vec3f(0.92);

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let size_f = vec2f(SIZE);

    let frag_id = id.xy / SIZE;
    let frag_idx = frag_id.y * COUNT.x + frag_id.x;

    let frag_coord = vec2f(id.xy) % size_f - 0.5 * size_f;

    let t = time.elapsed % custom.gap / custom.gap;
    let n = floor(time.elapsed / custom.gap);
    let se = vec2f(-40., 40.) * select(-1., 1., n % 2. == 0.);

    var x = mix(se.x, se.y, t);
    // sine
    if frag_idx == 0 {
        x = mix(se.x, se.y, i_ease_sine(t));
    } else if frag_idx == 1 {
        x = mix(se.x, se.y, o_ease_sine(t));
    } else if frag_idx == 2 {
        x = mix(se.x, se.y, io_ease_sine(t));
    }
    // quad
    else if frag_idx == 3 {
        x = mix(se.x, se.y, i_ease_quad(t));
    } else if frag_idx == 4 {
        x = mix(se.x, se.y, o_ease_quad(t));
    } else if frag_idx == 5 {
        x = mix(se.x, se.y, io_ease_quad(t));
    }
    // cubic
    else if frag_idx == 6 {
        x = mix(se.x, se.y, i_ease_cubic(t));
    } else if frag_idx == 7 {
        x = mix(se.x, se.y, o_ease_cubic(t));
    } else if frag_idx == 8 {
        x = mix(se.x, se.y, io_ease_cubic(t));
    }
    // quart
    else if frag_idx == 9 {
        x = mix(se.x, se.y, i_ease_quart(t));
    } else if frag_idx == 10 {
        x = mix(se.x, se.y, o_ease_quart(t));
    } else if frag_idx == 11 {
        x = mix(se.x, se.y, io_ease_quart(t));
    }
    // quint
    else if frag_idx == 12 {
        x = mix(se.x, se.y, i_ease_quint(t));
    } else if frag_idx == 13 {
        x = mix(se.x, se.y, o_ease_quint(t));
    } else if frag_idx == 14 {
        x = mix(se.x, se.y, io_ease_quint(t));
    }
    // expo
    else if frag_idx == 15 {
        x = mix(se.x, se.y, i_ease_expo(t));
    } else if frag_idx == 16 {
        x = mix(se.x, se.y, o_ease_expo(t));
    } else if frag_idx == 17 {
        x = mix(se.x, se.y, io_ease_expo(t));
    }
    // circ
    else if frag_idx == 18 {
        x = mix(se.x, se.y, i_ease_circ(t));
    } else if frag_idx == 19 {
        x = mix(se.x, se.y, o_ease_circ(t));
    } else if frag_idx == 20 {
        x = mix(se.x, se.y, io_ease_circ(t));
    }
    // back
    else if frag_idx == 21 {
        x = mix(se.x, se.y, i_ease_back(t));
    } else if frag_idx == 22 {
        x = mix(se.x, se.y, o_ease_back(t));
    } else if frag_idx == 23 {
        x = mix(se.x, se.y, io_ease_back(t));
    }
    // elastic
    else if frag_idx == 24 {
        x = mix(se.x, se.y, i_ease_elastic(t));
    } else if frag_idx == 25 {
        x = mix(se.x, se.y, o_ease_elastic(t));
    } else if frag_idx == 26 {
        x = mix(se.x, se.y, io_ease_elastic(t));
    }
    // bounce
    else if frag_idx == 27 {
        x = mix(se.x, se.y, i_ease_bounce(t));
    } else if frag_idx == 28 {
        x = mix(se.x, se.y, o_ease_bounce(t));
    } else if frag_idx == 29 {
        x = mix(se.x, se.y, io_ease_bounce(t));
    }

    var d = sdf_circle(frag_coord - vec2f(x, 0.), 8.0);
    d = smoothstep(-1., 1., d);

    var col = mix(COLOR_0, COLOR_1, d);

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn sdf_circle(p: vec2f, r: f32) -> f32 {
    return length(p) - r;
}

fn normalize_range(a: f32, b: f32, x: f32) -> f32 {
    return clamp((x - a) / (b - a), 0., 1.);
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