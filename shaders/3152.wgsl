#define COUNT 3
#define ITERATIONS (COUNT * 4 * 16 * 16 * 163)
#workgroup_count attractors COUNT 4 6

struct Params { a: f32, b: f32, c: f32, d: f32, t: f32 };
#storage param Params
#storage buf array<array<atomic<u32>,3>>

fn pcg3d(vin: vec3u) -> vec3u {
    var v = vin * 1664525u + 1013904223u;
    v.x += v.y*v.z; v.y += v.z*v.x; v.z += v.x*v.y;
    v ^= v >> vec3u(16u);
    v.x += v.y*v.z; v.y += v.z*v.x; v.z += v.x*v.y;
    return v;
}

fn pcg3df(vin: vec3u) -> vec3f {
    return vec3f(pcg3d(vin)) / f32(0xffffffffu);
}


fn iterate(aid: u32, x: f32, y: f32,
           a: f32, b: f32, c: f32, d: f32) -> vec2f {
    switch aid {
        case 0u: {  // De Jong
            return vec2f(sin(a*y) - cos(b*x),
                         sin(c*x) - cos(d*y));
        }
        case 1u: {  // Clifford
            return vec2f(sin(a*y) + c*cos(a*x),
                         sin(b*x) + d*cos(b*y));
        }
        case 2u: {  // Svensson
            return vec2f(d*sin(a*x) - sin(b*y),
                         c*cos(a*x) + cos(b*y));
        }
        case 3u: {  // Bedhead
            return vec2f(sin(x*y/b)*y + cos(a*x - y),
                         x + sin(y)/b);
        }
        case 4u: {  // Hopalong (Martin)
            let s = select(-1., 1., x >= 0.);
            return vec2f(y - s*sqrt(abs(b*x - c)),
                         a - x);
        }
        case 5u: {  // Tinkerbell
            return vec2f(x*x - y*y + a*x + b*y,
                         2.*x*y   + c*x + d*y);
        }
        default: { return vec2f(0.); }
    }
}

// pass 1: clear buffer + advance time
@compute @workgroup_size(16, 16)
fn clear(@builtin(global_invocation_id) id: vec3u) {
    let scsz = textureDimensions(screen);
    if (id.x >= scsz.x || id.y >= scsz.y) { return; }
    let idx = id.x + id.y * scsz.x;
    atomicStore(&buf[idx][0], 0u);
    atomicStore(&buf[idx][1], 0u);
    atomicStore(&buf[idx][2], 0u);

    if (id.x != 0u || id.y != 0u) { return; }
    param.t += pow(10., custom.animationSpeed / 3.33 - 5.);
    let t = custom.timeOffset + param.t;
    param.a = 4.*sin(t*1.03);
    param.b = 4.*sin(t*1.07);
    param.c = 4.*sin(t*1.09);
    param.d = 4.*sin(t*1.13);
}

// pass 2: iterate all 6 attractors
@compute @workgroup_size(16, 16)
fn attractors(@builtin(global_invocation_id) id: vec3u) {
    let scsz = textureDimensions(screen);
    let aid  = id.z;                        // 0‥5 from z-dispatch

    // 3×2 grid geometry
    let cell_w = scsz.x / 3u;
    let cell_h = scsz.y / 2u;
    let txsz   = min(cell_w, cell_h);
    let cx     = aid % 3u;
    let cy     = 1u - aid / 3u;             // flip so row 0 = top
    let x_base = cx * cell_w + (cell_w - txsz) / 2u;
    let y_base = cy * cell_h + (cell_h - txsz) / 2u;

    let rnd = pcg3df(vec3u(id.xy, u32(time.frame) * 7u + aid));
    let t   = custom.timeOffset + param.t;

    // per-attractor: params, start point, viewport
    var a = 0.; var b = 1.; var c = 0.; var d = 0.;
    var x1 = 0.; var y1 = 0.;
    var scale = 0.25;
    var off_x = 0.; var off_y = 0.;

    switch aid {
        case 0u: {  // De Jong  — original params verbatim
            a = param.a; b = param.b; c = param.c; d = param.d;
            x1 = 2.*sin(6.28*rnd.x);
            y1 = 2.*sin(6.28*rnd.y);
            scale = 0.25;
        }
        case 1u: {  // Clifford — smaller range for c,d (amplitudes)
            a = 2.0*sin(t*0.97);  b = 2.0*sin(t*1.01);
            c = 1.5*sin(t*1.11);  d = 1.5*sin(t*1.17);
            x1 = 2.*sin(6.28*rnd.x);
            y1 = 2.*sin(6.28*rnd.y);
            scale = 0.19;
        }
        case 2u: {  // Svensson
            a = 2.5*sin(t*1.05);  b = 2.5*sin(t*0.93);
            c = 2.5*sin(t*1.15);  d = 2.5*sin(t*0.89);
            x1 = 2.*sin(6.28*rnd.x);
            y1 = 2.*sin(6.28*rnd.y);
            scale = 0.14;
        }
        case 3u: {  // Bedhead — keep |b| away from 0
            a = 0.8*sin(t*1.03);
            b = -0.6 - 0.3*abs(sin(t*1.1));   // b ∈ [−0.9, −0.6]
            x1 = rnd.x*2. - 1.;
            y1 = rnd.y*2. - 1.;
            scale = 0.08;
        }
        case 4u: {  // Hopalong — keep b > 0 for nice spirals
            a = 4.*sin(t*0.97);
            b = 1. + 0.5*sin(t*1.07);         // b ∈ [0.5, 1.5]
            c = 3.*sin(t*1.13);
            x1 = rnd.x*2. - 1.;
            y1 = rnd.y*2. - 1.;
            scale = 0.015;
        }
        case 5u: {  // Tinkerbell — small perturbation of known-stable set
            a =  0.9    + 0.08*sin(t*1.03);
            b = -0.6013 + 0.05*sin(t*1.07);
            c =  2.0    + 0.20*sin(t*1.09);
            d =  0.5    + 0.10*sin(t*1.13);
            x1 = -0.72 + 0.2*(rnd.x - 0.5);  // start near attractor
            y1 = -0.64 + 0.2*(rnd.y - 0.5);
            scale = 0.25;
            off_x = 0.5;  off_y = 0.25;       // re-centre in cell
        }
        default: {}
    }

    // warmup
    for (var i = 0; i < 64; i++) {
        let p = iterate(aid, x1, y1, a, b, c, d);
        x1 = p.x;  y1 = p.y;
        if (abs(x1) > 1e4 || abs(y1) > 1e4) {
            x1 = 0.1*rnd.x;  y1 = 0.1*rnd.y;
        }
    }

    // main accumulation
    for (var i = 0; i < 163; i++) {
        let p  = iterate(aid, x1, y1, a, b, c, d);
        let x2 = p.x;
        let y2 = p.y;

        // divergence bailout
        if (abs(x2) > 1e4 || abs(y2) > 1e4) {
            x1 = 0.1*rnd.x;  y1 = 0.1*rnd.y;
            continue;
        }

        let fx = (x2 + off_x) * scale * f32(txsz) * 0.96
               + f32(txsz) * 0.5;
        let fy = (y2 + off_y) * scale * f32(txsz) * 0.96
               + f32(txsz) * 0.5;

        if (fx >= 0. && fx < f32(txsz) &&
            fy >= 0. && fy < f32(txsz))
        {
            let sx  = x_base + u32(fx);
            let sy  = y_base + u32(fy);
            let idx = sy * scsz.x + sx;
            let dx  = x2 - x1;
            let dy  = y2 - y1;
            atomicAdd(&buf[idx][0], u32(256.*min(abs(dx), 4.)));
            atomicAdd(&buf[idx][1], u32(256.*min(abs(dy), 4.)));
            atomicAdd(&buf[idx][2], 256u);
        }
        x1 = x2;  y1 = y2;
    }
}

// pass 3: tone-map to screen
@compute @workgroup_size(16, 16)
fn fragment(@builtin(global_invocation_id) id: vec3u) {
    let scsz = textureDimensions(screen);
    if (id.x >= scsz.x || id.y >= scsz.y) { return; }

    let txsz = min(scsz.x / 3u, scsz.y / 2u);
    let idx  = id.x + (scsz.y - id.y - 1u) * scsz.x;

    var col = vec3f(f32(atomicLoad(&buf[idx][0])),
                    f32(atomicLoad(&buf[idx][1])),
                    f32(atomicLoad(&buf[idx][2])));

    col = col * f32(txsz) * f32(txsz) / (2048. * ITERATIONS);
    col = pow(col, vec3f(1.5));

    textureStore(screen, id.xy, vec4f(col, 1.));
}