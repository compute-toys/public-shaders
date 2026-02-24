#workgroup_count compute_pendulms 1 1 1
#workgroup_count init_pendulums 1 1 1
#workgroup_count main_image 1 1 1
#dispatch_once init_pendulums

const width = 16;
const height = 16;
const pi = 3.141592653;

// #storage pend1_angles array<atomic<i32>>
// #storage pend2_angles array<atomic<i32>>
// #storage pend1_vels array<atomic<i32>>
// #storage pend2_vels array<atomic<i32>>

const G: f32 = 9.81;
const M1: f32 = 1.0;
const M2: f32 = 1.0;
const L1: f32 = 1.0;
const L2: f32 = 1.0;
// const DT: f32 = 0.01;
#define DT time.delta


@compute @workgroup_size(width, height)
fn init_pendulums(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= width || id.y >= height) { return; }
    passStore(0, vec2<i32>(id.xy), vec4(pi_to_normal(map_range(f32(id.x), 0, width, pi/2-0.01, pi/2)), 0.5, pi_to_normal(map_range(f32(id.y), 0, height, pi/2-0.01, pi/2)), 0.5));
    // let c = id.x + (screen_size.y * id.y);
}

fn normal_to_pi(v: f32) -> f32 {
    return map_range(v, 0, 1, -pi, pi);
}
fn pi_to_normal(v: f32) -> f32 {
    return map_range(v, -pi, pi, 0, 1);
}

// State represents: {theta1, theta2, omega1, omega2}
// theta = angle, omega = angular velocity
fn step_double_pendulum(state: vec4<f32>, dt: f32) -> vec4<f32> {
    let G: f32 = 9.81;
    let M1: f32 = 1.0;
    let M2: f32 = 1.0;
    let L1: f32 = 1.0;
    let L2: f32 = 1.0;


    // Runge-Kutta 4th Order Integration
    let k1 = get_derivs(state);
    let k2 = get_derivs(state + 0.5 * dt * k1);
    let k3 = get_derivs(state + 0.5 * dt * k2);
    let k4 = get_derivs(state + dt * k3);

    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

fn get_derivs(s: vec4<f32>) -> vec4<f32> {
        let t1 = s.x; let t2 = s.y;
        let w1 = s.z; let w2 = s.w;
        
        let delta = t1 - t2;
        let den = (2.0 * M1 + M2 - M2 * cos(2.0 * t1 - 2.0 * t2));

        // Angular acceleration for theta1 (alpha1)
        let a1 = (-G * (2.0 * M1 + M2) * sin(t1) - M2 * G * sin(t1 - 2.0 * t2) - 2.0 * sin(delta) * M2 * (w2 * w2 * L2 + w1 * w1 * L1 * cos(delta))) / (L1 * den);
        
        // Angular acceleration for theta2 (alpha2)
        let a2 = (2.0 * sin(delta) * (w1 * w1 * L1 * (M1 + M2) + G * (M1 + M2) * cos(t1) + w2 * w2 * L2 * M2 * cos(delta))) / (L2 * den);

        return vec4<f32>(w1, w2, a1, a2);
};

@compute @workgroup_size(width, height)
fn compute_pendulms(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= width || id.y >= height) { return; }
    let state = passLoad(0, vec2<i32>(id.xy), 0);
        // 1. Read current state
    var t1 = normal_to_pi(state.x);
    var t2 = normal_to_pi(state.z);
    var w1 = normal_to_pi(state.y);
    var w2 = normal_to_pi(state.w);

    let v = step_double_pendulum(vec4f(t1, t2, w1, w2), DT);

    passStore(0, vec2<i32>(id.xy), vec4f(pi_to_normal(v.x), pi_to_normal(v.z), pi_to_normal(v.y), pi_to_normal(v.w)));

    // // 5. Calculate Cartesian Coordinates
    // let x1 = L1 * sin(t1);
    // let y1 = -L1 * cos(t1);
    // let x2 = x1 + L2 * sin(t2);
    // let y2 = y1 - L2 * cos(t2);

    // output[0] = x1;
    // output[1] = y1;
    // output[2] = x2;
    // output[3] = y2;
}

@compute @workgroup_size(16, 16)
fn clear_screen(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    textureStore(screen, id.xy, vec4f(0, 0, 0, 1));
}

@compute @workgroup_size(width, height)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= width || id.y >= height) { return; }
    let v = passLoad(0, vec2<i32>(id.xy), 0);
    let a1 = normal_to_pi(v.x);
    let a2 = normal_to_pi(v.z);

    let sx = f32(screen_size.x/2);
    let sy = f32(screen_size.y/2);
    let x1 = sx + sin(a1)*(sy/2 - 10);
    let y1 = sy + cos(a1)*(sy/2 - 10);
    let x2 = x1 + sin(a2)*(sy/2 - 10);
    let y2 = y1 + cos(a2)*(sy/2 - 10);

    let col = hsl_to_rgb(map_range(f32(id.x), 0, width, 0, 1), map_range(f32(id.x), 0, width, 0, 1), 0.5);
    drawLine(i32(sx), i32(sy), i32(x1), i32(y1), col);
    drawLine(i32(x1), i32(y1), i32(x2), i32(y2), col);

    // let x = sx + u32(cos(a1)*f32(sy/2 - 10));
    // let y = sy + u32(sin(a1)*f32(sy/2 - 10));
    
    // drawLine(i32(sx), i32(sy), i32(x), i32(y));
    // let x1 = x + u32(cos(a2)*f32(sy/2 - 10));
    // let y1 = y + u32(sin(a2)*f32(sy/2 - 10));
    // drawLine(i32(x), i32(y), i32(x1), i32(y1));
    // let c = id.x + (screen_size.y * id.y);
    // let v = bitcast<f32>(atomicLoad(&atomic_storage[c]));
    // textureStore(screen, id.xy, vec4f(v.x, v.z, 1, 1.));
    // drawLine(0, 0, i32(id.x), i32(id.y));

    // // Pixel coordinates (centre of pixel, origin at bottom left)
    // let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);

    // // Normalised pixel coordinates (from 0 to 1)
    // let uv = fragCoord / vec2f(screen_size);

    // // Time varying pixel colour
    // var col = .5 + .5 * cos(time.elapsed + uv.xyx + vec3f(0.,2.,4.));

    // // Convert from gamma-encoded to linear colour space
    // col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
}

fn drawLine(x0: i32, y0: i32, x1: i32, y1: i32, c: vec3f) {
    var dx = abs(x1 - x0);
    var dy = abs(y1 - y0);
    var sx = select(-1, 1, x0 < x1);
    var sy = select(-1, 1, y0 < y1);
    var err = dx - dy;
    
    var x = x0;
    var y = y0;
    
    loop {
        textureStore(screen, vec2(x, y), vec4f(c.xyz, 1)); // Placeholder for plotting, e.g., writing to a storage buffer
        
        if (x == x1 && y == y1) { break; }
        
        let e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x += sx;
        }
        if (e2 < dx) {
            err += dx;
            y += sy;
        }
    }
}

fn map_range(v: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float {
    return out_min + (out_max - out_min) * (v - in_min) / (in_max - in_min);
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> vec3<f32> {
    var r: f32;
    var g: f32;
    var b: f32;

    if (s == 0.0) {
        r = l;
        g = l;
        b = l;
    } else {
        let q = select(l * (1.0 + s), l + s - l * s, l < 0.5);
        let p = 2.0 * l - q;

        r = hue_to_rgb_helper(p, q, h + 1.0/3.0);
        g = hue_to_rgb_helper(p, q, h);
        b = hue_to_rgb_helper(p, q, h - 1.0/3.0);
    }

    return vec3<f32>(r, g, b);
}

fn hue_to_rgb_helper(p_val: f32, q_val: f32, t_val: f32) -> f32 {
            var t = t_val;
            if (t < 0.0) { t += 1.0; }
            if (t > 1.0) { t -= 1.0; }
            if (t < 1.0/6.0) { return p_val + (q_val - p_val) * 6.0 * t; }
            if (t < 1.0/2.0) { return q_val; }
            if (t < 2.0/3.0) { return p_val + (q_val - p_val) * (2.0/3.0 - t) * 6.0; }
            return p_val;
        }