const mx = 1366.483*5;
const mn = 589.15;
const size: f32 = 5;


@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);
    let aspect = f32(screen_size.y) / f32(screen_size.x);


    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);

    // var uv_raw = vec2f((fragCoord / vec2f(screen_size)) * vec2f(1, (f32(screen_size.y)/f32(screen_size.x))));
    var uv_raw = fragCoord / vec2f(screen_size);
    var uv_a = vec2f((fragCoord / vec2f(screen_size)) * vec2f(1, (f32(screen_size.y)/f32(screen_size.x))));
    var uv = vec2f(remap_f32(uv_a.x, 0, 1, -1, 1), remap_f32(uv_a.y - pow(uv_a.y, 2)/2, 0, 1, -1, 1));
    let tri_coord = vec2f(remap_f32(uv_raw.x, 0, 1, -size, size), remap_f32(uv_raw.y, 0, 1, -size * aspect, size * aspect)); 
    // Time varying pixel colour
    // var col = .5 + .5 * cos(time.elapsed + uv.xyx + vec3f(0.,2.,4.));
    // var col = blackbody((uv.x * 39000) + 1000);
    // var col = vec3f(curlNoise2D(vec3f(uv.xy*3, time.elapsed)).xy, 0);

    let rate: f32 = 6;
    let coord = vec2f(uv.x, uv.y) * 8 + vec2f(0, -time.elapsed*rate);
    var val = ( snoise(vec3f(coord + curlNoise2D(vec3f(coord, time.elapsed * (rate/2))).xy/5, 0)) + 1 ) / 2;
    // var fade =  pow(clamp(remap_f32(distance(uv, vec2f(0, -0.55)), 0, 0.3, 1, 0), 0, 1), 0.8);
    let fade = remap_f32(clamp(sdf(vec2f(tri_coord.x, tri_coord.y/1.3), 0.75), -0.5, 0.5), -0.5, 0.5, 1, 0);
    
    var col = vec3f(val * fade);
    
    col = pow(col, vec3f(2.2))*10;
    col = blackbody(remap_f32(col.x, 0, 10, mn, mx)) * col.x;
    col = pow(col, vec3f(2.2));
    // col = vec3f(remap_f32(clamp(dist, -0.5, 0.5), -0.5, 0.5, 1, 0));
    

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn remap_f32(x: f32, in_min: f32, in_max: f32, out_min: f32, out_max: f32) -> f32 {
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

fn sdf(p: vec2f, test: f32) -> f32 {
    return sd_equilateral_triangle(p, 1 - test) - test;
}


// --- SECTION 1: Underlying 3D Noise (Simplex/Glow variant) ---
// Returns a value between -1.0 and 1.0
fn permute4(x: vec4f) -> vec4f { return ((x * 34.0) + 1.0) * x % vec4f(289.0); }
fn taylorInvSqrt(r: vec4f) -> vec4f { return 1.79284291400159 - 0.85373472095314 * r; }

fn snoise(v: vec3f) -> f32 {
    let C = vec2f(1.0/6.0, 1.0/3.0);
    let D = vec4f(0.0, 0.5, 1.0, 2.0);

    // First corner
    var i: vec3f  = floor(v + dot(v, C.yyy));
    let x0: vec3f = v - i + dot(i, C.xxx);

    // Other corners
    let g: vec3f = step(x0.yzx, x0.xyz);
    let l: vec3f = 1.0 - g;
    let i1: vec3f = min(g.xyz, l.zxy);
    let i2: vec3f = max(g.xyz, l.zxy);

    let x1: vec3f = x0 - i1 + C.xxx;
    let x2: vec3f = x0 - i2 + C.yyy;
    let x3: vec3f = x0 - D.yyy;

    // Permutations
    i = i % vec3f(289.0);
    let p = permute4(permute4(permute4(
                i.z + vec4f(0.0, i1.z, i2.z, 1.0))
            + i.y + vec4f(0.0, i1.y, i2.y, 1.0))
        + i.x + vec4f(0.0, i1.x, i2.x, 1.0));

    // Gradients
    let n_ = 0.142857142857; // 1.0/7.0
    let ns = n_ * D.wyz - D.xzx;

    let j = p - 49.0 * floor(p * ns.z * ns.z);

    let x_ = floor(j * ns.z);
    let y_ = floor(j - 7.0 * x_);

    let x = x_ * ns.x + ns.yyyy;
    let y = y_ * ns.x + ns.yyyy;
    let h = 1.0 - abs(x) - abs(y);

    let b0 = vec4f(x.xy, y.xy);
    let b1 = vec4f(x.zw, y.zw);

    let s0 = floor(b0) * 2.0 + 1.0;
    let s1 = floor(b1) * 2.0 + 1.0;
    let sh = -step(h, vec4f(0.0));

    let a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    let a1 = b1.xzyw + s1.xzyw * sh.zzww;

    var p0: vec3f = vec3f(a0.xy, h.x);
    var p1: vec3f = vec3f(a0.zw, h.y);
    var p2: vec3f = vec3f(a1.xy, h.z);
    var p3: vec3f = vec3f(a1.zw, h.w);

    // Normalise gradients
    let norm = taylorInvSqrt(vec4f(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 = p0 * norm.x;
    p1 = p1 * norm.y;
    p2 = p2 * norm.z;
    p3 = p3 * norm.w;

    // Mix contributions
    var m: vec4f = max(0.6 - vec4f(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), vec4f(0.0));
    m = m * m;
    return 42.0 * dot(m * m, vec4f(dot(x0, p0), dot(x1, p1), dot(x2, p2), dot(x3, p3)));
}

// --- SECTION 2: Curl Noise Calculations ---

// 2D Curl Noise: Returns a 2D velocity vector from a 3D point input
fn curlNoise2D(p: vec3f) -> vec2f {
    let e = 0.005; // Finite difference delta (step size)

    // Sample neighboring potential fields
    let n_x1 = snoise(p + vec3f(e, 0.0, 0.0));
    let n_x2 = snoise(p - vec3f(e, 0.0, 0.0));
    let n_y1 = snoise(p + vec3f(0.0, e, 0.0));
    let n_y2 = snoise(p - vec3f(0.0, e, 0.0));

    // Calculate partial derivatives
    let dy = (n_y1 - n_y2) / (2.0 * e);
    let dx = (n_x1 - n_x2) / (2.0 * e);

    // Return the rotated gradient (Curl in 2D)
    return vec2f(dy, -dx);
}

// 3D Curl Noise: Returns a 3D velocity vector from a 3D point input
fn curlNoise3D(p: vec3f) -> vec3f {
    let e = 0.005; // Finite difference delta

    // Sample offsets for all 3 axes
    let dx = vec3f(e, 0.0, 0.0);
    let dy = vec3f(0.0, e, 0.0);
    let dz = vec3f(0.0, 0.0, e);

    // Potential fields for X component
    let p_x1 = snoise(p + dx);
    let p_x2 = snoise(p - dx);
    // Potential fields for Y component
    let p_y1 = snoise(p + dy);
    let p_y2 = snoise(p - dy);
    // Potential fields for Z component
    let p_z1 = snoise(p + dz);
    let p_z2 = snoise(p - dz);

    // Direct mathematical cross product of the spatial derivatives
    let cx = ((p_y1 - p_y2) - (p_z1 - p_z2)) / (2.0 * e);
    let cy = ((p_z1 - p_z2) - (p_x1 - p_x2)) / (2.0 * e);
    let cz = ((p_x1 - p_x2) - (p_y1 - p_y2)) / (2.0 * e);

    return normalize(vec3f(cx, cy, cz));
}


fn blackbody(kelvin: f32) -> vec3f {
    // 1. Constrain temperature to the algorithm's valid range
    let t = clamp(kelvin, 1000.0, 40000.0) / 100.0;

    // 2. Calculate Red
    var r: f32;
    if t <= 66.0 {
        r = 255.0;
    } else {
        r = t - 60.0;
        r = 329.698727446 * pow(r, -0.1332047592);
    }

    // 3. Calculate Green
    var g: f32;
    if t <= 66.0 {
        g = t;
        g = 99.4708025861 * log(g) - 161.1195681661;
    } else {
        g = t - 60.0;
        g = 288.1221695283 * pow(g, -0.0755148492);
    }

    // 4. Calculate Blue
    var b: f32;
    if t >= 66.0 {
        b = 255.0;
    } else if t <= 19.0 {
        b = 0.0;
    } else {
        b = t - 10.0;
        b = 138.5177312231 * log(b) - 305.0447927307;
    }

    // 5. Clamp values to 0-255 range and normalize to 0.0 - 1.0
    return clamp(vec3f(r, g, b), vec3f(0.0), vec3f(255.0)) / 255.0;
}

fn sd_equilateral_triangle(point: vec2f, r: f32) -> f32 {
    let k: f32 = sqrt(3.0);
    var p = vec2f(point.xy);
    p.x = abs(p.x) - r;
    p.y = p.y + r/k;
    if( p.x+k*p.y>0.0 ) {
        p = vec2(p.x-k*p.y,-k*p.x-p.y)/2.0;
    }
    p.x -= clamp( p.x, -2.0*r, 0.0 );
    return -length(p)*sign(p.y);
}