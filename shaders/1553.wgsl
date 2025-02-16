#define PI 3.14159265358979323846f

// Rotate function, assuming a simple 2D rotation matrix
fn rotate(angle: f32) -> mat2x2<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return mat2x2<f32>(vec2<f32>(c, -s), vec2<f32>(s, c));
}


// Sign function definition
fn sign(x: f32) -> f32 {
    return select(-1.0, 1.0, x >= 0.0);
}

fn sdEquilateralTriangle(p: vec2<f32>, r: f32, angle: f32) -> f32 {
    let k = sqrt(3.0);
    var q = vec2<f32>(abs(p.x) - r, p.y + r / k);
    
    if (q.x + k * q.y > 0.0) {
        q = vec2<f32>(q.x - k * q.y, -k * q.x - q.y) / 2.0;
    }
    
    q.x -= clamp(q.x, -2.0 * r, 0.0);
    return max(-length(q * rotate(angle)) * sign(q.y), 0.1);
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = vec2f(f32(id.x) + 0.5, f32(screen_size.y - id.y) - 0.5);

    let o_trn_nor_pix = (vec2<f32>(id.xy) -  vec2f(screen_size)) * (0.5 / f32(screen_size.y));
    var o_col = vec4<f32>(0.0);
    var hue: vec4<f32>;
    // Initialize animation variables
    let t = time.elapsed * 0.5;
    var initial_angle = PI*0.0625;
    var angle = initial_angle;  // assuming `initial_angle` is defined
    let start = 40;  // equivalent to 1.5 in steps of 0.1
    let end = 0;     // equivalent to 0.036 in steps of 0.1
    var uv = fragCoord / vec2f(screen_size);

    for (var j: i32 = start; j > end; j -= 1) {
        let i = (f32(j) * 0.06666)-.05;  // Calculate `i` as `f32`
        let temp_angle = (t + i) * 3.0;
         angle -= sin(angle - sin(temp_angle));
        uv = (fragCoord + fragCoord - vec2f(screen_size)) / f32(screen_size.x);
        uv *= rotate(i + (angle +.45) + t);
        // Compute rounded triangle SDF
        let triangle = sdEquilateralTriangle(uv, i, angle + time.elapsed);
        // Compute anti-aliased alpha using SDF
        let alpha = min((f32(triangle) - 0.1) * (f32(screen_size.y) * f32(screen_size.x)) * 0.1, .925);
        hue = sin(i / 0.15 + angle / 2.0 + vec4<f32>(10.0, 2.0, 7.0, 0.0)) * 0.3 + 0.7;
        o_col = mix(hue, o_col, alpha);
        o_col *= mix(hue / hue, hue + 0.01 * alpha * uv.y / triangle, 0.1 / triangle);

    }

    // Convert from gamma-encoded to linear colour space
    o_col = pow(o_col, vec4f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(o_col.rgb, 1.));
}
