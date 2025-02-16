// Adapted from: https://ispc.github.io/example.html

fn mandel(c_re: f32, c_im: f32, count: i32) -> i32 {
    var z_re = c_re;
    var z_im = c_im;
    var i: i32;
    for (i = 0; i < count; i++) {
        if (z_re * z_re + z_im * z_im > 4.0) {
            break;
        }
        let new_re = z_re * z_re - z_im * z_im;
        let new_im = 2.0f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }
    return i;
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Mandelbrot region
    let p0 = float2(-2, -1);
    let p1 = float2(1, 1);
    let d = (p1 - p0) / float2(screen_size.xy);
    let uv = p0 + float2(id.xy) * d;

    var col = float3(float(mandel(uv.x, uv.y, 256))/256.0);

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, float4(col, 1.));
}