const COLOR_0: vec3f = vec3f(67, 119, 209) / 255.;
const COLOR_1: vec3f = vec3f(101, 209, 120) / 255.;

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    var fragCoord = vec2f(id.xy);
    // Normalised pixel coordinates (from 0 to 1)
    let uv = fragCoord / vec2f(screen_size);

    // apply scale
    fragCoord *= custom.scale;

    var tex_size = vec2f(textureDimensions(channel0));
    var tuv = fragCoord / tex_size;
    var bayer = textureSampleLevel(channel0, nearest_repeat, tuv, 0).r;
    // fix color space(from inverse gamma to linear)
    bayer = pow(bayer, 1. / 2.2);
    // Transform distribution to triangle
    bayer = uniform_to_triangle(bayer);

    tex_size = vec2f(textureDimensions(channel1));
    tuv = fragCoord / tex_size;
    var bnoise = textureSampleLevel(channel1, nearest_repeat, tuv, 0.).r;
    // fix color space(from inverse gamma to linear)
    bnoise = pow(bnoise, 1. / 2.2);
    // Transform distribution to triangle
    bnoise = uniform_to_triangle(bnoise);

    var col = vec3f(0.);

    // 0: 2 color, bayer
    if all(uv < vec2f(0.25, 0.5))
    {
        let ncols = 2.;
        
        var r = uv.y * 2.;  // scale to [0., 1.]
        r = floor(r * ncols + bayer) / ncols;

        col = mix(COLOR_0, COLOR_1, r);
    }
    // 1: 4 color, bayer
    else if all(uv < vec2f(0.5, 0.5))
    {
        let ncols = 4.;
        
        var r = uv.y * 2.;  // scale to [0., 1.]
        r = floor(r * ncols + bayer) / ncols;

        col = mix(COLOR_0, COLOR_1, r);
    }
    // 2: 8 color, bayer
    else if all(uv < vec2f(0.75, 0.5))
    {
        let ncols = 8.;
        
        var r = uv.y * 2.;  // scale to [0., 1.]
        r = floor(r * ncols + bayer) / ncols;

        col = mix(COLOR_0, COLOR_1, r);
    }
    // 3: 16 color, bayer
    else if all(uv < vec2f(1., 0.5))
    {
        let ncols = 16.;
        
        var r = uv.y * 2.;  // scale to [0., 1.]
        r = floor(r * ncols + bayer) / ncols;

        col = mix(COLOR_0, COLOR_1, r);
    }
    // 4: 2 color, bnoise
    else if all(uv < vec2f(0.25, 1.))
    {
        let ncols = 2.;
        
        var r = (uv.y - 0.5) * 2.;  // scale to [0., 1.]
        r = floor(r * ncols + bnoise) / ncols;

        col = mix(COLOR_0, COLOR_1, r);
    }
    // 5: 4 color, bnoise
    else if all(uv < vec2f(0.5, 1.))
    {
        let ncols = 4.;
        
        var r = (uv.y - 0.5) * 2.;  // scale to [0., 1.]
        r = floor(r * ncols + bnoise) / ncols;

        col = mix(COLOR_0, COLOR_1, r);
    }
    // 6:: 8 color, bnoise
    else if all(uv < vec2f(0.75, 1.))
    {
        let ncols = 8.;
        
        var r = (uv.y - 0.5) * 2.;  // scale to [0., 1.]
        r = floor(r * ncols + bnoise) / ncols;

        col = mix(COLOR_0, COLOR_1, r);
    }
    // 7: 16olor, bnoise
    else
    {
        let ncols = 16.;
        
        var r = (uv.y - 0.5) * 2.;  // scale to [0., 1.]
        r = floor(r * ncols + bnoise) / ncols;

        col = mix(COLOR_0, COLOR_1, r);
    }

    let screen_size_f = custom.scale * vec2f(screen_size);
    let g = abs(fragCoord.y - 0.5 * screen_size_f.y) < 1.
    || abs(fragCoord.x - 0.25 * screen_size_f.x) < 1.
    || abs(fragCoord.x - 0.50 * screen_size_f.x) < 1.
    || abs(fragCoord.x - 0.75 * screen_size_f.x) < 1.;
    col = mix(col, vec3f(0.08), select(0., 1., g));

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn uniform_to_triangle(v: f32) -> f32 {
    var n = v * 2.0 - 1.0;
    n = sign(n) * (1.0 - sqrt(max(0.0, 1.0 - abs(n)))); // [-1, 1], max prevents NaNs
    return n + 0.5; // [-0.5, 1.5]
}