// The top half of the screen has a checkerboard pattern and the bottom does not.
// Adjust the Gamma slider until there is no discontinuity between the two.
// That is your monitor's gamma. If you are unable to adjust the gamma until
// there is no discontinuity anywhere, your monitor is not displaying colors
// completely correctly and its settings may require adjustment.

// I read somewhere that some monitors actually used a value of 2 instead of the 
// standard 2.2 and wanted to test it. My monitor has 4 unlabeled gamma modes that
// seem to range from 2.05 to 2.4 and tint the screen various hues for some reason.

// from compute.toys source - lib/engine/blit.ts
fn srgb_to_linear(rgb: vec3<f32>) -> vec3<f32> {
    return select(
        pow((rgb + 0.055) * (1.0 / 1.055), vec3<f32>(2.4)),
        rgb * (1.0/12.92),
        rgb <= vec3<f32>(0.04045));
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);

    // Normalised pixel coordinates (from 0 to 1)
    let uv = fragCoord / vec2f(screen_size);

    var col = float3((uv.x * 4.) % 1);
    if(uv.x < 1./4.) {
        col *= float3(1, 0, 0);
    } else if(uv.x < 2./4.) {
        col *= float3(0, 1, 0);
    } else if(uv.x < 3./4.) {
        col *= float3(0, 0, 1);
    }
    var ch = id / uint(round(custom.CheckerSize));
    var checker = select(0., 1., (ch.x + ch.y) % 2 == 0);
    if(uv.y > 0.5) {
        col *= checker;
    } else {
        col *= 0.5;
    }

    col = pow(col, float3(1./custom.Gamma));

    // Convert from gamma-encoded to linear colour space
    col = srgb_to_linear(col);

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}
