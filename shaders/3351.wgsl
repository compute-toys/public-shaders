// NOTE(Constantine): Based on www.shadertoy.com/view/XdXXz4

fn circle(p: vec2<f32>, center: vec2<f32>, radius: f32) -> vec4<f32> {
    // Renders a smooth red circle on a transparent background
    let dist = length(p - center);
    let edge0 = radius + 0.005;
    let edge1 = radius - 0.005;
    return mix(vec4<f32>(1.0, 1.0, 1.0, 0.0), vec4<f32>(1.0, 0.0, 0.0, 1.0), smoothstep(edge0, edge1, dist));
}

fn scene(uv: vec2<f32>, t: f32) -> vec4<f32> {
    // Bouncing animation logic
    let centerY = sin(t * 16.0) * (sin(t) * 0.5 + 0.5) * 0.5;
    return circle(uv, vec2<f32>(0.0, centerY), 0.2);
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);

    // Normalize coordinates for the full screen
    var uv = fragCoord / vec2f(screen_size);
    uv = uv * 2.0 - vec2<f32>(1.0);
    
    // In WebGPU / WGSL, the viewport Y axis is pointing DOWN. 
    // To match GLSL/Shadertoy's upward orientation, we invert the Y-axis.
    uv.y = -uv.y;
    
    uv.x = uv.x * (f32(screen_size.x) / f32(screen_size.y)); // Correct aspect ratio
    
    // Time stepping logic
    let frametime: f32 = 60.0;
    let time = floor((time.elapsed + 3.0) * frametime) / frametime;
    
    // Accumulate motion blur samples
    var blurCol = vec4<f32>(0.0);
    let samples: i32 = 32;
    
    for (var i: i32 = 0; i < samples; i = i + 1) {
        // Sample the scene back in time
        let sampleTime = time - f32(i) * (1.0 / 15.0 / f32(samples));
        blurCol = blurCol + scene(uv, sampleTime);
    }
    
    // Divide by the total number of samples to get the correct average color/opacity
    blurCol = blurCol / f32(samples);
    
    // Convert from gamma-encoded to linear colour space
    var col = pow(blurCol.xyz, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}