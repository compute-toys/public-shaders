@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);

    // Normalised pixel coordinates (from 0 to 1)
    var uv = fragCoord / vec2f(screen_size)*2.-1.;
    uv.x *= f32(screen_size.x)/f32(screen_size.y);
    var size = floor(custom.length);
    var mousex = f32(mouse.pos.x)/f32(screen_size.x);
    var angel = 6.28318530718/size-(6.28318530718/(size*4))+(6.28318530718/4)+mousex*6.28318530718;
    var mathc = mat2x2(vec2(sin(angel),cos(angel)),
               vec2(-cos(angel),sin(angel)));
    var uvround = uv* mathc;

    var col:f32 = atan2(uvround.x,uvround.y);
    var theta_full:f32 = col + 6.28318530718;
    var theta_normalized = theta_full / 6.28318530718;

    // Time varying pixel colour
    var sinround = sin(theta_normalized*(size*6.28318530718));

    var dist = step(custom.step,length(uv)-sinround*custom.step*custom.fract);
    // Convert from gamma-encoded to linear colour space

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(vec3(dist), 1.));
}
