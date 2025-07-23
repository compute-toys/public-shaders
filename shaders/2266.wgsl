/*
    "Singularity" by @XorDev

    A quick and dirty port of my ShaderToy
    https://www.shadertoy.com/view/3csSWB
*/
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id : vec3u) {
    // Viewport resolution (in pixels)
    let screen_size : vec2<u32> = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let F : vec2<f32> = vec2<f32>(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);

    //Iterator and attenuation (distance-squared)
    var i : f32 = .2;
    var a : f32;
    
    //Resolution for scaling and centering
    let r : vec2<f32> = vec2<f32>(screen_size);
    //Centered ratio-corrected coordinates
    let p = ( F + F - r ) / r.y / .7;
    //Diagonal vector for skewing
    let d = vec2<f32>(-1, 1);
    //Blackhole center
    let b = p - i * d;
    //Rotate and apply perspective
    let D = .1 + i / dot(b, b);
    let c = p * mat2x2<f32>(1, 1, d.x / D, d.y / D);
    //Rotate into spiraling coordinates
    a = dot(c, c);
    let C = cos(time.elapsed * i + .5 * log(a) + vec4<f32>(0, 33, 11, 0));
    var v = c * mat2x2<f32>(C.x, C.y, C.z, C.w) / i;
    //Waves cumulative total for coloring
    var w = vec2<f32>(0);
    
    //Loop through waves
    for(; i < 9; i = i + 1)
    {
        //Distort coordinates
        v = v + .7 * sin(v.yx * i + .2*time.elapsed) / i + .5;
        w = w + (1 + sin(v));
    }
    //Acretion disk radius
    i = length(sin(v / .3) * .4 + c * (3 + d));
    
    //Red/blue gradient
    let O : vec4<f32> = 1 - exp(-exp(c.x * vec4<f32>(.6, -.4, -1, 0))
                   //Wave coloring
                   / w.xyyx
                   //Acretion disk brightness
                   / (2 + i * i / 4 - i)
                   //Center darkness
                   / (.5 + 1 / a)
                   //Rim highlight
                   / (.03 + abs(length(p) - .7))
             );
    
    // Output to screen
    textureStore(screen, id.xy, pow(O,vec4<f32>(2.2)));
}