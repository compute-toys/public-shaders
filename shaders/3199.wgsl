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

    var color = vec3f(0.0); // Color of circle
    var color2 = vec3f(0.0); // Color of bars

    var center = vec2f(0.5, 0.5); // Default point center
    
    if (mouse.click == 1) { 
        // normalizes the mouse coordinates like frag Coord
        let mouse_normal = vec2f(f32(mouse.pos.x), f32(screen_size.y) - f32(mouse.pos.y)) / vec2f(screen_size);
        
        // Sets center to normalized mouse position
        center = mouse_normal;
    } 

    // Creates circle around current center, sets color to change
    if (distance(uv, center) <= 0.15) {
        color = .5 + .5 * cos(time.elapsed + uv.xyx + vec3f(0.,2.,4.));
    }

    // Color bar logics
    if (uv.x < 0.33){
        color2 = vec3f(1, 0, 0);
    }
    else if (uv.x < 0.67){
        color2 = vec3f(0, 1, 0);
    }
    else{
        color2 = vec3f(0,0,1);
    }

    // Mixes the bar colors and the circle colors
    let color3 = mix(color, color2, 0.5);

    // Display results
    textureStore(screen, id.xy, vec4f(color3, 1));
}
