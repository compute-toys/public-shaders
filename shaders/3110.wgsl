@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);

    // Normalised pixel coordinates (from 0 to 1)
    let uv = (fragCoord * 2.0 - vec2f(screen_size)) / f32(screen_size.x);

    let a = atan2(uv.y, uv.x);
    let r = length(uv);
    
    var col = vec3f(1.0); 

    // from back to front
    for(var i = 0; i < 5; i = i + 1) {
        let fi = f32(i) / 5.; 
        
        // rotation by 2.5
        let offsetA = a + fi * 2.5; 
        
        let petals = 10.0 - f32(i); 

        let s = 0.5 + 0.1 * f32(i+1) * sin(petals * offsetA);
        var t = 0.16 + 0.3 * pow(s, 0.3);
        t += 0.05 * pow(0.3 + (0.4 + 0.05 * f32(i)) * cos(petals * 2.0 * offsetA), 2.0);
        
        // (1.2 -> 0.4)
        let scale = 1.2 - 0.8 * fi; 
        t *= scale;
        
        // Anti-aliasing
        let p = smoothstep(t + 0.005, t - 0.005, r);
        
        let edgeDist = (t - r) / t;
        let edgeShadow = smoothstep(0.0, 0.15, edgeDist);
        
        // Back is leaf
        var layerBase = vec3f(0.0);
        if (i == 0) {
            layerBase = mix(vec3f(0.3, 0.5, 0.2), vec3f(0.2, 0.45, 0.15), fi);
        } else {
            layerBase = mix(vec3f(0.6, 0.1, 0.2), vec3f(0.9, 0.25, 0.35), fi);

        }
        
        var layerCol = layerBase * (0.4 + 0.9 * (r / t));
        layerCol *= (0.6 + 0.4 * edgeShadow); // shadow
        layerCol = pow(layerCol, vec3f(2.2)); // fix color space

        col = mix(col, layerCol, p);
    }

    textureStore(screen, id.xy, vec4f(col, 1.));
}