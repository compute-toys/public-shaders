
fn f_n_sdf_rectangle(
    o: vec2f, 
    o_scl_nor: vec2f
)->f32{
    let n1 = abs(o.x)-o_scl_nor.x;
    let n2 = abs(o.y)-o_scl_nor.y;
    return max(n1,n2);
}

@compute @workgroup_size(16, 16)
fn main_image(
    @builtin(global_invocation_id) 
    o_trn_pixel: vec3u
) {
    // Viewport resolution (in pixels)
    let o_scl_screen = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    // if (o_trn_pixel.x >= o_scl_screen.x || o_trn_pixel.y >= o_scl_screen.y) { return; }


    // Pixel coordinates (centre of pixel, origin at bottom left)
    let o_trn_nor_pix = (vec2f(o_trn_pixel.xy)-vec2f(o_scl_screen)*.5)
        /vec2f(o_scl_screen).yy;
    let o_trn_nor_mou = (vec2f(mouse.pos.xy)-vec2f(o_scl_screen)*.5)
        /vec2f(o_scl_screen).yy;

    // Convert from gamma-encoded to linear colour space
    //o_col = pow(o_col, vec3f(2.2));
    var o_scl = vec2f(
        vec2f(
            sin(time.elapsed),
            cos(time.elapsed)
        )*.5+.5
    );

    if(mouse.click==1){
        o_scl = abs(o_trn_nor_mou);
    }
    var n = f_n_sdf_rectangle(o_trn_nor_pix,
        // vec2f(0.4, 0.6) // width and height of the recatngle
        o_scl
    );
    var o_col = vec3f(0.2, 0.8, 0.9);
    if(n < .0){
        o_col = vec3f(0.89, 0.5, 0.2);
    }
    let n2 = 1.-pow(n, 1./2.);
    o_col = pow(n, 1./2.)*o_col+sin(5.+99.*n2+time.elapsed)*.05;
    o_col += vec3f(smoothstep(0.9, 0.99, n2));    
    // o_col = fract(o_col);
    // Output to screen (linear o_colour space)
    textureStore(screen, o_trn_pixel.xy, vec4f(o_col, 1.));
}
