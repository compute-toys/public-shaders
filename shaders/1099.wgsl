
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

    var n = dot(o_trn_nor_pix, o_trn_nor_mou);
    let n_lines = 100.*length(o_trn_nor_mou);
    let n_tau = 6.2831;
    n = sin(n*n_lines*n_tau+time.elapsed*0.2)*.5+.5;
    let n_thickness = 0.5;
    var n_aa = 5.;
    n_aa = n_aa*(1./f32(o_scl_screen.x))*n_lines;
    if(o_trn_nor_pix.x > 0.){
        n = smoothstep(n_thickness+n_aa, n_thickness, n);
    }else{
        n = 1.-step(n_thickness, n);
    }
    var o_col = vec3(n);
    // Convert from gamma-encoded to linear colour space
    //o_col = pow(o_col, vec3f(2.2));

    // Output to screen (linear o_colour space)
    textureStore(screen, o_trn_pixel.xy, vec4f(o_col, 1.));
}
