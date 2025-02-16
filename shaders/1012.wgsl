const n_tau = 6.2831;
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
    var o_trn_nor_pix = (vec2f(o_trn_pixel.xy)-vec2f(o_scl_screen)*.5)
        /vec2f(o_scl_screen).yy;
    o_trn_nor_pix *=2;
    var o_col = vec3(o_trn_nor_pix.x, o_trn_nor_pix.y, 0.);
    // Convert from gamma-encoded to linear colour space
    //o_col = pow(o_col, vec3f(2.2));
    let n_its = 6.;
    var n_d = 0.;
    for(var n_it = 0.; n_it < n_its;n_it+=1.){
        let n_it_nor = n_it / n_its;
        let o_p = vec2(
            sin(n_it_nor*n_tau),
            cos(n_it_nor*n_tau)
        );
        var n =length(o_trn_nor_pix-o_p)*1.2;
        n = sin(n*20.)/n_its;
        n_d+=n;
    }
    let n_d_center = length(o_trn_nor_pix);
    // Output to screen (linear o_colour space)
    // let n2 = sin(n_d*5.+time.elapsed*20.+time.elapsed*n_d_center);
    let n2 = sin(n_d*3.+time.elapsed*2.);
    let n3 = sin(n_d*6.+time.elapsed*2.);
    let n4 = sin(n_d*9.+time.elapsed*2.);
    o_col = vec3f(n2,n3,n4);

    textureStore(screen, o_trn_pixel.xy, vec4f(o_col, 1.));
}
