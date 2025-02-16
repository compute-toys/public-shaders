const n_tau = 6.2831;

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) o_trn_pix: vec3u) {
    // Viewport resolution (in pixels)
    let o_scl_canvas = textureDimensions(screen);
    // Prevent overdraw for workgroups on the edge of the viewport
    if (o_trn_pix.x >= o_scl_canvas.x || o_trn_pix.y >= o_scl_canvas.y) { return; }
    // Pixel coordinates (centre of pixel, origin at bottom left)

    // Normalised pixel coordinates (from 0 to 1)
    let o_trn = (vec2f(o_trn_pix.xy)-vec2f(o_scl_canvas.xy)*.5)/vec2f(o_scl_canvas.yy);

    let n_its = 99.;
    var n_d_min1 = 1.;
    var n_d_min2 = 1.;
    var n_d_min3 = 1.;
    var n_it_min = 0.;
    var n_d_sum = 0.;
    var n_d_prod = 0.;
    for(var n_it = 0.; n_it < n_its; n_it+=1.){
        let n_it_nor = n_it / n_its;
        
        var o_p = vec2f(
            sin(n_it_nor*n_tau), 
            cos(n_it_nor*n_tau)
        )*vec2f(
            sin(n_it_nor*n_its+time.elapsed*0.2)
        );

        var n_d_cntr = length(o_p);
        n_d_cntr = (1.-n_d_cntr);
        var o_p2 = vec2f(
            sin(n_it_nor*n_tau), 
            cos(n_it_nor*n_tau)
        )*vec2f(
            sin(n_d_cntr*10.)
        );
        let o_diff = o_p2-o_trn;
        // euclidean /pythagorean / length
        var n_d2_1 = length(o_diff);
        //manhattan 
        var n_d2_2 = abs(o_diff.x) + abs(o_diff.y);
        // something ?
        var n_d2_3 = abs(o_diff.x) * abs(o_diff.y);

        if(n_d2_1 < n_d_min1){
            n_d_min1 = n_d2_1;
        }
        if(n_d2_2 < n_d_min2){
            n_d_min2 = n_d2_2;
        }
        if(n_d2_3 < n_d_min3){
            n_d_min3 = n_d2_3;
        }
    }
    // Convert from gamma-encoded to linear colour space
    var o_col = vec3f(
        n_d_min1,
        n_d_min2,
        n_d_min3,
    );
    //o_col = pow(o_col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(
        screen,
        o_trn_pix.xy,
        vec4f(o_col, 1.)
    );
}
