const n_tau = 6.2831;

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) o_trn_pix: vec3u) {
    // Viewport resolution (in pixels)
    let o_scl_canvas = textureDimensions(screen);
    // Prevent overdraw for workgroups on the edge of the viewport
    if (o_trn_pix.x >= o_scl_canvas.x || o_trn_pix.y >= o_scl_canvas.y) { return; }
    // Pixel coordinates (centre of pixel, origin at bottom left)

    // Normalised pixel coordinates (from 0 to 1)
    let o_trn_nor =  vec2f(o_trn_pix.xy)/vec2f(o_scl_canvas.xy);
    let o_trn = (vec2f(o_trn_pix.xy)-vec2f(o_scl_canvas.xy)*.5)/vec2f(o_scl_canvas.yy);
    let n_its = 33.;
    var n_d_min_euler = 1.;
    var n_d_min_manhattan = 1.;
    var n_it_nor_min_euler = 0.;
    var n_it_nor_min_manhattan = 0.;
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

        let o_diff = o_p-o_trn;
        // euclidean /pythagorean / length
        var n_d_euler = length(o_diff);
        //manhattan 
        var n_d_manhattan = abs(o_diff.x) + abs(o_diff.y);

        if(n_d_euler < n_d_min_euler){
            n_d_min_euler = n_d_euler;
            n_it_nor_min_euler = n_it_nor;
        }
        if(n_d_manhattan < n_d_min_manhattan){
            n_d_min_manhattan = n_d_manhattan;
            n_it_nor_min_manhattan = n_it_nor;
        }

    }
    // Convert from gamma-encoded to linear colour space
    const nl = 4;
    let a_n = array<f32, nl>(
        n_it_nor_min_euler, 
        n_it_nor_min_manhattan, 
        n_d_min_euler,
        n_d_min_manhattan
    );
    let n_idx: u32 = u32(floor((o_trn_nor.x)*f32(nl)));
    var n_col = a_n[n_idx];
    if(o_trn_nor.y > 0.5){
        n_col = 1.-n_col;
        n_col = pow(n_col,2.2);
    }
    var o_col = vec3f(n_col);
    let n_ratio = 1./f32(o_scl_canvas.x);
    let o_border = fract(o_trn_nor*vec2f(nl, nl/2.));
    var n_line = smoothstep(
        max(o_border.x, o_border.y), 1.-n_ratio*20.,1.);
    o_col *= vec3(n_line);

    // Output to screen (linear colour space)
    textureStore(
        screen,
        o_trn_pix.xy,
        vec4f(o_col, 1.)
    );
}
