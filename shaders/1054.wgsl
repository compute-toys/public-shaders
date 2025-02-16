
fn f_n_sin(
    n_freq: f32
)->f32{
    let n_tau = 6.283185307179586;
    var n = ((n_freq) % n_tau);
    if(n < 0.){
        n = n_tau - abs(n); 
    }
    // let n = n_freq;//(n_freq) % n_tau;
    var n2 = n;
    let n_its = 3.+fract(time.elapsed*.1)*12.;
    var n_exp = 3.;
    for(var n_it = 3.; n_it < n_its; n_it+=1){
        var n_factorial = 1.;
        for(var n_it2 =2.; n_it2 <= n_exp; n_it2+=1.){
            n_factorial*=n_it2;
        }
        n2 += sign(((n_it+1.) % 2.)-.5) * (pow(n, n_exp)/n_factorial); 
        n_exp+=2.;
    }
    return n2;
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
    var o_trn_nor_pix = (vec2f(o_trn_pixel.xy)-vec2f(o_scl_screen)*.5)
        /vec2f(o_scl_screen).yy;
    o_trn_nor_pix *=10.;
    var n = f_n_sin(o_trn_nor_pix.x+time.elapsed) - o_trn_nor_pix.y;
    if(o_trn_nor_pix.x > 0.){
        n = sin(o_trn_nor_pix.x+time.elapsed) - o_trn_nor_pix.y;
    }
    var o_col = vec3(
        n
    );
    let n_abs = abs(n);
    let oc1 = vec3f(0.6, 0.1, 0.6);
    let oc2 = vec3f(0.1, 0.2, 0.6);
    if(sign(n) == sign(o_trn_nor_pix.x)){
        o_col = oc1 * n_abs + sin(n*12.)*.1;
    }else{
        o_col = oc2 * n_abs + sin(n*12.)*.1;
    }
    o_col *= smoothstep(0.0,0.1,abs(o_trn_nor_pix.x));

    // o_col.x = (1.-length(o_trn_nor_pix));
    // Convert from gamma-encoded to linear colour space
    //o_col = pow(o_col, vec3f(2.2));
    
    // Output to screen (linear o_colour space)
    textureStore(screen, o_trn_pixel.xy, vec4f(o_col, 1.));
}
