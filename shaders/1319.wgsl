
fn f_n(
    o_trn_nor_pix: vec2f, 
    n_it_nor: f32
)->f32{
    
    var n_t = time.elapsed*.3;
    let nl = length(o_trn_nor_pix);
    let n_tau = radians(360.);

    var n_amp = 0.1+(sin((n_t+n_it_nor)*n_tau)*.5+.5)*.2;
    let o = vec2f(
        cos(n_it_nor*n_tau),
        sin(n_it_nor*n_tau)
    )*n_amp;
    var n = length(o_trn_nor_pix.xy-o.xy);
    var n_radius = (sin((n_t+n_it_nor)*n_tau)*.5+.5)*.2;
    n = abs(n-n_radius);
    var b = 0.003;//+(sin(nl*99.+n_t)*.5+.5)*.002; 
    //n = pow(n, 1./10.);
    n = -b*(1./(n+b))+1.005; 
    n =pow(n, 1.1);
    // n = smoothstep(0.2, 0.0, n);
    //n = clamp(n, 0., 1.);
    return n;

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
    o_trn_nor_pix*=1.2;

    let n_its = 9.;
    let n_it_nor_one = 1./n_its;
    var o_col = vec3f(0.);
    for(var n_it_nor = 0.; n_it_nor < 1.; n_it_nor+=n_it_nor_one){
        var n = f_n(o_trn_nor_pix.xy, n_it_nor);
        o_col += clamp(vec3f(
            1.-f_n(o_trn_nor_pix.xy, n_it_nor+.01),
            1.-f_n(o_trn_nor_pix.xy, n_it_nor+.02),
            1.-f_n(o_trn_nor_pix.xy, n_it_nor+.03),
        ),vec3f(0.), vec3f(1.));
    }
    // nm = length(o_trn_nor_pix);
    // Convert from gamma-encoded to linear colour space
    //o_col = pow(o_col, vec3f(2.2));

    // Output to screen (linear o_colour space)
    textureStore(screen, o_trn_pixel.xy, vec4f(o_col, 1.));
}
