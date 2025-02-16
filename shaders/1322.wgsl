const n_tau = radians(360.);
fn f_n(
    o_trn_nor_pix: vec2f, 
    n_it_nor: f32
)->f32{
    
    var n_t = time.elapsed;
    let nl = length(o_trn_nor_pix);
    let n_tau = radians(360.);
    let o = vec2f(
        sin(n_t*1.+n_it_nor*n_tau*sin(n_t*0.1)),
        sin(n_t*3.+n_it_nor*n_tau)
    )*.25;
    var n = length(o_trn_nor_pix.xy-o.xy);
    n = smoothstep(0., 1., n);
    var b = 0.002;//+(sin(nl*99.+n_t)*.5+.5)*.002; 
    n = -b*(1./(n+b))+1.003; 
    //n =pow(n, 20.);
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

    let n_its = 18.;
    let n_it_nor_one = 1./n_its;
    var o_col = vec3f(0.);
    for(var n_it_nor = 0.; n_it_nor < 1.; n_it_nor+=n_it_nor_one){
        var n = f_n(o_trn_nor_pix.xy, n_it_nor);
        o_col += clamp(vec3f(
            1.-f_n(o_trn_nor_pix.xy,  (n_it_nor+(1./3.))),
            1.-f_n(o_trn_nor_pix.xy,  (n_it_nor+(2./3.))),
            1.-f_n(o_trn_nor_pix.xy, (n_it_nor+(3./3.))),
        ),vec3f(0.), vec3f(1.));
    }
    // nm = length(o_trn_nor_pix);
    // Convert from gamma-encoded to linear colour space
    //o_col = pow(o_col, vec3f(2.2));

    // Output to screen (linear o_colour space)
    textureStore(screen, o_trn_pixel.xy, vec4f(o_col, 1.));
}
