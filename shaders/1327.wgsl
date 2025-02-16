const n_tau = radians(360.);
fn f_n(
    o_trn_nor_pix: vec2f
)->f32{
    let o_scl_screen = vec2f(textureDimensions(screen));
    var o_trn_nor = o_trn_nor_pix * 6.;
    let naa = 1./(o_scl_screen.x);
    var n1 = length(o_trn_nor-vec2f(-0.0, 0.));
    n1 = abs(n1-.5)/.5;
    var n2 = length(o_trn_nor-vec2f(0.5, 0.));
    n2 = abs(n2-1.)/1.;
    var n3 = length(o_trn_nor-vec2f(1.5, 0.));
    n3 = abs(n3-2.)/2.;
    return n1*n2*n3;

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

    let n_t = time.elapsed;
    var n = f_n(o_trn_nor_pix);
    var n2 = f_n(vec2f(-o_trn_nor_pix.x,o_trn_nor_pix.y));
    //n = min(n, n2);
    // n = n * n2;
    let n3 = n + n2;
    n = min(n, n2)+n3;
    let b = 0.6;
     n = -b*(1./(n+b))+1.;
     n = 1.-n;
     //n = sin( n*3.*n_tau+time.elapsed)*.5+.5;
     var o_col = vec3f(
        sin(n*10.+(time.elapsed+(1./3.)*.02*n_tau))*.5+.5,
        sin(n*10.+(time.elapsed+(2./3.)*.04*n_tau))*.5+.5,
        sin(n*10.+(time.elapsed+(3./3.)*.06*n_tau))*.5+.5
     );
    //o_col = vec3f(n);
    // nm = length(o_trn_nor_pix);
    // Convert from gamma-encoded to linear colour space
    //o_col = pow(o_col, vec3f(2.2));

    // Output to screen (linear o_colour space)
    textureStore(screen, o_trn_pixel.xy, vec4f(o_col, 1.));
}
