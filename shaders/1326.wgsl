const n_tau = radians(360.);
fn f_n(
    o_trn_nor_pix: vec2f
)->f32{
    let o_scl_screen = vec2f(textureDimensions(screen));

    let naa = 1./(o_scl_screen.x);
    var n1 = length(o_trn_nor_pix-vec2f(-0.0, 0.));
    let nm1 = smoothstep(0.5, 0.5+naa*4., n1);
    n1 = abs(n1-.5)/.5;

    var n2 = abs(length(o_trn_nor_pix-vec2f(0.0, -0.5))-.5)/.5;
    var n3 = abs(length(o_trn_nor_pix-vec2f(0.0, 0.5))-.5)/.5;

    var n4 = abs(length(o_trn_nor_pix-vec2f(-.25, 0.))-.25)/.25;
    var n5 = abs(length(o_trn_nor_pix-vec2f(.25, 0.))-.25)/.25;

    let n2n3 = n2*n3*(1.-nm1)+nm1;
    return n1*n2n3*n4*n5;
    // return n1;

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
    o_trn_nor_pix= -1.*abs(o_trn_nor_pix);

    let n_t = time.elapsed;
    var n = f_n(o_trn_nor_pix.xy);
    let b = 0.02;
     n = -b*(1./(n+b))+1.;
     n = 1.-n;
     //n = sin( n*3.*n_tau+time.elapsed)*.5+.5;
     let o_col = vec3f(
        sin(n*10.+(time.elapsed+(1./3.)*.1*n_tau))*.5+.5,
        sin(n*10.+(time.elapsed+(2./3.)*.1*n_tau))*.5+.5,
        sin(n*10.+(time.elapsed+(3./3.)*.1*n_tau))*.5+.5
     );
    // nm = length(o_trn_nor_pix);
    // Convert from gamma-encoded to linear colour space
    //o_col = pow(o_col, vec3f(2.2));

    // Output to screen (linear o_colour space)
    textureStore(screen, o_trn_pixel.xy, vec4f(o_col, 1.));
}
