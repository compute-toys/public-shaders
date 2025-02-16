
@compute @workgroup_size(16, 16)
fn main_image(
    @builtin(global_invocation_id) 
    o_trn_pixel: vec3u
) {
    // Viewport resolution (in pixels)
    let o_scl_screen = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    // if (o_trn_pixel.x >= o_scl_screen.x || o_trn_pixel.y >= o_scl_screen.y) { return; }

    var nr = 0.;
    // Pixel coordinates (centre of pixel, origin at bottom left)
    var o_trn_nor_pix = (vec2f(o_trn_pixel.xy)-vec2f(o_scl_screen)*.5)
        /vec2f(o_scl_screen).yy;
    var n_tau = radians(360.);
    var n = length(o_trn_nor_pix);
    n = sin((1./(.1+n)*11.)+time.elapsed*9.)*.5+.5;
    var o_col = vec3f(n);
    textureStore(screen, o_trn_pixel.xy, vec4f(o_col, 1.));

}
