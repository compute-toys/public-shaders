
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) o_trn_pixel: vec3u) {
    // Viewport resolution (in pixels)
    let o_scl_screen = textureDimensions(screen);

    var o_trn_nor_mou = (vec2f(mouse.pos.xy)-vec2f(o_scl_screen)*.5)
        /vec2f(o_scl_screen).yy;
    // Prevent overdraw for workgroups on the edge of the viewport
    if (o_trn_pixel.x >= o_scl_screen.x || o_trn_pixel.y >= o_scl_screen.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    // Pixel coordinates (centre of pixel, origin at bottom left)
    var o_trn_nor_pix = (vec2f(o_trn_pixel.xy)-vec2f(o_scl_screen)*.5)
        /vec2f(o_scl_screen).yy;
    let o_diff_mou_pix = o_trn_nor_mou-o_trn_nor_pix;
    let o = o_trn_nor_pix;

    let n_t = time.elapsed*0.1;
    let o_rotated = vec2f(
        o_trn_nor_pix.x*sin(n_t)-o_trn_nor_pix.y*cos(n_t),
        o_trn_nor_pix.x*cos(n_t)+o_trn_nor_pix.y*sin(n_t)
    );
    let o_rotated2 = vec2f(
        o_trn_nor_pix.x*sin(-n_t)-o_trn_nor_pix.y*cos(-n_t),
        o_trn_nor_pix.x*cos(-n_t)+o_trn_nor_pix.y*sin(-n_t)
    );
    let n_t1 = sin(n_t*9.);
    let n1 = sin(o_rotated.x*20.);
    let n2 = sin(o_rotated2.x*20.);
    let n3 =n_t1 * n2+(1.-n_t1)*n1;
    let n4 = sin(n3*3.);
    let n_len = 1.-length(o_diff_mou_pix);
    
    var o_col = vec3f(n4);
    o_col = pow(n_len,2.)*(vec3f(1.)-o_col)+o_col;

    // Output to screen (linear colour space)
    textureStore(screen, o_trn_pixel.xy, vec4f(o_col, 1.));
}
