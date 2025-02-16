
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) o_trn_pixel: vec3u) {
    // Viewport resolution (in pixels)
    let o_scl_screen = textureDimensions(screen);

    var o_trn_nor_mou = (vec2f(mouse.pos.xy)-vec2f(o_scl_screen)*.5)
        /vec2f(o_scl_screen).yy;
    // Prevent overdraw for worAkgroups on the edge of the viewport
    if (o_trn_pixel.x >= o_scl_screen.x || o_trn_pixel.y >= o_scl_screen.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    // Pixel coordinates (centre of pixel, origin at bottom left)
    var o_trn_nor_pix = (vec2f(o_trn_pixel.xy)-vec2f(o_scl_screen)*.5)
        /vec2f(o_scl_screen).yy;
    o_trn_nor_pix *= sin(time.elapsed*.2)*3.;
    var o_p = o_trn_nor_mou;
    let n_time = time.elapsed *.2;
    if(mouse.click == 0){
        o_p = vec2f(
            sin(n_time*5),
            cos(n_time*7.),
        )*0.2;
    }
    let o_diff_mou_pix = o_p-o_trn_nor_pix;
    let o = o_trn_nor_pix;

    let n_t = time.elapsed;
    let o_rotated = vec2f(
        o_trn_nor_pix.x*sin(n_t)-o_trn_nor_pix.y*cos(n_t),
        o_trn_nor_pix.x*cos(n_t)+o_trn_nor_pix.y*sin(n_t)
    );
    var n1 = sin(length(o_trn_nor_pix)*9.);
    var n2 = sin(length(o_trn_nor_pix)*3.);
    var n4 = length(o_diff_mou_pix);
    n4 = sin(n4*12.);
    
    var n = sin(n1 + n4);
    var o_col = vec3f(
        sin(n*3.+n_t),
        sin(n*6.+n_t*3.),
        sin(n*9.+n_t*6.),
    );

    // Output to screen (linear colour space)
    textureStore(screen, o_trn_pixel.xy, vec4f(o_col, 1.));
}
