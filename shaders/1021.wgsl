
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
    o_trn_nor_pix=vec2f(
        o.x* cos(time.elapsed)-o.y*sin(time.elapsed), 
        o.x*sin(time.elapsed)+o.y*cos(time.elapsed)
    )*cos(time.elapsed)*2.;
    let nm = (sin((o_trn_nor_pix.x)+time.elapsed)*.5+.5)*.5;
    let n1 = sin(o_trn_nor_pix.x*33.*nm+time.elapsed);
    let n2 = sin(o_trn_nor_pix.y*33.*nm);
    var n3 = n1*n2;
    let n_fa = length(o_diff_mou_pix);
    
    // n3 = o_trn_nor_pix.x+o_trn_nor_pix.y;
    n3 = sin(n3*18.*n_fa+time.elapsed*3.);

    let o_col = vec3f(n3);


    // Output to screen (linear colour space)
    textureStore(screen, o_trn_pixel.xy, vec4f(o_col, 1.));
}
