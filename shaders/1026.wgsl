
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
    //o_trn_nor_pix*=sin(time.elapsed)*0.2;
    o_trn_nor_pix *= sin(o_trn_nor_pix.x*6.2831)+cos(o_trn_nor_pix.y*6.2831);
    o_trn_nor_pix+=vec2f(sin(time.elapsed), cos(time.elapsed))*.2;
    let o = vec2f(
        sin(o_trn_nor_pix.x*33.),
        cos(o_trn_nor_pix.y*33.)
    );
    let nt = time.elapsed*.3;
    var n2 = min(abs(o.x), abs(o.y));
    let n = length(o_trn_nor_pix);

    let n3 = sin(n2*12.+nt*9.+n*33.);
    let n4 = sin(n2*n*22.-nt*12.+n*111.);
    let n5 = tanh(n2*9.+nt*3.+n*12.);

    let o_col = vec3f(
        n3,
        n4,
        n5*n3+0.1*(1.-n3)*n4
    );
    // Output to screen (linear colour space)
    textureStore(screen, o_trn_pixel.xy, vec4f(o_col, 1.));
}
