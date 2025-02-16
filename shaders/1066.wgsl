
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
    let o_trn_nor_pix = (vec2f(o_trn_pixel.xy)-vec2f(o_scl_screen)*.5)
        /vec2f(o_scl_screen).yy;

    var o_trn_nor_mou = (vec2f(mouse.pos)-vec2f(o_scl_screen)*.5)
        /vec2f(o_scl_screen).yy;

    if(mouse.click == 0){
        o_trn_nor_mou = vec2f(
            sin(time.elapsed),
            cos(time.elapsed)
        );
    }
    // Convert from gamma-encoded to linear colour space
    //o_col = pow(o_col, vec3f(2.2));
    let n1 = length(o_trn_nor_pix);
    let n2 = max(abs(o_trn_nor_pix.x), abs(o_trn_nor_pix.y));
    var nt = length(o_trn_nor_mou-o_trn_nor_pix);
    nt = sin(nt*33.*sin(time.elapsed*.2)); 
    var n = nt * n1 + (1.-nt)*n2;
    n = sin(n*33.+time.elapsed);
    let o_col = vec3f(
        sin(n*3.+time.elapsed+1.*n2),
        sin(n*3.+time.elapsed+2.*n2),
        sin(n*3.+time.elapsed+3.*n2),
    );
    // Output to screen (linear o_colour space)
    textureStore(screen, o_trn_pixel.xy, vec4f(o_col, 1.));
}
