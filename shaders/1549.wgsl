
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
    var nang = (atan2(o_trn_nor_pix.x, o_trn_nor_pix.y)/n_tau);

    //o_trn_nor_pix *= 0.05;
    let nt = time.elapsed*2.;
    
    var ncirc = length(o_trn_nor_pix);
    var nsqr = max(abs(o_trn_nor_pix.x), abs(o_trn_nor_pix.y));
    var nintrpl = 1.-ncirc;
    
    //nintrpl = 0.9;
    var n = nintrpl*ncirc+(1.-nintrpl)*nsqr;
    n += (sin(nang*n_tau*3.+nt)*.5+.5)*nsqr*.2;
    var c = 1.;//sin(nt*.2)*.5+.5;
    nr = (sin(n*n_tau*3.-nt*.5)*.5+.5)*0.5;
    var ncoloff = sin(ncirc*n_tau);//sin(nt*n_tau)*.5+.5;
    var o_col = vec3(
        sin((1./nr)+(sin(nr*n_tau*20.))+nt+ncoloff*1.)*.5+.5,
        sin((1./nr)+(sin(nr*n_tau*20.))+nt+ncoloff*2.)*.5+.5,
        sin((1./nr)+(sin(nr*n_tau*20.))+nt+ncoloff*3.)*.5+.5
    );
    // Convert from gamma-encoded to linear colour space
    //o_col = pow(o_col, vec3f(2.2));

    // Output to screen (linear o_colour space)
    textureStore(screen, o_trn_pixel.xy, vec4f(o_col, 1.));
    if(o_trn_nor_pix.x > 0.){
        //textureStore(screen, o_trn_pixel.xy, vec4f(vec3f(nr), 1.));
    }
}
