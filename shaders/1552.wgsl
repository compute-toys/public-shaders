
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
    o_trn_nor_pix*= 2.+sin(time.elapsed*.1)*.9;
    var o_trn_nor_pix2 = o_trn_nor_pix*vec2f(10.,10.);
    o_trn_nor_pix2+=.5;
    
    var o_trn_nor_pix2_flr = floor(o_trn_nor_pix2);
    o_trn_nor_pix2 = fract(o_trn_nor_pix2)-.5;
    o_trn_nor_pix2 *=2.;
    var n_tau = radians(360.);
    var nang = (atan2(o_trn_nor_pix.x, o_trn_nor_pix.y)/n_tau);
    var nang2 = (atan2(o_trn_nor_pix2_flr.x, o_trn_nor_pix2_flr.y)/n_tau);
    nang2 = sin(nang2*n_tau*4.)*.5+.5;
    var ndcntr2 = length(o_trn_nor_pix2_flr);
    //o_trn_nor_pix *= 0.05;
    let nt = time.elapsed*2.;
    var o_t = vec3f(
        time.elapsed*2.+nang2*1.,
        time.elapsed*2.+nang2*2.,
        time.elapsed*2.+nang2*3.,
    );
    var ncirc = length(o_trn_nor_pix2);
    var nsqr = max(abs(o_trn_nor_pix2.x), abs(o_trn_nor_pix2.y));
    var nintrpl = .5;
    nintrpl = 0.1;
    var nradius = (sin(nang2+nt+ndcntr2*1.2)*.5+.5)*.5;
    var oradius = vec3f(
        (sin(nang2+o_t[0]+ndcntr2*1.2)*.5+.5)*.5,
        (sin(nang2+o_t[1]+ndcntr2*1.2)*.5+.5)*.5,
        (sin(nang2+o_t[2]+ndcntr2*1.2)*.5+.5)*.5,
    );
    var ointpl = vec3f(
        oradius*2.
    );

    nintrpl = nradius*2.;
    nradius -= 0.1;
    //nintrpl = 0.9;
    var n = nintrpl*ncirc+(1.-nintrpl)*nsqr;
    var on = vec3f(
        ointpl[0]*ncirc+(1.-ointpl[0])*nsqr,
        ointpl[1]*ncirc+(1.-ointpl[1])*nsqr,
        ointpl[2]*ncirc+(1.-ointpl[2])*nsqr
    );
    on  = abs(on-oradius);
    on = vec3f(
        1.-pow(on[0], 1./3.),
        1.-pow(on[1], 1./3.),
        1.-pow(on[2], 1./3.),
    );
    n = abs(n-nradius);
    n = pow(n, 1./3.);
    var o_col = vec3(
        on[0],
        on[1],
        on[2],
    );
    // Convert from gamma-encoded to linear colour space
    //o_col = pow(o_col, vec3f(2.2));

    // Output to screen (linear o_colour space)
    textureStore(screen, o_trn_pixel.xy, vec4f(o_col, 1.));
    if(o_trn_nor_pix.x > 0.){
        //textureStore(screen, o_trn_pixel.xy, vec4f(vec3f(nr), 1.));
    }
}
