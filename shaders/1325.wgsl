const n_tau = radians(360.);
fn f_n_quadrant(o: vec2<f32>) -> f32 {
    let n_x_bit = u32(o.x > 0.); // 0 if uv.x < 1.0, 1 if uv.x >= 1.0
    let n_y_bit = u32(o.y > 0.); // 0 if uv.y < 1.0, 1 if uv.y >= 1.0

    // Combine the bits to form the quadrant number
    let n_quadrant = (n_x_bit << 1) | n_y_bit;

    return float(n_quadrant);
}
fn f_o_rotd(o_p: vec2<f32>, n_radians: f32) -> vec2<f32> {
    let n_cos = cos(n_radians);
    let n_sin = sin(n_radians);

    return mat2x2<f32>(n_cos, -n_sin, n_sin, n_cos)*o_p.xy;
}
fn f_n(
    o_trn_nor_pix: vec2f, 
    n_t: f32, 
    n_its: f32
)->f32{
    
    // let n_its = 9.;
    var n_sum = 0;
    for(var n_it = 1; n_it<=int(n_its);n_it+=1){
        n_sum+=(n_it*2);
    }
    let o_trn_nor2 = o_trn_nor_pix*float(n_sum);
    let n_it_nor_one = 1./n_its;
    var n = 1.;
    var o = vec2f(
        0.,
        0.
    );
    
    var n_min = 1.;
    for(var n_it_nor = 0.; n_it_nor < 1.; n_it_nor+=n_it_nor_one){
        let n_it_norr = 1.-n_it_nor;

        let n_it = floor(n_it_nor*n_its);
        let n_itr = n_its-n_it;//iteration reversed
        if(n_it > 0){
            
            o.y -= n_itr;
        }
        o.x = sin((n_t+n_it_nor)*n_tau)*n_itr*4.*n_it_nor;
        var nl = length(o_trn_nor2-o);
        
        let n_radius = n_itr;
        nl = abs(nl - n_radius)/n_radius;
        nl = smoothstep(0., 10., nl);
        var b = 1./n_its*.03;
        nl = -b*(1./(nl+b))+1.0; 
        n *= nl;
        n_min = min(nl,n_min);
    }
    return n;

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
    let n_quadrant_nor = f_n_quadrant(o_trn_nor_pix)/4.;
    o_trn_nor_pix*=1.2;
    o_trn_nor_pix= -1.*abs(o_trn_nor_pix);
    o_trn_nor_pix = f_o_rotd(o_trn_nor_pix, -n_tau*(1./8.));

    let n_t = time.elapsed;
    let o_col = clamp(vec3f(
        1.-f_n(o_trn_nor_pix.xy,  ((1./3.)+n_t), 18.),
        1.-f_n(o_trn_nor_pix.xy,  ((2./3.)+n_t), 18.),
        1.-f_n(o_trn_nor_pix.xy, ((3./3.)+n_t),18.),
    ),vec3f(0.), vec3f(1.));
    // nm = length(o_trn_nor_pix);
    // Convert from gamma-encoded to linear colour space
    //o_col = pow(o_col, vec3f(2.2));

    // Output to screen (linear o_colour space)
    textureStore(screen, o_trn_pixel.xy, vec4f(o_col, 1.));
}
