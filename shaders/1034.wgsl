const n_tau = 6.2831;
fn f_o_rotated(
    o: vec2f, 
    n_radians: f32
)-> vec2f{
    return vec2f(
        -sin(n_radians)*o.x+cos(n_radians)*o.y, 
        cos(n_radians)*o.x+sin(n_radians)*o.y
    );
}
fn f_n_signed_square(
    o: vec2f, 
    n_radius: f32
)->f32{
    return (max(abs(o.x),abs(o.y))-n_radius);
}
@compute @workgroup_size(16, 16)
fn main_image(
    @builtin(global_invocation_id) o_trn: vec3u
) {
    
    let o_scl = textureDimensions(screen);
    if (o_trn.x >= o_scl.x || o_trn.y >= o_scl.y) { return; }
    // Pixel coordinates (centre of pixel, origin at bottom left)
    var o_trn_nor = (vec2f(o_trn.xy)-vec2f(o_scl.xy)*.5)/vec2f(o_scl.yy);
    o_trn_nor*=2.;
    let n_its = 3.;
    var o_prod = vec3f(0.);    
    for(var n_it = 0.; n_it < n_its; n_it +=1.){
        let n_it_nor = n_it / n_its;
        let o_p = vec2f(
            sin(n_it_nor*n_tau), 
            cos(n_it_nor*n_tau)
        )*(sin(time.elapsed)*.5+.5);
        var n = length(o_trn_nor - o_p);
        let n_pixels_blur = 3.;
        let n_ratio = 1./f32(max(o_scl.x, o_scl.y));
        let n_radius = .5;
        n = smoothstep(
            n_radius, 
            n_radius-n_pixels_blur*n_ratio, 
            n
        );
        var oc = vec3f(0.);
        oc[u32(n_it)] = n;
        o_prod += oc;//vec3f(n)[u32(n_it)];
    }

    let o_col = vec3f(
        o_prod
    );
    // Output to screen (linear colour space)
    textureStore(screen, o_trn.xy, vec4f(o_col, 1.));
}
