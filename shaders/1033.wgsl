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
    let n1 = f_n_signed_square(
        f_o_rotated(
            o_trn_nor,
            time.elapsed
        ),
        sin(time.elapsed)*.5+.5
    );
    var o_col = vec3f(.8, .3, .1);
    if(n1 < .0){ o_col = vec3f(.2, .1, .9);}
    var n = abs(n1);
    let n_plus_minus = sin(n*99.)*.01;
    o_col = o_col * n + n_plus_minus;
    // Output to screen (linear colour space)
    textureStore(screen, o_trn.xy, vec4f(o_col, 1.));
}
