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
    o: vec2f
)->f32{
    return max(abs(o.x),abs(o.y));
}
@compute @workgroup_size(16, 16)
fn main_image(
    @builtin(global_invocation_id) o_trn: vec3u
) {
    
    let o_scl = textureDimensions(screen);
    if (o_trn.x >= o_scl.x || o_trn.y >= o_scl.y) { return; }
    // Pixel coordinates (centre of pixel, origin at bottom left)
    var o_trn_nor = (vec2f(o_trn.xy)-vec2f(o_scl.xy)*.5)/vec2f(o_scl.yy);
    let n1 = f_n_signed_square(
        f_o_rotated(
            o_trn_nor,
            0.
        )
    );
    let n2 = f_n_signed_square(
        f_o_rotated(
            o_trn_nor,
            n_tau*time.elapsed*.2
        )
    );
    let n_t = time.elapsed;
    let nd = length(o_trn_nor);
    let nr = sin(n1*33.*sin(nd*n_tau*2.*sin(nd*3.*sin(nd*3.+n_t))));
    let ng = sin(n1*11.*sin(nd*n_tau*5.*sin(nd*3.*sin(nd*3.+n_t+.1))));
    let nb = sin(n1*3.*sin(nd*n_tau*3.*sin(nd*3.*sin(nd*3.+n_t+.2))));
    let o_col = vec3f(
        nr,
        ng,
        nb
    ).zxy;
    // Output to screen (linear colour space)
    textureStore(screen, o_trn.xy, vec4f(o_col, 1.));
}
