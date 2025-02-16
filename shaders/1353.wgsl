const n_tau = 6.2831;
const o_trn__mouse_last_frame = vec2(0);

@compute @workgroup_size(16, 16)
fn main_image(
    @builtin(global_invocation_id) o_trn: vec3u
) {
    var o_scl = textureDimensions(screen);
    
    let n_scl_min = min(f32(o_scl.x), f32(o_scl.y));
    let naa = 1./n_scl_min;
    let o_trn_nor = (vec2f(o_trn.xy)-vec2f(o_scl.xy)*.5) / n_scl_min; 
    let o_trn_nor_mou = (vec2f(mouse.pos.xy)-vec2f(o_scl.xy)*.5) / n_scl_min;
    let n = abs(o_trn_nor.x-o_trn_nor_mou.x);
    var o = float4(n);

    if(o_trn.y > 0){
        o = textureLoad(
            pass_in, 
            o_trn.xy-vec2(0, 1), 
            0, 0
        );
    }

    textureStore(pass_out, o_trn.xy, 0, o);
    textureStore(screen, o_trn.xy, o);
}
