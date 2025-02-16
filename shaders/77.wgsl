@compute @workgroup_size(16, 16)
fn pass1(@builtin(global_invocation_id) id: uint3) 
{
    let col = float4(float(id.x) / float(textureDimensions(screen).x));
    textureStore(pass_out, int2(id.xy), 0, col);
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    //render result
    let col = textureLoad(pass_in, int2(id.xy), 0, 0);
    textureStore(screen, int2(id.xy), col);
}