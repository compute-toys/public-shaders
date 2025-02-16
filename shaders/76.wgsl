#storage stor1 array<float>

@compute @workgroup_size(16, 16)
fn pass1(@builtin(global_invocation_id) id: uint3) { 
    //store a value
    let idx = id.y * textureDimensions(screen).x + id.x;
    stor1[idx] = float(id.x) / float(textureDimensions(screen).x);
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    //render result
    let idx = id.y * textureDimensions(screen).x + id.x;
    let val = stor1[idx]; 
    let col = float3(val);
    textureStore(screen, int2(id.xy), float4(col, 1.));
}