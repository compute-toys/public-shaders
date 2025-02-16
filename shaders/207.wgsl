#storage state atomic<uint>

@compute @workgroup_size(1)
fn clear_state(@builtin(global_invocation_id) id: uint3) {
    atomicStore(&state, 0);
}

@compute @workgroup_size(1, 256)
// @compute @workgroup_size(256, 1)
// @compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    let screen_size = textureDimensions(screen);
    if (any(id.xy >= screen_size)) { return; }

    let coord = float2(id.xy) + 0.5;
    let uv = coord / float2(screen_size);

    var col = float3(pow(uv.xy, vec2(2.2)), 0.0);

    let order = atomicAdd(&state, 1);

    let a = float(order) / float(screen_size.x) / float(screen_size.y);
    col *= step(float(a), custom.clamp * (float(time.frame%999)/999));

    textureStore(screen, id.xy, float4(col, 1.));
}
