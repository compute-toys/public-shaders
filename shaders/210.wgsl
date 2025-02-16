#storage state atomic<uint>
#define PERIOD 3000

#workgroup_count clear_state 1 1 1
@compute @workgroup_size(1)
fn clear_state(@builtin(global_invocation_id) id: uint3) {
    atomicStore(&state, 0);
}

//@compute @workgroup_size(1, 256)
//@compute @workgroup_size(256, 1)
@compute @workgroup_size(16, 16)
//@compute @workgroup_size(1, 1)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    let screen_size = textureDimensions(screen);
    if (any(id.xy >= screen_size)) { return; }

    if(time.frame%PERIOD == 0)
    {
        //get index of thread
        let order = atomicAdd(&state, 1);
        //store it into a texture
        textureStore(pass_out, int2(id.xy), 0, float4(float(order)/ float(screen_size.x*screen_size.y)));
    }
    
    let order = textureLoad(pass_in, int2(id.xy), 0, 0).x;
    var time = float(time.frame%PERIOD)/float(PERIOD);
    var gradient = 0.05;
    if(mouse.click == 1)
    {
        time = float(mouse.pos.x) / float(screen_size.x);
        gradient = float(mouse.pos.y) / float(screen_size.y);
    }

    let transition0 = step(time, order);
    let transition1 = smoothstep(time - 0.02*gradient/0.05, time, order);
    let transition2 = smoothstep(time - gradient, time, order);
    var color = mix(float3(1.0,1.0,1.0),float3(0.0,0.0,0.0),transition0) * transition1 * transition2;
    color += mix(float3(0.1,0.1,1.0),float3(0.0,0.0,0.0),transition1) * transition2;
    color += mix(float3(0.0,0.0,0.05),float3(0.0,0.0,0.0),transition2);
    textureStore(screen, id.xy, float4(color, 1.0));
}
