//https://compute.toys/view/15
fn hash44(p: float4) -> float4 {
    var p4 = fract(p  * float4(.1031, .1030, .0973, .1099));
    p4 += dot(p4, p4.wzxy+33.33);
    return fract((p4.xxyz+p4.yzzw)*p4.zywx);
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if id.x >= screen_size.x || id.y >= screen_size.y { return; }

    var state = textureLoad(pass_in, id.xy, 0, 0);

    var rand = vec4f(vec2f(id.xy), f32(time.frame), custom.seed);
    rand = hash44(rand);
    let spawn = vec2u(rand.xy * vec2f(screen_size));

    if id.x == spawn.x && id.y == spawn.y && time.frame <= 5 {
        state = vec4f(1., 0., 0., 0.);
        textureStore(pass_out, id.xy, 0, state);
    } else {
        state.y = max(state.y, state.x);
        state.x = 0.;
        for(var i = 0; i < 2; i++) {
            rand = hash44(rand);
            let dir = rand.x * radians(360.);
            
            let sp = vec2f(id.xy) + vec2f(cos(dir), sin(dir)) / rand.y;

            let sip = vec2i(sp + 0.5);

            if (sip.x < i32(screen_size.x) && sip.y < i32(screen_size.y) && sip.x >= 0 && sip.y >= 0) { 
                let ss = textureLoad(pass_in, sip, 0, 0);
                state.x = max(state.x, ss.x - custom.subtraction * rand.z);
            }
        }


        var col = vec3f(pow(state.y, 3.), pow(state.y, 2.), pow(state.y, 1.));
        

        textureStore(pass_out, id.xy, 0, state);
        textureStore(screen, id.xy, vec4f(col, 1.));
    }
}
