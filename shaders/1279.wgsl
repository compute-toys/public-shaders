const pi = 3.1415926535897931;

fn hash(p: float3) -> float {
    var p3  = fract(p * .1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
} // Just took hash13 from Dave_Hoskins/hash instead of importing everything

@compute @workgroup_size(16, 16)
fn splat(@builtin(global_invocation_id) id: vec3u){
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);

    // Normalised pixel coordinates (from 0 to 1)
    let uv = fragCoord / vec2f(screen_size);
    let cuv = (2. * fragCoord - vec2f(screen_size)) / float(screen_size.y);
    

    var c = uv;
    var f = 0.;
    if (time.frame > 10){

        let t = passLoad(0, int2(id.xy), 0).xyz;
        c = t.xy;
        f = t.z;

        let h = floor(
            hash(vec3(vec2f(id.xy), 
                float(time.frame)
                //time.elapsed // Broken, cool look
            ))*3.
        ) + select(0, time.elapsed*custom.spin, custom.spin > .0);

        let corner = (
            vec2f(sin(2 * h * pi / 3.), cos(2 * h * pi / 3.))
            - select(vec2(0, .2), vec2(0), custom.spin > .0)
        )//  * (3. - 2.*float(mouse.pos.y)/float(screen_size.y))
        ;

        c = mix(c, corner, .5);


        let coord = int2(
            (c * float(screen_size.y) + vec2f(screen_size)) / 2.
        );

        let prev = passLoad(0, coord, 0);
        passStore(
            0,
            coord,
            vec4(prev.xy, min(1, prev.z + .01), 1)  
        );

    }
    
    passStore(0, int2(id.xy), vec4(c, f, 1));
}

@compute @workgroup_size(16, 16)
fn fade(@builtin(global_invocation_id) id: vec3u){
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    passStore(0, int2(id.xy), 
        passLoad(0, int2(id.xy), 0)*vec4(1,1,.99,1)
    );
}


@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) coord: vec3u) {
    let t = passLoad(0, int2(coord.xy), 0);
    textureStore(screen, coord.xy, vec4(t.z) + vec4(t.xy*.02, 0,  0));
}
