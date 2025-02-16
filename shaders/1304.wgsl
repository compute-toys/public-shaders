const pi = 3.1415926535897931;

#storage stuff array<array<atomic<u32>, SCREEN_HEIGHT>, SCREEN_WIDTH>
// 0-2048. fixed point using 1<<20 (as opposed to 1<<31 which is 0-1)


fn hash(p: float3) -> float {
    var p3  = fract(p * .1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
} // Just took hash13 from Dave_Hoskins/hash instead of importing everything

@compute @workgroup_size(16, 16)
fn particles(@builtin(global_invocation_id) id: vec3u){
    let r = textureDimensions(screen);
    if (id.x >= r.x || id.y >= r.y) { return; }
    let U = vec2f(f32(id.x) + .5, f32(r.y - id.y) - .5);
    let cuv = (2.*U-vec2f(r)) / float(r.y);
    let aspect = float(r.x)/float(r.y);

    let pos = mix(
        vec3f(
            sin(cuv.x / aspect * pi)*cos(cuv.y * pi/2.),
            cos(cuv.x / aspect * pi)*cos(cuv.y * pi/2.) + .5,
            sin(cuv.y * pi/2.)
        ),
        vec3f(
            cuv.x, 
            cos(cuv.x*20.)*.02*sin(cuv.y*10.) 
                + sin(time.elapsed + hash(vec3(0, U.xy))*2.*pi) * .02, 
            cuv.y
        ),
        // #if ${true}
        // 1.
        // #else
        smoothstep(-.5,.0,length(cuv)-(sin(time.elapsed)*.5+.5)*(sqrt(aspect*aspect + 1.)+.5))
        // #endif
    );

    passStore(0, int2(id.xy), 
        vec4(
            pos,
            hash(vec3(U.xy, 0))
        )
    );
}

@compute @workgroup_size(16, 16)
fn clear(@builtin(global_invocation_id) id: vec3u) {
    textureStore(screen, int2(id.xy), vec4f(0.005,.007,.010,1.));
    atomicStore(&stuff[int(id.x)][int(id.y)], 1<<31);
}

@compute @workgroup_size(16, 16)
fn splat(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let r = textureDimensions(screen);
    if (id.x >= r.x || id.y >= r.y) { return; }
    let U = vec2f(f32(id.x) + .5, f32(r.y - id.y) - .5);
    let uv = U / vec2f(r);
    let cuv = (2.*U-vec2f(r)) / float(r.y);
    let muv = 
    vec2(pi,pi/2.)*
    (2.*vec2f(mouse.pos)-vec2f(r)) / vec2f(r).y;


    var o = vec3f(0);
    let camf = vec3f(sin(muv.x)*cos(muv.y), cos(muv.x)*cos(muv.y), sin(muv.y));
    let camr = normalize(cross(camf, vec3(0, 0, 1)));
    let camu = cross(camr, camf);
    o -= camf*2.;

    let t = passLoad(0, int2(id.xy), 0);

    var local = vec3f(0);
    local.x = dot(t.xyz-o, camr);
    local.y = dot(t.xyz-o, camf);
    local.z = dot(t.xyz-o, camu);

    if (local.y < 0.) { return; }

    var coord = local.xz / local.y*2.;
    if (abs(coord.y) > 1. || abs(coord.x) > float(r.x)/float(r.y)){ return; }
    

    coord = .5*((coord * float(r.y)) + vec2f(r));
    let ic = int2(coord);

    // // Method 1
    // if (local.y < f32(atomicLoad(&depth[ic.x][ic.y]))/f32(1<<20)){
    //     let tex = textureSampleLevel(channel0, bilinear, uv, 0);
    //     textureStore(screen, int2(ic), 
    //         //passLoad(1, int2(id.xy), 0) +
    //         vec4f(tex.xyz/(1.+tex.xyz), local.y)
    //     );
    //     atomicStore(&depth[ic.x][ic.y], u32(local.y*f32(1<<20)));
    // }

    // // Method 2
    // let atom = atomicMin(&depth[ic.x][ic.y], u32(local.y*f32(1<<20)));
    // if (atom >= u32(local.y*f32(1<<20))){ // if old depth was more than current
    //                                      // particle, overwrite
    //     let tex = textureSampleLevel(channel0, bilinear, uv, 0);
    //     textureStore(screen, ic, 
    //         vec4f(tex.xyz/(1.+tex.xyz), local.y)
    //     );
    // }

    // // Method 3
    var tex = textureSampleLevel(channel0, bilinear, uv, 0);
    tex /= (1.+tex);
    var data = u32(min(local.y/8., 1.) * 255.) << 24;
    data += u32(tex.r * 255.) << 16;
    data += u32(tex.g * 255.) << 8;
    data += u32(tex.b * 255.);
    atomicMin(&stuff[ic.x][ic.y], data);
}

// Necessary part of Method 3 - remove if using any of the others
@compute @workgroup_size(16, 16)
fn img(@builtin(global_invocation_id) id: vec3u) {
    let ic = int2(id.xy);
    let data = atomicLoad(&stuff[ic.x][ic.y]);
    //d,r,g,b;
    let d = (data>>24);
    let r = (data>>16)%256;
    let g = (data>>8)%256;
    let b = data%256;

    textureStore(screen, ic, 
        vec4f(f32(r),f32(g),f32(b),1.)/f32(256)
    );
}
