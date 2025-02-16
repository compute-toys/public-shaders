const pi = 3.1415926535897931;
const bg = vec4f(0.005,.007,.010,1.);
const saturation = .4;

#storage stuff array<array<atomic<u32>, SCREEN_HEIGHT>, SCREEN_WIDTH>

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

    var pos: vec3f;
    var a: u32;
    var b: u32;
    var i: u32;
    var data: f32;

    if (time.frame < 20){
        a = u32(r.x * r.y);
        b = u32(pow(float(a), 1./3.));
        i = u32(U.x) + u32(U.y)*u32(r.x);

        pos = vec3f(
            float(i%b),
            float((i/b)%b),
            float(i/b/b)
        )/float(b) - .5;

        data = float(i)/float(a) + .5;
    } else {
        let t = passLoad(0, int2(id.xy), 0);
        pos = t.xyz;
        data = t.w;

        pos += vec3f(-sin(atan2(pos.y, pos.x)), cos(atan2(pos.y, pos.x)), 0)*.01;
        pos += vec3f(-sin(atan2(pos.z, pos.x)), 0, cos(atan2(pos.z, pos.x)))*.01;
        pos += vec3f(0, -sin(atan2(pos.z, pos.y)), cos(atan2(pos.z, pos.y)))*.01;

        pos.z += sin(pos.y*(10.+sin(time.elapsed)))
        *(.001 + .002*t.w)
        *sin(time.elapsed*.1);
        pos += normalize(pos)*exp(-length(pos))*.001;
    }

    passStore(0, int2(id.xy), 
        vec4(
            pos,
            data
        )
    );
}

@compute @workgroup_size(16, 16)
fn clear(@builtin(global_invocation_id) id: vec3u) {
    textureStore(screen, int2(id.xy), bg); // Clear screen
    atomicStore(&stuff[int(id.x)][int(id.y)], 1<<31); // Clear atomic screen buffer
}

@compute @workgroup_size(16, 16)
fn splat(@builtin(global_invocation_id) id: vec3u) {
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

    let coord = local.xz / local.y*2.;
    if (abs(coord.y) > 1. || abs(coord.x) > float(r.x)/float(r.y)){ return; }
    
    let ic = int2(.5*((coord * float(r.y)) + vec2f(r)));

    let coeff = f32(1<<15)-1.;
    var data = u32(min(local.y/8., 1.) * coeff) << 16;
    data += u32(t.w * coeff);
    atomicMin(&stuff[ic.x][ic.y], data);
}

@compute @workgroup_size(16, 16)
fn img(@builtin(global_invocation_id) id: vec3u) {
    let ic = int2(id.xy);
    let data = atomicLoad(&stuff[ic.x][ic.y]);
    //d,r,g,b;
    if (data == 1<<31){
        return;
    } else {
        let d = (data>>16);
        let a = data%(1<<15);
        let b = float(a)/float(1<<15);
        let depth = float(d)/float(1<<15);

        var col: vec3f;
        if (d < 1<<15){
            col = mix(
                vec3f(b),
                (cos(b*pi*2. + vec3f(0,2.*pi/3.,4.*pi/3.))*.5+.5)
                //*(cos(b*2.*pi*50.)*.5 + .5)
                ,
                saturation
            );
        }

        textureStore(screen, ic, 
            vec4f(col,1.) / (depth*10.)
        );
    }
}
