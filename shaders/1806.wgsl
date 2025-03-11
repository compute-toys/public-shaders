const pi = 3.1415926535897931;
const bg = vec4f(0.005,.007,.010,1.);
const saturation = .4;

#storage stuff array<array<atomic<u32>, SCREEN_HEIGHT>, SCREEN_WIDTH>

fn dot2(p: vec2f) -> f32 {
    return dot(p, p);
}

@compute @workgroup_size(16, 16)
fn particles(@builtin(global_invocation_id) id: vec3u){
    let r = textureDimensions(screen);
    if (id.x >= r.x || id.y >= r.y) { return; }
    let U = vec2f(f32(id.x) + .5, f32(r.y - id.y) - .5);
    let cuv = (2.*U-vec2f(r)) / f32(r.y);
    let aspect = f32(r.x)/f32(r.y);

    var pos: vec3f;
    var a: u32;
    var b: u32;
    var i: u32;
    var data: f32;

    if (time.frame < 20){
        a = u32(r.x * r.y);
        b = u32(pow(f32(a), 1./3.));
        i = u32(U.x) + u32(U.y)*u32(r.x);

        pos = vec3f(
            f32(i%b),
            f32((i/b)%b),
            f32(i/b/b)
        )/f32(b) - .5;

        data = f32(i)/f32(a);
    } else {
        let t = passLoad(0, int2(id.xy), 0); // Next step : PHYSICS
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
    textureStore(screen, int2(id.xy), vec4(0,1,0,0)); // Clear screen
    atomicStore(&stuff[int(id.x)][int(id.y)], 0); // Clear atomic screen buffer
}

@compute @workgroup_size(16, 16)
fn splat(@builtin(global_invocation_id) id: vec3u) {
    let r = textureDimensions(screen);
    if (id.x >= r.x || id.y >= r.y) { return; }

    if (id.x > u32(f32(r.x)*custom.particles)){ return; } // Skip some of the particles

    let U = vec2f(f32(id.x) + .5, f32(r.y - id.y) - .5);
    let uv = U / vec2f(r);
    let cuv = (2.*U-vec2f(r)) / f32(r.y);
    let muv = 
    vec2(pi,pi/2.)*
    (2.*vec2f(mouse.pos)-vec2f(r)) / vec2f(r).y;

    var o = vec3f(0);
    let camf = vec3f(sin(muv.x)*cos(muv.y), cos(muv.x)*cos(muv.y), sin(muv.y));
    let camr = normalize(cross(camf, vec3(0, 0, 1)));
    let camu = cross(camr, camf);
    o -= camf*3.;

    let t = passLoad(0, int2(id.xy), 0);

    var local = vec3f(0);
    local.x = dot(t.xyz-o, camr);
    local.y = dot(t.xyz-o, camf);
    local.z = dot(t.xyz-o, camu);

    if (local.y < 0.) { return; }

    let coord = local.xz / local.y*2.;
    if (abs(coord.y) > 1. || abs(coord.x) > f32(r.x)/f32(r.y)){ return; }
    
    let ic = int2(.5*((coord * f32(r.y)) + vec2f(r)));

    // depth is local.y
    let size = abs(custom.radius/local.y - custom.radius/custom.focus) + 1.;
    let iSize = i32(size)+1;

    var screenOut: u32;

    for (var x = -iSize; x < iSize; x++){
        for (var y = -iSize; y < iSize; y++){
            var l = dot2(vec2f(f32(x), f32(y)));
            if (l < size){
                let prev = atomicLoad(&stuff[ic.x + x][ic.y + y]);
                
                if (custom.grayscale < .5){

                    var r = ((prev>>24)       );
                    var g = ((prev>>16)%(1<<7));
                    var b = ((prev>>8 )%(1<<7));
                    var layer = ( prev     %(1<<7));

                    let x = t.w;
                    let col = 
                        vec3(x*x, pow(x*2.-1., 2.), 1.-x);
                        //cos(t.w + vec3f(0,2.*pi/3.,4.*pi/3.))*.5+.5;

                    var rOut = u32(col.r * custom.brightness
                        * f32(1<<7)
                        * smoothstep(size, size-sqrt(2.), l) / (size*size)
                        * (1.-f32(r)/f32(1<<7))
                    );

                    var gOut = u32(col.g * custom.brightness 
                        * f32(1<<7)
                        * smoothstep(size, size-sqrt(2.), l) / (size*size)
                        * (1.-f32(g)/f32(1<<7))
                    );

                    var bOut = u32(col.b * custom.brightness
                        * f32(1<<7)
                        * smoothstep(size, size-sqrt(2.), l) / (size*size)
                        * (1.-f32(b)/f32(1<<7))
                    );

                    let layerOut: u32 = 1; // Beware, at high values this does spill into blue

                    screenOut = 
                        (rOut<<24) + 
                        (gOut<<16) + 
                        (bOut<<8 ) + 
                        layerOut;
                } else {

                    screenOut = u32(custom.brightness*f32(1<<31)
                        * smoothstep(size, size-sqrt(2.), l) / (size*size)
                        * (1.-f32(prev)/f32(1<<31))
                    );

                }
                
                atomicAdd(&stuff[ic.x + x][ic.y + y], screenOut);
            }
        }
    }


}

@compute @workgroup_size(16, 16)
fn img(@builtin(global_invocation_id) id: vec3u) {
    let ic = int2(id.xy);
    let data = atomicLoad(&stuff[ic.x][ic.y]);

    var O: vec4f;

    if (custom.grayscale < .5){
        let r = f32((data>>24)       )/f32(1<<7);
        let g = f32((data>>16)%(1<<7))/f32(1<<7);
        let b = f32((data>>8 )%(1<<7))/f32(1<<7);
        let l = f32( data     %(1<<7));

        O = 
            vec4(r,g,b,1.) // (f32(l))
            //vec4(l) / (1<<7)
        ;
    } else {
        O = vec4f(f32(data)/f32(1<<31));
    }

    O = pow(O, vec4f(2.2));
    O += bg;

    textureStore(screen, ic, O);
}
