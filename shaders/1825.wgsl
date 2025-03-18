const pi = 3.1415926535897931;
const bg = vec4f(0.005,.007,.010,1.);

#storage stuff array<array<atomic<u32>, SCREEN_HEIGHT>, SCREEN_WIDTH>

const starRadius = .1;
const satRadius = starRadius*.4;
const gravConst = .03;

fn pos(res: f32) -> vec2f {
    let r = .7;
    let tm = f32(time.frame)*.0067/r;
    return vec2f(cos(tm), sin(tm))*r;

}

fn hash(p: float3) -> float2
{
    var p3 = fract(p * float3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx+33.33);
    return fract((p3.xx+p3.yz)*p3.zy);
}

fn dot2(p: vec2f) -> float {
    return dot(p,p);
}

fn sqr(p: float) -> float {
    return p*p;
}

@compute @workgroup_size(16, 16)
fn clear(@builtin(global_invocation_id) id: vec3u) {
    textureStore(screen, id.xy, vec4f(0)); // Clear display
    atomicStore(&stuff[int(id.x)][int(id.y)], 0); // Clear atomic screen buffer
}

@compute @workgroup_size(16, 16)
fn physics(@builtin(global_invocation_id) id: vec3u) {

    let r = vec2f(textureDimensions(screen));
    if (f32(id.x) >= r.x || f32(id.y) >= r.y) { return; }
    let U = vec2f(f32(id.x) + .5, r.y - f32(id.y) - .5);
    let uv = U / vec2f(r);
    let cuv = (2.*U-vec2f(r)) / float(r.y);
    let muv = select(
        vec2(pi,pi/2.)*
        (2.*vec2f(mouse.pos)-vec2f(r)) / vec2f(r).y,
        vec2f(time.elapsed, cos(time.elapsed*.2)),
        vec2f(mouse.pos) == vec2f(r)/2.
    );
    
    var t = passLoad(0, vec2i(id.xy), 0);
    // xy : pos
    // zw : vel

    let screenDim = vec2f(r.x/r.y, 1.);

    var h = hash(vec3f(uv, time.elapsed));
    let borderDistance = 30.;
    if (time.frame < 20 || 
        (
            // t.x >  screenDim.x*borderDistance || 
            // t.x < -screenDim.x*borderDistance || 
            // t.y >  screenDim.y*borderDistance || 
            // t.y < -screenDim.y*borderDistance
            //dot2(t.xy) > borderDistance*borderDistance
            false
            //dot2(t.zw) > .1 && dot2(t.xy) > borderDistance*borderDistance
        )
    ){
        // Initialize pos and vel
        var p = vec2f(h-.5)*.2;
        p = vec2f(cos(uv.x*pi*2.), sin(uv.x*pi*2.))*sqrt(uv.y);
        h = hash(vec3f(p.xy-50., time.elapsed))*2.-1.;

        t = vec4(
            p*.01 - pos(r.y)
            //vec2(-.9,0)
            //p
            ,t.zw
        );
        
        t = vec4f(t.xy, 
            //vec2f(0, -.05)//+(vec2f(cos(h.x*pi*2.), sin(h.x*pi*2.))*sqrt(h.y))*.01
            (vec2f(-t.y, t.x))*.1 + 
            (vec2f(cos(h.x*pi*2.), sin(h.x*pi*2.))*sqrt(h.y))*.01
            //*(sin(time.elapsed)*.3 + .7)
            //vec2(0,.08)
        );

        passStore(1, vec2i(id.xy), vec4(0));
    }

    // if (dot2(t.xy) < starRadius*starRadius){
    //     t = vec4f(t.xy, reflect(t.zw, normalize(t.xy)));
    // }

    // if (dot2(pos(r.y) - t.xy) < satRadius*satRadius){
    //     t = vec4f(t.xy, reflect(t.zw, normalize(pos(r.y) - t.xy)));
    // }

    // if (dot2(t.zw) > .1 && dot2(t.xy) > borderDistance){ // kill the ultra powerful jets from getting too close since they will be imprecise
    //     let hPos = h*2.-1.;
    //     t = vec4f(hPos, vec2(-hPos.y, hPos.x)*.1); 
    // }

    // Forces
    var dir = -t.xy;
    let f1 = (gravConst * starRadius*starRadius * normalize(dir)) / dot2(dir);

    //let cf = -.00074*normalize(dir) / length(dir);

    dir = pos(r.y)-t.xy;
    let f2 = (gravConst * satRadius*satRadius * normalize(dir)) / dot2(dir);


    // if (dot2(t.xy - pos()) < .2){
    //     t = vec4(0);
    // }

    // Pos
    t = vec4(
        t.xy + t.zw*.1,
        t.zw
    );

    // Vel
    t = vec4(
        t.xy,
        t.zw + f1 + f2
    );

    passStore(0, vec2i(id.xy), t);

    //t = vec4f(0,.4,0,0);
    //t = vec4f(cos(tm), sin(tm), 0, 0);
    let p = vec2(t.x, t.y)*r.y*.5 + r/2.;
    let pv = vec2(t.x+t.z, t.y+t.w)*r.y*.5 + r/2.;

    // Splat
    if (p.x > 0 && p.x < r.x && p.y > 0 && p.y < r.y){
        //textureStore(screen, vec2u(p), vec4(1));
        //atomicAdd(&stuff[int(p.x)][int(p.y)], 1);

        let iters = 1+int(length(p-pv)); // line segment substeps. 1 px/px + 1
        for (var i = 0; i<iters; i++){ 
            var draw = round(mix(p, pv, (f32(i)-.5)/f32(iters)));
            if (draw.x < r.x && draw.x > 0 && draw.y > 0 && draw.y < r.y){
                atomicAdd(&stuff[int(draw.x)][u32(draw.y)], 
                    u32(1.+exp(-f32(iters))*10.) // not proud of this hack but I
                                                // think it looks better than just 1
                );
            }
        }
    }

}

@compute @workgroup_size(16, 16)
fn img(@builtin(global_invocation_id) id: vec3u) {
    let r = vec2f(textureDimensions(screen));
    if (f32(id.x) >= r.x || f32(id.y) >= r.y) { return; }
    let U = vec2f(f32(id.x) + .5, r.y - f32(id.y) - .5);
    let uv = U / vec2f(r);
    let cuv = (2.*U-vec2f(r)) / float(r.y);

    let val = atomicLoad(&stuff[int(id.x)][int(u32(r.y)-id.y)]);
    let x = 1.-1./(1.+custom.brightness * f32(val));

    var col = vec4(x);

    col += .01*vec4(1., .4, 0., 1.)*starRadius/sqr(length(cuv));
    col += .01*vec4(0, .4, .8, 1.)*satRadius/sqr(length(pos(r.y) - cuv));
    col /= col + 1;

    let t = passLoad(1, vec2i(id.xy), 0);

    col = mix(t, col, 1e-1);
    textureStore(screen, id.xy, col); // move line up to disable accumulation
    
    passStore(1, vec2i(id.xy), col);
}
