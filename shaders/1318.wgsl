const pi = 3.1415926535897931;
const eps = 1e-3;
const ior = 1.5;
const bg =  vec3f(0.005,.007,.010);// vec3f(.079,.104,.12);

const res = 1024;
#storage stuff array<array<vec3f, res>, res>

const lightDir = vec3f(0,0,-1);

// Yoinked from https://www.shadertoy.com/view/ls2Bz1 //

// --- Spectral Zucconi --------------------------------------------
// By Alan Zucconi
// Based on GPU Gems: https://developer.nvidia.com/sites/all/modules/custom/gpugems/books/GPUGems/gpugems_ch08.html
// But with values optimised to match as close as possible the visible spectrum
// Fits this: https://commons.wikimedia.org/wiki/File:Linear_visible_spectrum.svg
// With weighter MSE (RGB weights: 0.3, 0.59, 0.11)
fn sat (x: vec3f) -> vec3f {
    return min(vec3(1), max(vec3(0),x));
}

fn bump3y (x: vec3f, yoffset: vec3f) -> vec3f {
	var y = vec3(1) - x * x;
	y = sat(y-yoffset);
	return y;
}

fn spectral_zucconi (w: f32) -> vec3f {
    // w: [400, 700]
	// x: [0,   1]
	let x = saturate((w - 400.0)/ 300.0);

	const cs = vec3f(3.54541723, 2.86670055, 2.29421995);
	const xs = vec3f(0.69548916, 0.49416934, 0.28269708);
	const ys = vec3f(0.02320775, 0.15936245, 0.53520021);

	return bump3y (	cs * (x - xs), ys);
}

// [ END YOINK ] //

fn spectrum(l: f32) -> vec3f {
    // return pow(
    //     (sin(l/50. + vec3f(0,1,2) + 2)*.5+.5) * (exp(-pow((l-550)/100, 2))),
    //     vec3f(2.2)
    // );
    // return pow(
    //     exp(-pow((l - vec3f(590, 550, 500))/vec3f(50,70,50), vec3f(2)))
    //     + exp(-pow((l - vec3f(460, 0, 0))/vec3f(40,1,1), vec3f(2)))*.2
    //     ,vec3f(2.2)
    // );
    return spectral_zucconi(l);
}

fn refr(
    incoming: vec3f, 
    norm: vec3f, 
    index: f32, 
    tir: ptr<function, bool>
) -> vec3f {

    var i = incoming;
    var n = norm;
    var mu = index;
    *tir = false;

    if (dot(i, n) < 0){
        mu = 1/mu;
        n = -n;
    } else {
        if (dot(i,n) < cos(asin(1/mu))){
            *tir = true;
            return reflect(i, n);
        }
    }

    var t = refract(i, -n, mu);
    // (
    //     n*sqrt(1-mu*mu*(1 - dot(n, i)*dot(n, i)))
    //     + mu * (i - dot(n,i)*n)
    // );
    // https://physics.stackexchange.com/questions/435512/snells-law-in-vector-form

    return t;
}

const k = 1103515245u;
fn hash(y: uint3 ) -> vec3f {
    var x = y;
    x = (uint3(x.x>>8u, x.y>>8u, x.z>>8u)^x.yzx)*k;
    x = (uint3(x.x>>8u, x.y>>8u, x.z>>8u)^x.yzx)*k;
    x = (uint3(x.x>>8u, x.y>>8u, x.z>>8u)^x.yzx)*k;
    
    return vec3f(x)*(1.0/float(0xffffffffu));
} //hash by IQ https://www.shadertoy.com/view/XlXcW4. Ported to wgsl by ahs3n.

fn df(p: vec3f) -> float {
    return 
    length(p-vec3f(0,0,1.))-.6 + sin(p.x*30.)*.01
    // max(abs(p.x), max(abs(p.y), abs(p.z-.7)))-.3
    //max(length(p.xy)-.2, -p.z-.1)// + sin(p.z*50.)*.002
    // max(
    //     max(-p.z + .2, abs(p.x)-.9), 
    //     max(abs(p.y)-.9, p.z-.5 - sin((p.x + p.y)*20.)*.004)
    // )
    // max(abs(p.x)-.5, max(abs(p.y)-.3, abs(p.z-.7) -.2- p.x*.3))
    ;
}

fn baseDF(p: vec3f) -> float {
    return max(p.z, max(abs(p.x), abs(p.y))-1);
}

fn dfSum(p: vec3f) -> float {
    return min(df(p), baseDF(p));
}

fn normal(p: vec3f) -> vec3f {
    let d = dfSum(p);
    return normalize(
        vec3f(
            d - dfSum(p - vec3f(eps,0,0)),
            d - dfSum(p - vec3f(0,eps,0)),
            d - dfSum(p - vec3f(0,0,eps))
        )
    );
}

fn ray(o: vec3f, dir: vec3f, negative: bool) -> vec4f { 
    /*
    vec4f(
        depth,
        hit?1:0,
        ground?1:0,
        [empty]
    )
    */
    var p = o; 
    var t = 0.; 
    var d = 0.;
    var ground = false;
    var hit = false;

    for (var i = 0; i < 1<<7; i++){
        p = o+dir*t;
        d = min(
            df(p) * select(1.,-1.,negative),
            baseDF(p)
        );

        t += d;
    }

    if (d < eps){
        hit = true;
    }

    if (baseDF(p) < df(p) * select(1.,-1.,negative) && p.z < eps && hit){
        ground = true;
    }

    return vec4f(
        t,
        select(0.,1.,hit),
        select(0.,1.,ground),
        0
    );
}

fn path(
    o: vec3f, 
    dir: vec3f, 
    mu: f32,
    rayDirOut: ptr<function, vec3f>, 
    rayOutPos: ptr<function, vec3f>    
) -> vec4f {

    var hit = ray(o, dir, false);

    /*
    vec4f(
        depth,
        hit?1:0,
        ground?1:0,
        [empty]
    )
    */

    var p = o + hit.x * dir;
    var rd = dir;

    if (hit.y > .5 && hit.z < .5){
        var norm = normal(p);
        p -= norm*eps*3;
        var tir: bool;
        rd = refr(dir, norm, mu, &tir);
        var tHit: vec4f;
        var inside = true;

        for (var i = 0; i < 16; i++){

            // if (i == 2){
            //     return vec4f(p,1);
            // }

            tHit = ray(p, rd, inside);

            p += rd*tHit.x;

            if (tHit.y < .5 || tHit.z > .5){ // no hit (sky) or ground
                break;
            }


            norm = normal(p);
            p += norm*eps*3;

            let prd = rd;
            rd = refr(rd, norm, ior, &tir);

            let a = dot(norm, prd)>0;
            let b = dot(norm, rd )>0;

            if (!tir){
                inside = !inside;
            }
        }

        hit = tHit;
    } 

    *rayDirOut = rd;
    *rayOutPos = p;

    return hit;
}

@compute @workgroup_size(16, 16)
fn trace(@builtin(global_invocation_id) id: vec3u) {
    let h = hash(uint3(id.xy, time.frame));
    var o = vec3f(
        h.xy*2.-1.,
        0
    );
    //o.x = 0.;
    var dir = normalize(lightDir);
    o -= dir*20.;

    var rd: vec3f;
    var p: vec3f;

    let wavelength = 350+h.z*400;

    var hit = path(o, dir, ior / wavelength * 500., &rd, &p);

    if (hit.z > .5){
        stuff
            [max(0, min(int(p.x*res/2)+res/2, res))]
            [max(0, min(int(p.y*res/2)+res/2, res))] 
            += spectrum(wavelength);
    }
}

@compute @workgroup_size(16, 16)
fn img(@builtin(global_invocation_id) id: vec3u) {
    let r = vec2f(textureDimensions(screen));
    if (f32(id.x) >= r.x || f32(id.y) >= r.y) { return; }
    let U = vec2f(f32(id.x) + .5, r.y - f32(id.y) - .5);
    let uv = U / vec2f(r);
    let cuv = (2.*U-vec2f(r)) / float(r.y);
    let muv = select(
        vec2(pi,pi/2.)*
        (2.*vec2f(mouse.pos)-vec2f(r)) / vec2f(r).y,
        vec2f(time.elapsed, cos(time.elapsed*.2)*.5+.5),
        f32(mouse.pos.x) == f32(r.x)/2. && f32(mouse.pos.y) == f32(r.y)/2.
    );

    var o = vec3f(0);
    let camf = vec3f(sin(muv.x)*cos(muv.y), cos(muv.x)*cos(muv.y), sin(-muv.y));
    let camr = normalize(cross(camf, vec3(0,0,1)));
    let camu = cross(camr, camf);
    let camUV = cuv*.3;
    var dir = vec3f(sin(camUV.x)*cos(camUV.y), cos(camUV.x)*cos(camUV.y), sin(camUV.y));
    dir = camf * dir.y + camr * dir.x + camu * dir.z;

    o = vec3(0,0,.2)-camf*3.;

    var col = vec3f(0);
    var rd: vec3f;
    var op: vec3f;
    var hit = path(o, dir, ior, &rd, &op);
    
    if (hit.y < .5) { // Sky

        // col = vec3f(.005,.007,.009);

        col = exp(rd.z*2.)*bg;

        // col = vec3f(
        //     exp(-rd.z*2.), 
        //     exp(-rd.z), 
        //     exp(rd.z)
        // ) * (rd.z*.5+.5) * (sin(5.*atan2(rd.y, rd.x))*.5+.5);

    } else if (hit.z > .5){ // Ground
        
        op.x = max(-1. + 1./res, min(1. - 1./res, op.x));
        op.y = max(-1. + 1./res, min(1. - 1./res, op.y));

        col = vec3f(
            (stuff
                [max(0, min(int(op.x*res/2)+res/2, res))]
                [max(0, min(int(op.y*res/2)+res/2, res))])
        ) / f32(time.frame) * (res*res / (r.x*r.y));


        col = mix(exp(rd.z*2.)*bg, col, exp(op.z*3.));

        col *= vec3f(.533,.533,1);


        col /= col + 1.;

    } else {
        
        //col = vec3f(1,0,1); // Failed to terminate
        
    }
    //col = hit.xyz;

    // vignette 
    // col = vec3f(1);
    col = mix(bg, col, 1./(1.+5.*dot(uv-.5, uv-.5)));

    // col = spectrum(350 + uv.x*400);

    textureStore(screen, id.xy, vec4f(col, 1.));
}