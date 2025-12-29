

// IDEA:
// https://www.youtube.com/shorts/7g5Cj_1XPkM




@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);
    let uv = fragCoord / vec2f(screen_size);
    let f = fragCoord - vec2f(f32(screen_size.x)/2., f32(screen_size.y)/2.);

    var col = vec3f(0);

    //BACKGROUND

    //col = vec3f(f.x+f.y);

    // STARS
    if random(f) < custom.a {
        col = mix(col, vec3f(1), sin(f.x + f.y + time.elapsed));
    }

    // PLANET
    let dis = distance(vec2f(.5,.5), f);
    if dis <= 180. {
        // normalize
        let g = f / 180.;
        
        // to 3d coords
        let p = vec3f(g.x, g.y, sqrt(1.-(g.x * g.x)-(g.y * g.y)));

        // rotation matrix
        let t = time.elapsed * .2;
        let m = mat3x3(
            cos(t),-sin(t),0,
            0,0,1,
            sin(t),cos(t),0,
        );
        let o = (m * p).xyz;

        // lookup noise
        let n = simplex3d(float3(o) * 3.);
        // color layers
        if n <= 0. {
            col = c4;
        } else if n <= .1 {
            col = c7;
        } else if n <= .2 {
            col = c5;
        } else {
            col = c6;
        }
        //col = vec3f(n);


    } else {
        // "ATHMOSPHERE"
        let dis2 = distance(vec2f(.5,.5), f);
        if dis2 <= 210. { 
            col = mix(col, vec3f(.23), .8);
        }

    }

    // CLOUDS (move faster + morph)
    let dis2 = distance(vec2f(.5,.5), f);
    if dis2 <= 195. { 
        let g = f / 195;
        
        let p = vec3f(g.x, g.y, sqrt(1.-(g.x * g.x)-(g.y * g.y)));

        let t = .5 * time.elapsed + 5000.;
        let m = mat3x3(
            cos(t),-sin(t),0,
            0,0,1,
            sin(t),cos(t),0,
        );
        let o = (m * p).xyz;

        let n = simplex3d(time.elapsed/4 + float3(o) * 1.4);
        if n <= -.1 {
            col = mix(col, c8, .8);
        }
    }

        col = pow(col, vec3f(2.2));
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn random (st: vec2f) -> f32 {
    return fract(sin(dot(st.xy,
                         vec2(12.9828,78.231)))
                 * 43758.5453123);
}



// Color Palette
// https://lospec.com/palette-list/ecoaction
const c1 = vec3f(0.03561, 0.01765, 0.02218);
const c2 = vec3f(0.0666, 0.0369, 0.05285);
const c3 = vec3f(0.05448, 0.04666, 0.07422);
const c4 = vec3f(0.07428, 0.09992, 0.13283);
const c5 = vec3f(0.1329, 0.21221, 0.09994);
const c6 = vec3f(0.24617, 0.31393, 0.10455);
const c7 = vec3f(0.6866, 0.6172, 0.42343);
const c8 = vec3f(0.87146, 0.87147, 0.77585);


// Simplex Noise 3D: https://www.shadertoy.com/view/XsX3zB

/* discontinuous pseudorandom uniformly distributed in [-0.5, +0.5]^3 */
fn random3(c: float3) -> float3
{
    var j = 4096.0*sin(dot(c,vec3(17.0, 59.4, 15.0)));
    var r = float3(0.);
    r.z = fract(512.0*j);
    j *= .125;
    r.x = fract(512.0*j);
    j *= .125;
    r.y = fract(512.0*j);
    return r - 0.5;
}

/* skew constants for 3d simplex functions */
const F3 = 0.3333333;
const G3 = 0.1666667;

/* 3d simplex noise */
fn simplex3d(p: float3) -> float
{
    /* 1. find current tetrahedron T and it's four vertices */
    /* s, s+i1, s+i2, s+1.0 - absolute skewed (integer) coordinates of T vertices */
    /* x, x1, x2, x3 - unskewed coordinates of p relative to each of T vertices*/

    /* calculate s and x */
    let s = floor(p + dot(p, vec3(F3)));
    let x = p - s + dot(s, vec3(G3));

    /* calculate i1 and i2 */
    let e = step(vec3(0.0), x - x.yzx);
    let i1 = e*(1.0 - e.zxy);
    let i2 = 1.0 - e.zxy*(1.0 - e);

    /* x1, x2, x3 */
    let x1 = x - i1 + G3;
    let x2 = x - i2 + 2.0*G3;
    let x3 = x - 1.0 + 3.0*G3;

    /* 2. find four surflets and store them in d */
    var w = float4(0.);
    var d = float4(0.);

    /* calculate surflet weights */
    w.x = dot(x, x);
    w.y = dot(x1, x1);
    w.z = dot(x2, x2);
    w.w = dot(x3, x3);

    /* w fades from 0.6 at the center of the surflet to 0.0 at the margin */
    w = max(0.6 - w, float4(0.0));

    /* calculate surflet components */
    d.x = dot(random3(s), x);
    d.y = dot(random3(s + i1), x1);
    d.z = dot(random3(s + i2), x2);
    d.w = dot(random3(s + 1.0), x3);

    /* multiply d by w^4 */
    w *= w;
    w *= w;
    d *= w;

    /* 3. return the sum of the four surflets */
    return dot(d, vec4(52.0));
}

/* const matrices for 3d rotation */
const rot1 = mat3x3<f32>(-0.37, 0.36, 0.85,-0.14,-0.93, 0.34,0.92, 0.01,0.4);
const rot2 = mat3x3<f32>(-0.55,-0.39, 0.74, 0.33,-0.91,-0.24,0.77, 0.12,0.63);
const rot3 = mat3x3<f32>(-0.71, 0.52,-0.47,-0.08,-0.72,-0.68,-0.7,-0.45,0.56);

/* directional artifacts can be reduced by rotating each octave */
fn simplex3d_fractal(m: float3) -> float
{
    return   0.5333333*simplex3d(m*rot1)
            +0.2666667*simplex3d(2.0*m*rot2)
            +0.1333333*simplex3d(4.0*m*rot3)
            +0.0666667*simplex3d(8.0*m);
}