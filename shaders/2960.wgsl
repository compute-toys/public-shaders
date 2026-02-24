@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);

    // Normalised pixel coordinates (from 0 to 1)
    let uv = (fragCoord / vec2f(screen_size.xx)) * vec2f(1, 0.5) + vec2f(0, 0.275);


    // Time varying pixel colour
    var col = (
        fire(uv.xy, simplex2d(vec2f(time.elapsed/4, 0)))

    );

    // Convert from gamma-encoded to linear colour space
    // col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn fire(uv: vec2f, a: f32) -> vec3f {
    let screen_size = textureDimensions(screen);
    return colorRamp((cubicInOut(cubicInOut(select(remap(distance(uv.xy, vec2f(0.5, (f32(screen_size.y)/f32(screen_size.x))/2)), 0.125, 0.3, 1, 0), 1.0, distance(uv.xy, vec2f(0.5, (f32(screen_size.y)/f32(screen_size.x))/2))<0.125)*remap(pattern(vec3f((uv.xy + vec2f(a, 0))*5 + vec2f(0, -time.elapsed*2), time.elapsed*0.75)), -1, 1, 0, 1)))));
}


fn cubicInOut(t: float) -> float {
  return select(0.5 * pow(2.0 * t - 2.0, 3.0) + 1.0, 4.0 * t * t * t, t < 0.5);
}


fn remap(value: f32, in_min: f32, in_max: f32, out_min: f32, out_max: f32) -> f32 {
    return ((value - in_min) / (in_max - in_min)) * (out_max - out_min) + out_min;
}

fn colorRamp(t: f32) -> vec3<f32> {
    let clamped_t = clamp(t, 0.0, 1.0);

    // Define color stops and positions (e.g., black at 0.0, red at 0.5, white at 1.0)
    let stops = array<f32, 3>(0.0, 0.5, 1.0);
    let colors = array<vec4<f32>, 4>(
        vec4<f32>(0.0, 0.0, 0.0, 1.0), // Black
        vec4<f32>(247.0/255.0, 25.0/255.0, 37.0/255.0, 1.0),
        vec4<f32>(247.0/255.0, 170.0/255.0, 37.0/255.0, 1.0), // Red
        vec4<f32>(1.0, 1.0, 1.0, 1.0)  // White
    );

    for (var i = 0u; i < 3u; i = i + 1u) {
        if (clamped_t >= stops[i] && clamped_t <= stops[i+1]) {
            // Calculate interpolation factor (0.0 to 1.0) for the current segment
            let segment_start = stops[i];
            let segment_end = stops[i+1];
            let segment_t = (clamped_t - segment_start) / (segment_end - segment_start);
            // Interpolate between the two adjacent colors
            return mix(colors[i], colors[i+1], segment_t).xyz;
        }
    }
    // Fallback for t=1.0 edge case due to loop condition
    return colors[2].xyz;
}

// Simplex Noise (http://en.wikipedia.org/wiki/Simplex_noise), a type of gradient noise
// that uses N+1 vertices for random gradient interpolation instead of 2^N as in regular
// latice based Gradient Noise.

// Simplex Noise 2D: https://www.shadertoy.com/view/Msf3WH

fn hash(p: float2) -> float2 // replace this by something better
{
    let p2 = float2( dot(p,float2(127.1,311.7)), dot(p,float2(269.5,183.3)) );
    return -1.0 + 2.0*fract(sin(p2)*43758.5453123);
}

fn simplex2d(p: float2) -> float
{
    let K1 = 0.366025404; // (sqrt(3)-1)/2;
    let K2 = 0.211324865; // (3-sqrt(3))/6;
    let i = floor( p + (p.x+p.y)*K1 );
    let a = p - i + (i.x+i.y)*K2;
    let o = step(a.yx,a.xy);
    let b = a - o + K2;
    let c = a - 1.0 + 2.0*K2;
    let h = max( 0.5-float3(dot(a,a), dot(b,b), dot(c,c) ), float3(0.) );
    let n = h*h*h*h*float3( dot(a,hash(i+0.0)), dot(b,hash(i+o)), dot(c,hash(i+1.0)));
    return dot( n, float3(70.0) );
}


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
const rot2 = mat3x3<f32>(-0.55,-0.39, 0.74, 0.33,-0.91,-0.24,0.77, 0.12,0.63);
const rot1 = mat3x3<f32>(-0.37, 0.36, 0.85,-0.14,-0.93, 0.34,0.92, 0.01,0.4);
const rot3 = mat3x3<f32>(-0.71, 0.52,-0.47,-0.08,-0.72,-0.68,-0.7,-0.45,0.56);

/* directional artifacts can be reduced by rotating each octave */
fn simplex3d_fractal(m: float3) -> float
{
    return   0.5333333*simplex3d(m*rot1)
            +0.2666667*simplex3d(2.0*m*rot2)
            +0.1333333*simplex3d(4.0*m*rot3)
            +0.0666667*simplex3d(8.0*m);
}

fn pattern( v: vec3f ) -> float
{
    let p = v.xy;
    let q = vec2f( simplex3d_fractal( vec3f(p + vec2(0.0,0.0), v.z) ),
                   simplex3d_fractal( vec3f(p + vec2(5.2,1.3), v.z ) ) );

    return simplex3d( vec3f(p + 4.0*q,v.z) );
}