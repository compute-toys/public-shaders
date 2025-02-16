
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let t = time.elapsed * 0.3;
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    //let fragCoord = vec2f(f32(id.x) +.5, f32(screen_size.y - id.y) - .5);
    //let fragCoord = vec2f(f32(id.x), f32(id.y));

    // Normalised pixel coordinates (from 0 to 1)
    //let uv = fragCoord / vec2f(screen_size);
    //let uv = vec2f(1) - vec2f(f32(id.x ) , f32(id.y) ) / vec2f(f32(screen_size.x), f32(screen_size.y) );

        
    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);
    var uv = fragCoord / vec2f(screen_size) ;


    //let uv_pixel = floor(uv * (f32(screen_size.x) / 4.)) / (f32(screen_size.y)/ 4.);
    let uv_pixel = floor(uv * (vec2f(screen_size) / 4.)) / (vec2f(screen_size)/ 4.);

    let col1 = vec4f(0.510, 0.776, 0.486, 1.0);
    let col2 = vec4f(0.200, 0.604, 0.318, 1.0);
    let col3 = vec4f(0.145, 0.490, 0.278, 1.0);
    let col4 = vec4f(0.059, 0.255, 0.251, 1.0);

/*
    var f = simplex3d_fractal(vec3f(uv_pixel.x, (uv_pixel.y + t ) * 0.05, 0.)*custom.octave * 10.0 + custom.octave * 10.0);
    f = 0.5 + 0.5*f;
    f *= smoothstep( 0.0, 0.005, abs(uv.x - 0.6) );

    f = pow(f, 2.2);
    
    var displace = vec3f(f,f,f);
*/
    var displace = textureSampleLevel(channel0, nearest, vec2(1. - (uv_pixel.y +  mix(.3, 1.4, fract( t ))) * 0.05, uv_pixel.x  ) , 0).xyz;
    //var displace = textureSampleLevel(channel0, nearest, vec2(uv_pixel.x , 1. - (uv_pixel.y +  mix(.3, 1.4, fract( t ))) * 0.05 ) , 0).xyz;

    displace *= 0.5;
    displace.x -= 1.;
    displace.y -= 1.0;
    displace.y *= 0.5;

    var uv_temp = uv_pixel;
    uv_temp.y *= 0.2;
    uv_temp.y += mix(.3, 1.4, fract( t ));

/*
    var f_1 = simplex3d_fractal(vec3f((uv_temp + displace.xy), 1.)*custom.octave * 10.0 + custom.octave * 10.0);
    
    f_1 = 0.5 + 0.5*f_1;
    f_1 *= smoothstep( 0.0, 0.005, abs(uv_temp.x - 0.6) );

    f_1 = pow(f_1, 2.2);
    var color = vec4f(f_1, f_1, f_1, 1.);
*/

    var color = textureSampleLevel(channel0, nearest, (uv_temp + displace.xy), 0);

    let noise = floor(color * 10.0) / 5.0;
    let dark = mix(col1, col2, uv.y);
    let bright = mix(col3, col4, uv.y);
    color = mix(dark, bright, noise);

    let inv_uv = 1.0 - uv_pixel.y;
    color = vec4f(color.rgb - 0.45 * pow(uv_pixel.y, 8.), color.a);
    color = vec4f(color.rgb, color.a - 0.2 * pow(uv_pixel.y, 8.));
    color += pow(inv_uv, 8.);


    color = vec4f(pow(color.xyz, vec3f(2.2)), 1.);

    color.a -= 0.2;

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, color);
}

//======================
// simplex noise (not used, but could be; instead of texture)

fn hash(p: vec2f) -> vec2f // replace this by something better
{
    let p2 = vec2f( dot(p,vec2(127.1,311.7)), dot(p,vec2f(269.5,183.3)) );
    return -1.0 + 2.0*fract(sin(p2)*43758.5453123);
}

fn simplex2d(p: vec2f) -> f32
{
    let K1 = 0.366025404; // (sqrt(3)-1)/2;
    let K2 = 0.211324865; // (3-sqrt(3))/6;
    let i = floor( p + (p.x+p.y)*K1 );
    let a = p - i + (i.x+i.y)*K2;
    let o = step(a.yx,a.xy);
    let b = a - o + K2;
    let c = a - 1.0 + 2.0*K2;
    let h = max( 0.5-vec3f(dot(a,a), dot(b,b), dot(c,c) ), vec3f(0.) );
    let n = h*h*h*h*vec3f( dot(a,hash(i+0.0)), dot(b,hash(i+o)), dot(c,hash(i+1.0)));
    return dot( n, vec3f(70.0) );
}


// Simplex Noise 3D: https://www.shadertoy.com/view/XsX3zB

/* discontinuous pseudorandom uniformly distributed in [-0.5, +0.5]^3 */
fn random3(c: vec3f) -> vec3f
{
    var j = 4096.0*sin(dot(c,vec3f(17.0, 59.4, 15.0)));
    var r = vec3f(0.);
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
fn simplex3d(p: vec3f) -> f32
{
    /* 1. find current tetrahedron T and it's four vertices */
    /* s, s+i1, s+i2, s+1.0 - absolute skewed (integer) coordinates of T vertices */
    /* x, x1, x2, x3 - unskewed coordinates of p relative to each of T vertices*/

    /* calculate s and x */
    let s = floor(p + dot(p, vec3f(F3)));
    let x = p - s + dot(s, vec3f(G3));

    /* calculate i1 and i2 */
    let e = step(vec3f(0.0), x - x.yzx);
    let i1 = e*(1.0 - e.zxy);
    let i2 = 1.0 - e.zxy*(1.0 - e);

    /* x1, x2, x3 */
    let x1 = x - i1 + G3;
    let x2 = x - i2 + 2.0*G3;
    let x3 = x - 1.0 + 3.0*G3;

    /* 2. find four surflets and store them in d */
    var w = vec4(0.);
    var d = vec4(0.);

    /* calculate surflet weights */
    w.x = dot(x, x);
    w.y = dot(x1, x1);
    w.z = dot(x2, x2);
    w.w = dot(x3, x3);

    /* w fades from 0.6 at the center of the surflet to 0.0 at the margin */
    w = max(0.6 - w, vec4(0.0));

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
fn simplex3d_fractal(m: vec3f) -> f32
{
    return   0.5333333*simplex3d(m*rot1)
            +0.2666667*simplex3d(2.0*m*rot2)
            +0.1333333*simplex3d(4.0*m*rot3)
            +0.0666667*simplex3d(8.0*m);
}