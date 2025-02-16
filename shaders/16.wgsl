/*
MIT License

Copyright (c) 2013 Inigo Quilez <https://iquilezles.org/>
Copyright (c) 2013 Nikita Miropolskiy
Copyright (c) 2022 David A Roberts <https://davidar.io/>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/


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
let F3 = 0.3333333;
let G3 = 0.1666667;

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
let rot1 = mat3x3<f32>(-0.37, 0.36, 0.85,-0.14,-0.93, 0.34,0.92, 0.01,0.4);
let rot2 = mat3x3<f32>(-0.55,-0.39, 0.74, 0.33,-0.91,-0.24,0.77, 0.12,0.63);
let rot3 = mat3x3<f32>(-0.71, 0.52,-0.47,-0.08,-0.72,-0.68,-0.7,-0.45,0.56);

/* directional artifacts can be reduced by rotating each octave */
fn simplex3d_fractal(m: float3) -> float
{
    return   0.5333333*simplex3d(m*rot1)
            +0.2666667*simplex3d(2.0*m*rot2)
            +0.1333333*simplex3d(4.0*m*rot3)
            +0.0666667*simplex3d(8.0*m);
}


@stage(compute) @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    let screen_size = uint2(textureDimensions(screen));
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    let fragCoord = float2(id.xy) + .5;
    let resolution = float2(screen_size);
    let p = fragCoord / resolution.x;
    let p3 = float3(p, time.elapsed*0.025);

    var uv = p * float2(resolution.x / resolution.y, 1.) + time.elapsed * .25;
    var f = 0.;
    if (p.x < .6) { // left: value noise
        //f = simplex2d( 16.0*uv );
        f = simplex3d(p3*32.0);
    } else { // right: fractal noise (4 octaves)
        //uv *= 5.0;
        //let m = mat2x2<f32>( 1.6,  1.2, -1.2,  1.6 );
        //f  = 0.5000*simplex2d( uv ); uv = m*uv;
        //f += 0.2500*simplex2d( uv ); uv = m*uv;
        //f += 0.1250*simplex2d( uv ); uv = m*uv;
        //f += 0.0625*simplex2d( uv ); uv = m*uv;
        f = simplex3d_fractal(p3*8.0+8.0);
    }
    f = 0.5 + 0.5*f;
    f *= smoothstep( 0.0, 0.005, abs(p.x - 0.6) );

    f = pow(f, 2.2); // perceptual gradient to linear colour space
    textureStore(screen, int2(id.xy), float4(f, f, f, 1.));
}
