#storage atomic_storage array<atomic<i32>>

// Highly useful references:
// https://www.w3.org/TR/WGSL/#atomic-type
// michael0884's 3D Atomic rasterizer https://compute.toys/view/21

// Precision is equal to 1/atomic_scale
#define atomic_scale 256.0

fn atomic_encode(x: f32) -> i32 {
    return i32(x * atomic_scale);
}

fn atomic_decode(x: i32) -> f32 {
    return f32(x) / atomic_scale;
}

const buddhabulb_iterations = 10u;

// http://www.mimirgames.com/articles/programming/digits-of-pi-needed-for-floating-point-numbers/
const     PI = 3.141592653589793; // Pi
const TWO_PI = 6.283185307179586; // 2 * Pi
const INV_PI = 0.318309886183790; // 1 / Pi

var<private> rng_state : uint4;

// http://www.pcg-random.org/
fn pcg4d(a: uint4) -> uint4
{
    var v = a * 0x0019660Du + 0x3C6EF35Fu;
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    v = v ^ (v >> uint4(16u)); // vec4<uint> >> uint isn't implemented yet
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;

    return v;
}

fn rand() -> float
{
    rng_state = pcg4d(rng_state);

    let divisor = 1. / float(0xFFFFFFFFu);

    return float(rng_state.x) * divisor;
}

fn rand2() -> float2
{
    rng_state = pcg4d(rng_state);

    let divisor = 1. / float(0xFFFFFFFFu);

    return float2(rng_state.xy) * divisor;
}

fn rand3() -> float3
{
    rng_state = pcg4d(rng_state);

    let divisor = 1. / float(0xFFFFFFFFu);

    return float3(rng_state.xyz) * divisor;
}

fn rand4() -> float4
{
    rng_state = pcg4d(rng_state);

    let divisor = 1. / float(0xFFFFFFFFu);

    return float4(rng_state) * divisor;
}

fn palette(t: float, a: float3, b: float3, c: float3, d: float3) -> float3
{
    return a + b * cos(TWO_PI * (c * t + d));
}

fn draw_point(pos: float3, col: float3) {
    let screen_size = uint2(textureDimensions(screen));

    var p2d = pos.xy;

    //let fragCoord = float2(float(id.x) + .5, float(screen_size.y - id.y) - .5);
    //let uv = (fragCoord - 0.5 * screen_size) / float2(screen_size);
    p2d *= 4.*custom.zoom + .5;
    p2d -= 4.*float2((custom.x_offset- .5), -(custom.y_offset - .5));

    p2d.y *= float(screen_size.x) / float(screen_size.y);
	p2d = ( (.5 * p2d) + .5 ) * float2(screen_size);

	//let coord = int2(p2d);
    let coord = p2d;

    //if(coord.x >= 0 && coord.x < int(screen_size.x) && coord.y >= 0 && coord.y < int(screen_size.y)) {
    if(coord.x >= 0. && coord.x < float(screen_size.x) && coord.y >= 0. && coord.y < float(screen_size.y)) {
        //assert(0, coord.x == 0);
        let index = uint(coord.x) + (screen_size.x * uint(coord.y));

        atomicAdd(&atomic_storage[4u * index + 0u], atomic_encode(col.r));
        atomicAdd(&atomic_storage[4u * index + 1u], atomic_encode(col.g));
        atomicAdd(&atomic_storage[4u * index + 2u], atomic_encode(col.b));
        atomicAdd(&atomic_storage[4u * index + 3u], 1);
    }
}

@compute @workgroup_size(16, 16)
fn clear(@builtin(global_invocation_id) id: uint3) {
    let screen_size = uint2(textureDimensions(screen));

    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let index = id.x + screen_size.x * id.y;

    atomicStore(&atomic_storage[4u * index + 0u], 0);
    atomicStore(&atomic_storage[4u * index + 1u], 0);
    atomicStore(&atomic_storage[4u * index + 2u], 0);
    atomicStore(&atomic_storage[4u * index + 3u], 0);
}

@compute @workgroup_size(16, 16)
fn draw(@builtin(global_invocation_id) id: uint3) {
    let screen_size = uint2(textureDimensions(screen));

    rng_state = uint4(id.xyz + 1u, time.frame + 1u);

    for (var j: uint = 0u; j < 15u; j++) {
    let c = 4. * (rand3() - 0.5);

    var orbit: array<float3, buddhabulb_iterations>;

    // https://iquilezles.org/articles/mandelbulb/

    var w = c;
    var m = dot(w, w);

    var trap = abs(w);

    var dz = 1.;

    var orbit_escaped = false;
    var orbit_length = 0u;

    for (var i: uint = 0u; i < buddhabulb_iterations; i++) {
        // polynomial version (no trigonometrics, but MUCH slower)

        orbit[i] = w;

        if (m > 256.) {
            orbit_escaped = true;
            orbit_length = i;

            break;
        }

        let m2 = m  * m ;
        let m4 = m2 * m2;
        dz = 8. * sqrt(m4*m2*m)*dz + 1.;

        let x = w.x; let x2 = x * x; let x4 = x2 * x2;
        let y = w.y; let y2 = y * y; let y4 = y2 * y2;
        let z = w.z; let z2 = z * z; let z4 = z2 * z2;

        let k3 = x2 + z2;
        let k2 = 1. / sqrt(k3*k3*k3*k3*k3*k3*k3);
        let k1 = x4 + y4 + z4 - 6.*y2*z2 - 6.*x2*y2 + 2.*z2*x2;
        let k4 = x2 - y2 + z2;

        w.x = c.x +  64.*x*y*z*(x2 - z2)*k4*(x4 - 6.*x2*z2 + z4)*k1*k2;
        w.y = c.y + -16.*y2*k3*k4*k4 + k1*k1;
        w.z = c.z +  -8.*y*k4*(x4*x4 - 28.*x4*x2*z2 + 70.*x4*z4 - 28.*x2*z2*z4 + z4*z4)*k1*k2;

        trap = min(trap, abs(w));

        m = dot(w, w);
    }

    if (orbit_escaped) {
        for (var i: uint = 1u; i < orbit_length; i++) {
            //draw_point(orbit[i], float3(.1*float(orbit_length)));
            draw_point(orbit[i], palette(float(orbit_length)*.3, float3(0.5, 0.5, 0.5), float3(0.5, 0.5, 0.5), float3(1.0, 1.0, 1.0), float3(0.00, 0.33, 0.67)));
            //draw_point(orbit[i].pos, trap);
        }
    }
    }
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    // Viewport resolution (in pixels)
    let screen_size = uint2(textureDimensions(screen));

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)

    // Time varying pixel colour
    //var col = float3(0); // As of May 31, 2022 on Google Chrome Canary, this is invalid
    var col = float3(0.); // But for some reason this isn't? WGSL moment

    var fragColor : float4;

    let index = id.x + screen_size.x * id.y;

    fragColor.r = atomic_decode(atomicLoad(&atomic_storage[4u * index + 0u]));
    fragColor.g = atomic_decode(atomicLoad(&atomic_storage[4u * index + 1u]));
    fragColor.b = atomic_decode(atomicLoad(&atomic_storage[4u * index + 2u]));
    fragColor.a = f32(atomicLoad(&atomic_storage[4u * index + 3u]));

    // Convert from gamma-encoded to linear colour space
    //col = pow(col, float3(2.2));

    if (custom.reset < .5 && time.frame > 0) {
        let texel = textureLoad(pass_in, int2(id.xy), 0, 0);

        fragColor = float4(fragColor.rgb, 1.) + texel;

        textureStore(pass_out, int2(id.xy), 0, fragColor);
    }
    else {
        textureStore(pass_out, int2(id.xy), 0, float4(0.));
    }

    if (fragColor.a != 0.) {
        col = fragColor.rgb / fragColor.a;
    }
    else {
        col = fragColor.rgb;
    }

    //textureStore(pass_out, int2(id.xy), 0, float4(col, fragColor.a));

    // Output to screen (linear colour space)
    textureStore(screen, int2(id.xy), float4((10.*custom.exposure * col)-(.5*custom.contrast), 1.));
}
