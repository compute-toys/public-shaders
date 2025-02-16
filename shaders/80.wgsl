// Number of bodies in the simulation
//#define N 16384
//#define N_64 256 // N/64

#define N 16384
#define N_64 256 // N/64
#define N_I 4

//#define dt 0.0166666666666666666667
#define dt 0.01

struct SimulationData {
    m: array<f32, N>,   // body masses
    r: array<vec2<f32>, N>, // body positions
    v: array<vec2<f32>, N>, // body velocities
    a: array<vec2<f32>, N>, // body accelerations
}

const TWO_PI = 6.2831853071795864769252867665590057683943387987502116419498891846;

#storage simulationData SimulationData

#storage atomic_storage array<atomic<i32>>

// Highly useful references:
// https://www.w3.org/TR/WGSL/#atomic-type
// michael0884's 3D Atomic rasterizer https://compute.toys/view/21

// https://www.jcgt.org/published/0009/03/02/
fn pcg4d(a: uint4) -> uint4{
    var v = a * 0x0019660Du + 0x3C6EF35Fu;
    v.x += v.y*v.w; v.y += v.z*v.x; v.z += v.x*v.y; v.w += v.y*v.z;
    v ^= (v >> uint4(16u));
    v.x += v.y*v.w; v.y += v.z*v.x; v.z += v.x*v.y; v.w += v.y*v.z;
    return v;
}

fn urand4(rng_state: ptr<function, vec4u>) -> vec4<f32> { 
    *rng_state = pcg4d(*rng_state);

    return vec4<f32>(*rng_state)/f32(0xFFFFFFFFu); 
}

fn udir2(rng_state: ptr<function, vec4u>) -> vec2f {
    *rng_state = pcg4d(*rng_state);

    var z = f32((*rng_state).x)/f32(0xFFFFFFFFu);
    var t = TWO_PI * z;
    var c = cos(t); var s = sin(t);
    return vec2f(c, s);
}

fn nrand4(sigma: float, mean: float4, rng_state: ptr<function, vec4u>) -> vec4<f32> {
    let Z = urand4(rng_state);
    return mean + sigma * sqrt(-2.0 * log(Z.xxyy)) * 
           float4(cos(TWO_PI * Z.z),sin(TWO_PI * Z.z),cos(TWO_PI * Z.w),sin(TWO_PI * Z.w));
}

#workgroup_count init 256 1 1
@compute @workgroup_size(64)
fn init(@builtin(global_invocation_id) id: vec3<u32>) {
    if(time.frame > 0) { return; } // only initialize on frame 0

    var rng_state: vec4<u32> = vec4<u32>(id.x + 256u, id.x + 3456342u, id.x + 11234u, 1345u);

    simulationData.m[id.x] = 15.0 / f32(N);
    simulationData.r[id.x] = udir2(&rng_state);
    simulationData.r[id.x] *= 0.5 * pow(urand4(&rng_state).x, 2.0);
    //simulationData.r[id.x] = nrand4(0.2, vec4f(0.0, 0.0, 0.0, 0.0), &rng_state).yz;
    simulationData.v[id.x] = nrand4(0.0, vec4f(0.0, 0.0, 0.0, 0.0), &rng_state).yz;
    simulationData.v[id.x] += 0.25 * normalize(vec2f(simulationData.r[id.x].y, -simulationData.r[id.x].x)) * pow(length(simulationData.r[id.x]), 0.1);
    simulationData.v[id.x] += -1.0 * normalize(simulationData.r[id.x]) * pow(length(simulationData.r[id.x]), 3.0);
    simulationData.a[id.x] = vec2f(0.0, 0.0);
}

#workgroup_count kick0 N_64 1 1
@compute @workgroup_size(64)
fn kick0(@builtin(global_invocation_id) id: vec3<u32>) {
    simulationData.v[id.x] += 0.5 * dt * simulationData.a[id.x];
}

#workgroup_count drift N_64 1 1
@compute @workgroup_size(64)
fn drift(@builtin(global_invocation_id) id: vec3<u32>) {
    simulationData.r[id.x] += dt * simulationData.v[id.x];
}

#workgroup_count update_acceleration N_64 1 1
@compute @workgroup_size(64)
fn update_acceleration(@builtin(global_invocation_id) id: vec3<u32>) {
    let m = simulationData.m[id.x];
    let r = simulationData.r[id.x];

    simulationData.a[id.x] = vec2f(0.0, 0.0);

    const G = 1.;

    for(var i = 0; i < N/N_I; i++) {
        if(i == i32(id.x)) { continue; }

        let _m = simulationData.m[i];
        let _r = simulationData.r[i];

        let diff = _r - r;

        let r2 = (diff.x*diff.x)+(diff.y*diff.y);

        simulationData.a[id.x] += normalize(diff)*float(N_I)*G*m*_m/(r2+0.001);
    }

    simulationData.a[id.x] += normalize(-r)*G*m*0.01/(dot(-r,-r)+0.001); // central black hole
}

#workgroup_count kick1 N_64 1 1
@compute @workgroup_size(64)
fn kick1(@builtin(global_invocation_id) id: vec3<u32>) {
    simulationData.v[id.x] += 0.5 * dt * simulationData.a[id.x];
}

@compute @workgroup_size(16, 16)
fn clear(@builtin(global_invocation_id) id: vec3<u32>) {
    let screen_size = int2(textureDimensions(screen));
    let idx0 = int(id.x) + int(screen_size.x * int(id.y));

    atomicStore(&atomic_storage[idx0*4+0], 0);
    atomicStore(&atomic_storage[idx0*4+1], 0);
    atomicStore(&atomic_storage[idx0*4+2], 0);
    atomicStore(&atomic_storage[idx0*4+3], 0);
}

#define R 1

#workgroup_count draw N_64 1 1
@compute @workgroup_size(64)
fn draw(@builtin(global_invocation_id) id: vec3<u32>) {
    let screen_size = uint2(textureDimensions(screen));

    //vec2 uv = (fragCoord - 0.5 * iResolution.xy) / min(iResolution.x, iResolution.y);

    var x: vec2f = simulationData.r[id.x];
    //x += (vec2f(mouse.pos.xy) - 0.5 * vec2f(f32(screen_size.x), f32(screen_size.y)))/vec2f(f32(screen_size.x), f32(screen_size.y));
    x -= 2.0 * vec2f(custom.offset_x - 0.5, -custom.offset_y + 0.5);
    //x *= (5.0 * exp(10.0 * custom.scale - 1.0)) + 0.1;
    x *= (5.0 * custom.scale) + 0.1;
    x *= min(f32(screen_size.x), f32(screen_size.y));
    x += 0.5 * vec2<f32>(f32(screen_size.x), f32(screen_size.y));

    /*
    if(i32(x.x) >= 0 && i32(x.x) < i32(screen_size.x) && i32(x.y) >= 0 && i32(x.y) < i32(screen_size.y)) {
        let index: i32 = i32(x.x) + (i32(screen_size.x) * i32(x.y));

        atomicAdd(&atomic_storage[4*index+0], i32(1));
        atomicAdd(&atomic_storage[4*index+1], i32(1));
        atomicAdd(&atomic_storage[4*index+2], i32(1));
        atomicAdd(&atomic_storage[4*index+3], i32(1));
    }
    */
    for(var i: i32 = -1; i <= 1; i++) {
        for(var j: i32 = -1; j <= 1; j++) {
            let coord_x: i32 = i32(x.x)+i;
            let coord_y: i32 = i32(x.y)+j;
            if(coord_x >= 0 && coord_x < i32(screen_size.x) && coord_y >= 0 && coord_y < i32(screen_size.y)) {
                let index: i32 = coord_x + (i32(screen_size.x) * coord_y);
                var accum: i32 = 1;
                if(i == 0) {accum = 2;}
                if(j == 0) {accum = 2;}
                if(i == 0 && j == 0) {accum = 3;}
                atomicAdd(&atomic_storage[4*index+0], accum);
                atomicAdd(&atomic_storage[4*index+1], accum);
                atomicAdd(&atomic_storage[4*index+2], accum);
                atomicAdd(&atomic_storage[4*index+3], accum);
            }
        }
    }
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    // Viewport resolution (in pixels)
    let screen_size = uint2(textureDimensions(screen));

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = float2(float(id.x) + .5, float(screen_size.y - id.y) - .5);

    // Normalised pixel coordinates (from 0 to 1)
    let uv = fragCoord / float2(screen_size);

    let r = atomicLoad(&atomic_storage[4*(id.x+(screen_size.x * id.y))+0]);
    let g = atomicLoad(&atomic_storage[4*(id.x+(screen_size.x * id.y))+1]);
    let b = atomicLoad(&atomic_storage[4*(id.x+(screen_size.x * id.y))+2]);
    //let a = atomicLoad(&atomic_storage[4*(id.x+(screen_size.x * id.y))+3]);

    // Time varying pixel colour
    var col = 0.0625*vec3f(f32(r), f32(g), f32(b));

    // Convert from gamma-encoded to linear colour space
    col = pow(col, float3(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, int2(id.xy), float4(col, 1.));
}
