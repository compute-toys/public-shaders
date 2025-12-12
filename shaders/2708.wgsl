// number of bodies (must be a factor of 64)
#define N 1024
//#define N 2048
//#define N 4096
//#define N 8192
//#define N 16384
//#define N 32768
//#define N 65535

// divide force calculations across this many frames
#define N_DIV 4

#define TAU 6.2831853071795864769252867665590057683943387987502116419498891846156328125724179972560696506842341360

struct simulation {
    mas: array<f32, N>,       // mass
    pos: array<vec3<f32>, N>, // position
    vel: array<vec3<f32>, N>, // velocity
    acc: array<vec3<f32>, N>  // acceleration
};

#storage simulation_storage simulation

#storage atomic_storage array<atomic<i32>>

const INV_PCG32_MAX = 1.0 / f32(0xFFFFFFFFu);

// https://www.jcgt.org/published/0009/03/02/
fn pcg4d(s: vec4<u32>) -> vec4<u32> {
    var v = u32(0x0019660Du) * s + u32(0x3C6EF35Fu);

    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;

    v ^= v >> vec4<u32>(16u);

    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;

    return v;
}

// multiply by another sqrt(x) for a random point on a unit disk
fn udir2(x: f32) -> vec2<f32> {
    return vec2<f32>(cos(TAU*x), sin(TAU*x));
}

fn udir3(x: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(0.000, 0.000, 0.000);
}

// https://en.wikipedia.org/wiki/Box–Muller_transform

fn randn2(x: vec2<f32>) -> vec2<f32> {
    return sqrt(-2.0*log(x.x   )) * vec2<f32>(cos(TAU*x.y), sin(TAU*x.y));
}

fn randn3(x: vec4<f32>) -> vec3<f32> {
    return sqrt(-2.0*log(x.xxy )) * vec3<f32>(cos(TAU*x.z), sin(TAU*x.z), cos(TAU*x.w));
}

fn randn4(x: vec4<f32>) -> vec4<f32> {
    return sqrt(-2.0*log(x.xxyy)) * vec4<f32>(cos(TAU*x.z), sin(TAU*x.z), cos(TAU*x.w), sin(TAU*x.w));
}

// https://en.wikipedia.org/wiki/Rotation_matrix
fn rotate2d(v: vec2<f32>, theta: f32) -> vec2<f32> {
    return mat2x2<f32>(cos(theta), sin(theta),-sin(theta), cos(theta)) * v;
}

#dispatch_once simulation_init
#workgroup_count simulation_init N/256 1 1
@compute @workgroup_size(16, 16)
fn simulation_init(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx: u32 = (16u * id.x) + id.y;

    if(u32(N) <= idx) { return; }

    var s = vec4<u32>(idx+1u, 420u, 69u, 1337u);

    s = pcg4d(s);
    s = pcg4d(s); // second round for better statistical quality

    simulation_storage.mas[idx] = 1.000 / f32(N);

    let pos = vec3<f32>(1.000, 0.125, 1.000) * randn3(INV_PCG32_MAX*vec4<f32>(s));

    //simulation_storage.pos[idx] = vec3<f32>(0.0, 0.0, 0.0);
    simulation_storage.pos[idx] = pos;

    s = pcg4d(s);

    //let vel = vec3<f32>(0.000, 0.000, 0.000);
    //let vel = randn3(INV_U32_MAX*vec4<f32>(s));
    //let vel = 0.01 * normalize(vec3<f32>(pos.z, pos.y,-pos.x));
    let vel = 0.010 * vec3<f32>(pos.z, pos.y,-pos.x) + 0.001 * randn3(INV_PCG32_MAX*vec4<f32>(s));

    simulation_storage.vel[idx] = vel;

    simulation_storage.acc[idx] = vec3<f32>(0.0, 0.0, 0.0);
}

@compute @workgroup_size(16, 16)
fn visualization_clear(@builtin(global_invocation_id) id: vec3<u32>) {
    let image_size = textureDimensions(screen);

    if(image_size.x <= id.x || image_size.y <= id.y) { return; }

    atomicStore(&atomic_storage[4u*(image_size.x*id.y+id.x)+0u], 0);
    atomicStore(&atomic_storage[4u*(image_size.x*id.y+id.x)+1u], 0);
    atomicStore(&atomic_storage[4u*(image_size.x*id.y+id.x)+2u], 0);
    atomicStore(&atomic_storage[4u*(image_size.x*id.y+id.x)+3u], 0);
}

#workgroup_count visualization_draw N/256 1 1
@compute @workgroup_size(16, 16)
fn visualization_draw(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx: u32 = (16u * id.x) + id.y;

    if(u32(N) <= idx) { return; }

    var s = vec4<u32>(idx, 42067u, 1337u, time.frame);

    s = pcg4d(s);
    s = pcg4d(s); // second round for better statistical quality

    let image_size = textureDimensions(screen);

    var pos = simulation_storage.pos[idx];

    pos *= exp2(custom.scale);
    pos = vec3<f32>(rotate2d(pos.xy, 0.125*TAU             ), pos.z).xyz;
    pos = vec3<f32>(rotate2d(pos.xz, 0.500*TAU*custom.theta), pos.y).xzy;

    for(var i: u32 = 0u; i < 16u; i++) {
        s = pcg4d(s);

        let offset = 0.750 * randn2(INV_PCG32_MAX*vec2<f32>(s.xy));

        var coord = vec2<i32>(f32(image_size.y)*pos.xy+0.5*vec2<f32>(image_size)-0.5+offset);

        if(0 <= coord.x && coord.x < i32(image_size.x)
        && 0 <= coord.y && coord.y < i32(image_size.y)) {
            let pixel_index = image_size.x*u32(coord.y)+u32(coord.x);

            atomicAdd(&atomic_storage[4u*pixel_index+0u], 1);
            atomicAdd(&atomic_storage[4u*pixel_index+1u], 1);
            atomicAdd(&atomic_storage[4u*pixel_index+2u], 1);
            atomicAdd(&atomic_storage[4u*pixel_index+3u], 1);
        }
    }
}

@compute @workgroup_size(16, 16)
fn visualization_output(@builtin(global_invocation_id) id: vec3<u32>) {
    let image_size = textureDimensions(screen);

    if(image_size.x <= id.x || image_size.y <= id.y) { return; }

    var value = vec4<i32>(0);

    value.r = atomicLoad(&atomic_storage[4u*(image_size.x*id.y+id.x)+0u]);
    value.g = atomicLoad(&atomic_storage[4u*(image_size.x*id.y+id.x)+1u]);
    value.b = atomicLoad(&atomic_storage[4u*(image_size.x*id.y+id.x)+2u]);
    value.a = atomicLoad(&atomic_storage[4u*(image_size.x*id.y+id.x)+3u]);

    var color = vec3<f32>(value.rgb);

    color *= 0.500 * 0.125 * exp2(custom.exposure);

    color = tanh(color);

    color = pow(color, vec3<f32>(2.200));

    textureStore(screen, id.xy, vec4<f32>(color, 1.0));
}

#workgroup_count simulation_force N/256 1 1
@compute @workgroup_size(16, 16)
fn simulation_force(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx: u32 = (16u * id.x) + id.y;

    if(u32(N) <= idx) { return; }

    let i = idx;

    let m_i = simulation_storage.mas[idx];
    let r_i = simulation_storage.pos[idx];

    var f = vec3<f32>(0.0, 0.0, 0.0);

    #ifndef N_DIV
    for(var j = 0u; j < u32(N); j++) {
    #else
    for(var j: u32 = (time.frame % u32(N_DIV)) * (u32(N)/4u); j < (u32(N)/u32(N_DIV))*( (time.frame % u32(N_DIV))+1u ); j++) {
    #endif
        if(i == j) {
            continue;
        }

        let m_j = simulation_storage.mas[j];
        let r_j = simulation_storage.pos[j];

        let r_ij = r_j - r_i;

        let r2 = dot(r_ij, r_ij);

        let inv_r2 = 1.0 / (     r2 +0.000001);
        let inv_r1 = 1.0 / (sqrt(r2)+0.000001);

        f += 4.0 * m_i * m_j * inv_r2 * (r_j - r_i) * inv_r1;
    }

    simulation_storage.acc[idx] = f;
}

#workgroup_count simulation_kick0 N/256 1 1
@compute @workgroup_size(16, 16)
fn simulation_kick0(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx: u32 = (16u * id.x) + id.y;

    if(u32(N) <= idx) { return; }

    simulation_storage.vel[idx] += 0.5 * custom.dt * simulation_storage.acc[idx];
}

#workgroup_count simulation_drift N/256 1 1
@compute @workgroup_size(16, 16)
fn simulation_drift(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx: u32 = (16u * id.x) + id.y;

    if(u32(N) <= idx) { return; }

    simulation_storage.pos[idx] += custom.dt * simulation_storage.vel[idx];
}

#workgroup_count simulation_kick1 N/256 1 1
@compute @workgroup_size(16, 16)
fn simulation_kick1(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx: u32 = (16u * id.x) + id.y;

    if(u32(N) <= idx) { return; }

    simulation_storage.vel[idx] += 0.5 * custom.dt * simulation_storage.acc[idx];
}
