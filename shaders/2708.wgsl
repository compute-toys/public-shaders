// timestep size
//#define dt 0.03

// # of bodies (must be a factor of 64)
#define N 65535

const BODY_MASS = 3.0 / N;

struct simulation {
    //mas: array<f32, N>,
    pos: array<vec3<f32>, N>,
    vel: array<vec3<f32>, N>,
    acc: array<vec3<f32>, N>
};

#storage simulation_storage simulation

#storage atomic_storage array<atomic<i32>>

// https://www.jcgt.org/published/0009/03/02/
fn pcg4d(s: vec4<u32>) -> vec4<u32> {
    var v = u32(0x0019660Du) * s + u32(0x3C6EF35Fu);

    v.x += v.y*v.w;
    v.y += v.z*v.x;
    v.z += v.x*v.y;
    v.w += v.y*v.z;

    v ^= v >> vec4<u32>(16u);

    v.x += v.y*v.w;
    v.y += v.z*v.x;
    v.z += v.x*v.y;
    v.w += v.y*v.z;

    return v;
}

const TAU = 6.2831853071795864769252867665590057683943387987502116419498891846156328125724179972560696506842341360;

const INV_U32_MAX = 1.0 / f32(0xFFFFFFFFu);

// multiply by another sqrt(x) for a random point on a unit disk
fn udir2(x: f32) -> vec2<f32> {
    return vec2<f32>(cos(TAU*x), sin(TAU*x));
}

fn udir3(x: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(0., 0., 0.);
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

#dispatch_once simulation_init
#workgroup_count simulation_init N/256 1 1
@compute @workgroup_size(256)
fn simulation_init(@builtin(global_invocation_id) id: vec3<u32>) {
    if(id.x >= N) { return; }

    var s = vec4<u32>(id.x+1u, 420u, 69u, 1337u);

    s = pcg4d(s);
    s = pcg4d(s); // second round for better statistical quality

    //simulation_storage.mas[id.x] = BODY_MASS;

    let pos = vec3<f32>(1.000, 0.125, 1.000)*randn3(INV_U32_MAX*vec4<f32>(s));

    //simulation_storage.pos[id.x] = vec3<f32>(0.0, 0.0, 0.0);
    simulation_storage.pos[id.x] = pos;

    s = pcg4d(s);

    //let vel = vec3<f32>(0.000, 0.000, 0.000);
    //let vel = randn3(INV_U32_MAX*vec4<f32>(s));
    let vel = 0.01*normalize(vec3<f32>(pos.z, pos.y,-pos.x));

    simulation_storage.vel[id.x] = vel;

    simulation_storage.acc[id.x] = vec3<f32>(0.0, 0.0, 0.0);
}

@compute @workgroup_size(16, 16)
fn visualization_clear(@builtin(global_invocation_id) id: vec3<u32>) {
    let image_size = textureDimensions(screen);

    atomicStore(&atomic_storage[4u*(image_size.x*id.y+id.x)+0u], 0);
    atomicStore(&atomic_storage[4u*(image_size.x*id.y+id.x)+1u], 0);
    atomicStore(&atomic_storage[4u*(image_size.x*id.y+id.x)+2u], 0);
    atomicStore(&atomic_storage[4u*(image_size.x*id.y+id.x)+3u], 0);
}

#workgroup_count visualization_render N/256 1 1
@compute @workgroup_size(256)
fn visualization_render(@builtin(global_invocation_id) id: vec3<u32>) {
    if(id.x >= N) { return; }

    let image_size = textureDimensions(screen);

    var pos = simulation_storage.pos[id.x];

    pos *= custom.scale;

    let p = 0.125 * TAU;

    pos = vec3<f32>(pos.x*cos(p)-pos.y*sin(p), pos.x*sin(p)+pos.y*cos(p), pos.z);

    let t = TAU * custom.theta;

    pos = vec3<f32>(pos.x*cos(t)-pos.z*sin(t), pos.y, pos.y*sin(t)+pos.z*cos(t));

    var s = vec4<u32>(id.x, 42067u, 1337u, time.frame);

    s = pcg4d(s);

    for(var i = 0u; i < 16; i++) {
        s = pcg4d(s);

        let d = 0.5*randn2(INV_U32_MAX*vec2<f32>(s.xy));

        var x = i32((f32(image_size.y)*0.25*pos.x+0.5*f32(image_size.x))+d.x-0.5);
        var y = i32((f32(image_size.y)*0.25*pos.y+0.5*f32(image_size.y))+d.y-0.5);

        if(0 <= x && x < i32(image_size.x) && 0 <= y && y < i32(image_size.y)) {
            atomicAdd(&atomic_storage[4u*(image_size.x*u32(y)+u32(x))+0u], 1);
            atomicAdd(&atomic_storage[4u*(image_size.x*u32(y)+u32(x))+1u], 1);
            atomicAdd(&atomic_storage[4u*(image_size.x*u32(y)+u32(x))+2u], 1);
            atomicAdd(&atomic_storage[4u*(image_size.x*u32(y)+u32(x))+3u], 1);
        }
    }
}

@compute @workgroup_size(16, 16)
fn visualization_output(@builtin(global_invocation_id) id: vec3<u32>) {
    let image_size = textureDimensions(screen);

    if(image_size.x <= id.x || image_size.y <= id.y) { return; }

    //atomicLoad();

    // Pixel coordinates (centre of pixel, origin at bottom left)
    //let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);

    // Normalised pixel coordinates (from 0 to 1)
    //let uv = fragCoord / vec2f(screen_size);

    //// Time varying pixel colour
    //var col = .5 + .5 * cos(time.elapsed + uv.xyx + vec3f(0.,2.,4.));

    // Convert from gamma-encoded to linear colour space
    //col = pow(col, (2.2));

    var color = vec4<f32>(0.250, 0.250, 0.250, 1.000);

    color.r = f32(atomicLoad(&atomic_storage[4u*(image_size.x*id.y+id.x)+0u]));
    color.g = f32(atomicLoad(&atomic_storage[4u*(image_size.x*id.y+id.x)+1u]));
    color.b = f32(atomicLoad(&atomic_storage[4u*(image_size.x*id.y+id.x)+2u]));
    color.a = f32(atomicLoad(&atomic_storage[4u*(image_size.x*id.y+id.x)+3u]));

    color *= 0.125*0.500;

    color.r = pow(color.r, 2.2);
    color.g = pow(color.g, 2.2);
    color.b = pow(color.b, 2.2);

    textureStore(screen, id.xy, vec4<f32>(color.rgb, 1.000));
}

#workgroup_count simulation_force N/32 1 1
@compute @workgroup_size(32)
fn simulation_force(@builtin(global_invocation_id) id: vec3<u32>) {
    if(id.x >= N) { return; }

    //let m_i = simulation_storage.mas[id.x];
    let r_i = simulation_storage.pos[id.x];

    var f = vec3<f32>(0.0, 0.0, 0.0);

    for(var j = (time.frame % 4) * (N/4); j < ((time.frame % 4) * (N/4))+(N/4); j++) {
        if(id.x == j) {
            continue;
        }

        //let m_j = simulation_storage.mas[j];
        let r_j = simulation_storage.pos[j];

        let r_ij = r_j - r_i;

        let r2 = dot(r_ij, r_ij);

        let inv_r2 = 1.0 / (     r2 +0.000001);
        let inv_r1 = 1.0 / (sqrt(r2)+0.000001);

        //f += 4.0 * m_i * m_j * inv_r2 * (r_j - r_i) * inv_r1;
        f += 4.0 * BODY_MASS * BODY_MASS * inv_r2 * (r_j - r_i) * inv_r1;
    }

    simulation_storage.acc[id.x] = f;
}

#workgroup_count simulation_kick0 N/256 1 1
@compute @workgroup_size(256)
fn simulation_kick0(@builtin(global_invocation_id) id: vec3<u32>) {
    if(id.x >= N) { return; }

    simulation_storage.vel[id.x] += 0.5 * custom.dt * simulation_storage.acc[id.x];
}

#workgroup_count simulation_drift N/256 1 1
@compute @workgroup_size(256)
fn simulation_drift(@builtin(global_invocation_id) id: vec3<u32>) {
    if(id.x >= N) { return; }

    simulation_storage.pos[id.x] += custom.dt * simulation_storage.vel[id.x];
}

#workgroup_count simulation_kick1 N/256 1 1
@compute @workgroup_size(256)
fn simulation_kick1(@builtin(global_invocation_id) id: vec3<u32>) {
    if(id.x >= N) { return; }

    simulation_storage.vel[id.x] += 0.5 * custom.dt * simulation_storage.acc[id.x];
}
