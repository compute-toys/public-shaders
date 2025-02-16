#define CAMERA_ZOOM 40.0
#define SIMULATION_SPEED 5.0
#define PARTICLE_COUNT 256
#define PARTICLE_WGCOUNT 1
#define PARTICLE_WGSIZE 256

var<private> seed: u32;

#storage memory Memory
struct Memory {
    lastTime: f32,
    oldStartIndex: i32,
    newStartIndex: i32,
    particles: array<Particle, PARTICLE_COUNT * 2>,
    framebuffer: array<array<array<atomic<u32>, 3>, SCREEN_WIDTH>, SCREEN_HEIGHT>
};

struct Particle {
    pos: vec3<f32>,
    color: vec3<f32>
}

fn wangHash() -> u32 {
    seed = (seed ^ 61u) ^ (seed >> 16u);
    seed *= 9u;
    seed = seed ^ (seed >> 4);
    seed *= 668265261u;
    seed = seed ^ (seed >> 15);
    return seed;
}

fn rand01() -> f32 {
    return f32(wangHash()) / 4294967296.0;
}

#dispatch_once initParticles
#workgroup_count initParticles PARTICLE_WGCOUNT 1 1
@compute @workgroup_size(PARTICLE_WGSIZE)
fn initParticles(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= PARTICLE_COUNT) { return; }
    seed = (id.x * 1973u + 26699u) | 1u;

    memory.lastTime = time.elapsed;
    memory.oldStartIndex = 0;
    memory.newStartIndex = PARTICLE_COUNT;

    var pointInSphere: vec3<f32>;
    loop {
        pointInSphere = vec3<f32>(rand01(), rand01(), rand01()) * 2.0 - 1.0;
        if (dot(pointInSphere, pointInSphere) <= 1.0) { break; }
    };

    let particle = &memory.particles[id.x];
    (*particle).pos = normalize(pointInSphere);
    (*particle).color = vec3<f32>(rand01(), rand01(), rand01());
}

#workgroup_count updateParticles PARTICLE_WGCOUNT 1 1
@compute @workgroup_size(PARTICLE_WGSIZE)
fn updateParticles(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= PARTICLE_COUNT) { return; }

    var deltaTime = time.elapsed - memory.lastTime;
    memory.lastTime = time.elapsed;

    let newParticle = &memory.particles[memory.newStartIndex + i32(id.x)];
    *newParticle = memory.particles[memory.oldStartIndex + i32(id.x)];
    for (var i = 0; i < PARTICLE_COUNT; i++) {
        if (i == i32(id.x)) { continue; }
        let otherParticle = &memory.particles[memory.oldStartIndex + i];
        var push = (*newParticle).pos - (*otherParticle).pos;
        (*newParticle).pos += normalize(push) / (dot(push, push) + 1.0) * SIMULATION_SPEED * deltaTime;
    }

    (*newParticle).pos = normalize((*newParticle).pos);

    var temp = memory.oldStartIndex;
    memory.oldStartIndex = memory.newStartIndex;
    memory.newStartIndex = temp;
}

@compute @workgroup_size(16, 16)
fn clearImage(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= SCREEN_WIDTH || id.y >= SCREEN_HEIGHT) { return; }
    textureStore(screen, id.xy, vec4<f32>(0.0, 0.0, 0.0, 1.0));
    atomicStore(&memory.framebuffer[id.y][id.x][0], 0xffffff00);
    atomicStore(&memory.framebuffer[id.y][id.x][1], 0xffffff00);
    atomicStore(&memory.framebuffer[id.y][id.x][2], 0xffffff00);
}

#workgroup_count drawParticles PARTICLE_WGCOUNT 1 1
@compute @workgroup_size(PARTICLE_WGSIZE)
fn drawParticles(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= PARTICLE_COUNT) { return; }
    let particle = &memory.particles[memory.oldStartIndex + i32(id.x)];
    var center = vec2<f32>(f32(SCREEN_WIDTH), f32(SCREEN_HEIGHT)) / 2.0;

    var fd = -normalize(vec3<f32>(f32(mouse.pos.x) - center.x, center.y - f32(mouse.pos.y), 100.0));
    var rt = normalize(vec3<f32>(-fd.z, 0.0, fd.x));
    var up = cross(rt, fd);
    var pos = (*particle).pos * CAMERA_ZOOM * mat3x3<f32>(rt, up, fd) - vec3<f32>(0.0, 0.0, custom.cameraDistance);

    var z = -pos.z;
    if (z < custom.zNear || z > custom.zFar) { return; }
    var depthBits = u32((z - custom.zNear) / (custom.zFar - custom.zNear) * 16777215.0) << 8;
    var proj = pos.xy / z * custom.focalLength;
    var coords = vec2<i32>(
        i32(center.x + proj.x * f32(SCREEN_HEIGHT)),
        i32(center.y - proj.y * f32(SCREEN_HEIGHT))
    );

    var encodedR = depthBits | u32((*particle).color.r * 255.0);
    var encodedG = depthBits | u32((*particle).color.g * 255.0);
    var encodedB = depthBits | u32((*particle).color.b * 255.0);

    var radius = i32(custom.particleRadius / z * custom.focalLength);
    var sqradius = radius * radius;
    for (var i = -radius; i < radius; i++) {
        for (var j = -radius; j < radius; j++) {
            if (i * i + j * j > sqradius) { continue; }
            var pix = coords + vec2<i32>(i, j);
            if (
                pix.x >= 0 && pix.x < SCREEN_WIDTH &&
                pix.y >= 0 && pix.y < SCREEN_HEIGHT
            ) {
                atomicMin(&memory.framebuffer[pix.y][pix.x][0], encodedR);
                atomicMin(&memory.framebuffer[pix.y][pix.x][1], encodedG);
                atomicMin(&memory.framebuffer[pix.y][pix.x][2], encodedB);
            }
        }
    }
}

@compute @workgroup_size(16, 16)
fn decodeImage(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= SCREEN_WIDTH || id.y >= SCREEN_HEIGHT) { return; }
    textureStore(screen, id.xy, vec4<f32>(
        f32(atomicLoad(&memory.framebuffer[id.y][id.x][0]) & 0xff) / 255.0,
        f32(atomicLoad(&memory.framebuffer[id.y][id.x][1]) & 0xff) / 255.0,
        f32(atomicLoad(&memory.framebuffer[id.y][id.x][2]) & 0xff) / 255.0,
        1.0
    ));
}