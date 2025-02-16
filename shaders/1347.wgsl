#include "Dave_Hoskins/hash"

#define PARTICLE_COUNT 10000
#define PARTICLE_WGSIZE 10
#define PARTICLE_WGCOUNT 1000

// Hack to get around two storage buffer limit
#storage memory Memory

struct Particle {
    pos: vec3<f32>,
    vel: vec3<f32>,
    color: vec3<f32>
}

struct Memory {
    particles: array<Particle, PARTICLE_COUNT>,
    framebuf: array<array<array<atomic<u32>, 3>, SCREEN_WIDTH>, SCREEN_HEIGHT>,
    //occupancy: array<array<array<atomic<u32>, 100>, 100>, 100>
}

// Hue to RGB function from Fabrice's shadertoyunofficial blog
fn hue2rgb(hue: f32) -> vec3<f32> {
    return 0.6 + 0.6 * cos(6.3 * hue + vec3<f32>(0.0, 23.0, 21.0));
}

#dispatch_once initParticles
#workgroup_count initParticles PARTICLE_WGCOUNT 1 1
@compute @workgroup_size(PARTICLE_WGSIZE)
fn initParticles(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= PARTICLE_COUNT) { return; }
    let particle = &memory.particles[id.x];
    (*particle).pos = hash31(f32(id.x)) * 60.0 - 30.0;
    (*particle).vel = vec3<f32>(40.0 * normalize(vec2<f32>(-(*particle).pos.z, (*particle).pos.x)), 0.0).xzy;
    (*particle).color = hue2rgb(atan2((*particle).pos.z, (*particle).pos.x));
}

@compute @workgroup_size(16, 16)
fn clearImage(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= SCREEN_WIDTH || id.y >= SCREEN_HEIGHT) { return; }
    textureStore(screen, id.xy, vec4<f32>(0.0, 0.0, 0.0, 1.0));
    atomicStore(&memory.framebuf[id.y][id.x][0], 0xffffff00);
    atomicStore(&memory.framebuf[id.y][id.x][1], 0xffffff00);
    atomicStore(&memory.framebuf[id.y][id.x][2], 0xffffff00);
}

#workgroup_count splatParticles PARTICLE_WGCOUNT 1 1
@compute @workgroup_size(PARTICLE_WGSIZE)
fn splatParticles(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= PARTICLE_COUNT) { return; }
    let particle = &memory.particles[id.x];
    let center = vec2<f32>(f32(SCREEN_WIDTH), f32(SCREEN_HEIGHT)) / 2.0;

    let fd = -normalize(vec3<f32>(f32(mouse.pos.x) - center.x, center.y - f32(mouse.pos.y), 100.0));
    let rt = normalize(vec3<f32>(-fd.z, 0.0, fd.x));
    let up = cross(rt, fd);
    let pos = (*particle).pos * mat3x3<f32>(rt, up, fd) - vec3<f32>(0.0, 0.0, custom.cameraDistance);

    let z = -pos.z;
    if (z < custom.zNear || z > custom.zFar) { return; }
    let depthBits = u32((z - custom.zNear) / (custom.zFar - custom.zNear) * 16777215.0) << 8;
    let proj = pos.xy / z * custom.focalLength;
    let coords = vec2<i32>(i32(center.x + proj.x), i32(center.y - proj.y));

    let encodedR = depthBits | u32((*particle).color.r * 255.0);
    let encodedG = depthBits | u32((*particle).color.g * 255.0);
    let encodedB = depthBits | u32((*particle).color.b * 255.0);

    // Draw to the pixels covered by the particle
    let radius = i32(custom.particleRadius);
    let sqradius = radius * radius;
    for (var i = -radius; i < radius; i++) {
        for (var j = -radius; j < radius; j++) {
            if (i * i + j * j > sqradius) { continue; }
            let pix = coords + vec2<i32>(i, j);
            if (
                pix.x >= 0 && pix.x < SCREEN_WIDTH && // Is x coordinate in bounds
                pix.y >= 0 && pix.y < SCREEN_HEIGHT   // And y coordinate in bounds
            ) {
                atomicMin(&memory.framebuf[pix.y][pix.x][0], encodedR);
                atomicMin(&memory.framebuf[pix.y][pix.x][1], encodedG);
                atomicMin(&memory.framebuf[pix.y][pix.x][2], encodedB);
            }
        }
    }
}

@compute @workgroup_size(16, 16)
fn decodeImage(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= SCREEN_WIDTH || id.y >= SCREEN_HEIGHT) { return; }
    textureStore(screen, id.xy, vec4<f32>(
        f32(atomicLoad(&memory.framebuf[id.y][id.x][0]) & 0xff) / 255.0,
        f32(atomicLoad(&memory.framebuf[id.y][id.x][1]) & 0xff) / 255.0,
        f32(atomicLoad(&memory.framebuf[id.y][id.x][2]) & 0xff) / 255.0,
        1.0
    ));
}

#workgroup_count updateParticles PARTICLE_WGCOUNT 1 1
@compute @workgroup_size(PARTICLE_WGSIZE)
fn updateParticles(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= PARTICLE_COUNT) { return; }
    let me = &memory.particles[id.x];

    // Accumulate change in velocity
    var dv = vec3<f32>(0.0);
    for (var i = 0u; i < PARTICLE_COUNT; i++) {
        if (i == id.x) { continue; }
        let other = &memory.particles[i];
        let to = (*other).pos - (*me).pos;
        dv += normalize(to) / max(0.01, dot(to, to)) * custom.gravity * custom.deltaTime;
    }

    // Euler integration
    (*me).vel += dv;
    (*me).pos += (*me).vel * custom.deltaTime;
}