#include "Dave_Hoskins/hash"

#define noise_simplex_2d_hash hash22
#include "iq/noise_simplex_2d"

#define noise_simplex_3d_hash hash33
#include "nikat/noise_simplex_3d"

#define COUNT (1<<19)
#define PARTICLE_WORKGROUP_SIZE 128
#define PARTICLE_WORKGROUP_COUNT 8192
#storage PARTICLES array<Particle, COUNT>
#define SCREEN_AREA (SCREEN_WIDTH*SCREEN_HEIGHT)
// #storage DENSITIES array<i32, SCREEN_AREA>
#storage DENSITIES array<atomic<i32>, SCREEN_AREA>

struct Particle {
    pos: vec2f,
    vel: vec2f,
}

#dispatch_once initialization
#workgroup_count initialization PARTICLE_WORKGROUP_COUNT 1 1 
@compute @workgroup_size(PARTICLE_WORKGROUP_SIZE, 1, 1)
fn initialization(@builtin(global_invocation_id) id: vec3u) {
    // let pos = hash21(f32(id.x)) * vec2f(textureDimensions(screen));
    let hh = SCREEN_HEIGHT / 1u;
    let pos = vec2f(f32(id.x/hh), f32(id.x % hh)) * 1.0;

    PARTICLES[id.x] = Particle(pos, vec2f(0.0, 0.0));
}

#workgroup_count update PARTICLE_WORKGROUP_COUNT 1 1 
@compute @workgroup_size(PARTICLE_WORKGROUP_SIZE, 1, 1)
fn update(@builtin(global_invocation_id) id: vec3u) {
    let particle = &PARTICLES[id.x];

    // let ppos = particle.pos*0.003;
    // let noise = vec2f(
    //     noise_simplex_3d(vec3f(ppos, time.elapsed*0.02))*0.05,
    //     noise_simplex_3d(vec3f(ppos, 1347.0+time.elapsed*0.02))*0.05
    // );
    // particle.vel += noise;

    // particle.vel += (vec2f(SCREEN_WIDTH*0.5, SCREEN_HEIGHT*0.5)-particle.pos)*0.00001;

	if (mouse.click != 0) {
		let delta = vec2f(mouse.pos) - particle.pos;
		let r = length(delta);
		let accel = 0.1 / r;
		particle.vel += delta * accel;
	}
    particle.pos += particle.vel;
	particle.vel *= 0.995;
	
    // PARTICLES[id.x] = particle;

    let size = textureDimensions(screen);
    let pix = vec2i(round(particle.pos));
    if (pix.x < 0 || pix.x >= SCREEN_WIDTH || pix.y < 0 || pix.y >= SCREEN_HEIGHT) {
		return;
	}
    let density_index = pix.y * SCREEN_WIDTH + pix.x;
    // DENSITIES[density_index] += 1;
    atomicAdd(&DENSITIES[density_index], 1);
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // // Pixel coordinates (centre of pixel, origin at bottom left)
    // let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);

    // // Normalised pixel coordinates (from 0 to 1)
    // let uv = fragCoord / vec2f(screen_size);

    // // Time varying pixel colour
    // var col = .5 + .5 * cos(time.elapsed + uv.xyx + vec3f(0.,2.,4.));

    // // Convert from gamma-encoded to linear colour space
    // col = pow(col, vec3f(2.2));
    let density_index = id.y * SCREEN_WIDTH + id.x;
    // let density: f32 = f32(DENSITIES[density_index]);
    let density: f32 = f32(atomicLoad(&DENSITIES[density_index]));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(density * vec3f(0.01, 0.04, 0.1), 1.0));
    // DENSITIES[density_index] = 0;
    atomicStore(&DENSITIES[density_index], 0);
}
