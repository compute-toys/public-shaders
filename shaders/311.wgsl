#include "Dave_Hoskins/hash"

// sqrt of particle count
#define PARTICLES 64

// PARTICLES / 16
#define PARTICLES_16 4

// number of grid cells per dimension
#define GRID_RES 48

// total number of grid cells = pow(GRID_RES, 2)
#define GRID_SIZE (GRID_RES*GRID_RES+1)

// maximum number of particles per cell
#define GRID_CAP 20

// world size of grid extents
#define GRID_WIDTH 700

struct Atoms {
    count: array<atomic<u32>, GRID_SIZE>,
}

#storage atoms Atoms

struct Particle {
    position: float2,
    velocity: float2,
    pressure: float,
    rho: float, //neighbor density
}

struct Store {
    particles: array<array<Particle, GRID_CAP>, GRID_SIZE>,
    count: array<u32, GRID_SIZE>,
}

#storage store Store

const BOUNDS = float2(GRID_WIDTH,GRID_WIDTH);
const PI = 3.1415926536;

// Settings
const USE_GAS_PRESSURE = false;
const CLAMP_PRESSURE = false;

fn LoadParticle(id: int2) -> Particle {
    var p = Particle();
    p.position = passLoad(0, id, 0).xy;
    p.velocity = passLoad(0, id, 0).zw;
    p.pressure = passLoad(1, id, 0).x;
    p.rho      = passLoad(1, id, 0).y;
    return p;
}

fn SaveParticle(id: int2, p: Particle) {
    passStore(0, id, float4(p.position, p.velocity));
    passStore(1, id, float4(p.pressure, p.rho, 0., 0.));
}

fn grid_cell(pos: float2) -> int2 {
    return clamp(int2((pos / GRID_WIDTH) * float(GRID_RES)), int2(0), int2(GRID_RES));
}

fn grid_cell_id(cell: int2) -> int {
    return cell.x + cell.y * GRID_RES;
}

fn grid_id_to_cell(i: int) -> int2 {
    return int2(i, i / GRID_RES) % GRID_RES;
}

fn grid_cell_dist_min(cell1: int2, cell2: int2) -> float {
    let cell_width = GRID_WIDTH / float(GRID_RES);
    let cdist = distance(float2(cell1), float2(cell2)) * cell_width;
    return max(0., cdist - sqrt(2.) * cell_width);
}

fn grid_increment(cell_id: int) -> int {
    return int(atomicAdd(&atoms.count[cell_id], 1u));
}

fn grid_clear(cell_id: uint) {
    atomicStore(&atoms.count[cell_id], 0u);
}

fn grid_insert(p: Particle) {
    let cell = grid_cell(p.position);
    let cell_id = grid_cell_id(cell);

    let idx = grid_increment(cell_id);
    if (idx >= GRID_CAP) {assert(0,false); return; }

    store.particles[cell_id][idx] = p;
}

fn grid_neighbours() -> array<int2,9> {
    var r: array<int2, 9>;
    var n = 0;
    for (var i = -1; i <= 1; i += 1) {
        for (var j = -1; j <= 1; j += 1) {
            r[n] = int2(i,j);
            n += 1;
        }
    }
    return r;
}

fn border_force(p: float2) -> float2 {

    let d = min(min(p.x,p.y),min(BOUNDS.x-p.x,1e10));

    var dir = float2();
    if (d == p.x) {
        dir = vec2(1,0);
    } else if (d==p.y) {
        dir = float2(0,1);
    } else if (d==BOUNDS.x-p.x) {
        dir = float2(-1,0);
    } else {
        dir = float2(0,-1);
    }

    return exp(-max(d,0.)*max(d,0.))*dir;
}

const DENSITY = 0.036;

fn square(x: float) -> float {
    return x*x;
}

fn Kernel(d: float, h: float) -> float {
    return exp(-square(d/h))/(PI*square(h));
}

fn KernelGrad(d: float, h: float) -> float {
    return -2.*d*Kernel(d,h)/square(h);
}

fn SimulateParticles(id: uint3) {
    var p = LoadParticle(int2(id.xy));

    if(int(id.x) >= PARTICLES || int(id.y) >= PARTICLES) {   
        return;
    }

    if(time.frame < 20u) {
        p.position = hash42(float2(id.xy)).xy * GRID_WIDTH;
        p.velocity = (hash42(float2(id.xy)).zw - .5)/5.;
        p.pressure = 0.;
        p.rho = 5.;
        
        p.position.x *= 0.50; // Dambreak

        SaveParticle(int2(id.xy), p);
        return;
    }

    var force = float2();

    var avg_pressure = 0.;
    let scale = 0.21/DENSITY;
    var rho = Kernel(0., scale); // neighbor density
    var avg_velocity = vec2(p.velocity)*rho;

    let offsets = grid_neighbours();
    let cell = grid_cell(p.position);
        
    // neighbor particles
    for(var o = 0; o < 9; o += 1) {
        let offset = offsets[o];
        let ncell = cell + offset;
        if (any(ncell < int2(0)) || any(ncell >= int2(GRID_RES))) {
            continue;
        }
        let i = grid_cell_id(ncell);
        let gcount = int(store.count[i]);
        for(var j = 0; j < gcount; j += 1) {
            let q = store.particles[i][j];

            let dist = distance(p.position, q.position);

            let dv = q.velocity - p.velocity;
            let dx = q.position - p.position;
            let n_dir = dx/(dist + 0.001); // neighbor direction

            // SPH kernels
            let K = Kernel(dist, scale);
            let dK = KernelGrad(dist, scale);

            let dotv = dot(n_dir, dv); // divergence

            let pressure = -(q.pressure/square(q.rho) + p.pressure/square(p.rho))*n_dir*K;

            rho += K;
            avg_pressure += q.pressure*K;
            avg_velocity += q.velocity*K;

            let viscosity = 1.4*(3. + 3.*length(dv)) * n_dir * dotv * K;
            force += pressure + viscosity;
        }
    }

    // Gravity
    force -= 1e-3*vec2(0.,-1.);

    // Boundaries
    let bdf_p = border_force(p.position);
    force += 0.5 * bdf_p * abs(dot(bdf_p, p.velocity));

    // velocity reflection colliders
    if (BOUNDS.x - p.position.x < 2.) {p.velocity.x = -abs(p.velocity.x); p.position.x = BOUNDS.x - 2.;} // Right 
    if (p.position.x < 2.) {p.velocity.x = abs(p.velocity.x); p.position.x = 2.;} // Left
    if (BOUNDS.y - p.position.y < 2.) {p.velocity.y = -abs(p.velocity.y); p.position.y = BOUNDS.y - 2.;} // floor
    if (p.position.y < 2.) {p.velocity.y = abs(p.velocity.y);} // roof

    if (mouse.click == 1) {
        let mouse_pos = (float2(mouse.pos) - float2(SCREEN_WIDTH/2.0 - SCREEN_HEIGHT/2.0, 0))/SCREEN_HEIGHT*GRID_WIDTH;
        let d = distance(mouse_pos, p.position);
        force += 40.1*(vec2(0,-1) + normalize(p.position - mouse_pos)/3.0)/square(0.2*d*d+2.);
    }

    p.rho = rho;

    const r = 7.;
    const D = 1.;
    if (USE_GAS_PRESSURE) {
        let gasP = 0.03*(p.rho-0.01);
        p.pressure = gasP; // Gas
    } else {
        let waterP = 0.035*DENSITY*(pow(abs(p.rho/DENSITY), r) - D);
        p.pressure = min(waterP,0.04); // Water
    }
    if (CLAMP_PRESSURE) {
        p.pressure = max(p.pressure, 0.0); // pressure clamp
    }

    p.velocity += custom.Timescale * force;
    p.velocity -= custom.Timescale * p.velocity * custom.air_resistance * (0.5*tanh(8.*(length(p.velocity)-1.5))+0.5); // Air resistance?
    p.position += custom.Timescale * p.velocity;

    SaveParticle(int2(id.xy), p);
}

fn ClearGrid(id: uint3) {
    if (id.x >= GRID_SIZE) {
        return;
    }
    grid_clear(id.x);
}

fn BuildGrid(id: uint3) {
    if(int(id.x) >= PARTICLES || int(id.y) >= PARTICLES) {   
        return;
    }

    var p = LoadParticle(int2(id.xy));

    grid_insert(p);
}

fn FreezeGrid(id: uint3) {
    if(int(id.x) >= GRID_SIZE || int(id.y) >= 1) {   
        return;
    }
    let cell_id = int(id.x);
    store.count[cell_id] = uint(atomicLoad(&atoms.count[cell_id]));
}


@compute @workgroup_size(16, 16)
#workgroup_count f1 PARTICLES_16 PARTICLES_16 1
fn f1(@builtin(global_invocation_id) id: uint3) {
    SimulateParticles(id);
}

@compute @workgroup_size(256)
#workgroup_count g1 16 1 1
fn g1(@builtin(global_invocation_id) id: uint3) {
    ClearGrid(id);
}

@compute @workgroup_size(16, 16)
#workgroup_count h1 PARTICLES_16 PARTICLES_16 1
fn h1(@builtin(global_invocation_id) id: uint3) {
    BuildGrid(id);
}

@compute @workgroup_size(256)
#workgroup_count j1 16 1 1
fn j1(@builtin(global_invocation_id) id: uint3) {
    FreezeGrid(id);
}

@compute @workgroup_size(16, 16)
#workgroup_count f2 PARTICLES_16 PARTICLES_16 1
fn f2(@builtin(global_invocation_id) id: uint3) {
    SimulateParticles(id);
}

@compute @workgroup_size(256)
#workgroup_count g2 16 1 1
fn g2(@builtin(global_invocation_id) id: uint3) {
    ClearGrid(id);
}

@compute @workgroup_size(16, 16)
#workgroup_count h2 PARTICLES_16 PARTICLES_16 1
fn h2(@builtin(global_invocation_id) id: uint3) {
    BuildGrid(id);
}

@compute @workgroup_size(256)
#workgroup_count j2 16 1 1
fn j2(@builtin(global_invocation_id) id: uint3) {
    FreezeGrid(id);
}

@compute @workgroup_size(16, 16)
#workgroup_count f3 PARTICLES_16 PARTICLES_16 1
fn f3(@builtin(global_invocation_id) id: uint3) {
    SimulateParticles(id);
}

@compute @workgroup_size(256)
#workgroup_count g3 16 1 1
fn g3(@builtin(global_invocation_id) id: uint3) {
    ClearGrid(id);
}

@compute @workgroup_size(16, 16)
#workgroup_count h3 PARTICLES_16 PARTICLES_16 1
fn h3(@builtin(global_invocation_id) id: uint3) {
    BuildGrid(id);
}

@compute @workgroup_size(256)
#workgroup_count j3 16 1 1
fn j3(@builtin(global_invocation_id) id: uint3) {
    FreezeGrid(id);
}

@compute @workgroup_size(16, 16)
#workgroup_count f4 PARTICLES_16 PARTICLES_16 1
fn f4(@builtin(global_invocation_id) id: uint3) {
    SimulateParticles(id);
}

@compute @workgroup_size(256)
#workgroup_count g4 16 1 1
fn g4(@builtin(global_invocation_id) id: uint3) {
    ClearGrid(id);
}

@compute @workgroup_size(16, 16)
#workgroup_count h4 PARTICLES_16 PARTICLES_16 1
fn h4(@builtin(global_invocation_id) id: uint3) {
    BuildGrid(id);
}

@compute @workgroup_size(256)
#workgroup_count j4 16 1 1
fn j4(@builtin(global_invocation_id) id: uint3) {
    FreezeGrid(id);
}

fn mat_inv(M : float2x2) -> float2x2 {
    return float2x2(M[1][1], -M[0][1], -M[1][0], M[0][0])*(1.0/determinant(M));
}

@compute @workgroup_size(16, 16)
fn Accumulate(@builtin(global_invocation_id) id: uint3) {
    let screen_size = uint2(textureDimensions(screen));
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    
    let pos = (float2(float(id.x), float(id.y)) - float2(SCREEN_WIDTH/2.0 - SCREEN_HEIGHT/2.0, 0.0))/SCREEN_HEIGHT*GRID_WIDTH;
    if (any(pos <= float2(0)) || any(pos >= float2(GRID_WIDTH))) {return;}

    let offsets = grid_neighbours();
    let cell = grid_cell(pos);
        
    var col = float3(0.0);
    // neighbor particles
    for(var o = 0; o < 9; o += 1) {
        let offset = offsets[o];
        let ncell = cell + offset;
        if (any(ncell < int2(0)) || any(ncell >= int2(GRID_RES))) {
            continue;
        }
        let i = grid_cell_id(ncell);
        let gcount = int(store.count[i]);
        for(var j = 0; j < gcount; j += 1) {
            let q = store.particles[i][j];

            // Render as 2d gaussian stretched by particle's velocity
            let b = -q.velocity.x/sqrt(length(q.velocity) + 0.0001);
            let a = q.velocity.y/sqrt(length(q.velocity) + 0.0001);

            let c = 5.0;
            let k = 200.0;
            let M = float2x2(a*a + b*b*k + c, a*b*(1.0-k),a*b*(1.0-k), a*a*k+b*b + c)*mix(0.01,0.06, custom.blur_strength);

            let delta = pos-q.position;
            col += 10*(exp(-0.5*dot(delta, mat_inv(M)*delta))/sqrt(determinant(M))/2/PI) *
                    (float3(0.0005,0.0001,1.0) + smoothstep(0, 1.0, length(q.velocity))*float3(1,1,0));
            // if (distance(pos, q.position) <1.0) {
                // col = float3(0,0,1.0) + smoothstep(0, 1.0, length(q.velocity))*float3(1,1,0);
            // }
        }
    }

    textureStore(screen, int2(id.xy), float4(col, 1.));
}
