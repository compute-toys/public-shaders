
//Check Uniforms
//Mode 0 - additive blending (atomicAdd)
//Mode 1 - closest sample (atomicMax)

#define MAX_RASTER_AREA 8192.0

//Particle count
#define group_size 32
#define group_count 4096
#define N (group_size * group_count)
#define SIMULATION_GROUP_SIZE 256
#define SIMULATION_GROUPS 512 // N / SIMULATION_GROUP_SIZE

//Rasterization grid parameters 
#define GRID_SIZE_X 128
#define GRID_SIZE_Y 64
#define GRID_SIZE_Z 128
#define GRID_COUNT 1048576 // GRID_SIZE_X*GRID_SIZE_Y*GRID_SIZE_Z
#define GRID_GROUP_SIZE 256
#define GRID_GROUPS 4096 // GRID_COUNT/GRID_GROUP_SIZE

//Simulation parameters
#define KERNEL_VOXEL_RADIUS 2
#define MAX_COMPUTED_DISTANCE 4
#define RADIUS_SCALE custom.RadiusScale
#define DENSITY_RADIUS (1.0 * RADIUS_SCALE)
#define PRESSURE custom.Pressure
#define PRESSURE_RAD (0.85 * RADIUS_SCALE)
#define VISCOSITY custom.Viscosity
#define SPIKE_KERNEL custom.Spike
#define SPIKE_RAD (0.75 * RADIUS_SCALE)
#define MAX_VELOCITY 1.0
#define DELTA_TIME custom.TimeStep
#define BOUNDARY_FORCE 5.0
#define REST_DENSITY custom.RestDensity
#define GRAVITY custom.Gravity
#define GRAVITY_DOWN custom.GravityDown
#define GRAVITY_OSCILLATION custom.GravityOscill
#define SIM_TIME (f32(time.frame)/120.0)

#storage sim Simulation

const PI = 3.14159265;
const TWO_PI = 6.28318530718;
const INVPI = 0.31830988618;
const size3d = vec3i(GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z);
const dt = 1.0;


struct Particle 
{
    pos: vec3f,
    vel: vec3f,
    force: vec3f,
    mass: f32,
    density: f32,
}

struct avec3i
{
    x: atomic<i32>,
    y: atomic<i32>,
    z: atomic<i32>,
}

struct AtomicCell 
{
    pos: avec3i,
    mass: atomic<i32>,

    vel: avec3i,
    density: atomic<i32>,
}

struct Cell 
{
    pos: vec3f,
    mass: f32,

    vel: vec3f,
    density: f32,
}

const quant_scale = 65536.0;

fn FloatToInt(f: f32) -> i32
{
    return i32(f * quant_scale);
}

fn IntToFloat(i: i32) -> f32
{
    return f32(i) / quant_scale;
}

struct Simulation 
{
    // N particles
    pos: array<vec3f, N>,
    vel: array<vec3f, N>,
    density: array<f32, N>,
    atomic_grid: array<AtomicCell, size3d.x * size3d.y * size3d.z>,
    grid: array<Cell, size3d.x * size3d.y * size3d.z>,
}

fn GetIdxFromPos(p: vec3i) -> int 
{
    return p.x + p.y * size3d.x + p.z * size3d.x * size3d.y;
}

fn PosFromIdx(idx: int) -> vec3i 
{
    let z = idx / (size3d.x * size3d.y);
    let y = (idx - z * size3d.x * size3d.y) / size3d.x;
    let x = idx - z * size3d.x * size3d.y - y * size3d.x;
    return vec3i(x, y, z);
}

fn AddParticleToGrid0(p: Particle) 
{
    let idx = GetIdxFromPos(vec3i(p.pos));
    let cell = &sim.atomic_grid[idx];
    atomicAdd(&cell.pos.x, FloatToInt(p.pos.x * p.mass));
    atomicAdd(&cell.pos.y, FloatToInt(p.pos.y * p.mass));
    atomicAdd(&cell.pos.z, FloatToInt(p.pos.z * p.mass));
    atomicAdd(&cell.mass, FloatToInt(p.mass));
}

fn AddParticleToGrid1(p: Particle) 
{
    let idx = GetIdxFromPos(vec3i(p.pos));
    let cell = &sim.atomic_grid[idx];
    atomicAdd(&cell.vel.x, FloatToInt(p.vel.x * p.mass));
    atomicAdd(&cell.vel.y, FloatToInt(p.vel.y * p.mass));
    atomicAdd(&cell.vel.z, FloatToInt(p.vel.z * p.mass));
    atomicAdd(&cell.density, FloatToInt(p.density * p.mass));
}

fn SubtractParticle(p0: Particle, p1: Particle) -> Particle 
{
    var p = Particle();
    p.mass = p0.mass - p1.mass;
    if (p.mass <= 0.0) {
        p.mass = 0.0;
        return p;
    }
    let inv_mass = 1.0 / p.mass;
    p.pos = (p0.pos * p0.mass - p1.pos * p1.mass) * inv_mass;
    p.vel = (p0.vel * p0.mass - p1.vel * p1.mass) * inv_mass;
    p.density = (p0.density * p0.mass - p1.density * p1.mass) * inv_mass;
    return p;
}

fn GetAvgParticleFromAtomicGrid(cell_idx: vec3i) -> Particle 
{
    let idx = GetIdxFromPos(cell_idx);
    let cell = &sim.atomic_grid[idx];
    let mass = IntToFloat(atomicLoad(&cell.mass));
    if(mass <= 0.0) { return Particle(); }
    let pos = vec3f(
        IntToFloat(atomicLoad(&cell.pos.x)),
        IntToFloat(atomicLoad(&cell.pos.y)),
        IntToFloat(atomicLoad(&cell.pos.z))
    ) / mass;
    let vel = vec3f(
        IntToFloat(atomicLoad(&cell.vel.x)),
        IntToFloat(atomicLoad(&cell.vel.y)),
        IntToFloat(atomicLoad(&cell.vel.z))
    ) / mass;
    let dens = IntToFloat(atomicLoad(&cell.density)) / mass;
    return Particle(pos, vel, vec3f(0.0), mass, dens);
}

fn GetAvgParticleFromGridID(idx: i32) -> Particle 
{
    let cell = sim.grid[idx];
    return Particle(cell.pos, cell.vel, vec3f(0.0), cell.mass, cell.density);
}

fn GetAvgParticleFromGrid(cell_idx: vec3i) -> Particle 
{
    let idx = GetIdxFromPos(cell_idx);
    return GetAvgParticleFromGridID(idx);
}

fn SaveParticle(idx: uint, p: Particle) 
{
    sim.pos[idx] = p.pos;
    sim.vel[idx] = p.vel;
    sim.density[idx] = p.density;
}

fn GetParticle(idx: uint) -> Particle 
{
    return Particle(sim.pos[idx], sim.vel[idx], vec3f(0.0), 1.0, sim.density[idx]);
}

var<private> state : uint4;

fn pcg4d(a: uint4) -> uint4
{
	var v = a * 1664525u + 1013904223u;
    v.x += v.y*v.w; v.y += v.z*v.x; v.z += v.x*v.y; v.w += v.y*v.z;
    v = v ^  ( v >> uint4(16u) );
    v.x += v.y*v.w; v.y += v.z*v.x; v.z += v.x*v.y; v.w += v.y*v.z;
    return v;
}

fn rand4() -> float4
{ 
    state = pcg4d(state);
    return float4(state)/float(0xffffffffu); 
}

fn nrand4(sigma: float, mean: float4) -> float4
{
    let Z = rand4();
    return mean + sigma * sqrt(-2.0 * log(Z.xxyy)) * 
           float4(cos(TWO_PI * Z.z),sin(TWO_PI * Z.z),cos(TWO_PI * Z.w),sin(TWO_PI * Z.w));
}

fn minv(a: vec3f) -> f32 {
    return min(min(a.x, a.y), a.z);
}

fn maxv(a: vec3f) -> f32 {
    return max(max(a.x, a.y), a.z);
}

fn distance2border(p: vec3f) -> f32 {
    // Scalar ‘1.0’ is broadcast into (1.0, 1.0, 1.0) when subtracted from size3d
    let a = (vec3f(size3d) - 1.0) - p;
    return min(minv(p), minv(a)) + 1.0;
}

fn border_grad(p: vec3f) -> vec4f {
    let dx = 0.001;
    // 4-component k so we can use .xyyx, .xxxx, etc.
    let k = vec4f(1.0, -1.0, 0.0, 1.0);

    return (
        k.xyyx * distance2border(p + k.xyy * dx) +
        k.yyxx * distance2border(p + k.yyx * dx) +
        k.yxyx * distance2border(p + k.yxy * dx) +
        k.xxxx * distance2border(p + k.xxx * dx)
    ) / vec4f(4.0 * dx, 4.0 * dx, 4.0 * dx, 4.0);
}

// Simple helper to square a number
fn sqr(x: f32) -> f32 {
    return x * x;
}

// Simple helper to cube a number
fn cub(x: f32) -> f32 {
    return x * x * x;
}

fn Pressure(rho: f32) -> f32 {
    return (rho / REST_DENSITY - 1.0) / max(rho * rho, 0.001);
}

fn GaussianNorm(d: f32) -> f32 {
    return 1.0 / (cub(d) * sqrt(cub(TWO_PI)));
}

// Gaussian kernel
fn Gaussian(r: f32, d: f32) -> f32 {
    return GaussianNorm(d) * exp(-0.5 * sqr(r / d));
}

// Gradient of the Gaussian kernel
fn GaussianGrad(dx: vec3f, d: f32) -> vec3f {
    let r = length(dx);
    return - (GaussianNorm(d) / sqr(d)) * exp(-0.5 * sqr(r / d)) * dx;
}

// Returns the gradient and the value of the kernel
fn GaussianGrad2(dx: vec3f, d: f32) -> vec4f {
    let r = length(dx);
    return GaussianNorm(d) * exp(-0.5 * sqr(r / d)) * vec4f(-dx / sqr(d), 1.0);
}

#workgroup_count ClearGrid GRID_GROUPS 1 1
@compute @workgroup_size(GRID_GROUP_SIZE)
fn ClearGrid(@builtin(global_invocation_id) id: uint3) {
    let idx = int(id.x);
    let cell = &sim.atomic_grid[idx];
    atomicStore(&cell.pos.x, 0);
    atomicStore(&cell.pos.y, 0);
    atomicStore(&cell.pos.z, 0);
    atomicStore(&cell.vel.x, 0);
    atomicStore(&cell.vel.y, 0);
    atomicStore(&cell.vel.z, 0);
    atomicStore(&cell.mass, 0);
    atomicStore(&cell.density, 0);
}

#workgroup_count Particle2Grid group_count 1 1
@compute @workgroup_size(group_size)
fn Particle2Grid(@builtin(global_invocation_id) id: uint3) {
    let idx = id.x;
    var p = GetParticle(idx);

    //Scatter particle to atomic_grid
    AddParticleToGrid0(p);
}

#workgroup_count CopyAtomicToRegular1 GRID_GROUPS 1 1
@compute @workgroup_size(GRID_GROUP_SIZE)
fn CopyAtomicToRegular1(@builtin(global_invocation_id) id: uint3) {
    let idx = int(id.x);
    let p = GetAvgParticleFromAtomicGrid(PosFromIdx(idx));
    sim.grid[idx] = Cell(p.pos, p.mass, p.vel, p.density);
}

fn ComputeDensityTerm(p0: Particle, p1: Particle) -> f32 
{
    if (p1.mass <= 0.0) {return 0.0;}
    let dx = p0.pos - p1.pos;
    let r = length(dx);
    return p1.mass * Gaussian(r, DENSITY_RADIUS);
}

#workgroup_count ComputeDensity SIMULATION_GROUPS 1 1
@compute @workgroup_size(SIMULATION_GROUP_SIZE)
fn ComputeDensity(@builtin(global_invocation_id) id: uint3) {
    let idx = id.x;
    var p = GetParticle(idx);

    // Compute density
    p.density = p.mass * GaussianNorm(DENSITY_RADIUS); // Self density
    let cell_idx = vec3i(p.pos);
    for (var i = -KERNEL_VOXEL_RADIUS; i <= KERNEL_VOXEL_RADIUS; i++) {
    for (var j = -KERNEL_VOXEL_RADIUS; j <= KERNEL_VOXEL_RADIUS; j++) {
    for (var k = -KERNEL_VOXEL_RADIUS; k <= KERNEL_VOXEL_RADIUS; k++) {
        // Skip if center 
        let di = abs(i) + abs(j) + abs(j);
        if(di == 0 || di > MAX_COMPUTED_DISTANCE) {continue;}
        let cell = GetAvgParticleFromGrid(cell_idx + vec3i(i, j, k));
        // Compute neighbor particle density contribution
        p.density += ComputeDensityTerm(p, cell);
    } } }
    // Get central particle
    let cell = GetAvgParticleFromGrid(cell_idx);
    // Unfuse self from average particle
    let cell_without_p = SubtractParticle(cell, p);
    // Compute central neighbor particle density contribution
    p.density += ComputeDensityTerm(p, cell_without_p);

    // Save particle state
    SaveParticle(idx, p); 
}

#workgroup_count ParticleDensity2Grid group_count 1 1
@compute @workgroup_size(group_size)
fn ParticleDensity2Grid(@builtin(global_invocation_id) id: uint3) {
    let idx = id.x;
    var p = GetParticle(idx);

    //Scatter particle to atomic_grid
    AddParticleToGrid1(p);
}

#workgroup_count CopyAtomicToRegular2 GRID_GROUPS 1 1
@compute @workgroup_size(GRID_GROUP_SIZE)
fn CopyAtomicToRegular2(@builtin(global_invocation_id) id: uint3) {
    let idx = int(id.x);
    let p = GetAvgParticleFromAtomicGrid(PosFromIdx(idx));
    sim.grid[idx] = Cell(p.pos, p.mass, p.vel, p.density);
}

fn ComputeForce(p: Particle, incoming: Particle) -> vec3f {
    // Make a local mutable copy of p so we can update it.
    var outP = p;

    // Early exit if either mass is zero
    if (incoming.mass <= 0.0 || outP.mass <= 0.0) {
        return vec3f(0.0);
    }

    let dx = incoming.pos - outP.pos;
    let ggrad = GaussianGrad(dx, PRESSURE_RAD);

    // If gradient is negligible, do nothing
    if (length(ggrad) < 1e-5) {
        return vec3f(0.0);
    }

    let dv = incoming.vel - outP.vel;
    let d  = length(dx);
    let dir = dx / max(d, 1e-3);

    let mass1 = f32(incoming.mass);

    let pressure = 0.5 * outP.density * (Pressure(outP.density) + Pressure(incoming.density));

    // Forces
    let F_SPH  = -PRESSURE * pressure * ggrad;
    let F_VISC = VISCOSITY * dot(dir, dv) * ggrad;
    let F_SPIKE = SPIKE_KERNEL * Gaussian(d, SPIKE_RAD) * dir;

    // Final force
    return -(F_SPH + F_VISC + F_SPIKE) * mass1;
}

fn ClampVector(v: vec3f, lim: f32) -> vec3f {
    let l = length(v);
    return select(v, lim * v / l, l > lim);
}

#workgroup_count Simulate SIMULATION_GROUPS 1 1
@compute @workgroup_size(SIMULATION_GROUP_SIZE)
fn Simulate(@builtin(global_invocation_id) id: uint3) {
    let idx = id.x;
    var p = GetParticle(idx);

    // Initialize random number generator
    state = uint4(idx, 0, 0, 0);

    // If first frame, initialize particle positions
    if (time.frame == 0 || all(p.pos <= vec3f(0.01))) {
        // Random position
        p.pos = (0.1 + 0.5 * rand4().xyz) * vec3f(size3d);
        // Random velocity
        p.vel = nrand4(0.001, float4(0.0, 0.0, 0.0, 0.0)).xyz;
    }

    // Compute forces
    p.force = vec3f(0.0);

    // Compute SPH forces
    let cell_idx = vec3i(p.pos);
    // Get central particle
    let ccell = GetAvgParticleFromGrid(cell_idx);
    // Unfuse self from average particle
    let ccell_without_p = SubtractParticle(ccell, p);
    // Compute central neighbor particle force contribution
    p.force += ComputeForce(p, ccell_without_p);

    for (var i = -KERNEL_VOXEL_RADIUS; i <= KERNEL_VOXEL_RADIUS; i++) {
    for (var j = -KERNEL_VOXEL_RADIUS; j <= KERNEL_VOXEL_RADIUS; j++) {
    for (var k = -KERNEL_VOXEL_RADIUS; k <= KERNEL_VOXEL_RADIUS; k++) {
        // Skip if center 
        let di = abs(i) + abs(j) + abs(j);
        if(di == 0 || di > MAX_COMPUTED_DISTANCE) {continue;}
        let cell = GetAvgParticleFromGrid(cell_idx + vec3i(i, j, k));
        // Compute neighbor particle force contribution
        p.force += ComputeForce(p, cell);
    } } }
  

    // Boundary forces
    let border = border_grad(p.pos);
    let bound = BOUNDARY_FORCE * normalize(border.xyz) * exp(-0.3 * border.w * border.w);
    p.force += bound;

    // Gravity
    p.force += GRAVITY * vec3f(GRAVITY_OSCILLATION*sin(1.5*SIM_TIME), GRAVITY_DOWN, GRAVITY_OSCILLATION*cos(0.75*SIM_TIME)); //gravity

    // Integrate
    p.vel += DELTA_TIME * p.force / p.mass;

    // Clamp velocity
    p.vel = ClampVector(p.vel, MAX_VELOCITY);

    p.pos += DELTA_TIME * p.vel;

    // Clamp position
    p.pos = clamp(p.pos, vec3f(0.0), vec3f(size3d) - 1.0);

    // Save particle state
    SaveParticle(idx, p); 
}

@compute @workgroup_size(16, 16)
fn Clear(@builtin(global_invocation_id) id: uint3) {
    let screen_size = int2(textureDimensions(screen));
    let idx0 = int(id.x) + int(screen_size.x * int(id.y));

    atomicStore(&gstate.screen[idx0*4+0], 0);
    atomicStore(&gstate.screen[idx0*4+1], 0);
    atomicStore(&gstate.screen[idx0*4+2], 0);
    atomicStore(&gstate.screen[idx0*4+3], 0);
}

const DEPTH_MIN = 0.1;
const DEPTH_MAX = 20.0;
const DEPTH_BITS = 16u;

fn SimToWorld(pos: vec3f) -> vec3f {
    return pos/f32(size3d.y) - 0.5;
}

fn WorldToSim(pos: vec3f) -> vec3f {
    return (pos + 0.5) * f32(size3d.y);
}

#workgroup_count Rasterize group_count 1 1
@compute @workgroup_size(group_size)
fn Rasterize(@builtin(global_invocation_id) id: uint3) {

    SetCamera();

    let idx = id.x;
    let particle = GetParticle(idx);
    let pos = particle.pos;
    var col = abs(particle.vel) + 0.25*vec3f(0.9,0.9,1.0);

    let lightDir = -normalize(vec3f(1.0, 0.5, 0.5));
    let shadowDensity = RayMarchDensityConstantStepSize(pos+lightDir, lightDir, 0.5);
    col =col*exp(-0.25*vec3f(2,1.5,1)*shadowDensity);

    if(custom.RenderMode < 0.5)
    {
        let cameraDir = normalize(WorldToSim(camera.pos) - pos);
        let cameraOcclusion = RayMarchDensityConstantStepSize(pos, cameraDir, 0.5);
        col =col*exp(-0.1*vec3f(2,1.5,1)*cameraOcclusion);
    }

    let rot = vec3f(0.0, 0.0, 0.0);
    let q = axis_angle_to_quaternion(rot);
    RasterizeEllipsoid(col, Ellipsoid(SimToWorld(pos), particle.density * custom.R0 *vec3f(1.0), q));
}

// TO DEBUG RASTERIZED AVERAGES OF PARTICLES ON GRID

// #workgroup_count RasterizeGrid GRID_GROUPS 1 1
// @compute @workgroup_size(GRID_GROUP_SIZE)
// fn RasterizeGrid(@builtin(global_invocation_id) id: uint3) {

//     SetCamera();

//     let idx = int(id.x);
//     let particle = GetAvgParticleFromGridID(idx);
//     let pos = particle.pos;
//     if(particle.mass < 1.0) {return;}
//     let col = vec3f(1.0, 0.2, 0.2);
//     let rot = vec3f(0.0, 0.0, 0.0);
//     let q = axis_angle_to_quaternion(rot);
//     RasterizeEllipsoid(col, Ellipsoid(pos/f32(size3d.y) - 0.5,particle.density * custom.R*vec3f(1.0), q));
// }

fn Sample(pos: int2) -> vec4f
{
    let screen_size = int2(textureDimensions(screen));
    let idx = pos.x + screen_size.x * pos.y;

    var color: vec4f;
    if(custom.RenderMode < 0.5)
    {
        let x = float(atomicLoad(&gstate.screen[idx*4+0]))/(256.0);
        let y = float(atomicLoad(&gstate.screen[idx*4+1]))/(256.0);
        let z = float(atomicLoad(&gstate.screen[idx*4+2]))/(256.0);
        color = vec4f(x,y,z,0);
    }
    else
    {
        let xdata = atomicLoad(&gstate.screen[idx*4+0]);
        let x = Unpack(xdata);
        let y = Unpack(atomicLoad(&gstate.screen[idx*4+1]));
        let z = Unpack(atomicLoad(&gstate.screen[idx*4+2]));
        let depth = UnpackDepth(xdata);
        color = vec4f(x,y,z,f32(depth));
    }

    return abs(color);
}

fn skyTexture(rd: vec3f) -> vec3f
{
    return textureSampleLevel(channel0, bilinear, INVPI * vec2f(.5 * atan2(rd.z, rd.x), asin(rd.y)) + .5, 0.).rgb;
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) 
{
    SetCamera();

    let screen_size = uint2(textureDimensions(screen));

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = float2(float(id.x) + .5, float(id.y) + .5);

    var color = Sample(int2(id.xy));
    let depth = color.w;
    let ray = RayFromPixel(camera, vec2f(id.xy));

    // Sample background from cube map
    let sky = skyTexture(ray.rd);
    color = 1.5*color;
    if(custom.RenderMode < 0.5)  {
        let absorb = exp(-20.0*length(color));
        color = vec4f(color.xyz + absorb*sky, 0);
    } else {
        if(depth < 1.0) {
            color = vec4f(sky, 0);
        }
    }
    
    // Output to screen (linear colour space)
    textureStore(screen, int2(id.xy), float4(tanh(color.xyz), 1.));
}

fn SetCamera()
{
    camera.fov = gstate.fov;
    camera.cam[0] = gstate.camera * float3(1,0,0);
    camera.cam[1] = gstate.camera * float3(0,1,0);
    camera.cam[2] = gstate.camera * float3(0,0,1);
    camera.pos = gstate.pos;
    camera.size = float2(textureDimensions(screen));
}


////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
/////// RENDERING UTILITIES BELOW //////////
////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////
////////////////////////////////////////////

fn RayBoxIntersection(
    rayOrigin: vec3f,
    rayDir: vec3f,
    boxMin: vec3f,
    boxMax: vec3f
) -> vec2f {
    // Compute 1 / rayDir, handle zero to avoid division by zero.
    let invDir = 1.0 / rayDir;

    // Parametric values for intersection with "min" and "max" planes on each axis
    let t1 = (boxMin.x - rayOrigin.x) * invDir.x;
    let t2 = (boxMax.x - rayOrigin.x) * invDir.x;
    let t3 = (boxMin.y - rayOrigin.y) * invDir.y;
    let t4 = (boxMax.y - rayOrigin.y) * invDir.y;
    let t5 = (boxMin.z - rayOrigin.z) * invDir.z;
    let t6 = (boxMax.z - rayOrigin.z) * invDir.z;

    // tMin = largest of the "lower" intersection values
    let tMin = max(
        max(min(t1, t2), min(t3, t4)),
        min(t5, t6)
    );
    // tMax = smallest of the "upper" intersection values
    let tMax = min(
        min(max(t1, t2), max(t3, t4)),
        max(t5, t6)
    );

    return vec2f(tMin, tMax);
}

fn RayMarchDensityConstantStepSize(
    rayOrigin: vec3f,
    rayDir: vec3f,
    stepSize: f32
) -> f32 {
    // Treat simulation grid as an axis-aligned box from [0,0,0] to [size3d.x, size3d.y, size3d.z].
    let boxMin = vec3f(0.0, 0.0, 0.0);
    let boxMax = vec3f(
        f32(size3d.x),
        f32(size3d.y),
        f32(size3d.z)
    );

    // 1. Find the parametric intersection (tMin, tMax) of the ray with the bounding box.
    let intersection = RayBoxIntersection(rayOrigin, rayDir, boxMin, boxMax);
    var tMin = intersection.x;
    var tMax = intersection.y;

    // If there's no intersection, or tMin > tMax, bail out.
    if (tMin > tMax) {
        return 0.0;
    }

    // If the near intersection is behind the origin, clamp it to 0.0 so we start from the ray origin.
    if (tMin < 0.0) {
        tMin = 0.0;
    }

    // If tMax is behind the origin, the entire box is behind the viewer => no density.
    if (tMax < 0.0) {
        return 0.0;
    }

    // 2. Compute the distance we need to march inside the box.
    let totalDist = tMax - tMin;
    if (totalDist <= 0.0) {
        return 0.0;
    }

    // Determine how many steps we can take with this step size.
    let numStepsF = floor(totalDist / stepSize);
    let numSteps = max(i32(numStepsF), 0);

    var accumulatedDensity = 0.0;

    // 3. March in constant increments from tMin to tMax.
    for (var i = 0; i < numSteps; i = i + 1) {
        let t = tMin + f32(i) * stepSize;
        let samplePos = rayOrigin + rayDir * t;
        // Convert position to integral cell coordinates by flooring (nearest-lower cell).
        let cellCoord = vec3i(floor(samplePos));
        let idx = GetIdxFromPos(cellCoord);
        accumulatedDensity += sim.grid[idx].density * stepSize;
    }

    return accumulatedDensity;
}

//project to clip space
fn Project(cam: Camera, p: float3) -> float3
{
    let td = distance(cam.pos, p);
    let dir = (p - cam.pos)/td;
    let screen = dir*cam.cam;
    return float3(screen.xy*cam.size.y/(cam.fov*screen.z) + 0.5*cam.size,screen.z*td);
}

fn RayFromPixel(cam: Camera, uv: float2) -> Ray 
{
    let p = (uv + vec2f(0.5, 0.0) - 0.5 * cam.size) * (1.0 / cam.size.y);
    let rayDirWorld = normalize(cam.cam * float3(p.x, p.y, cam.fov));
    return Ray(cam.pos, rayDirWorld);
}

struct Ellipsoid {
    c: vec3f, // center
    r: vec3f, // radii
    q: vec4f, // quaternion
};

struct ProjEllipsoidResult {
    center : vec2f,
    axis1  : vec2f,
    axis2  : vec2f,
    size   : vec2f,
    depth : f32,
    projectionFailed : bool,
};

fn parametricEllipse(proj: ProjEllipsoidResult, t: f32) -> vec2f {
    let angle = t;
    let cosA  = cos(angle);
    let sinA  = sin(angle);

    // x = center + (axis1 * (size.x * cos(t))) + (axis2 * (size.y * sin(t)))
    let point = proj.center
               + proj.axis1 * (proj.size.x * cosA)
               + proj.axis2 * (proj.size.y * sinA);

    return point;
}

fn ellipseLineCoefficients(proj: ProjEllipsoidResult, y0: f32) -> vec3f {
    let cx = proj.center.x;
    let cy = proj.center.y;

    let a1x = proj.axis1.x;
    let a1y = proj.axis1.y;
    let a2x = proj.axis2.x;
    let a2y = proj.axis2.y;

    let sx = proj.size.x;
    let sy = proj.size.y;

    let d = y0 - cy;

    let alpha = (a1x * a1x) / (sx * sx) + (a2x * a2x) / (sy * sy);
    let beta  = 2.0 * ((a1x * a1y) / (sx * sx) + (a2x * a2y) / (sy * sy)) * d;
    let gamma = ((a1y * a1y) / (sx * sx) + (a2y * a2y) / (sy * sy)) * (d * d);

    return vec3f(alpha, beta - 2.0 * alpha * cx, alpha * cx * cx - beta * cx + gamma - 1.0);
}

fn solveQuadratic(a: f32, b: f32, c: f32) -> vec2f {
    let disc = b*b - 4.0*a*c;
    if (disc < 0.0) {
        return vec2f(0.0, 0.0);
    }

    let sqrt_disc = sqrt(disc);
    let x1 = (-b - sqrt_disc) / (2.0 * a);
    let x2 = (-b + sqrt_disc) / (2.0 * a);
    return vec2f(x1, x2);
}

fn ellipseLineMinMaxX(y: f32, proj: ProjEllipsoidResult) -> vec2f {
    let abc = ellipseLineCoefficients(proj, y);
    return solveQuadratic(abc.x, abc.y, abc.z);
}

struct Intersection {
    t: f32,
    normal: vec3f,
};

fn intersectEllipsoid(r: Ray, ell: Ellipsoid) -> Intersection {
    // Transform the ray into ellipsoid unit space.
    let ro = posToUnitSpace(r.ro, ell);
    let rd = vecToUnitSpace(r.rd, ell);

    // Ray-sphere intersection
    let a = dot(rd, rd);
    let b = 2.0 * dot(rd, ro);
    let c = dot(ro, ro) - 1.0;
    
    let solution = solveQuadratic(a, b, c);
    let t = min(solution.x, solution.y);
    if (t <= 0.0) {
        return Intersection(0.0, vec3f(0.0));
    }

    // Compute the normal at the intersection point.
    let normal = normalize(qrotate((ro + t * rd) / ell.r, ell.q));

    return Intersection(t, normal);
}


// A safe divide that mimics your Python safe_divide
fn safe_divide(a: f32, b: f32) -> f32 {
    let epsilon = 1e-2;
    let abs_b = abs(b);
    if abs_b < epsilon {
        // emulate: a / (epsilon * sign(b))
        return a / (epsilon * sign(b));
    }
    return a / b;
}

fn safe_sqrt(x: f32) -> f32 { return sqrt(max(x, 0.0)); }


fn axis_angle_to_quaternion(axis_angle: vec3f) -> vec4f {
    let angle = length(axis_angle);
    let safeAngle = select(angle, 1e-6, angle < 1e-6);
    let axis = axis_angle / safeAngle;
    let half = 0.5 * angle;
    let s    = sin(half);
    let c    = cos(half);
    return vec4f(c, axis.x * s, axis.y * s, axis.z * s);
}

fn qrotate(v: vec3f, q: vec4f) -> vec3f {
    let t = 2.0 * cross(q.yzw, v);
    return v + q.x * t + cross(q.yzw, t);
}

fn conj(q:vec4f) -> vec4f {
    return vec4f(-q.x, q.yzw);
}

fn vecToUnitSpace(v: vec3f, e: Ellipsoid) -> vec3f {
    return qrotate(v, conj(e.q)) / e.r;
}

fn vecToWorldSpace(v: vec3f, e: Ellipsoid) -> vec3f {
    return qrotate(v * e.r, e.q);
}

fn posToUnitSpace(p: vec3f, e: Ellipsoid) -> vec3f {
    return vecToUnitSpace(p - e.c, e);
}

fn posToWorldSpace(p: vec3f, e: Ellipsoid) -> vec3f {
    return vecToWorldSpace(p, e) + e.c;
}

// Extract ellipse from conic parameters (a, b, c, d, e, f)
fn extractEllipse(a: f32, b: f32, c: f32, d: f32, e: f32, f0: f32) -> ProjEllipsoidResult {

    var result = ProjEllipsoidResult();
    result.projectionFailed = true;

    let delta = c*c - 4.0*a*b;
    if (delta >= 0.0) {
        return result;
    }

    // Compute center (h, k)
    let h = safe_divide(2.0*b*d - c*e, delta);
    let k = safe_divide(2.0*a* e - c*d, delta);

    // Plug (h,k) back into the conic to find F'
    let F_prime = a*h*h + b*k*k + c*h*k + d*h + e*k + f0;

    let diff_ba = b - a;
    let sum_ba  = b + a;
    let J       = sqrt(diff_ba * diff_ba + c*c);
    let lambda1 = 0.5 * (sum_ba + J);
    let lambda2 = 0.5 * (sum_ba - J);

    // Angles for the ellipse axes
    let r  = safe_divide(diff_ba, c);
    let ca = 0.5 * sign(c) / sqrt(1.0 + r*r);
    let ch = sqrt(0.5 + ca) * sqrt(0.5);
    let sh = sqrt(0.5 - ca) * sqrt(0.5) * sign(diff_ba);

    let cos_theta = ch - sh;
    let sin_theta = ch + sh;

    let denom1 = -safe_divide(F_prime, lambda1);
    let denom2 = -safe_divide(F_prime, lambda2);

    if (denom1 <= 0.0 || denom2 <= 0.0) {
        return result;
    }

    let a1 = sqrt(denom1);
    let a2 = sqrt(denom2);

    result.center           = vec2f(h, k);
    result.axis1            = vec2f(cos_theta, sin_theta);
    result.axis2            = vec2f(-sin_theta, cos_theta);
    result.size             = vec2f(a1, a2);
    result.projectionFailed = false;

    return result;
}

// Project an ellipsoid into screen-space (building conic parameters, then extracting ellipse).
fn projEllipsoid(ellips: Ellipsoid, cam: Camera) -> ProjEllipsoidResult {
    // 1) Clamp radii
    let rad_max    = max(ellips.r.x, max(ellips.r.y, ellips.r.z));
    let rad_clamped = clamp(
        ellips.r,
        vec3f(rad_max * 0.01, rad_max * 0.01, rad_max * 0.01),
        vec3f(rad_max,       rad_max,       rad_max)
    );
    let e0 = Ellipsoid(ellips.c, rad_clamped, ellips.q);

    let v0 = vecToUnitSpace(cam.cam[0], e0);
    let v1 = vecToUnitSpace(cam.cam[1], e0);
    let v2 = vecToUnitSpace(cam.cam[2], e0);
    let ro = posToUnitSpace(cam.pos, e0);
    let focal_length = 1.0 / cam.fov;

    let s  = dot(ro, ro) - 1.0;
    let vv0 = ro * dot(v0, ro) - v0 * s;
    let vv1 = ro * dot(v1, ro) - v1 * s;
    let vv2 = ro * dot(v2, ro) - v2 * s;

    let a = dot(v0, vv0);
    let b = dot(v1, vv1);
    let c = 2.0 * dot(v0, vv1);  
    let d = 2.0 * focal_length * dot(v0, vv2);
    let e = 2.0 * focal_length * dot(v1, vv2);
    let f = focal_length * focal_length * dot(v2, vv2);

    var ellipse = extractEllipse(a, b, c, d, e, f);
    let min_res = min(cam.size.x, cam.size.y);
    let i = ellipse.center.x * min_res + 0.5 * cam.size.x;
    let j = ellipse.center.y * min_res + 0.5 * cam.size.y;
    ellipse.center = vec2f(i, j);
    ellipse.size = ellipse.size * min_res;
    ellipse.depth = dot(cam.cam[2], e0.c - cam.pos);

    return ellipse;
}


fn Pack(a: uint, b: uint) -> int
{
    return int(a + (b << (31u - DEPTH_BITS)));
}

fn Unpack(a: int) -> float
{
    let mask = (1 << (DEPTH_BITS - 1u)) - 1;
    return float(a & mask)/256.0;
}

fn UnpackDepth(a: int) -> int
{
    return a >> (31u - DEPTH_BITS);
}

fn ClosestPoint(color: float3, depth: float, index: int)
{
    let inverseDepth = 1.0/depth;
    let scaledDepth = (inverseDepth - 1.0/DEPTH_MAX)/(1.0/DEPTH_MIN - 1.0/DEPTH_MAX);
    
    if(scaledDepth > 1.0 || scaledDepth < 0.0) 
    {
        return;
    }

    let uintDepth = uint(scaledDepth*float((1u << DEPTH_BITS) - 1u));
    let uintColor = uint3(color * 256.0);

    // Load prev depth
    let packed = atomicLoad(&gstate.screen[index*4+0]);
    let prevDepth = UnpackDepth(packed);

    if(uintDepth < u32(prevDepth)) { return; }
    
    atomicMax(&gstate.screen[index*4+0], Pack(uintColor.x, uintDepth));
    atomicMax(&gstate.screen[index*4+1], Pack(uintColor.y, uintDepth));
    atomicMax(&gstate.screen[index*4+2], Pack(uintColor.z, uintDepth));
}

fn AdditiveBlend(color: float3, depth: float, index: int)
{
    let scaledColor = 256.0 * color/depth;

    atomicAdd(&gstate.screen[index*4+0], int(scaledColor.x));
    atomicAdd(&gstate.screen[index*4+1], int(scaledColor.y));
    atomicAdd(&gstate.screen[index*4+2], int(scaledColor.z));
}

fn RasterizePixel(color: float3, depth: float, pos: int2)
{
    let screen_size = int2(camera.size);
    if(pos.x < 0 || pos.x >= screen_size.x || 
        pos.y < 0 || pos.y >= screen_size.y || depth < 0.0)
    {
        return;
    }

    let idx = pos.x + screen_size.x * pos.y;
    
    if(custom.RenderMode < 0.5)
    {
        AdditiveBlend(color, depth, idx);
    }
    else
    {
        ClosestPoint(color, depth, idx);
    }
}

fn RasterizeEllipsoid(color: float3, ellipsoid: Ellipsoid)
{
    let projection = projEllipsoid(ellipsoid, camera);

    if(projection.projectionFailed) { return; }

    // Get the bounding box of the ellipse
    let u = projection.axis1 * projection.size.x;
    let v = projection.axis2 * projection.size.y;
    let size = sqrt(u * u + v * v);
    let bbox_min = projection.center - size;
    let bbox_max = projection.center + size;

    // Skip if bbox is too large
    let area = projection.size.x * projection.size.y * PI;
    if(area > MAX_RASTER_AREA) { return; }

    // Rasterize the ellipse
    let imin_y = max(int(floor(bbox_min.y)), 0);
    let imax_y = min(int(ceil(bbox_max.y)), int(camera.size.y));
    for (var j = imin_y; j < imax_y; j++) {
        let bounds = ellipseLineMinMaxX(f32(j), projection);
        let imin_x = max(int(floor(bounds.x)), 0);
        let imax_x = min(int(ceil(bounds.y)), int(camera.size.x));
        for (var i = imin_x; i < imax_x; i++) {
            let p = vec2f(f32(i), f32(j));
            let ray = RayFromPixel(camera, p);
            let intersection = intersectEllipsoid(ray, ellipsoid);
            if (intersection.t > 0.0) 
            {   
                var pix_color = color;
                if(custom.RenderMode > 0.5)  {
                   pix_color *= (dot(intersection.normal, vec3f(1.0, 0.0, 0.0)) * 0.15 + 0.85);
                }
                //pix_color = intersection.normal * 0.5 + 0.5;
                RasterizePixel(pix_color, intersection.t, int2(p.xy));
            } 
            else // misses
            {
                 //RasterizePixel(vec3f(1.0, 0.0, 0.0), projection.depth, int2(p.xy));
            }
        }
    }
}

#storage gstate GlobalState

struct Camera 
{
  pos: float3,
  cam: float3x3,
  fov: float,
  size: float2
}

struct Ray
{
    ro: float3,
    rd: float3,
}

struct GlobalState
{
    initialized: uint,

    pos: float3,
    dposdt: float3,

    rot: float4,
    drotdt: float3,

    mouse: float2,
    dmousedt: float2,

    prevpos: float3,
    prevrot: float4,
    prevmouse: float2,

    camera: float3x3,
    fov: float,

    simulation_on: int,

    screen: array<atomic<i32>>
}

var<private> camera : Camera;

fn matrixCompMult(a: float3x3, b: float3x3) -> float3x3 
{
    return float3x3(a[0] * b[0], a[1] * b[1], a[2] * b[2]);
}

fn outerProduct(a: float3, b: float3) -> float3x3 
{
    return float3x3(a[0] * b, a[1] * b, a[2] * b);
}

// Return quaternion from axis and angle
fn aa2q(axis: float3, ang: float) -> float4 
{
    let g: float2 = float2(sin(ang), cos(ang)) * 0.5;
    return normalize(float4(axis * g.x, g.y));
}

// Return AxisAngle of NORMALIZED quaternion input
fn q2aa(q: float4) -> float4 
{
    return float4(q.xyz / sqrt(1.0 - q.w * q.w), acos(q.w) * 2.0);
}

// Return q2, rotated by q1, order matters (is non commutative) : (aka quaternion multiplication == AxisAngleRotation)
fn qq2q(q1: float4, q2: float4) -> float4
{
    return float4(q1.xyz * q2.w + q2.xyz * q1.w + cross(q1.xyz, q2.xyz), (q1.w * q2.w) - dot(q1.xyz, q2.xyz));
}

fn float3x3f(value: f32) -> float3x3
{
    return float3x3(
        float3(value, 0.0, 0.0),
        float3(0.0, value, 0.0),
        float3(0.0, 0.0, value)
    );
}

fn float3x3d(value: f32) -> float3x3
{
    return float3x3(
        float3(value, value, value),
        float3(value, value, value),
        float3(value, value, value),
    );
}

fn quaternion_to_matrix(quat: float4) -> float3x3 {
    var m: float3x3 = float3x3(
        float3(0.0, 0.0, 0.0),
        float3(0.0, 0.0, 0.0),
        float3(0.0, 0.0, 0.0)
    );

    let x = quat.x; let y = quat.y; let z = quat.z; let w = quat.w;
    let x2 = x + x;  let y2 = y + y;  let z2 = z + z;
    let xx = x * x2; let xy = x * y2; let xz = x * z2;
    let yy = y * y2; let yz = y * z2; let zz = z * z2;
    let wx = w * x2; let wy = w * y2; let wz = w * z2;

    m[0] = float3(1.0 - (yy + zz), xy - wz, xz + wy);
    m[1] = float3(xy + wz, 1.0 - (xx + zz), yz - wx);
    m[2] = float3(xz - wy, yz + wx, 1.0 - (xx + yy));

    return m;
}

#define INITIALIZED_OK 100

#workgroup_count UpdateGlobalState 1 1 1
@compute @workgroup_size(1)
fn UpdateGlobalState() 
{
    let dt: f32 = 1.0;
    let speed: f32 = 0.025;
    let mouse_sens: f32 = 0.005;
    let roll_speed: f32 = 0.005;
    
    gstate.fov = 1.0;

    if(gstate.initialized != INITIALIZED_OK)
    {
        gstate.pos = float3(0.5, -0.0, -1.5);
        gstate.dposdt = float3(0.0);
        gstate.rot = float4(0.0, 0.0, 0.0, 1.0);
        gstate.drotdt = float3(0.0);
        gstate.mouse = float2(0.0);
        gstate.dmousedt = float2(0.0);
        gstate.prevpos = float3(0.0);
        gstate.prevrot = float4(0.0, 0.0, 0.0, 1.0);
        gstate.prevmouse = float2(0.0);
        gstate.camera = float3x3f(0.0);
        gstate.simulation_on = 1;
        gstate.initialized = INITIALIZED_OK;
    }

    // Get velocities
    var cv: float3 = gstate.dposdt;
    let cav: float3 = gstate.drotdt;

    // Update position
    if(keyDown(87u)) { cv += speed * dt * gstate.camera * float3(0.0,0.0,1.0); } // W
    if(keyDown(83u)) { cv += speed * dt * gstate.camera * float3(0.0,0.0,-1.0); } // S
    if(keyDown(65u)) { cv += speed * dt * gstate.camera * float3(-1.0,0.0,0.0); } // A
    if(keyDown(68u)) { cv += speed * dt * gstate.camera * float3(1.0,0.0,0.0); } // D

    gstate.pos += dt * cv;
    cv += -cv * tanh(100.0 * dt);

    // Update camera orientation
    let dmouse: float2 = dt * mouse_sens * (float2(mouse.pos) - gstate.prevmouse);

    if(length(dmouse) < 0.1)
    {
        // Rotate around y axis
        gstate.rot = qq2q(gstate.rot, aa2q(gstate.camera * float3(0,1,0), -dmouse.x));
        // Rotate around x axis
        gstate.rot = qq2q(gstate.rot, aa2q(gstate.camera * float3(1,0,0), dmouse.y));
    }

   // Roll camera
    if(keyDown(81u)) { gstate.rot = qq2q(gstate.rot, aa2q(gstate.camera * float3(0,0,1), roll_speed * dt)); } // Q
    if(keyDown(69u)) { gstate.rot = qq2q(gstate.rot, aa2q(gstate.camera * float3(0,0,1), -roll_speed * dt)); } // E

    // Turn on and off simulation (M key)
    if(keyDown(77u)) { gstate.simulation_on = 1 - gstate.simulation_on; }

    gstate.rot = normalize(gstate.rot);
    gstate.dposdt = cv;

    // Update previous states
    gstate.prevpos = gstate.pos;
    gstate.prevrot = gstate.rot;
    gstate.prevmouse = float2(mouse.pos);

    // Update camera orientation
    gstate.camera = quaternion_to_matrix(gstate.rot);
}