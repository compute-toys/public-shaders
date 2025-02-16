// Pan around by clicking and dragging with the mouse or using the WASD
// keys. Zoom in and out with the plus/minus or E/Q keys
//
// 2D O(n log n) Barnes-Hut n-body gravity simulation using a mipmap-like
// quadtree instead of a pointer-based quadtree. Uses leapfrog integration.
// 
// General steps are:
//  1. Do the first "kick" and "drift" for leapfrog integration
//  2. Calculate the acceleration for each body by constructing and using the
//     quadtree
//      a) Calculate the bounding box for the bodies to determine the size of
//         the quadtree
//      b) Atomically splat the bodies into the quadtree leaves
//      c) Fill in the rest of the quadtree levels as you would a mipmap
//      d) Do a depth first traversal of the quadtree in the regular Barnes-Hut
//         fashion to calculate the net force for each body
//  3. Do the final "kick" for leapfrog integration
//  4. Splat the bodies onto the screen for presentation
// 
// References:
//   2D O(N²) N-body Simulation - Zi7ar21
//   https://compute.toys/view/80
// 
//   float atomic add - lomateron
//   https://compute.toys/view/88
//
//   matplotlib colormaps + turbo - mattz
//   https://www.shadertoy.com/view/3lBXR3
//
//   Pan & Zoom - fadaaszhi
//   https://compute.toys/view/677

// Try this if you can:
// const PARTICLE_WIDTH = 1;
// const N_BODIES = 1 << 20;
// Make sure to restart the shader if you change any parameters!
const PARTICLE_WIDTH = 2;
const N_BODIES = 1 << 18;
#define QT_HEIGHT 11
const PI = 3.1415927;
const MAX_FLOAT = bitcast<f32>(0x7f7fffff);

struct PanZoom {
    top_left: vec2f,
    scale: f32,
    _mouse: Mouse,
    _dt: f32,
}

struct Body {
    m: f32,
    x: vec2f,
    v: vec2f,
    a: vec2f,
}

#define BOUNDS_WG_COUNT 64

struct Global {
    pz: PanZoom,
    previous_time_elapsed: f32,
    delta_time: f32,
    bodies: array<Body, N_BODIES>,
    wg_x_min: array<vec2f, BOUNDS_WG_COUNT>,
    wg_x_max: array<vec2f, BOUNDS_WG_COUNT>,
    x_min: vec2f,
    x_max: vec2f,
    splat: array<atomic<u32>>,
}

#storage g Global
#storage quadtree array<vec3f>
#define dt g.delta_time
#define bodies g.bodies

fn pan_and_zoom() {
    if time.frame == 0 {
        g.pz.top_left = vec2f(0.0);
        g.pz.scale = 1.0;
        g.pz._dt = 1.0 / 60.0;
    } else {
        g.pz._dt = select(g.pz._dt, dt, dt > 0.0);

        if g.pz._mouse.click == 1 && mouse.click == 1 {
            let delta_mouse_pos = vec2f(mouse.pos) - vec2f(g.pz._mouse.pos);
            g.pz.top_left -= delta_mouse_pos * g.pz.scale;
        }

        let zoom_direction = 
            f32(keyDown(81) || keyDown(189)) -
            f32(keyDown(69) || keyDown(187));
        let zoom_factor = pow(
            2.0,
            10.0 * custom.zoom_speed * zoom_direction * g.pz._dt,
        );
        g.pz.top_left -=
            (vec2f(mouse.pos) + 0.5) * g.pz.scale * (zoom_factor - 1.0);
        g.pz.scale *= zoom_factor;
        let screen_height = f32(textureDimensions(screen).y);
        g.pz.top_left += vec2f(
            f32(keyDown(68)) - f32(keyDown(65)),
            f32(keyDown(83)) - f32(keyDown(87))
        ) * 10.0 * custom.pan_speed * g.pz.scale * screen_height * g.pz._dt;
    }

    g.pz._mouse = mouse;
}

fn to_world(screen_coords: vec2f) -> vec2f {
    let screen_size = vec2f(textureDimensions(screen));
    let p = g.pz.top_left + g.pz.scale * screen_coords;
    return (2.0 * p - screen_size) / screen_size.y;
}

fn to_screen(world_coords: vec2f) -> vec2f {
    let screen_size = vec2f(textureDimensions(screen));
    let p = (world_coords * screen_size.y + screen_size) / 2.0;
    return (p - g.pz.top_left) / g.pz.scale;
}

fn atomic_float_add(index: u32, value: f32) {
    let a = &g.splat[index];
    var old = atomicLoad(a);

    loop {
        var n = bitcast<u32>(bitcast<f32>(old) + value);
        var r = atomicCompareExchangeWeak(a, old, n);

        if r.exchanged {
            break;
        }

        old = r.old_value;
    }
}

#workgroup_count update_globals 1 1 1
@compute @workgroup_size(1)
fn update_globals() {
    if time.frame == 0 {
        dt = 1.0 / 60.0;
    } else {
        dt = time.elapsed - g.previous_time_elapsed;
    }

    pan_and_zoom();
    dt = custom.time_scale * min(dt, 1.0 / 30.0);
    g.previous_time_elapsed = time.elapsed;
}

var<private> rng_state: u32;

fn rand1u() -> u32 {
    let state = rng_state * 747796405 + 2891336453;
    let word = ((state >> ((state >> 28) + 4)) ^ state) * 277803737;
    rng_state = (word >> 22u) ^ word;
    return rng_state;
}

fn rand1f() -> f32 {
    return f32(rand1u()) / f32(0xffffffffu);
}

// Initialize the position, velocity and mass of each body in the first frame
#dispatch_once initialize_bodies
#workgroup_count initialize_bodies 64 1 1
@compute @workgroup_size(256)
fn initialize_bodies(
    @builtin(global_invocation_id) id: vec3u,
    @builtin(num_workgroups) nwgs: vec3u,
) {
    for (
        var body_index = id.x;
        body_index < N_BODIES;
        body_index += 256 * nwgs.x * nwgs.y * nwgs.z
    ) {
        rng_state = body_index;
        let b = &bodies[body_index];
        let theta = rand1f() * 2.0 * PI;
        (*b).m = 0.35 / f32(N_BODIES);
        (*b).x = vec2f(cos(theta), sin(theta)) * rand1f();
        (*b).v = 0.6 * vec2f(-(*b).x.y, (*b).x.x) / (dot((*b).x, (*b).x) + 0.2);
        (*b).a = vec2f(0.0);
    }
}

// Integrate velocity and position for the first "kick-drift" of leapfrog
#workgroup_count kick_drift 64 1 1
@compute @workgroup_size(256)
fn kick_drift(
    @builtin(global_invocation_id) id: vec3u,
    @builtin(num_workgroups) nwgs: vec3u,
) {
    for (
        var body_index = id.x;
        body_index < N_BODIES;
        body_index += 256 * nwgs.x * nwgs.y * nwgs.z
    ) {
        let b = &bodies[body_index];
        (*b).v += (*b).a * dt / 2.0;
        (*b).x += (*b).v * dt;
    }
}

// Each workgroup computes the bounds for a chunk of bodies using shared memory
const calculate_bounds0_wgs = 256;
var<workgroup> wg_x_min: array<vec2f, calculate_bounds0_wgs>;
var<workgroup> wg_x_max: array<vec2f, calculate_bounds0_wgs>;

#workgroup_count calculate_bounds0 BOUNDS_WG_COUNT 1 1
@compute @workgroup_size(calculate_bounds0_wgs)
fn calculate_bounds0(
    @builtin(global_invocation_id) id: vec3u,
    @builtin(num_workgroups) nwgs: vec3u,
    @builtin(workgroup_id) wg_id: vec3u,
    @builtin(local_invocation_index) index: u32,
) {
    var x_min = vec2f(MAX_FLOAT);
    var x_max = vec2f(-MAX_FLOAT);
    
    for (
        var body_index = id.x;
        body_index < N_BODIES;
        body_index += 256 * nwgs.x * nwgs.y * nwgs.z
    ) {
        let x = bodies[body_index].x;
        x_min = min(x, x_min);
        x_max = max(x, x_max);
    }

    wg_x_min[index] = x_min;
    wg_x_max[index] = x_max;
    workgroupBarrier();

    for (var i = u32(calculate_bounds0_wgs) / 2; i > 0; i /= 2) {
        if index < i {
            wg_x_min[index] = min(wg_x_min[index], wg_x_min[index + i]);
            wg_x_max[index] = max(wg_x_max[index], wg_x_max[index + i]);
        }

        workgroupBarrier();
    }

    if index == 0 {
        g.wg_x_min[wg_id.x] = wg_x_min[0];
        g.wg_x_max[wg_id.x] = wg_x_max[0];
    }
}

// Compute the bounds of all the bodies using the previous pass's results
#workgroup_count calculate_bounds1 1 1 1
@compute @workgroup_size(1)
fn calculate_bounds1() {
    var x_min = vec2f(MAX_FLOAT);
    var x_max = vec2f(-MAX_FLOAT);

    for (var i = 0; i < BOUNDS_WG_COUNT; i++) {
        x_min = min(g.wg_x_min[i], x_min);
        x_max = max(g.wg_x_max[i], x_max);
    }

    g.x_min = x_min;
    g.x_max = x_max;
}

// Clear the splat buffer in preparation for making the quadtree
#workgroup_count clear_splat 64 1 1
@compute @workgroup_size(256)
fn clear_splat(
    @builtin(global_invocation_id) id: vec3u,
    @builtin(num_workgroups) nwgs: vec3u,
) {
    for (
        var i = id.x;
        i < 1 << (2 * QT_HEIGHT);
        i += 256 * nwgs.x * nwgs.y * nwgs.z
    ) {
        atomicStore(&g.splat[3 * i + 0], bitcast<u32>(0.0));
        atomicStore(&g.splat[3 * i + 1], bitcast<u32>(0.0));
        atomicStore(&g.splat[3 * i + 2], bitcast<u32>(0.0));
    }
}

fn qt_coords_to_index(depth: u32, coords: vec2u) -> u32 {
    return coords.x + (coords.y << depth) + ((1u << (2 * depth)) - 1) / 3;
}

// Splat each body's mass and weighted position into the quadtree leaves
#workgroup_count splat_bodies 64 1 1
@compute @workgroup_size(256)
fn splat_bodies(
    @builtin(global_invocation_id) id: vec3u,
    @builtin(num_workgroups) nwgs: vec3u,
) {
    for (
        var body_index = id.x;
        body_index < N_BODIES;
        body_index += 256 * nwgs.x * nwgs.y * nwgs.z
    ) {
        let b = bodies[body_index];
        let x_normalized = (b.x - g.x_min) / (g.x_max - g.x_min);
        var coords = min(
            vec2u(x_normalized * pow(2.0, f32(QT_HEIGHT))),
            vec2u(1 << QT_HEIGHT) - 1
        );
        let mx_m = vec3f(b.m * b.x, b.m);
        let i = coords.x + (coords.y << QT_HEIGHT);
        atomic_float_add(3 * i + 0, mx_m[0]);
        atomic_float_add(3 * i + 1, mx_m[1]);
        atomic_float_add(3 * i + 2, mx_m[2]);
    }
}

// Copy over the atomic splat into a regular non-atomic array for performance
#workgroup_count copy_splat_to_quadtree 64 1 1
@compute @workgroup_size(256)
fn copy_splat_to_quadtree(
    @builtin(global_invocation_id) id: vec3u,
    @builtin(num_workgroups) nwgs: vec3u,
) {
    for (
        var i = id.x;
        i < 1 << (2 * QT_HEIGHT);
        i += 256 * nwgs.x * nwgs.y * nwgs.z
    ) {
        let coords = vec2u(i & ((1 << QT_HEIGHT) - 1), i >> QT_HEIGHT);
        quadtree[qt_coords_to_index(QT_HEIGHT, coords)] = vec3f(
            bitcast<f32>(atomicLoad(&g.splat[3 * i + 0])),
            bitcast<f32>(atomicLoad(&g.splat[3 * i + 1])),
            bitcast<f32>(atomicLoad(&g.splat[3 * i + 2]))
        );
    }
}

// Calculate the rest of the levels of the quadtree as the sum of its children
#dispatch_count calculate_quadtree QT_HEIGHT
#workgroup_count calculate_quadtree 64 1 1
@compute @workgroup_size(256)
fn calculate_quadtree(
    @builtin(global_invocation_id) id: vec3u,
    @builtin(num_workgroups) nwgs: vec3u,
) {
    let depth = QT_HEIGHT - dispatch.id - 1;

    for (
        var i = id.x;
        i < 1u << (2 * depth);
        i += 256 * nwgs.x * nwgs.y * nwgs.z
    ) {
        let coords = vec2u(i & ((1u << depth) - 1), i >> depth);
        quadtree[qt_coords_to_index(depth, coords)] =
            quadtree[qt_coords_to_index(depth + 1, 2 * coords + vec2u(0, 0))] +
            quadtree[qt_coords_to_index(depth + 1, 2 * coords + vec2u(1, 0))] +
            quadtree[qt_coords_to_index(depth + 1, 2 * coords + vec2u(0, 1))] +
            quadtree[qt_coords_to_index(depth + 1, 2 * coords + vec2u(1, 1))];
    }
}

// Calculate the net force/acceleration for each body by traversing the quadtree
#workgroup_count calculate_accelerations 64 1 1
@compute @workgroup_size(256)
fn calculate_accelerations(
    @builtin(global_invocation_id) id: vec3u,
    @builtin(num_workgroups) nwgs: vec3u,
) {
    const G = 1.0;

    for (
        var body_index = id.x;
        body_index < N_BODIES;
        body_index += 256 * nwgs.x * nwgs.y * nwgs.z
    ) {
        let b = &bodies[body_index];
        (*b).a = vec2f(0.0);

        if false {
            // O(N^2) way
            for (var i = 0; i < N_BODIES; i++) {
                let b1 = bodies[i];
                let r = b1.x - (*b).x;
                
                if all(r == vec2f(0.0)) {
                    continue;
                }

                (*b).a += normalize(r) * G * b1.m / (dot(r, r) + 0.01);
            }
        } else {
            // Barnes-Hut way
            let x_normalized = ((*b).x - g.x_min) / (g.x_max - g.x_min);
            var p = vec2u(0);
            var depth = 0u;
            var bound_size = g.x_max - g.x_min;
            let square_width = sqrt(bound_size.x * bound_size.y);

            // Depth first traversal of the quadtree
            // depth stores the current node depth and p contains the position
            // of the node within that depth of the quadtree
            loop {
                let size = f32(1 << depth);
                let xm_m = quadtree[qt_coords_to_index(depth, p)];
                let m = xm_m.z;
                let x = xm_m.xy / m;
                let s = square_width / size;
                let r = x - (*b).x;
                let d = length(r);
                let counted = depth == QT_HEIGHT || s / d < custom.theta;
                
                // Check if the Barnes-Hut criteria is satisfied in order to
                // count this quadtree cell as a single body
                if m != 0.0 && counted {
                    if any(r != vec2f(0.0) ) {
                        (*b).a += normalize(r) * G * m / (dot(r, r) + 0.001);
                    }
                }
                
                // If we did count it (or if it has no mass) then stop
                // descending at this node and move onto the next node
                if m == 0.0 || counted {
                    while p.x % 2 == 1 && p.y % 2 == 1 {
                        depth--;
                        p /= 2;
                    }
                    
                    if p.y % 2 == 0 {
                        p.y++;
                    } else {
                        p.x++;
                        p.y--;
                    }
                    
                    // Exit the loop if back at root node if we're done
                    if depth == 0 && p.y == 1 {
                        break;
                    }
                    
                    continue;
                }

                // Otherwise descend further
                depth++;
                p *= 2;
            }
        }
    }
}

// Integrate velocity for the final "drift" of leapfrog
#workgroup_count kick 64 1 1
@compute @workgroup_size(256)
fn kick(
    @builtin(global_invocation_id) id: vec3u,
    @builtin(num_workgroups) nwgs: vec3u,
) {
    for (
        var body_index = id.x;
        body_index < N_BODIES;
        body_index += 256 * nwgs.x * nwgs.y * nwgs.z
    ) {
        let b = &bodies[body_index];
        (*b).v += (*b).a * dt / 2.0;
    }
}

// Clear the splat buffer in preparation for drawing the bodies to the screen
@compute @workgroup_size(16, 16)
fn clear_splat2(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);

    if any(id.xy >= screen_size) {
        return;
    }

    atomicStore(&g.splat[id.x + id.y * screen_size.x], bitcast<u32>(0.0));
}

// Splat each body's mass onto the screen
#workgroup_count draw_bodies 64 1 1
@compute @workgroup_size(256)
fn draw_bodies(
    @builtin(global_invocation_id) id: vec3u,
    @builtin(num_workgroups) nwgs: vec3u,
) {
    let screen_size = textureDimensions(screen);

    for (
        var body_index = id.x;
        body_index < N_BODIES;
        body_index += 256 * nwgs.x * nwgs.y * nwgs.z
    ) {
        let b = bodies[body_index];
        let x = to_screen(b.x);
        let r = PARTICLE_WIDTH - 1;
        let a = b.m / f32(PARTICLE_WIDTH * PARTICLE_WIDTH);

        for (var i = -r; i <= r; i++) {
            for (var j = -r; j <= r; j++) {
                let c = vec2i(floor(x)) + vec2i(i, j);

                if all(clamp(c, vec2i(0), vec2i(screen_size - 1)) == c) {
                    atomic_float_add(u32(c.x) + u32(c.y) * screen_size.x, a);
                }
            }
        }
    }
}

fn sRGB_to_linear(color: vec3f) -> vec3f {
    let higher = pow((color + 0.055) / 1.055, vec3(2.4));
    let lower = color / 12.92;
    return select(lower, higher, color > vec3(0.04045));
}

fn inferno(t: f32) -> vec3f {
    // https://www.shadertoy.com/view/3lBXR3
    const c0 = vec3f(0.00021894037, 0.0016510046, -0.019480899);
    const c1 = vec3f(0.10651341949, 0.5639564368, 3.9327123889);
    const c2 = vec3f(11.6024930825, -3.972853966, -15.94239411);
    const c3 = vec3f(-41.703996131, 17.436398882, 44.354145199);
    const c4 = vec3f(77.1629356994, -33.40235894, -81.80730926);
    const c5 = vec3f(-71.319428245, 32.626064264, 73.209519858);
    const c6 = vec3f(25.1311262248, -12.24266895, -23.07032500);
    return sRGB_to_linear(c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6))))));
}

// Color each pixel on screen based on the mass inside it
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);

    if any(id.xy >= screen_size) {
        return;
    }

    let x = atomicLoad(&g.splat[id.x + screen_size.x * id.y]);
    var color = inferno(1.0 - exp(-bitcast<f32>(x) * 3e5 / g.pz.scale));
    textureStore(screen, id.xy, vec4f(color, 1.0));
}