// ─────────────────────────────────────────────────────────────
//  Falling sand sim — Pressure-based water
//  Data layout per cell:
//    .x = material ID
//    .y = horizontal velocity (momentum)
//    .z = pressure (water depth above)
//    .w = (unused - could be temperature)
// ─────────────────────────────────────────────────────────────

const AIR   : f32 = 0.0;
const WATER : f32 = 1.0;
const ROCK  : f32 = 2.0;

const PRESSURE_SAMPLE_DEPTH : i32 = 16;  // how far up to look for pressure calc
const PRESSURE_FLOW_STRENGTH : f32 = 0.8;
const VELOCITY_DAMPING : f32 = 0.85;
const VELOCITY_TRANSFER : f32 = 0.6;

fn density(m: f32) -> f32 {
    if (m == AIR)   { return 0.0; }
    if (m == WATER) { return 1.0; }
    return 1e6;
}
fn movable(m: f32) -> bool { return m != ROCK; }
fn isLiquid(m: f32) -> bool { return m == WATER; }

fn hash(p: vec3<u32>) -> u32 {
    var h = (p.x * 1597334673u) ^ (p.y * 3812015801u) ^ (p.z * 2798796415u);
    h = (h ^ (h >> 16u)) * 2246822519u;
    h = (h ^ (h >> 13u)) * 3266489917u;
    return h ^ (h >> 16u);
}

fn hashf(p: vec3<u32>) -> f32 {
    return f32(hash(p) & 0xFFFFu) / 65535.0;
}

fn loadCell(p: vec2<i32>, dim: vec2<i32>) -> vec4<f32> {
    if (p.x < 0 || p.y < 0 || p.x >= dim.x || p.y >= dim.y) { 
        return vec4<f32>(ROCK, 0.0, 0.0, 0.0); 
    }
    return passLoad(0, p, 0);
}

fn storeCell(p: vec2<i32>, dim: vec2<i32>, cell: vec4<f32>) {
    if (p.x < 0 || p.y < 0 || p.x >= dim.x || p.y >= dim.y) { return; }
    passStore(0, p, cell);
}

// Sample pressure by counting water cells above (reads are safe anywhere)
fn samplePressure(p: vec2<i32>, dim: vec2<i32>) -> f32 {
    var pressure = 0.0;
    for (var dy = 1; dy <= PRESSURE_SAMPLE_DEPTH; dy++) {
        let above = loadCell(p - vec2<i32>(0, dy), dim);
        if (above.x == WATER) {
            pressure += 1.0;
        } else if (above.x == ROCK) {
            // rock caps pressure (water under rock dome)
            pressure += 2.0;
            break;
        } else {
            break; // air breaks the column
        }
    }
    return pressure;
}

// ───────────── INIT ─────────────
#dispatch_once init
@compute @workgroup_size(16,16)
fn init(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = vec2<i32>(textureDimensions(screen));
    if (i32(id.x) >= dim.x || i32(id.y) >= dim.y) { return; }
    let uv = vec2<f32>(id.xy) / vec2<f32>(dim);

    var m = AIR;

    // floor
    if (uv.y > 0.95) { m = ROCK; }

    // walls
    if (uv.x < 0.02 || uv.x > 0.98) { m = ROCK; }

    // a rocky bowl (thicker)
    let d = length(uv - vec2<f32>(0.5, 0.85));
    if (d > 0.28 && d < 0.34 && uv.y > 0.58 && uv.y < 0.92) { m = ROCK; }

    // a platform with a gap for waterfall
    if (abs(uv.y - 0.45) < 0.02 && (uv.x < 0.42 || uv.x > 0.58)) { m = ROCK; }

    // small reservoir walls on the platform
    if (abs(uv.x - 0.25) < 0.015 && uv.y > 0.35 && uv.y < 0.45) { m = ROCK; }
    if (abs(uv.x - 0.75) < 0.015 && uv.y > 0.35 && uv.y < 0.45) { m = ROCK; }

    // starting blob of water (larger)
    if (length(uv - vec2<f32>(0.5, 0.2)) < 0.12) { m = WATER; }

    passStore(0, vec2<i32>(id.xy), vec4<f32>(m, 0.0, 0.0, 0.0));
}

// ───────────── SPAWN ─────────────
@compute @workgroup_size(16,16)
fn spawn(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = vec2<i32>(textureDimensions(screen));
    let p = vec2<i32>(id.xy);
    if (p.x >= dim.x || p.y >= dim.y) { return; }

    var cell = loadCell(p, dim);

    // faucet
    let fc = vec2<i32>(dim.x / 2, 8);
    if (length(vec2<f32>(p - fc)) < 4.0 && cell.x == AIR) {
        cell = vec4<f32>(WATER, 0.0, 0.0, 0.0);
    }

    // mouse
    if (mouse.click > 0) {
        let dist = length(vec2<f32>(p - mouse.pos));
        if (dist < 8.0 && cell.x != ROCK) {
            cell = vec4<f32>(WATER, 0.0, 0.0, 0.0);
        }
    }

    storeCell(p, dim, cell);
}

// ───────────── PRESSURE PROPAGATION ─────────────
// Run before update to compute pressure field
@compute @workgroup_size(16,16)
fn pressure(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = vec2<i32>(textureDimensions(screen));
    let p = vec2<i32>(id.xy);
    if (p.x >= dim.x || p.y >= dim.y) { return; }

    var cell = loadCell(p, dim);
    
    if (cell.x == WATER) {
        // compute pressure from water column above
        cell.z = samplePressure(p, dim);
    } else {
        cell.z = 0.0;
    }
    
    storeCell(p, dim, cell);
}

// ───────────── UPDATE (Margolus 2×2) ─────────────
@compute @workgroup_size(16,16)
fn update(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = vec2<i32>(textureDimensions(screen));

    // 4-phase Margolus for better mixing
    let phase = time.frame & 3u;
    let off = vec2<i32>(i32(phase & 1u), i32(phase >> 1u));
    let base = vec2<i32>(id.xy) * 2 - off;

    if (base.x < 0 || base.y < 0 || base.x + 1 >= dim.x || base.y + 1 >= dim.y) { return; }

    // Load 2x2 block: a b
    //                 c d   (y+ is down)
    var a = loadCell(base + vec2<i32>(0, 0), dim);
    var b = loadCell(base + vec2<i32>(1, 0), dim);
    var c = loadCell(base + vec2<i32>(0, 1), dim);
    var d = loadCell(base + vec2<i32>(1, 1), dim);

    let rnd = hash(vec3<u32>(id.xy, time.frame));
    let rndf = hashf(vec3<u32>(id.xy, time.frame));
    let rndf2 = hashf(vec3<u32>(id.xy + vec2<u32>(1337u), time.frame));

    // 1. GRAVITY — vertical fall with velocity transfer
    
    // Left column (a over c)
    if (movable(a.x) && movable(c.x) && density(a.x) > density(c.x)) {
        let vel_transfer = a.y * VELOCITY_TRANSFER;
        let t = a; a = c; c = t;
        c.y = vel_transfer; // carry momentum down
    }
    
    // Right column (b over d)
    if (movable(b.x) && movable(d.x) && density(b.x) > density(d.x)) {
        let vel_transfer = b.y * VELOCITY_TRANSFER;
        let t = b; b = d; d = t;
        d.y = vel_transfer;
    }

    // 2. DIAGONAL SLIDE — with randomization
    
    if ((rnd & 2u) == 0u) {
        if (movable(a.x) && movable(d.x) && density(a.x) > density(d.x)) {
            let t = a; a = d; d = t;
            d.y += 0.3; // impart rightward velocity
        }
        if (movable(b.x) && movable(c.x) && density(b.x) > density(c.x)) {
            let t = b; b = c; c = t;
            c.y -= 0.3; // impart leftward velocity
        }
    } else {
        if (movable(b.x) && movable(c.x) && density(b.x) > density(c.x)) {
            let t = b; b = c; c = t;
            c.y -= 0.3;
        }
        if (movable(a.x) && movable(d.x) && density(a.x) > density(d.x)) {
            let t = a; a = d; d = t;
            d.y += 0.3;
        }
    }

    // 3. PRESSURE-DRIVEN HORIZONTAL FLOW
    
    // Bottom row (c ↔ d) — most important for leveling
    if (movable(c.x) && movable(d.x) && c.x != d.x) {
        let pressure_diff = c.z - d.z;
        let vel_bias = (c.y - d.y) * 0.5;
        let flow_score = pressure_diff * PRESSURE_FLOW_STRENGTH + vel_bias;
        
        // Water flows toward lower pressure
        if (c.x == WATER && d.x == AIR && (flow_score > rndf * 2.0 - 1.0 || c.z > 1.0)) {
            let vel = c.y;
            let t = c; c = d; d = t;
            d.y = vel * VELOCITY_DAMPING + 0.4; // rightward momentum
        } else if (d.x == WATER && c.x == AIR && (-flow_score > rndf * 2.0 - 1.0 || d.z > 1.0)) {
            let vel = d.y;
            let t = c; c = d; d = t;
            c.y = vel * VELOCITY_DAMPING - 0.4; // leftward momentum
        }
    }
    
    // Top row (a ↔ b) — also flows, helps with waves
    if (movable(a.x) && movable(b.x) && a.x != b.x) {
        let pressure_diff = a.z - b.z;
        let vel_bias = (a.y - b.y) * 0.5;
        let flow_score = pressure_diff * PRESSURE_FLOW_STRENGTH + vel_bias;
        
        if (a.x == WATER && b.x == AIR && (flow_score > rndf2 * 2.0 - 1.0 || a.z > 2.0)) {
            let vel = a.y;
            let t = a; a = b; b = t;
            b.y = vel * VELOCITY_DAMPING + 0.3;
        } else if (b.x == WATER && a.x == AIR && (-flow_score > rndf2 * 2.0 - 1.0 || b.z > 2.0)) {
            let vel = b.y;
            let t = a; a = b; b = t;
            a.y = vel * VELOCITY_DAMPING - 0.3;
        }
    }

    // 4. VELOCITY-DRIVEN FLOW (momentum carries water)
    
    // If water has strong horizontal velocity, it pushes harder
    // Bottom row
    if (c.x == WATER && d.x == AIR && c.y > 0.5) {
        let t = c; c = d; d = t;
        d.y = c.y * VELOCITY_DAMPING;
    } else if (d.x == WATER && c.x == AIR && d.y < -0.5) {
        let t = c; c = d; d = t;
        c.y = d.y * VELOCITY_DAMPING;
    }
    
    // Top row
    if (a.x == WATER && b.x == AIR && a.y > 0.5) {
        let t = a; a = b; b = t;
        b.y = a.y * VELOCITY_DAMPING;
    } else if (b.x == WATER && a.x == AIR && b.y < -0.5) {
        let t = a; a = b; b = t;
        a.y = b.y * VELOCITY_DAMPING;
    }

    // 5. DAMPING — velocity decays over time
    a.y *= VELOCITY_DAMPING;
    b.y *= VELOCITY_DAMPING;
    c.y *= VELOCITY_DAMPING;
    d.y *= VELOCITY_DAMPING;

    // Clamp velocities
    a.y = clamp(a.y, -2.0, 2.0);
    b.y = clamp(b.y, -2.0, 2.0);
    c.y = clamp(c.y, -2.0, 2.0);
    d.y = clamp(d.y, -2.0, 2.0);

    storeCell(base + vec2<i32>(0, 0), dim, a);
    storeCell(base + vec2<i32>(1, 0), dim, b);
    storeCell(base + vec2<i32>(0, 1), dim, c);
    storeCell(base + vec2<i32>(1, 1), dim, d);
}

// ───────────── RENDER ─────────────
@compute @workgroup_size(16,16)
fn render(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = vec2<i32>(textureDimensions(screen));
    let p = vec2<i32>(id.xy);
    if (p.x >= dim.x || p.y >= dim.y) { return; }

    let cell = passLoad(0, p, 0);
    let m = cell.x;
    let vel = cell.y;
    let pressure = cell.z;

    var col = vec3<f32>(0.02, 0.02, 0.04); // air

    if (m == WATER) {
        // Base water color
        let baseCol = vec3<f32>(0.1, 0.4, 0.9);
        
        // Darken by pressure (deeper = darker)
        let depthFactor = 1.0 - clamp(pressure / 20.0, 0.0, 0.5);
        col = baseCol * depthFactor;
        
        // Tint by velocity (moving water is lighter/foam-ish)
        let speed = abs(vel);
        col += vec3<f32>(0.15, 0.2, 0.1) * clamp(speed, 0.0, 1.0);
        
        // Surface highlight: check if air above
        let above = passLoad(0, p - vec2<i32>(0, 1), 0).x;
        if (above == AIR) {
            col += vec3<f32>(0.2, 0.25, 0.3); // surface shine
        }
    }
    
    if (m == ROCK) {
        // Textured rock using hash
        let h = hashf(vec3<u32>(u32(p.x), u32(p.y), 0u));
        col = mix(vec3<f32>(0.3, 0.28, 0.25), vec3<f32>(0.45, 0.4, 0.35), h);
    }

    textureStore(screen, p, vec4<f32>(col, 1.0));
}