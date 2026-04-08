// ─────────────────────────────────────────────────────────────
//  Falling sand style sim — Rock + Water
//  Grid lives in pass channel 0 (x component = material id)
//  Margolus neighbourhood → race-free parallel update
// ─────────────────────────────────────────────────────────────

const AIR   : f32 = 0.0;
const WATER : f32 = 1.0;
const ROCK  : f32 = 2.0;

fn density(m: f32) -> f32 {
    if (m == AIR)   { return 0.0; }
    if (m == WATER) { return 1.0; }
    return 1e6; // rock = infinite
}
fn movable(m: f32) -> bool { return m != ROCK; }

fn hash(p: vec3<u32>) -> u32 {
    var h = (p.x * 1597334673u) ^ (p.y * 3812015801u) ^ (p.z * 2798796415u);
    h = (h ^ (h >> 16u)) * 2246822519u;
    h = (h ^ (h >> 13u)) * 3266489917u;
    return h ^ (h >> 16u);
}

fn loadCell(p: vec2<i32>, dim: vec2<i32>) -> f32 {
    if (p.x < 0 || p.y < 0 || p.x >= dim.x || p.y >= dim.y) { return ROCK; }
    return passLoad(0, p, 0).x;
}
fn storeCell(p: vec2<i32>, dim: vec2<i32>, m: f32) {
    if (p.x < 0 || p.y < 0 || p.x >= dim.x || p.y >= dim.y) { return; }
    passStore(0, p, vec4<f32>(m, 0.0, 0.0, 0.0));
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

    // a rocky bowl
    let d = length(uv - vec2<f32>(0.5, 0.85));
    if (d > 0.30 && d < 0.34 && uv.y > 0.6 && uv.y < 0.9) { m = ROCK; }

    // a slanted ledge
    if (abs(uv.y - (0.4 + uv.x*0.2)) < 0.015 && uv.x < 0.6) { m = ROCK; }

    // starting blob of water
    if (length(uv - vec2<f32>(0.5, 0.15)) < 0.08) { m = WATER; }

    passStore(0, vec2<i32>(id.xy), vec4<f32>(m,0.,0.,0.));
}

// ───────────── SPAWN (mouse + faucet) ─────────────
@compute @workgroup_size(16,16)
fn spawn(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = vec2<i32>(textureDimensions(screen));
    let p   = vec2<i32>(id.xy);
    if (p.x >= dim.x || p.y >= dim.y) { return; }

    var m = loadCell(p, dim);

    // constant faucet top-centre
    let fc = vec2<i32>(dim.x/2, 8);
    if (length(vec2<f32>(p - fc)) < 3.0 && m == AIR) { m = WATER; }

    // mouse pours water
    if (mouse.click > 0) {
        if (length(vec2<f32>(p - mouse.pos)) < 6.0 && m != ROCK) { m = WATER; }
    }

    storeCell(p, dim, m);
}

// ───────────── UPDATE (Margolus 2×2) ─────────────
@compute @workgroup_size(16,16)
fn update(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = vec2<i32>(textureDimensions(screen));

    // alternate the 2×2 lattice offset each frame
    let off  = i32(time.frame & 1u);
    let base = vec2<i32>(id.xy) * 2 - vec2<i32>(off, off);

    if (base.x >= dim.x || base.y >= dim.y) { return; }

    //  a b
    //  c d      (+y is down on screen)
    var a = loadCell(base + vec2<i32>(0,0), dim);
    var b = loadCell(base + vec2<i32>(1,0), dim);
    var c = loadCell(base + vec2<i32>(0,1), dim);
    var d = loadCell(base + vec2<i32>(1,1), dim);

    let rnd = hash(vec3<u32>(id.xy, time.frame));

    // 1. vertical fall
    if (movable(a) && movable(c) && density(a) > density(c)) { let t=a; a=c; c=t; }
    if (movable(b) && movable(d) && density(b) > density(d)) { let t=b; b=d; d=t; }

    // 2. diagonal slide (randomise order to avoid bias)
    if ((rnd & 2u) == 0u) {
        if (movable(a) && movable(d) && density(a) > density(d)) { let t=a; a=d; d=t; }
        if (movable(b) && movable(c) && density(b) > density(c)) { let t=b; b=c; c=t; }
    } else {
        if (movable(b) && movable(c) && density(b) > density(c)) { let t=b; b=c; c=t; }
        if (movable(a) && movable(d) && density(a) > density(d)) { let t=a; a=d; d=t; }
    }

    // 3. horizontal flow along bottom row (makes water level out)
    if ((rnd & 1u) == 0u && movable(c) && movable(d) && c != d) {
        if (c == WATER && d == AIR) { let t=c; c=d; d=t; }
        else if (d == WATER && c == AIR) { let t=c; c=d; d=t; }
    }

    storeCell(base + vec2<i32>(0,0), dim, a);
    storeCell(base + vec2<i32>(1,0), dim, b);
    storeCell(base + vec2<i32>(0,1), dim, c);
    storeCell(base + vec2<i32>(1,1), dim, d);
}

// ───────────── RENDER ─────────────
@compute @workgroup_size(16,16)
fn render(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = vec2<i32>(textureDimensions(screen));
    if (i32(id.x) >= dim.x || i32(id.y) >= dim.y) { return; }

    let m = passLoad(0, vec2<i32>(id.xy), 0).x;

    var col = vec3<f32>(0.03, 0.03, 0.05);            // air / background
    if (m == WATER) { col = vec3<f32>(0.15, 0.45, 0.95); }
    if (m == ROCK)  { col = vec3<f32>(0.40, 0.36, 0.32); }

    textureStore(screen, vec2<i32>(id.xy), vec4<f32>(col, 1.0));
}