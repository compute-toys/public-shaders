#include "Dave_Hoskins/hash"

// [2k+0]: pixel index -> cluster id
// [2k+1]: cluster id -> reduced cluster id
#storage cluster array<atomic<u32>>

struct State {
    last_p: f32,
    last_reset: u32,
}

#storage state State

fn connected(a: vec2u, b: vec2u) -> bool {
    return hash12(vec2f(a + b)) < custom.p;
}

fn merge_cells(a: vec2u, b: vec2u, R: vec2u) {
    let ak = dot(a, vec2u(1, R.x));
    let bk = dot(b, vec2u(1, R.x));

    // get cluster ids
    let ac = atomicLoad(&cluster[2 * ak]);
    let bc = atomicLoad(&cluster[2 * bk]);

    // get reduced cluster ids
    let acr = atomicLoad(&cluster[2 * ac + 1]);
    let bcr = atomicLoad(&cluster[2 * bc + 1]);

    // get merged cluster id
    let mc = min(acr, bcr);

    // update everything
    atomicMin(&cluster[2 * ak], mc);
    atomicMin(&cluster[2 * bk], mc);
    atomicMin(&cluster[2 * ac + 1], mc);
    atomicMin(&cluster[2 * bc + 1], mc);
}

@compute @workgroup_size(16, 16)
fn init(@builtin(global_invocation_id) id: vec3u) {
    let R = textureDimensions(screen);
    if any(id.xy >= R) { return; }

    // only reset on start-up of when `p` decreases
    if time.frame > 3 && custom.p >= state.last_p {
        return;
    }

    // every pixel is its own cluster
    let k = dot(id.xy, vec2u(1, R.x));
    atomicStore(&cluster[2 * k], k);
    atomicStore(&cluster[2 * k + 1], k);

    if all(id == vec3u(0)) {
        state.last_reset = time.frame;
    }
}

const NEIGHBORS = array(
    vec2i(1, 0),
    vec2i(-1, 0),
    vec2i(0, 1),
    vec2i(0, -1),
);

@compute @workgroup_size(16, 16)
fn walk(@builtin(global_invocation_id) id: vec3u) {
    let R = textureDimensions(screen);
    if any(id.xy >= R) { return; }

    for (var k = 0u; k < 4; k++) {
        let curr = vec2i(id.xy) + NEIGHBORS[k];

        // skip if out of bounds
        if any(curr >= vec2i(R)) || any(curr < vec2i(0)) {
            continue;
        }

        // do not merge disconnected clusters
        if !connected(id.xy, vec2u(curr)) {
            continue;
        }

        merge_cells(id.xy, vec2u(curr), R);
    }
}

@compute @workgroup_size(16, 16)
fn present(@builtin(global_invocation_id) id: vec3u) {
    let R = textureDimensions(screen);
    if any(id.xy >= R) { return; }

    if all(id == vec3(0)) {
        state.last_p = custom.p;
    }

    // reduce flickering
    if time.frame - state.last_reset < 4 {
        return;
    }

    // get cluster for the current pixel
    let k = dot(id.xy, vec2u(1, R.x));
    var c = atomicLoad(&cluster[2 * k]);

    // pick a color for the cluster
    let col = hash31(f32(c));
    textureStore(screen, id.xy, vec4f(col, 1.0));
}
