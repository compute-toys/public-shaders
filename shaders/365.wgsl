const BASE: u32 = 8;

const W_CUBE = BASE * BASE;  // 64
const W_IMG = W_CUBE * BASE; // 512
const N_PIX = W_IMG * W_IMG; // 262,144

struct State {
    img: array<vec4f, N_PIX>,
    cube: array<u32, N_PIX>,
    out: array<vec4f, N_PIX>,
}

#storage state State

#workgroup_count prepare 32 32 1
@compute @workgroup_size(16, 16)
fn prepare(@builtin(global_invocation_id) id: vec3u) {
    if any(id.xy >= vec2u(W_IMG)) { return; }

    let R = textureDimensions(channel0).xy;
    let low = vec2u(vec2f(R) * vec2f(id.xy) / f32(W_IMG));
    let high = vec2u(vec2f(R) * vec2f(id.xy + 1) / f32(W_IMG));

    var acc = vec4f(0);
    var white = 0.0;

    for (var x = low.x; x < high.x; x++) {
        for (var y = low.y; y < high.y; y++) {
            acc += textureLoad(channel0, vec2u(x, y), 0);
            white += 1.0;
        }
    }

    let idx = dot(id.xy, vec2u(1, W_IMG));
    state.img[idx] = acc / white;

    if time.frame < 3 {
        var p = id.xy;
        for (var k = 2u; k < 32; k++) {
            p.y += (p.x * 31 - k * (p.x / k)) % W_IMG;
            p = p.yx % W_IMG;
        }

        state.cube[idx] = dot(p, vec2u(1, W_IMG));
    }
}

#workgroup_count sort 4 4 1
@compute @workgroup_size(16, 16)
fn sort(@builtin(global_invocation_id) id: vec3u) {
    if any(id.xy >= vec2u(W_IMG)) { return; }

    var dir = vec3u(1);
    dir[(time.frame + 1) % 3] = W_CUBE;
    dir[(time.frame + 2) % 3] = W_CUBE * W_CUBE;

    let base = dot(dir.yz, id.xy);
    for (var k = time.frame % 2; k < W_CUBE - 1; k += 2) {
        let cube_a = base + dir.x * k;
        let cube_b = cube_a + dir.x;

        let img_a = state.cube[cube_a];
        let img_b = state.cube[cube_b];

        let val_a = state.img[img_a][time.frame % 3];
        let val_b = state.img[img_b][time.frame % 3];

        if val_a < val_b {
            continue;
        }

        state.cube[cube_a] = img_b;
        state.cube[cube_b] = img_a;
    }
}

#workgroup_count recolor 16 16 16
@compute @workgroup_size(4, 4, 4)
fn recolor(@builtin(global_invocation_id) id: vec3u) {
    if any(id >= vec3u(W_CUBE)) { return; }

    let cube_idx = dot(id, vec3u(1, W_CUBE, W_CUBE * W_CUBE));
    let img_idx = state.cube[cube_idx];

    let col = vec3f(id) / f32(W_CUBE);
    state.out[img_idx] = vec4f(col, 1.0);
}

@compute @workgroup_size(16, 16)
fn present(@builtin(global_invocation_id) id: vec3u) {
    let R = textureDimensions(screen);
    if any(id.xy >= R) { return; }

    var raw = vec2i(2 * id.xy) - vec2i(R.xy);
    if mouse.click != 0 {
        raw /= 2;
        raw -= vec2i(2 * mouse.pos) - vec2i(R.xy);
    }

    let coord = (raw + i32(W_IMG)) / 2;
    if any(coord < vec2i(0)) || any(coord >= vec2(i32(W_IMG))) {
        let col = vec3f(17, 22, 26) / 256.0;
        let out = vec4f(pow(col, vec3f(2.2)), 1.0);
        textureStore(screen, id.xy, out);
        return;
    }

    let k = dot(vec2u(coord), vec2u(1, W_IMG));
    let col = pow(state.out[k], vec4(2.2));

    textureStore(screen, id.xy, col);
}
