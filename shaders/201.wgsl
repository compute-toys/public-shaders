#include "Dave_Hoskins/hash"

const CLOTH_R = vec2u(64, 32);
const GAP_SIZE = 6.0;
const DRAG = 0.997;
const DT = 0.25;

struct MouseLast {
    pos: int2,
    click: int,
    time: float,
}

#storage mouse_last MouseLast

fn connected(a: vec2i, b: vec2i) -> bool {
    let v = vec4(vec2f(a), vec2f(b));
    return hash44(v).x * hash44(v.zwxy).x > 0.05;
}

#dispatch_count sim 4
#workgroup_count sim 4 2 1
@compute @workgroup_size(16, 16)
fn sim(@builtin(global_invocation_id) id: uint3) {
    let R = textureDimensions(screen);
    if any(id.xy >= CLOTH_R) { return; }

    if time.frame < 5 || id.y < 1 {
        var pos = GAP_SIZE * (vec2f(id.xy) - vec2f(CLOTH_R.xx) / 2.0);
        pos += vec2f(R) / 2.0;

        passStore(0, vec2i(id.xy), vec4f(pos, 0, 0));
        return;
    }

    let tex = passLoad(0, vec2i(id.xy), 0);
    let pos = tex.xy;
    let vel = tex.zw;
    var acc = vec2f(0);

    const DIRS = array(
        vec2i( 1,  0),
        vec2i(-1,  0),
        vec2i( 0,  1),
        vec2i( 0, -1),
    );

    for(var k=0; k<4; k++) {
        let n_id: vec2i = vec2i(id.xy) + DIRS[k];
        if any(n_id < vec2i(0)) || any(n_id >= vec2i(CLOTH_R)) {
            continue;
        }

        if !connected(vec2i(id.xy), n_id) {
            continue;
        }

        let n_tex = passLoad(0, n_id, 0);
        var n_pos = n_tex.xy;

        let d: vec2f = n_pos - pos;
        let d_len: f32 = length(d);

        acc += (d_len - GAP_SIZE) * normalize(d);
    }

    if (mouse.click | mouse_last.click) != 0 {
        let w = pos - vec2f(mouse.pos);
        let pd = vec2f(mouse.pos - mouse_last.pos);
        let td = time.elapsed - mouse_last.time;
        acc += 1e-5 * pd * td / dot(w, w);
    }

    let gravity = 0.1;
    acc.y += gravity;

    let t_vel = (DRAG * vel) + acc * DT; 
    let t_pos = pos + t_vel * DT;
    passStore(0, vec2i(id.xy), vec4f(t_pos, t_vel));
}

#storage image array<atomic<u32>>

fn draw_point(a: vec2f) {
    for (var dy = 0; dy < 2; dy++) {
        for (var dx = 0; dx < 2; dx++) {
            let p = vec2i(a) + vec2i(dx, dy);
            let d = 1.0 - abs(a - vec2f(p));
            let w = d.x * d.y;

            let idx = p.x + SCREEN_WIDTH * p.y;
            atomicMax(&image[idx], u32(1.5e8 * w));
        }
    }
}

fn draw_line(a: vec2f, b: vec2f) {
    let iter = 2 * u32(length(a - b));

    for (var k = 0u; k <= iter; k++) {
        let p = mix(a.xy, b.xy, f32(k) / f32(iter));
        draw_point(p);
    }
}

#workgroup_count sim 4 2 1
@compute @workgroup_size(16, 16)
fn draw(@builtin(global_invocation_id) id: uint3) {
    let R = textureDimensions(screen);
    if any(id.xy >= CLOTH_R) { return; }

    let a = vec2i(id.xy);
    let b = a + vec2i(1, 0);
    let c = a + vec2i(0, 1);

    let ta = passLoad(0, a, 0);
    let tb = passLoad(0, b, 0);
    let tc = passLoad(0, c, 0);

    if b.x < i32(CLOTH_R.x) && connected(a, b) {
        draw_line(ta.xy, tb.xy);
    }

    if c.y < i32(CLOTH_R.y) && connected(a, c) {
        draw_line(ta.xy, tc.xy);
    }
}

@compute @workgroup_size(16, 16)
fn present(@builtin(global_invocation_id) id: uint3) {
    let R = textureDimensions(screen);
    if any(id.xy >= R) { return; }

    if all(id == vec3u(0)) {
        mouse_last.pos = mouse.pos;
        mouse_last.click = mouse.click;
        mouse_last.time = time.elapsed;
    }

    let idx = id.x + R.x * id.y;
    let val = atomicExchange(&image[idx], 0);

    var col = f32(val) / 1e8;
    col += 0.005 * hash13(vec3f(vec3u(id.xy, time.frame)));
    textureStore(screen, id.xy, vec4f(col));
}