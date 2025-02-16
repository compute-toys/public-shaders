#include "Dave_Hoskins/hash"

#storage accumulator array<atomic<u32>>

const ITER = 3000u;
const RADIUS = 2.0;
const ZOOM = 1.2;

@compute @workgroup_size(16, 16)
fn splat(
    @builtin(global_invocation_id) id: vec3u,
    @builtin(local_invocation_index) idx: u32,
) {
    let R = textureDimensions(screen);

    var seed = vec4f(id.xyzz + time.frame);
    for (var k = 0u; k < 4u; k++) {
        seed = fract(hash44(seed) + seed.wxyz);
    }

    let rad = RADIUS * sqrt(seed.x);
    let ang = 2.0 * acos(-1.0) * seed.y;
    let c = rad * vec2f(cos(ang), sin(ang));

    var z = c;
    var steps = 0u;

    for (; steps < ITER; steps++) {
        z = float2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        if dot(z,z) > RADIUS * RADIUS { break; }
    }

    if steps >= ITER {
        return;
    }

    z = c;
    var k = 0u;
    var white = 0u;

    for (; k < steps; k++) {
        z = float2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;

        let coord = vec2i(z * f32(R.y) / ZOOM + vec2f(R)) / 2;
        if any(coord < vec2i(0)) || any(coord >= vec2i(R.xy))  {
            continue;
        }

        let off = i32(100 * steps > ITER) + i32(10 * steps > ITER);
        let idx = coord.x + i32(R.x) * coord.y;
        atomicAdd(&accumulator[3 * idx + off], 1);
        white += 1;
    }

    atomicAdd(&accumulator[3 * R.x * R.y + idx], white);
}

@compute @workgroup_size(16, 16)
fn present(
    @builtin(global_invocation_id) id: vec3u,
    @builtin(local_invocation_index) idx: u32,
) {
    let R = textureDimensions(screen);
    if any(id.xy >= R) { return; }

    let off = id.x + R.x * id.y;
    var col = vec3f(vec3u(
        atomicLoad(&accumulator[3 * off + 2]),
        atomicLoad(&accumulator[3 * off + 1]),
        atomicLoad(&accumulator[3 * off + 0]),
    ));

    let white = atomicLoad(&accumulator[3 * R.x * R.y + idx]);
    col *= 0.5 * f32(R.x * R.y) / (256.0 * pow(ZOOM, 2.0) * f32(white));
    col = pow(col, vec3f(1.5));

    textureStore(screen, id.xy, vec4f(col, 1.0));
}
