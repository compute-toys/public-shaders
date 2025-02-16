#define PI 3.14159265

#storage offset vec2f
#storage zoom f32

fn complexMultiply(a: vec2f, b: vec2f) -> vec2f {
    return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn complexMod(z: vec2f) -> f32 {
    return z.x * z.x + z.y * z.y;
}

@compute @workgroup_size(1, 1)
#dispatch_once init
fn init() {
    offset = vec2(0.0, 0.0);
    zoom = 1.0f;
}

@compute @workgroup_size(1, 1)
fn update() {
    if (keyDown(69)) {
        zoom *= 1.000002f;
    }
    if (keyDown(81)) {
        zoom /= 1.000002f;
    }
    if (keyDown(68)) {
        offset.x += 2e-6 / zoom;
    }
    if (keyDown(65)) {
        offset.x -= 2e-6 / zoom;
    }
    if (keyDown(87)) {
        offset.y += 2e-6 / zoom;
    }
    if (keyDown(83)) {
        offset.y -= 2e-6 / zoom;
    }
}

@compute @workgroup_size(8, 8)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let resolution = textureDimensions(screen);
    if ((id.x >= resolution.x) || (id.y >= resolution.y)) {
        return;
    }
    let fragCoord = vec2f(f32(id.x) + 0.5f, f32(resolution.y - id.y) - 0.5f);

    let uv = (2.0f * fragCoord - vec2f(resolution)) / f32(resolution.y) + offset * zoom;
    var color = vec3f(cos(PI * 1.1f), sin(PI * 1.1f), cos(PI * 1.1f + PI * 0.5f));
    var c = uv / zoom;
    var z = uv / zoom;
    for (var i = 0u; i < 1000u; i = i + 1u) {
        z = complexMultiply(complexMultiply(z, z), complexMultiply(z, z)) + c;
        if (complexMod(z) <= 4.0f) {
            let x = PI * f32(i % 360u + 220u) / 200.0f;
            color = vec3f(cos(x), sin(x), cos(x + PI * 0.5f));
            if ((i + 1u) == 1000u) {
                color = vec3(0.0f, 0.0f, 0.0f);
            }
        }
    }
    
    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(color, 1.));
}