#storage state array<vec3f>
#storage payload array<vec3f>

const UV_PAYLOAD: bool = false;

const size: vec2u = vec2u(512, 512);
const dt: f32 = 0.05;

fn get_u(x: i32, y: i32) -> vec2f {
    if x <= 0 || x >= i32(size.x) || y <= 0 || y >= i32(size.y) {
        return vec2f(0.0);
    }
    return state[x + y * i32(size.y)].xy;
}

fn get_q(x: i32, y: i32) -> vec3f {
    let cx = clamp(x, 0, i32(size.x - 1));
    let cy = clamp(y, 0, i32(size.y - 1));
    return payload[cx + cy * i32(size.y)];
}

fn get_p(x: i32, y: i32) -> f32 {
    if x < 0 || x > i32(size.x - 1) || y < 0 || y > i32(size.y - 1) {
        return 0.0;
    }
    return state[x + y * i32(size.y)].z;
}


fn sample_u(uv: vec2f) -> vec2f {
    let uv_c = uv - 0.5;
    let gx = i32(floor(uv_c.x));
    let gy = i32(floor(uv_c.y));
    let f = fract(uv_c);
    
    let v00 = get_u(gx, gy);
    let v10 = get_u(gx + 1, gy);
    let v01 = get_u(gx, gy + 1);
    let v11 = get_u(gx + 1, gy + 1);
    
    return mix(mix(v00, v10, f.x), mix(v01, v11, f.x), f.y);
}

fn sample_q(uv: vec2f) -> vec3f {
    let uv_c = uv - 0.5;
    let gx = i32(floor(uv_c.x));
    let gy = i32(floor(uv_c.y));
    let f = fract(uv_c);
    
    let v00 = get_q(gx, gy);
    let v10 = get_q(gx + 1, gy);
    let v01 = get_q(gx, gy + 1);
    let v11 = get_q(gx + 1, gy + 1);
    
    return mix(mix(v00, v10, f.x), mix(v01, v11, f.x), f.y);
}

#dispatch_once init
@compute
@workgroup_size(16, 16)
fn init(@builtin(global_invocation_id)gid: vec3u) {
    if gid.x > size.x - 1 || gid.y > size.y - 1 {
        return;
    }
    let uv = vec2f(gid.xy) / vec2f(size);
    if UV_PAYLOAD {
        payload[gid.x + gid.y * size.y] = vec3f(uv, 1.0);
    } else {
        payload[gid.x + gid.y * size.y] = textureSampleLevel(channel0, bilinear, uv, 0).rgb;
    }
}

@compute
@workgroup_size(16, 16)
fn solve(@builtin(global_invocation_id)gid: vec3u) {
    if gid.x > size.x - 1 || gid.y > size.y - 1 {
        return;
    }
    // Advect
    let x = i32(gid.x);
    let y = i32(gid.y);
    let pos = vec2f(gid.xy) + 0.5;
    let vel = get_u(x, y);
    let prev_pos = pos - vel * dt;
    let prev_u = sample_u(prev_pos);
    let prev_q = sample_q(prev_pos);
    state[x + y * i32(size.y)].x = prev_u.x;
    state[x + y * i32(size.y)].y = prev_u.y;
    payload[x + y * i32(size.y)] = prev_q;

    // Force
    let uv = (vec2f(f32(x), f32(y)) / vec2f(size) - 0.5) * 2.0;
    let f = smoothstep(0.2, 0.0, length(uv - vec2f(-1.0, 0.0))) * vec2f(100.0, 0.0);
    state[x + y * i32(size.y)].x += f.x * dt;
    state[x + y * i32(size.y)].y += f.y * dt;


    // Project
    let dx = 1.0;
    let dy = 1.0;
    let div = 
        (sample_u(pos + vec2(dx, 0.0)).x - sample_u(pos - vec2(dx, 0.0)).x) / (2.0 * dx) +
        (sample_u(pos + vec2(0.0, dy)).y - sample_u(pos - vec2(0.0, dy)).y) / (2.0 * dy);
    for(var i = 0; i < 20; i ++) {
        var p_n: f32 = 0.0;
        p_n += (get_p(x - 1, y) + get_p(x + 1, y)) * dy * dy;
        p_n += (get_p(x, y - 1) + get_p(x, y + 1)) * dx * dx;
        p_n -= div * dx * dx * dy * dy;
        p_n /= 2.0 * (dx * dx + dy * dy);
        state[x + y * i32(size.y)].z = p_n;
    }
    let residual = vec2(
        (get_p(x + 1, y) - get_p(x - 1, y)) / (2.0 * dx),
        (get_p(x, y + 1) - get_p(x, y - 1)) / (2.0 * dy),
    );
    state[x + y * i32(size.y)].x -= residual.x;
    state[x + y * i32(size.y)].y -= residual.y;

}




@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    if (id.x >= size.x || id.y >= size.y) { return; }
    var col: vec3f;
    if UV_PAYLOAD {
        let uv = vec3f(payload[id.x + id.y * size.y]).xy;
        col = textureSampleLevel(channel0, bilinear, uv, 0).rgb;
    } else {
        col = payload[id.x + id.y * size.y];
    };
    textureStore(screen, id.xy, vec4f(col, 1.));
}
