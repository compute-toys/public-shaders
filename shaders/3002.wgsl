#storage state State

struct State {
    angle: vec2f, // yaw pitch
    angular_velocity: vec2f,
    zoom: f32,
    last_mouse_pos: vec2i,
    was_clicking: i32,
}

struct BoxAligned {
    low: vec3f,
    high: vec3f,
    colors: array<vec3f, 6>,
}

struct Ray {
    origin: vec3f,
    direction: vec3f,
}

struct Camera {
    pos: vec3f,
    up: vec3f,
    look: vec3f,
    fov: f32,
}

fn argmin(v: vec3f) -> u32 {
    let lt_xy = u32(v.x < v.y);
    let lt_xz = u32(v.x < v.z);
    let lt_yz = u32(v.y < v.z);
    return (1u - lt_xy) * lt_yz + 2u * (1u - lt_xz) * (1u - lt_yz);
}

fn argmax(v: vec3f) -> u32 {
    let gt_xy = u32(v.x > v.y);
    let gt_xz = u32(v.x > v.z);
    let gt_yz = u32(v.y > v.z);
    return (1u - gt_xy) * gt_yz + 2u * (1u - gt_xz) * (1u - gt_yz);
}

fn color_ray(box_obj: BoxAligned, ray: Ray) -> vec4f {
    let t_low = (box_obj.low - ray.origin) / ray.direction;
    let t_high = (box_obj.high - ray.origin) / ray.direction;
    let t_close = min(t_low, t_high);
    let t_far = max(t_low, t_high);
    
    let t_close_result = max(t_close.x, max(t_close.y, t_close.z));
    let t_far_result = min(t_far.x, min(t_far.y, t_far.z));
    
    if (t_close_result > t_far_result) { return vec4f(0.0); }
    
    let t_close_index = argmax(t_close);
    let t_far_index = argmin(t_far);
    
    let close_dir_is_pos = u32(ray.direction[t_close_index] > 0.0);
    let far_dir_is_pos = u32(ray.direction[t_far_index] > 0.0);
    
    let color_close = box_obj.colors[t_close_index * 2u + close_dir_is_pos];
    let color_far = box_obj.colors[t_far_index * 2u + far_dir_is_pos];
    
    return vec4f(color_close * color_far, 1.0);
}

fn rotate(v: vec3f, axis: vec3f, angle: f32) -> vec3f {
    return mix(dot(axis, v) * axis, v, cos(angle)) + cross(axis, v) * sin(angle);
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    
    let fragCoord = vec2f(f32(id.x) + 0.5, f32(id.y) + 0.5);
    var uv = (fragCoord / vec2f(screen_size)) * 2.0 - 1.0;
    uv.x *= f32(screen_size.x) / f32(screen_size.y);
    
    var camera = Camera(
        vec3f(4.0, 0.0, 0),
        vec3f(0.0, 0.0, 1.0),
        vec3f(-1.0, 0.0, 0.0),
        3.14 / 2.0
    );
    
    let zAxis = vec3f(0.0, 0.0, 1.0);
    camera.pos = rotate(camera.pos, zAxis, state.angle.x);
    camera.look = rotate(camera.look, zAxis, state.angle.x);
    
    let localRight = normalize(cross(camera.look, camera.up));
    camera.pos = rotate(camera.pos, localRight, state.angle.y);
    camera.look = rotate(camera.look, localRight, state.angle.y);
    
    let look = normalize(camera.look);
    let up_ = normalize(camera.up);
    let right = normalize(cross(look, up_));
    let up = cross(right, look);
    
    let ray_direction = normalize(
        look + 
        uv.x * tan(camera.fov / 2.0) * right + 
        uv.y * tan(camera.fov / 2.0) * up
    );
    
    let ray = Ray(camera.pos, ray_direction);
    
    let colors = array<vec3f, 6>(
        vec3f(1.0, 0.0, 1.0), vec3f(1.0, 0.0, 1.0),
        vec3f(1.0, 1.0, 0.0), vec3f(1.0, 1.0, 0.0),
        vec3f(0.0, 1.0, 1.0), vec3f(0.0, 1.0, 1.0)
    );
    
    let box_obj = BoxAligned(vec3f(-1.0), vec3f(1.0), colors);
    
    let col = color_ray(box_obj, ray);
    
    textureStore(screen, id.xy, col);
}

@compute @workgroup_size(1)
#workgroup_count handle_mouse 1 1 1
fn handle_mouse() {
    let screen_size = vec2i(textureDimensions(screen));
    let friction = 0.96;
    let sensitivity = 1.5;
    let zoom_stiffness = 0.1;
    // state.angular_velocity = vec2f(0, 0.1);
    
    if mouse.click == 1 {
        let norm_delta = vec2f(mouse.pos - state.last_mouse_pos) / vec2f(f32(screen_size.y)) * vec2f(-1, 1) * f32(state.was_clicking);
        state.angular_velocity = norm_delta * sensitivity;
        state.angle += state.angular_velocity;
        state.last_mouse_pos = mouse.pos;
        state.was_clicking = 1;
    } else {
        state.angle += state.angular_velocity;
        state.angular_velocity *= friction;
        state.was_clicking = 0;
    }

    // Prevent jittering
    state.angle.y = clamp(state.angle.y, -1.5, 1.5);
    // state.angle.y = 0;

    state.zoom = mix(state.zoom, mouse.zoom, zoom_stiffness);
}
