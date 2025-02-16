// Pan around by clicking and dragging with the mouse or using the WASD
// keys. Zoom in and out with the plus/minus or E/Q keys

struct PanZoom {
    top_left: vec2f,
    scale: f32,
    _mouse: Mouse,
    _time: Time,
}

#storage pz PanZoom

#workgroup_count pan_and_zoom 1 1 1
@compute @workgroup_size(1)
fn pan_and_zoom() {
    if time.frame == 0 {
        pz.top_left = vec2f(0.0);
        pz.scale = 1.0;
    } else {
        if pz._mouse.click == 1 && mouse.click == 1 {
            let delta_mouse_pos = vec2f(mouse.pos) - vec2f(pz._mouse.pos);
            pz.top_left -= delta_mouse_pos * pz.scale;
        }

        let zoom_direction = 
            f32(keyDown(81) || keyDown(189)) -
            f32(keyDown(69) || keyDown(187));
        let delta_time = time.elapsed - pz._time.elapsed;
        let zoom_factor = pow(
            2.0,
            10.0 * custom.zoom_speed * zoom_direction * delta_time,
        );
        pz.top_left -=
            (vec2f(mouse.pos) + 0.5) * pz.scale * (zoom_factor - 1.0);
        pz.scale *= zoom_factor;
        let screen_height = f32(textureDimensions(screen).y);
        pz.top_left += vec2f(
            f32(keyDown(68)) - f32(keyDown(65)),
            f32(keyDown(83)) - f32(keyDown(87))
        ) * 10.0 * custom.pan_speed * pz.scale * screen_height * delta_time;
    }

    pz._mouse = mouse;
    pz._time = time;
}

fn to_world(screen_coords: vec2f) -> vec2f {
    return pz.top_left + pz.scale * screen_coords;
}

fn to_screen(world_coords: vec2f) -> vec2f {
    return (world_coords - pz.top_left) / pz.scale;
}

// -------------------------------------------------------------------------- //

fn sRGB_to_linear(color: vec3f) -> vec3f {
    let higher = pow((color + 0.055) / 1.055, vec3f(2.4));
    let lower = color / 12.92;
    return select(lower, higher, color > vec3f(0.04045));
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);

    if any(id.xy >= screen_size) {
        return;
    }

    var color = render_image(to_world(vec2f(id.xy) + 0.5));

    if any(id.xy == mouse.pos) {
        color = vec3f(1.0, 0.0, 0.0);
    }

    textureStore(screen, id.xy, vec4f(sRGB_to_linear(color), 1.0));
}

// -------------------------------------------------------------------------- //

// mandelbrot feathers - ronwnor
// https://www.shadertoy.com/view/7tffDj

const TAU = 6.2831853;

fn rot(a: f32) -> mat2x2f {
    return mat2x2(cos(a), sin(a), -sin(a), cos(a));
}

fn render_image(frag_coord: vec2f) -> vec3f {
    let resolution = vec2f(textureDimensions(screen));
    let zoom = 6.0;
    let location = vec2(-0.1640316, -1.025873);
    var c = ((2.0 * frag_coord - resolution) / resolution.y);
    c = c * exp(-zoom) * vec2f(1.0, -1.0) + location;
    var z = c;
    var i: f32;

    for (i = 0.0; i < 64.0; i += 1.0) {
        z = vec2f(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        
        if !((z * rot(cos(TAU * time.elapsed * i / 40.0))).y < 8.0) {
            break;
        }
    }

    return cos(TAU * (i / 64.0 + vec3f(0.4, 0.5, 0.6))) * 0.5 + 0.5;
}