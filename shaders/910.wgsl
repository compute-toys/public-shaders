
struct Camera {
    origin: vec3f,
    forward: vec3f,
    upward: vec3f,
}

struct Line {
    start: vec3f,
    direction: vec3f,
    /// >=2 for directing
    /// >=1 for line
    usage: f32,
    end: vec3f,
    width: f32,
}

struct GlobalState {
    camera: Camera,
    lines_size: u32,
    lines: array<Line>,
}

#storage global_state GlobalState

const DISTANCE_EYE_ORIGIN: f32 = 50.0;
const ROTATE_SPEED = 0.1;

fn calculate_camera(angle: f32) -> Camera {
    let r = 100.;
    let origin = vec3f(cos(angle) * r, 20., -sin(angle) * r);
    let forward = vec3f(-sin(angle), 0., -cos(angle));
    return Camera(origin, forward, vec3f(0., 1., 0.));
}

// On generating random numbers, with help of y= [(a+x)sin(bx)] mod 1", W.J.J. Rey, 22nd European Meeting of Statisticians 1998
fn rand11(n: f32) -> f32 { return fract(sin(n) * 43758.5453123); }
fn rand22(n: vec2f) -> f32 { return fract(sin(dot(n, vec2f(12.9898, 4.1414))) * 43758.5453); }

@compute @workgroup_size(1, 1)
#dispatch_once initialization
fn initialization() {
    global_state.camera = calculate_camera(1.3);

    let zero = vec3f(0., 0., 0.);

    global_state.lines_size = 0u;
    var c: u32 = 0u;
    for (var i = -9i; i <= 10i; i++) {
        c = global_state.lines_size;
        global_state.lines[c] = Line(vec3f(30. * f32(i), 0., 0.), vec3f(0., 0., -1.), 0., zero, 0.02);
        global_state.lines_size = c + 1u;
    }
    for (var i = -9i; i <= 10i; i++) {
        c = global_state.lines_size;
        global_state.lines[c] = Line(vec3f(0., 0., 30. * f32(i)), vec3f(1., 0., 0.), 0., zero, 0.02);
        global_state.lines_size = c + 1u;
    }

    for (var j=0u; j < 4u; j ++) {
        let r_x = rand11(f32(j)) - 0.5;
        let r_y = rand11(f32(j)+0.5) - 0.5;
        var base = vec3f(r_x * 200., 0. , r_y * 200.);
        let base_direction = normalize(base);
        base = base + base_direction * 100.;
        if j == 0u {
            base = vec3f(100., 0., -40);
        }
        
        global_state.lines[c] = Line(base, normalize(vec3f(0., 1., 0.)), 3., base + vec3f(0., 1., 0.) * 40., 51.);
        global_state.lines[c+1] = Line(base, normalize(vec3f(0., 1., 0.)), 1., base + vec3f(0., 1., 0.) * 40., 0.4);
        global_state.lines_size = c + 2u;

        for (var i = 0u; i < 40u; i++) {
            c = global_state.lines_size;
            let move_down = 0.4 * f32(i);
            let main = base + vec3f(0., 40. - move_down, 0.);
            let angle = f32(i) * 0.72;
            let ratio = f32(i+1) / 50.;
            let direct = 20 * ratio * (vec3f(1., 0., 0.) * sin(angle) + vec3f(0., 0., -1.) * cos(angle)) + move_down * vec3f(0., -1, 0.);
            let end = main + direct;
            let direction = normalize(end - main);
            global_state.lines[c] = Line(main, direction , 1., end, 0.2);

            global_state.lines_size = c + 1u;
        }
    }

}

const KEY_W = 87u;
const KEY_A = 65u;
const KEY_S = 83u;
const KEY_D = 68u;

const KEY_I = 73u;
const KEY_J = 74u;
const KEY_K = 75u;
const KEY_L = 76u;


#workgroup_count update_camera 1 1 1
@compute @workgroup_size(1, 1)
fn update_camera() {
    let forward = global_state.camera.forward;
    let upward = global_state.camera.upward;
    let right = normalize(cross(forward, upward));
    let v_move = 0.008;
    let v_pos = 0.001;
    let v_angle = 0.00002;
    if keyDown(KEY_W) {
        global_state.camera.origin += forward * time.elapsed * v_move;
    }
    if keyDown(KEY_S) {
        global_state.camera.origin -= forward * time.elapsed * v_move;
    }

    if keyDown(KEY_A) {
        let angle = -v_angle * time.elapsed;
        let next_f = forward * cos(angle) + right * sin(angle);
        global_state.camera.forward = next_f;
    }
    if keyDown(KEY_D) {
        let angle = v_angle * time.elapsed;
        let next_f = forward * cos(angle) + right * sin(angle);
        global_state.camera.forward = next_f;
    }
    if keyDown(KEY_I) {
        global_state.camera.origin += upward * time.elapsed * v_pos;        
    }
    if keyDown(KEY_K) {
        global_state.camera.origin -= upward * time.elapsed * v_pos;        
    }
    if keyDown(KEY_J) {
        global_state.camera.origin -= right * time.elapsed * v_pos;
    }
    if keyDown(KEY_L) {
        global_state.camera.origin += right * time.elapsed * v_pos;
    }
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    let narrow = f32(min(screen_size.x, screen_size.y));
    let x = (f32(id.x) - f32(screen_size.x) * 0.5) / narrow * 20.;
    let y = -(f32(id.y) - f32(screen_size.y) * 0.5) / narrow * 20.;

    var color = vec4f(0., 0., 0., 0.);
    let forward = global_state.camera.forward;
    let upward = global_state.camera.upward;
    let origin = global_state.camera.origin;
    let rightward = cross(forward, upward);

    // Drawring

    let lines_size = global_state.lines_size;
    let eye = origin - DISTANCE_EYE_ORIGIN * forward;
    let pixel = origin + x * rightward + y * upward;
    let ray_direction = pixel - eye;

    var skip_direct = 0u;
    for (var i = 0u; i < lines_size; i++) {
        if skip_direct > 0 {
            skip_direct -= 1;
            continue;
        }
        let line = global_state.lines[i];
        
        let perp_line = cross(ray_direction, line.direction);
        let perp_direction = normalize(perp_line);
        let pick_line = line.start - eye;
        let distance_line_ray = abs(dot(perp_direction, pick_line));

        // thanks to https://math.stackexchange.com/a/4764188/54238
         
        if distance_line_ray > line.width {
            continue;
        }
        let t1 = dot(cross(line.start - eye, line.direction), perp_line) / dot(perp_line, perp_line);
        
        if t1 < 0.001 {
            if line.usage >= 2. {
                skip_direct = u32(line.width);
                continue;
            }
            continue;
        }
        if line.usage >= 2. {
            continue;
        }
            
        
        let l = length(ray_direction * t1);
        if line.usage >= 1. {
            let closest = eye + t1 * ray_direction;
            let to_start = line.start - closest;
            let to_end = line.end - closest;
            if dot(to_start, to_end) > 0. {
                continue;
            }
        }
        
        if l > 300. {
            color = mix(color, vec4f(1., 1., 1., 1.), 0.1);
        } else {
            color = mix(color, vec4f(1., 1., 1., 1.), 0.5);
        }
    }

    // HUD UIs

    let r = length(vec2f(x, y));

    if r < 10.0 && r > 9.98 {
        // draws a circle with radius 10
        color = mix(color, vec4f(1., 1., 1., 1.), 0.04);
    }
    if distance(vec2f(x, y), (vec2f(5., 5.,))) < 0.1 {
        // draw a dot at (5,5)
        color = mix(color, vec4f(1., 1., 1., 1.), 0.04);
    }
    if abs(x) < 0.01 || abs(y) < 0.01 {
        color = mix(color, vec4f(1., 1., 1., 1.), 0.04);
    }

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, color);
}
