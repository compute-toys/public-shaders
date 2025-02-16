
struct Camera {
    origin: vec3f,
    forward: vec3f,
    upward: vec3f,
}

struct Bounce {
    start: vec3f,
    direction: vec3f,
    width: f32,
}

struct GlobalState {
    camera: Camera,
    objects_size: u32,
    bounce_balls: array<Bounce>,
}

#storage global_state GlobalState

const DISTANCE_EYE_ORIGIN: f32 = 40.0;
const OBJECTS_SIZE = 240u;

// On generating random numbers, with help of y= [(a+x)sin(bx)] mod 1", W.J.J. Rey, 22nd European Meeting of Statisticians 1998
fn rand11(n: f32) -> f32 { return fract(sin(n) * 43758.5453123); }
fn rand22(n: vec2f) -> f32 { return fract(sin(dot(n, vec2f(12.9898, 4.1414))) * 43758.5453); }

const PI = 3.14159265358532374;
const PHI = 2.61803398875;

fn fibo_grid_n(n: u32, total: u32) -> vec3f {
    let theta = 2.0 * PI * f32(n) / PHI;
    let phi = acos(1. - 2. * (f32(n) + 0.5) / f32(total));
    
    let x = sin(phi) * cos(theta);
    let y = sin(phi) * sin(theta);
    let z = cos(phi);
    return vec3f(x, y, z);
}

@compute @workgroup_size(1, 1)
#dispatch_once initialization
fn initialization() {
    global_state.camera =  Camera(vec3f(0., 0., 400.), vec3f(0., 0., -1.), vec3f(0., 1., 0.));

    let zero = vec3f(0., 0., 0.);

    global_state.objects_size = 0u;
    var c: u32 = 0u;
    for (var i = 0u; i < OBJECTS_SIZE; i++) {
        c = global_state.objects_size;
        let v = fibo_grid_n(i, OBJECTS_SIZE) * 4.;
        global_state.bounce_balls[c] = Bounce(vec3f(0., 100., 0.), v, 1.);
        global_state.objects_size = c + 1u;
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
    let v_move = 0.032;
    let v_pos = 0.008;
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
#workgroup_count move_balls 1 1 1
@compute @workgroup_size(1, 1)
fn move_balls() {
    let total = global_state.objects_size;
    let a = vec3f(0., -0.004, 0.);

    let c0 = vec3f(0., 50., 0.);
    let r0 = 150.;

    for (var i = 0u; i < total; i++) {
        let ball = global_state.bounce_balls[i];
        let next_pos = ball.start + ball.direction * 0.03 * time.elapsed;

        if abs(distance(next_pos, c0)) >= r0 {
            // quite inaccurate anyway....
            let pointer = normalize(next_pos - c0) * r0;
            let reverted_direction = normalize(c0 - pointer);
            let v_perp = dot(ball.direction, reverted_direction) * reverted_direction;
            let v_hori = ball.direction - v_perp;
            let next_v = v_hori * 0.96 - v_perp * 0.88;
            global_state.bounce_balls[i].direction = next_v;
        } else {
            global_state.bounce_balls[i].start = next_pos;
            global_state.bounce_balls[i].direction += a * time.elapsed;
        }
        
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

    let objects_size = global_state.objects_size;
    let eye = origin - DISTANCE_EYE_ORIGIN * forward;
    let pixel = origin + x * rightward + y * upward;
    let ray_direction = normalize(pixel - eye);

    for (var i = 0u; i < objects_size; i++) {
        let ball = global_state.bounce_balls[i];
        
        let pick_line = ball.start - eye;
        let pick_unit = normalize(pick_line);
        if dot(pick_unit, forward) < 0.5 {
            continue;
        }

        let ray_distance = dot(ray_direction, pick_line);
        let near_point = eye + ray_direction * ray_distance;

        let distance_ball_ray = abs(distance(ball.start, near_point));

        // thanks to https://math.stackexchange.com/a/4764188/54238
         
        if distance_ball_ray > ball.width {
            continue;
        }
            
        
        let l = distance_ball_ray;
        
        if l < .1 {
            // color = mix(color, vec4f(1., 1., 1., 1.), 0.001);
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
