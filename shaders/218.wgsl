#storage state Controller

const KEY_W     = 87;
const KEY_A     = 65;
const KEY_S     = 83;
const KEY_D     = 68;
const KEY_E     = 69;
const KEY_Q     = 81;

const KEY_SHIFT = 16;
const KEY_SPACE = 32;

const KEY_LEFT  = 37;
const KEY_UP    = 38;
const KEY_RIGHT = 39;
const KEY_DOWN  = 40;

const PI  = 3.1415926535897932384626;
const TAU = 2.0 * PI;

const MAX_DIS      = 1e3;
const MAX_RAYMARCH = 512;

var<private> seed: u32;
var<private> coord: float2;
var<private> screen_pixel_size: float2;
var<private> pixel_radius: float;
var<private> uv: float2;

var<private> camera: Camera;
var<private> ray: Ray;

struct Camera {
    lookfrom: float3,
    lookat:   float3,
    vup:      float3,
    vfov:     float,
    aspect:   float,
    aperture: float,
    focus:    float,
}

struct Controller {
    premouse:    float3,
    lookat:      float3,
    target_p:    float3,
    position:    float3,
    target_r:    float2,
    rotation:    float2,
    resolution:  float2,
    moving:      uint,
}

struct Ray {
    origin:    float3,
    direction: float3,
}

fn hash4(s: vec4<u32>) -> u32 {
    let q = s * vec4(1597334673u, 3812015801u, 2798796415u, 1979697957u);
	return (q.x ^ q.y ^ q.z ^ q.w) * 1597334673u;
}

fn rand2() -> float2 {
    seed = seed * 747796405u + 2891336453u;
    var n = seed * vec2(1597334673u, 3812015801u);
    n = (n.x ^ n.y) * vec2(1597334673u, 3812015801u);
    return vec2<f32>(n) * (1.0 / float(0xffffffffu));
}

fn random_in_unit_disk() -> float2 {
    let r = rand2() * vec2(1.0, TAU);
    return sqrt(r.x) * vec2(sin(r.y), cos(r.y));
}

fn camera_gen_ray(c: Camera, uv: float2) -> Ray {
    let z = normalize(c.lookfrom - c.lookat);
    let x = normalize(cross(c.vup, z));
    let y = cross(z, x);

    let theta = radians(c.vfov);
    let hh = tan(theta * 0.5);
    let hw = c.aspect * hh;
    let mm = c.focus * mat3x3(hw * x, hh * y, z);
    
    let lens_radius = c.aperture * 0.5;
    let lrud = lens_radius * random_in_unit_disk();
    
    let ro = c.lookfrom + mat2x3(x, y) * lrud;
    let po = c.lookfrom + mm * vec3(2*uv-1, -1);
    let rd = normalize(po - ro);
    
    return Ray(ro, rd);
}

fn keyboard_mov_input() -> float3 {
    return vec3(float(keyDown(KEY_D)) - float(keyDown(KEY_A)), 
                float(keyDown(KEY_E)) - float(keyDown(KEY_Q)),
                float(keyDown(KEY_S)) - float(keyDown(KEY_W)));
}

fn keyboard_rot_input() -> float2 {
    return vec2(float(keyDown(KEY_RIGHT)) - float(keyDown(KEY_LEFT)), 
                float(keyDown(KEY_DOWN))  - float(keyDown(KEY_UP)));
}

fn camera_rotation(v: float2) -> mat3x3<float> {
    let s = sin(v.yx); let c = cos(v.yx);
    let rotX = mat3x3(1.0, 0.0, 0.0, 0.0, c.x, -s.x, 0.0, s.x, c.x);
    let rotY = mat3x3(c.y, 0.0, s.y, 0.0, 1.0, 0.0, -s.y, 0.0, c.y);
    return rotY * rotX;
}

#workgroup_count init_camera 1 1 1
@compute @workgroup_size(1)
fn init_camera(@builtin(global_invocation_id) id: uint3) {
    if (any(id != vec3(0u))) { return; }
    if (time.frame != 0u) { return; }

    state.position = vec3(0.0, 2.0, 5.0);
    state.rotation = vec2(0.0, 0.0);
    
    state.target_p = vec3(0.0, 0.3, 4.0);
    state.target_r = vec2(0.0, 0.0);
    state.premouse = vec3(0.0);
}

#workgroup_count process_camera 1 1 1
@compute @workgroup_size(1)
fn process_camera(@builtin(global_invocation_id) id: uint3) {
    if (any(id != vec3(0u))) { return; }

    let screen_size = float2(textureDimensions(screen));
    let resolution  = state.resolution;

    var target_p = state.target_p;
    var position = state.position;
    var rotation = state.rotation;
    var target_r = state.target_r;
    let premouse = state.premouse;

    let velocity = 10.0 + 15.0 * float(keyDown(KEY_SHIFT));

    var delta_mouse = float2(mouse.pos) - premouse.xy;
    let mouse_moved = any(delta_mouse != vec2(0));

    delta_mouse *= 600 * premouse.z / screen_size;
    
    target_r += (delta_mouse + 3 * keyboard_rot_input()) * time.delta;
    target_r.y = clamp(target_r.y, -PI * 0.499, PI * 0.499);

    rotation += (target_r - rotation) * saturate(time.delta * 16.0);
    
    let camera_matrix = camera_rotation(rotation);
    let direction = camera_matrix * keyboard_mov_input();

    target_p += direction * saturate(time.delta * velocity);
    position += (target_p - position) * saturate(time.delta * 10.0);
    
    let moving = length(target_r - rotation) * screen_size.x > 1.0 ||
                 length(target_p - position) * screen_size.x > 1.0 ||
                 any(screen_size != resolution.xy)                 ||
                 keyDown(KEY_SPACE);
    
    state.lookat     = position + camera_matrix * vec3(0, 0, -1);
    state.target_p   = target_p;
    state.position   = position;
    state.target_r   = target_r;
    state.rotation   = rotation;
    state.resolution = screen_size;
    state.moving     = uint(moving);
    state.premouse   = vec3(float2(mouse.pos), float(mouse.click > 0 && (mouse_moved || premouse.z > 0)));
}

fn spherical_map(p: float3) -> float2 {
    var uv = vec2(atan2(p.z, p.x), asin(p.y));
    return uv * vec2(1.0 / TAU, 1.0 / PI) + 0.5;
}

fn environment(lod: float) -> float4 {
    let uv = spherical_map(-ray.direction);
    let col = textureSampleLevel(channel0, bilinear, uv, lod);
    return col.rgba;
}

fn sd_sphere(p: float3, r: float) -> float {
    return length(p) - r;
}

fn sd_box(p: float3, b: float3) -> float {
    let q = abs(p) - b;
    return length(max(q, vec3(0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sdf_scene(p: float3) -> float {
    let box = sd_box(p - vec3(0, 0.5, 0), vec3(1.0));
    let ground = sd_sphere(p - vec3(0, -100.52, 0), 100.0);
    return min(box, ground);
}

fn calc_normal(p: float3) -> float3 {
    let e = vec2(1, -1) * 0.5773 * 0.005;
    return normalize(e.xyy*sdf_scene(p + e.xyy) + 
                     e.yyx*sdf_scene(p + e.yyx) + 
                     e.yxy*sdf_scene(p + e.yxy) + 
                     e.xxx*sdf_scene(p + e.xxx));
}

fn raymarch_sdf(uv: float2) -> float4 {
    var color: float3;
    var t = 0.0;
    var w = 1.6;
    var s = 0.0;
    var d = 0.0;
    var hit = false;
    var pos: float3;

    for (var i = 0; i < MAX_RAYMARCH && t < MAX_DIS && !hit; i++) {
        pos = ray.origin + ray.direction * t;
        let dis = sdf_scene(pos);
        
        if (w > 1.0 && d + dis < s) {
            s -= w * s;
            t += s;
            w = 1.0;
            continue;
        }
        
        hit = dis < pixel_radius * t;
        s = w * dis;
        d = dis;
        t += s;
    }

    if (!hit) {
        color = environment(0).rgb;
    } else {
        let normal = calc_normal(pos);
        color = 0.5 + 0.5 * normal;
    }
    return vec4(color, 1.0);
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    let screen_size = textureDimensions(screen);
    if (any(id.xy > screen_size)) { return; }

    seed = hash4(vec4(id.xy, time.frame, u32(time.elapsed)));
    coord = float2(id.xy) + rand2();
    screen_pixel_size = 1.0 / float2(screen_size);
    pixel_radius = min(screen_pixel_size.x, screen_pixel_size.y);
    uv = coord * screen_pixel_size;
    uv.y = 1.0 - uv.y;

    camera.lookfrom = state.position;
    camera.lookat   = state.lookat;
    camera.vup      = vec3(0.0, 1.0,  0.0);
    camera.aspect   = screen_pixel_size.y / screen_pixel_size.x;
    camera.vfov     = 180.0 * clamp(custom.vfov, 0.1, 0.9);
    camera.aperture = 0.1 * pow(custom.aperture, 4.0);
    camera.focus    = pow(4.0, custom.focus);

    ray = camera_gen_ray(camera, uv);

    var color: float4;
    var data:  float4;

    color = raymarch_sdf(uv);

    data = textureLoad(pass_in, id.xy, 0, 0);

    data *= float(!bool(state.moving));
    
    data += color;

    textureStore(pass_out, id.xy, 0, data);
    textureStore(screen, id.xy, float4(data.rgb / data.a, 1.0));
}