#storage world array<SDFObject>
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

const SHAPE_SPHERE   = 0;
const SHAPE_BOX      = 1;
const SHAPE_CYLINDER = 2;

const MAX_DIS = 1e3;
const VISIBILITY   = 1e-3;
const MAX_RAYMARCH = 512;
const MAX_RAYTRACE = 128;

const ENV_IOR = 1.000277;

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
    color:     float3,
}

struct Material {
    albedo:       float3,
    emission:     float3,
    roughness:    float,
    metallic:     float,
    transmission: float,
    ior:          float,
}

struct Transform {
    position: float3,
    rotation: mat3x3<float>,
    scale:    float3,
};

struct SDFObject {
    shape: int,
    transform: Transform,
    material: Material,
}

fn hash4(s: vec4<u32>) -> u32 {
    let q = s * vec4(1597334673u, 3812015801u, 2798796415u, 1979697957u);
	return (q.x ^ q.y ^ q.z ^ q.w) * 1597334673u;
}

fn rand1() -> float {
    seed = seed * 747796405u + 2891336453u;
    let word = ((seed >> ((seed >> 28u) + 4u)) ^ seed) * 277803737u;
    return f32((word >> 22u) ^ word) * (1.0 / float(0xffffffffu));
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
    
    return Ray(ro, rd, vec3(1));
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

#workgroup_count init 1 1 1
@compute @workgroup_size(1)
fn init(@builtin(global_invocation_id) id: uint3) {
    if (any(id != vec3(0u))) { return; }
    if (time.frame != 0u) { return; }

    state.position = vec3(0.0, 2.0, 5.0);
    state.rotation = vec2(0.0, 0.0);
    
    state.target_p = vec3(0.0, 0.3, 4.0);
    state.target_r = vec2(0.0, 0.0);
    state.premouse = vec3(0.0);

    world[0] = SDFObject(SHAPE_SPHERE,
                    Transform(vec3(0, -100.501, 0),
                       rotate(vec3(0, 0, 0)),
                              vec3(100, 100, 100)),
                    Material(vec3(1.0, 1.0, 1.0)*0.6,
                             vec3(1.0, 1.0, 1.0),
                             0.9, 0.1, 0.0, 1.635));
    world[1] = SDFObject(SHAPE_SPHERE,
                    Transform(vec3(0, 1.0, 0),
                       rotate(vec3(0, 0, 0)),
                              vec3(0.5, 0.5, 0.5)),
                    Material(vec3(1.0, 1.0, 1.0),
                             vec3(1.0, 100.0, 1.0),
                             0.0, 1.0, 0.0, 1.00));
    world[2] = SDFObject(SHAPE_CYLINDER,
                    Transform(vec3(-1, -0.2, 0),
                       rotate(vec3(0, 0, 0)),
                              vec3(0.3, 0.3, 0.3)),
                    Material(vec3(1.0, 0.2, 0.2),
                             vec3(1.0, 1.0, 1.0),
                             0.1, 0.0, 0.0, 1.460));
    world[3] = SDFObject(SHAPE_SPHERE,
                    Transform(vec3(1, -0.2, 0),
                       rotate(vec3(0, 0, 0)),
                              vec3(0.3, 0.3, 0.3)),
                    Material(vec3(0.2, 0.2, 1.0),
                             vec3(1.0, 1.0, 1.0),
                             0.2, 1.0, 0.0, 1.100));
    world[4] = SDFObject(SHAPE_SPHERE,
                    Transform(vec3(0, -0.2, 2),
                       rotate(vec3(0, 0, 0)),
                              vec3(0.3, 0.3, 0.3)),
                    Material(vec3(1.0, 1.0, 1.0)*0.9,
                             vec3(1.0, 1.0, 1.0),
                             0.0, 0.0, 1.0, 1.500));
    world[5] = SDFObject(SHAPE_BOX,
                    Transform(vec3(0, 0, 5.0),
                       rotate(vec3(0, 0, 0)),
                              vec3(2, 1, 0.2)),
                    Material(vec3(1.0, 1.0, 0.2)*0.9,
                             vec3(1.0, 1.0, 1.0),
                             0.0, 1.0, 0.0, 0.470));
    world[6] = SDFObject(SHAPE_BOX,
                    Transform(vec3(0, 0, -1.0),
                       rotate(vec3(0, 0, 0)),
                              vec3(2, 1, 0.2)),
                    Material(vec3(1.0, 1.0, 1.0)*0.5,
                             vec3(1.0, 1.0, 1.0),
                             0.0, 1.0, 0.0, 2.950));
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

fn adjust(rgb: float3, brightness: float, gamma: float) -> float3 {
    return pow(rgb, vec3(gamma)) * brightness;
}

fn spherical_map(p: float3) -> float2 {
    var uv = vec2(atan2(p.z, p.x), asin(p.y));
    return uv * vec2(1.0 / TAU, 1.0 / PI) + 0.5;
}

fn environment(brightness: float, lod: float) -> float4 {
    let uv = spherical_map(-ray.direction);
    let col = textureSampleLevel(channel0, bilinear, uv, lod);
    return vec4(adjust(col.rgb, brightness, 2.2), col.a);
}

fn sd_sphere(p: float3, r: float) -> float {
    return length(p) - r;
}

fn sd_box(p: float3, b: float3) -> float {
    let q = abs(p) - b;
    return length(max(q, vec3(0))) + min(max(q.x, max(q.y, q.z)), 0.0) - 0.03;
}

fn sd_cylinder(p: float3, rh: float2) -> float {
    let d = abs(vec2(length(p.xz), p.y)) - rh;
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2(0)));
}

fn signed_distance(obj: SDFObject, pos: float3) -> float {
    let position = obj.transform.position;
    let rotation = obj.transform.rotation;
    let scale    = obj.transform.scale;
    
    let p = rotation * (pos - position);
    
    switch (obj.shape) {
        case SHAPE_SPHERE:
            { return sd_sphere(p, scale.x); }
        case SHAPE_BOX:
            { return sd_box(p, scale); }
        case SHAPE_CYLINDER:
            { return sd_cylinder(p, scale.xy); }
        default:
            { return sd_sphere(p, scale.x); }
    }
}

fn nearest_object(p: float3, p_idx: ptr<function, u32>) -> float {
    var min_dis = 1e9;
    for (var i = 0u; i < 7u; i++) {
        let new_dis = signed_distance(world[i], p);
        if (new_dis < min_dis) {
            min_dis = new_dis;
            *p_idx = i;
        }
    }
    return min_dis;
}

fn calc_normal(obj: SDFObject, p: float3) -> float3 {
    let e = vec2(1, -1) * 0.5773 * 0.005;
    return normalize( e.xyy*signed_distance(obj, p + e.xyy) + 
                      e.yyx*signed_distance(obj, p + e.yyx) + 
                      e.yxy*signed_distance(obj, p + e.yxy) + 
                      e.xxx*signed_distance(obj, p + e.xxx) );
}

fn hemispheric_sampling(nor: float3) -> float3 {
    let r  = rand2() * vec2(2.0, TAU) - vec2(1, 0);
    let z  = r.x;
    let xy = sqrt(1.0 - z*z) * vec2(sin(r.y), cos(r.y));
    return normalize(nor + vec3(xy, z));
}

fn brightness(rgb: float3) -> float {
    return dot(rgb, vec3(0.299, 0.587, 0.114));
}

fn fresnel_schlick(NoI: float, F0: float, roughness: float) -> float {
    return mix(mix(pow(abs(1.0 + NoI), 5.0), 1.0, F0), F0, roughness);
}

fn sample(uv: float2) -> float4 {
    var color: float4;

    for (var i = 0; i < MAX_RAYTRACE; i++) {
        let roulette_prob = 1.0 - float(i / MAX_RAYTRACE);
        if (rand1() > roulette_prob) {
            ray.color *= 1.0 - roulette_prob;
            break;
        }

        var t = 0.0;
        var w = 1.6;
        var s = 0.0;
        var d = 0.0;
        var index = 0u;
        var hit = false;
        var pos: float3;

        for (var i = 0; i < MAX_RAYMARCH && t < MAX_DIS && !hit; i++) {
            pos = ray.origin + ray.direction * t;
            let dis = abs(nearest_object(pos, &index));
            
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
            ray.color *= environment(1.0 , 0.0).rgb;
            break;
        }

        var normal = calc_normal(world[index], pos);

        let bias = 2.5 * pixel_radius;

        let obj = world[index];

        let albedo       = obj.material.albedo;
        let roughness    = obj.material.roughness;
        let metallic     = obj.material.metallic;
        let transmission = obj.material.transmission;
        let ior          = obj.material.ior;
        
        let outer = dot(normal, ray.direction) < 0.0;
        
        let I  =  ray.direction;
        normal *= select(-1.0, 1.0, outer);
        var N  =  normal;
        var C  =  ray.color;
        var L: float3;
        
        let hemispheric_sample = hemispheric_sampling(normal);
        
        let alpha = roughness * roughness;
        N =  normalize(mix(normal, hemispheric_sample, alpha));

        let NoI = dot(N, I);

        let eta = select(ior / ENV_IOR, ENV_IOR / ior, outer);
        let k   = 1.0 - eta * eta * (1.0 - NoI * NoI);
        var F0  = (eta - 1.0) / (eta + 1.0);
        F0 *= 2.0*F0;
        let F   = fresnel_schlick(NoI, F0, roughness);

        let r2 = rand2();
        if (r2.x < F + metallic || k < 0.0) {
            L = I - 2.0 * NoI * N;
            C *= float(dot(L, normal) > 0.0);
        } else if (r2.y < transmission) {
            L = eta * I - (sqrt(k) + eta * NoI) * N;
        } else {
            L = hemispheric_sample;
        }

        C *= albedo;

        ray.color     = C;
        ray.direction = L;
        ray.origin = pos + normal * bias * sign(dot(L, normal));

        let intensity = brightness(ray.color);
        ray.color    *= obj.material.emission;
        let visible   = brightness(ray.color);
        
        if (intensity < visible || visible < VISIBILITY) { break; }
    
    }

    color = vec4(ray.color, 1.0);

    return color;
}

fn rotate(a: float3) -> mat3x3<float> {
    let s = sin(a); let c = cos(a);
    return mat3x3(vec3( c.z,  s.z,    0),
                  vec3(-s.z,  c.z,    0),
                  vec3(   0,    0,    1)) *
           mat3x3(vec3( c.y,    0, -s.y),
                  vec3(   0,    1,    0),
                  vec3( s.y,    0,  c.y)) *
           mat3x3(vec3(   1,    0,    0),
                  vec3(   0,  c.x,  s.x),
                  vec3(   0, -s.x,  c.x));
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    let screen_size = textureDimensions(screen);
    if (any(id.xy > screen_size)) { return; }
    if (time.frame == 0u) { return; }

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

    color = sample(uv);

    color = vec4(adjust(color.rgb, 1.0, 1.0 / 2.2), color.a);

    data = textureLoad(pass_in, id.xy, 0, 0);

    data *= float(1 - state.moving);
    
    data += color;

    textureStore(pass_out, id.xy, 0, data);
    textureStore(screen, id.xy, float4(data.rgb / data.a, 1.0));
}