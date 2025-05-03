#storage atomic_storage array<atomic<i32>>

const PI = 3.14159265359;
const TAU = 2.0 * PI;
const PARTICLE_COUNT = 4096;
const ATOMIC_SCALE = 1024.0;

struct Camera {
    pos: vec3<f32>,
    cam: mat3x3<f32>,
    fov: f32,
    size: vec2<f32>
}

var<private> camera: Camera;
var<private> state: vec4<u32>;

fn pcg4d(a: vec4<u32>) -> vec4<u32> {
    var v = a * 1664525u + 1013904223u;
    v.x += v.y*v.w; v.y += v.z*v.x; v.z += v.x*v.y; v.w += v.y*v.z;
    v = v ^ (v >> vec4<u32>(16u));
    v.x += v.y*v.w; v.y += v.z*v.x; v.z += v.x*v.y; v.w += v.y*v.z;
    return v;
}

fn rand4() -> vec4<f32> { 
    state = pcg4d(state);
    return vec4<f32>(state) / f32(0xffffffffu); 
}

fn nrand4(sigma: f32, mean: vec4<f32>) -> vec4<f32> {
    let Z = rand4();
    return mean + sigma * sqrt(-2.0 * log(Z.xxyy)) * 
           vec4<f32>(cos(TAU * Z.z), sin(TAU * Z.z), cos(TAU * Z.w), sin(TAU * Z.w));
}

fn disk(rng: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(sin(TAU * rng.x), cos(TAU * rng.x)) * sqrt(rng.y);
}

fn GetCameraMatrix(ang: vec2<f32>) -> mat3x3<f32> {
    let x_dir = vec3<f32>(cos(ang.x) * sin(ang.y), cos(ang.y), sin(ang.x) * sin(ang.y));
    let y_dir = normalize(cross(x_dir, vec3<f32>(0.0, 1.0, 0.0)));
    let z_dir = normalize(cross(y_dir, x_dir));
    return mat3x3<f32>(-x_dir, y_dir, z_dir);
}

fn SetCamera() {
    let screen_size = vec2<f32>(textureDimensions(screen));
    let screen_size_u = vec2<u32>(screen_size);
    let ang = vec2<f32>(mouse.pos.xy) * vec2<f32>(-TAU, PI) / screen_size + vec2<f32>(0.3, 0.4);
    
    camera.fov = 0.8;
    camera.cam = GetCameraMatrix(ang); 
    camera.pos = -(camera.cam * vec3<f32>(12.0, 0.0, 0.0));
    camera.size = screen_size;
}

// Project 3D point to screen space
fn Project(cam: Camera, p: vec3<f32>) -> vec3<f32> {
    let td = distance(cam.pos, p);
    let dir = (p - cam.pos) / td;
    let screen = dir * cam.cam;
    return vec3<f32>(screen.yz * cam.size.y / (cam.fov * screen.x) + 0.5 * cam.size, screen.x * td);
}

fn ForceField(pos: vec3<f32>, t: f32) -> vec4<f32> {
    let a0 = vec3<f32>(
        sin(t * 1.5) * 3.0, 
        cos(t * 0.3) * 2.0, 
        sin(t * 0.7) * cos(t * 0.2) * 3.0
    );
    
    let a1 = vec3<f32>(
        cos(t * 0.6) * 2.0,
        sin(t * 0.4) * 3.0,
        cos(t * 1.5) * sin(t * 0.3) * 2.0
    );
    
    let d0 = distance(pos, a0);
    let d1 = distance(pos, a1);
    
    let F0 = (a0 - pos) * (1.0 / (d0 * d0 * d0 + 1e-3) - 0.4 / (d0 * d0 * d0 * d0 + 1e-3));
    
    let F1 = (a1 - pos) * (1.0 / (d1 * d1 * d1 + 1e-3) - 0.4 / (d1 * d1 * d1 * d1 + 1e-3));
    
    let angle = atan2(pos.z, pos.x);
    let spiral = vec3<f32>(
        cos(angle + d0 * 0.5) * 0.2,
        sin(t * 0.2) * 0.1,
        sin(angle + d0 * 0.5) * 0.2
    );
    
    return 0.15 * vec4<f32>(F0 + F1 + spiral, 0.0);
}

#workgroup_count SimulateParticles 64 64 1
@compute @workgroup_size(16, 16)
fn SimulateParticles(@builtin(global_invocation_id) id: vec3<u32>) {
    let pix = vec2<i32>(id.xy);
    
    if (pix.x >= 64 || pix.y >= 64) {
        return;
    }
    state = vec4<u32>(id.x, id.y, id.z, time.frame);
    var position = textureLoad(pass_in, pix, 0, 0);
    var velocity = textureLoad(pass_in, pix, 1, 0);
    
    if (time.frame == 0u || mouse.click == 1) {
        let rng = rand4();
        
        // Initialize in a sphere
        let theta = rng.x * TAU;
        let phi = acos(2.0 * rng.y - 1.0);
        let r = pow(rng.z, 0.33) * 5.0;
        
        position = vec4<f32>(
            r * sin(phi) * cos(theta),
            r * sin(phi) * sin(theta),
            r * cos(phi),
            1.0
        );
        
        velocity = vec4<f32>(nrand4(0.1, vec4<f32>(0.0)).xyz, 0.0);
    }

    let t = time.elapsed;
    let dt = 0.1;
    let force = ForceField(position.xyz, t);
    velocity += (force - 0.2 * velocity) * dt;
    let freq = 0.5 + 0.5 * sin(t * 0.2);
    
    velocity += vec4<f32>(
        sin(position.y * freq + t) * 0.4,
        cos(position.z * freq + t * 0.7) * 0.02,
        sin(position.x * freq + t * 1.3) * 0.02,
        0.0
    );
    
    position += velocity * dt;
    
    textureStore(pass_out, pix, 0, position);
    textureStore(pass_out, pix, 1, velocity);
}

@compute @workgroup_size(16, 16)
fn Clear(@builtin(global_invocation_id) id: vec3<u32>) {
    let screen_size = vec2<i32>(textureDimensions(screen));
    let idx = i32(id.x) + i32(screen_size.x * i32(id.y));
    
    atomicStore(&atomic_storage[idx * 4 + 0], 0);
    atomicStore(&atomic_storage[idx * 4 + 1], 0);
    atomicStore(&atomic_storage[idx * 4 + 2], 0);
    atomicStore(&atomic_storage[idx * 4 + 3], 0);
}

fn AdditiveBlend(color: vec3<f32>, depth: f32, index: i32) {
   let scaledColor = vec3<i32>(ATOMIC_SCALE* color / (depth * depth + 0.2));
    if (scaledColor.x > 0) {
        atomicAdd(&atomic_storage[index * 4 + 0], scaledColor.x);
    }
    
    if (scaledColor.y > 0) {
        atomicAdd(&atomic_storage[index * 4 + 1], scaledColor.y);
    }
    
    if (scaledColor.z > 0) {
        atomicAdd(&atomic_storage[index * 4 + 2], scaledColor.z);
    }
}

fn RasterizePoint(pos: vec3<f32>, color: vec3<f32>, size: f32) {
    let screen_size = vec2<i32>(camera.size);
    let projectedPos = Project(camera, pos);
    
    let radius = max(4.0, min(15.0, size / (projectedPos.z * 1.1)));
    let samples = max(3.0, radius * 2.0);
    
    for(var i: i32 = 0; i < i32(samples); i++) {
        let rng = rand4();
        let offset = disk(rng.xy) * radius;
        let screenCoord = vec2<i32>(projectedPos.xy + offset);
        
        if (screenCoord.x < 0 || screenCoord.x >= screen_size.x || 
            screenCoord.y < 0 || screenCoord.y >= screen_size.y || 
            projectedPos.z < 0.0) {
            continue;
        }
        
        let idx = screenCoord.x + screen_size.x * screenCoord.y;
        AdditiveBlend(color, projectedPos.z, idx);
    }
}

@compute @workgroup_size(16, 16)
fn Rasterize(@builtin(global_invocation_id) id: vec3<u32>) {
    // Setup camera
    SetCamera();
    
    let pix = vec2<i32>(id.xy);
    if (pix.x >= 64 || pix.y >= 64) {
        return;
    }
    
    state = vec4<u32>(id.x, id.y, id.z, time.frame);
    
    let position = textureLoad(pass_in, pix, 0, 0);
    let velocity = textureLoad(pass_in, pix, 1, 0);
    
    let speed = length(velocity.xyz);
    let dir = normalize(velocity.xyz + vec3<f32>(0.001));
    
    let t = time.elapsed * 0.3;
    let hue = fract(length(position.xyz) * 0.1 + t * 0.05);
    
    let color = 0.5 + 0.5 * vec3<f32>(
        cos(TAU * (hue + 0.0/3.0)),
        cos(TAU * (hue + 1.0/3.0)),
        cos(TAU * (hue + 2.0/3.0))
    );
    
    let finalColor = color * (0.1 + 2.0 * speed);
    
    let particleIndex = pix.x + pix.y * 64;
    let size = 1.0 + 0.0 * speed + sin(time.elapsed + f32(particleIndex) * 0.01) * 0.5;
    
    // Render the particle
    RasterizePoint(position.xyz, finalColor, size);
}

// Sample atomic buffer
fn Sample(pos: vec2<i32>) -> vec3<f32> {
    let screen_size = vec2<i32>(textureDimensions(screen));
    let idx = pos.x + screen_size.x * pos.y;
    
    // Get the atomic values
    let r = f32(atomicLoad(&atomic_storage[idx * 4 + 0]));
    let g = f32(atomicLoad(&atomic_storage[idx * 4 + 1]));
    let b = f32(atomicLoad(&atomic_storage[idx * 4 + 2]));
    
    return vec3<f32>(r, g, b) / ATOMIC_SCALE;
}

fn aces_tonemap(color: vec3<f32>) -> vec3<f32> {
    let a = 1.0;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Final render pass
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    let screen_size = vec2<u32>(textureDimensions(screen));
    
    if (id.x >= screen_size.x || id.y >= screen_size.y) {
        return;
    }
    
    var color = Sample(vec2<i32>(id.xy));
    
    color *= 1.5;
    
    let oldColor = textureLoad(pass_in, vec2<i32>(id.xy), 2, 0).rgb;
    
    if (mouse.click == 1) {
        color = oldColor * 1.95;
    } else {
        color = mix(color, oldColor, 0.8);
    }
    
    let neighbors = array<vec2<i32>, 4>(
        vec2<i32>(-1, 0),
        vec2<i32>(1, 0),
        vec2<i32>(0, -1),
        vec2<i32>(0, 1)
    );
    
    var glow = vec3<f32>(0.0);
    for (var i = 0; i < 12; i++) {
        let npos = vec2<i32>(id.xy) + neighbors[i];
        if (npos.x >= 0 && npos.x < i32(screen_size.x) && 
            npos.y >= 0 && npos.y < i32(screen_size.y)) {
            glow += Sample(npos);
        }
    }
    
    color += glow * 1.1;
    
    color = aces_tonemap(color);
    color = pow(color, vec3<f32>(0.8));
    
    let t = time.elapsed * 0.1;
    let tint = vec3<f32>(
        0.05 * sin(t) + 1.0,
        0.05 * sin(t + PI * 2.0/3.0) + 1.0,
        0.05 * sin(t + PI * 4.0/3.0) + 1.0
    );
    color *= tint;
    
    textureStore(pass_out, vec2<i32>(id.xy), 2, vec4<f32>(color, 1.0));
    
    textureStore(screen, vec2<i32>(id.xy), vec4<f32>(color, 1.0));
}