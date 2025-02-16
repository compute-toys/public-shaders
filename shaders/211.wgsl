const SAMPLES = 8;
const PI = 3.141592564;
const LIGHT_RADIUS = 0.005;

const materials = array<vec3<f32>, 4>(
    vec3<f32>(1.000,1.000,1.000),   // white
    vec3<f32>(1.000,0.067,0.157),   // red
    vec3<f32>(0.027,0.945,0.259),   // green
    vec3<f32>(0.318,0.553,0.992)    // blue
);

const points = array<vec2<f32>, 16>(
    vec2(.1,-.25), 
    vec2(.3,-.25), 
    vec2(.1,-.05),
    vec2(.3,-.05), 
    vec2(-.9,-.4), 
    vec2(.8,-.4),  
    vec2(-.9,-1.), 
    vec2(.8,1.),   
    vec2(-.4,-.3), 
    vec2(-.2,-.3), 
    vec2(-.4,-.1), 
    vec2(-.2,-.1),
    vec2(-.05,-.05),
    vec2(-.05,-.15),
    vec2(0,-.1),
    vec2(-.1,-.1)
);

const segments = array<vec3<i32>, 15>(
    vec3(0,1,1),   // vec3(a,b,c)
    vec3(0,2,1),   // a = endpoint a index
    vec3(1,3,1),   // b = endpoint b index
    vec3(2,3,1),   // c = material index
    vec3(4,5,0),
    vec3(4,6,0),
    vec3(5,7,0),
    vec3(8,9,3),
    vec3(8,10,3),
    vec3(9,11,3),
    vec3(10,11,3),
    vec3(12,14,2),
    vec3(14,13,2),
    vec3(13,15,2),
    vec3(15,12,2)
);

fn segment_intersect(ro: vec2<f32>, rd: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let v1: vec2<f32> = ro - a;
    let v2: vec2<f32> = b - a;
    let v3: vec2<f32> = vec2<f32>(-rd.y, rd.x);

    let d: f32 = dot(v2, v3);
    let t1: f32 = cross(vec3<f32>(v2, 0.0), vec3<f32>(v1, 0.0)).z / d;
    let t2: f32 = dot(v1, v3) / d;

    if (t1 >= 0.0 && (t2 >= 0.0 && t2 <= 1.0)) {
        return t1;
    }
    return 1000.0;
}

fn scene_intersect(ro: vec2<f32>, rd: vec2<f32>) -> vec4<f32> {
    var v0: f32 = 1000.0;
    var col: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);

    for (var i = 0; i < 15; i = i + 1) {
        let a: vec2<f32> = points[segments[i].x];
        let b: vec2<f32> = points[segments[i].y];

        let v1: f32 = segment_intersect(ro, rd, a, b);
        if (v1 < v0) {
            col = materials[segments[i].z];
            v0 = v1;
        }
    }
    return vec4<f32>(col, v0);
}

fn line(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 { 
    var pp = p - a;
    var bb = b - a;
    let h: f32 = clamp(dot(pp, bb) / dot(bb, bb), 0.0, 1.0);
    return length(pp - bb * h);
}

fn scene_dist(p: vec2<f32>) -> vec4<f32> {
    var v0: f32 = 1000.0;
    var col: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);

    for (var i = 0; i < 15; i = i + 1) {
        let a: vec2<f32> = points[segments[i].x];
        let b: vec2<f32> = points[segments[i].y];

        let v1: f32 = line(p, a, b);
        if (v1 < v0) {       
            col = materials[segments[i].z];
            v0 = v1;
        }
    }
    return vec4<f32>(col, v0);
}

// https://blog.demofox.org/2022/01/01/interleaved-gradient-noise-a-different-kind-of-low-discrepancy-sequence
fn interleaved_gradient_noise(pixel_coordinates: vec2<f32>) -> f32 {
    let frame = f32(time.frame % 64u);
    let xy = pixel_coordinates + 5.588238 * frame;
    return fract(52.9829189 * fract(0.06711056 * xy.x + 0.00583715 * xy.y));
}

fn scene_normal(p: vec2<f32>) -> vec2<f32> {
    let epsilon: vec2<f32> = vec2<f32>(0.001, -0.001);
    return normalize(vec2<f32>(scene_dist(p + epsilon.xx).w) - vec2<f32>(scene_dist(p - epsilon.xy).w, scene_dist(p - epsilon.yx).w));
}

fn aces(x: vec3<f32>) -> vec3<f32> {
    let a: f32 = 2.51;
    let b: f32 = 0.03;
    let c: f32 = 2.43;
    let d: f32 = 0.59;
    let e: f32 = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let frag_coord = float2(float(id.x) + .5, float(screen_size.y - id.y) - .5);

    let p: vec2<f32> = (2.0 * frag_coord - float2(screen_size) - 0.5) / f32(screen_size.x);
    let rand = interleaved_gradient_noise(frag_coord.xy);
    
    var spot: vec3<f32>;
    var gi: vec3<f32>;

    let light_pos: vec2<f32> = vec2<f32>(sin(time.elapsed * 0.5) * 0.75, cos(time.elapsed * 0.25) * 0.25 + 0.25);
    var light_dir: vec2<f32> = normalize(vec2<f32>(sin(time.elapsed * 1.5), -1.0));

    let light_falloff = custom.LIGHT_FALLOFF * 3.0;

    if (mouse.click > 0){
        light_dir = normalize(2.0 * vec2<f32>(f32(mouse.pos.x), -f32(mouse.pos.y)) / f32(screen_size.x) - vec2<f32>(1.0, 0.561) - light_pos);
    }

    if (scene_intersect(p, normalize(light_pos - p)).w > distance(p, light_pos)) {
        spot = vec3<f32>(max((0.5 * dot(normalize(p - light_pos), light_dir) - 0.5) / LIGHT_RADIUS + 1.0, 0.0));
    }

    var hit: vec2<f32>;
    for (var i = 0; i < SAMPLES; i = i + 1) {
        var ray_origin: vec2<f32> = light_pos;
        let rot: f32 = 0.08 * PI * ((f32(i) + rand) / f32(SAMPLES) - 0.5) + atan2(light_dir.y, light_dir.x);
        var ray_direction = vec2<f32>(cos(rot), sin(rot));
        let light_dir_sampled = ray_direction;

        var dist: f32 = scene_intersect(ray_origin, ray_direction).w;
        hit = ray_origin + ray_direction * dist;
        let normal: vec2<f32> = scene_normal(hit - ray_direction * 0.01);

        ray_origin = p;
        ray_direction = normalize(hit - p);
        let hit_dist: f32 = min(distance(p, hit) / light_falloff, 1.0);

        var light_ray: vec4<f32> = scene_intersect(ray_origin, ray_direction);
        dist = light_ray.w;

        if (dist + 0.01 > distance(p, hit)) {
            var contribution = 1.0;
            if scene_dist(p).w < .005 {
                contribution = dot(scene_normal(p), light_dir_sampled) * 0.5 + 0.5;
            }

            gi += 1.0 / f32(SAMPLES) * light_ray.rgb * clamp(dot(-ray_direction, normal), 0.0, 1.0) * (1.0 - sqrt(2.0 * hit_dist - hit_dist * hit_dist)) * contribution;
        }
    }

    let scene: vec4<f32> = scene_dist(p);
    var color = spot * 0.5 + gi;
    if scene.w > 0.005 {
        color *= vec3<f32>(0.25);
    } else {
        color *= 3.0 * scene.rgb;
    }

    textureStore(screen, id.xy, float4(aces(color), 1.));
}
