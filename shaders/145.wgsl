var<private> seed = u32(0);
const arpeture_size = 0.0;
const bri = 0.6;
const gamma = 2.2;
const PI = 3.14159265359;
var<private> rgb: vec3<f32>;
var<private> refl: f32;


fn random() -> f32 {
    seed = seed * 747796405u + 2891336453u;
    let word = ((seed >> ((seed >> 28u) + 4u)) ^ seed) * 277803737u;
    return f32((word >> 22u) ^ word) * bitcast<f32>(0x2f800004u);
}

var<private> point: vec3<f32>;
var<private> lightpos: vec3<f32>;

var<private> sde: f32;
var<private> pos: vec3<f32>;
var<private> maxl: f32 = 1000.0;

fn plane_sde(ray: vec3<f32>, nor: vec3<f32>, planey: f32, col: vec3<f32>, r: f32) -> f32 {
    if (dot(ray, nor) + planey < sde) {
        sde = dot(ray, nor) + planey;
        refl = r;
        rgb = col;
    }
    return dot(ray, nor) + planey;
}

fn sphere_sde(ray: vec3<f32>, position: vec3<f32>, radius: f32, col: vec3<f32>, r: f32) -> f32 {
    if (length(position - ray) - radius < sde) {
        sde = length(position - ray) - radius;
        rgb = col;
        refl = r;
    }
    return length(position - ray) - radius;
}

fn calc_sdf(ray: vec3<f32>) -> f32 {
    sde = maxl;
    plane_sde(ray, vec3<f32>(0.0, 1.0, 0.0), 10.0, vec3<f32>(bri, bri, bri), 0.0);
    sphere_sde(ray, vec3<f32>(-10.0, -5.0, 80.0), 5.0, vec3<f32>(bri, bri, bri), 0.5);
    sphere_sde(ray, vec3<f32>(16.0, 5.0, 80.0), 15.0, vec3<f32>(0.0, bri, bri), 0.0);
    sphere_sde(ray, vec3<f32>(-20.0, -5.0, 50.0), 5.0, vec3<f32>(0.0, bri, 0.0), 0.0);
    return sde;
}

fn calcNormal(p: vec3<f32>) -> vec3<f32> {
    let eps: f32 = 0.0001;
    let h = vec2<f32>(eps, 0.0);
    return normalize(vec3<f32>(calc_sdf(p + h.xyy) - calc_sdf(p - h.xyy),
                               calc_sdf(p + h.yxy) - calc_sdf(p - h.yxy),
                               calc_sdf(p + h.yyx) - calc_sdf(p - h.yyx)));
}

fn raycol(orig: vec3<f32>, dir: vec3<f32>, ml: f32) -> vec4<f32> {
    var len: f32 = 0.0;
    sde = ml;
    var steps: f32 = 0.0;
    pos = orig;

    while (len < ml && sde > 0.01) {
        sde = calc_sdf(pos);
        pos += dir * sde;
        len += sde;
        steps += 1.0;
    }
    return vec4<f32>(pos, steps);
}

fn sphrand() -> vec3<f32> {
    let theta = 2.0 * 3.14159265359 * random();
    let phi = acos(2.0 * random() - 1.0);
    let sin_phi = sin(phi);
    return vec3<f32>(sin_phi * cos(theta), sin_phi * sin(theta), cos(phi));
}

fn sample_cosine_hemisphere(normal: vec3<f32>) -> vec3<f32> {
    let cos_theta = 2.0 * random() - 1.0;
    let phi = 2.0 * PI * random();
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let sin_phi = sin(phi);
    let cos_phi = cos(phi);
    let unit_sphere_direction = normalize(vec3(sin_theta * cos_phi, cos_theta, sin_theta * sin_phi));
    return normal + unit_sphere_direction;
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let resolution = vec2<f32>(screen_size);
    let fragCoord = float2(float(id.x) + .5, float(screen_size.y - id.y) - .5);

    seed = time.frame;

    var colour: vec3<f32> = vec3<f32>(0.0);

    for (var i: i32 = 0; i < 1; i = i + 1) {
        let uv = (fragCoord.xy - 0.5 * resolution) / resolution.y + sphrand().xy / resolution.xy;
        let pos = vec3<f32>(sphrand().x * arpeture_size, sphrand().y * arpeture_size, -1.0);
        let direction = normalize(normalize(vec3<f32>(uv.x, uv.y, 1.0)) * 80.0 - pos);

        var o: vec4<f32>;
        var steps: f32;
        var newdir = direction;
        o = raycol(pos, direction, 500.0);
        if (sde <= 0.01) {
            sde = 0.01;
            var tcol: vec3<f32> = vec3<f32>(1.0);
            var bounces: f32 = 0.0;

            while (sde <= 0.01 && bounces < 10.0) {
                let tref = refl;
                var pre: vec3<f32>;
                pre = rgb;
                var offset = sphrand();
                offset *= 20.0;
                lightpos = vec3<f32>(-50.0, 10.0, -10.0) + offset;
                steps = o.w;
                let inter = o.xyz + calcNormal(o.xyz) / 10.0;
                if (tref < random()) {
                    raycol(inter, normalize(lightpos-inter), length(lightpos-inter));
                    if (sde <= 0.01) {
                          pre *= 1.0;
                    } else {
                          pre *= 1.0 + max(dot(calcNormal(inter), normalize(lightpos - inter)), 0.0) * 0.8;
                    }
                    newdir = sample_cosine_hemisphere(calcNormal(o.xyz));          
                } else {
                    newdir = reflect(newdir, calcNormal(o.xyz));
                }
                o = raycol(inter, normalize(newdir), 500.0);
                tcol *= pre;
                bounces += 1.0;
            }
            colour += tcol;
        } else {
            colour += vec3(0.8);
        }
    }

    colour /= 1.0;
    var data = textureLoad(pass_in, id.xy, 0, 0);
    data += vec4(pow(colour, vec3(1.0 / gamma)), 1.0);

    textureStore(pass_out, id.xy, 0, data);
    textureStore(screen, id.xy, float4(data.xyz/data.w, 1.0));
}