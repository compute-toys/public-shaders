// Sampling borrowed from oneshade (https://compute.toys/view/1358)
// Hash borrowed from Poisson (https://www.shadertoy.com/view/dssXRj)
#storage image array<array<vec4f, SCREEN_WIDTH>, SCREEN_HEIGHT>

const lDir = normalize(vec3f(5, -5, 5));

var<private> seed: f32;

struct material {
    col: vec3f,
    emission: f32,
    smoothness: f32,
    specular: f32,
    refractIndex: f32
}

fn hash22(p: vec2f) -> vec2f {
	var p3 = fract(vec3f(p.xyx) * vec3f(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

fn hash1() -> f32 {
    seed += 0.1;
    return fract(sin(seed) * 4568.7564);
}

fn hash2() -> vec2f {
    return vec2f(hash1(), hash1());
}

fn hash3() -> vec3f {
    return vec3f(hash1(), hash1(), hash1());
}

fn RandomDir2D() -> vec2f {
    var curr: vec2f;
    while(true) {
        curr = hash2() * 2.0 - 1.0;
        if(length(curr) < 1.0) {
            return curr;
        }
    }
    return vec2f(0);
}

fn RandomDir() -> vec3f {
    var d: vec3f;
    while(true) {
        d = hash3() * 2.0 - 1.0;
        if(length(d) <= 1.0) {
            return normalize(d);
        }
    }
    return vec3f(0);
}

fn rotate2D(t: vec2f) -> mat3x3f {
    var stx = sin(t.x);
    var ctx = cos(t.x);
    var sty = sin(t.y);
    var cty = cos(t.y);
    var xRotation = mat3x3f(
        1, 0, 0,
        0, ctx, -stx,
        0, stx, ctx
    );
    
    var yRotation = mat3x3f(
        cty, 0, -sty,
        0, 1, 0,
        sty, 0, cty
    );
    
    return xRotation * yRotation;
}

// Triplanar mapping function from iq (https://iquilezles.org/articles/biplanar/)
fn triplanar(p: vec3f, n: vec3f, s: f32) -> vec3f {
    var z = textureSampleLevel(channel1, bilinear, fract(p.xy * 1.0), 0.0).rgb;
    var y = textureSampleLevel(channel1, bilinear, fract(p.zx * 1.0), 0.0).rgb;
    var x = textureSampleLevel(channel1, bilinear, fract(p.yz * 1.0), 0.0).rgb;
    
    var sn = abs(n);
    
    return pow((x * sn.x + y * sn.y + z * sn.z) / (sn.x + sn.y + sn.z), vec3f(1.0 / 2.2));
}

fn sphere(p: vec3f, r: f32) -> f32 {
    return length(p) - r;
}

fn box(p: vec3f, s: vec3f, r: f32) -> f32 {
    var q = abs(p) - s;
    return length(max(q, vec3f(0))) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

fn scene(p: vec3f) -> f32 {
    return (box(p - vec3f(0, -2, 0), vec3f(1000, 0.5, 1000), 0.0) - pow(textureSampleLevel(channel0, bilinear, fract(p.xz * 0.01 - 0.0), 0.0).r, 1.0 / 2.2) * 8.0) * 0.5;
}

fn raymarch(o: vec3f, d: vec3f, h: ptr<function, bool>) -> f32 {
    var t = 0.0;
    *h = false;

    for (var i = 0u; i < 300u && *h == false && t <= 1000.0; i++) {
        var s = scene(o + d * t);

        t += s;

        *h = s < t * 0.003 && t >= 0.0;
    }

    return t;
}

fn getNormal(p: vec3f) -> vec3f {
    var e = vec2f(0, 0.001);
    return normalize(vec3f(
        scene(p + e.yxx) - scene(p - e.yxx),
        scene(p + e.xyx) - scene(p - e.xyx),
        scene(p + e.xxy) - scene(p - e.xxy)
    ));
}

// https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
fn ACES(color: vec3f) -> vec3f {	
	var m1 = mat3x3f(
        0.59719, 0.07600, 0.02840,
        0.35458, 0.90834, 0.13383,
        0.04823, 0.01566, 0.83777
	);
	var m2 = mat3x3f(
        1.60475, -0.10208, -0.00327,
        -0.53108,  1.10813, -0.07276,
        -0.07367, -0.00605,  1.07602
	);
	var v = m1 * color;    
	var a = v * (v + 0.0245786) - 0.000090537;
	var b = v * (0.983729 * v + 0.4329510) + 0.238081;
	return clamp(m2 * (a / b), vec3f(0), vec3f(1.0));	
}

#dispatch_once initialize
@compute @workgroup_size(16, 16)
fn initialize(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= SCREEN_WIDTH || id.y >= SCREEN_HEIGHT) { return; }
    image[id.y][id.x] = vec4f(0);
}

fn PathTrace(origin: ptr<function, vec3f>, direction: ptr<function, vec3f>) -> vec3f {
    var col = vec3f(1);
    var depth: f32;
    var initD = *direction;
    for (var i = 0u; i < 12u; i++) {
        var hit = false;
        var normal: vec3f;
        var mat: material;
        var t = raymarch(*origin, *direction, &hit);
        
        if (i < 1u) { depth = t; }

        if (hit) {
            var p = *origin + *direction * t;
            
            *origin = p - normal * 0.01;
            normal = getNormal(p);

            var diffuseDir = normalize(RandomDir() + normal);

            // var reflectDir = normalize(reflect(*direction, normal));
            
            // *direction = mix(diffuseDir, reflectDir, mat.smoothness * specular);

            *direction = diffuseDir;

            col *= triplanar(p, normal, 0.0) * 0.5;
            
            // if(mat.emission > 1.0) { break; }
        } else {
            col *= mix(vec3f(0.6, 0.8, 1), vec3f(1, 0.9, 0.7) * 500.0, pow(max(0.0, dot(*direction, -lDir)), 500.0));
            break;
        }
    }

    return mix(col, mix(vec3f(0.6, 0.8, 1), vec3f(1, 0.9, 0.7) * 500.0, pow(max(0.0, dot(initD, -lDir)), 500.0)), 1.0 - exp(-depth * vec3(0.001, 0.0012, 0.0015)));
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);

    seed = time.elapsed + dot(sin(fragCoord), vec2f(443.712, 983.234));
    seed += hash1() * 434.251;

    // Normalized pixel coordinates (from 0 to 1)
    let uv = ((fragCoord + hash2() - 0.5) - 0.5 * vec2f(screen_size)) / f32(screen_size.y);

    var camRot = rotate2D(radians(vec2f(mouse.pos.yx)));
    var origin = vec3f(0, 0, -100) * camRot;
    var direction = normalize(vec3f(uv, 1.0)) * camRot;
    var focalPoint = origin + direction * 100.0;
    
    origin += (vec3f(RandomDir2D(), 0) * camRot) * 0.1;
    direction = normalize(focalPoint - origin);

    var col = PathTrace(&origin, &direction);

    var sample: vec4f;
    
    if (mouse.click > 0) {
        sample = vec4f(0, 0, 0, 1);
    } else {
        sample = image[id.y][id.x];
    }

    sample.w += 1.0;

    sample = vec4f(mix(sample.rgb, col, 1.0 / sample.w), sample.w);

    image[id.y][id.x] = sample;
    
    col = ACES(image[id.y][id.x].rgb);

    textureStore(screen, id.xy, vec4f(image[id.y][id.x].rgb, 1));
}
