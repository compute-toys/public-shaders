#include "Dave_Hoskins/hash"
const PI: f32 = 3.1415926;

const nSpheres: u32 = 7;
const nLights : u32 = 1;
const nBounces: u32 = 10;

const nShadowRays: u32 = 1;
const nSamples   : u32 = 1;

const nRMSteps: u32 = 32;
const RMSurf: f32 = 0.001;
const RMMiss: f32 = 10;

#storage spheres array<Sphere, nSpheres>
#storage lights  array<Light , nLights >

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);
    // Normalised pixel coordinates (from 0 to 1)
    var uv = fragCoord / vec2f(screen_size) * 2 - 1;
    let last = textureLoad(pass_in, int2(id.xy), 0, 0);
    uv.x *= f32(screen_size.x) / f32(screen_size.y);
    let col = main(uv/2+.5, last);
    //let colCorrected = vec4f(ACESFilm(col.xyz), 1.0);
    let colCorrected = pow(col, vec4f(1.0/2.2));

    textureStore(pass_out, int2(id.xy), 0, col);
    textureStore(screen, id.xy, colCorrected);

}

fn ACESFilm(x: float3) -> float3
{
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x*(a*x + b)) / (x*(c*x + d) + e), float3(0.), float3(1.));
}

var<workgroup> seed: f32;
fn rand(p: vec2f) -> vec3f {
    return hash32(p*(seed+5000+f32(time.elapsed)));
}
fn inUnit(p: vec3f) -> vec3f {
    var P = p*(seed+5000+f32(time.elapsed));
    var r: vec3f = hash33(P)*2-1;
    while (dot(r, r) > 1.0) {
        P *= 2.0;
        r = hash33(P)*2-1;
    }
    if (length(r) < 0.001) { r = vec3f(1,0,0); }
    return normalize(r);
}

fn texture(tex: texture_2d<f32>, uv: vec2f) -> vec4f {
    return textureSampleLevel(tex, nearest, uv, 0);
}

struct Material {
    diffuse: vec3f,
    emission: vec3f,
    refracted: f32,
    ior: f32,

    roughness: f32,
    specular: f32,
    roughSpecular: f32
};

struct Hit {
    didHit: bool,
    pos: vec3f,
    norm: vec3f,
    depth: f32,

    material: Material
};

struct Ray {
    origin: vec3f,
    direction: vec3f
};

struct Light {
    pos: vec3f,
    radius: f32,

    color: vec3f,
    strength: f32
};

fn rAt(r: Ray, t: f32) -> vec3f {
    return r.origin+r.direction*t;
}

    // TODO: lookfrom/lookat camera
fn rCam(uv: vec2f) -> Ray {
    let org = vec3f(0,0.6,-1);

    let ndc = uv * 2 - 1;
    let screen = vec3f(ndc, 0.8) + org;
    let dir = normalize(screen-org);
    return Ray(org, dir);
}

struct Sphere {
    center: vec3f,
    radius: f32,
    material: Material
};

fn hit_sphere(center: vec3f, radius: f32, r: Ray) -> f32 {
    let oc = r.origin - center;
    let a = dot(r.direction, r.direction);
    let half_b = dot(oc, r.direction);
    let c = dot(oc, oc) - radius*radius;
    let discriminant = half_b*half_b - a*c;
    var out: f32;
    if (discriminant < 0) {
        out = -1.0;
    } else {
        out = (-half_b - sqrt(discriminant) ) / a;
    }
    return out;
}

fn sphereIntersect(sph: Sphere, r: Ray) -> Hit {
    let dst = hit_sphere(sph.center, sph.radius, r);
    var h: Hit;
    h.didHit = dst > 0.0;
    if (h.didHit) {
        h.pos = rAt(r, dst);
        h.norm = (h.pos-sph.center)/sph.radius;
        h.depth = dst;
        h.material = sph.material;
    }
    return h;
}

    // adapted from https://iquilezles.org/articles/distfunctions/
fn sdCappedCylinder( p: vec3f,  h: f32, r: f32 ) -> f32
{
  let d = abs(vec2f(length(p.xz),p.y)) - vec2f(r,h);
  return min(max(d.x,d.y),0.0) + length(max(d,vec2f(0.0)));
}

fn box3x3(tex: texture_2d<f32>, uv: vec2f, level: u32) -> vec4f {
    let e = vec3f(-1,0,1);
    var col = vec4f(0.0);
    let R: vec2f = vec2f(textureDimensions(tex).xy);
    let C = uv*R;
    col += textureSampleLevel(tex, nearest, (C+e.xx)/R, 0);
    col += textureSampleLevel(tex, nearest, (C+e.xx)/R, 0);
    col += textureSampleLevel(tex, nearest, (C+e.xx)/R, 0);
    col += textureSampleLevel(tex, nearest, (C+e.xx)/R, 0);
    col += textureSampleLevel(tex, nearest, (C+e.xx)/R, 0);
    col += textureSampleLevel(tex, nearest, (C+e.xx)/R, 0);
    col += textureSampleLevel(tex, nearest, (C+e.xx)/R, 0);
    col += textureSampleLevel(tex, nearest, (C+e.xx)/R, 0);
    col += textureSampleLevel(tex, nearest, (C+e.xx)/R, 0);
    col /= 9;
    return col;
}

    // Ray marching
fn map(p: vec3f, material: ptr<function, Material>) -> float {
    var dist: f32 = RMMiss;
    (*material) = Material(vec3f(1.0), vec3f(0.0), 0.0, 0.0, 1.0, 0.0, 0.0);

    var cyl = sdCappedCylinder(p - vec3f(-0.5, -0.1, 0.5), 0.4, 0.4)-0.05;
    let innerCyl = sdCappedCylinder(p - vec3f(-0.5, 0.1, 0.5), 0.5, 0.37)-0.01;
    cyl = max(cyl, -innerCyl);
    if (cyl < dist) {
        (*material) = Material(vec3f(1.0), vec3f(0.0), 1.0, 2.5, 0.001, 0.15, 0.05);
        dist = cyl;
    }

    var dFloor = p.y + 0.5;
    if (dFloor < dist) {
        if (dFloor < .1) {
            var color: vec3f = textureSampleLevel(channel1, nearest, fract(p.xz/8.0), 0).xyz;
            var height: f32 = length(textureSampleLevel(channel1, nearest, fract(p.xz/8.0), 2).xyz);
            dFloor += height / 5.0;
            (*material) = Material(color, vec3f(0.0), 0.0, 0.0, 0.7, 0.02, 0.1);
        }
        dist = dFloor;
    }

        // milk :)
    var dMilk = sdCappedCylinder(p - vec3f(-0.5, -0.1, 0.5), 0.3, 0.36) - 0.01;
    if (dMilk < dist) {
        (*material) = Material(vec3f(0.8), vec3f(0.0), 0.2, 1.333, 0.8, 0.01, 0.001);
        dist = dMilk;
    }

    return dist;
}
fn normal(pos: vec3f, dist: f32) -> vec3f {
    let epsilon: f32 = RMSurf*2.0;
    var material: Material;
    return normalize(
        vec3f(
            dist - map(pos - vec3f(epsilon, 0, 0), &material),
            dist - map(pos - vec3f(0, epsilon, 0), &material),
            dist - map(pos - vec3f(0, 0, epsilon), &material)
        )
    );
}

fn march(r: Ray) -> Hit {
    var t: f32 = 0.0;
    var scene: f32;
    var p: vec3f;

    var material: Material;

    for (var i: u32 = 0; i < nRMSteps; i++) {
        p = rAt(r, t);
        scene = map(p, &material);
        let uScene = abs(scene);
        if (uScene < RMSurf || uScene > RMMiss) { break; }
        t += uScene;
    }

    let hit: bool = abs(scene) < RMMiss;
    var out: Hit;
    out.didHit = hit;
    if (hit) {
        out.norm = normal(p, scene);
        out.pos = p;
        out.depth = t;
        out.material = material;
    }
    return out;
}

fn sceneIntersect(r: Ray) -> Hit {
    var close = Hit(false, vec3f(0), vec3f(0), 100000, Material(vec3f(0.0), vec3f(0.0), 0.0, 0.0, 0.0, 0.0, 0.0));
    for (var i: u32 = 0; i < nSpheres; i++) {
        let h = sphereIntersect(spheres[i], r);
        if (h.didHit && h.depth < close.depth) { close = h; }
    }

    let RMScene = march(r);
    if ((RMScene.didHit && RMScene.depth < close.depth) || !close.didHit) {
        close = RMScene;
    }

    return close;
}

fn trace(ray: Ray) -> vec3f {
    var r = ray;
    var color: vec3f = vec3f(1.0);
    var light: vec3f = vec3f(0.0);

    var ior: f32 = 1.0;

    for (var bounce: u32 = 0; bounce < nBounces; bounce++) {
        let h = sceneIntersect(r);
        if (!h.didHit || dot(color, color) < 0.01*0.01) { break; } // could do skybox here

            // some hits reflect for "shininess"
        if (hash13(h.pos) > h.material.specular-0.001) {

            var direct: vec3f = vec3f(0.0);
            for (var lI: u32 = 0; lI < nLights; lI++) {
                for (var r: u32 = 0; r < nShadowRays; r++) {
                    let l = lights[lI];
                    let inL = l.pos + l.radius*inUnit(h.pos*f32(r+1));
                    let toL = inL - h.pos;
                    let dstL = length(toL);
                    let falloff = 1.0 / dot(toL,toL); // inverse square law
                    let diffuse = max(0.0, dot(toL, h.norm) / dstL);

                    let shadowR = Ray(h.pos+h.norm*0.001, toL/dstL);
                    let shadowH = sceneIntersect(shadowR);
                    var shadowT = 1.0f;
                    if (shadowH.didHit && shadowH.depth < dstL) { shadowT = 0.0; }

                    direct += diffuse * falloff * l.color * l.strength * shadowT;
                }
            }
                // TODO: make DLS work with <1 roughness
            if (h.material.roughness >= 0.95 && h.material.refracted < 0.05) {
            light += color * h.material.diffuse * (direct/f32(nShadowRays)); }
            light += color * h.material.emission;
            color *= h.material.diffuse;

            var material: Material;
            let scene: f32 = map(h.pos, &material);

            r.origin = h.pos + h.norm*(5.0*RMSurf*sign(scene));
            var d = normalize(inUnit(h.pos) + h.norm);
            let re = reflect(r.direction, h.norm);

            var rd = mix(re, d, h.material.roughness);
            if (hash13(h.pos*86.0) < h.material.refracted+0.001) {
                let n = h.norm * sign(dot(h.norm, -r.direction));
                var newIOR = h.material.ior;
                if (dot(n, h.norm) < 0.0) {
                    newIOR = 1.0;
                }
                rd = refract(r.direction, n, ior/newIOR);
                rd += h.material.roughness*normalize(inUnit(h.pos*323)+r.direction);
                rd = normalize(rd);
                ior = newIOR;
                r.origin = h.pos - h.norm*(5.0*RMSurf*sign(scene));
            }
            r.direction = rd;
        } else {
            light += color * h.material.emission;
            var material: Material;
            let scene: f32 = map(h.pos, &material);
            r.origin = h.pos + h.norm*(5.0*RMSurf*sign(scene));
            let d = normalize(inUnit(h.pos) + h.norm);
            let re = reflect(r.direction, h.norm);
            r.direction = mix(re, d, h.material.roughSpecular);
        }
    }

    return light;
}

fn main(uv: vec2f, last: vec4f) -> vec4f {
    lights[0] = Light(vec3f(0, 1.6, 0.5), 0.3, vec3f(1.0), 1.0);

    //spheres[0] = Sphere(vec3f(-0.5, -0.09, 0.5),    0.4, Material(vec3f(0.8, 0.8, 0.95), vec3f(0.0), 1.0, 2.5, 0.0, 0.15, 0.2));
    spheres[0] = Sphere(vec3f(0.5, -0.1, 0.5),    0.4, Material(vec3f(0.8, 0.8, 0.8), vec3f(0.0), 0.0, 0.0, 1.0, 0.0, 0.0));
    //spheres[1] = Sphere(vec3f(0, -100000.5, 0), 100000, Material(vec3f(0.8), vec3f(0.0), 0.0, 0.0, 1.0, 0.0, 0.0));
    spheres[2] = Sphere(vec3f(-100001.25, 0, 0), 100000, Material(vec3f(0.8, 0.0, 0.0), vec3f(0.0), 0.0, 0.0, 1.0, 0.0, 0.0));
    spheres[3] = Sphere(vec3f( 100001.25, 0, 0), 100000, Material(vec3f(0.0, 0.8, 0.0), vec3f(0.0), 0.0, 0.0, 1.0, 0.0, 0.0));
    spheres[4] = Sphere(vec3f(0, 100001.5, 0), 100000, Material(vec3f(0.8, 0.8, 0.8), vec3f(0.0), 0.0, 0.0, 1.0, 0.0, 0.0));
    spheres[5] = Sphere(vec3f(0, 0, 100001), 100000, Material(vec3f(0.8, 0.8, 0.8), vec3f(0.0), 0.0, 0.0, 1.0, 0.0, 0.0));
    spheres[6] = Sphere(lights[0].pos, lights[0].radius*0.99, Material(vec3f(0.0), lights[0].color, 0.0, 0.0, 1.0, 0.0, 0.0));

    let cam = rCam(uv);
    var col: vec3f;
    for (var s: u32 = 0; s < nSamples; s++) {
        seed += f32(s);
        var c = cam;
        c.origin += 0.002 * inUnit(c.origin);
        var thisC = trace(c);
        thisC = clamp(thisC, vec3f(0), vec3f(1));
            // nans
        if (thisC.x != thisC.x) { thisC.x = 0.0; }
        if (thisC.y != thisC.y) { thisC.y = 0.0; }
        if (thisC.z != thisC.z) { thisC.z = 0.0; }
        
        col += thisC;
    }
    col /= f32(nSamples);
    var out = vec4f(col, 1.0);

    // accumulation
    out = mix(last, out, 1.0/f32(time.frame+1));

    return out;
}