// Easy to use template for your first SDF scene
// Other 3D SDF functions: https://gist.github.com/munrocket/f247155fc22ecb8edf974d905c677de1

const EPS = 0.0001;
const FAR = 80.0;
const PI = 3.1415926;

fn rotX(p: vec3f, a: f32) -> vec3f { let r = p.yz * cos(a) + vec2f(-p.z, p.y) * sin(a); return vec3f(p.x, r); }
fn rotY(p: vec3f, a: f32) -> vec3f { let r = p.xz * cos(a) + vec2f(-p.z, p.x) * sin(a); return vec3f(r.x, p.y, r.y); }
fn rotM(p: vec3f, m: vec2f) -> vec3f { return rotY(rotX(p, -PI * m.y), 2 * PI * m.x); }

fn sdSphere(p: vec3f, r: f32) -> f32 {
    return length(p) - r;
}

fn sdBox(p: vec3f, b: vec3f) -> f32 {
  let q = abs(p) - b;
  return length(max(q, vec3f(0.))) + min(max(q.x, max(q.y, q.z)), 0.);
}

fn opSubtract(d1: f32, d2: f32) -> f32 {
    return max(d1, -d2);
}

fn opLimArray(p: vec3f, c: f32, lim: vec3f) -> vec3f {
    return p - c * clamp(round(p / c), -lim, lim);
}

fn map(p0: vec3f) -> f32 {
    let p = p0; //opLimArray(p0, 2.5, vec3f(6.));
    return opSubtract(sdBox(p, vec3f(1)), sdSphere(p, 1.3));
}

fn march(ro: vec3f, rd: vec3f) -> vec3f {
    var p: vec3f;
    var s: f32;
    for (var i = 0; i < 99; i++) {
        p = ro + s * rd;
        let ds = map(p);
        s += ds;
        if (ds < EPS|| s > FAR) { break; }
    }
    return p;
}

fn normal(p: vec3f) -> vec3f {
    let e = vec2f(0., EPS);
    return normalize(vec3f(
        map(p + e.yxx) - map(p - e.yxx),
        map(p + e.xyx) - map(p - e.xyx),
        map(p + e.xxy) - map(p - e.xxy)
    ));
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let res = textureDimensions(screen);
    if (id.x >= res.x || id.y >= res.y) { return; }
    let uv = (2.*(vec2f(id.xy) + .5) - vec2f(res)) / f32(res.y);
    
    var ro = vec3f(0, 0, 4); // Coordinate system: X→, Y↑, Z⊙
    var rd = normalize(vec3f(uv, -2));  
    ro = rotM(ro, vec2f(mouse.pos) / vec2f(res) - .5);
    rd = rotM(rd, vec2f(mouse.pos) / vec2f(res) - .5);

    let p = march(ro, rd);
    let n = normal(p);
    let l = normalize(vec3f(-1,0,1));

    let bg = vec3f(0);
    var col = vec3f(n*.5+.5);
    col = mix(col, bg, map(p));
    textureStore(screen, vec2u(id.x, res.y-1-id.y), vec4f(col, 1.));
}