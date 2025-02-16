// CC BY-NC-SA © 2023 munrocket

const EPS = 0.0002;
const FAR = 2.5;
const AA = 1.;

const PI = 3.1415926;
const PHI = 1.6180339;
const TAU = 6.2831853;
const dihedIcos = 0.5 * acos (sqrt (5.) / 3.);
const pentaHeight = 1.5 * acos(dot(normalize(vec3(PHI+1., 1., PHI)), normalize(vec3(.5))));
const R0 = .8;
const R1 = 1.;
const R2 = .007;
const R3 = .004;

fn mod1(x: f32, y: f32) -> f32 { return x - y * floor(x / y); }

fn rot(v: vec2f, a: f32) -> vec2f { return v * cos(a) + vec2f(-v.y, v.x) * sin(a); }
fn rotX(p: vec3f, a: f32) -> vec3f { return vec3f(p.x, rot(p.yz, a)); }
fn rotY(p: vec3f, a: f32) -> vec3f { return vec3f(rot(p.zx, a), p.y).xzy; }
fn rotZ(p: vec3f, a: f32) -> vec3f { return vec3f(rot(p.xy, a), p.z); }
fn rotM(p: vec3f, m: vec2f) -> vec3f { return rotY(rotX(p, PI * m.y), 2. * PI * m.x); }

fn rand(n: f32) -> f32 { return fract(sin(n) * 43758.5453123); }
fn noise(p: f32) -> f32 {
  let fl = floor(p);
  return mix(rand(fl), rand(fl + 1.), fract(p));
}

fn icoSym (q: vec3f) -> vec3f {
  var p = q;
  let dihedIcos = 0.5 * acos (sqrt (5.) / 3.);
  let w = 2. * PI / 3.;
  p.z = abs(p.z);
  p = rotX(p, -dihedIcos);
  p.x = -abs(p.x);
  for (var k = 0; k < 4; k++) {
    p = rotX(p, dihedIcos);
    p.y = -abs(p.y);
    p = rotX(p, -dihedIcos);
    if (k < 3) {
        p = rotZ(p, -w);
    }
  }
  p.z = - p.z;
  let a = mod1(atan2(p.x, p.y) + 0.5 * w, w) - 0.5 * w;
  let t = vec2f(cos(a), sin(a)) * length(p.xy);
  p = vec3f(t.yx, p.z);
  p.x -= 2. * p.x * step (0., p.x);
  return p;
}

fn sdBezier(p: vec2f, A: vec2f, B: vec2f, C: vec2f) -> vec2f {
  let a = B - A;
  let b = A - 2. * B + C;
  let c = a * 2.;
  let d = A - p;
  let kk = 1. / dot(b, b);
  let kx = kk * dot(a, b);
  let ky = kk * (2. * dot(a, a) + dot(d, b)) / 3.;
  let kz = kk * dot(d, a);

  let p1 = ky - kx * kx;
  let p3 = p1 * p1 * p1;
  let q = kx * (2.0 * kx * kx - 3.0 * ky) + kz;
  var h: f32 = q * q + 4. * p3;

  var res: vec2f;
  if (h >= 0.) {
    h = sqrt(h);
    let x = (vec2f(h, -h) - q) / 2.;
    let uv = sign(x) * pow(abs(x), vec2f(1. / 3.));
    let t = clamp(uv.x + uv.y - kx, 0., 1.);
    let f = d + (c + b * t) * t;
    res = vec2f(dot(f, f), t);
  } else {
    let z = sqrt(-p1);
    let v = acos(q / (p1 * z * 2.)) / 3.;
    let m = cos(v);
    let n = sin(v) * 1.732050808;
    let t = clamp(vec2f(m + m, -n - m) * z - kx, vec2f(0.0), vec2f(1.0));
    let f = d + (c + b * t.x) * t.x;
    var dis: f32 = dot(f, f);
    res = vec2f(dis, t.x);

    let g = d + (c + b * t.y) * t.y;
    dis = dot(g, g);
    res = select(res, vec2f(dis, t.y), dis < res.x);
  }
  res.x = sqrt(res.x);
  return res;
}

fn map(v: vec3f) -> f32 {
    let ang = TAU / 15. / floor(1. / custom.nn);
    let R0 = R0 * custom.R0;
    let R1 = R1 * custom.R1;

    var p = icoSym(v);
    var q = p;
    
    var phi = atan2(p.x, p.y);
    var sym = floor((phi + .5 * ang)/ang);
    p = rotZ(p, ang * sym);
    
    var d: f32; var d2: f32; var dr: f32;
    
    dr = R1 * custom.noise1 * noise(sym / custom.noise2);
    
    d = sdBezier(
        p.yz, vec2(R2, R3),
        vec2(mix(-1.,1.,custom.shape1), mix(-2.*R0, 0., custom.shape2)),
        vec2(R1, -R0) + dr).x;
    d = length(vec2(d, p.x)) - R3;
    p = vec3f(p.x, p.y - R1 - dr, p.z + R0 - dr);
    d = min(d, length(p) - R2);
    
    q = rotX(q, -dihedIcos);
    q = rotY(q, pentaHeight);
    
    phi = atan2(q.x, q.y);
    sym = floor((phi + 0.5 * ang)/ang);
    q = rotZ(q, ang * sym);
    
    dr = R1 * custom.noise1 * noise(sym / custom.noise2);
    
    d2 = sdBezier(
        q.yz, vec2(R2, R3),
        vec2(mix(-1.,1.,custom.shape1), mix(-2.*R0, 0., custom.shape2)),
        vec2(R1, -R0) + dr).x;
    d2 = length(vec2(d2, q.x)) - R3;
    q = vec3f(q.x, q.y - R1 - dr, q.z + R0 - dr);
    d = min(d, length(q) - R2);
    
    d = min(d, d2);
    
    return d;
}

fn march(ro: vec3f, rd: vec3f) -> vec3f {
    var t: f32;
    var col: vec3f;
    var p: vec3f;
    let R0 = R0 * custom.R0;
    for(var i = 0; i < 100; i++) {
      	p = ro + t * rd;
        let dt = 0.4 * map(p);
        let c1 = vec3(custom.r1,custom.g1,custom.b1);
        let c2 = vec3(custom.r2,custom.g2,custom.b2);
        col += pow(0.00025/max(0., dt + 0.008), 1.1)       // accum
            * pow(1. - (t + R0 - length(ro)) / FAR, 6.)   // fade
            * mix(c1, c2, length(p)*1.5-.5);              // col
        t += max(EPS, dt);
        if (t > FAR) {break;}
    }
    return col;
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let res = textureDimensions(screen);
    if (id.x >= res.x || id.y >= res.y) { return; }
    var ro: vec3f; var rd: vec3f; var uv: vec2f; var col: vec3f;

    for (var i = 0.; i < AA; i += 1.) {
    for (var j = 0.; j < AA; j += 1.) {
        let dxy = (vec2f(i, j) + .5) / AA;
        uv = (2.*(vec2f(id.xy) + dxy) - vec2f(res)) / f32(res.y);
        ro = vec3(0., 0.08+.08*cos(time.elapsed), 1.65 - .3 * cos(time.elapsed));
        rd = normalize(vec3f(uv, -2));
        let mousepos = .5 - vec2f(mouse.pos) / vec2f(res) + .04 * vec2f(time.elapsed, 0.);
        ro = rotM(ro, mousepos);
        rd = rotM(rd, mousepos);
        col += vec3(custom.r3,custom.g3,custom.b3) + march(ro, rd) / AA / AA;
    }}
    
    textureStore(screen, vec2u(id.x, res.y-1-id.y), vec4f(col*col, 1.));
}