override PI = 3.1415926;
override EPS = 0.00001;
override FAR = 100.0;

fn sdSphere(p: vec3f, r: f32) -> f32 {
    return length(p) - r;
}

fn sdBox(p: vec3f, r: f32) -> f32 {
    return max(max(abs(p.x), abs(p.y)), abs(p.z)) - r;
} //length(max(abs(mod(p,c)-0.5*c)-b,0.0))-r

fn opLimArray(p: vec3f, c: f32, lim: vec3f) -> vec3f {
    return p - c * clamp(round(p / c), -lim, lim);
}

fn repeat(p: vec3f) -> vec3f {
    return vec3f(fract(p.x) + 0.8, p.y, p.z);
}

fn map(p0: vec3f) -> f32 {
    let p = p0; //opLimArray(p0, 2.5, vec3f(6.));
    let r = sin(time.elapsed*12)*0.15 + 1.2;
    return sdBox(repeat(p), 1.);
    //return max(sdBox(repeat(p), 1.), -sdSphere(repeat(p), r));
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
    let e = vec2f(EPS, 0.);
    return normalize(vec3f(
        map(p + e.xyy) - map(p - e.xyy),
        map(p + e.yxy) - map(p - e.yxy),
        map(p + e.yyx) - map(p - e.yyx)
    ));
}

fn rotX(p: vec3f, a: f32) -> vec3f { let s = sin(a); let c = cos(a); let r = p.yz * mat2x2f(c, s, -s, c); return vec3f(p.x, r.x, r.y); }
fn rotY(p: vec3f, a: f32) -> vec3f { let s = sin(a); let c = cos(a); let r = p.zx * mat2x2f(c, s, -s, c); return vec3f(r.y, p.y, r.x); }
fn rotM(p0: vec3f, mouse: vec2u, res: vec2u) -> vec3f { let m = vec2f(mouse) / vec2f(res); var p = rotX(p0, PI * (m.y - .5)); return rotY(p, 2 * PI * m.x); }

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let res = textureDimensions(screen);
    if (id.x >= res.x || id.y >= res.y) { return; }
    let fragCoord = vec2f(id.xy) + .5;
    let uv = (2.*fragCoord - vec2f(res)) / f32(res.y);
    
    var ro = vec3f(0,0,5);
    var rd = normalize(vec3f(uv, -2));
    ro = rotM(ro, mouse.pos, res);
    rd = rotM(rd, mouse.pos, res);

    let p = march(ro, rd);
    let n = normal(p);
    let l = normalize(vec3f(-1,0,1));

    let bg = vec3f(0);
    var col = vec3f(n*.5+.5);
    col = mix(col, bg, map(p));
    textureStore(screen, vec2u(id.x, res.y-1-id.y), vec4f(col, 1.));
}