fn opUnion(d1: f32, d2: f32) -> f32 {
  return min(d1, d2);
}

fn opSmoothUnion(d1: f32, d2: f32, k: f32) -> f32 {
  let h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0., 1.);
    return mix(d2, d1, h) - k * h * (1. - h);
}

fn sdSphere(p: vec3f, r: f32) -> f32 {
  return length(p) - r;
}

fn opTranslate(p: vec3f, t: vec3f) -> vec3f {
  return p - t;
}

fn opRotateY(p: vec3f, a: f32) -> vec3f {
  let s = sin(a); let c = cos(a);
    return vec3f(c * p.x - s * p.z, p.y, s * p.x + c * p.z);
}

fn opRotateX(p: vec3f, a: f32) -> vec3f {
  let s = sin(a); let c = cos(a);
    return vec3f(p.x, c * p.y + s * p.z, -s * p.y + c * p.z);
}

fn opRotateZ(p: vec3f, a: f32) -> vec3f {
  let s = sin(a); let c = cos(a);
    return vec3f(c * p.x + s * p.y, -s * p.x + c * p.y, p.z);
}

fn opScale(p: vec3f, s: f32) -> vec3f {
  return p / s;
}

// Auto-generated scene (with materials)

struct SdResult {
  dist:    f32,
  blend:   f32,
  normal:  vec3<f32>,
  ao:      f32,
  color:   vec4<f32>,
  shapeID: u32,
};



fn selectSdResult(a: SdResult, b: SdResult, cond: bool) -> SdResult {
  if (cond) {
    return a;
  } else {
    return b;
  }
}

fn opUnionMat(a: SdResult, b: SdResult) -> SdResult {
  let closer = a.dist < b.dist;
  var out = selectSdResult(a, b, closer);
  out.blend = select(1.0, 0.0, closer);
  return out;
}

fn opSubtractMat(a: SdResult, b: SdResult) -> SdResult {
  var out = a;
  out.dist  = max(a.dist, -b.dist);
  out.blend = 0.0;
  return out;
}

fn opSmoothUnionMat(a: SdResult, b: SdResult, k: f32) -> SdResult {
  let h    = clamp(0.5 + 0.5 * (b.dist - a.dist) / k, 0.0, 1.0);
  let dist = mix(b.dist, a.dist, h) - k * h * (1.0 - h);
  // you can store h and then do the mix(child.color) later
  return SdResult(dist, h, vec3<f32>(0), 1.0, vec4<f32>(0), 0u);
}

fn opSmoothSubtractMat(a: SdResult, b: SdResult, k: f32) -> SdResult {
  let h    = clamp(0.5 - 0.5 * (a.dist + b.dist) / k, 0.0, 1.0);
  let dist = mix(a.dist, -b.dist, h) + k * h * (1.0 - h);
  return SdResult(dist, h, vec3<f32>(0), 1.0, vec4<f32>(0), 0u);
}


fn sdScene(p: vec3<f32>) -> SdResult {
  var p_0: vec3<f32> = p;
  var p_1: vec3<f32> = p_0;
  var p_2: vec3<f32> = p_1;
  p_2 = opTranslate(p_2, vec3<f32>(-0.009656, 0.402661, -0.362939));
  p_2 = opRotateY(p_2, 0.000000);
  p_2 = opRotateX(p_2, 0.000000);
  p_2 = opRotateZ(p_2, 0.000000);
  p_2 = opScale(p_2, 1.000000);
  var d_3: f32       = sdSphere(p_2, 1.000000);
  var r_4: SdResult  = SdResult(d_3, 0.0, vec3<f32>(0), 1.0, vec4<f32>(0.000000,0.400000,1.000000,1.000000), 3u);
  var p_5: vec3<f32> = p_1;
  p_5 = opTranslate(p_5, vec3<f32>(-0.207946, 0.127463, -0.454293));
  p_5 = opRotateY(p_5, 0.000000);
  p_5 = opRotateX(p_5, 0.000000);
  p_5 = opRotateZ(p_5, 0.000000);
  p_5 = opScale(p_5, 1.000000);
  var d_6: f32       = sdSphere(p_5, 1.000000);
  var r_7: SdResult  = SdResult(d_6, 0.0, vec3<f32>(0), 1.0, vec4<f32>(0.466667,1.000000,0.000000,1.000000), 9u);
  var p_8: vec3<f32> = p_1;
  p_8 = opTranslate(p_8, vec3<f32>(0.363921, -0.281342, 0.277673));
  p_8 = opRotateY(p_8, 0.000000);
  p_8 = opRotateX(p_8, 0.000000);
  p_8 = opRotateZ(p_8, 0.000000);
  p_8 = opScale(p_8, 1.000000);
  var d_9: f32       = sdSphere(p_8, 1.000000);
  var r_10: SdResult  = SdResult(d_9, 0.0, vec3<f32>(0), 1.0, vec4<f32>(5.000000,1.000000,0.000000,1.000000), 15u);
  var r_11: SdResult  = r_4;
  var c_12: vec4<f32> = r_11.color;
  var i_13: u32       = r_11.shapeID;
  var r_14: SdResult  = opSmoothUnionMat(r_11, r_7, 0.100000);
  var c_15: vec4<f32> = mix(c_12, r_7.color, r_14.blend);
  var i_16: u32       = select(i_13, r_7.shapeID, r_14.blend > 0.5);
  r_11 = SdResult(r_14.dist, 0.0, vec3<f32>(0), 1.0, c_15, i_16);
  c_12 = c_15;
  i_13 = i_16;
  var r_17: SdResult  = opSmoothUnionMat(r_11, r_10, 0.100000);
  var c_18: vec4<f32> = mix(c_12, r_10.color, r_17.blend);
  var i_19: u32       = select(i_13, r_10.shapeID, r_17.blend > 0.5);
  r_11 = SdResult(r_17.dist, 0.0, vec3<f32>(0), 1.0, c_18, i_19);
  c_12 = c_18;
  i_13 = i_19;
  var r_20: SdResult  = r_11;
  var c_21: vec4<f32> = r_20.color;
  var i_22: u32       = r_20.shapeID;
  return r_20;
}
fn map(p: vec3f) -> f32 {
    return sdScene(p).dist;
}

fn march(ro: vec3f, rd: vec3f) -> vec4f {
    var p: vec3f;
    var s = 0.;
    for(var i = 0; i <= 99; i++) {
        p = ro + rd * s;
        let ds = map(p);
        s += ds;
        if (ds < .001 || s > 80.) { break; }
    }
    return vec4f(p, s / 80.);
}

fn normal(p: vec3f) -> vec3f {
    let e = vec2f(0., 0.0001);
    return normalize(vec3f(
        map(p + e.yxx) - map(p - e.yxx),
        map(p + e.xyx) - map(p - e.xyx),
        map(p + e.xxy) - map(p - e.xxy)
    ));
}

fn normal4(p: vec3f) -> vec3f {
    let e = vec2f(-.5, .5) * 0.001;
    var n = map(p + e.yxx) * e.yxx;
    n += map(p + e.xxy) * e.xxy;
    n += map(p + e.xyx) * e.xyx;
    n += map(p + e.yyy) * e.yyy;
    return normalize(n);
}

fn rotX(p: vec3f, a: f32) -> vec3f { let s = sin(a); let c = cos(a); let r = p.yz * mat2x2f(c, s, -s, c); return vec3f(p.x, r.x, r.y); }
fn rotY(p: vec3f, a: f32) -> vec3f { let s = sin(a); let c = cos(a); let r = p.zx * mat2x2f(c, s, -s, c); return vec3f(r.y, p.y, r.x); }
fn rotM(p: vec3f, m: vec2f) -> vec3f { return rotY(rotX(p, 3.14159265 * m.y), 2. * 3.14159265 * m.x); }

const AA = 3.;
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    let res = textureDimensions(screen);
    if (id.x >= res.x || id.y >= res.y) { return; }
    var ro: vec3f; var rd: vec3f; var uv: vec2f; var col: vec3f;

    for (var i = 0.; i < AA; i += 1.) {
    for (var j = 0.; j < AA; j += 1.) {
        let dxy = (vec2f(i, j) + .5) / AA;
        let uv = (2.*(vec2f(id.xy) + dxy) - vec2f(res)) / f32(res.y);
         // new: orbit twice as far on Z
        var ro = vec3f(0.0, 0.0, 18.4);
        var rd = normalize(vec3f(uv, -2.));

        rd = rotM(rd, vec2f(mouse.pos) / vec2f(res) - .5);
        ro = rotM(ro, vec2f(mouse.pos) / vec2f(res) - .5);

        let m = march(ro, rd);
        let test = sdScene(m.xyz);

        let n = normal(m.xyz);
        let l = normalize(vec3f(-.4, 1., .5));
        let bg = vec3f(0.);
        let c = (n * .5 + .5) * (dot(n, l) * .5 + .5);
        col += select(test.color.rgb, bg, m.w > 1.) / AA / AA;
    }}

    textureStore(screen, vec2u(id.x, res.y-1-id.y), vec4f(col, 1.));
}
