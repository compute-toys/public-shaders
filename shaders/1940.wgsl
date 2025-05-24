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

fn opSmoothSubtract(d1: f32, d2: f32, k: f32) -> f32 {
  let h = clamp(0.5 - 0.5 * (d1 + d2) / k, 0., 1.);
    return mix(d1, -d2, h) + k * h * (1. - h);
}

fn sdBox(p: vec3f, b: vec3f) -> f32 {
  let q = abs(p) - b;
    return length(max(q, vec3f(0.))) + min(max(q.x, max(q.y, q.z)), 0.);
}

fn opTransform(p: vec3f, transform: mat4x4<f32>) -> vec3f {
  let q = transform * vec4f(p, 1.);
    return q.xyz;
}

fn sdEllipsoid(p: vec3f, r: vec3f) -> f32 {
  let k0 = length(p / r);
    let k1 = length(p / (r * r));
    return k0 * (k0 - 1.) / k1;
}

fn sdTorus(p: vec3f, R: f32, r: f32) -> f32 {
  let q = vec2f(length(p.xz) - R, p.y);
    return length(q) - r;
}



fn sdScene(p: vec3<f32>) -> f32 {
  var p_0: vec3<f32> = p;
  p_0 = opTransform(p_0, mat4x4<f32>(0.898549, 0.000000, 0.000000, 0.000000, 0.000000, 0.898549, 0.000000, 0.000000, 0.000000, 0.000000, 0.898549, 0.000000, 1.019985, -0.072677, -4.361393, 1.000000));
  var p_2: vec3<f32> = p_0;
  var p_4: vec3<f32> = p_2;
  var p_6: vec3<f32> = p_4;
  p_6 = opTranslate(p_6, vec3<f32>(0.956082, 0.208331, -7.925380));
  p_6 = opRotateY(p_6, 0.000000);
  p_6 = opRotateX(p_6, 0.000000);
  p_6 = opRotateZ(p_6, 0.000000);
  p_6 = opScale(p_6, 0.500000);
  var d_7: f32       = sdSphere(p_6, 1.000000);
  d_7 = d_7 * 0.556453;
  var p_8: vec3<f32> = p_4;
  p_8 = opTranslate(p_8, vec3<f32>(0.536555, -0.260050, -7.649010));
  p_8 = opRotateY(p_8, 0.000000);
  p_8 = opRotateX(p_8, 0.000000);
  p_8 = opRotateZ(p_8, 0.000000);
  p_8 = opScale(p_8, 0.968609);
  var d_9: f32       = sdSphere(p_8, 1.000000);
  d_9 = d_9 * 1.077970;
  var p_10: vec3<f32> = p_4;
  p_10 = opTranslate(p_10, vec3<f32>(0.801841, -0.090772, -7.735627));
  p_10 = opRotateY(p_10, 0.000000);
  p_10 = opRotateX(p_10, 0.000000);
  p_10 = opRotateZ(p_10, 0.000000);
  p_10 = opScale(p_10, 0.859917);
  var d_11: f32       = sdSphere(p_10, 1.000000);
  d_11 = d_11 * 0.957006;
  var d_5: f32       = opUnion(d_7, d_9);
  d_5 = opUnion(d_5, d_11);
  var p_12: vec3<f32> = p_2;
  var p_14: vec3<f32> = p_12;
  p_14 = opTranslate(p_14, vec3<f32>(-0.755682, 0.132760, -7.878069));
  p_14 = opRotateY(p_14, 0.000000);
  p_14 = opRotateX(p_14, 0.000000);
  p_14 = opRotateZ(p_14, 0.000000);
  p_14 = opScale(p_14, 0.500000);
  var d_15: f32       = sdSphere(p_14, 1.000000);
  d_15 = d_15 * 0.556453;
  var p_16: vec3<f32> = p_12;
  p_16 = opTranslate(p_16, vec3<f32>(-0.435087, -0.440131, -7.489189));
  p_16 = opRotateY(p_16, 0.000000);
  p_16 = opRotateX(p_16, 0.000000);
  p_16 = opRotateZ(p_16, 0.000000);
  p_16 = opScale(p_16, 1.000000);
  var d_17: f32       = sdSphere(p_16, 1.000000);
  d_17 = d_17 * 1.112905;
  var p_18: vec3<f32> = p_12;
  p_18 = opTranslate(p_18, vec3<f32>(-0.593721, -0.190727, -7.703767));
  p_18 = opRotateY(p_18, 0.000000);
  p_18 = opRotateX(p_18, 0.000000);
  p_18 = opRotateZ(p_18, 0.000000);
  p_18 = opScale(p_18, 0.859917);
  var d_19: f32       = sdSphere(p_18, 1.000000);
  d_19 = d_19 * 0.957006;
  var d_13: f32       = opUnion(d_15, d_17);
  d_13 = opUnion(d_13, d_19);
  var p_20: vec3<f32> = p_2;
  p_20 = opTransform(p_20, mat4x4<f32>(1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, -0.425031, 0.089615, -0.077146, 1.000000));
  var p_22: vec3<f32> = p_20;
  p_22 = opTranslate(p_22, vec3<f32>(0.233596, -0.386102, -0.205559));
  p_22 = opRotateY(p_22, 0.000000);
  p_22 = opRotateX(p_22, 0.000000);
  p_22 = opRotateZ(p_22, 0.000000);
  p_22 = opScale(p_22, 0.965349);
  var d_23: f32       = sdBox(p_22, vec3<f32>(0.250000, 1.500000, 1.000000));
  d_23 = d_23 * 1.074342;
  var p_24: vec3<f32> = p_20;
  p_24 = opTranslate(p_24, vec3<f32>(0.049646, 1.740092, -0.717682));
  p_24 = opRotateY(p_24, -0.088847);
  p_24 = opRotateX(p_24, -0.323150);
  p_24 = opRotateZ(p_24, 0.012580);
  p_24 = opScale(p_24, 1.000000);
  var d_25: f32       = sdBox(p_24, vec3<f32>(1.000000, 1.250000, 1.000000));
  d_25 = d_25 * 1.112905;
  var p_26: vec3<f32> = p_20;
  p_26 = opTranslate(p_26, vec3<f32>(1.728225, -1.551498, -2.669053));
  p_26 = opRotateY(p_26, -1.144997);
  p_26 = opRotateX(p_26, -1.299369);
  p_26 = opRotateZ(p_26, 0.885635);
  p_26 = opScale(p_26, 1.000000);
  var d_27: f32       = sdBox(p_26, vec3<f32>(1.000000, 1.250000, 1.000000));
  d_27 = d_27 * 1.112905;
  var p_28: vec3<f32> = p_20;
  p_28 = opTranslate(p_28, vec3<f32>(-0.019461, -0.556386, 1.464793));
  p_28 = opRotateY(p_28, -3.119381);
  p_28 = opRotateX(p_28, -0.710298);
  p_28 = opRotateZ(p_28, -1.946950);
  p_28 = opScale(p_28, 1.000000);
  var d_29: f32       = sdBox(p_28, vec3<f32>(1.000000, 1.250000, 1.000000));
  d_29 = d_29 * 1.112905;
  var d_21: f32       = opSmoothSubtract(d_23, d_25, 1.150000);
  d_21 = opSmoothSubtract(d_21, d_27, 1.150000);
  d_21 = opSmoothSubtract(d_21, d_29, 1.150000);
  var p_30: vec3<f32> = p_2;
  p_30 = opTransform(p_30, mat4x4<f32>(1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.064199, 0.179786, -0.142959, 1.000000));
  var p_32: vec3<f32> = p_30;
  p_32 = opTranslate(p_32, vec3<f32>(0.410738, -0.682779, -4.809686));
  p_32 = opRotateY(p_32, 0.070558);
  p_32 = opRotateX(p_32, 0.000274);
  p_32 = opRotateZ(p_32, 0.004733);
  p_32 = opScale(p_32, 0.946218);
  var d_33: f32       = sdEllipsoid(p_32, vec3<f32>(-0.500000, 1.500000, 4.250000));
  d_33 = d_33 * 1.053051;
  var p_34: vec3<f32> = p_30;
  p_34 = opTranslate(p_34, vec3<f32>(0.152516, -1.761336, -5.498321));
  p_34 = opRotateY(p_34, 0.000000);
  p_34 = opRotateX(p_34, 0.000000);
  p_34 = opRotateZ(p_34, 0.000000);
  p_34 = opScale(p_34, 1.000000);
  var d_35: f32       = sdEllipsoid(p_34, vec3<f32>(0.750000, 1.500000, -2.500000));
  d_35 = d_35 * 1.112905;
  var d_31: f32       = opSmoothUnion(d_33, d_35, 2.100000);
  var p_36: vec3<f32> = p_2;
  var p_38: vec3<f32> = p_36;
  p_38 = opTranslate(p_38, vec3<f32>(1.138073, -1.092974, -5.262986));
  p_38 = opRotateY(p_38, -2.162047);
  p_38 = opRotateX(p_38, 0.010939);
  p_38 = opRotateZ(p_38, 0.000925);
  p_38 = opScale(p_38, 1.000000);
  var d_39: f32       = sdBox(p_38, vec3<f32>(0.010000, 0.250000, 0.750000));
  d_39 = d_39 * 1.112905;
  var p_40: vec3<f32> = p_36;
  p_40 = opTranslate(p_40, vec3<f32>(-1.000000, -1.500000, -5.000000));
  p_40 = opRotateY(p_40, -1.000000);
  p_40 = opRotateX(p_40, 0.000000);
  p_40 = opRotateZ(p_40, 0.000000);
  p_40 = opScale(p_40, 1.000000);
  var d_41: f32       = sdBox(p_40, vec3<f32>(0.010000, 0.250000, 0.750000));
  d_41 = d_41 * 1.112905;
  var d_37: f32       = opUnion(d_39, d_41);
  var p_42: vec3<f32> = p_2;
  var p_44: vec3<f32> = p_42;
  p_44 = opTranslate(p_44, vec3<f32>(0.000000, -0.500000, -8.750000));
  p_44 = opRotateY(p_44, 0.000000);
  p_44 = opRotateX(p_44, -1.500000);
  p_44 = opRotateZ(p_44, 0.000000);
  p_44 = opScale(p_44, 1.000000);
  var d_45: f32       = sdTorus(p_44, 0.300000, 0.000010);
  d_45 = d_45 * 1.112905;
  var d_43: f32       = d_45;
  var d_3: f32       = opSmoothUnion(d_5, d_13, 0.600000);
  d_3 = opSmoothUnion(d_3, d_21, 0.600000);
  d_3 = opSmoothUnion(d_3, d_31, 0.600000);
  d_3 = opSmoothUnion(d_3, d_37, 0.600000);
  d_3 = opSmoothUnion(d_3, d_43, 0.600000);
  var d_1: f32       = d_3;
  return d_1;
}

fn map(p: vec3f) -> f32 {
    return sdScene(p);
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
        let n = normal(m.xyz);
        let l = normalize(vec3f(-.4, 1., .5));
        let bg = vec3f(0.);
        let c = (n * .5 + .5) * (dot(n, l) * .5 + .5);
        col += select(c, bg, m.w > 1.) / AA / AA;
    }}

    textureStore(screen, vec2u(id.x, res.y-1-id.y), vec4f(col, 1.));
}
