const SPLAT_ITER = 10;
const SPLAT_ITER_F = float(SPLAT_ITER);
const PI = acos(-1.0);
const TAU = 2.0 * PI;
const DEG2RAD = PI / 180.0;
const FOV = DEG2RAD * 45.0;
const NEAR = 0.1;

#storage computeTex array<array<float3, SCREEN_HEIGHT>, SCREEN_WIDTH>

var<private> seed: uint3;

fn cis(t: f32) -> float2 {
    return float2(cos(t), sin(t));
}

fn rotate2D(t: f32) -> mat2x2<f32> {
    let c = cos(t);
    let s = sin(t);
    return mat2x2<f32>(c, s, -s, c);
}

fn hash3u(sp: uint3) -> uint3 {
    var s = sp * 1145141919u + 1919810u;

    s.x += s.y * s.z;
    s.y += s.z * s.x;
    s.z += s.x * s.y;

    s.x ^= s.x >> 16;
    s.y ^= s.y >> 16;
    s.z ^= s.z >> 16;

    s.x += s.y * s.z;
    s.y += s.z * s.x;
    s.z += s.x * s.y;

    return s;
}

fn hash3f(s: float3) -> float3 {
    let r = hash3u(bitcast<uint3>(s));
    return float3(r) / float(0xffffffffu);
}

fn random3f() -> float3 {
    seed = hash3u(seed);
    return float3(seed) / float(0xffffffffu);
}

fn uniformSphere(xi: float2) -> float3 {
  let phi = TAU * xi.x;
  let sint = 1.0 - 2.0 * xi.y;
  let cost = sqrt(1.0 - sint * sint);
  return float3(
    cost * cos(phi),
    cost * sin(phi),
    sint
  );
}

fn posHolo(prog: float) -> float3 {
    let t = time.elapsed - exp(-2.0 + 2.0 * sin(time.elapsed)) * prog;

    // plane
    var xi = random3f();
    var pos = 2.0 * xi - 1.0;
    pos = float3(pos.x, 0.0, pos.z);

    // edges / dots
    xi = random3f();
    if (xi.x < 0.2) {
        pos.x = sign(pos.x);
    } else if (xi.x < 0.4) {
        pos.z = sign(pos.z);
    } else if (xi.y < 0.1) {
        var v = pos.xz;
        v *= 6.0;
        v = floor(v);
        v += 0.5;
        v /= 6.0;
        pos = float3(v, mix(0.03, 0.06, xi.z)).yzx;
    }

    // noise
    let noisescale = exp2(-2.0 + 2.0 * cyclic(float3(7.0, 2.0, 2.0 * t), 0.5, 1.0).x);
    var np = 2.0 * pos;
    np += cyclic(float3(2.0, 3.0, 2.0 * t), 0.5, 2.0);
    np.z -= 10.0 * t;
    pos += float3(0, 1, 0) * noisescale * cyclic(np, 0.5, 2.0);

    // rotate
    var rt = 4.0 * t;
    rt += cyclic(float3(5.0, 7.0, rt), 0.5, 1.0).x;
    pos = float3(pos.zx * rotate2D(rt), pos.y).yzx;

    // 0b5vr logo
    xi = random3f();
    if (xi.z < 0.2) {
        pos = 0.5 * (random3f() - 0.5);
        xi = random3f();

        // emphasizing cube edge
        if (xi.x < 0.5) {
            pos.y = 0.25 * sign(pos.y);
            pos.z = 0.25 * sign(pos.z);
        } else {
            pos.z = 0.25 * sign(pos.z);
        }

        // each xyz axes
        if (xi.y < 1.0 / 3.0) {
            pos = pos.zxy;
        } else if (xi.y < 2.0 / 3.0) {
            pos = pos.yzx;
        }

        // scale and position
        if (xi.z < 3.0 / 13.0) {
            pos.x *= 3.0;
            pos += float3(0.0, -1.0, 0.0);
        } else if (xi.z < 6.0 / 13.0) {
            pos.x *= 3.0;
            pos += float3(0.0, 1.0, 0.0);
        } else if (xi.z < 9.0 / 13.0) {
            pos.y *= 3.0;
            pos += float3(-1.0, 0.0, 0.0);
        } else if (xi.z < 12.0 / 13.0) {
            pos.y *= 3.0;
            pos += float3(1.0, 0.0, 0.0);
        }
        pos *= 0.3;

        // rotate
        pos = float3(pos.zx * rotate2D(3.1 * t), pos.y).yzx;
        pos = float3(pos.x, pos.yz * rotate2D(2.7 * t));

        // hologram like
        xi = random3f();
        if (xi.x < 0.4) {
            pos.y = floor(pos.y * 80.0) / 80.0;
        }

        // reposition
        pos.y += 0.6;
    }

    // holo
    xi = random3f();
    if (xi.x < 0.2) {
        pos = mix(pos, float3(0.0, 2.0, 0.0), xi.y);
    }

    // arc
    xi = random3f();
    if (xi.x < 0.1) {
        pos.y = mix(0.0, 4.0, floor(100.0 * xi.z) / 100.0);
        pos = float3(
            1.5 * cis(PI * xi.y + 4.0 * t * hash3f(pos.yyy).x),
            pos.y
        ).yzx;
    }

    // color explosion
    pos *= mix(1.0, 1.5, pow(prog, 6.0));

    return pos;
}

fn orthBas(z_: float3) -> mat3x3<f32> {
    let z = normalize(z_);
    let up = select(float3(0.0, 1.0, 0.0), float3(0.0, 0.0, 1.0), abs(z.y) > 0.99);
    let x = normalize(cross(up, z));
    return mat3x3<f32>(x, cross(z, x), z);
}

fn cyclic(p_: float3, pers: f32, lacu: f32) -> float3 {
    var p = p_;
    var b = orthBas(float3(-4, 3, -2));
    var sum = float4(0.0);
    
    for (var i = 0; i < 5; i++) {
        p *= b;
        p += sin(p.zxy);
        sum += float4(
            cross(cos(p), sin(p.yzx)),
            1.0
        );
        sum /= pers;
        p *= lacu;
    }
  
    return sum.xyz / sum.w;
}

fn colfuck(t: f32) -> float3 {
    return 3.0 * (0.5 - 0.5 * cos(TAU * saturate(1.5 * t - 0.25 * float3(0.0, 1.0, 2.0))));
}

fn movefuck(t_: f32, heck: float3) -> float3 {
    var t = t_;
    t += cyclic(heck + 2.0 + t, 0.5, 1.0).x;

    return mix(
        hash3f(heck + floor(t)),
        hash3f(heck + floor(t) + 1.0),
        smoothstep(0.0, 0.1, fract(t))
    );
}

@compute @workgroup_size(16, 16)
fn clear(@builtin(global_invocation_id) id: vec3u) {
    // clear the previous compute tex
    computeTex[id.x][id.y] = float3(0.0, 0.0, 0.0);
}

@compute @workgroup_size(16, 16)
fn splat(@builtin(global_invocation_id) id: uint3) {
    let resolution = textureDimensions(screen);
    if (id.x >= resolution.x || id.y >= resolution.y) { return; }

    let aspect = float(resolution.x) / float(resolution.y);
    seed = uint3(id.xy, time.frame);

    for (var i = 0; i < SPLAT_ITER; i ++) {
        let prog = float(i) / SPLAT_ITER_F + random3f().x;
        let t = time.elapsed - 0.1 * prog;

        var pos = posHolo(prog);

        // camera
        let co = float3(0.0, 0.8, 2.0);
        let ct = float3(0.0, 0.2, 0.0);
        let cb = orthBas(co - ct);
        pos -= co;
        pos *= cb;

        // camera animation
        pos = float3(
            pos.xy * rotate2D(0.4 * movefuck(t, float3(1.0, 2.0, -1.0)).x),
            pos.z
        );
        pos += 0.2 * movefuck(t, float3(6.0, 7.0, -1.0));

        if (pos.z < 0.0) {
            // projection
            pos.y *= -1.0;
            pos /= -pos.z;
            pos *= 2.0;
            pos.x /= aspect;
            pos = 0.5 + 0.5 * pos;

            if (0.0 < pos.x && pos.x < 1.0 && 0.0 < pos.y && pos.y < 1.0) {
                let col = colfuck(prog);
                let screenPos = uint2(pos.xy * float2(resolution));
                computeTex[screenPos.x][screenPos.y] += 0.4 * col / SPLAT_ITER_F;
            }
        }
    }
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    let resolution = textureDimensions(screen);
    if (id.x >= resolution.x || id.y >= resolution.y) { return; }

    let aspect = float(resolution.x) / float(resolution.y);

    // A noise function to add visual effect variations
    var fuck = cyclic(float3(time.elapsed, 1.2, 3.8), 0.5, 2.0);

    // the initial uv
    var uv = float2(id.xy) / float2(resolution);

    if (fuck.x > 0.2) {
        // kaleidoscope effect, modify uv
        uv -= 0.5;
        uv.x *= aspect;
        uv = abs(uv);
        uv *= rotate2D(time.elapsed);
        uv.x /= aspect;
        uv += 0.5;
    }

    var coord = int2((uv % 1.0) * float2(resolution));
    var col = computeTex[coord.x][coord.y];

    // color grading
    col = pow(col, float3(0.4545));
    col = smoothstep(
        float3(0.0, -0.1, -0.2),
        float3(1.0, 1.0, 1.1),
        col
    );
    col = pow(col, float3(2.2));

    if (fuck.y > 0.4) {
        // feedback with negative effect and scaling
        uv -= 0.5;
        uv /= 1.1;
        uv += 0.5;
        var coord = int2((uv % 1.0) * float2(resolution));
        col = mix(
            1.0 - textureSampleLevel(pass_in, bilinear, uv, 0, 0).xyz,
            col,
            0.5
        );
    } else {
        // ordinary denoising feedback
        col = mix(
            textureLoad(pass_in, coord, 0, 0).xyz,
            col,
            0.2
        );
    }

    textureStore(screen, id.xy, float4(col, 1.0));
    textureStore(pass_out, id.xy, 0, float4(col, 1.0));
}
