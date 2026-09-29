// 色（RGB）を格納するStorage Buffer
#storage computeTex array<array<array<atomic<i32>, 3>, SCREEN_HEIGHT>, SCREEN_WIDTH>

//----GPUの処理能力に余裕がある場合は、値を大きくする----
//const numSamples = 10;
const numSamples = 1;
//-----------------------------------------------

const PI = acos(-1.); // 円周率
const PI2 = PI * 2.;
const BPM = 132.;

var<private> seed = 0.; // 疑似乱数のシード

// 浮動小数点数の剰余
fn fmod(a: f32, b: f32) -> f32 {
    return a - floor(a / b) * b;
}

// 2Dの回転行列
fn rotate2D(a: f32) -> mat2x2f {
    let s = sin(a);
    let c = cos(a);
    return mat2x2f(c, s, -s, c);
}

// 疑似乱数
fn hash(p: f32) -> f32 {
    const k = 1103515245u;
    var x = bitcast<u32>(p);
    x = ((x >> 8u) ^ x) * k;
    x = ((x >> 8u) ^ x) * k;
    x = ((x >> 8u) ^ x) * k;
    return f32(x) / f32(0xFFFFFFFFu); // 0.0～1.0
}

// 呼び出される度に異なる値を返す
fn random() -> f32 {
    seed += 1.;
    return hash(seed);
}

// 原点を中心とする半径1の円内部に一様分布する疑似乱数
fn hash_disc() -> vec2f {
    let r = sqrt(random());
    let a = random() * PI2;
    return vec2f(cos(a), sin(a)) * r;
}

// Storage Bufferに色（RGB）を加算する
fn add(p: vec2u, v: vec3f) {
    let q = vec3i(v * 2048.);
    atomicAdd(&computeTex[p.x][p.y][0], q.x);
    atomicAdd(&computeTex[p.x][p.y][1], q.y);
    atomicAdd(&computeTex[p.x][p.y][2], q.z);
}

// Storage Bufferから色（RGB）を読み出す
fn load(p: vec2u) -> vec3f {
    return vec3f(f32(atomicLoad(&computeTex[p.x][p.y][0])),
                 f32(atomicLoad(&computeTex[p.x][p.y][1])),
                 f32(atomicLoad(&computeTex[p.x][p.y][2]))) / 2048.;
}

// カメラの姿勢行列
fn camera(direction: vec3f) -> mat3x3f {
    let dir = normalize(direction);
    //let u = abs(dir.y) < 0.999 ? vec3f(0, 1, 0) : vec3f(0, 0, 1);
    let u = select(vec3f(0., 0., 1.), vec3f(0., 1., 0.), abs(dir.y) < 0.999);
    let side = normalize(cross(dir, u));
    let up = cross(side, dir);
    return mat3x3f(side, up, dir);
}

// 3D空間上の点posを、2Dの画面上に投影する
// 参考:evvvvil氏によるDOF（被写界深度）のコード
// https://github.com/evvvvil/bonzomatic-compute-examples/blob/main/03-dof.glsl
fn proj(pos: vec3f, ro: vec3f, camera: mat3x3f, fov: f32, dofFocus: f32, dofAmount: f32) -> vec2i {
    let resolution = vec2f(textureDimensions(screen));
    var p = pos - ro;
    p *= camera;
    if(p.z < 0.) { // カメラの背後は描画しない
        return vec2i(-1, -1);
    }

    p /= vec3f(vec2f(p.z * tan(fov / 360. * PI)), 1.);
    p += vec3f(hash_disc(), 0.) * abs(p.z - dofFocus) * dofAmount;

    //let q = (p.xy * vec2f(resolution.y / resolution.x, 1.) * 0.5 + 0.5) * resolution.xy;
    let q = (p.xy * min(resolution.x, resolution.y) + resolution) * 0.5;
    return vec2i(q);
}

// Cyclic Noise
// 参考:0b5vr氏による記事
// https://scrapbox.io/0b5vr/Cyclic_Noise
fn cyclic(pos: vec3f, pers: f32, lacu: f32) -> vec3f {
    var p = pos;
    var sum = vec4f(0.);
    let rot = camera(vec3f(3., 1., -2.));
    for(var i = 0; i < 5; i++) {
        p *= rot;
        p += sin(p.zxy);
        sum += vec4f(cross(cos(p), sin(p.yzx)), 1.);
        sum /= pers;
        p *= lacu;
    }
    return sum.xyz / sum.w;
}

// 3D空間上の点の座標
fn particle(p: vec3f) -> vec3f {
    const pers = 0.5;
    const lacu = 1.5;
    let res = cyclic(p, pers, lacu);

    return res * 0.03;
}

// Storage Bufferをクリアする
@compute @workgroup_size(16, 16)
fn clear_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if(id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    //computeTex[id.x][id.y] = vec3f(0., 0., 0.);
    atomicStore(&computeTex[id.x][id.y][0], 0);
    atomicStore(&computeTex[id.x][id.y][1], 0);
    atomicStore(&computeTex[id.x][id.y][2], 0);
}

// 色を加算する
@compute @workgroup_size(16, 16)
fn add_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if(id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let resolution = vec2f(screen_size);
    let fragCoord = vec2f(f32(id.x) + 0.5, f32(screen_size.y - id.y) - 0.5);
    // 画面上の座標を正規化
    let uv = vec2f(fragCoord * 2. - resolution) / min(resolution.x, resolution.y);
    let BPMTime = time.elapsed * BPM / 60. * 0.5;

    let sampleSeed = hash(fract(time.elapsed * 0.1) + hash(fragCoord.x + hash(fragCoord.y)));

    var ro = vec3f(0, 0, 0.05); // カメラの座標
    let temp = ro.xz * rotate2D(time.elapsed * 1.);
    ro = vec3f(temp.x, ro.y, temp.y);

    let ta = vec3f(0, 0, 0); // カメラのターゲット座標
    let dir = normalize(ta - ro); // カメラの方向ベクトル
    var fov = 60.; // FOV（視野角）
    let amp = pow(sin(fract(BPMTime * 2.) * PI2) * 0.5 + 0.5, 4.);
    fov -= amp * 10.;
    let cam = camera(dir); // カメラの姿勢行列

    // DOF（被写界深度）のパラメータを設定
    let dofFocus = length(ta - ro); // DOF（被写界深度）の焦点までの距離
    let dofAmount = 3.; // DOF（被写界深度）の浅さ

    // 一定時間毎にノイズを切り替える
    let h = hash(floor(BPMTime));

    for(var i = 0; i < numSamples; i++) {
        seed = sampleSeed + f32(i) * PI;

        // 3D空間上の点の座標
        let pos = particle(vec3f(random() * 0.05, random() * 100. * fract(BPMTime), h * 500.));

        let u = proj(pos, ro, cam, fov, dofFocus, dofAmount);
        if(u.x < 0 || SCREEN_WIDTH <= u.x || u.y < 0 || SCREEN_HEIGHT <= u.y ) {
            continue; // 画面外は描画しない
        }
        add(vec2u(u), vec3f(0.1)); // 色をStorage Bufferに加算
    }
}

@compute @workgroup_size(16, 16)
fn read_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if(id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    var col = vec3f(0.); // 画面上の色

    let resolution = vec2f(screen_size);
    let fragCoord = vec2f(f32(id.x) + 0.5, f32(screen_size.y - id.y) - 0.5);
    // 画面上の座標を正規化
    let uv = vec2f(fragCoord * 2. - resolution) / min(resolution.x, resolution.y);
    let BPMTime = time.elapsed * BPM / 60. * 0.5;

    // 色収差
    var dis = uv * resolution.x * 0.05;
    let amp = pow(sin(fract(BPMTime * 2.) * PI2) * 0.5 + 0.5, 4.);
    dis *= amp;
    let u = abs(fragCoord / resolution - 0.5);
    dis *= smoothstep(0.5, 0.41, max(u.x, u.y));

    // Storage Bufferから色を読み取る
    col.r += load(vec2u(fragCoord + dis)).r;
    col.g += load(vec2u(fragCoord)).g;
    col.b += load(vec2u(fragCoord - dis)).b;

    col /= vec3f(f32(numSamples));

    if(fmod(BPMTime, 2.) < 1.) {
        //col = 0.5 - col;
        const c = pow(0.5, 2.2);
        col = vec3f(c) - col;
    }

    textureStore(screen, id.xy, vec4f(col, 1.));
}