// 色（RGB）を格納するstorage buffer
#storage computeTex array<array<array<atomic<i32>, 3>, SCREEN_HEIGHT>, SCREEN_WIDTH>

//----GPUの処理能力に余裕がある場合は、値を大きくする----
const numSamples = 10;
//-----------------------------------------------

const PI = acos(-1.); // 円周率
const PI2 = PI * 2.;
const BPM = 148.;
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

/*fn hash(p: f32) -> f32 {
    const k = 1103515245u;
    let x = bitcast<u32>(p);
    let uvx = f32(x % 256);
    let uvy = f32((x / 256) % 256);
    return textureSampleLevel(channel0, nearest, vec2f(uvx, uvy) / vec2f(256.), 0.).r; // 0.0～1.0
}*/

// 呼び出される度に異なる値を返す
fn random() -> f32 {
    seed += 1.;
    return hash(seed);
}

// 原点を中心とする半径1の円内部の一様な疑似乱数
fn hash_disc() -> vec2f {
    let r = sqrt(random());
    let a = random() * PI2;
    return vec2f(cos(a), sin(a)) * r;
}

// storage bufferに色（RGB）を加算する
fn add(p: vec2u, v: vec3f) {
    let q = vec3i(v * 2048.);
    atomicAdd(&computeTex[p.x][p.y][0], q.x);
    atomicAdd(&computeTex[p.x][p.y][1], q.y);
    atomicAdd(&computeTex[p.x][p.y][2], q.z);
}

// storage bufferから色（RGB）を読み出す
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

// HSVからRGB色を計算する
fn hsv(h: f32, s: f32, v: f32) -> vec3f {
    var res = fract(h + vec3f(0., 2., 1.) / 3.);
    res = clamp(abs(res * 6. - 3.) - 1., vec3f(0.), vec3f(1.));
    res = (res - 1.) * s + 1.;
    res *= v;
    return res;
}

// 明るすぎる色を抑えるトーンマッピング
// 参考:
// https://hikita12312.hatenablog.com/entry/2017/08/27/002859
fn reinhard(col: vec3f, L: f32) -> vec3f {
    return col / (1. + col) * (1. + col / (L * L));
}

// 矩形波を滑らかにした波形
fn smoothSqWave(p: f32, f: f32) -> f32 {
    let x = p - 0.5;
    let odd = fmod(floor(x), 2.);
    let factor = f * (odd * 2. - 1.);
    let res = smoothstep(0.5 - factor, 0.5 + factor, fract(x));
    return res * 2. - 1.;
}

// 3D空間上の蝶に含まれる点の座標をランダムに返す
fn butterfly(phase: f32) -> vec3f {
    let a = random() * PI2 * 2.;
    let r = sqrt(random()); // 円内部で一様にする
    let s = sign(a - PI2); // 右の羽: -1.0, 左の羽: 1.0
    var p = vec3f(cos(a), 0., sin(a)) * r;
    p.x += s;
    p *= 0.5;
    p.x *= 0.75 + p.z * 1.2; // 蝶の形状を作る

    let si = sin(phase);
    p = vec3f(p.xy * rotate2D(s * si), p.z); // 羽ばたかせる
    p.y += si * 0.5; // 上下に揺らす

    return p * 0.04; // 蝶の大きさを調整
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
    var uv = vec2f(fragCoord * 2. - resolution) / min(resolution.x, resolution.y);
    
    var T = time.elapsed * BPM / 60. * 0.5; // BPMに合わせて速さを変更した時間

    // モーションブラー
    let sampleSeed = dot(fragCoord.xy, vec2(1.3723, 1.8329)) + time.elapsed;
    //let sampleSeed = f32(id.y * screen_size.x + id.x) + time.elapsed * .1;
    seed = sampleSeed;
    // T += random() * 0.03;

    // Cyclic Noiseのパラメーター
    const pers = 0.5;
    const lacu = 1.5;

    // カメラ関係の変数を計算
    var ro = vec3f(0, 0.001, 0.5); // カメラの座標（レイの原点）
    let temp = ro.xz * rotate2D(T * 0.5);
    ro = vec3f(temp.x, ro.y, temp.y); // カメラの座標をy軸まわりに回転させる
    const sp = 0.2; // 蝶が飛ぶスピード
    var ta = vec3f(0.); // カメラのターゲット座標
    ta += cyclic(vec3(T * sp), pers, lacu);
    let dofFocus = length(ta - ro); // DOF（被写界深度）の焦点までの距離
    let dir = normalize(ta - ro); // カメラの方向ベクトル
    let cam = camera(dir); // カメラの姿勢行列
    var fov = 60.; // FOV（視野角）
    fov += smoothSqWave(T * 0.5, 0.1) * 30.; // FOVを時間によって変化させる

    // 個々の蝶の座標や向き、色を計算
    seed = fragCoord.y * 1.3724; // 画面の縦の解像度分の蝶を描画
    var init = vec3(random(), random(), random()) - 0.5;
    init += uv.y * 0.2;
    let fT = fract(T * 2.);
    let bPhase = (fT + random()) * PI2; // 羽の回転に使う角度
    let bPos = cyclic(init + T * sp, pers, lacu); // 3D空間上の蝶の中心の座標
    let bDir = normalize(bPos - cyclic(init - 0.01 + T * sp, pers, lacu)); // 蝶の方向ベクトル
    let bXZ = normalize(vec3(bDir.x, 0, bDir.z));
    let rotYZ = rotate2D(sign(bDir.y) * acos(dot(bXZ, bDir))); // yz平面内（x軸まわり）の回転行列
    let rotXZ = rotate2D(sign(bDir.x) * acos(dot(vec3(0, 0, 1), bXZ))); // xz平面内（y軸まわり）の回転行列
    let bCol = hsv(random(), 0.9, 1.); // 蝶の色

    const tL = 0.5; // トレイル（飛跡）の長さ
    const rate = 0.5; // Storage Bufferにおけるトレイルの使用割合（1.0にすると蝶が消える）
    let x = fragCoord.x / resolution.x / rate;
    let tPos = cyclic(init + T * sp - x * tL, pers, lacu); // トレイル上の点の座標

    for(var i = 0u; i < numSamples; i++) {
        seed = fragCoord.x * 1.6116 + f32(i) + T;
        var pp = butterfly(bPhase); // 蝶に含まれる点の座標

        // 点の座標を蝶の方向に合わせて回転させる
        var temp = pp.yz * rotYZ;
        pp = vec3f(pp.x, temp);
        temp = pp.xz * rotXZ;
        pp = vec3f(temp.x, pp.y, temp.y);

        seed = sampleSeed + f32(i);
        //vec3 pos = x < 1. ? tPos : bPos + pp;
        let pos = select(bPos + pp, tPos, x < 1.); // Storage Bufferの左側でトレイル、右側で蝶を描画する
        let u = proj(pos, ro, cam, fov, dofFocus, 0.05);
        if(u.x < 0 || SCREEN_WIDTH <= u.x || u.y < 0 || SCREEN_HEIGHT <= u.y ) {
            continue; // 画面外は描画しない
        }
        add(vec2u(u), bCol); // 色をStorage Bufferに加算
    }
}

@compute @workgroup_size(16, 16)
fn read_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if(id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    var col = vec3f(0.); // 画面上の色

    let resolution = vec2f(screen_size);
    let fragCoord = vec2f(f32(id.x) + 0.5, f32(screen_size.y - id.y) - 0.5);

    var uv = vec2f(fragCoord * 2. - resolution) / min(resolution.x, resolution.y);

    var T = time.elapsed * BPM / 60. * 0.5; // BPMに合わせて速さを変更した時間

    // モーションブラー
    let sampleSeed = dot(fragCoord, vec2(1.3723, 1.8329)) + time.elapsed;
    //let sampleSeed = f32(id.y * screen_size.x + id.x) + time.elapsed * .1;
    seed = sampleSeed;
    // T += random() * 0.03;

    // 色収差
    var dis = uv * resolution.x * 0.04;
    let amp = pow(sin(T * 8.) * 0.5 + 0.5, 4.);
    dis *= amp;
    let u = abs(fragCoord / resolution - 0.5);
    dis *= smoothstep(0.5, 0.44, max(u.x, u.y));
    // Storage Bufferから色を読み取る
    col.r += load(vec2u(fragCoord + dis)).r;
    col.g += load(vec2u(fragCoord)).g;
    col.b += load(vec2u(fragCoord - dis)).b;
    
    col /= vec3f(f32(numSamples));

    col = reinhard(col, 10.); // トーンマッピング
    col = pow(col, vec3f(1. / 2.2)); // ガンマ補正

    textureStore(screen, id.xy, vec4f(col, 1.));
}
