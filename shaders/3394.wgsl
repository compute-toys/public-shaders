#storage world array<u32>

// ===== 世界：64 x 16 x 64 = 65536 格 =====
const WX : i32 = 64;
const WY : i32 = 16;
const WZ : i32 = 64;

// 状态槽紧跟在方块后面：65536 = pos.x, +1 = pos.y, +2 = pos.z, +3 = 帧号
const STATE : u32 = 65536u;

const EYE   : f32 = 1.62;
const SPEED : f32 = 6.0;
const FOV_K : f32 = 1.0;

fn inBounds(p: vec3i) -> bool {
    return all(p >= vec3i(0)) && all(p < vec3i(WX, WY, WZ));
}

fn idxOf(p: vec3i) -> u32 {
    return u32(p.x) + u32(p.z) * u32(WX) + u32(p.y) * u32(WX * WZ);
}

fn heightAt(x: i32, z: i32) -> i32 {
    let fx = f32(x);
    let fz = f32(z);
    let hf = 5.0 + 2.5 * sin(fx * 0.18) * cos(fz * 0.15)
                 + 1.8 * sin(fx * 0.07 + fz * 0.11)
                 + 0.6 * sin(fx * 0.53) * sin(fz * 0.47);
    return i32(clamp(hf, 0.0, f32(WY) - 1.0));
}

fn getBlock(p: vec3i) -> u32 {
    if (!inBounds(p)) { return 0u; }
    let e = world[idxOf(p)];
    if (e != 0u) { return e - 1u; }
    let h = heightAt(p.x, p.z);
    if (p.y < h)  { return 2u; }
    if (p.y == h) { return 1u; }
    return 0u;
}

fn setBlock(p: vec3i, v: u32) {
    if (!inBounds(p)) { return; }
    world[idxOf(p)] = v + 1u;
}

fn groundAt(x: f32, z: f32) -> f32 {
    let bx = i32(floor(x));
    let bz = i32(floor(z));
    var top = 0;
    for (var y = WY - 1; y >= 0; y = y - 1) {
        if (getBlock(vec3i(bx, y, bz)) != 0u) {
            top = y + 1;
            break;
        }
    }
    return f32(top);
}

// ===== 玩家状态 =====
fn loadPos() -> vec3f {
    return vec3f(bitcast<f32>(world[STATE]),
                 bitcast<f32>(world[STATE + 1u]),
                 bitcast<f32>(world[STATE + 2u]));
}

fn storePos(p: vec3f, f: u32) {
    world[STATE]      = bitcast<u32>(p.x);
    world[STATE + 1u] = bitcast<u32>(p.y);
    world[STATE + 2u] = bitcast<u32>(p.z);
    world[STATE + 3u] = f;
}

// ★ 这里修好了上下颠倒：y 取反，让屏幕顶部对应 +y
fn screenUV(px: vec2f, size: vec2f) -> vec2f {
    let n = px / size - vec2f(0.5);
    return vec2f(n.x * (size.x / size.y), -n.y);
}

// ===== 体素 DDA =====
struct Hit {
    hit  : bool,
    cell : vec3i,
    nrm  : vec3i,
    t    : f32,
};

fn raycast(ro: vec3f, rd: vec3f) -> Hit {
    var h: Hit;
    h.hit  = false;
    h.cell = vec3i(0, 0, 0);
    h.nrm  = vec3i(0, 0, 0);
    h.t    = 0.0;

    let inv = 1.0 / rd;
    let ta  = (vec3f(0.0, 0.0, 0.0) - ro) * inv;
    let tb  = (vec3f(f32(WX), f32(WY), f32(WZ)) - ro) * inv;
    let lo  = min(ta, tb);
    let hi  = max(ta, tb);
    let tn  = max(max(lo.x, lo.y), lo.z);
    let tf  = min(min(hi.x, hi.y), hi.z);
    if (tf < max(tn, 0.0)) { return h; }

    var t  = max(tn, 0.0) + 1e-4;
    let p0 = ro + rd * t;
    var c  = vec3i(floor(p0));

    var sx: i32 = 0;
    var sy: i32 = 0;
    var sz: i32 = 0;
    var dx: f32 = 1e30;
    var dy: f32 = 1e30;
    var dz: f32 = 1e30;
    var tx: f32 = 1e30;
    var ty: f32 = 1e30;
    var tz: f32 = 1e30;

    if (abs(rd.x) > 1e-8) {
        sx = i32(sign(rd.x));
        dx = abs(1.0 / rd.x);
        tx = t + (f32(c.x) + step(0.0, rd.x) - p0.x) / rd.x;
    }
    if (abs(rd.y) > 1e-8) {
        sy = i32(sign(rd.y));
        dy = abs(1.0 / rd.y);
        ty = t + (f32(c.y) + step(0.0, rd.y) - p0.y) / rd.y;
    }
    if (abs(rd.z) > 1e-8) {
        sz = i32(sign(rd.z));
        dz = abs(1.0 / rd.z);
        tz = t + (f32(c.z) + step(0.0, rd.z) - p0.z) / rd.z;
    }

    var n = vec3i(0, 0, 0);
    for (var k = 0; k < 256; k++) {          // 世界变大，步数上限也抬上去
        if (!inBounds(c)) { break; }
        if (getBlock(c) != 0u) {
            h.hit  = true;
            h.cell = c;
            h.nrm  = n;
            h.t    = t;
            return h;
        }
        if (tx < ty && tx < tz) {
            t = tx; c.x = c.x + sx; tx = tx + dx; n = vec3i(-sx, 0, 0);
        } else if (ty < tz) {
            t = ty; c.y = c.y + sy; ty = ty + dy; n = vec3i(0, -sy, 0);
        } else {
            t = tz; c.z = c.z + sz; tz = tz + dz; n = vec3i(0, 0, -sz);
        }
    }
    return h;
}

// ===== 入口 =====
@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let res  = vec2f(f32(SCREEN_WIDTH), f32(SCREEN_HEIGHT));
    var mpos = vec2f(mouse.pos);
    if (mpos.x <= 0.0 && mpos.y <= 0.0) { mpos = res * 0.5; }

    // 鼠标位置 -> 朝向
    let yaw   = (mpos.x / res.x - 0.5) * 4.0;
    let pitch = -(mpos.y / res.y - 0.5) * 1.4;

    let fw = vec3f(sin(yaw) * cos(pitch), sin(pitch), cos(yaw) * cos(pitch));

    // ---- 相机位置：每帧只更新一次 ----
    var pos = loadPos();
    if (bitcast<f32>(world[STATE + 1u]) == 0.0) {
        pos = vec3f(32.0, 14.0, 6.0);            // 新出生点（世界中心偏南）
    }

    if (world[STATE + 3u] != time.frame) {
        let rgt = vec3f(-cos(yaw), 0.0, sin(yaw));

        var input = vec3f(0.0);
        if (keyDown(87)) { input = input + fw * vec3f(1.0, 0.0, 1.0); }  // W
        if (keyDown(83)) { input = input - fw * vec3f(1.0, 0.0, 1.0); }  // S
        if (keyDown(68)) { input = input + rgt; }                        // D
        if (keyDown(65)) { input = input - rgt; }                        // A

        if (length(input) > 1e-5) {
            input = normalize(input) * SPEED * time.delta;
            let nx = pos.x + input.x;
            let nz = pos.z + input.z;
            let g  = groundAt(nx, nz);
            if (g - (pos.y - EYE) <= 1.2) {
                pos = vec3f(nx, g + EYE, nz);
            }
        } else {
            pos = vec3f(pos.x, groundAt(pos.x, pos.z) + EYE, pos.z);
        }

        if (id.x == 0u && id.y == 0u) { storePos(pos, time.frame); }
    }

    // ---- 挖掘 / 放置 ----
    if (id.x == 0u && id.y == 0u && mouse.click != 0) {
        let mh = raycast(pos, fw);
        if (mh.hit) {
            if (mouse.click == 1) {
                setBlock(mh.cell, 0u);
            } else {
                var mat = 1u;
                if (keyDown(50)) { mat = 2u; }
                setBlock(mh.cell + mh.nrm, mat);
            }
        }
    }

    // ---- 渲染 ----
    let rt = normalize(cross(fw, vec3f(0.0, 1.0, 0.0)));
    let up = cross(rt, fw);
    let uv = screenUV(vec2f(id.xy), res);
    let rd = normalize(uv.x * rt + uv.y * up + FOV_K * fw);

    let h = raycast(pos, rd);
    var col: vec3f;

    if (h.hit) {
        let p = pos + rd * h.t;
        let f = p - vec3f(h.cell);

        var tuv: vec2f;
        if (h.nrm.x != 0)      { tuv = vec2f(f.z, 1.0 - f.y); }
        else if (h.nrm.y != 0) { tuv = vec2f(f.x, f.z); }
        else                   { tuv = vec2f(f.x, 1.0 - f.y); }

        var tex: vec3f;
        if (getBlock(h.cell) == 2u) {
            tex = textureSampleLevel(channel1, nearest, tuv, 0.0).rgb;
        } else {
            tex = textureSampleLevel(channel0, nearest, tuv, 0.0).rgb;
        }

        var shade = 0.80;
        if (h.nrm.y != 0)      { shade = select(0.5, 1.0, h.nrm.y > 0); }
        else if (h.nrm.x != 0) { shade = 0.70; }

        col = tex * shade;
    } else {
        col = mix(vec3f(0.55, 0.72, 0.95), vec3f(0.15, 0.38, 0.80),
                  clamp(rd.y * 1.5 + 0.5, 0.0, 1.0));
    }

    // 屏幕中心准星
    let dc = abs(vec2f(id.xy) - res * 0.5);
    if (max(dc.x, dc.y) < 2.0) { col = vec3f(1.0); }

    textureStore(screen, id.xy, vec4f(col, 1.0));
}

//LZX-2026-11-27-001