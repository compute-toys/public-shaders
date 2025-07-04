// CC BY-NC-SA © 2025 munrocket

const AA = 1.;
const PI = 3.14159265;

const H = 0.6;
const R = 1.8;
const S = .01;

const COUNT = 256*32;
#define COUNT_256 32

#calcdefine TILES_X SCREEN_WIDTH / 16
#calcdefine TILES_Y SCREEN_HEIGHT / 16
#define TILE_CAPACITY 1024

struct Particle { p: vec4f, v: vec4f }
struct Tile {
    amount: atomic<u32>,
    ids: array<u32, TILE_CAPACITY>
}

#storage tiles array<array<Tile, TILES_Y>, TILES_X>
#storage particle array<Particle, COUNT>

alias packed = vec4u;
var<workgroup> wg_mem: array<packed, TILE_CAPACITY>;
fn pack_Particle(s: Particle) -> packed {
    return vec4u(
        pack2x16float(vec2f(s.p.x, s.p.y)),
        pack2x16float(vec2f(s.p.z, s.p.w)),
        pack2x16float(vec2f(s.v.x, s.v.y)),
        pack2x16float(vec2f(s.v.z, s.v.w))
    );
}
fn unpack_Particle(data: packed) -> Particle {
    return Particle(
        vec4f(unpack2x16float(data.x), unpack2x16float(data.y)),
        vec4f(unpack2x16float(data.z), unpack2x16float(data.w))
    );
}

fn pcg3df(p: vec3u) -> vec3f {
    var v = p * 1664525u + 1013904223u;
    v.x += v.y*v.z; v.y += v.z*v.x; v.z += v.x*v.y;
    v ^= v >> vec3u(16u);
    v.x += v.y*v.z; v.y += v.z*v.x; v.z += v.x*v.y;
    return vec3f(v) / f32(0xffffffff);
}

fn rot(v: vec2f, a: f32) -> vec2f { return v * cos(a) + vec2f(v.y, -v.x) * sin(a); }
fn rotX(p: vec3f, a: f32) -> vec3f { return vec3f(p.x, rot(p.yz, a)); }
fn rotY(p: vec3f, a: f32) -> vec3f { return vec3f(rot(p.xz, a), p.y).xzy; }
fn rotM(p: vec3f, m: vec2f) -> vec3f { return rotX(rotY(p, 2. * 3.14159265 * m.x), 3.14159265 * (.5 - m.y)); }

fn project(v: vec4f, res: vec2f) -> vec3f {
    var p = rotM(v.xyz, vec2f(mouse.pos+vec2i(0,i32(res.y*.2))) / res);
    p.y += .2;
    p.z = 1.5*mouse.zoom - p.z;
    let rmax = max(res.x, res.y);
    let uv = vec2f(p.xy) / p.z;
    let pixel = (uv * rmax + res) * .5;
    let size = v.w * rmax / p.z;
    return vec3f(pixel, size);
}

fn sine(x: f32) -> f32 {
    return .6 * sin(x) + .3 * cos(2 * x) + .1 * sin(4 * x);
}
fn wave(a: f32, r: f32, q: f32) -> f32 {
    return sine(q * (a*3. + 30.*pow(r, 0.3)));
}
fn fbm(a: f32, r: f32) -> f32 {
    var c = 1.; let p = 1.3; let q = 0.9;
    var f = c * wave(a, r, 1); c *= q;
    f = f + c * wave(a, r, p); c *= q;
    f = f + c * wave(a, r, p * p); c *= q;
    f = f + c * wave(a, r, p * p * p); c *= q;
    f = f / 3.43;
    return f;
}

#dispatch_once init
@compute @workgroup_size(256)
#workgroup_count init COUNT_256 1 1
fn init(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wid: vec3u
) {
    let random = pcg3df(vec3(gid.x, wid.x, lid.x));
    let r = R * pow(random.x, 0.9) + .08;
    let a = random.y * 2 * PI;
    let x = r * sin(a);
    let y = H * fbm(a, r) * exp(-r*r*1.5);
    let z = r * cos(a);
    let w = S + S * random.z; 
    let v = pow(custom.g*(.4+y*.15)/(r), 0.5);
    let o = vec3f(custom.ox, custom.oy, custom.oz);
    let s = Particle(
        vec4f(x, y, z, w),
        vec4f(v * cross(o - vec3f(x, y, z), vec3f(0,1,0)), 0)
    );
    particle[gid.x] = s;
}

@compute @workgroup_size(1)
#workgroup_count clear TILES_X TILES_Y 1
fn clear(@builtin(global_invocation_id) wid: vec3u) {
    atomicStore(&(tiles[wid.x][wid.y].amount), 0u);
}

fn acc(r: vec3f) -> vec3f {
    return -normalize(r) * custom.g * (max(exp(-dot(r,r)*4.), .95) + .01);
}

@compute @workgroup_size(256)
#workgroup_count setup COUNT_256 1 1
fn setup(@builtin(global_invocation_id) gid: vec3u) {
    //physics
    var s = particle[gid.x];
    let dt = time.delta * custom.time;
    let m = 1.;//s.p.w*s.p.w;
    let o = vec3f(custom.ox, custom.oy, custom.oz);
    var a = acc(s.p.xyz - o);
    s.p += vec4f((s.v.xyz + a * dt * 0.5) * dt, 0.);
    a += acc(s.p.xyz - o);
    s.v += vec4f(a * 0.5 * dt, 0.);
    particle[gid.x] = s;

    let res = vec2f(textureDimensions(screen));
    let sp = project(s.p, res);
    let lb = (sp.xy - sp.z) / 16. - 1.;
    let rt = min(res / 16., (sp.xy + sp.z) / 16.);
    var id: u32;
    for (var i = 0.; i < TILES_X; i += 1.) {
        for (var j = 0.; j < TILES_Y; j += 1.) {
            if (lb.x < i && i < rt.x && lb.y < j && j < rt.y) {
                id = atomicAdd(&(tiles[u32(i)][u32(j)].amount), 1u);
                tiles[u32(i)][u32(j)].ids[id % TILE_CAPACITY] = gid.x;
            }
        }
    }
}

@compute @workgroup_size(16, 16)
#workgroup_count rasterizer TILES_X TILES_Y 1
fn rasterizer(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(local_invocation_index) ldx: u32,
    @builtin(workgroup_id) wid: vec3u
) {
    // copy to wg memory
    for (var i = 0u; i < u32(TILE_CAPACITY / 16 / 16); i++) {
        let id = i * 16u * 16u + ldx;
        wg_mem[id] = pack_Particle(particle[tiles[wid.x][wid.y].ids[id]]);
    }
    workgroupBarrier();

    let amount = max(min(atomicLoad(&tiles[wid.x][wid.y].amount), TILE_CAPACITY), 0);
    let res = vec2f(textureDimensions(screen));
    var col = vec3f(0.001, .0, .001);
    //col += vec3f(1., 5., 1.)*f32(amount)/TILE_CAPACITY;
    for (var i = 0.; i < AA; i += 1.) { for (var j = 0.; j < AA; j += 1.) {
        let dxy = (vec2f(i, j) + .5) / AA;
        let pix = vec2f(gid.xy) + dxy;
        for (var id = 0u; id < amount; id++) {
            let s = unpack_Particle(wg_mem[id]);
            let p = project(s.p, res);
            let L = length(s.p);
            var size = p.z;
            size *= exp(-L*L/12.);
            let r = length(pix - p.xy);
            let f = exp(-L*L/20.) * (1. - exp(-max(0, pow(.04*size/r, 1.2) - .01)));
            col += .7*mix(vec3f(.2,.5,1.), vec3f(1.1, .5, .2), length(s.v*1.5)) * f / AA / AA;
        }
    }}
    textureStore(screen, vec2u(gid.x, u32(res.y)-1-gid.y), vec4f(pow(col, vec3(2.2)), 1.));
}