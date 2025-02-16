// CC BY-NC-SA © 2023 munrocket

const AA = 1.;

const COUNT = 256*32;
#define COUNT_256 32
#define point_size 1.41421356237 * .7

// change after preprocessor eval feature:
#define TILES_X 70 //SCREEN_WIDTH/16
#define TILES_Y 50 //SCREEN_HEIGHT/16
#define TILE_CAPACITY 2048
struct Tile {
    amount: atomic<u32>,
    ids: array<u32, TILE_CAPACITY>
}

#storage tiles array<array<Tile, TILES_Y>, TILES_X>
#storage points array<vec3f,COUNT>
var<workgroup> ids: array<u32, TILE_CAPACITY>;

@compute @workgroup_size(256)
#workgroup_count init COUNT_256 1 1
fn init(@builtin(global_invocation_id) gid: vec3u) {

    // LORENZ
    var h = .02 * custom.h;
    var s = 20. * custom.s;
    var b = 4. * custom.b;
    var r = 40. * custom.r;
    var x0 = mix(-10., 10., custom.x0);
    var y0 = mix(-10., 10., custom.y0);
    var z0 = mix(-30., 30., custom.z0);
    var x1: f32;    var y1: f32;    var z1: f32;
    for (var i = 0u; i < gid.x; i++) {
        x1 = x0 + h * s * (y0 - x0);
        y1 = y0 + h * (x0 * (r - z0) - y0);
        z1 = z0 + h * (x0 * y0 - b * z0);
        x0 = x1; y0 = y1; z0 = z1;
    }
    points[gid.x] = vec3f(y0, z0-r, x0);

    /*// AIZAWA
    var h = .01;
    var a = 0.95;   var b = .7;     var c = 1.;
    var d = 3.5;    var e = .25;    var f = .1;
    var x0 = 1.;    var y0 = -1.;   var z0 = 1.;
    var x1: f32;    var y1: f32;    var z1: f32;
    for (var i = 0u; i < gid.x; i++) {
        x1 = x0 + h * ((z0 - b) * x0 - d * y0);
        y1 = y0 + h * (d * x0 + (z0 - b) * y0);
        z1 = z0 + h * (c + a * z0 - z0*z0*z0/3.
            - (x0*x0 + y0*y0) * (1. + e * z0) + f * z0 * x0*x0*x0);
        x0 = x1; y0 = y1; z0 = z1;
    }
    points[gid.x] = vec3f(10.*y0, 10.*z0-5., 10.*x0);*/

    /*//HALVORSEN
    var h = .01;
    var a = 1.89;
    var x0 = 3.;    var y0 = -3.;   var z0 = -3.;
    var x1: f32;    var y1: f32;    var z1: f32;
    for (var i = 0u; i < gid.x; i++) {
        x1 = x0 + h * (-a * x0 - 4. * y0 - 4. * z0 - y0*y0);
        y1 = y0 + h * (-a * y0 - 4. * z0 - 4. * x0 - z0*z0);
        z1 = z0 + h * (-a * z0 - 4. * x0 - 4. * y0 - x0*x0);
        x0 = x1; y0 = y1; z0 = z1;
    }
    points[gid.x] = vec3f(x0, y0, z0);*/

}

fn rotX(p: vec3f, a: f32) -> vec3f { let s = sin(a); let c = cos(a); let r = p.yz * mat2x2f(c, s, -s, c); return vec3f(p.x, r.x, r.y); }
fn rotY(p: vec3f, a: f32) -> vec3f { let s = sin(a); let c = cos(a); let r = p.zx * mat2x2f(c, s, -s, c); return vec3f(r.y, p.y, r.x); }
fn rotM(p: vec3f, m: vec2f) -> vec3f { return rotX(rotY(p, -2. * 3.14159265 * m.x), 3.14159265 * (.5 - m.y)); }

fn project(v: vec3f) -> vec3f {
    //in coordinate system: X→, Y↑, Z⊙
    let res = vec2f(textureDimensions(screen));
    let p = rotM(v, vec2f(mouse.pos) / res);
    let ro = 60. * custom.r;
    let uv = vec2f(p.xy) / (ro - p.z);
    let rmax = max(res.x, res.y);
    let pixel = (uv * rmax + res) * .5;
    let size = point_size * rmax / (ro - p.z);
    return vec3f(pixel, size);
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
  var r: vec2f;
  if (h >= 0.) {
    h = sqrt(h);
    let x = (vec2f(h, -h) - q) / 2.;
    let uv = sign(x) * pow(abs(x), vec2f(1. / 3.));
    let t = clamp(uv.x + uv.y - kx, 0., 1.);
    let f = d + (c + b * t) * t;
    r = vec2f(dot(f, f), t);
  } else {
    let z = sqrt(-p1);
    let v = acos(q / (p1 * z * 2.)) / 3.;
    let m = cos(v);
    let n = sin(v) * 1.732050808;
    let t = clamp(vec2f(m + m, -n - m) * z - kx, vec2f(0.0), vec2f(1.0));
    let f = d + (c + b * t.x) * t.x;
    var dis: f32 = dot(f, f);
    r = vec2f(dis, t.x);
    let g = d + (c + b * t.y) * t.y;
    dis = dot(g, g);
    r = select(r, vec2f(dis, t.y), dis < r.x);
  }
  r.x = sqrt(r.x);
  return r;
}

@compute @workgroup_size(1)
#workgroup_count clear TILES_X TILES_Y 1
fn clear(@builtin(global_invocation_id) wid: vec3u) {
    atomicStore(&(tiles[wid.x][wid.y].amount), 0u);
}

@compute @workgroup_size(256)
#workgroup_count setup COUNT_256 1 1
fn setup(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wid: vec3u
) {
    if (gid.x < COUNT-2) {
        let res = vec2f(textureDimensions(screen));
        let id = gid.x;
        let p0 = points[id];
        let p1 = points[id+1u];
        let p2 = points[id+2u];
        let a = project(mix(p0, p1, 0.5));
        let b = project(p1);
        let c = project(mix(p1, p2, 0.5));
        let size = point_size * mix(a.z, c.z, .5) / 16.;
        for (var i = 0.; i < TILES_X; i += 1.) {
           for (var j = 0.; j < TILES_Y; j += 1.) {
                if (sdBezier(vec2f(i,j)+.5, a.xy/16., b.xy/16., c.xy/16.).x < size) {
                    let id = atomicAdd(&(tiles[u32(i)][u32(j)].amount), 1u);
                    if (id < TILE_CAPACITY) {
                        tiles[u32(i)][u32(j)].ids[id] = gid.x;
                    }
                }
            }
        }
    }
}

@compute @workgroup_size(16, 16)
fn rasterizer(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(local_invocation_index) ldx: u32,
    @builtin(workgroup_id) wid: vec3u
) {
    let th_size = u32(TILE_CAPACITY / 256);
    for (var i = 0u; i < th_size; i++) {
        let th_id = ldx * th_size + i;
        ids[th_id] = tiles[wid.x][wid.y].ids[th_id];
    }
    workgroupBarrier();

    let amount = atomicLoad(&tiles[wid.x][wid.y].amount);
    var col = vec3f(0.);
    //col += vec3f(f32(amount)/TILE_CAPACITY);
    for (var i = 0u; i < amount; i++) {
        let id = ids[i];
        let p0 = points[id];
        let p1 = points[id+1u];
        let p2 = points[id+2u];
        let a = project(mix(p0, p1, 0.5));
        let b = project(p1);
        let c = project(mix(p1, p2, 0.5));
        var aa_col = vec3f(0.);
        var aa_rt = vec2f(0);
        for (var i = 0.; i < AA; i += 1.) { for (var j = 0.; j < AA; j += 1.) {
            let dxy = (vec2f(i, j) + .5) / AA;
            let pix = vec2f(gid.xy) + dxy;
            aa_rt += sdBezier(pix, a.xy, b.xy, c.xy);
        }}
        aa_rt /= (AA * AA);
        let z = mix(a.z, c.z, aa_rt.y);
        let r = max(0., aa_rt.x - 2.*custom.thickness);
        let size = point_size * z;
        let sid = f32(id) + aa_rt.y;
        let t0 = 0.;
        let T = (sid + t0) / COUNT;
        let base_col = (.5 + .5 * sin(vec3f(500. * T) + vec3f(1,2,3))) * (.5 + .5 * sin(50. * T));
        let bloom = 1. - exp(-max(0., pow(.015*size/(r+.003), 1.5) - .003));
        let falloff = smoothstep(0., 30., sid) * (1. - smoothstep(COUNT-30., COUNT, sid));
        col = max(col, step(r, size) * base_col * bloom * falloff);
    }
    let res = textureDimensions(screen);
    if (gid.x < res.x && gid.y < res.y) {
        textureStore(screen, vec2u(gid.x, res.y-1-gid.y), vec4f(col, 1.));
    }
}