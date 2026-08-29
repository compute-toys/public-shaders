// Holographic Radiance Cascades (flatland) — compute.toys dialect.
//
// Implements Freeman/Sannikov/Margel, "Holographic Radiance Cascades"
// (arXiv:2505.02041; reference impls: github.com/entropylost/amitabha,
// Yaazarai/Volumetric-HRC). There is NO raymarching anywhere: four
// frustums (E, N, W, S in image space) each seed T_0 by sampling the
// scene at probe positions, then EXTEND ("merge up") builds level-n rays
// by compositing two level n-1 rays (even rays: exact composition, odd
// rays: averaged cross-composition, paper Eq. 18-20), then MERGE ("merge
// down") resolves cascades top-down with the parity-dependent connection
// (even probes Richardson-average with the coarser level) and cone-arc
// weights (paper Eq. 13, computed analytically here), with sky injected
// at the top cascade. Level 0 accumulates per-direction fluence; a
// resolve pass applies the opacity-gated cross blur (Eq. 21) that
// cancels the even/odd checkerboard.
//
// Structure: the whole per-frame loop (4 frustums x (seed + extends +
// merges)) is ONE entry point run 4*NC times via #dispatch_count,
// decoding (pair, phase, level, side) from dispatch.id + thread index —
// dispatch barriers provide the seed -> extend -> merge ordering each
// level needs. Opposite frustums (E+W, then N+S) run CONCURRENTLY in
// each dispatch: they share probe/slice geometry and write disjoint
// buffer slots, and this matters because the compute.toys engine copies
// the full pass texture after EVERY dispatch, so per-frame cost is
// dominated by dispatch count, not shader work. Per-pass parameters are
// recomputed in-shader; the scene is the analytic SDF shared with
// rc-vanilla (occupancy = distance < 0.5, like the marcher's hit test);
// all intermediate buffers are packed into the two #storage arrays with
// manual offsets: `rays` = 2 slots x per-level ray buffers (f16
// rgb+transmittance), `aux` = 2x2 merge ping-pong (rgb9e5) + 4-direction
// fluence + resolved probe fluence. NC_MAX 10 with probe spacing 2
// supports screens up to 2048 px wide; the default 1280x720 uses all 10
// levels (~45 MB rays, ~9 MB aux).
//
// Scene, output convention (linear, mean radiance = fluence/2pi) and the
// direct-emission overlay are identical to rc-vanilla / rc-bilinear-fix —
// compare with identical --frames/--size renders.
//
// Findings (1280x720, 8 frames, vs rc-vanilla / rc-bilinear-fix): no
// ringing at all and no starburst residual near emitters — penumbrae stay
// clean arbitrarily close to lights, and shadows are sharper than both
// raymarched variants. Faint diagonal frustum seams at +-45 degrees from
// small emitters are the visible HRC artifact. Cost in this harness is
// all dispatch-count overhead: 43 dispatches/frame vs vanilla's 7 (~5x
// vanilla wall-clock; ~110 ms/frame at 720p here) even though HRC's
// per-thread work is far lower — on compute.toys itself the same
// per-dispatch pass-copy applies. The frustum pairing above already
// halved this from a naive sequential schedule's 83 dispatches
// (byte-identical output, ~200 ms/frame).
//
// Paste-ready for compute.toys.

#define NC_MAX 10u
#define PROBE_SPACING 2u

#storage rays array<vec2u>
#storage aux array<u32>

const SKY = vec3f(0.02, 0.03, 0.05);
const TAU = 6.28318530718;

// --- scene (identical to rc-vanilla — keep verbatim for A/B) --------------

fn sd_circle(p: vec2f, c: vec2f, r: f32) -> f32 { return length(p - c) - r; }

fn sd_box(p: vec2f, c: vec2f, b: vec2f, rot: f32) -> f32 {
    var q = p - c;
    let s = sin(rot);
    let co = cos(rot);
    q = mat2x2f(co, -s, s, co) * q;
    let d = abs(q) - b;
    return length(max(d, vec2f(0.0))) + min(max(d.x, d.y), 0.0);
}

struct Hit { d: f32, emis: vec3f }

fn scene(p: vec2f, res: vec2f) -> Hit {
    var h: Hit;
    h.d = 1e9;
    h.emis = vec3f(0.0);

    // Draggable light (orbits when the mouse is up).
    var lp = res * 0.5
        + vec2f(cos(time.elapsed * 0.7), sin(time.elapsed * 0.7)) * res.y * 0.25;
    if (mouse.click > 0) { lp = vec2f(mouse.pos); }
    let dl = sd_circle(p, lp, 12.0);
    if (dl < h.d) { h.d = dl; h.emis = vec3f(3.0, 2.8, 2.5); }

    let d1 = sd_circle(p, res * vec2f(0.15, 0.25), 10.0);
    if (d1 < h.d) { h.d = d1; h.emis = vec3f(4.0, 1.2, 0.3); }
    let d2 = sd_circle(p, res * vec2f(0.85, 0.7), 9.0);
    if (d2 < h.d) { h.d = d2; h.emis = vec3f(0.4, 1.6, 4.0); }

    var o = sd_box(p, res * vec2f(0.35, 0.6), vec2f(90.0, 14.0), 0.4);
    o = min(o, sd_box(p, res * vec2f(0.62, 0.35), vec2f(14.0, 110.0), 0.0));
    o = min(o, sd_box(p, res * vec2f(0.78, 0.8), vec2f(60.0, 60.0), 0.7));
    if (o < h.d) { h.d = o; h.emis = vec3f(0.0); }

    return h;
}

// Binary occupancy sample: solid where the marcher would report a hit.
fn seed_sample(wp: vec2f, res: vec2f) -> vec4f {
    let h = scene(wp, res);
    return select(vec4f(0.0), vec4f(h.emis, 1.0), h.d < 0.5);
}

// --- layout helpers -------------------------------------------------------

fn cdiv(a: u32, b: u32) -> u32 { return (a + b - 1u) / b; }
fn clog2(v: u32) -> u32 { return 32u - countLeadingZeros(max(v, 2u) - 1u); }
fn np2(v: u32) -> u32 { return 1u << clog2(v); }

// Ray-buffer row width (in entries) for probe count pc at a cascade level.
fn ray_w(pc: u32, level: u32) -> u32 {
    if (level == 0u) { return pc; }
    return cdiv(pc, 1u << level) * ((1u << level) + 1u);
}

// A level's region must fit either frustum orientation (E/W: probe axis x,
// N/S: probe axis y) since the two slot regions are reused across pairs.
fn ray_region(level: u32, probes: vec2u) -> u32 {
    return max(ray_w(probes.x, level) * probes.y,
               ray_w(probes.y, level) * probes.x);
}

// Offset of (slot, level)'s region in `rays`; slots sized for NC_MAX levels.
fn ray_off(slot: u32, level: u32, probes: vec2u) -> u32 {
    var o = 0u;
    for (var l = 0u; l < NC_MAX; l++) {
        if (l < level) { o += ray_region(l, probes); }
    }
    var stride = 0u;
    for (var l = 0u; l < NC_MAX; l++) { stride += ray_region(l, probes); }
    return slot * stride + o;
}

// aux: [4x merge (slot,pingpong)][fluence 4*pw*ph][resolved pw*ph]
fn merge_region(probes: vec2u) -> u32 {
    return max(np2(probes.x) * probes.y, np2(probes.y) * probes.x);
}
fn merge_off(slot: u32, buf: u32, probes: vec2u) -> u32 {
    return (slot * 2u + buf) * merge_region(probes);
}
fn fluence_off(probes: vec2u) -> u32 { return 4u * merge_region(probes); }
fn resolved_off(probes: vec2u) -> u32 {
    return fluence_off(probes) + probes.x * probes.y * 4u;
}

fn pack_ray_data(v: vec4f) -> vec2u {
    return vec2u(pack2x16float(v.xy), pack2x16float(v.zw));
}
fn unpack_ray_data(p: vec2u) -> vec4f {
    return vec4f(unpack2x16float(p.x), unpack2x16float(p.y));
}

fn pack_e5(c: vec3f) -> u32 {
    let max_c = max(c.r, max(c.g, c.b));
    var exp_shared = 0;
    var scale = 0.0;
    if (max_c >= 6.10352e-5) {
        exp_shared = clamp(i32(ceil(log2(max_c))) + 15, 0, 31);
        scale = exp2(f32(-exp_shared + 15 + 9));
    }
    let r = u32(clamp(c.r * scale, 0.0, 511.0));
    let g = u32(clamp(c.g * scale, 0.0, 511.0));
    let b = u32(clamp(c.b * scale, 0.0, 511.0));
    return r | (g << 9u) | (b << 18u) | (u32(exp_shared) << 27u);
}

fn unpack_e5(p: u32) -> vec3f {
    let r = f32(p & 0x1FFu);
    let g = f32((p >> 9u) & 0x1FFu);
    let b = f32((p >> 18u) & 0x1FFu);
    let e = f32((p >> 27u) & 0x1Fu) - 15.0 - 9.0;
    return vec3f(r, g, b) * exp2(e);
}

struct Ray { rad: vec3f, trans: f32 }

// Paper Eq. 7: Merge(<r_n,t_n>, <r_f,t_f>) = <r_n + t_n*r_f, t_n*t_f>
fn composite_ray(near: Ray, far: Ray) -> Ray {
    return Ray(near.rad + far.rad * near.trans, near.trans * far.trans);
}

// Paper Eq. 13 cone arc for `level`, sub-bin s in [0, 2^(level+1)).
fn cone_arc(level: u32, s: i32) -> f32 {
    let n = i32(2u << level);
    return atan2(f32(2 * s - n + 2), f32(n)) - atan2(f32(2 * s - n), f32(n));
}

// --- extend/merge loads ----------------------------------------------------

fn load_prev(probe_idx: i32, ray_idx: i32, slice_idx: i32, slot: u32,
             level: u32, pc: u32, sc: u32, probes: vec2u) -> Ray {
    let prev_level = level - 1u;
    let prev_num_probes = i32(cdiv(pc, 1u << prev_level));
    let prev_num_rays = i32(1u << prev_level) + 1;
    if (probe_idx < 0 || probe_idx >= prev_num_probes ||
        ray_idx < 0 || ray_idx >= prev_num_rays ||
        slice_idx < 0 || slice_idx >= i32(sc)) {
        return Ray(vec3f(0.0), 1.0);
    }
    let prev_w = i32(ray_w(pc, prev_level));
    var tex: i32;
    if (prev_level == 0u) {
        tex = probe_idx;
    } else {
        tex = (probe_idx << prev_level) + probe_idx + ray_idx;
    }
    let r = unpack_ray_data(
        rays[ray_off(slot, prev_level, probes) + u32(slice_idx * prev_w + tex)]);
    return Ray(r.rgb, r.a);
}

fn load_ray(probe_idx: i32, ray_idx: i32, slice_idx: i32, slot: u32,
            level: u32, pc: u32, sc: u32, probes: vec2u) -> Ray {
    let level_probes = i32(cdiv(pc, 1u << level));
    let num_rays = i32(1u << level) + 1;
    if (probe_idx < 0 || probe_idx >= level_probes ||
        ray_idx < 0 || ray_idx >= num_rays ||
        slice_idx < 0 || slice_idx >= i32(sc)) {
        return Ray(vec3f(0.0), 1.0);
    }
    var tex: i32;
    var row_w: i32;
    if (level == 0u) {
        tex = probe_idx;
        row_w = level_probes;
    } else {
        tex = (probe_idx << level) + probe_idx + ray_idx;
        row_w = level_probes * num_rays;
    }
    let r = unpack_ray_data(
        rays[ray_off(slot, level, probes) + u32(slice_idx * row_w + tex)]);
    return Ray(r.rgb, r.a);
}

// --- passes ---------------------------------------------------------------

@compute @workgroup_size(16, 16)
fn clear_fluence(@builtin(global_invocation_id) id: vec3u) {
    let res = vec2u(textureDimensions(screen));
    if (id.x >= res.x || id.y >= res.y) { return; }
    let idx = id.y * res.x + id.x;
    let probes = res / PROBE_SPACING;
    if (idx < probes.x * probes.y * 4u) {
        aux[fluence_off(probes) + idx] = 0u;
    }
}

// 2 pairs (E+W, then N+S) x (1 seed + (nc-1) extends + nc merges) = 4*nc.
#dispatch_count hrc_pass 40

@compute @workgroup_size(16, 16)
fn hrc_pass(@builtin(global_invocation_id) id: vec3u) {
    let res = vec2u(textureDimensions(screen));
    if (id.x >= res.x || id.y >= res.y) { return; }
    let idx = id.y * res.x + id.x; // flat, unique in [0, res.x*res.y)

    let probes = res / PROBE_SPACING;
    let pmax = max(probes.x, probes.y);
    let nc = min(clog2(pmax), NC_MAX);

    let per_pair = 2u * nc;
    let pair = dispatch.id / per_pair;
    let k = dispatch.id - pair * per_pair;
    if (pair >= 2u) { return; } // idle dispatches when nc < NC_MAX

    // Pair 0 = E+W frustums (probe axis x), pair 1 = N+S (probe axis y).
    let pc = select(probes.y, probes.x, pair == 0u);
    let sc = select(probes.x, probes.y, pair == 0u);

    // Threads for ONE frustum of this phase; the two opposite frustums of
    // the pair run side by side in this dispatch on disjoint buffer slots.
    var n1: u32;
    if (k == 0u) {
        n1 = pc * sc;
    } else if (k < nc) {
        n1 = ray_w(pc, k) * sc;
    } else {
        let level = nc - 1u - (k - nc);
        n1 = (cdiv(pc, 1u << level) << level) * sc;
    }
    let side = idx / n1;
    if (side >= 2u) { return; }
    let sub = idx - side * n1;
    let dir = pair + 2u * side; // 0=E, 1=N, 2=W, 3=S
    let slot = side;
    let spacing = f32(PROBE_SPACING);

    if (k == 0u) {
        // -- Phase A: seed — T_0 is a direct occupancy sample at the probe.
        let probe = sub % pc;
        let slice = sub / pc;
        let pf = (f32(probe) + 0.5) * spacing;
        let sf = (f32(slice) + 0.5) * spacing;
        var wp: vec2f;
        switch (dir) {
            case 0u: { wp = vec2f(pf, sf); }                    // E: +x
            case 1u: { wp = vec2f(sf, pf); }                    // N: +y
            case 2u: { wp = vec2f(f32(res.x) - pf, sf); }       // W: -x
            default: { wp = vec2f(sf, f32(res.y) - pf); }       // S: -y
        }
        let s = seed_sample(wp, vec2f(res));
        var rd = vec4f(0.0, 0.0, 0.0, 1.0);
        if (s.a > 0.001) { rd = vec4f(s.rgb, 0.0); }
        rays[ray_off(slot, 0u, probes) + slice * pc + probe] = pack_ray_data(rd);

    } else if (k < nc) {
        // -- Phase B: extend ("merge up") — level k rays from two level k-1
        // rays. Even rays: exact composition (Eq. 18); odd rays: averaged
        // cross-composition (Eq. 19-20). Branchless: for even rays
        // lower == upper so both cross terms are identical.
        let level = k;
        let interval = i32(1u << level);
        let num_rays = interval + 1;
        let curr_w = ray_w(pc, level);
        let texel_x = i32(sub % curr_w);
        let slice = i32(sub / curr_w);
        let level_probes = i32(cdiv(pc, 1u << level));
        let probe_idx = texel_x / num_rays;
        let ray_idx = texel_x - probe_idx * num_rays;
        if (probe_idx >= level_probes) { return; }

        let prev_interval = interval / 2;
        let lower = ray_idx / 2;
        let upper = (ray_idx + 1) / 2;

        let off_a = lower * 2 - prev_interval;
        let cross_a = composite_ray(
            load_prev(probe_idx * 2, lower, slice, slot, level, pc, sc, probes),
            load_prev(probe_idx * 2 + 1, upper, slice + off_a, slot, level, pc, sc, probes),
        );
        let off_b = upper * 2 - prev_interval;
        let cross_b = composite_ray(
            load_prev(probe_idx * 2, upper, slice, slot, level, pc, sc, probes),
            load_prev(probe_idx * 2 + 1, lower, slice + off_b, slot, level, pc, sc, probes),
        );

        let avg = vec4f((cross_a.rad + cross_b.rad) * 0.5,
                        (cross_a.trans + cross_b.trans) * 0.5);
        rays[ray_off(slot, level, probes) + u32(slice) * curr_w + u32(texel_x)] =
            pack_ray_data(avg);

    } else {
        // -- Phase C: merge ("merge down") — resolve R_{nc-1}..R_0 top-down.
        // Parity-dependent connection: even probes composite two adjacent
        // fine intervals and Richardson-average with the coarser level's
        // result; odd probes composite a single interval directly. Level 0
        // accumulates into this direction's fluence slot.
        let k_merge = k - nc;
        let level = nc - 1u - k_merge;
        let ndirs = 1u << level; // angular bins at this level
        let xw = cdiv(pc, ndirs) << level;
        let probe_ang = sub % xw;
        let slice = i32(sub / xw);
        let probe_idx = i32(probe_ang >> level);
        let ang_bin = i32(probe_ang & (ndirs - 1u));

        let f_nc = min(clog2(pc), nc);
        let is_top = level == f_nc - 1u;
        let ms = i32(np2(pc)); // merge row stride
        var miw = 0; // merge_in width (0 = nothing above to read)
        if (level + 1u < f_nc) {
            miw = i32(cdiv(pc, ndirs * 2u) * (ndirs * 2u));
        }
        let read_base = merge_off(slot, 1u - (k_merge & 1u), probes);
        let write_base = merge_off(slot, k_merge & 1u, probes);

        let is_even = (probe_idx % 2 == 0);
        let far_step = select(1, 2, is_even);

        var result = vec3f(0.0);
        for (var side_ray = 0; side_ray < 2; side_ray++) {
            let sub_bin = ang_bin * 2 + side_ray;
            let ray_idx = ang_bin + side_ray;
            let weight = cone_arc(level, sub_bin);

            let ray = load_ray(probe_idx, ray_idx, slice, slot, level, pc, sc, probes);
            let slice_off = ray_idx * 2 - i32(ndirs);

            let far_x = ((probe_idx + far_step) << level) + sub_bin;
            let far_slice = slice + slice_off * far_step;
            // Constant sky: the prefix-sum lookup reduces to sky * arc.
            var far_fluence: vec3f;
            if (is_top || far_x < 0 || far_x >= miw ||
                far_slice < 0 || far_slice >= i32(sc)) {
                far_fluence = SKY * weight;
            } else {
                far_fluence = unpack_e5(aux[read_base + u32(far_slice * ms + far_x)]);
            }

            if (is_even) {
                let ext = load_ray(probe_idx + 1, ray_idx, slice + slice_off,
                                   slot, level, pc, sc, probes);
                let c_rad = ray.rad + ext.rad * ray.trans;
                let c_trans = ray.trans * ext.trans;
                let merged = c_rad * weight + far_fluence * c_trans;
                let coarse_x = (probe_idx << level) + sub_bin;
                var coarse = vec3f(0.0);
                if (!is_top && coarse_x >= 0 && coarse_x < miw) {
                    coarse = unpack_e5(aux[read_base + u32(slice * ms + coarse_x)]);
                }
                result += (merged + coarse) * 0.5;
            } else {
                result += ray.rad * weight + far_fluence * ray.trans;
            }
        }

        if (ndirs > 1u) {
            let out_x = (probe_idx << level) + ang_bin;
            aux[write_base + u32(slice * ms + out_x)] = pack_e5(result);
        } else {
            // Level 0: accumulate into this direction's fluence slot.
            // Threads map 1:1 to (probe, slice) and the two concurrent
            // frustums write disjoint direction slots, so plain
            // read-modify-write is race-free.
            let fw = i32(probes.x);
            let fh = i32(probes.y);
            var fc: vec2i;
            switch (dir) {
                case 0u: { fc = vec2i(probe_idx - 1, slice); }
                case 1u: { fc = vec2i(slice, probe_idx - 1); }
                case 2u: { fc = vec2i(fw - probe_idx, slice); }
                default: { fc = vec2i(slice, fh - probe_idx); }
            }
            if (fc.x >= 0 && fc.x < fw && fc.y >= 0 && fc.y < fh) {
                let fi = fluence_off(probes) + dir * probes.x * probes.y
                       + u32(fc.y) * probes.x + u32(fc.x);
                aux[fi] = pack_e5(unpack_e5(aux[fi]) + result);
            }
        }
    }
}

// Resolve: opacity-gated cross blur (paper Eq. 21, cancels the even/odd
// checkerboard) summed over the 4 directions; total fluence stored in
// radiance-average scale (/ 2pi), matching rc-vanilla's output convention.

fn probe_solid(px: i32, py: i32, res: vec2f) -> bool {
    let wp = (vec2f(f32(px), f32(py)) + 0.5) * f32(PROBE_SPACING);
    return scene(wp, res).d < 0.5;
}

@compute @workgroup_size(16, 16)
fn hrc_resolve(@builtin(global_invocation_id) id: vec3u) {
    let res = vec2u(textureDimensions(screen));
    let probes = res / PROBE_SPACING;
    let pw = i32(probes.x);
    let ph = i32(probes.y);
    let x = i32(id.x);
    let y = i32(id.y);
    if (x >= pw || y >= ph) { return; }

    let f_base = fluence_off(probes);
    let center_solid = probe_solid(x, y, vec2f(res));
    var offsets = array<vec2i, 4>(
        vec2i(-1, 0), vec2i(1, 0), vec2i(0, -1), vec2i(0, 1),
    );

    var total = vec3f(0.0);
    for (var d = 0u; d < 4u; d++) {
        let d_base = f_base + d * probes.x * probes.y;
        var sum = unpack_e5(aux[d_base + u32(y * pw + x)]) * 4.0;
        var wt = 4.0;
        for (var i = 0; i < 4; i++) {
            let nx = clamp(x + offsets[i].x, 0, pw - 1);
            let ny = clamp(y + offsets[i].y, 0, ph - 1);
            if (probe_solid(nx, ny, vec2f(res)) == center_solid) {
                sum += unpack_e5(aux[d_base + u32(ny * pw + nx)]);
                wt += 1.0;
            }
        }
        total += sum / wt;
    }

    aux[resolved_off(probes) + id.y * probes.x + id.x] = pack_e5(total / TAU);
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let res = vec2u(textureDimensions(screen));
    if (id.x >= res.x || id.y >= res.y) { return; }
    let pos = vec2f(id.xy) + 0.5;

    let probes = res / PROBE_SPACING;
    let base = resolved_off(probes);

    // Bilinear upscale of the resolved probe-fluence grid.
    let pcf = vec2f(probes);
    let local = clamp(pos / vec2f(res) * pcf - 0.5, vec2f(0.0), pcf - 1.0);
    let i0 = vec2u(floor(local));
    let i1 = min(i0 + 1u, probes - 1u);
    let f = fract(local);
    let s00 = unpack_e5(aux[base + i0.y * probes.x + i0.x]);
    let s10 = unpack_e5(aux[base + i0.y * probes.x + i1.x]);
    let s01 = unpack_e5(aux[base + i1.y * probes.x + i0.x]);
    let s11 = unpack_e5(aux[base + i1.y * probes.x + i1.x]);
    var fluence = mix(mix(s00, s10, f.x), mix(s01, s11, f.x), f.y);

    // Direct emission overlay for crisp shapes.
    let h = scene(pos, vec2f(res));
    if (h.d < 0.0) { fluence = h.emis; }

    textureStore(screen, vec2i(id.xy), vec4f(fluence, 1.0));
}
