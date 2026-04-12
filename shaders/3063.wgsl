#storage height_map array<vec3f>


fn hash( p: vec2f ) -> vec2f // replace this by something better
{
	let a = vec2( dot(p,vec2(127.1,311.7)), dot(p,vec2(269.5,183.3)) );
	return (-1.0 + 2.0*fract(sin(a)*43758.5453123)) * 0.5 + 0.5;
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
  var p3 = fract(vec3<f32>(p.xyx) * vec3<f32>(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.xx + p3.yz) * p3.zy);
}

// Returns vec3(value, derivative_x, derivative_y)
fn noised(p: vec2f) -> vec3f {
    let scaled_p = p * custom.hill_frequency * 0.01;
    let i = floor(scaled_p);
    let f = fract(scaled_p);

    // Quintic interpolation (C2 continuous) — smoother than cubic for derivatives
    let u  = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let du = f * f * (f * (f * 30.0 - 60.0) + 30.0);  // analytical derivative of u

    let ga = hash22(i + vec2f(0.0, 0.0));
    let gb = hash22(i + vec2f(1.0, 0.0));
    let gc = hash22(i + vec2f(0.0, 1.0));
    let gd = hash22(i + vec2f(1.0, 1.0));

    let va = dot(ga, f - vec2f(0.0, 0.0));
    let vb = dot(gb, f - vec2f(1.0, 0.0));
    let vc = dot(gc, f - vec2f(0.0, 1.0));
    let vd = dot(gd, f - vec2f(1.0, 1.0));

    let value = va + u.x*(vb-va) + u.y*(vc-va) + u.x*u.y*(va-vb-vc+vd);

    // Chain rule through the bilinear blend
    let ddx = du.x * (vb-va + u.y*(va-vb-vc+vd)) + ga.x + u.x*(gb.x-ga.x) + u.y*(gc.x-ga.x) + u.x*u.y*(ga.x-gb.x-gc.x+gd.x);
    let ddy = du.y * (vc-va + u.x*(va-vb-vc+vd)) + ga.y + u.x*(gb.y-ga.y) + u.y*(gc.y-ga.y) + u.x*u.y*(ga.y-gb.y-gc.y+gd.y);

    return vec3f(value * 0.5 + 0.5, ddx, ddy);
}

const OCTAVES = 4;
const LACUNARITY = 4.0;
const GAIN = 0.5;

fn fractal_noise(p: vec2f) -> vec3f {
    var n  = vec3f(0.0);
    var nf = 1.0;
    var na = 1.0;

    for (var i = 0; i < OCTAVES; i++) {
        let s = noised(p * nf);
        // vec3(1, nf, nf) applies chain rule to derivatives: d/dp[f(nf*p)] = nf * f'(nf*p)
        n  += s * na * vec3f(1.0, nf, nf);
        na *= GAIN;
        nf *= LACUNARITY;
    }
    return n;
}

fn get_height(id: vec2i) -> vec3f {
    let idx = id.y * vec2i(textureDimensions(screen)).x + id.x;

    return height_map[idx];
}

fn get_angle_dir(theta: f32) -> vec2f {
    return vec2f(cos(theta), sin(theta));
}

fn get_sine_wave(id: vec2f, dir: vec2f, freq: f32) -> f32 {
    let phase = dot(id, dir) * freq;
    return cos(phase);
}

fn dir_to_rad(dir: vec2f) -> f32 {
    return atan2(dir.y, dir.x);
}


fn rotate(p: vec2f, angle: f32) -> vec2f {
    return vec2f(
        p.x * cos(angle) - p.y * sin(angle),
        p.x * sin(angle) + p.y * cos(angle) 
    );
}

// rotate `p` around `pivot` by `angle`
fn rotate_around(pivot: vec2f, p: vec2f, angle: f32) -> vec2f {
    let rel = p - pivot;            // move pivot to origin
    let rot = rotate(rel, angle);   // rotate
    return rot;             // move back
}

const right_angle = 3.1423  / 2.0;


// Returns vec3(height_offset, deriv_x, deriv_y)
fn voronoi_gullies(id: vec2i, scale: f32) -> vec3f {
    let s = custom.voronoi_scale;
    let p = vec2i(floor(vec2f(id) / s));
    let f = fract(vec2f(id) / s);

    // Get slope from stored height+deriv — .yz is the gradient
    let stored   = get_height(id);   // vec3(h, dx, dy)
    let grad     = stored.yz;

    // sideDir is perpendicular to slope, pre-multiplied by 2pi (matches Rune's formulation)
    // This is the direction the sine waves run along
    let sideDir  = vec2f(-grad.y, grad.x) * 2.0 * 3.14159265 * scale;

    var weight_sum  = 0.0;
    var sum_height  = 0.0;
    var sum_deriv   = vec2f(0.0);

    for (var dy: i32 = -1; dy <= 1; dy++) {
        for (var dx: i32 = -1; dx <= 1; dx++) {
            let b = vec2f(f32(dx), f32(dy));
            let r = b - f + hash(vec2f(p) + b);   // vector from cell point to p
            let d = dot(r, r);

            let w = max(0.0, exp(-d * 2.0) - 0.01111);   // bell weight, zero at dist 1.5
            weight_sum += w;

            // waveInput: how far along sideDir we are from this cell point
            // sideDir already encodes frequency (length = slope magnitude * 2pi * scale)
            let waveInput = dot(r, sideDir);

            // Analytical height and derivative of cos(waveInput)
            let stripe      =  cos(waveInput);
            let stripe_grad = -sin(waveInput) * sideDir;  // d/dr[cos(dot(r, sideDir))] = -sin * sideDir

            sum_height += stripe * w;
            sum_deriv  += stripe_grad * w;
        }
    }

    if (weight_sum < 0.0001) { return vec3f(0.0); }

    let h = sum_height / weight_sum;
    let d = sum_deriv  / weight_sum;

    return vec3f(h, d);
}


@compute @workgroup_size(16, 16)
fn get_height_map(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let height = noised(vec2f(id.xy));

    let idx = id.y * textureDimensions(screen).x + id.x;
    height_map[idx] = height;
}

@compute @workgroup_size(16, 16)
fn gullies_1(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let idx = id.y * screen_size.x + id.x;

    let height =  height_map[idx];

    let gullies = voronoi_gullies(vec2i(id.xy), 1.0 * custom.ridge_frequency);
   
    height_map[idx] = height - (gullies * 2.0 - 1.0) * custom.gully_depth * 0.01;
}

@compute @workgroup_size(16, 16)
fn gullies_2(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let idx = id.y * screen_size.x + id.x;

    let height =  height_map[idx];

    let gullies = voronoi_gullies(vec2i(id.xy), 2.0 * custom.ridge_frequency);
   
    height_map[idx] = height - (gullies * 2.0 - 1.0) * custom.gully_depth * 0.5* 0.01;
}

@compute @workgroup_size(16, 16)
fn gullies_4(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let idx = id.y * screen_size.x + id.x;

    let height =  height_map[idx];

    let gullies = voronoi_gullies(vec2i(id.xy), 4.0 * custom.ridge_frequency);
   
    height_map[idx] = height - (gullies * 2.0 - 1.0) * custom.gully_depth * 0.25* 0.01;
}


const EPS = 0.0001;
const FAR = 80.0;
const PI = 3.1415926;

fn rotX(p: vec3f, a: f32) -> vec3f { let c = cos(a); let s = sin(a); return vec3f(p.x, p.y*c - p.z*s, p.y*s + p.z*c); }
fn rotY(p: vec3f, a: f32) -> vec3f { let c = cos(a); let s = sin(a); return vec3f(p.x*c + p.z*s, p.y, -p.x*s + p.z*c); }
fn rotM(p: vec3f, m: vec2f) -> vec3f { return rotY(rotX(p, -PI * m.y), 2.0 * PI * m.x); }

fn getHeight(coord: vec2i, res: vec2u) -> f32 {
    let c = clamp(coord, vec2i(0), vec2i(res) - vec2i(1));
    let idx = u32(c.y) * res.x + u32(c.x);
    return height_map[idx].x;
}

// Sample height at a continuous UV position [0..1]^2
fn sampleHeight(uv: vec2f, res: vec2u) -> f32 {
    let tc = uv * vec2f(res);
    let ci = vec2i(i32(tc.x), i32(tc.y));
    let f  = fract(tc);
    let h00 = getHeight(ci + vec2i(0,0), res);
    let h10 = getHeight(ci + vec2i(1,0), res);
    let h01 = getHeight(ci + vec2i(0,1), res);
    let h11 = getHeight(ci + vec2i(1,1), res);
    return mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
}

// Ray-march against the heightmap
// World: heightmap occupies [0,1]^2 in XZ, Y = height * heightScale
fn marchTerrain(ro: vec3f, rd: vec3f, heightScale: f32, hmRes: vec2u) -> f32 {
    var t = 0.0;
    var prevY = ro.y;
    var prevT = 0.0;
    for (var i = 0; i < 1024; i++) {
        let p   = ro + rd * t;
        let uv  = p.xz;                          // world XZ → UV
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
            t += 0.005;
            if (t > FAR) { break; }
            continue;
        }
        let terrY = sampleHeight(uv, hmRes) * heightScale;
        if (p.y < terrY) {
            // Binary-search refinement between prevT and t
            var lo = prevT;
            var hi = t;
            for (var j = 0; j < 8; j++) {
                let mid  = (lo + hi) * 0.5;
                let mp   = ro + rd * mid;
                let muv  = mp.xz;
                let mty  = sampleHeight(muv, hmRes) * heightScale;
                if (mp.y < mty) { hi = mid; } else { lo = mid; }
            }
            return (lo + hi) * 0.5;
        }
        prevT = t;
        prevY = p.y - terrY;
        // Adaptive step: step faster when high above terrain
        t += max(0.001, (p.y - terrY) * 0.3);
        if (t > FAR) { break; }
    }
    return -1.0;
}

fn terrainNormal(p: vec3f, heightScale: f32, hmRes: vec2u) -> vec3f {
    let eps = 1.0 / f32(hmRes.x);
    let hR = sampleHeight(p.xz + vec2f( eps, 0.0), hmRes) * heightScale;
    let hL = sampleHeight(p.xz + vec2f(-eps, 0.0), hmRes) * heightScale;
    let hU = sampleHeight(p.xz + vec2f(0.0,  eps), hmRes) * heightScale;
    let hD = sampleHeight(p.xz + vec2f(0.0, -eps), hmRes) * heightScale;
    return normalize(vec3f(hL - hR, 2.0 * eps, hD - hU));
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let res = textureDimensions(screen);
    if (id.x >= res.x || id.y >= res.y) { return; }

    let hmRes      = vec2u(res.x, res.y);
    let heightScale = custom.terrain_height;

    let uv = (2.0 * (vec2f(id.xy) + 0.5) - vec2f(res)) / f32(res.y);

    var ro = vec3f(0.5, heightScale * 0.5, 1.5);
    var rd = normalize(vec3f(uv, -2.0 * mouse.zoom));
    let mouseNorm = vec2f(mouse.pos) / vec2f(res) - 0.5;
    ro = rotM(ro - vec3f(0.5, 0.0, 0.5), mouseNorm) + vec3f(0.5, 0.0, 0.5);
    rd = rotM(rd, mouseNorm);

    // Sky: deep zenith blue → warm horizon
    let skyZenith  = vec3f(0.10, 0.30, 0.65);
    let skyHorizon = vec3f(0.72, 0.58, 0.42);
    let skyT       = clamp(1.0 - rd.y * 2.5, 0.0, 1.0);
    let bg         = mix(skyZenith, skyHorizon, skyT * skyT);

    var col = bg;

    let lightDir = normalize(vec3f(-1.0, 3.0, -1.0));

    let t = marchTerrain(ro, rd, heightScale, hmRes);
    if (t > 0.0) {
        let p = ro + rd * t;
        let n = terrainNormal(p, heightScale, hmRes);
        let h = sampleHeight(p.xz, hmRes);

        // Slope: 0 = flat, 1 = vertical cliff
        let slope = 1.0 - clamp(n.y, 0.0, 1.0);

        // Small noise to break up colour bands — reuse fbm2D you already have
        let detailNoise = noised(p.xz * 18.0).x * 0.5 + 0.5;
        let h_noisy = clamp(h + (detailNoise - 0.5) * 0.04, 0.0, 1.0);

        // --- Colour zones by height ---
        let deepWater  = vec3f(0.04, 0.18, 0.40);
        let shallowWater = vec3f(0.10, 0.38, 0.55);
        let wetSand    = vec3f(0.62, 0.56, 0.38);
        let drySand    = vec3f(0.80, 0.73, 0.52);
        let lowGrass   = vec3f(0.30, 0.56, 0.18);
        let highGrass  = vec3f(0.20, 0.42, 0.12);
        let rock       = vec3f(0.42, 0.37, 0.30);
        let darkRock   = vec3f(0.28, 0.24, 0.20);
        let snow       = vec3f(0.93, 0.94, 0.97);
        let snowShadow = vec3f(0.72, 0.78, 0.88);

        var baseCol: vec3f;
        if (h_noisy < 0.08) {
            baseCol = mix(deepWater, shallowWater, h_noisy / 0.08);
        } else if (h_noisy < 0.14) {
            baseCol = mix(shallowWater, wetSand, (h_noisy - 0.08) / 0.06);
        } else if (h_noisy < 0.22) {
            baseCol = mix(wetSand, drySand, (h_noisy - 0.14) / 0.08);
        } else if (h_noisy < 0.42) {
            baseCol = mix(lowGrass, highGrass, (h_noisy - 0.22) / 0.20);
        } else if (h_noisy < 0.68) {
            baseCol = mix(highGrass, rock, (h_noisy - 0.42) / 0.26);
        } else if (h_noisy < 0.82) {
            baseCol = mix(rock, darkRock, (h_noisy - 0.68) / 0.14);
        } else {
            let snowT = clamp((h_noisy - 0.82) / 0.18, 0.0, 1.0);
            baseCol = mix(darkRock, mix(snowShadow, snow, snowT), snowT);
        }

        // Steep slopes go to bare rock regardless of height
        let rockCol = mix(rock, darkRock, clamp(slope * 2.0 - 0.5, 0.0, 1.0));
        baseCol = mix(baseCol, rockCol, smoothstep(0.35, 0.70, slope));

        // --- Lighting ---
        let diffuse  = max(dot(n, lightDir), 0.0);
        let ambient  = 0.18;
        // Soft warm bounce light from below (fake GI)
        let bounce   = max(dot(n, -lightDir * vec3f(1.0, 0.0, 1.0)), 0.0) * 0.07;
        let sunColor = vec3f(1.0, 0.93, 0.78);

        var shaded = baseCol * (ambient + bounce)
                   + baseCol * sunColor * diffuse * 0.85;

        // Water specular
        if (h < 0.10) {
            let refl    = reflect(rd, n);
            let spec    = pow(max(dot(refl, lightDir), 0.0), 64.0);
            let waterMask = clamp((0.10 - h) / 0.10, 0.0, 1.0);
            shaded += vec3f(1.0, 0.96, 0.80) * spec * 0.6 * waterMask;
        }

        // Snow rim light
        if (h > 0.78) {
            let rimLight = pow(clamp(1.0 - dot(n, -rd), 0.0, 1.0), 3.0);
            shaded += snowShadow * rimLight * 0.15;
        }

        // Fog: blend to horizon colour rather than flat sky
        let horizonCol = mix(skyHorizon, vec3f(0.85, 0.75, 0.55), 0.4);
        let fog = clamp((t - FAR * 0.3) / (FAR * 0.7), 0.0, 1.0);
        col = mix(shaded, horizonCol, fog * fog);

    } else {
        // Sun disc
        let sunDot = dot(rd, lightDir);
        let sun    = smoothstep(0.9985, 0.9995, sunDot);
        let glow   = smoothstep(0.90, 0.9985, sunDot) * 0.25;
        col = bg + vec3f(1.0, 0.90, 0.60) * (sun + glow);
    }

    textureStore(screen, vec2u(id.x, res.y - 1u - id.y), vec4f(col, 1.0));
}