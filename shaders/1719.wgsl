///////////////////////////////////////////////////////////////////////////////
// 1) The Link SDF (from Inigo Quilez) with ridges
///////////////////////////////////////////////////////////////////////////////
fn sdLink(p: vec3f, halfLen: f32, bigR: f32, smallR: f32, ridgeAmp: f32, ridgeFreq: f32) -> f32 {
    // Orient the link so that its long axis is along Y.
    let q = vec3f(
        p.x,
        max(abs(p.y) - halfLen, 0.0),
        p.z
    );
    // Compute the 2D vector for the torus profile:
    let k = vec2f(length(q.xy) - bigR, q.z);
    // Determine the angle in the cross–section, then modulate the minor radius.
    let a = atan2(k.y, k.x);
    return length(k) - (smallR + ridgeAmp * sin(ridgeFreq * a));
}

///////////////////////////////////////////////////////////////////////////////
// 2) Rotation helpers
///////////////////////////////////////////////////////////////////////////////
fn rotateZ(p: vec3f, angle: f32) -> vec3f {
    let c = cos(angle);
    let s = sin(angle);
    return vec3f(
        c * p.x - s * p.y,
        s * p.x + c * p.y,
        p.z
    );
}

fn rotateY(p: vec3f, angle: f32) -> vec3f {
    let c = cos(angle);
    let s = sin(angle);
    return vec3f(
        c * p.x - s * p.z,
        p.y,
        s * p.x + c * p.z
    );
}

fn rotateX(p: vec3f, angle: f32) -> vec3f {
    let c = cos(angle);
    let s = sin(angle);
    return vec3f(
        p.x,
        c * p.y - s * p.z,
        s * p.y + c * p.z,
    );
}

///////////////////////////////////////////////////////////////////////////////
// 3)  scene SDF: union of two links (with ridges)
///////////////////////////////////////////////////////////////////////////////
fn map(p: vec3f) -> f32 {
    // Ridge parameters
    let ridgeAmp = 0.02;
    let ridgeFreq = 9.0;

    // ------------------
    // Link #1: Oriented along Y.
    // Parameters: half length, major radius, tube radius.
    var pY =  p - vec3f(0., 0., 0.0);           // shift in -Y so they interlock

    var pR = rotateY(pY, 1.5708 * time.elapsed*.5);  
    pR = rotateZ(pR, 1.5708 * time.elapsed*.25);
    let d1 = sdLink(pR, 0.6, 0.6, 0.15, ridgeAmp, ridgeFreq);

    // ------------------
    // Link #2: Rotate by 90° and offset slightly for interlocking.
    var p2 = rotateY(pR, 1.5708);
    p2 = p2 - vec3f(0.0, 1.75, 0.0);
    let d2 = sdLink(p2, 0.6, 0.6, 0.15, ridgeAmp, ridgeFreq);

    var p3 = rotateY(pR, 1.5708);

    p3 = p3 - vec3f(0.0, -1.75, 0.0);
    let d3 = sdLink(p3, 0.6, 0.6, 0.15, ridgeAmp, ridgeFreq);
    var d1d2 = min(d1, d2);

    return min(d1d2, d3);
}

fn march(ro: vec3f, rd: vec3f) -> vec4f {
    var t = 0.0;
    var p: vec3f;
    for (var i = 0; i < 100; i++) {
        p = ro + rd * t;
        let dist = map(p);
        t += dist;
        if (dist < 0.001 || t > 50.0) {
            break;
        }
    }
    return vec4f(p, t / 50.0);
}

fn normal(p: vec3f) -> vec3f {
    let e = 0.0001;
    let d0 = map(p);
    let nx = map(p + vec3f(e, 0.0, 0.0)) - d0;
    let ny = map(p + vec3f(0.0, e, 0.0)) - d0;
    let nz = map(p + vec3f(0.0, 0.0, e)) - d0;
    return normalize(vec3f(nx, ny, nz));
}

@compute @workgroup_size(8, 8)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let res = textureDimensions(screen);

    // Skip out–of–bounds pixels
    if (id.x >= res.x || id.y >= res.y) {
        return;
    }

    // Tiny antialiasing
    let AA = 2.0;
    var color = vec3f(0.0);

    for (var i = 0.0; i < AA; i += 1.0) {
        for (var j = 0.0; j < AA; j += 1.0) {
            let dxy = (vec2f(i, j) + 0.5) / AA;
            let uv  = (2.0 * (vec2f(id.xy) + dxy) - vec2f(res)) / f32(res.y);

            // Camera setup
            let ro = vec3f(0.0, 0.0, 3.5);
            let rd = normalize(vec3f(uv, -1.5));

            // Ray march to compute hit position
            let r = march(ro, rd);
            let tFrac = r.w;
            let hitPos = r.xyz;

            if (tFrac < 1.0) {
                // We hit the geometry.
                let n = normal(hitPos);
                // Predefined light direction.
                let lightDir = normalize(vec3f(0.4, 0.8, 0.3));
                // Compute view direction.
                let viewDir = normalize(ro - hitPos);
                // Reflection vector for the view direction.
                let R = reflect(-viewDir, n);

                // Fake environment: a simple vertical gradient.
                let envFactor = clamp(0.5 * (R.y + 1.0), 0.0, 1.0);
                // Darker at the “bottom” and brighter (a subtle blue–tint) at the “top.”
                let envColor = mix(vec3f(0.04, 0.04, 0.06), vec3f(0.9, 0.95, 1.0), envFactor);

                // Blinn–Phong specular: a high exponent for a sharp mirror–like highlight.
                let halfVec = normalize(lightDir + viewDir);
                let spec = pow(max(dot(n, halfVec), 0.0), 128.0);

                // Fresnel term enhances the edge–reflection.
                let fresnel = pow(1.0 - max(dot(viewDir, n), 0.0), 3.0);

                // Combine the specular and the environment reflection.
                // Here the metal color is nearly pure reflection with a slight tint.
                let chromeColor = mix(envColor, vec3f(1.0, 1.0, 1.0), fresnel) * spec;
                // Add a little ambient term from the environment.
                let finalColor = chromeColor + envColor * 0.2;

                color += finalColor / (AA * AA);
            } else {
                // Background remains a dark bluish tone.
                color += vec3f(0.02, 0.02, 0.04) / (AA * AA);
            }
        }
    }

    textureStore(screen, vec2u(id.x, res.y - 1u - id.y), vec4f(color, 1.0));
}
