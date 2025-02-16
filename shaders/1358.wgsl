#storage image array<array<vec4<f32>, SCREEN_WIDTH>, SCREEN_HEIGHT>

// Constants
#define TAU 6.28318530718
#define PI 3.14159265359
#define RHO 1.57079632679
#define EPSILON 0.001

var<private> seed: u32;

// Structs
struct Ray {
    orig: vec3<f32>,
    dir: vec3<f32>
}

struct HitInfo {
    fromInside: bool,
    hitDist: f32,
    hitPos: vec3<f32>,
    surfNor: vec3<f32>,
    surfUv: vec2<f32>,
    objId: i32
}

struct Material {
    albedo: vec3<f32>,
    specular: vec3<f32>,
    absorption: vec3<f32>,
    emission: vec3<f32>,
    specularAmount: f32,
    specularRoughness: f32,
    refractionAmount: f32,
    refractionRoughness: f32,
    refractiveIndex: f32
}

// Defaults
fn setHitInfoMiss(hit: ptr<function, HitInfo>) {
    (*hit).fromInside = false;
    (*hit).hitDist = -1.0;
    (*hit).hitPos = vec3<f32>(0.0);
    (*hit).surfNor = vec3<f32>(0.0);
    (*hit).surfUv = vec2<f32>(0.0);
    (*hit).objId = -1;
}

fn setMaterialBase(mtl: ptr<function, Material>) {
    (*mtl).albedo = vec3<f32>(0.0);
    (*mtl).specular = vec3<f32>(0.0);
    (*mtl).absorption = vec3<f32>(0.0);
    (*mtl).emission = vec3<f32>(0.0);
    (*mtl).specularAmount = 0.0;
    (*mtl).specularRoughness = 0.0;
    (*mtl).refractionAmount = 0.0;
    (*mtl).refractionRoughness = 0.0;
    (*mtl).refractiveIndex = 1.0;
}

// Schlick aproximation
fn getFresnel(n1: f32, n2: f32, normal: vec3<f32>, incident: vec3<f32>, f0: f32, f90: f32) -> f32 {
    var r0 = (n1 - n2) / (n1 + n2);
    r0 *= r0;
    var cosX = -dot(normal, incident);
    if (n1 > n2) {
        var n = n1 / n2;
        var sinT2 = n * n * (1.0 - cosX * cosX);
        if (sinT2 > 1.0) { return f90; } // Total internal reflection
        cosX = sqrt(1.0 - sinT2);
    }

    var x = 1.0 - cosX;
    var x2 = x * x;
    return mix(f0, f90, mix(r0, 1.0, x2 * x2 * x));
}

// RNG utilities
fn wangHash() -> u32 {
    seed = (seed ^ 61u) ^ (seed >> 16u);
    seed *= 9u;
    seed ^= seed >> 4;
    seed *= 668265261u;
    seed ^= seed >> 15;
    return seed;
}

fn rand01() -> f32 {
    return f32(wangHash()) / 4294967296.0;
}

fn randInUnitDisc() -> vec2<f32> {
    var a = rand01() * TAU;
    return vec2<f32>(cos(a), sin(a)) * sqrt(rand01());
}

fn randUnitVec() -> vec3<f32> {
    var z = rand01() * 2.0 - 1.0;
    var a = rand01() * TAU;
    var r = sqrt(1.0 - z * z);
    var x = r * cos(a);
    var y = r * sin(a);
    return vec3<f32>(x, y, z);
}

// Camera basis matrix
fn getCameraBasis(an: vec2<f32>) -> mat3x3<f32> {
    var matrix: mat3x3<f32>;
    var c = cos(an);
    var s = sin(an);
    matrix[0] = vec3<f32>(c.x, 0.0, s.x);
    matrix[1] = vec3<f32>(s.x * -s.y, c.y, c.x * s.y);
    matrix[2] = vec3<f32>(s.x * c.y, s.y, -c.x * c.y);
    return matrix;
}

// Replaces the current hit with a new one if it is closer
fn addRayHit(curHit: ptr<function, HitInfo>, newHit: ptr<function, HitInfo>, objId: i32) {
    if ((*newHit).hitDist > 0.0 && ((*curHit).hitDist < 0.0 || (*newHit).hitDist < (*curHit).hitDist)) {
        *curHit = *newHit;
        (*curHit).objId = objId;
    }
}

// Matrix math utilities
fn inverse(m: mat4x4<f32>) -> mat4x4<f32> {
    let a00 = m[0][0]; let a01 = m[0][1]; let a02 = m[0][2]; let a03 = m[0][3];
    let a10 = m[1][0]; let a11 = m[1][1]; let a12 = m[1][2]; let a13 = m[1][3];
    let a20 = m[2][0]; let a21 = m[2][1]; let a22 = m[2][2]; let a23 = m[2][3];
    let a30 = m[3][0]; let a31 = m[3][1]; let a32 = m[3][2]; let a33 = m[3][3];

    let b00 = a00 * a11 - a01 * a10;
    let b01 = a00 * a12 - a02 * a10;
    let b02 = a00 * a13 - a03 * a10;
    let b03 = a01 * a12 - a02 * a11;
    let b04 = a01 * a13 - a03 * a11;
    let b05 = a02 * a13 - a03 * a12;
    let b06 = a20 * a31 - a21 * a30;
    let b07 = a20 * a32 - a22 * a30;
    let b08 = a20 * a33 - a23 * a30;
    let b09 = a21 * a32 - a22 * a31;
    let b10 = a21 * a33 - a23 * a31;
    let b11 = a22 * a33 - a23 * a32;

    let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    return mat4x4<f32>(
        (a11 * b11 - a12 * b10 + a13 * b09) / det,
        (a02 * b10 - a01 * b11 - a03 * b09) / det,
        (a31 * b05 - a32 * b04 + a33 * b03) / det,
        (a22 * b04 - a21 * b05 - a23 * b03) / det,
        (a12 * b08 - a10 * b11 - a13 * b07) / det,
        (a00 * b11 - a02 * b08 + a03 * b07) / det,
        (a32 * b02 - a30 * b05 - a33 * b01) / det,
        (a20 * b05 - a22 * b02 + a23 * b01) / det,
        (a10 * b10 - a11 * b08 + a13 * b06) / det,
        (a01 * b08 - a00 * b10 - a03 * b06) / det,
        (a30 * b04 - a31 * b02 + a33 * b00) / det,
        (a21 * b02 - a20 * b04 - a23 * b00) / det,
        (a11 * b07 - a10 * b09 - a12 * b06) / det,
        (a00 * b09 - a01 * b07 + a02 * b06) / det,
        (a31 * b01 - a30 * b03 - a32 * b00) / det,
        (a20 * b03 - a21 * b01 + a22 * b00) / det
    );
}

// Transformations
fn Identity() -> mat4x4<f32> {
    return mat4x4<f32>(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
}

fn Translate(t: vec3<f32>) -> mat4x4<f32> {
    var matrix = Identity();
    matrix[3] = vec4<f32>(t, 1.0);
    return matrix;
}

fn Scale(s: vec3<f32>) -> mat4x4<f32> {
    var matrix = Identity();
    matrix[0][0] = s.x;
    matrix[1][1] = s.y;
    matrix[2][2] = s.z;
    return matrix;
}

fn Rotate(a: f32, i: u32, j: u32) -> mat4x4<f32> {
    var matrix = Identity();
    var co = cos(a);
    var si = sin(a);
    matrix[i][i] = co;
    matrix[i][j] = si;
    matrix[j][i] = -si;
    matrix[j][j] = co;
    return matrix;
}

// Primitive intersectors
// Ray vs. Plane intersection
fn planeIsect(hit: ptr<function, HitInfo>, ray: ptr<function, Ray>, mat: mat4x4<f32>) {
    setHitInfoMiss(hit);
    var matInv = inverse(mat);
    var rayOrig = (matInv * vec4<f32>((*ray).orig, 1.0)).xyz;
    var rayDir = (matInv * vec4<f32>((*ray).dir, 0.0)).xyz;

    (*hit).hitDist = -rayOrig.y / rayDir.y;
    (*hit).hitPos = rayOrig + rayDir * (*hit).hitDist;
    (*hit).fromInside = rayOrig.y < 0.0;
    (*hit).surfNor = transpose(matInv)[1].xyz * sign(rayOrig.y);
    (*hit).surfUv = (*hit).hitPos.xz;
    (*hit).hitPos = (mat * vec4((*hit).hitPos, 1.0)).xyz;
}

// Ray vs. Sphere intersection
fn sphereIsect(hit: ptr<function, HitInfo>, ray: ptr<function, Ray>, mat: mat4x4<f32>) {
    setHitInfoMiss(hit);
    var matInv = inverse(mat);
    var rayOrig = (matInv * vec4<f32>((*ray).orig, 1.0)).xyz;
    var rayDir = (matInv * vec4<f32>((*ray).dir, 0.0)).xyz;

    var a = dot(rayDir, rayDir);
    var b = dot(rayOrig, rayDir);
    var c = dot(rayOrig, rayOrig) - 1.0;

    var dis = b * b - a * c;
    if (dis < 0.0) { return; }

    (*hit).hitDist = (-b - sign(c) * sqrt(dis)) / a;
    (*hit).hitPos = rayOrig + rayDir * (*hit).hitDist;
    (*hit).fromInside = c < 0.0;
    (*hit).surfNor = (transpose(matInv) * vec4<f32>((*hit).hitPos, 0.0)).xyz * sign(c);
    (*hit).surfUv = vec2<f32>(atan2((*hit).hitPos.z, (*hit).hitPos.x), atan2((*hit).hitPos.y, length((*hit).hitPos.xz)));
    (*hit).hitPos = (mat * vec4<f32>((*hit).hitPos, 1.0)).xyz;
}

// Not so primitive intersectors
// Ray vs. Bilinear Patch intersection
fn cross2D(a: vec2<f32>, b: vec2<f32>) -> f32 { return a.x * b.y - a.y * b.x; }
fn patchIsect(hit: ptr<function, HitInfo>, ray: ptr<function, Ray>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>, solid: bool) {
    setHitInfoMiss(hit);
    var m = (*ray).dir.yz / (*ray).dir.x;
    var p = a - b + c - d;
    var q = d - a;
    var r = b - a;
    var ip = p.yz - p.x * m;
    var iq = q.yz - q.x * m;
    var ir = r.yz - r.x * m;
    var ia = (a.yz - (*ray).orig.yz) - (a.x - (*ray).orig.x) * m;

    // Reversed coefficients (solving for 1/x seems most stable)
    var qa = cross2D(iq, ia);
    var qb = cross2D(ip, ia) + cross2D(iq, ir);
    var qc = cross2D(ip, ir);

    var dis = qb * qb - 4.0 * qa * qc;
    if (dis > 0.0) {
        var v = 2.0 * qa / (vec2<f32>(-1.0, 1.0) * sqrt(dis) - qb);
        var u = -(ia.x + ir.x * v) / (ip.x * v + iq.x);
        var t = (p.x * u * v + q.x * u + r.x * v + a.x - (*ray).orig.x) / (*ray).dir.x;

        // Check the validity of both intersections
        var valid = vec2<bool>(
            t.x > 0.0 && u.x >= 0.0 && u.x <= 1.0 && v.x >= 0.0 && v.x <= 1.0,
            t.y > 0.0 && u.y >= 0.0 && u.y <= 1.0 && v.y >= 0.0 && v.y <= 1.0
        );

        // Choose closest intersection in the patch
        if (valid.y && (!valid.x || t.y < t.x)) {
            u = u.yx;
            v = v.yx;
            t = t.yx;
            valid = valid.yx;
        }

        if (valid.x) {
            (*hit).hitDist = t.x;
            (*hit).hitPos = (*ray).orig + (*ray).dir * t.x;
            (*hit).surfNor = cross(p * u.x + r, p * v.x + q);
            var viewAngle = dot((*hit).surfNor, -(*ray).dir);
            (*hit).fromInside = solid && viewAngle < 0.0;
            (*hit).surfNor *= sign(viewAngle);
            (*hit).surfUv = vec2<f32>(u.x, v.x);
        }
    }
}

// https://64.github.io/tonemapping/#uncharted-2
fn Uncharted2(color: vec3<f32>) -> vec3<f32> {
    var toned = color * 2.0;

    let A = custom.UnchartedA; let B = custom.UnchartedB; let C = custom.UnchartedC;
    let D = custom.UnchartedD; let E = custom.UnchartedE; let F = custom.UnchartedF;
    toned = (((A * toned + C * B) * toned + D * E) / ((A * toned + B) * toned + D * F)) - E / F;

    let whiteMax = custom.whiteLevel;
    toned /= (((A * whiteMax + C * B) * whiteMax + D * E) / ((A * whiteMax + B) * whiteMax + D * F)) - E / F;

    return toned;
}

// Thanks @fad
fn sphericalMap(dir: vec3<f32>) -> vec2f {
    var uv = vec2<f32>(atan2(dir.z, dir.x), asin(dir.y));
    return uv * vec2<f32>(0.5 / PI, -1.0 / PI) + 0.5;
}

fn traceRay(result: ptr<function, HitInfo>, ray: ptr<function, Ray>) {
    setHitInfoMiss(result);
    var hit: HitInfo;

    planeIsect(&hit, ray, Identity());
    addRayHit(result, &hit, 1);

    sphereIsect(&hit, ray, Translate(vec3<f32>(0.0, 0.7, -3.0)) * Scale(vec3<f32>(0.4)));
    addRayHit(result, &hit, 2);

    sphereIsect(&hit, ray, Translate(vec3<f32>(3.5, 1.5, -5.0)) * Scale(vec3<f32>(0.75)));
    addRayHit(result, &hit, 3);

    var jellyMat = Translate(vec3<f32>(0.0, 1.5, -4.5)) * Rotate(-0.25, 2, 1) * Scale(vec3<f32>(0.75));
    var topTwist = Rotate(-1.5, 0, 2);
    var bottomTwist = Rotate(0.0, 0, 2);
    var wiggle = 0.5;
    var v0 = (jellyMat * bottomTwist * vec4<f32>(-1.0, -1.0 + wiggle, -1.0, 1.0)).xyz;
    var v1 = (jellyMat * bottomTwist * vec4<f32>(-1.0, -1.0 - wiggle,  1.0, 1.0)).xyz;
    var v2 = (jellyMat *    topTwist * vec4<f32>(-1.0,  1.0 - wiggle, -1.0, 1.0)).xyz;
    var v3 = (jellyMat *    topTwist * vec4<f32>(-1.0,  1.0 + wiggle,  1.0, 1.0)).xyz;
    var v4 = (jellyMat * bottomTwist * vec4<f32>( 1.0, -1.0 - wiggle, -1.0, 1.0)).xyz;
    var v5 = (jellyMat * bottomTwist * vec4<f32>( 1.0, -1.0 + wiggle,  1.0, 1.0)).xyz;
    var v6 = (jellyMat *    topTwist * vec4<f32>( 1.0,  1.0 + wiggle, -1.0, 1.0)).xyz;
    var v7 = (jellyMat *    topTwist * vec4<f32>( 1.0,  1.0 - wiggle,  1.0, 1.0)).xyz;

    patchIsect(&hit, ray, v7, v5, v4, v6, true);
    addRayHit(result, &hit, 4);
    patchIsect(&hit, ray, v2, v0, v1, v3, true);
    addRayHit(result, &hit, 4);
    patchIsect(&hit, ray, v2, v3, v7, v6, true);
    addRayHit(result, &hit, 4);
    patchIsect(&hit, ray, v1, v0, v4, v5, true);
    addRayHit(result, &hit, 4);
    patchIsect(&hit, ray, v6, v4, v0, v2, true);
    addRayHit(result, &hit, 4);
    patchIsect(&hit, ray, v3, v1, v5, v7, true);
    addRayHit(result, &hit, 4);

    sphereIsect(&hit, ray, Translate(vec3(1.3, 1.5, -3.0)) * Scale(vec3(0.75)));
    addRayHit(result, &hit, 5);

    sphereIsect(&hit, ray, Translate(vec3(3.0, 1.5, -2.0)) * Scale(vec3(0.75)));
    addRayHit(result, &hit, 6);

    sphereIsect(&hit, ray, Translate(vec3(1.0, 1.2, -1.0)) * Scale(vec3(0.25)));
    addRayHit(result, &hit, 7);
}

fn setObjMaterial(mtl: ptr<function, Material>, surfUv: vec2<f32>, objId: i32) {
    setMaterialBase(mtl);
    switch (objId) {
        case 1: {
            (*mtl).albedo = vec3<f32>(fract((floor(surfUv.x) + floor(surfUv.y)) / 2.0) * 2.0);
            (*mtl).specular = vec3<f32>(1.0, 0.4, 0.2);
            (*mtl).specularAmount = 0.5;
            (*mtl).specularRoughness = 0.7;
            return;
        }

        case 2: {
            (*mtl).absorption = vec3<f32>(0.0, 0.0, 2.0);
            (*mtl).specular = vec3<f32>(1.0);
            (*mtl).specularAmount = 0.1;
            (*mtl).specularRoughness = 0.3;
            (*mtl).refractionAmount = 0.9;
            (*mtl).refractionRoughness = 0.5;
            (*mtl).refractiveIndex = 1.5;
            return;
        }

        case 3: {
            (*mtl).specular = vec3<f32>(1.0);
            (*mtl).absorption = vec3<f32>(1.0, 0.5, 0.0);
            (*mtl).specularAmount = 0.1;
            (*mtl).refractionAmount = 0.9;
            (*mtl).refractiveIndex = 1.5;
            return;
        }

        case 4: {
            (*mtl).albedo = vec3<f32>(fract((floor(surfUv.x * 8.0) + floor(surfUv.y * 8.0)) / 2.0) * 2.0);
            return;
        }

        case 5: {
            (*mtl).albedo = vec3<f32>(0.9, 0.25, 0.25);
            (*mtl).specular = vec3<f32>(0.8);
            (*mtl).specularAmount = 0.02;
            return;
        }

        case 6: {
            (*mtl).albedo = vec3<f32>(0.9, 0.25, 0.25);
            (*mtl).specular = vec3<f32>(0.8);
            (*mtl).absorption = vec3<f32>(0.0, 1.5, 3.0);
            (*mtl).specularAmount = 0.02;
            (*mtl).refractionAmount = 0.98;
            (*mtl).refractiveIndex = 1.5;
            return;
        }

        case 7: {
            (*mtl).albedo = vec3<f32>(0.0, 1.0, 0.0);
            (*mtl).specular = vec3<f32>(0.0, 1.0, 0.0);
            (*mtl).specularAmount = 1.0;
            (*mtl).specularRoughness = 0.5;
            return;
        }

        default: {
            return;
        }
    }
}

#dispatch_once initialize
@compute @workgroup_size(16, 16)
fn initialize(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= SCREEN_WIDTH || id.y >= SCREEN_HEIGHT) { return; }
    image[id.y][id.x] = vec4<f32>(0.0);
}

@compute @workgroup_size(16, 16)
fn pathtrace(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= SCREEN_WIDTH || id.y >= SCREEN_HEIGHT) { return; }

    seed = (id.x * 1973u + id.y * 9277u + time.frame * 26699u) | 1u;

    // Jitter within the pixel for antialiasing
    var coord = vec2<f32>(f32(id.x) + rand01(), f32(id.y) + rand01());
    var viewportCenter = vec2<f32>(f32(SCREEN_WIDTH), f32(SCREEN_HEIGHT)) / 2.0;
    var uv = (coord - viewportCenter) / f32(SCREEN_HEIGHT);
    uv.y = -uv.y;

    // Calculate a ray for the pixel
    var mousePos = (vec2<f32>(f32(mouse.pos.x), f32(mouse.pos.y)) - viewportCenter) / f32(SCREEN_HEIGHT);
    mousePos.y = -mousePos.y;
    if (mouse.pos.x == 0 && mouse.pos.y == 0) { mousePos = vec2<f32>(0.07, -0.04); }
    var camBasis = getCameraBasis(mousePos * TAU);

    // Adding depth of field as a bonus
    // Based on this lesson https://pathtracing.home.blog/depth-of-field/
    var primaryRay = Ray(vec3<f32>(custom.cameraPosX, custom.cameraPosY, custom.cameraPosZ), camBasis * normalize(vec3<f32>(uv, custom.lensDistance)));
    var aperturePoint = camBasis * vec3<f32>(randInUnitDisc() * custom.apertureRadius, 0.0);
    var ray = Ray(primaryRay.orig + aperturePoint, normalize(primaryRay.dir * custom.focusDistance - aperturePoint));

    // Pathtrace!
    var acc = vec3<f32>(0.0);
    var throughput = vec3<f32>(1.0);
    for (var bounce = 0u; bounce < u32(custom.maxRayBounces); bounce++) {
        var hit: HitInfo;
        traceRay(&hit, &ray);
        if (hit.hitDist < 0.0) {
            acc += pow(textureSampleLevel(channel0, bilinear, sphericalMap(ray.dir), 0.0).rgb, vec3(2.2)) * throughput;
            break;
        }

        hit.surfNor = normalize(hit.surfNor);
        var mtl: Material;
        setObjMaterial(&mtl, hit.surfUv, hit.objId);

        // If the ray hit inside, some light gets absorbed
        if (hit.fromInside) { throughput *= exp(-mtl.absorption * hit.hitDist); }

        // Initial chances of reflecting or refracting
        var specularChance = mtl.specularAmount;
        var refractionChance = mtl.refractionAmount;

        if (specularChance > 0.0) {
            // Adjust specular chance to account for the Fresnel effect
            specularChance = getFresnel(
                select(1.0, mtl.refractiveIndex, hit.fromInside),
                select(mtl.refractiveIndex, 1.0, hit.fromInside),
                ray.dir, hit.surfNor, mtl.specularAmount, 1.0
            );

            // Make sure diffuse / refraction ratio is the same
            // Diffuse chance is implied (1 - specularChance - refractionChance)
            var correctionRatio = (1.0 - specularChance) / (1.0 - mtl.specularAmount);
            refractionChance *= correctionRatio;
        }

        // Choose whether to diffuse, reflect, or refract
        var doSpecular = 0.0;
        var doRefraction = 0.0;
        var rayProbability = 1.0 - specularChance - refractionChance;
        var selector = rand01();
        if (specularChance > 0.0 && selector < specularChance) {
            doSpecular = 1.0;
            rayProbability = specularChance;
        } else if (refractionChance > 0.0 && selector < specularChance + refractionChance) {
            doRefraction = 1.0;
            rayProbability = refractionChance;
        }

        // Step to the intersection and push off the surface a tiny bit
        ray.orig = hit.hitPos + hit.surfNor * select(EPSILON, -EPSILON, doRefraction == 1.0);

        // Calculate a new ray direction
        // Diffuse uses a random reflection from a cosine distribution about the normal
        // Specular uses the perfect reflection across the normal
        // Refraction uses the perfect refraction across the normal
        // Squaring the roughness is just a convention to make roughness appear more linear
        var diffuseRay = normalize(hit.surfNor + randUnitVec());
        var specularRay = reflect(ray.dir, hit.surfNor);
        specularRay = normalize(mix(specularRay, diffuseRay, mtl.specularRoughness * mtl.specularRoughness));
        var refractionRay = refract(ray.dir, hit.surfNor, select(1.0 / mtl.refractiveIndex, mtl.refractiveIndex, hit.fromInside));
        refractionRay = normalize(mix(refractionRay, normalize(-hit.surfNor + randUnitVec()), mtl.refractionRoughness * mtl.refractionRoughness));
        ray.dir = mix(diffuseRay, specularRay, doSpecular);
        ray.dir = mix(ray.dir, refractionRay, doRefraction);

        // Accumulate light emission from the surface
        acc += mtl.emission * throughput;

        // Update the throughput for diffuse and specular reflections only
        if (doRefraction == 0.0) { throughput *= mix(mtl.albedo, mtl.specular, doSpecular); }

        // Adjust the throughput to account for the actions that got discarded
        throughput /= max(EPSILON, rayProbability);

        // Russian roulette optimization
        // Increases the chance of terminating as the throughput decreases
        // Surviving samples get boosted to make up for the eliminated ones
        var stopChance = max(throughput.r, max(throughput.g, throughput.b));
        if (rand01() > stopChance) { break; }
        throughput /= stopChance;
    }

    // Load previously accumulated samples (or reset on click)
    var sample = select(image[id.y][id.x], vec4<f32>(0.0), mouse.click > 0);

    if (length(acc) < custom.outlierThreshold) { // Reject outliers
        // Combine the new sample with the current average
        sample.w += 1.0;
        sample = vec4<f32>(mix(sample.rgb, acc, 1.0 / sample.w), sample.w);
    }

    image[id.y][id.x] = sample;
}

@compute @workgroup_size(16, 16)
fn tonemap(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= SCREEN_WIDTH || id.y >= SCREEN_HEIGHT) { return; }
    textureStore(screen, id.xy, vec4<f32>(pow(Uncharted2(image[id.y][id.x].rgb), vec3<f32>(0.4545)), 1.0));
}