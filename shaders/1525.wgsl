
const TAU    = 6.28318530717958647693;
const PI     = 3.14159265358979323846;
const INVPI  = 0.31830988618379067154;

fn sqr(x: f32) -> f32 { return x * x; }

fn inverse(m: mat4x4f) -> mat4x4f
{
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
    
    return mat4x4f(a11 * b11 - a12 * b10 + a13 * b09,
                   a02 * b10 - a01 * b11 - a03 * b09,
                   a31 * b05 - a32 * b04 + a33 * b03,
                   a22 * b04 - a21 * b05 - a23 * b03,
                   a12 * b08 - a10 * b11 - a13 * b07,
                   a00 * b11 - a02 * b08 + a03 * b07,
                   a32 * b02 - a30 * b05 - a33 * b01,
                   a20 * b05 - a22 * b02 + a23 * b01,
                   a10 * b10 - a11 * b08 + a13 * b06,
                   a01 * b08 - a00 * b10 - a03 * b06,
                   a30 * b04 - a31 * b02 + a33 * b00,
                   a21 * b02 - a20 * b04 - a23 * b00,
                   a11 * b07 - a10 * b09 - a12 * b06,
                   a00 * b09 - a01 * b07 + a02 * b06,
                   a31 * b01 - a30 * b03 - a32 * b00,
                   a20 * b03 - a21 * b01 + a22 * b00) * (1. / det);
}

fn translate(p : vec3f) -> mat4x4f
{
    return mat4x4f(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, p.x, p.y, p.z, 1);
}


fn rot(a: f32) -> mat2x2f { return mat2x2f(cos(a), -sin(a), sin(a), cos(a));}

fn rotateX(a: f32) -> mat4x4f
{
    return mat4x4f(1,      0,       0, 0,
                   0, cos(a), -sin(a), 0,
                   0, sin(a),  cos(a), 0,
			       0,      0,       0, 1);
}

fn rotateY(a: f32) -> mat4x4f
{
    return mat4x4f( cos(a), 0, sin(a), 0,
                         0, 1,      0, 0,
                   -sin(a), 0, cos(a), 0,
			             0, 0,      0, 1);
}

fn rotateZ(a: f32) -> mat4x4f
{
    return mat4x4f(cos(a), -sin(a), 0, 0,
                   sin(a),  cos(a), 0, 0,
                        0,       0, 1, 0,
			            0,       0, 0, 1);
}

const noTransform = mat4x4f(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);

struct ray
{
    o: vec3f,
    d: vec3f
}

struct material
{
    baseColor: vec3f,
    emission: vec3f,
    
    anisotropic: f32,
    metallic: f32,
    roughness: f32,
    subsurface: f32,
    specularTint: f32,
    sheen: f32,
    sheenTint: f32,
    clearcoat: f32,
    clearcoatRoughness: f32,
    specTrans: f32,
    IOR: f32,
    ax: f32,
    ay: f32
}

const initMat = material(vec3f(0), vec3f(0), 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.);

fn DIFFUSE(col: vec3f, roughness: f32) -> material
{
    var mat = initMat;
    mat.baseColor = col;
    mat.roughness = roughness;
    
    return mat;
}

fn METAL(col: vec3f, roughness: f32) -> material
{
    var mat = initMat;
    mat.baseColor = col;
    mat.metallic  = 1.;
    mat.roughness = roughness;
    
    return mat;
}

fn GLASS(col: vec3f, IOR: f32, roughness: f32) -> material
{
    var mat = initMat;
    mat.baseColor = col;
    mat.roughness = roughness;
    mat.specTrans = 1.;
    mat.IOR       = IOR;
    
    return mat;
}

fn LIGHT(col: vec3f) -> material
{
    var mat = initMat;
    mat.emission = col;
    
    return mat;
}

fn VOLUME(col: vec3f) -> material
{
    var mat = initMat;
    mat.baseColor = col;
    
    return mat;
}

struct hitRecord
{
    t: f32,
    eta: f32,
    p: vec3f,
    n: vec3f,
    mat: material,
    isVolume: bool
}

struct cam
{
    o: vec3f,
    llc: vec3f,
    hor: vec3f,
    ver: vec3f,
    u: vec3f,
    v: vec3f,
    w: vec3f,
    rad: f32
}

var<private> seed: vec4u;

fn PCG(x: vec4u) -> vec4u
{
    var v = x;
    
    v = v * vec4(1664525) + vec4(1013904223);
    
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    
    v ^= v >> vec4(16);
    
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    
    return v;
}

fn rand4() -> vec4f
{
    seed = PCG(seed);
    return vec4f(seed) / exp2(32);
}

fn rand() -> f32 { return rand4().x; }

fn rand2() -> vec2f { return rand4().xy; }

fn hash44(p: vec4f) -> vec4f
{
	return vec4f(PCG(bitcast<vec4u>(p))) / exp2(32);
}

fn hash14(p: f32) -> vec4f
{
	return hash44(vec4f(p));
}

fn hash24(p: vec2f) -> vec4f
{
	return hash44(p.xyxy);
}

fn hash34(p: vec3f) -> vec4f
{
	return hash44(p.xyzx);
}

fn hash22(p: vec2f) -> vec2f
{
	return hash24(p).xy;
}

fn hash41(p: vec4f) -> f32
{
    return hash44(p).x;
}

fn ranDir() -> vec2f
{
    let phi = TAU * rand();
    return vec2f(cos(phi), sin(phi));
}

fn ranDisk() -> vec2f { return sqrt(rand()) * ranDir(); }

// Sampling Functions

fn GTR1(NoH: f32, a: f32) -> f32
{
    if (a >= 1.)
    {
        return INVPI;
    }

    return INVPI * (a * a - 1.) / (log(a * a) * (1. + (a * a - 1.) * NoH * NoH));
}


fn sampleGTR1(a: f32) -> vec3f
{
    let cos2T = (1. - pow(a * a, rand())) / (1. - a * a);
    return vec3f(sqrt(1. - cos2T) * ranDir(), sqrt(cos2T));
}

fn sampleGGXVNDF(V: vec3f, ax: f32, ay: f32) -> vec3f
{
    let Vh = normalize(vec3f(ax * V.x, ay * V.y, V.z));
    
    var t   = ranDisk();
        t.y = mix(sqrt(1. - t.x * t.x), t.y, .5 + .5 * Vh.z);
    
    let l2 = dot(Vh.xy, Vh.xy);
    
    let T1 = select(vec3(1, 0, 0), vec3(-Vh.y, Vh.x, 0) / sqrt(l2), l2 > 0.) ;
    let T2 = cross(Vh, T1);
    let N  = mat3x3f(T1, T2, Vh) * vec3f(t, sqrt(1. - dot(t, t)));

    return normalize(vec3f(ax * N.x, ay * N.y, max(0., N.z)));
}

fn GTR2Aniso(NoH: f32, HoX: f32, HoY: f32, ax: f32, ay: f32) -> f32
{
    if(ax * ay == 0.)
    {
        return 1.;
    }
    
    let a = HoX / ax;
    let b = HoY / ay;
    let c = a * a + b * b + NoH * NoH;
    return INVPI / (ax * ay * c * c);
}

fn smithG(NoV: f32, alphaG: f32) -> f32
{
    let a = alphaG * alphaG;
    let b = NoV * NoV;
    
    return 2. * NoV / (NoV + sqrt(a + b - a * b));
}

fn smithGAniso(NoV: f32, VoX: f32, VoY: f32, ax: f32, ay: f32) -> f32
{
    if(ax * ay == 0.)
    {
        return 1.;
    }

    let a = VoX * ax;
    let b = VoY * ay;
    let c = NoV;
    
    return 2. * NoV / (NoV + sqrt(a * a + b * b + c * c));
}

fn schlickWeight(u: f32) -> f32
{
    let m = 1. - u;
    let m2 = m * m;
    
    return m2 * m2 * m;
}


fn dielectricFresnel(cosI: f32, eta: f32) -> f32
{
    let sin2T = eta * eta * (1. - cosI * cosI);

    // Total internal reflection
    
    if (sin2T > 1.)
    {
        return 1.;
    }

    let cosT = sqrt(1. - sin2T);

    let rs = (eta * cosT - cosI) / (eta * cosT + cosI);
    let rp = (eta * cosI - cosT) / (eta * cosI + cosT);

    return .5 * (rs * rs + rp * rp);
}

fn ranCos() -> vec3f
{
    let r = rand();
    return vec3f(sqrt(r) * ranDir(), sqrt(1. - r));
}

fn ranSph() -> vec3f
{
    let h = rand() * 2. - 1.;
	return vec3f(sqrt(1. - h * h) * ranDir(), h);
}

fn ranHemi(n: vec3f) -> vec3f
{
    let d = ranSph();
    return d * sign(dot(d, n));
}

fn onb(N: vec3f, T: ptr<function, vec3f>, B: ptr<function, vec3f>) 
{
    let s = select(-1., 1., N.z > 0.);
    let a = s + N.z;
    let b = -N.x * N.y / a;
    
    *T = s * vec3(s - N.x * N.x / a, b, -N.x);
    *B =     vec3(b, s - N.y * N.y / a, -N.y);
}

// Disney BSDF

fn luma(v: vec3f) -> f32 { return dot(v, vec3(.3, .6, .1)); }

fn tint(mat: material, eta: f32, F0: ptr<function, f32>, Csheen: ptr<function, vec3f>, Cspec: ptr<function, vec3f>)
{
    let lum = luma(mat.baseColor);
    let tint = select(vec3(1), mat.baseColor / lum, lum > 0.);

    *F0 = sqr((1. - eta) / (1. + eta));
    
    *Cspec  = mix(vec3(1), tint, mat.specularTint) * *F0;
    *Csheen = mix(vec3(1), tint, mat.sheenTint);
}

fn diffuse(mat: material, Csheen: vec3f, V: vec3f, L: vec3f, H: vec3f, pdf: ptr<function, f32>) -> vec3f
{
    let LoH = dot(L, H);
    let Rr  = 2. * mat.roughness * LoH * LoH;

    // Diffuse
    
    let FL     = schlickWeight(L.z);
    let FV     = schlickWeight(V.z);
    let Fretro = Rr * (FL + FV + FL * FV * (Rr - 1.));
    let Fd     = (1. - .5 * FL) * (1. - .5 * FV);

    // Fake subsurface
    
    let Fss90 = .5 * Rr;
    let Fss   = mix(1., Fss90, FL) * mix(1., Fss90, FV);
    let ss    = 1.25 * (Fss * (1. / (L.z + V.z) - .5) + .5);

    // Sheen
    
    let Fsheen = schlickWeight(LoH) * mat.sheen * Csheen;

    *pdf = L.z * INVPI;
    
    return INVPI * mat.baseColor * mix(Fd + Fretro, ss, mat.subsurface) + Fsheen;
}

fn reflection(mat: material, V: vec3f, L: vec3f, H: vec3f, F: vec3f, pdf: ptr<function, f32>) -> vec3f
{
    let D  = GTR2Aniso(H.z, H.x, H.y, mat.ax, mat.ay);
    let G1 = smithGAniso(abs(V.z), V.x, V.y, mat.ax, mat.ay);
    let G2 = smithGAniso(abs(L.z), L.x, L.y, mat.ax, mat.ay) * G1;

    *pdf = .25 * G1 * D / V.z;
    
    return vec3f(F * D * G2 / (4. * L.z * V.z));
}

fn refraction(mat: material, eta: f32, V: vec3f, L: vec3f, H: vec3f, F: vec3f, pdf: ptr<function, f32>) -> vec3f
{
    let LoH = dot(L, H);
    let VoH = dot(V, H);

    let D  = GTR2Aniso(H.z, H.x, H.y, mat.ax, mat.ay);
    let G1 = smithGAniso(abs(V.z), V.x, V.y, mat.ax, mat.ay);
    let G2 = smithGAniso(abs(L.z), L.x, L.y, mat.ax, mat.ay) * G1;
    
    let jacobian = abs(LoH) / sqr(LoH + VoH * eta);

    *pdf = G1 * max(0., VoH) * D * jacobian / V.z;
    
    return sqrt(mat.baseColor) * (1. - F) * D * G2 * abs(VoH) * jacobian * eta * eta / abs(L.z * V.z);
}

fn clearcoat(mat: material, V: vec3f, L: vec3f, H: vec3f, pdf: ptr<function, f32>) -> f32
{
    let VoH = dot(V, H);

    let F = mix(.04, 1., schlickWeight(VoH));
    let D = GTR1(H.z, mat.clearcoatRoughness);
    let G = smithG(L.z, .25) * smithG(V.z, .25);

    *pdf = .25 * D * H.z / VoH;
    
    return F * D * G;
}

fn toWorld(x: vec3f, y: vec3f, z: vec3f, v: vec3f) -> vec3f
{
    return mat3x3f(x, y, z) * v;
}

fn toLocal(x: vec3f, y: vec3f, z: vec3f, v: vec3f) -> vec3f
{
    return v * mat3x3f(x, y, z);
}

fn DisneyBSDF(rec: ptr<function, hitRecord>, V: vec3f, N: vec3f, L: vec3f, pdf: ptr<function, f32>) -> vec3f
{
    if(rec.isVolume)
    {
        *pdf = .25 * INVPI;
        return rec.mat.baseColor * *pdf;
    }
    
    let aspect = sqrt(1. - rec.mat.anisotropic * .9);
    rec.mat.ax = rec.mat.roughness / aspect;
    rec.mat.ay = rec.mat.roughness * aspect;
    
    *pdf = 0.;
    var f = vec3f(0);

    var T: vec3f;
    var B: vec3f;
    onb(N, &T, &B);

    let Vh = toLocal(T, B, N, V);
    let Lh = toLocal(T, B, N, L);

    var H = normalize(Lh + select(Vh * rec.eta, Vh, Lh.z > 0.));

    H *= sign(H.z);

    // Tint colors
    
    var Csheen: vec3f;
    var Cspec: vec3f;
    var F0: f32;
    tint(rec.mat, rec.eta, &F0, &Csheen, &Cspec);

    // Model weights
    
    let dielectricW = (1. - rec.mat.metallic) * (1. - rec.mat.specTrans);
    let metalW      = rec.mat.metallic;
    let glassW      = (1. - rec.mat.metallic) * rec.mat.specTrans;

    // Lobe probabilities
    
    let schlickW = schlickWeight(V.z);

    var diffP       = dielectricW * luma(rec.mat.baseColor);
    var dielectricP = dielectricW * mix(luma(Cspec), 1., schlickW);
    var metalP      = metalW * mix(luma(rec.mat.baseColor), 1., schlickW);
    var glassP      = glassW;
    var clearCoatP  = .25 * rec.mat.clearcoat;

    // Normalize probabilities
    
    let norm = 1. / (diffP + dielectricP + metalP + glassP + clearCoatP);
    
    diffP       *= norm;
    dielectricP *= norm;
    metalP      *= norm;
    glassP      *= norm;
    clearCoatP  *= norm;

    let reflect = Lh.z > 0.;

    var tmpPdf = 0.;
    let VoH = abs(dot(Vh, H));

    // Diffuse
    if (diffP > 0. && reflect)
    {
        f += diffuse(rec.mat, Csheen, Vh, Lh, H, &tmpPdf) * dielectricW;
        *pdf += tmpPdf * diffP;
    }

    // Dielectric Reflection
    if (dielectricP > 0. && reflect)
    {
        let F = (dielectricFresnel(VoH, 1. / rec.eta) - F0) / (1. - F0);

        f += reflection(rec.mat, Vh, Lh, H, mix(Cspec, vec3(1), F), &tmpPdf) * dielectricW;
        *pdf += tmpPdf * dielectricP;
    }

    // Metallic Reflection
    if (metalP > 0.0 && reflect)
    {
        // Tinted to base color
        let F = mix(rec.mat.baseColor, vec3(1), schlickWeight(VoH));

        f += reflection(rec.mat, Vh, Lh, H, F, &tmpPdf) * metalW;
        *pdf += tmpPdf * metalP;
    }

    // Glass/Specular BSDF
    if (glassP > 0.0)
    {
        // Dielectric fresnel (achromatic)
        let F = dielectricFresnel(VoH, rec.eta);

        if (reflect)
        {
            f += reflection(rec.mat, Vh, Lh, H, vec3(F), &tmpPdf) * glassW;
            *pdf += tmpPdf * glassP * F;
        }
        else
        {
            f += refraction(rec.mat, rec.eta, Vh, Lh, H, vec3(F), &tmpPdf) * glassW;
            *pdf += tmpPdf * glassP * (1. - F);
        }
    }

    // Clearcoat
    if (clearCoatP > 0. && reflect)
    {
        f += clearcoat(rec.mat, Vh, Lh, H, &tmpPdf) * .25 * rec.mat.clearcoat;
        *pdf += tmpPdf * clearCoatP;
    }

    return f * abs(Lh.z);
}

fn DisneySample(rec: ptr<function, hitRecord>, V: vec3f, L: ptr<function, vec3f>, pdf: ptr<function, f32>) -> vec3f
{
    if(rec.isVolume)
    {
        *L = ranSph();
        *pdf = .25 * INVPI;
        return rec.mat.baseColor * *pdf;
    }
    
    let aspect = sqrt(1. - rec.mat.anisotropic * .9);
    let ax = rec.mat.roughness / aspect;
    let ay = rec.mat.roughness * aspect;
    
    *pdf = 0.;

    var N = rec.n;
    var T: vec3f;
    var B: vec3f;
    onb(N, &T, &B);

    var Vh = toLocal(T, B, N, V);

    // Tint colors
    
    var Csheen: vec3f;
    var Cspec: vec3f;
    var F0: f32;
    tint(rec.mat, rec.eta, &F0, &Csheen, &Cspec);

    // Model weights
    
    let dielectricW = (1. - rec.mat.metallic) * (1. - rec.mat.specTrans);
    let metalW      = rec.mat.metallic;
    let glassW      = (1. - rec.mat.metallic) * rec.mat.specTrans;

    // Lobe probabilities
    
    let schlickW = schlickWeight(V.z);

    var diffP       = dielectricW * luma(rec.mat.baseColor);
    var dielectricP = dielectricW * mix(luma(Cspec), 1., schlickW);
    var metalP      = metalW * mix(luma(rec.mat.baseColor), 1., schlickW);
    var glassP      = glassW;
    var clearCoatP  = .25 * rec.mat.clearcoat;

    // Normalize probabilities
    
    let norm = 1. / (diffP + dielectricP + metalP + glassP + clearCoatP);
    
    diffP       *= norm;
    dielectricP *= norm;
    metalP      *= norm;
    glassP      *= norm;
    clearCoatP  *= norm;

    // CDF of the sampling probabilities
    var cdf: vec3f;
    cdf.x = diffP;
    cdf.y = cdf.x + dielectricP + metalP;
    cdf.z = cdf.y + glassP;

    // Sample a lobe based on its importance
    let r = rand();
    
    if (r < cdf.x) // Diffuse
    {
        *L = ranCos();
    }
    else if (r < cdf.y) // Dielectric + Metallic reflection
    {
        let H = sampleGGXVNDF(Vh, ax, ay);
        *L = reflect(-Vh, H);
    }
    else if (r < cdf.z) // Glass
    {
        let H = sampleGGXVNDF(Vh, ax, ay);
        
        let F = dielectricFresnel(abs(dot(Vh, H)), rec.eta);
        
        *L = select(refract(-Vh, H, rec.eta), reflect(-Vh, H), rand() < F);
    }
    else // Clearcoat
    {
        let H = sampleGTR1(rec.mat.clearcoatRoughness);

        *L = reflect(-Vh, H);
    }

    *L = toWorld(T, B, N, *L);
    Vh = toWorld(T, B, N, Vh);

    return DisneyBSDF(rec, Vh, N, *L, &*pdf);
}

fn iSphere(txx: mat4x4f, rad: f32, r: ray, tmin: f32, tmax: f32, rec: ptr<function, hitRecord>) -> bool
{
    var ro = (inverse(txx) * vec4(r.o, 1)).xyz;
	var rd = (inverse(txx) * vec4(r.d, 0)).xyz;
    
    var b  = dot(ro, rd);
    var qc = ro - b * rd;
    var d  = rad * rad - dot(qc, qc);

    if (d < 0.) { return false; }
    
    d = sqrt(d);
    
    var N = -b - d;
    var F = -b + d;
    
    var t = select(N, F, N < tmin);
    
    if(t < tmin || t > tmax) { return false; }
    
    (*rec).t = t;
    (*rec).p = r.o + r.d * t;
    (*rec).n = (txx * vec4(ro + rd * t, 0) / rad).xyz;
    
    return true;
}

fn iBox(txx: mat4x4f, rad: vec3f, r: ray, tmin: f32, tmax: f32, rec: ptr<function, hitRecord>) -> bool
{
    var ro = (inverse(txx) * vec4(r.o, 1)).xyz;
	var rd = (inverse(txx) * vec4(r.d, 0)).xyz;
         
    var k = rad * sign(rd);
                  
	var t1 = (-ro - k) / rd;
	var t2 = (-ro + k) / rd;
    
    var N = max(max(t1.x, t1.y), t1.z);
    var F = min(min(t2.x, t2.y), t2.z);
    
    if(N > F) { return false; }
    
    var res = select(vec4(F, -step(t2, vec3(F))),
                     vec4(N,  step(vec3(N), t1)), N > tmin);
                     
    if(res.x < tmin || res.x > tmax) { return false; }
    
    (*rec).t = res.x;
    (*rec).p = r.o + r.d * res.x;
    (*rec).n = (txx * vec4(-sign(rd) * res.yzw, 0)).xyz;
	
	return true;
}

fn worldHit(r: ray, tmin: f32, tmax: f32, rec: ptr<function, hitRecord>) -> bool
{
    rec.t = tmax;
    
    var hit = false;
    
    if(iBox(noTransform, vec3(1e3, .01, 1e3), r, tmin, rec.t, rec))
    {
        hit = true;
        rec.mat = DIFFUSE(vec3f(.5), 1.);
    }
    
    if (iSphere(translate(vec3(0, 1, 0)), 1., r, tmin, rec.t, rec))
    {
        hit = true;
        rec.mat = GLASS(vec3(1), 1.5, 0.);
    }
    
    if (iSphere(translate(vec3(-4, 1, 0)), 1., r, tmin, rec.t, rec))
    {
        hit = true;
        rec.mat = DIFFUSE(vec3(.4, .2, .1), 1.);
    }

    if (iSphere(translate(vec3(4, 1, 0)), 1., r, tmin, rec.t, rec))
    {
        hit = true;
        rec.mat = METAL(vec3(.7, .6, .5), 0.);
    }
    
    let ro = r.o.xz;
    let rd = r.d.xz;
    var p = floor(ro);
    var s = sign(rd);
    var m: vec2f;
    var d = (p - ro + .5 + s * .5) / rd;
    var cen: vec2f;
    
    var bhit = false;

    for(var i = 0; i < 40; i++)
    {
        for(var j = 0; j < 4; j++)
        {
            cen = p - vec2f(vec2i(j / 2, j % 2));
            
            if (abs(cen).x < 12. && abs(cen).y < 12.)
            {
                cen += .2 + .9 * hash22(cen);

                if (sqr(abs(abs(cen.x) - 2.) - 2.) + sqr(cen.y) > .8)
                {
                    if (iSphere(translate(vec3f(cen.x, .2, cen.y)), .2, r, tmin, rec.t, rec))
                    {
                        bhit = true; 
                        let ran = hash24(cen);

                        if(ran.w < .8)
                        {
                            if (ran.w < .3)
                            {
                                rec.mat = DIFFUSE(ran.xyz * ran.xyz * .9 + .1, 1.);
                            }
                            else if (ran.w < .5)
                            {
                                rec.mat = METAL(.5 * ran.xyz + .5, .5 * hash41(ran));
                            }
                            else
                            {
                                rec.mat = GLASS(.5 * ran.xyz + .5, 1.5, 0.);
                            }
                        }
                        else
                        {
                            rec.mat = LIGHT(7. * ran.xyz);
                        }
                    }
                }
            }
        }

        if(bhit)
        {
            hit = true;
            break;
        }
            
        m = step(d, d.yx);
        d += m / abs(rd);
        p += m * s;
    }
    
    return hit;
}

fn skyTexture(rd: vec3f) -> vec3f
{
    return textureSampleLevel(channel0, bilinear, INVPI * vec2f(.5 * atan2(rd.z, rd.x), -asin(rd.y)) + .5, 0.).rgb;
}

fn color(r: ray) -> vec3f
{
    var col = vec3f(1);
    var emitted = vec3f(0);
    var pdf: f32;
	var rec: hitRecord;

    var Ray = r;
    
    for (var i = 0; i < 10; i++)
    {
        if (worldHit(Ray, .01, 1e5, &rec))
        {
            emitted += col * rec.mat.emission;
            
            if(luma(col) < 1e-6)
            {
                break;
            }

            var scattered: ray;
            scattered.o = rec.p;
            
            rec.eta = rec.mat.IOR;
            if(dot(rec.n, r.d) > 0.)
            {
                rec.n = -rec.n;
            } 
            else
            {
                rec.eta = 1. / rec.eta;
            }
            let BSDF = DisneySample(&rec, -r.d, &scattered.d, &pdf);
            if(pdf > 0.)
            {
                col *= BSDF / pdf;
            }
            else
            {
                break;
            }

            Ray = scattered;
        }
        
        else
        {
            emitted += col * skyTexture(Ray.d);
            break;
    	}
    }
    
    return emitted;
}

fn camera(ro: vec3f, lp: vec3f, vup: vec3f, vfov: f32, aperture: f32, d: f32) -> cam
{
    let R = vec2f(textureDimensions(screen));
    
    var c: cam;
    
    let hh = tan(radians(vfov) / 2.) * d;
    let hw = R.x / R.y * hh;
          
    c.rad = aperture / 2.;
    c.o = ro;
    c.w = normalize(ro - lp);
    c.u = normalize(cross(vup, c.w));
    c.v = cross(c.w, c.u);
    c.llc = c.o - hw * c.u - hh * c.v - d * c.w;
    c.hor = 2. * hw * c.u;
    c.ver = 2. * hh * c.v;
    
    return c;
}

fn getRay(c: cam, uv: vec2f) -> ray
{
    var rd = c.rad * ranDisk();

    var ro = c.o + c.u * rd.x + c.v * rd.y;
    
    return ray(ro, normalize(c.llc + uv.x * c.hor + uv.y * c.ver - ro));
}

#storage image array<array<vec4f, SCREEN_WIDTH>, SCREEN_HEIGHT>

fn tonemap(v: vec3f) -> vec3f
{	
	var c = v;
    
    c *= mat3x3(.59719, .35458, .04823,
                .076  , .90834, .01566,
                .0284 , .13383, .83777);
    
    c = (c * (c + .0245786) - .000090537) / (c * (.983729 * c + .432951) + .238081);
	
    return c * mat3x3(1.60475, -.53108, -.07367,
                      -.10208, 1.10813, -.00605,
                      -.00327, -.07276, 1.07602);	
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u)
{
    let screen_size = textureDimensions(screen);

    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    seed = vec4u(id, time.frame);

    let uv = (vec2f(f32(id.x), f32(screen_size.y - id.y)) + rand2()) / vec2f(screen_size);

    let c = camera(vec3(13, 2, 3), vec3(0), vec3(0, 1, 0), 20., .5, 10.);

    let col = color(getRay(c, uv));

    var sample = select(image[id.y][id.x], vec4f(0), mouse.click > 0);

    sample.w += 1;
    sample = vec4f(mix(sample.rgb, col, 1. / sample.w), sample.w);

    image[id.y][id.x] = sample;

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(tonemap(sample.xyz), sample.w));
}
