import std;

typealias vec2 = float2;
typealias vec3 = float3;
typealias vec4 = float4;
typealias mat3 = float3x3;

// ==== SDF3D ===

interface SDF3D {
    float sdf3d(vec3 p);
}

struct SDF3DSphere : SDF3D {
    float radius;

    __init(float r){
        radius = r;
    }

    float sdf3d(vec3 p){
        return length(p) - radius;
    }
}

struct SDF3DBox : SDF3D{
    vec3 size;

    __init(vec3 s){
        size = s;
    }

    float sdf3d(vec3 p){
        vec3 q = abs(p) - size;
        return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
    }
}

struct SDF3DTorus : SDF3D {
    vec2 radii;

    __init(vec2 r){
        radii = r;
    }

    float sdf3d(vec3 p){
        vec2 q = vec2(length(p.xz) - radii.x, p.y);
        return length(q) - radii.y;
    }
}

struct SDF3DCylinder : SDF3D {
    float radius;
    float height;

    __init(float r, float h){
        radius = r;
        height = h;
    }

    float sdf3d(vec3 p){
        vec2 d = abs(vec2(length(p.xz), p.y)) - vec2(radius, height);
        return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
    }
}

struct SDF3DUnion<S1: SDF3D, S2: SDF3D> : SDF3D {
    S1 sdfa;
    S2 sdfb;
    float k;

    __init(S1 a, S2 b, float blend) {
        sdfa = a;
        sdfb = b;
        k = blend;
    }

    float sdf3d(vec3 p) {
        float a = sdfa.sdf3d(p);
        float b = sdfb.sdf3d(p);
        if(k <= 0.0) return min(a, b);
        float h = clamp(0.5 + 0.5*(b-a)/k, 0.0, 1.0);
        return lerp(b, a, h) - k*h*(1.0-h);
    }
}

struct SDF3DSubtract<S1: SDF3D, S2: SDF3D> : SDF3D {
    S1 sdfa;
    S2 sdfb;
    float k;

    __init(S1 a, S2 b, float blend) {
        sdfa = a;
        sdfb = b;
        k = blend;
    }

    float sdf3d(vec3 p) {
        float a = sdfa.sdf3d(p);
        float b = sdfb.sdf3d(p);
        if(k <= 0.0) return max(a, -b);
        float h = clamp(0.5 - 0.5*(a+b)/k, 0.0, 1.0);
        return lerp(a, -b, h) + k*h*(1.0-h);
    }
}

struct SDF3DIntersect<S1: SDF3D, S2: SDF3D> : SDF3D {
    S1 sdfa;
    S2 sdfb;
    float k;

    __init(S1 a, S2 b, float blend) {
        sdfa = a;
        sdfb = b;
        k = blend;
    }

    float sdf3d(vec3 p) {
        float a = sdfa.sdf3d(p);
        float b = sdfb.sdf3d(p);
        if(k <= 0.0) return max(a, b);
        float h = clamp(0.5 - 0.5*(b-a)/k, 0.0, 1.0);
        return lerp(b, a, h) + k*h*(1.0-h);
    }
}

struct SDF3DTranslate<S: SDF3D> : SDF3D {
    S sdf;
    vec3 offset;

    __init(S shape, vec3 translation) {
        sdf = shape;
        offset = translation;
    }

    float sdf3d(vec3 p) {
        return sdf.sdf3d(p - offset);
    }
}

struct SDF3DRotate<S: SDF3D> : SDF3D {
    S sdf;
    mat3 rotation;

    __init(S shape, mat3 rot) {
        sdf = shape;
        rotation = rot;
    }

    float sdf3d(vec3 p) {
        return sdf.sdf3d(mul(transpose(rotation), p));
    }
}

struct SDF3DScale<S: SDF3D> : SDF3D {
    S sdf;
    float scale;

    __init(S shape, float s) {
        sdf = shape;
        scale = s;
    }

    float sdf3d(vec3 p) {
        return sdf.sdf3d(p / scale) * scale;
    }
}

namespace sdf3d {

    func sphere(float radius) -> SDF3DSphere {
        return SDF3DSphere(radius);
    }

    func box(vec3 size) -> SDF3DBox {
        return SDF3DBox(size);
    }

    func torus(vec2 radii) -> SDF3DTorus {
        return SDF3DTorus(radii);
    }

    func cylinder(float radius, float height) -> SDF3DCylinder {
        return SDF3DCylinder(radius, height);
    }

    func union<S1: SDF3D, S2: SDF3D>(S1 sdf1, S2 sdf2, float k = 0.0) -> SDF3DUnion<S1, S2> {
        return SDF3DUnion<S1, S2>(sdf1, sdf2, k);
    }

    func subtract<S1: SDF3D, S2: SDF3D>(S1 sdf1, S2 sdf2, float k = 0.0) -> SDF3DSubtract<S1, S2> {
        return SDF3DSubtract<S1, S2>(sdf1, sdf2, k);
    }

    func intersect<S1: SDF3D, S2: SDF3D>(S1 sdf1, S2 sdf2, float k = 0.0) -> SDF3DIntersect<S1, S2> {
        return SDF3DIntersect<S1, S2>(sdf1, sdf2, k);
    }

    func translate<S: SDF3D>(S shape, vec3 offset) -> SDF3DTranslate<S> {
        return SDF3DTranslate<S>(shape, offset);
    }

    func rotate<S: SDF3D>(S shape, mat3 rotation) -> SDF3DRotate<S> {
        return SDF3DRotate<S>(shape, rotation);
    }

    func scale<S: SDF3D>(S shape, float scale) -> SDF3DScale<S> {
        return SDF3DScale<S>(shape, scale);
    }

}

// ==== Raymarcher Parameters ===

struct CameraParams {
    vec3 position;
    vec3 target;
    vec3 up;
    float fov;
}

struct RaymarchParams {
    int maxSteps;
    float epsilon;
    float maxDistance;
    float stepScale;
}

struct LightingParams {
    vec3 lightDir;
    vec3 lightColor;
    vec3 skyColorTop;
    vec3 skyColorBottom;
    float lightIntensity;
    float ambientIntensity;
}

struct ShadowParams {
    int maxSteps;
    float softness;
    float minDist;
    float maxDist;
}

struct AOParams {
    int samples;
    float stepSize;
    float intensity;
    float falloff;
}

struct MaterialParams {
    vec3 albedo;
    float roughness;
    float metallic;
}

struct FogParams {
    float density;
    vec3 color;
}

struct RenderParams {
    CameraParams camera;
    RaymarchParams raymarch;
    LightingParams lighting;
    ShadowParams shadow;
    AOParams ao;
    MaterialParams material;
    FogParams fog;
    float normalEpsilon;
    float gamma;
}

// ==== Raymarcher ===

interface Raymarcher {
    vec3 render(vec2 uv, RenderParams params);
}

struct RaymarchSDF3D<T: SDF3D> : Raymarcher {
    T scene;
    
    vec3 calcNormal(vec3 p, float epsilon) {
        vec2 e = vec2(epsilon, 0.0);
        return normalize(vec3(
            scene.sdf3d(p + e.xyy) - scene.sdf3d(p - e.xyy),
            scene.sdf3d(p + e.yxy) - scene.sdf3d(p - e.yxy),
            scene.sdf3d(p + e.yyx) - scene.sdf3d(p - e.yyx)
        ));
    }
    
    float calcAO(vec3 pos, vec3 nor, AOParams params) {
        float occ = 0.0;
        float sca = 1.0;
        for(int i = 0; i < params.samples; i++) {
            float h = params.stepSize + params.stepSize * float(i) / float(params.samples - 1);
            float d = scene.sdf3d(pos + h * nor);
            occ += (h - d) * sca;
            sca *= params.falloff;
        }
        return clamp(1.0 - params.intensity * occ, 0.0, 1.0);
    }
    
    float softShadow(vec3 ro, vec3 rd, ShadowParams params) {
        float hit = 1.0;
        float t = params.minDist;
        
        for(int i = 0; i < params.maxSteps; i++) {
            float h = scene.sdf3d(ro + rd * t);
            if(h < 0.001) return 0.0;
            t += h;
            hit = min(hit, params.softness * h / t);
            if(t >= params.maxDist) break;
        }
        
        return clamp(hit, 0.0, 1.0);
    }
    
    // PBR Helper functions
    float saturate(float x) { return clamp(x, 0.0, 1.0); }
    vec3 saturate(vec3 x) { return clamp(x, 0.0, 1.0); }
    
    float pow5(float x) {
        float x2 = x * x;
        return x2 * x2 * x;
    }
    
    float D_GGX(float linearRoughness, float NoH, vec3 h) {
        float oneMinusNoHSquared = 1.0 - NoH * NoH;
        float a = NoH * linearRoughness;
        float k = linearRoughness / (oneMinusNoHSquared + a * a);
        float d = k * k * (1.0 / 3.14159265359);
        return d;
    }
    
    float V_SmithGGXCorrelated(float linearRoughness, float NoV, float NoL) {
        float a2 = linearRoughness * linearRoughness;
        float GGXV = NoL * sqrt((NoV - a2 * NoV) * NoV + a2);
        float GGXL = NoV * sqrt((NoL - a2 * NoL) * NoL + a2);
        return 0.5 / (GGXV + GGXL);
    }
    
    vec3 F_Schlick(vec3 f0, float VoH) {
        return f0 + (vec3(1.0) - f0) * pow5(1.0 - VoH);
    }
    
    float F_Schlick(float f0, float f90, float VoH) {
        return f0 + (f90 - f0) * pow5(1.0 - VoH);
    }
    
    float Fd_Burley(float linearRoughness, float NoV, float NoL, float LoH) {
        // Burley 2012, "Physically-Based Shading at Disney"
        float f90 = 0.5 + 2.0 * linearRoughness * LoH * LoH;
        float lightScatter = F_Schlick(1.0, f90, NoL);
        float viewScatter = F_Schlick(1.0, f90, NoV);
        return lightScatter * viewScatter * (1.0 / 3.14159265359);
    }
    
    float Fd_Lambert() {
        return 1.0 / 3.14159265359;
    }
    
    // Indirect lighting helpers
    vec3 Irradiance_SphericalHarmonics(vec3 n) {
        return max(
            vec3(0.754554516862612, 0.748542953903366, 0.790921515418539)
            + vec3(-0.083856548007422, 0.092533500963210, 0.322764661032516) * (n.y)
            + vec3(0.308152705331738, 0.366796330467391, 0.466698181299906) * (n.z)
            + vec3(-0.188884931542396, -0.277402551592231, -0.377844212327557) * (n.x)
            , 0.0);
    }
    
    vec2 PrefilteredDFG_Karis(float roughness, float NoV) {
        const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
        const vec4 c1 = vec4(1.0, 0.0425, 1.040, -0.040);
        
        vec4 r = roughness * c0 + c1;
        float a004 = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;
        
        return vec2(-1.04, 1.04) * a004 + r.zw;
    }
    
    vec3 Tonemap_ACES(vec3 x) {
        const float a = 2.51;
        const float b = 0.03;
        const float c = 2.43;
        const float d = 0.59;
        const float e = 0.14;
        return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
    }
    
    vec3 OECF_sRGBFast(vec3 linear, float gamma) {
        return pow(linear, vec3(1.0 / gamma));
    }
    
    vec3 render(vec2 uv, RenderParams params) {
        // Camera setup
        vec3 ro = params.camera.position;
        vec3 ta = params.camera.target;
        vec3 ww = normalize(ta - ro);
        vec3 uu = normalize(cross(ww, params.camera.up));
        vec3 vv = normalize(cross(ww,uu));
        
        float fovScale = tan(params.camera.fov * 0.5 * 3.14159265 / 180.0);
        vec3 rd = normalize(uv.x * uu * fovScale + uv.y * vv * fovScale + ww);
        
        // Sky gradient
        float skyMix = rd.y * 0.5 + 0.5;
        vec3 skyColor = lerp(params.lighting.skyColorBottom, params.lighting.skyColorTop, skyMix);
        
        // Raymarching
        float t = 0.0;
        vec3 color = skyColor;
        
        for(int i = 0; i < params.raymarch.maxSteps; i++) {
            vec3 p = ro + rd * t;
            float d = scene.sdf3d(p);
            if(d < params.raymarch.epsilon || t > params.raymarch.maxDistance) break;
            t += d * params.raymarch.stepScale;
        }
        
        // Shading
        if(t < params.raymarch.maxDistance) {
            vec3 position = ro + rd * t;
            
            vec3 v = normalize(-rd);
            vec3 n = calcNormal(position, params.normalEpsilon);
            vec3 l = normalize(params.lighting.lightDir);
            vec3 h = normalize(v + l);
            vec3 r = normalize(reflect(rd, n));
            
            float NoV = abs(dot(n, v)) + 1e-5;
            float NoL = saturate(dot(n, l));
            float NoH = saturate(dot(n, h));
            float LoH = saturate(dot(l, h));
            float VoH = saturate(dot(v, h));
            
            // Material properties
            vec3 baseColor = params.material.albedo;
            float roughness = params.material.roughness;
            float metallic = params.material.metallic;
            
            float linearRoughness = roughness * roughness;
            vec3 diffuseColor = (1.0 - metallic) * baseColor;
            vec3 f0 = vec3(0.04) * (1.0 - metallic) + baseColor * metallic;
            
            float shadow = softShadow(position, l, params.shadow);
            
            // Specular BRDF
            float D = D_GGX(linearRoughness, NoH, h);
            float V = V_SmithGGXCorrelated(linearRoughness, NoV, NoL);
            vec3 F = F_Schlick(f0, VoH);
            vec3 Fr = (D * V) * F;
            
            // Diffuse BRDF
            vec3 Fd = diffuseColor * Fd_Burley(linearRoughness, NoV, NoL, LoH);
            
            // Direct lighting
            color = Fd + Fr;
            color *= (params.lighting.lightIntensity * shadow * NoL) * params.lighting.lightColor;
            
            // Indirect diffuse
            vec3 indirectDiffuse = Irradiance_SphericalHarmonics(n) * Fd_Lambert();
            
            // Indirect specular (simple reflection)
            vec3 indirectSpecular = skyColor;
            
            // Trace reflection
            float tRefl = 0.0;
            for(int i = 0; i < 40; i++) {
                vec3 p = position + r * tRefl;
                float d = scene.sdf3d(p);
                if(d < params.raymarch.epsilon || tRefl > 10.0) break;
                tRefl += d * 0.9;
            }
            
            if(tRefl < 10.0) {
                // Hit something with reflection
                vec3 reflPos = position + r * tRefl;
                vec3 reflNorm = calcNormal(reflPos, params.normalEpsilon);
                float reflNoL = saturate(dot(reflNorm, l));
                vec3 reflColor = params.material.albedo * 0.8; // Simplified material
                indirectSpecular = reflColor * reflNoL * params.lighting.lightColor;
            }
            
            // Indirect contribution
            vec2 dfg = PrefilteredDFG_Karis(roughness, NoV);
            vec3 specularColor = f0 * dfg.x + dfg.y;
            vec3 ibl = diffuseColor * indirectDiffuse + indirectSpecular * specularColor;
            
            float ao = calcAO(position, n, params.ao);
            color += ibl * params.lighting.ambientIntensity * ao;
            
            // Fog
            color = lerp(color, params.fog.color, 1.0 - exp(-params.fog.density * t * t));
        }
        
        // Tone mapping
        color = Tonemap_ACES(color);
        
        // Gamma correction (sRGB)
        color = OECF_sRGBFast(color,params.gamma);
        
        return color;
    }
}

namespace raymarcher {
    func raymarchsdf3d<S: SDF3D>(S sdf) -> RaymarchSDF3D<S> {
        return RaymarchSDF3D<S>(sdf);
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main_image(uint3 id : SV_DispatchThreadID)
{
    uint width, height;
    screen.GetDimensions(width, height);
    if (id.x >= width || id.y >= height) return;
    
    vec2 uv = (vec2(id.xy) - 0.5 * vec2(width, height)) / float(height);
    
    float t = time.elapsed;
    
    // Create rotation matrices
    mat3 rotY = mat3(cos(t), 0.0, sin(t),
                     0.0, 1.0, 0.0,
                     -sin(t), 0.0, cos(t));
    
    mat3 rotX = mat3(1.0, 0.0, 0.0,
                     0.0, cos(t * 0.7), -sin(t * 0.7),
                     0.0, sin(t * 0.7), cos(t * 0.7));
    
    // Setup all render parameters
    RenderParams params;
    
    // Camera
    params.camera.position = vec3(3.0 * sin(t * 0.3), 1.5, 3.0 * cos(t * 0.3));
    params.camera.target = vec3(0.0, 0.0, 0.0);
    params.camera.up = vec3(0.0, 1.0, 0.0);
    params.camera.fov = 60.0;
    
    // Raymarching
    params.raymarch.maxSteps = 512;
    params.raymarch.epsilon = 0.002;
    params.raymarch.maxDistance = 20.0;
    params.raymarch.stepScale = 1.0;
    
    // Lighting
    params.lighting.lightDir = vec3(0.6, 0.7, -0.7);
    params.lighting.lightColor = vec3(0.98, 0.92, 0.89);
    params.lighting.skyColorTop = vec3(0.65, 0.85, 1.0);
    params.lighting.skyColorBottom = vec3(0.65, 0.85, 1.0) + vec3(0.0, 0.72, 0.72);
    params.lighting.lightIntensity = 2.0;
    params.lighting.ambientIntensity = 0.65;
    
    // Shadows
    params.shadow.maxSteps = 100;
    params.shadow.softness = 10.0;
    params.shadow.minDist = 0.02;
    params.shadow.maxDist = 2.5;
    
    // Ambient Occlusion
    params.ao.samples = 8;
    params.ao.stepSize = 0.02;
    params.ao.intensity = 2.0;
    params.ao.falloff = 0.85;
    
    // Material
    params.material.albedo = vec3(0.9, 0.6, 0.3);
    params.material.roughness = 0.2;
    params.material.metallic = 0.0;
    
    // Fog
    params.fog.density = 0.001;
    params.fog.color = 0.8 * vec3(0.7, 0.8, 1.0);
    
    // Other
    params.normalEpsilon = 0.0005;
    params.gamma = 2.2;
    
    // Build scene and render
    vec3 pixelcolor = raymarcher::raymarchsdf3d(
        sdf3d::union(
            // Central morphing structure
            sdf3d::intersect(
                sdf3d::rotate(
                    sdf3d::box(vec3(0.6, 0.6, 0.6)),
                    rotY
                ),
                sdf3d::sphere(0.8),
                0.05
            ),
            
            // Orbiting torus
            sdf3d::union(
                sdf3d::translate(
                    sdf3d::rotate(
                        sdf3d::torus(vec2(0.3, 0.1)),
                        rotX
                    ),
                    vec3(sin(t * 1.5), 0.0, cos(t * 1.5))
                ),
                
                // Smaller details
                sdf3d::union(
                    sdf3d::translate(
                        sdf3d::sphere(0.2),
                        vec3(0.0, sin(t * 2.0) * 0.8, 0.0)
                    ),
                    
                    // Cylinder with holes
                    sdf3d::translate(
                        sdf3d::subtract(
                            sdf3d::cylinder(0.3, 0.4),
                            sdf3d::union(
                                sdf3d::translate(sdf3d::box(vec3(0.1, 0.5, 0.1)), vec3(0.2, 0.0, 0.0)),
                                sdf3d::translate(sdf3d::box(vec3(0.1, 0.5, 0.1)), vec3(-0.2, 0.0, 0.0)),
                                0.05
                            ),
                            0.05
                        ),
                        vec3(-1.0, -0.5, 0.0)
                    ),
                    0.1
                ),
                0.15
            ),
            0.2
        )
    ).render(uv, params);

    screen[id.xy] = vec4(pixelcolor, 1.0);
}