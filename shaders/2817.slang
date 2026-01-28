import std;
import shadertoy;

// Shader Rally - Ported to compute.toys (Slang)
// Original by @P_Malin
// 
// Controls: WASD to drive, Space = brake, G = toggle gravity, V = toggle wheels
// Mouse controls camera angle and distance

// This implements the full 4-pass system:
// Buffer A: Physics simulation
// Buffer B: Tyre tracks
// Buffer C: Main rendering
// Main Image: Post processing (motion blur, depth of field)

#define FAST_VERSION
#define RAYTRACE_COUNT 2
#define MOTION_BLUR_TAPS 8

#define MAX_PIXEL_COUNT (1920*1080)
[StorageBuffer(MAX_PIXEL_COUNT)]
RWStructuredBuffer<vec4> ChBufferA;
[StorageBuffer(MAX_PIXEL_COUNT)]
RWStructuredBuffer<vec4> ChBufferB;
[StorageBuffer(MAX_PIXEL_COUNT)]
RWStructuredBuffer<vec4> ChBufferC;
[StorageBuffer(MAX_PIXEL_COUNT)]
RWStructuredBuffer<vec4> ChBufferD;

// External texture channels
// In compute.toys, these may need to be declared as separate resources
// Try uncommenting these if compute.toys supports them:
// [StorageBuffer(MAX_PIXEL_COUNT)]
// RWStructuredBuffer<vec4> channel0;  // Terrain texture
// [StorageBuffer(MAX_PIXEL_COUNT)]
// RWStructuredBuffer<vec4> channel1;  // Wood texture

uint PixelID(ivec2 id) { return id.x + id.y * SCREEN_SIZE.x; }

public vec4 texelFetch(RWStructuredBuffer<vec4> channel, ivec2 coord, int lod) {
    return channel[PixelID(coord)];
}

public vec4 textureLod(RWStructuredBuffer<vec4> channel, vec2 coord, float lod) {
    coord = coord*iResolution.xy - 0.5;
    ivec2 icoord = ivec2(floor(coord));
    vec2 fcoord = fract(coord);
    vec4 v00 = texelFetch(channel, icoord + ivec2(0,0), 0);
    vec4 v01 = texelFetch(channel, icoord + ivec2(0,1), 0);
    vec4 v10 = texelFetch(channel, icoord + ivec2(1,0), 0);
    vec4 v11 = texelFetch(channel, icoord + ivec2(1,1), 0);
    return mix(mix(v00, v01, fcoord.y), mix(v10, v11, fcoord.y), fcoord.x); 
}

public vec4 texture(RWStructuredBuffer<vec4> channel, vec2 coord) {
    return textureLod(channel, coord, 0.0);
}

// Texture channels - channel0 and channel1 are declared in std.slang as Texture2D<float4>
// They are accessed using the texture() or textureLod() functions from shadertoy.slang
// channel0: Terrain texture
// channel1: Wood texture
// 
// Note: channel0 and channel1 are automatically available from std.slang import
// Use: texture(channel0, coord) or textureLod(channel0, coord, lod)

// Address constants
static const ivec2 addrVehicle = ivec2(0, 0);
static const ivec2 offsetVehicleBody = ivec2(1, 0);
static const ivec2 offsetBodyPos = ivec2(0, 0);
static const ivec2 offsetBodyRot = ivec2(1, 0);
static const ivec2 offsetBodyMom = ivec2(2, 0);
static const ivec2 offsetBodyAngMom = ivec2(3, 0);
static const ivec2 offsetVehicleWheel0 = ivec2(5, 0);
static const ivec2 offsetVehicleWheel1 = ivec2(7, 0);
static const ivec2 offsetVehicleWheel2 = ivec2(9, 0);
static const ivec2 offsetVehicleWheel3 = ivec2(11, 0);
static const ivec2 offsetWheelState = ivec2(0, 0);
static const ivec2 offsetWheelContactState = ivec2(1, 0);
static const ivec2 addrCamera = ivec2(0, 1);
static const ivec2 offsetCameraPos = ivec2(0, 0);
static const ivec2 offsetCameraTarget = ivec2(1, 0);
static const ivec2 addrPrevCamera = ivec2(0, 2);

// Material IDs
static const int MAT_TERRAIN = 1;
static const int MAT_CAR_BODY = 3;
static const int MAT_CHROME = 4;
static const int MAT_WHEEL = 6;
static const int MAT_SUSPENSION = 7;
static const int MAT_WOOD = 2;

// Constants
static const float PI = 3.14159265359;
static const float kFarClip = 1000.0;
static const vec2 MOD2 = vec2(4.438975, 3.972973);
static const vec3 MOD3 = vec3(0.1031, 0.11369, 0.13787);
static const float HASHSCALE = 0.1031;

// Keyboard constants
static const int KEY_SPACE = 32;
static const int KEY_A = 65;
static const int KEY_D = 68;
static const int KEY_S = 83;
static const int KEY_W = 87;
static const int KEY_G = 71;
static const int KEY_V = 86;
static const int KEY_COMMA = 188;
static const int KEY_PER = 190;

// Helper functions for state storage
vec4 LoadVec4(ivec2 vAddr) {
    return texelFetch(ChBufferA, vAddr, 0);
}

vec3 LoadVec3(ivec2 vAddr) {
    return LoadVec4(vAddr).xyz;
}

bool AtAddress(vec2 p, ivec2 c) {
    return all(equal(floor(p), vec2(c)));
}

void StoreVec4(ivec2 vAddr, vec4 vValue, inout vec4 fragColor, vec2 fragCoord) {
    fragColor = AtAddress(fragCoord, vAddr) ? vValue : fragColor;
}

void StoreVec3(ivec2 vAddr, vec3 vValue, inout vec4 fragColor, vec2 fragCoord) {
    StoreVec4(vAddr, vec4(vValue, 0.0), fragColor, fragCoord);
}

bool KeyIsPressed(int key) {
    return keyDown(key);
}

bool KeyIsToggled(int key) {
    // Simplified - would need state tracking for proper toggle
    return keyDown(key);
}

// Hash functions
float Hash(float p) {
    vec2 p2 = fract(vec2(p) * MOD2);
    p2 += dot(p2.yx, p2.xy + 19.19);
    return fract(p2.x * p2.y);
}

vec3 Hash31(float p) {
    vec3 p3 = fract(vec3(p) * MOD3);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract(vec3((p3.x + p3.y) * p3.z, (p3.x + p3.z) * p3.y, (p3.y + p3.z) * p3.x));
}

vec3 Hash32(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * HASHSCALE);
    p3 += dot(p3, p3.yxz + 19.19);
    return fract(vec3((p3.x + p3.y) * p3.z, (p3.x + p3.z) * p3.y, (p3.y + p3.z) * p3.x));
}

float SmoothNoise(vec2 o) {
    vec2 p = floor(o);
    vec2 f = fract(o);
    float n = p.x + p.y * 57.0;
    
    float a = Hash(n + 0.0);
    float b = Hash(n + 1.0);
    float c = Hash(n + 57.0);
    float d = Hash(n + 58.0);
    
    vec2 f2 = f * f;
    vec2 f3 = f2 * f;
    vec2 t = 3.0 * f2 - 2.0 * f3;
    
    float u = t.x;
    float v = t.y;
    
    return a + (b - a) * u + (c - a) * v + (a - b + d - c) * u * v;
}

float FBM(vec2 p, float ps) {
    float f = 0.0;
    float tot = 0.0;
    float a = 1.0;
    
    for (int i = 0; i < 3; i++) {
        f += SmoothNoise(p) * a;
        p *= 2.0;
        tot += a;
        a *= ps;
    }
    return f / tot;
}

// Rotation functions
vec2 Rotate(vec2 vPos, float t) {
    float s = sin(t);
    float c = cos(t);
    return vec2(c * vPos.x + s * vPos.y, -s * vPos.x + c * vPos.y);
}

vec2 Rotate(vec2 vPos, vec2 sc) {
    return vec2(sc.y * vPos.x + sc.x * vPos.y, -sc.x * vPos.x + sc.y * vPos.y);
}

vec3 RotY(vec3 vPos, float t) {
    vec3 result;
    result.y = vPos.y;
    result.xz = Rotate(vPos.xz, t);
    return result;
}

vec3 RotX(vec3 vPos, float t) {
    vec3 result;
    result.x = vPos.x;
    result.yz = Rotate(vPos.yz, t);
    return result;
}

vec3 RotY(vec3 vPos, vec2 sc) {
    vec3 result;
    result.y = vPos.y;
    result.xz = Rotate(vPos.xz, sc);
    return result;
}

vec3 RotX(vec3 vPos, vec2 sc) {
    vec3 result;
    result.x = vPos.x;
    result.yz = Rotate(vPos.yz, sc);
    return result;
}

// Quaternion functions
mat3 QuatToMat3(vec4 q) {
    vec4 qSq = q * q;
    float xy2 = q.x * q.y * 2.0;
    float xz2 = q.x * q.z * 2.0;
    float yz2 = q.y * q.z * 2.0;
    float wx2 = q.w * q.x * 2.0;
    float wy2 = q.w * q.y * 2.0;
    float wz2 = q.w * q.z * 2.0;
    
    return mat3(
        vec3(qSq.w + qSq.x - qSq.y - qSq.z, xy2 - wz2, xz2 + wy2),
        vec3(xy2 + wz2, qSq.w - qSq.x + qSq.y - qSq.z, yz2 - wx2),
        vec3(xz2 - wy2, yz2 + wx2, qSq.w - qSq.x - qSq.y + qSq.z)
    );
}

vec4 QuatMul(vec4 lhs, vec4 rhs) {
    return vec4(
        lhs.y * rhs.z - lhs.z * rhs.y + lhs.x * rhs.w + lhs.w * rhs.x,
        lhs.z * rhs.x - lhs.x * rhs.z + lhs.y * rhs.w + lhs.w * rhs.y,
        lhs.x * rhs.y - lhs.y * rhs.x + lhs.z * rhs.w + lhs.w * rhs.z,
        lhs.w * rhs.w - lhs.x * rhs.x - lhs.y * rhs.y - lhs.z * rhs.z
    );
}

vec4 QuatFromAxisAngle(vec3 vAxis, float fAngle) {
    return vec4(normalize(vAxis) * sin(fAngle), cos(fAngle));
}

vec4 QuatFromVec3(vec3 vRot) {
    float l = length(vRot);
    if (l <= 0.0) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    return QuatFromAxisAngle(vRot, l);
}

// Vector operations
vec3 Vec3Parallel(vec3 x, vec3 n) {
    float d = dot(x, n);
    return x - n * d;
}

vec3 Vec3Perp(vec3 x, vec3 n) {
    return x - Vec3Parallel(x, n);
}

vec3 ObjToWorld(vec3 v, mat3 m) {
    return v * m;
}

vec3 WorldToObj(vec3 v, mat3 m) {
    return m * v;
}

// Distance functions
float sdBox(vec3 p, vec3 b) {
    vec3 d = abs(p) - b;
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, vec3(0.0)));
}

float SdSphere(vec3 p, float r) {
    return length(p) - r;
}

float smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

float UdRoundBox(vec3 p, vec3 b, float r) {
    return length(max(abs(p) - b, 0.0)) - r;
}

// Terrain
float GetTerrainDistance(vec3 vPos) {
    float fbm = FBM(vPos.xz * vec2(0.5, 1.0), 0.5);
    float fTerrainHeight = fbm * fbm;
    fTerrainHeight = fTerrainHeight * (sin(vPos.x * 0.1) + 1.0) * 0.5 + vPos.y + 3.0;
    return fTerrainHeight;
}

// Scene structures for physics (Buffer A)
struct SurfaceInfo {
    vec3 vUVW;
    int iId;
};

struct ClosestSurface {
    float fDist;
    SurfaceInfo surface;
};

struct C_Intersection {
    vec3 vPos;
    float fDist;
    vec3 vNormal;
    SurfaceInfo surface;
};

ClosestSurface GetEnvironmentClosestSurface(vec3 vPos) {
    ClosestSurface terrainClosest;
    terrainClosest.surface.iId = MAT_TERRAIN;
    terrainClosest.surface.vUVW = vec3(vPos.xz, 0.0);
    terrainClosest.fDist = GetTerrainDistance(vPos);
    return terrainClosest;
}

ClosestSurface GetSceneClosestSurface(vec3 vPos) {
    return GetEnvironmentClosestSurface(vPos);
}

vec3 GetSceneNormal(vec3 vPos) {
    const float fDelta = 0.0001;
    vec3 vDir1 = vec3(1.0, -1.0, -1.0);
    vec3 vDir2 = vec3(-1.0, -1.0, 1.0);
    vec3 vDir3 = vec3(-1.0, 1.0, -1.0);
    vec3 vDir4 = vec3(1.0, 1.0, 1.0);
    
    ClosestSurface c1 = GetSceneClosestSurface(vPos + vDir1 * fDelta);
    ClosestSurface c2 = GetSceneClosestSurface(vPos + vDir2 * fDelta);
    ClosestSurface c3 = GetSceneClosestSurface(vPos + vDir3 * fDelta);
    ClosestSurface c4 = GetSceneClosestSurface(vPos + vDir4 * fDelta);
    
    vec3 vNormal = vDir1 * c1.fDist + vDir2 * c2.fDist + vDir3 * c3.fDist + vDir4 * c4.fDist;
    return normalize(vNormal);
}

void TraceScene(out C_Intersection outIntersection, vec3 vOrigin, vec3 vDir) {
    float t = 0.1;
    const int kRaymarchMaxIter = 32;
    const float kFarClipPhysics = 10.0;
    
    for (int i = 0; i < kRaymarchMaxIter; i++) {
        float fClosestDist = GetSceneClosestSurface(vOrigin + vDir * t).fDist;
        t += fClosestDist;
        if (abs(fClosestDist) < 0.01) {
            break;
        }
        if (t > kFarClipPhysics) {
            t = kFarClipPhysics;
            break;
        }
    }
    
    outIntersection.fDist = t;
    outIntersection.vPos = vOrigin + vDir * t;
    
    if (t >= kFarClipPhysics) {
        outIntersection.surface.iId = 0;
        outIntersection.surface.vUVW = vec3(0.0);
        outIntersection.vNormal = vec3(0.0, 1.0, 0.0);
    } else {
        outIntersection.vNormal = GetSceneNormal(outIntersection.vPos);
        outIntersection.surface = GetSceneClosestSurface(outIntersection.vPos).surface;
    }
}

// Physics structures
struct Body {
    vec3 vPos;
    vec4 qRot;
    vec3 vMomentum;
    vec3 vAngularMomentum;
    mat3 mRot;
    float fMass;
    float fIT;
    vec3 vForce;
    vec3 vTorque;
};

void BodyLoadState(out Body body, ivec2 addr) {
    body.vPos = LoadVec3(addr + offsetBodyPos);
    body.qRot = LoadVec4(addr + offsetBodyRot);
    body.vMomentum = LoadVec3(addr + offsetBodyMom);
    body.vAngularMomentum = LoadVec3(addr + offsetBodyAngMom);
}

void BodyStoreState(ivec2 addr, Body body, inout vec4 fragColor, vec2 fragCoord) {
    StoreVec3(addr + offsetBodyPos, body.vPos, fragColor, fragCoord);
    StoreVec4(addr + offsetBodyRot, body.qRot, fragColor, fragCoord);
    StoreVec3(addr + offsetBodyMom, body.vMomentum, fragColor, fragCoord);
    StoreVec3(addr + offsetBodyAngMom, body.vAngularMomentum, fragColor, fragCoord);
}

void BodyResetForFrame(inout Body body) {
    body.vForce = vec3(0.0);
    body.vTorque = vec3(0.0);
}

void BodyCalculateDerivedState(inout Body body) {
    body.mRot = QuatToMat3(body.qRot);
}

void BodyApplyGravity(inout Body body, float dT) {
    float fAccel_MpS = -9.81;
    body.vForce.y += body.fMass * fAccel_MpS;
}

void BodyIntegrate(inout Body body, float dT) {
    if (!KeyIsToggled(KEY_G)) {
        BodyApplyGravity(body, dT);
    }
    
    body.vMomentum += body.vForce * dT;
    body.vAngularMomentum += body.vTorque * dT;
    
    vec3 vVel = body.vMomentum / body.fMass;
    vec3 vAngVel = body.vAngularMomentum / body.fIT;
    
    body.vPos += vVel * dT;
    vec4 qAngDelta = QuatFromVec3(vAngVel * dT);
    body.qRot = QuatMul(qAngDelta, body.qRot);
    body.qRot = normalize(body.qRot);
}

void BodyApplyForce(inout Body body, vec3 vPos, vec3 vForce) {
    body.vForce += vForce;
    body.vTorque += cross(vPos - body.vPos, vForce);
}

void BodyApplyImpulse(inout Body body, vec3 vPos, vec3 vImpulse) {
    body.vMomentum += vImpulse;
    body.vAngularMomentum += cross(vPos - body.vPos, vImpulse);
}

vec3 BodyPointVelocity(Body body, vec3 vWorldPos) {
    vec3 vVel = body.vMomentum / body.fMass;
    vec3 vAngVel = body.vAngularMomentum / body.fIT;
    return vVel + cross(vAngVel, vWorldPos - body.vPos);
}

void BodyCollideShapeSphere(inout Body body, vec3 vSphereOrigin, float fSphereRadius, float dT) {
    vec3 vSphereWorld = ObjToWorld(vSphereOrigin, body.mRot) + body.vPos;
    
    ClosestSurface closest = GetSceneClosestSurface(vSphereWorld);
    float fDepth = fSphereRadius - closest.fDist;
    
    if (fDepth < 0.0) return;
    
    vec3 vNormal = GetSceneNormal(vSphereWorld);
    vec3 vHitPos = vSphereWorld - vNormal * closest.fDist;
    vec3 vPointVel = BodyPointVelocity(body, vHitPos);
    
    float fDot = dot(vPointVel, vNormal);
    if (fDot >= 0.0) return;
    
    float fRestitution = 0.5;
    vec3 vRelativePos = vHitPos - body.vPos;
    float fDenom = 1.0 / body.fMass;
    float fCr = dot(cross(cross(vRelativePos, vNormal), vRelativePos), vNormal);
    fDenom += fCr / body.fIT;
    
    float fImpulse = -((1.0 + fRestitution) * fDot) / fDenom;
    fImpulse += fDepth / fDenom;
    vec3 vImpulse = vNormal * fImpulse;
    
    vec3 vFriction = Vec3Perp(vPointVel, vNormal) * body.fMass;
    float fLimit = 100000.0;
    float fMag = length(vFriction);
    if (fMag > 0.0) {
        vFriction = normalize(vFriction);
        fMag = min(fMag, fLimit);
        vFriction = vFriction * fMag;
        vImpulse += vFriction * dT;
    }
    
    BodyApplyImpulse(body, vHitPos, vImpulse);
}

void BodyCollide(inout Body body, float dT) {
    BodyCollideShapeSphere(body, vec3(0.7, 0.7, 1.5), 0.5, dT);
    BodyCollideShapeSphere(body, vec3(-0.7, 0.7, 1.5), 0.5, dT);
    BodyCollideShapeSphere(body, vec3(0.7, 0.7, -1.5), 0.5, dT);
    BodyCollideShapeSphere(body, vec3(-0.7, 0.7, -1.5), 0.5, dT);
    BodyCollideShapeSphere(body, vec3(0.5, 1.0, 0.0), 0.7, dT);
    BodyCollideShapeSphere(body, vec3(-0.5, 1.0, 0.0), 0.7, dT);
}

// Wheel structure
struct Wheel {
    float fSteer;
    float fRotation;
    float fExtension;
    float fAngularVelocity;
    vec2 vContactPos;
    float fOnGround;
    float fSkid;
    vec3 vBodyPos;
    float fRadius;
    bool bIsDriven;
    bool bSteering;
};

void WheelLoadState(out Wheel wheel, ivec2 addr) {
    vec4 vState = LoadVec4(addr + offsetWheelState);
    wheel.fSteer = vState.x;
    wheel.fRotation = vState.y;
    wheel.fExtension = vState.z;
    wheel.fAngularVelocity = vState.w;
    wheel.vContactPos = vec2(0.0);
    wheel.fOnGround = 0.0;
    wheel.fSkid = 0.0;
}

void WheelStoreState(ivec2 addr, Wheel wheel, inout vec4 fragColor, vec2 fragCoord) {
    vec4 vState = vec4(wheel.fSteer, wheel.fRotation, wheel.fExtension, wheel.fAngularVelocity);
    StoreVec4(addr + offsetWheelState, vState, fragColor, fragCoord);
    vec4 vState2 = vec4(wheel.vContactPos.xy, wheel.fOnGround, wheel.fSkid);
    StoreVec4(addr + offsetWheelContactState, vState2, fragColor, fragCoord);
}

C_Intersection WheelTrace(vec3 vPos, vec3 vDir, Wheel wheel) {
    C_Intersection intersection;
    TraceScene(intersection, vPos - vDir * wheel.fRadius, vDir);
    return intersection;
}

float ClampTyreForce(inout vec3 vVel, float fLimit) {
    float fSkid = 0.0;
    float fMag = length(vVel);
    if (fMag > 0.0) {
        vVel = normalize(vVel);
    } else {
        vVel = vec3(0.0);
    }
    if (fMag > fLimit) {
        fSkid = fMag - fLimit;
        fMag = fLimit;
    }
    vVel = vVel * fMag;
    return fSkid;
}

void WheelUpdate(inout Body body, inout Wheel wheel, float dT) {
    vec3 vWheelWorld = ObjToWorld(wheel.vBodyPos, body.mRot) + body.vPos;
    vec3 vWheelDown = ObjToWorld(vec3(0.0, -1.0, 0.0), body.mRot);
    
    float fSuspensionTravel = 0.25;
    C_Intersection intersection = WheelTrace(vWheelWorld, vWheelDown, wheel);
    
    float fTravel = clamp(intersection.fDist - wheel.fRadius, 0.0, fSuspensionTravel);
    float fWheelExt = fTravel / fSuspensionTravel;
    wheel.fOnGround = 1.0 - fWheelExt;
    
    float delta = (wheel.fExtension - fTravel) / fSuspensionTravel;
    float fForce = (1.0 - fWheelExt) * 5000.0 + delta * 15000.0;
    vec3 vForce = Vec3Perp(intersection.vNormal, vWheelDown) * fForce;
    
    vec3 vIntersectWorld = intersection.vPos;
    wheel.vContactPos = vIntersectWorld.xz;
    vec3 vVelWorld = BodyPointVelocity(body, vIntersectWorld);
    vec3 vVelBody = WorldToObj(vVelWorld, body.mRot);
    vec3 vVelWheel = RotY(vVelBody, wheel.fSteer);
    
    float fWScale = wheel.fRadius;
    float fWheelMOI = 20.0;
    
    if (wheel.bIsDriven) {
        fWheelMOI = 30.0;
        if (KeyIsPressed(KEY_W)) {
            wheel.fAngularVelocity += 2.0;
        }
        if (KeyIsPressed(KEY_S)) {
            wheel.fAngularVelocity -= 2.0;
        }
    }
    
    if (KeyIsPressed(KEY_SPACE)) {
        wheel.fAngularVelocity = 0.0;
    }
    
    vVelWheel.z -= wheel.fAngularVelocity * fWScale;
    vec3 vForceWheel = vVelWheel * body.fMass;
    float fLimit = 9000.0 * (1.0 - fWheelExt);
    wheel.fSkid = ClampTyreForce(vForceWheel, fLimit);
    
    vec3 vForceBody = RotY(vForceWheel, -wheel.fSteer);
    wheel.fAngularVelocity += ((vForceWheel.z / fWScale) / fWheelMOI) * dT;
    vec3 vForceWorld = ObjToWorld(vForceBody, body.mRot);
    vForceWorld = Vec3Parallel(vForceWorld, intersection.vNormal);
    vForce -= vForceWorld;
    BodyApplyForce(body, vIntersectWorld, vForce);
    
    wheel.fExtension = fTravel;
    wheel.fRotation += wheel.fAngularVelocity * dT;
}

void WheelUpdateSteerAngle(float fSteerAngle, inout Wheel wheel) {
    if (!wheel.bSteering) {
        wheel.fSteer = 0.0;
    } else {
        float turningCircle = wheel.vBodyPos.z / tan(fSteerAngle);
        float wheelTurningCircle = turningCircle - wheel.vBodyPos.x;
        wheel.fSteer = atan(abs(wheel.vBodyPos.z) / wheelTurningCircle);
    }
}

// Vehicle structure
struct Vehicle {
    Body body;
    Wheel wheel[4];
    float fSteerAngle;
};

void VehicleLoadState(out Vehicle vehicle, ivec2 addr) {
    BodyLoadState(vehicle.body, addr + offsetVehicleBody);
    WheelLoadState(vehicle.wheel[0], addr + offsetVehicleWheel0);
    WheelLoadState(vehicle.wheel[1], addr + offsetVehicleWheel1);
    WheelLoadState(vehicle.wheel[2], addr + offsetVehicleWheel2);
    WheelLoadState(vehicle.wheel[3], addr + offsetVehicleWheel3);
    vec4 vParam0 = LoadVec4(addr + ivec2(0, 0));
    vehicle.fSteerAngle = vParam0.x;
}

void VehicleStoreState(ivec2 addr, Vehicle vehicle, inout vec4 fragColor, vec2 fragCoord) {
    BodyStoreState(addr + offsetVehicleBody, vehicle.body, fragColor, fragCoord);
    WheelStoreState(addr + offsetVehicleWheel0, vehicle.wheel[0], fragColor, fragCoord);
    WheelStoreState(addr + offsetVehicleWheel1, vehicle.wheel[1], fragColor, fragCoord);
    WheelStoreState(addr + offsetVehicleWheel2, vehicle.wheel[2], fragColor, fragCoord);
    WheelStoreState(addr + offsetVehicleWheel3, vehicle.wheel[3], fragColor, fragCoord);
    vec4 vParam0 = vec4(vehicle.fSteerAngle, 0.0, 0.0, 0.0);
    StoreVec4(addr + ivec2(0, 0), vParam0, fragColor, fragCoord);
}

void VehicleResetForFrame(inout Vehicle vehicle) {
    BodyResetForFrame(vehicle.body);
}

void VehicleSetup(inout Vehicle vehicle) {
    vehicle.body.fMass = 1000.0;
    vehicle.body.fIT = 1000.0;
    
    vehicle.wheel[0].vBodyPos = vec3(-0.9, -0.1, 1.25);
    vehicle.wheel[1].vBodyPos = vec3(0.9, -0.1, 1.25);
    vehicle.wheel[2].vBodyPos = vec3(-0.9, -0.1, -1.25);
    vehicle.wheel[3].vBodyPos = vec3(0.9, -0.1, -1.25);
    
    vehicle.wheel[0].fRadius = 0.45;
    vehicle.wheel[1].fRadius = 0.45;
    vehicle.wheel[2].fRadius = 0.45;
    vehicle.wheel[3].fRadius = 0.45;
    
    vehicle.wheel[0].bIsDriven = false;
    vehicle.wheel[1].bIsDriven = false;
    vehicle.wheel[2].bIsDriven = true;
    vehicle.wheel[3].bIsDriven = true;
    
    vehicle.wheel[0].bSteering = true;
    vehicle.wheel[1].bSteering = true;
    vehicle.wheel[2].bSteering = false;
    vehicle.wheel[3].bSteering = false;
}

// Camera structure
struct Camera {
    vec3 vPos;
    vec3 vTarget;
};

void CameraLoadState(out Camera cam, ivec2 addr) {
    cam.vPos = LoadVec3(addr + offsetCameraPos);
    cam.vTarget = LoadVec3(addr + offsetCameraTarget);
}

void CameraStoreState(Camera cam, ivec2 addr, inout vec4 fragColor, vec2 fragCoord) {
    StoreVec3(addr + offsetCameraPos, cam.vPos, fragColor, fragCoord);
    StoreVec3(addr + offsetCameraTarget, cam.vTarget, fragColor, fragCoord);
}

// ============================================
// BUFFER A: Physics Simulation
// ============================================
void mainImageA(out vec4 fragColor, in vec2 fragCoord) {
    // Only process state storage pixels (small area)
    if (fragCoord.x > 13.0 || fragCoord.y > 3.0) {
        fragColor = vec4(0.0);
        return;
    }
    
    Vehicle vehicle;
    VehicleLoadState(vehicle, addrVehicle);
    VehicleSetup(vehicle);
    VehicleResetForFrame(vehicle);
    
    if (iFrame < 1) {
        vehicle.body.vPos = vec3(0.0, -2.5, 0.0);
        vehicle.body.vMomentum = vec3(0.0);
        vehicle.body.qRot = vec4(0.0, 0.0, 0.0, 1.0);
        vehicle.body.vAngularMomentum = vec3(0.0, 0.5, 0.0);
        vehicle.fSteerAngle = 0.0;
    }
    
    BodyCalculateDerivedState(vehicle.body);
    
    if (KeyIsPressed(KEY_A)) {
        vehicle.fSteerAngle += 0.05;
    }
    if (KeyIsPressed(KEY_D)) {
        vehicle.fSteerAngle -= 0.05;
    }
    vehicle.fSteerAngle *= 0.9;
    
    float fSteerAngle = vehicle.fSteerAngle / (1.0 + length(vehicle.body.vMomentum) * 0.0001);
    WheelUpdateSteerAngle(fSteerAngle, vehicle.wheel[0]);
    WheelUpdateSteerAngle(fSteerAngle, vehicle.wheel[1]);
    WheelUpdateSteerAngle(fSteerAngle, vehicle.wheel[2]);
    WheelUpdateSteerAngle(fSteerAngle, vehicle.wheel[3]);
    
    float dT = 1.0 / 60.0;
    
    if (!KeyIsToggled(KEY_V)) {
        WheelUpdate(vehicle.body, vehicle.wheel[0], dT);
        WheelUpdate(vehicle.body, vehicle.wheel[1], dT);
        WheelUpdate(vehicle.body, vehicle.wheel[2], dT);
        WheelUpdate(vehicle.body, vehicle.wheel[3], dT);
    }
    
    BodyCollide(vehicle.body, dT);
    BodyIntegrate(vehicle.body, dT);
    
    fragColor = vec4(0.0);
    VehicleStoreState(addrVehicle, vehicle, fragColor, fragCoord);
    
    Camera prevCam;
    CameraLoadState(prevCam, addrCamera);
    CameraStoreState(prevCam, addrPrevCamera, fragColor, fragCoord);
    
    Camera cam;
    vec2 vMouse = iMouse.xy / iResolution.xy;
    float fAngle = (-vMouse.x * 2.0 + 1.0) * 3.14;
    float fDistance = 8.0 - vMouse.y * 6.0;
    
    cam.vTarget = vec3(0.0, 1.0, 0.0) * vehicle.body.mRot + vehicle.body.vPos;
    cam.vPos = vec3(0.0, 0.0, -fDistance) * vehicle.body.mRot + vehicle.body.vPos + vec3(0.0, 2.0, 0.0);
    cam.vPos -= cam.vTarget;
    cam.vPos = RotY(cam.vPos, fAngle);
    cam.vPos += cam.vTarget;
    
    CameraStoreState(cam, addrCamera, fragColor, fragCoord);
}

// ============================================
// BUFFER B: Tyre Tracks
// ============================================
void mainImageB(out vec4 fragColor, in vec2 fragCoord) {
    fragColor = vec4(0.0, 0.0, 0.0, 1.0);
    
    Camera cam;
    CameraLoadState(cam, addrCamera);
    Camera prevCam;
    CameraLoadState(prevCam, addrPrevCamera);
    
    float fRange = 20.0;
    vec2 vPrevOrigin = floor(prevCam.vPos.xz);
    vec2 vCurrOrigin = floor(cam.vPos.xz);
    
    vec2 vFragOffset = ((fragCoord / iResolution.xy) * 2.0 - 1.0) * fRange;
    vec2 vFragWorldPos = vFragOffset + vCurrOrigin;
    
    vec2 vPrevFragOffset = vFragWorldPos - vPrevOrigin;
    vec2 vPrevUV = ((vPrevFragOffset / fRange) + 1.0) / 2.0;
    vec4 vPrevSample = textureLod(ChBufferB, vPrevUV, 0.0);
    
    vec4 vWheelContactState[4];
    vWheelContactState[0] = LoadVec4(addrVehicle + offsetVehicleWheel0 + offsetWheelContactState);
    vWheelContactState[1] = LoadVec4(addrVehicle + offsetVehicleWheel1 + offsetWheelContactState);
    vWheelContactState[2] = LoadVec4(addrVehicle + offsetVehicleWheel2 + offsetWheelContactState);
    vWheelContactState[3] = LoadVec4(addrVehicle + offsetVehicleWheel3 + offsetWheelContactState);
    
    fragColor = vPrevSample;
    
    if (vPrevUV.x < 0.0 || vPrevUV.x >= 1.0 || vPrevUV.y < 0.0 || vPrevUV.y >= 1.0) {
        fragColor = vec4(0.0);
    }
    
    for (int w = 0; w < 4; w++) {
        vec2 vContactPos = vWheelContactState[w].xy;
        float fDist = length(vFragWorldPos - vContactPos);
        
        if (vWheelContactState[w].z > 0.01) {
            float fAmount = smoothstep(0.25, 0.1, fDist);
            fragColor.x = max(fragColor.x, fAmount * vWheelContactState[w].z);
            fragColor.y = max(fragColor.y, fAmount * vWheelContactState[w].w * 0.01);
        }
    }
    
    fragColor.x = clamp(fragColor.x, 0.0, 1.0);
    fragColor.y = clamp(fragColor.y, 0.0, 1.0);
    
    if (iFrame < 1) {
        fragColor.x = 0.0;
    }
}

// ============================================
// BUFFER C: Main Rendering
// ============================================

// Helper functions for vehicle rendering
ClosestSurface ClosestSurfaceUnion(ClosestSurface a, ClosestSurface b) {
    if (a.fDist < b.fDist) {
        return a;
    }
    return b;
}

vec2 Segment(vec3 vPos, vec3 vP0, vec3 vP1) {
    vec3 pa = vPos - vP0;
    vec3 ba = vP1 - vP0;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return vec2(length(pa - ba * h), h);
}

float SdCapsule(vec3 vPos, vec3 vP0, vec3 vP1, float r0, float r1) {
    vec2 vC = Segment(vPos, vP0, vP1);
    return vC.x - mix(r0, r1, vC.y);
}

float PlaneDist(vec3 vPos, vec3 vNormal, float fDist) {
    return dot(vNormal, vPos) - fDist;
}

float CarBodyMin(float a, float b, float k) {
    return smin(a, b, k);
}

float CarBodyMax(float a, float b, float k) {
    return -CarBodyMin(-a, -b, k);
}

float WheelArchCombine(float a, float b) {
    float size = 0.1;
    float r = clamp(1.0 - abs(b) / size, 0.0, 1.0);
    a -= r * r * size;
    return CarBodyMax(a, b, 0.1);
}

float GetWheelArchDist(vec3 vPos) {
    vPos.y = max(vPos.y, 0.0);
    return 0.45 - length(vec2(length(vPos.zy), vPos.x));
}

// Vehicle state for rendering
struct VehicleState {
    vec3 vPos;
    vec4 qRot;
    mat3 mRot;
    vec4 vWheelState0;
    vec4 vWheelState1;
    vec4 vWheelState2;
    vec4 vWheelState3;
    vec4 vWheelSC0;
    vec4 vWheelSC1;
    vec4 vWheelSC2;
    vec4 vWheelSC3;
};

VehicleState SetupSceneState() {
    VehicleState state;
    state.vPos = LoadVec3(addrVehicle + offsetVehicleBody + offsetBodyPos);
    state.qRot = LoadVec4(addrVehicle + offsetVehicleBody + offsetBodyRot);
    state.mRot = QuatToMat3(state.qRot);
    
    vec4 vWheelState0 = LoadVec4(addrVehicle + offsetVehicleWheel0);
    vec4 vWheelState1 = LoadVec4(addrVehicle + offsetVehicleWheel1);
    vec4 vWheelState2 = LoadVec4(addrVehicle + offsetVehicleWheel2);
    vec4 vWheelState3 = LoadVec4(addrVehicle + offsetVehicleWheel3);
    
    state.vWheelState0 = vWheelState0;
    state.vWheelState1 = vWheelState1;
    state.vWheelState2 = vWheelState2;
    state.vWheelState3 = vWheelState3;
    
    // Precompute sin/cos for wheel rotations
    state.vWheelSC0 = vec4(sin(vWheelState0.x), cos(vWheelState0.x), sin(vWheelState0.y), cos(vWheelState0.y));
    state.vWheelSC1 = vec4(sin(vWheelState1.x), cos(vWheelState1.x), sin(vWheelState1.y), cos(vWheelState1.y));
    state.vWheelSC2 = vec4(sin(vWheelState2.x), cos(vWheelState2.x), sin(vWheelState2.y), cos(vWheelState2.y));
    state.vWheelSC3 = vec4(sin(vWheelState3.x), cos(vWheelState3.x), sin(vWheelState3.y), cos(vWheelState3.y));
    
    return state;
}

// Wheel rendering constants
static const float g_fWheelR = 0.45;
static const float g_fWheelW = 0.25;

ClosestSurface GetWheelClosestSurface(vec3 vPos) {
    float theta = atan(vPos.z, vPos.y);
    float r = length(vPos.zy);
    float x = vPos.x;
    
    float fr = r * (1.0 / g_fWheelR);
    
    if (fr < 0.5) {
        x += 0.01 * clamp((0.5 - fr) * 30.0, 0.0, 1.0);
        if (fr < 0.3) {
            float unitr = fr / 0.3;
            x = x + sqrt(1.0 - unitr * unitr) * 0.05;
        }
    }
    
    float fRound = 0.1;
    float fWheelR = g_fWheelR - fRound;
    float fWheelW = g_fWheelW - fRound;
    vec2 rx = vec2(r, x);
    
    ClosestSurface closest;
    closest.surface.iId = MAT_WHEEL;
    closest.surface.vUVW = vPos.yzx;
    closest.fDist = length(max(abs(rx) - vec2(fWheelR, fWheelW), 0.0)) - fRound;
    
    return closest;
}

ClosestSurface GetCarBodyClosestSurface(vec3 vCarPos) {
    ClosestSurface closest;
    vec3 vAbsBodyPos = vCarPos - vec3(0.0, 0.3, 0.0);
    vec3 vBodyPos = vAbsBodyPos;
    vBodyPos.x = abs(vBodyPos.x);
    closest.surface.vUVW = vAbsBodyPos;
    closest.surface.iId = MAT_CAR_BODY;
    
    vec3 vFrontWheelPos = -vec3(0.0, -0.1, -1.25);
    vec3 vRearWheelPos = -vec3(0.0, -0.1, 1.25);
    vec3 vWheelPos = vBodyPos - vFrontWheelPos;
    float fSeparation = (vFrontWheelPos.z - vRearWheelPos.z) * 0.5;
    vWheelPos.z = abs(vWheelPos.z + fSeparation) - fSeparation;
    vWheelPos.x = abs(vWheelPos.x) - 0.8;
    float fWheelArchDist = GetWheelArchDist(vWheelPos);
    
    float fBodyBaseDist = kFarClip;
    {
        float fTopDist = PlaneDist(vBodyPos, normalize(vec3(0.0, 1.0, 0.0)), 0.8);
        float fFrontDist = PlaneDist(vBodyPos, normalize(vec3(0.0, 0.2, 1.0)), 1.9);
        float fSideDist = PlaneDist(vBodyPos, normalize(vec3(1.0, -0.1, 0.0)), 0.85);
        float fBaseDist = PlaneDist(vBodyPos, normalize(vec3(0.0, -1.0, 0.0)), -0.1);
        float fBackDist = PlaneDist(vBodyPos, normalize(vec3(0.0, 0.0, -1.0)), 2.0);
        
        float fX = abs(vBodyPos.x);
        fTopDist += fX * fX * 0.05;
        fFrontDist += fX * fX * 0.1;
        
        float fFrontTopDist = CarBodyMax(fTopDist, fFrontDist, 0.2);
        fBodyBaseDist = fFrontTopDist;
        fBodyBaseDist = CarBodyMax(fBodyBaseDist, fSideDist, 0.3);
        float fBaseBackDist = CarBodyMax(fBaseDist, fBackDist, 0.1);
        fBodyBaseDist = CarBodyMax(fBodyBaseDist, fBaseBackDist, 0.1);
    }
    
    fBodyBaseDist = WheelArchCombine(fBodyBaseDist, fWheelArchDist);
    
    float fBodyTopDist = kFarClip;
    {
        float fTopDist = PlaneDist(vBodyPos, normalize(vec3(0.0, 1.0, 0.0)), 1.3);
        float fFrontDist = PlaneDist(vBodyPos, normalize(vec3(0.0, 1.0, 0.7)), 1.1);
        float fSideDist = PlaneDist(vBodyPos, normalize(vec3(1.0, 0.4, 0.0)), 1.03);
        float fBaseDist = PlaneDist(vBodyPos, normalize(vec3(0.0, -1.0, 0.0)), -0.7);
        float fBackDist = PlaneDist(vBodyPos, normalize(vec3(0.0, 0.0, -1.0)), 0.55);
        
        float fX = abs(vBodyPos.x);
        fTopDist += fX * fX * 0.1;
        
        float fFrontTopDist = CarBodyMax(fTopDist, fFrontDist, 0.1);
        fBodyTopDist = fFrontTopDist;
        fBodyTopDist = CarBodyMax(fBodyTopDist, fSideDist, 0.1);
        float fBaseBackDist = CarBodyMax(fBaseDist, fBackDist, 0.1);
        fBodyTopDist = CarBodyMax(fBodyTopDist, fBaseBackDist, 0.1);
    }
    
    closest.fDist = fBodyBaseDist;
    closest.fDist = smin(closest.fDist, fBodyTopDist, 0.1);
    
    return closest;
}

ClosestSurface GetVehicleClosestSurface(VehicleState vehicleState, vec3 vPos) {
    vec3 vLocalPos = vehicleState.mRot * (vPos - vehicleState.vPos);
    ClosestSurface closest = GetCarBodyClosestSurface(vLocalPos);
    
    vec3 vWheelPos0 = vec3(-0.9, -0.1, 1.25);
    vec3 vWheelPos1 = vec3(0.9, -0.1, 1.25);
    vec3 vWheelPos2 = vec3(-0.9, -0.1, -1.25);
    vec3 vWheelPos3 = vec3(0.9, -0.1, -1.25);
    
    vec3 vWheelOrigin;
    vec4 vWheelState;
    vec4 vWheelSC;
    
    if (vLocalPos.z > 0.0) {
        if (vLocalPos.x < 0.0) {
            vWheelOrigin = vWheelPos0;
            vWheelState = vehicleState.vWheelState0;
            vWheelSC = vehicleState.vWheelSC0;
        } else {
            vWheelOrigin = vWheelPos1;
            vWheelState = vehicleState.vWheelState1;
            vWheelSC = vehicleState.vWheelSC1;
        }
    } else {
        if (vLocalPos.x < 0.0) {
            vWheelOrigin = vWheelPos2;
            vWheelState = vehicleState.vWheelState2;
            vWheelSC = vehicleState.vWheelSC2;
        } else {
            vWheelOrigin = vWheelPos3;
            vWheelState = vehicleState.vWheelState3;
            vWheelSC = vehicleState.vWheelSC3;
        }
    }
    
    vec3 vWheelPos = vWheelOrigin;
    float fWheelSide = sign(vWheelOrigin.x);
    vWheelPos.y -= vWheelState.z - g_fWheelR;
    vec3 vWheelLocalPos = vWheelPos - vLocalPos;
    vWheelLocalPos = RotY(vWheelLocalPos, vWheelSC.xy);
    vWheelLocalPos = RotX(vWheelLocalPos, vWheelSC.zw);
    vWheelLocalPos.x *= -fWheelSide;
    closest = ClosestSurfaceUnion(closest, GetWheelClosestSurface(vWheelLocalPos));
    
    return closest;
}

ClosestSurface GetSceneClosestSurfaceRender(VehicleState vehicleState, vec3 vPos) {
    ClosestSurface terrainClosest = GetEnvironmentClosestSurface(vPos);
    ClosestSurface vehicleClosest = GetVehicleClosestSurface(vehicleState, vPos);
    
    if (terrainClosest.fDist < vehicleClosest.fDist) {
        return terrainClosest;
    }
    return vehicleClosest;
}

vec3 GetSceneNormalRender(VehicleState vehicleState, vec3 vPos) {
    float fDelta = 0.0005;
    vec3 vDir1 = vec3(1.0, -1.0, -1.0);
    vec3 vDir2 = vec3(-1.0, -1.0, 1.0);
    vec3 vDir3 = vec3(-1.0, 1.0, -1.0);
    vec3 vDir4 = vec3(1.0, 1.0, 1.0);
    
    ClosestSurface c1 = GetSceneClosestSurfaceRender(vehicleState, vPos + vDir1 * fDelta);
    ClosestSurface c2 = GetSceneClosestSurfaceRender(vehicleState, vPos + vDir2 * fDelta);
    ClosestSurface c3 = GetSceneClosestSurfaceRender(vehicleState, vPos + vDir3 * fDelta);
    ClosestSurface c4 = GetSceneClosestSurfaceRender(vehicleState, vPos + vDir4 * fDelta);
    
    vec3 vNormal = vDir1 * c1.fDist + vDir2 * c2.fDist + vDir3 * c3.fDist + vDir4 * c4.fDist;
    return normalize(vNormal);
}

C_Intersection TraceSceneRender(VehicleState vehicleState, vec3 vOrigin, vec3 vDir) {
    C_Intersection intersection;
    float t = 0.1;
    const int kRaymarchMaxIter = 64;
    
    for (int i = 0; i < kRaymarchMaxIter; i++) {
        float fClosestDist = GetSceneClosestSurfaceRender(vehicleState, vOrigin + vDir * t).fDist;
        t += fClosestDist;
        if (abs(fClosestDist) < 0.01) {
            break;
        }
        if (t > kFarClip) {
            t = kFarClip;
            break;
        }
    }
    
    intersection.fDist = t;
    intersection.vPos = vOrigin + vDir * t;
    
    if (t >= kFarClip) {
        intersection.surface.iId = 0;
        intersection.surface.vUVW = vec3(0.0);
        intersection.vNormal = vec3(0.0, 1.0, 0.0);
    } else {
        intersection.vNormal = GetSceneNormalRender(vehicleState, intersection.vPos);
        intersection.surface = GetSceneClosestSurfaceRender(vehicleState, intersection.vPos).surface;
    }
    
    return intersection;
}

float TraceShadow(VehicleState vehicleState, vec3 vOrigin, vec3 vDir, float fDist) {
    const int kShadowIter = 16;
    const float kShadowFalloff = 10.0;
    float fShadow = 1.0;
    float t = 0.01;
    float fDelta = 2.5 / float(kShadowIter);
    
    for (int i = 0; i < kShadowIter; i++) {
        ClosestSurface closest = GetSceneClosestSurfaceRender(vehicleState, vOrigin + vDir * t);
        fShadow = min(fShadow, kShadowFalloff * closest.fDist / t);
        t += fDelta;
    }
    
    return clamp(fShadow, 0.0, 1.0);
}

float GetAmbientOcclusion(VehicleState vehicleState, vec3 vPos, vec3 vNormal) {
    float fAmbientOcclusion = 0.0;
    float fStep = 0.1;
    float fDist = 0.0;
    
    for (int i = 0; i <= 5; i++) {
        fDist += fStep;
        ClosestSurface closest = GetSceneClosestSurfaceRender(vehicleState, vPos + vNormal * fDist);
        float fAmount = fDist - closest.fDist;
        fAmbientOcclusion += max(0.0, fAmount * fDist);
    }
    
    return max(1.0 - fAmbientOcclusion, 0.0);
}

// Material structure
struct MaterialInfo {
    vec3 vAlbedo;
    float fR0;
    float fSmoothness;
    vec3 vBumpNormal;
};

// Get tyre track sample
vec4 GetTyreTrackSample(vec3 vPos, vec2 vTyreTrackOrigin) {
    float fRange = 20.0;
    vec2 vPrevFragOffset = vPos.xz - vTyreTrackOrigin;
    vec2 vPrevUV = ((vPrevFragOffset / fRange) + 1.0) / 2.0;
    
    if (vPrevUV.x < 0.0 || vPrevUV.x >= 1.0 || vPrevUV.y < 0.0 || vPrevUV.y >= 1.0) {
        return vec4(0.0);
    }
    
    return textureLod(ChBufferB, vPrevUV, 0.0);
}

void GetSurfaceInfo(out MaterialInfo mat, C_Intersection intersection, vec2 vTyreTrackOrigin) {
    mat.vBumpNormal = intersection.vNormal;
    mat.fR0 = 0.02;
    
    vec4 vTrackSample = GetTyreTrackSample(intersection.vPos, vTyreTrackOrigin);
    
    switch (intersection.surface.iId) {
        case MAT_TERRAIN: {
            // Use channel0 for terrain texture
            vec2 vUV = intersection.surface.vUVW.xy * 0.1;
            vec3 vTex = texture(channel0, vUV).rgb;
            
            // Apply bump mapping
            #ifndef FAST_VERSION
            float fBumpScale = 1.0;
            vec2 vDU = vec2(1.0 / 512.0, 0.0); // Approximate texture resolution
            vec2 vDV = vec2(0.0, 1.0 / 512.0);
            
            float fSampleW = texture(channel0, vUV - vDU).r;
            float fSampleE = texture(channel0, vUV + vDU).r;
            float fSampleN = texture(channel0, vUV - vDV).r;
            float fSampleS = texture(channel0, vUV + vDV).r;
            
            vec3 vNormalDelta = vec3(0.0);
            vNormalDelta.x += (fSampleW * fSampleW - fSampleE * fSampleE) * fBumpScale;
            vNormalDelta.z += (fSampleN * fSampleN - fSampleS * fSampleS) * fBumpScale;
            mat.vBumpNormal = normalize(mat.vBumpNormal + vNormalDelta);
            #endif
            
            mat.vAlbedo = vTex * vTex; // Gamma correction
            mat.fSmoothness = mat.vAlbedo.r * 0.3;
            
            // Apply tyre tracks
            float fDepth = vTrackSample.x * (1.0 + vTrackSample.y);
            vec3 vTrackColor = mix(mat.vAlbedo, vec3(0.9, 0.3, 0.01), 0.5);
            mat.vAlbedo = mix(mat.vAlbedo, vTrackColor, fDepth);
            mat.vAlbedo *= 1.0 - 0.6 * vTrackSample.g;
            mat.fSmoothness = mix(mat.fSmoothness, mat.fSmoothness * 0.75 + 0.25, fDepth);
        }
        break;
        
        case MAT_CAR_BODY: {
            mat.vAlbedo = vec3(0.0, 0.0, 1.0);
            mat.fSmoothness = 1.0;
            
            float fAbsX = abs(intersection.surface.vUVW.x);
            float fStripe = abs(fAbsX - 0.15);
            fStripe = smoothstep(0.1 + 0.01, 0.1 - 0.01, fStripe);
            mat.vAlbedo = mix(mat.vAlbedo, vec3(1.0, 1.0, 1.0), fStripe);
            
            if (intersection.surface.vUVW.y < 0.85) {
                float fLine = abs(intersection.surface.vUVW.z - 0.7);
                fLine = min(fLine, abs(intersection.surface.vUVW.z + 0.6));
                fLine = min(fLine, abs(fAbsX - 0.65));
                fLine = min(fLine, abs(intersection.surface.vUVW.y - 0.2));
                fLine = clamp((fLine - 0.005) / 0.01, 0.0, 1.0);
                mat.vAlbedo *= fLine;
                mat.fR0 *= fLine;
                mat.fSmoothness *= fLine;
            }
            
            if (fAbsX > 0.92) {
                mat.vAlbedo = vec3(0.02, 0.02, 0.02);
                mat.fSmoothness = 0.2;
            }
            
            // Window glass - detect window areas and make them transparent/reflective
            if (intersection.surface.vUVW.y > 0.85 && intersection.surface.vUVW.y < 1.2) {
                bool bFront = (intersection.surface.vUVW.z + intersection.surface.vUVW.y * 1.25) > 1.6;
                bool bRear = (intersection.surface.vUVW.z) < -0.45;
                bool bSide = (fAbsX + intersection.surface.vUVW.y * 0.3) > 0.9;
                
                // If it's a window area (not front+side or rear+side), make it glass
                if (!(bFront && bSide) && !(bRear && bSide)) {
                    mat.vAlbedo = vec3(0.0, 0.0, 0.0);  // Black glass
                    mat.fR0 = 0.02;  // Low reflectivity
                    mat.fSmoothness = 0.9;  // High smoothness for glass
                }
            }
            
            // Grill (front grille)
            vec3 vGrillDomain = intersection.surface.vUVW - vec3(0.0, 0.55, 1.85);
            float fGrillDist = UdRoundBox(vGrillDomain, vec3(0.85, 0.05, 0.0), 0.1);
            if (fGrillDist < 0.0) {
                mat.vAlbedo = vec3(0.0, 0.0, 0.0);
                mat.fR0 = 0.02;
            }
            
            // Headlights
            vec3 vLightDomain = intersection.surface.vUVW;
            vLightDomain.x = abs(vLightDomain.x);
            vLightDomain -= vec3(0.6, 0.56, 1.85);
            float fLightDist = UdRoundBox(vLightDomain, vec3(0.1, 0.04, 0.5), 0.05);
            if (fLightDist < 0.0) {
                mat.vAlbedo = vec3(0.5);
                mat.fR0 = 1.0;
                mat.fSmoothness = 0.8;
            }
            
            // Add dirt effect using channel1
            vec3 vDirt = (texture(channel1, intersection.surface.vUVW.zy).rgb + 
                         texture(channel1, intersection.surface.vUVW.xy).rgb) * 0.5;
            float fDirt = vDirt.r;
            float fMix = clamp(fDirt - intersection.surface.vUVW.y * 1.5 + 0.8, 0.0, 1.0);
            vDirt = vDirt * vDirt * 0.1;
            mat.vAlbedo = mix(mat.vAlbedo, vDirt, fMix);
            mat.fR0 = mix(mat.fR0, 0.01, fMix);
            mat.fSmoothness = mix(mat.fSmoothness, 0.01, fMix);
        }
        break;
        
        case MAT_CHROME: {
            mat.vAlbedo = vec3(0.1);
            mat.fSmoothness = 1.0;
            mat.fR0 = 0.9;
        }
        break;
        
        case MAT_WHEEL: {
            vec2 vUV = intersection.surface.vUVW.xy;
            mat.vAlbedo = texture(channel1, vUV).rgb;
            mat.vAlbedo = mat.vAlbedo * mat.vAlbedo;
            mat.vAlbedo *= 0.01;
            
            float len = length(vUV);
            float fR = len * (1.0 / g_fWheelR);
            if (fR < 0.5) {
                mat.fSmoothness = 1.0;
                mat.fR0 = 1.0;
            } else {
                mat.fSmoothness = 0.1;
            }
            
            // Add dirt effect
            vec3 vDirt = (texture(channel1, intersection.surface.vUVW.zy).rgb + 
                         texture(channel1, intersection.surface.vUVW.xy).rgb) * 0.5;
            float fDirt = sqrt(vDirt.r);
            float fMix = clamp(fDirt - (1.0 - fR) * 1.0 + 0.8, 0.0, 1.0);
            vDirt = vDirt * vDirt * 0.1;
            mat.vAlbedo = mix(mat.vAlbedo, vDirt, fMix);
            mat.fR0 = mix(mat.fR0, 0.01, fMix);
            mat.fSmoothness = mix(mat.fSmoothness, 0.01, fMix);
        }
        break;
        
        case MAT_SUSPENSION: {
            mat.vAlbedo = vec3(0.1);
            mat.fSmoothness = 1.0;
            mat.fR0 = 0.9;
            
            float fY = intersection.surface.vUVW.y;
            float fAngle = atan(intersection.surface.vUVW.x, intersection.surface.vUVW.y);
            fAngle -= fY * 30.0;
            float fFAngle = fract(fAngle / (3.1415 * 2.0));
            if (fFAngle < 0.5) {
                mat.fR0 = 0.0;
                mat.vAlbedo = vec3(0.0);
            }
            
            // Add dirt effect
            vec3 vDirt = (texture(channel1, intersection.surface.vUVW.zy).rgb + 
                         texture(channel1, intersection.surface.vUVW.xy).rgb) * 0.5;
            float fDirt = sqrt(vDirt.r);
            float fMix = clamp(fDirt + 0.1, 0.0, 1.0);
            vDirt = vDirt * vDirt * 0.1;
            mat.vAlbedo = mix(mat.vAlbedo, vDirt, fMix);
            mat.fR0 = mix(mat.fR0, 0.01, fMix);
            mat.fSmoothness = mix(mat.fSmoothness, 0.01, fMix);
        }
        break;
        
        case MAT_WOOD: {
            // Use channel1 for wood texture
            vec2 vUV = intersection.surface.vUVW.xz * 0.1;
            mat.vAlbedo = texture(channel1, vUV).rgb;
            mat.vAlbedo = mat.vAlbedo * mat.vAlbedo;
            mat.fSmoothness = mat.vAlbedo.r;
            mat.vAlbedo *= 1.0 - vTrackSample.g * 0.6;
        }
        break;
        
        default: {
            mat.vAlbedo = vec3(0.5);
            mat.fSmoothness = 0.5;
            mat.fR0 = 0.02;
        }
        break;
    }
}

vec3 GetSkyColour(vec3 vDir) {
    vec3 vSkyTop = vec3(0.1, 0.5, 0.8);
    vec3 vSkyBottom = vec3(0.02, 0.04, 0.06);
    return mix(vSkyBottom, vSkyTop, abs(vDir.y)) * 30.0;
}

float GetFogFactor(float fDist) {
    const float kFogDensity = 0.0025;
    return exp(fDist * -kFogDensity);
}

vec3 GetSceneColour(VehicleState vehicleState, vec3 vRayOrigin, vec3 vRayDir, vec2 vTyreTrackOrigin, out float fDepth) {
    vec3 vColour = vec3(0.0);
    float fRemaining = 1.0;
    fDepth = 0.0;
    float fFirstTrace = 1.0;
    
    vec3 vSunLightColour = vec3(1.0, 0.9, 0.6) * 10.0;
    vec3 vSunLightDir = normalize(vec3(0.4, -0.3, -0.5));
    
    for (int i = 0; i < RAYTRACE_COUNT; i++) {
        C_Intersection intersection = TraceSceneRender(vehicleState, vRayOrigin, vRayDir);
        
        float fHitDepth = intersection.fDist;
        if (intersection.surface.iId >= MAT_CAR_BODY) {
            fHitDepth = -fHitDepth;
        }
        fDepth = (fFirstTrace > 0.0) ? fHitDepth : fDepth;
        fFirstTrace = 0.0;
        
        if (intersection.surface.iId == 0) {
            vec3 vSkyColor = GetSkyColour(vRayDir);
            vColour += vSkyColor * fRemaining;
            break;
        }
        
        MaterialInfo mat;
        GetSurfaceInfo(mat, intersection, vTyreTrackOrigin);
        
        // Lighting
        vec3 vShadowRayDir = -vSunLightDir;
        vec3 vShadowRayOrigin = intersection.vPos + vShadowRayDir * 0.01;
        float fShadowFactor = TraceShadow(vehicleState, vShadowRayOrigin, vShadowRayDir, 10.0);
        
        float fNDotL = clamp(dot(-vSunLightDir, mat.vBumpNormal), 0.0, 1.0);
        vec3 vDiffuseLight = vSunLightColour * fNDotL * fShadowFactor;
        vDiffuseLight += 0.2 * GetAmbientOcclusion(vehicleState, intersection.vPos, mat.vBumpNormal);
        
        // Specular lighting
        vec3 vReflectDir = reflect(-vSunLightDir, mat.vBumpNormal);
        float fSpecDot = clamp(dot(-vRayDir, vReflectDir), 0.0, 1.0);
        float fSpecPower = pow(fSpecDot, 32.0) * mat.fSmoothness;
        vec3 vSpecularLight = vSunLightColour * fSpecPower * fShadowFactor;
        
        // Fresnel effect
        float fFresnel = mat.fR0 + (1.0 - mat.fR0) * pow(1.0 - dot(-mat.vBumpNormal, vRayDir), 5.0) * pow(mat.fSmoothness, 5.0);
        
        vec3 vResult = mix(mat.vAlbedo * vDiffuseLight, vSpecularLight, fFresnel);
        
        // Atmosphere
        float fFogFactor = GetFogFactor(intersection.fDist);
        vec3 vFogColour = vec3(0.02, 0.04, 0.06) * 25.0;
        vResult = mix(vFogColour, vResult, fFogFactor);
        
        vColour += vResult * fRemaining;
        fRemaining *= fFresnel;
        
        if (fRemaining < 0.05) {
            break;
        }
        
        vRayDir = normalize(reflect(vRayDir, mat.vBumpNormal));
        vRayOrigin = intersection.vPos;
    }
    
    vec3 vSkyColor = GetSkyColour(vRayDir);
    float fFogFactor = GetFogFactor(kFarClip);
    vec3 vFogColour = vec3(0.02, 0.04, 0.06) * 25.0;
    vec3 vSkyFinal = mix(vFogColour, vSkyColor, fFogFactor);
    vColour += vSkyFinal * fRemaining;
    
    return vColour;
}

vec2 GetWindowCoord(vec2 vUV) {
    vec2 vWindow = vUV * 2.0 - 1.0;
    vWindow.x *= iResolution.x / iResolution.y;
    return vWindow;
}

vec3 GetCameraRayDir(vec2 vWindow, vec3 vCameraPos, vec3 vCameraTarget) {
    vec3 vForward = normalize(vCameraTarget - vCameraPos);
    vec3 vRight = normalize(cross(vec3(0.0, 1.0, 0.0), vForward));
    vec3 vUp = normalize(cross(vForward, vRight));
    return normalize(vWindow.x * vRight + vWindow.y * vUp + vForward * 2.0);
}

void mainImageC(out vec4 fragColor, in vec2 fragCoord) {
    vec2 vUV = fragCoord.xy / iResolution.xy;
    
    vec3 vCameraPos = LoadVec3(addrCamera + offsetCameraPos);
    vec3 vCameraTarget = LoadVec3(addrCamera + offsetCameraTarget);
    vec2 vTyreTrackOrigin = floor(vCameraPos.xz);
    
    vec3 vRayOrigin = vCameraPos;
    vec3 vRayDir = GetCameraRayDir(GetWindowCoord(vUV), vCameraPos, vCameraTarget);
    
    float fDepth;
    VehicleState vehicleState = SetupSceneState();
    vec3 vResult = GetSceneColour(vehicleState, vRayOrigin, vRayDir, vTyreTrackOrigin, fDepth);
    vResult = max(vResult, vec3(0.0));
    
    fragColor = vec4(vResult, fDepth);
}

// ============================================
// MAIN IMAGE: Post Processing
// ============================================
vec2 GetUVFromWindowCoord(vec2 vWindow) {
    vec2 vScaledWindow = vWindow;
    vScaledWindow.x *= iResolution.y / iResolution.x;
    return vScaledWindow * 0.5 + 0.5;
}

vec2 GetCameraWindowCoord(vec3 vWorldPos, vec3 vCameraPos, vec3 vCameraTarget) {
    vec3 vForward = normalize(vCameraTarget - vCameraPos);
    vec3 vRight = normalize(cross(vec3(0.0, 1.0, 0.0), vForward));
    vec3 vUp = normalize(cross(vForward, vRight));
    
    vec3 vOffset = vWorldPos - vCameraPos;
    vec3 vCameraLocal;
    vCameraLocal.x = dot(vOffset, vRight);
    vCameraLocal.y = dot(vOffset, vUp);
    vCameraLocal.z = dot(vOffset, vForward);
    
    vec2 vWindowPos = vCameraLocal.xy / (vCameraLocal.z / 2.0);
    return vWindowPos;
}

float GetCoC(float fDistance, float fPlaneInFocus) {
    float fAperture = 0.03;
    float fFocalLength = 1.0;
    return abs(fAperture * (fFocalLength * (fDistance - fPlaneInFocus)) /
               (fDistance * (fPlaneInFocus - fFocalLength)));
}

vec3 Tonemap(vec3 x) {
    float a = 0.010;
    float b = 0.132;
    float c = 0.010;
    float d = 0.163;
    float e = 0.101;
    return (x * (a * x + b)) / (x * (c * x + d) + e);
}

vec3 ColorGrade(vec3 vColor) {
    vec3 vHue = vec3(1.0, 0.7, 0.2);
    vec3 vGamma = 1.0 + vHue * 0.6;
    vec3 vGain = vec3(0.9) + vHue * vHue * 8.0;
    vColor *= 1.5;
    float fMaxLum = 100.0;
    vColor /= fMaxLum;
    vColor = pow(vColor, vGamma);
    vColor *= vGain;
    vColor *= fMaxLum;
    return vColor;
}

vec3 ApplyPostFX(vec2 vUV, vec3 vInput) {
    vec2 vOffset = (vUV - 0.5) * sqrt(2.0);
    float fDist = dot(vOffset, vOffset);
    const float kStrength = 0.75;
    float fShade = mix(1.0, 1.0 - kStrength, fDist);
    vec3 vTemp = vInput * fShade;
    vTemp = vTemp * 2.0;
    vTemp = ColorGrade(vTemp);
    return Tonemap(vTemp);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 vUV = fragCoord.xy / iResolution.xy;
    vec4 vSample = textureLod(ChBufferC, vUV, 0.0);
    
    float fDepth = abs(vSample.w);
    vec3 vCameraPos = LoadVec3(addrCamera + offsetCameraPos);
    vec3 vCameraTarget = LoadVec3(addrCamera + offsetCameraTarget);
    
    vec3 vRayOrigin = vCameraPos;
    vec3 vRayDir = GetCameraRayDir(GetWindowCoord(vUV), vCameraPos, vCameraTarget);
    vec3 vWorldPos = vRayOrigin + vRayDir * fDepth;
    
    vec3 vPrevCameraPos = LoadVec3(addrPrevCamera + offsetCameraPos);
    vec3 vPrevCameraTarget = LoadVec3(addrPrevCamera + offsetCameraTarget);
    vec2 vPrevWindow = GetCameraWindowCoord(vWorldPos, vPrevCameraPos, vPrevCameraTarget);
    vec2 vPrevUV = GetUVFromWindowCoord(vPrevWindow);
    
    if (vSample.a < 0.0) {
        vPrevUV = vUV;
    }
    
    vec3 vResult = vec3(0.0);
    float fTot = 0.0;
    float fPlaneInFocus = length(vCameraPos - vCameraTarget);
    float fCoC = GetCoC(abs(fDepth), fPlaneInFocus);
    
    vResult.rgb = vSample.rgb * fCoC;
    fTot += fCoC;
    
    float fMotionBlurTaps = float(MOTION_BLUR_TAPS);
    float fGolden = 3.141592 * (3.0 - sqrt(5.0));
    
    for (int i = 1; i < MOTION_BLUR_TAPS; i++) {
        float f = float(i);
        vec2 vTapUV = mix(vUV, vPrevUV, f / fMotionBlurTaps - 0.5);
        
        float fRand = Hash(iTime + float(i) + vUV.x + vUV.y * 12.345);
        float fTheta = fRand * fGolden * fMotionBlurTaps;
        float fRadius = fCoC * sqrt(fRand * fMotionBlurTaps) / sqrt(fMotionBlurTaps);
        vTapUV += vec2(sin(fTheta), cos(fTheta)) * fRadius;
        
        vec4 vTapSample = textureLod(ChBufferC, vTapUV, 0.0);
        if (sign(vTapSample.a) == sign(vSample.a)) {
            float fCurrCoC = GetCoC(abs(vTapSample.a), fPlaneInFocus);
            float fWeight = fCurrCoC + 1.0;
            vResult += vTapSample.rgb * fWeight;
            fTot += fWeight;
        }
    }
    vResult /= fTot;
    
    vec3 vFinal = ApplyPostFX(vUV, vResult);
    fragColor = vec4(vFinal, 1.0);
}

// Workgroup size 
static const int WG_X = 16;
static const int WG_Y = 16;

#define ComputeFragColor(bfunc, id) \
    if(any(id.xy >= SCREEN_SIZE)) return; \
    vec4 fragColor; \
    bfunc(fragColor, vec2(id.xy) + 0.5);

#define BufferPass(bid, bfunc, id)  \
    ComputeFragColor(bfunc, id) \
    bid[PixelID(id.xy)] = fragColor; 

[shader("compute")] [numthreads(WG_X, WG_Y, 1)]
void BufferA(uint3 id : SV_DispatchThreadID) { 
    BufferPass(ChBufferA, mainImageA, id) 
}

[shader("compute")] [numthreads(WG_X, WG_Y, 1)]
void BufferB(uint3 id : SV_DispatchThreadID) { 
    BufferPass(ChBufferB, mainImageB, id) 
}

[shader("compute")] [numthreads(WG_X, WG_Y, 1)]
void BufferC(uint3 id : SV_DispatchThreadID) { 
    BufferPass(ChBufferC, mainImageC, id) 
}

[shader("compute")] [numthreads(WG_X, WG_Y, 1)]
void Image(uint3 id : SV_DispatchThreadID) {
    ComputeFragColor(mainImage, id)
    screen[ivec2(id.x, SCREEN_SIZE.y - 1 - id.y)] = pow(fragColor, vec4(1.0/2.2));
}
