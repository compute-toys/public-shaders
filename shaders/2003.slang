import std;

//import shadertoy
/* REMOVE AFTER SHADERTOY.SLANG FIX: https://github.com/compute-toys/compute.toys/pull/314 */

// GLSL types
public typealias vec2  = float2;    public typealias vec3  = float3;    public typealias vec4  = float4;
public typealias ivec2 = int2;      public typealias ivec3 = int3;      public typealias ivec4 = int4;
public typealias uvec2 = uint2;     public typealias uvec3 = uint3;     public typealias uvec4 = uint4;
public typealias bvec2 = bool2;     public typealias bvec3 = bool3;     public typealias bvec4 = bool4;
public typealias mat2  = float2x2;  public typealias mat3  = float3x3;  public typealias mat4  = float4x4;
public typealias mat2x2 = float2x2; public typealias mat2x3 = float2x3; public typealias mat2x4 = float2x4;
public typealias mat3x2 = float3x2; public typealias mat3x3 = float3x3; public typealias mat3x4 = float3x4;
public typealias mat4x2 = float4x2; public typealias mat4x3 = float4x3; public typealias mat4x4 = float4x4;

// GLSL functions
[Differentiable] public __generic<T> T mix(T a, T b, T c) where T : __BuiltinFloatingPointType { return a + (b - a) * c; }
[Differentiable] public __generic<T, int S> vector<T, S> mix(vector<T, S> a, vector<T, S> b, vector<T, S> c) where T : __BuiltinFloatingPointType { return a + (b - a) * c; }
[Differentiable] public __generic<T, int S0, int S1> matrix<T, S0, S1> mix(matrix<T, S0, S1> a, matrix<T, S0, S1> b, matrix<T, S0, S1> c) where T : __BuiltinFloatingPointType { return a + (b - a) * c; }

[Differentiable] public float fract(float a) { return frac(a); }
[Differentiable] public __generic<T, int S> vector<T, S> frac(vector<T, S> a) where T : __BuiltinFloatingPointType { return frac(a); }
[Differentiable] public __generic<T, int S0, int S1> matrix<T, S0, S1> frac(matrix<T, S0, S1> a) where T : __BuiltinFloatingPointType { return frac(a); }

[Differentiable] public __generic<T> T inversesqrt(T a) where T : __BuiltinFloatingPointType { return rsqrt(a); }
[Differentiable] public __generic<T, int S> vector<T, S> inversesqrt(vector<T, S> a) where T : __BuiltinFloatingPointType { return rsqrt(a); }
[Differentiable] public __generic<T, int S0, int S1> matrix<T, S0, S1> inversesqrt(matrix<T, S0, S1> a) where T : __BuiltinFloatingPointType { return rsqrt(a); }

[Differentiable] public __generic<T> T mod(T x, T y) where T : __BuiltinFloatingPointType { return x - no_diff floor(x / y) * y; }
[Differentiable] public __generic<T, int S> vector<T, S> mod(vector<T, S> x, T y) where T : __BuiltinFloatingPointType { return x - no_diff floor(x / y) * y; }
[Differentiable] public __generic<T, int S> vector<T, S> mod(vector<T, S> x, vector<T, S> y) where T : __BuiltinFloatingPointType { return x - no_diff floor(x / y) * y; }
[Differentiable] public __generic<T, int S0, int S1> matrix<T, S0, S1> mod(matrix<T, S0, S1> x, T y) where T : __BuiltinFloatingPointType { return x - no_diff floor(x / y) * y; }
[Differentiable] public __generic<T, int S0, int S1> matrix<T, S0, S1> mod(matrix<T, S0, S1> x, matrix<T, S0, S1> y) where T : __BuiltinFloatingPointType { return x - no_diff floor(x / y) * y; }

[Differentiable] public __generic<T> T sin(T x) where T : __BuiltinFloatingPointType { return sin(x); }
[Differentiable] public __generic<T> T cos(T x) where T : __BuiltinFloatingPointType { return cos(x); }

[Differentiable] public __generic<T> T atan(T y, T x) where T : __BuiltinFloatingPointType { return atan2(y, x); }
[Differentiable] public __generic<T, int S> vector<T, S> atan(vector<T, S> y, vector<T, S> x) where T : __BuiltinFloatingPointType { return atan2(y, x); }
[Differentiable] public __generic<T, int S0, int S1> matrix<T, S0, S1> atan(matrix<T, S0, S1> y, matrix<T, S0, S1> x) where T : __BuiltinFloatingPointType { return atan2(y, x); }

// HLSL/Slang doenst provide a way to get the inverse of a matrix, so we have to implement it ourselves
[Differentiable] float2x2 inverse(float2x2 m)
{
    float det = m._11 * m._22 - m._12 * m._21;
    float invD = 1.0 / det;
    return float2x2(m._22, -m._12,
                    -m._21, m._11) * invD;
}

[Differentiable] float3x3 inverse(float3x3 m)
{
    float3x3 inv;
    inv._11 = (m._22 * m._33 - m._23 * m._32);
    inv._12 = -(m._12 * m._33 - m._13 * m._32);
    inv._13 = (m._12 * m._23 - m._13 * m._22);

    inv._21 = -(m._21 * m._33 - m._23 * m._31);
    inv._22 = (m._11 * m._33 - m._13 * m._31);
    inv._23 = -(m._11 * m._23 - m._13 * m._21);

    inv._31 = (m._21 * m._32 - m._22 * m._31);
    inv._32 = -(m._11 * m._32 - m._12 * m._31);
    inv._33 = (m._11 * m._22 - m._12 * m._21);

    float det = m._11 * inv._11 + m._12 * inv._21 + m._13 * inv._31;
    return inv * (1.0 / det);
}

[Differentiable] float4x4 inverse(float4x4 m)
{
    float a00 = m._11, a01 = m._12, a02 = m._13, a03 = m._14;
    float a10 = m._21, a11 = m._22, a12 = m._23, a13 = m._24;
    float a20 = m._31, a21 = m._32, a22 = m._33, a23 = m._34;
    float a30 = m._41, a31 = m._42, a32 = m._43, a33 = m._44;

    float b00 = a00 * a11 - a01 * a10;
    float b01 = a00 * a12 - a02 * a10;
    float b02 = a00 * a13 - a03 * a10;
    float b03 = a01 * a12 - a02 * a11;
    float b04 = a01 * a13 - a03 * a11;
    float b05 = a02 * a13 - a03 * a12;
    float b06 = a20 * a31 - a21 * a30;
    float b07 = a20 * a32 - a22 * a30;
    float b08 = a20 * a33 - a23 * a30;
    float b09 = a21 * a32 - a22 * a31;
    float b10 = a21 * a33 - a23 * a31;
    float b11 = a22 * a33 - a23 * a32;

    float det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    float invD = 1.0 / det;

    float4x4 inv;
    inv._11 = (a11 * b11 - a12 * b10 + a13 * b09) * invD;
    inv._12 = (-a01 * b11 + a02 * b10 - a03 * b09) * invD;
    inv._13 = (a31 * b05 - a32 * b04 + a33 * b03) * invD;
    inv._14 = (-a21 * b05 + a22 * b04 - a23 * b03) * invD;

    inv._21 = (-a10 * b11 + a12 * b08 - a13 * b07) * invD;
    inv._22 = (a00 * b11 - a02 * b08 + a03 * b07) * invD;
    inv._23 = (-a30 * b05 + a32 * b02 - a33 * b01) * invD;
    inv._24 = (a20 * b05 - a22 * b02 + a23 * b01) * invD;

    inv._31 = (a10 * b10 - a11 * b08 + a13 * b06) * invD;
    inv._32 = (-a00 * b10 + a01 * b08 - a03 * b06) * invD;
    inv._33 = (a30 * b04 - a31 * b02 + a33 * b00) * invD;
    inv._34 = (-a20 * b04 + a21 * b02 - a23 * b00) * invD;

    inv._41 = (-a10 * b09 + a11 * b07 - a12 * b06) * invD;
    inv._42 = (a00 * b09 - a01 * b07 + a02 * b06) * invD;
    inv._43 = (-a30 * b03 + a31 * b01 - a32 * b00) * invD;
    inv._44 = (a20 * b03 - a21 * b01 + a22 * b00) * invD;

    return inv;
}

// HLSL/Slang matrices are transposed compared to GLSL matrices
[Differentiable] public __generic<int S> matrix<float, S, S> operator *(matrix<float, S, S> a, matrix<float, S, S> b) { return mul(b, a); }
[Differentiable] public __generic<int S0, int S1, int S2> matrix<float, S0, S2> operator *(matrix<float, S1, S2> a, matrix<float, S0, S1> b) { return mul(b, a); }
[Differentiable] public __generic<int S0, int S1> vector<float, S1> operator *(matrix<float, S0, S1> a, vector<float, S0> b) { return mul(b, a); }
[Differentiable] public __generic<int S0, int S1> vector<float, S0> operator *(vector<float, S1> a, matrix<float, S0, S1> b) { return mul(b, a); }

[Differentiable] public __generic<int S> void operator *=(inout matrix<float, S, S> a, matrix<float, S, S> b) { a = mul(b, a); }
[Differentiable] public __generic<int S> void operator *=(inout vector<float, S> a, matrix<float, S, S> b) { a = mul(b, a); }

public uint floatBitsToUint(float x) { return asuint(x); };
public int floatBitsToInt(float x) { return asint(x); };
public float uintBitsToFloat(uint x) { return asfloat(x); };
public float intBitsToFloat(int x) { return asfloat(x); };

public __generic<int S> vector<uint, S> floatBitsToUint(vector<float, S> x) { return asuint(x); }
public __generic<int S> vector<int, S> floatBitsToInt(vector<float, S> x) { return asint(x); }
public __generic<int S> vector<float, S> uintBitsToFloat(vector<uint, S> x) { return asfloat(x); }
public __generic<int S> vector<float, S> intBitsToFloat(vector<int, S> x) { return asfloat(x); }

public __generic<int S> vector<bool, S> lessThan(vector<float, S> x, vector<float, S> y) { return x < y; }
public __generic<int S> vector<bool, S> lessThanEqual(vector<float, S> x, vector<float, S> y) { return x <= y; }
public __generic<int S> vector<bool, S> greaterThan(vector<float, S> x, vector<float, S> y) { return x > y; }
public __generic<int S> vector<bool, S> greaterThanEqual(vector<float, S> x, vector<float, S> y) { return x >= y; }
public __generic<int S> vector<bool, S> equal(vector<float, S> x, vector<float, S> y) { return x == y; }
public __generic<int S> vector<bool, S> notEqual(vector<float, S> x, vector<float, S> y) { return x != y; }

public static const vec3 iResolution = vec3(SCREEN_SIZE, 1);
public static float iTime = time.elapsed;
public static float iTimeDelta = time.delta;
public static int iFrame = time.frame;
public static vec2 iMousePos = (mouse.pos.x == SCREEN_SIZE.x / 2 && mouse.pos.y == SCREEN_SIZE.y / 2) ? vec2(0, 0) : vec2(mouse.pos.x, SCREEN_SIZE.y - 1 - mouse.pos.y);
public static vec4 iMouse = vec4(iMousePos, (mouse.click == 1) ? iMousePos : 0.0);

public vec4 texelFetch(int channel, ivec2 coord, int lod) {
    return pass_in[ivec3(coord, channel)];
}

public vec4 textureLod(int channel, vec2 coord, float lod) {
    coord = coord*iResolution.xy - 0.5;
    ivec2 icoord = ivec2(floor(coord));
    vec2 fcoord = fract(coord);
    vec4 v00 = texelFetch(channel, icoord + ivec2(0,0), 0);
    vec4 v01 = texelFetch(channel, icoord + ivec2(0,1), 0);
    vec4 v10 = texelFetch(channel, icoord + ivec2(1,0), 0);
    vec4 v11 = texelFetch(channel, icoord + ivec2(1,1), 0);
    return mix(mix(v00, v01, fcoord.y), mix(v10, v11, fcoord.y), fcoord.x); 
}

public vec4 texture(int channel, vec2 coord) {
    return textureLod(channel, coord, 0.0);
}

public vec4 texelFetch(Texture2D<float4> tex, ivec2 coord, int lod) {
    return tex.Load(int3(coord, lod));
}

public vec4 textureLod(Texture2D<float4> tex, vec2 coord, float lod) {
    return tex.SampleLevel(bilinear, coord, lod);
}

public vec4 texture(Texture2D<float4> tex, vec2 coord) {
    return textureLod(tex, coord, 0.0);
}


// There are, however, some differences from Shadertoy:

// 1. All samples from iChannels that are mapped to Buffer A-D
//    must be replaced with ChBufferA-D
//    channel0/channel1 can be used normally, 
//    but note they are called channel0 not iChannel0
#define ChBufferA 0
#define ChBufferB 1
#define ChBufferC 2
#define ChBufferD 3

// 2. Globals must be static
static const vec3 someConst = vec3(0,2,4);

// 3. Compute shaders don't support screen derivatives, so dFdx/dFdy/fwidth won't work
//    They must be emulated, either analytically, or doing multiple passes
//    This also means that texture(ch, p) always samples the 0th MIP level
//    Technically can be fully emulated using subgroup operations and clever mapping,
//    but not supported here at the moment.

// 4. There are no sampler2D or other GLSL texture types
//    Those must be either replaced by an int (for the buffer index)
//    Or replaced with Texture2D<float4> which is the type of the input channel0/1


// MIT License. © 2020 munrocket

#define M_PI   3.1415926
#define M_PHI  1.6180339

/* quaternions */

vec4 qmult(vec4 p, vec4 q) {
    vec3 pv = p.xyz, qv = q.xyz;
    return vec4(p.w * qv + q.w * pv + cross(pv, qv), p.w * q.w - dot(pv, qv));
}

vec4 qrotor(vec3 axis, float phi) {
    return vec4(sin(phi*0.5) * axis, cos(phi*0.5));
}

vec4 qmouse(vec4 iMouse, vec3 iResolution, float iTime, float initRotation) {
    vec2 mouse = mix(vec2(0.5 - initRotation * sin(iTime), 0.5 - initRotation * cos(iTime)),
                                        iMouse.xy / iResolution.xy, step(0.0027, iMouse.y));
    vec4 rotY = qrotor(vec3(0., 1., 0.), M_PI - 2.*M_PI * mouse.x);
    vec4 rotX = qrotor(vec3(1., 0., 0.), M_PI * mouse.y - 0.5*M_PI);
    return qmult(rotY, rotX);
}

vec3 rotate(vec3 point, vec4 qrotor) {
    vec3 rv = qrotor.xyz;
    return qmult(qrotor, vec4(point * qrotor.w - cross(point, rv), dot(point, rv))).xyz;
}

float sdCapsule( vec3 p, vec3 a, vec3 b, float r )
{
  vec3 pa = p - a, ba = b - a;
  float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
  return length( pa - ba*h ) - r;
}

float opUnion( float d1, float d2, float k ) {
    float h = clamp( 0.5 + 0.5*(d2-d1)/k, 0.0, 1.0 );
    return mix( d2, d1, h ) - k*h*(1.0-h);
}

/* noise */

float mod289( float x ) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}
vec4 mod289( vec4 x ) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}
vec4 perm(vec4 x) {
    return mod289(((x * 34.0) + 1.0) * x);
}
float noise3(vec3 p) {
    vec3 a = floor(p);
    vec3 d = p - a;
    d = d * d * (3.0 - 2.0 * d);

    vec4 b = a.xxyy + vec4(0.0, 1.0, 0.0, 1.0);
    vec4 k1 = perm(b.xyxy);
    vec4 k2 = perm(k1.xyxy + b.zzww);

    vec4 c = k2 + a.zzzz;
    vec4 k3 = perm(c);
    vec4 k4 = perm(c + 1.0);

    vec4 o1 = fract(k3 * (1.0 / 41.0));
    vec4 o2 = fract(k4 * (1.0 / 41.0));

    vec4 o3 = o2 * d.z + o1 * (1.0 - d.z);
    vec2 o4 = o3.yw * d.x + o3.xz * (1.0 - d.x);

    return o4.y * d.y + o4.x * (1.0 - d.y);
}

#define MAX_STEPS 100.
#define MAX_DIST 2.5
#define MIN_STEP 0.001

static const float dihedIcos = 0.5 * acos (sqrt (5.) / 3.);
static const float pentaHeight = 1.5*acos(dot(normalize(vec3(M_PHI+1., 1., M_PHI)), normalize(vec3(.5))));
const float nn = 10. * 3.;
const float an = 6.2831853 / nn;
const float r0 = .7;
const float r1 = .16;
const float r2 = .008;
const float r3 = .005;

/// icosahedral symmetry https://www.shadertoy.com/view/MdXfWS
vec2 Rot2D (vec2 q, float a) {return q * cos (a) + q.yx * sin (a) * vec2 (-1., 1.);}
vec3 IcosSym (vec3 p) {
  const float dihedIcos = 0.5 * acos (sqrt (5.) / 3.);
  float a, w;
  w = 2. * M_PI / 3.;
  p.z = abs (p.z);
  p.yz = Rot2D (p.yz, - dihedIcos);
  p.x = - abs (p.x);
  for (int k = 0; k < 4; k ++) {
    p.zy = Rot2D (p.zy, - dihedIcos);
    p.y = - abs (p.y);
    p.zy = Rot2D (p.zy, dihedIcos);
    if (k < 3) p.xy = Rot2D (p.xy, - w);
  }
  p.z = - p.z;
  a = mod (atan (p.x, p.y) + 0.5 * w, w) - 0.5 * w;
  p.yx = vec2 (cos (a), sin (a)) * length (p.xy);
  p.x -= 2. * p.x * step (0., p.x);
  return p;
}
///

#define COUNT 3.
#define MARGIN 1.6

float map(vec3 p) {
    float cube = max(abs(p.x), max(abs(p.y), abs(p.z))) - .5 * COUNT * MARGIN;
    p.x = MARGIN * (fract(p.x / MARGIN - .5) - .5);
    
    p = IcosSym(p.xyz);
    p.yz = Rot2D (p.yz, -dihedIcos);
    
    float d = length(vec3(p.x, p.y, p.z + .65)) - .045;
    float d2 = sdCapsule(p,  vec3(0), vec3(0., 0., -.65), .02);
    
    
    p.xz = Rot2D (p.xz, pentaHeight);
    d = min(d, length(vec3(p.x, p.y, p.z + .60)) - .045);
    d2 = min(d2, sdCapsule(p,  vec3(0), vec3(0., 0., -.60), .02));
    
    d = min(d, length(p)-.45);
    d = opUnion(d2, d, 0.05);
    d = max(d, -length(p)+.40);
    
    d += noise3(p*70.0)*0.008;
    
    return max(cube, d);
}

vec3 march(vec3 camera, vec3 dir, out vec3 pos) {
    float dist, t = 0.;
    vec3 col;
    float zerolvl = 0.1;
    for(float i = 0.; i < MAX_STEPS; i++) {
      	pos = camera + t * dir;
        dist = 0.65 * map(pos);
        col += pow(0.002/max(0., dist + .65*zerolvl), 1.1)                 // accum
             * pow(1. - (t + .5 - length(camera)) / MAX_DIST, 6.)          // fade
             * mix(vec3(1.,.03,0.), vec3(.05,1.,.01), length(pos)*1.2-.2); //col
        t += max(MIN_STEP, dist);
        if(t > MAX_DIST) break;
    }
    return col;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
    
    vec2 uv = (2.*fragCoord - iResolution.xy) / min(iResolution.x, iResolution.y);
    vec3 dir = normalize(vec3(uv, -2.5));
    vec3 camera = vec3(0., 0.04+.04*cos(iTime), 1.65 - .3 * cos(iTime));
    
    vec4 mouse = qmouse(iMouse, iResolution, iTime, .1);
    dir = rotate(dir, mouse);
    camera = rotate(camera, mouse);
    
    vec3 pos, col = vec3(.005, .04, .08);
    col += march(camera, dir, pos) + 0.1 * noise3(70. * vec3(uv,0) + 100. * iTime);
    fragColor = vec4(pow(col, vec3(0.99)), .0);
}

// void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
//     fragColor = texture(ChBufferD, fragCoord / iResolution.xy);
// }

////////////////////////////////////////////////////////////
//Compute dispatches emulating fullscreen fragment shaders//
////////////////////////////////////////////////////////////

// 5. Unlike fragment shaders here you actually can control how the GPU spawns
//    the "pixel" threads.  Depending on the algorithm it might be preferrable 
//    to change the size of the workgroups

// Workgroup size 
static const int WG_X = 16;
static const int WG_Y = 16;

#define ComputeFragColor(bfunc, id) \
    if(any(id.xy >= SCREEN_SIZE)) return; \
    vec4 fragColor; \
    bfunc(fragColor, vec2(id.xy) + 0.5);

#define BufferPass(bid, bfunc, id)  \
    ComputeFragColor(bfunc, id) \
    pass_out[int3(id.xy, bid)] = fragColor; 

[shader("compute")] [numthreads(WG_X, WG_Y, 1)]
void Image(uint3 id : SV_DispatchThreadID) {
    ComputeFragColor(mainImage, id)
    screen[ivec2(id.x, SCREEN_SIZE.y - 1 - id.y)] = pow(fragColor, 2.2);
}