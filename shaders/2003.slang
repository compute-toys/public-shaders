import std;
import shadertoy;

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
    vec3 camera = vec3(0., 0.04, 1.9+.4*sin(iTime));
    
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