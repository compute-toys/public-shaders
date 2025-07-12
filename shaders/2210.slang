import std;
import shadertoy;

#define ChBufferA 0
#define ChBufferB 1
#define ChBufferC 2
#define ChBufferD 3

#define R iResolution.xy

#define LOAD(ch, pos) texelFetch(ch, ivec2(pos), 0)
#define LOAD3D(ch, pos) texelFetch(ch, ivec2(dim2from3(pos)), 0)

#define PI 3.1415926535
#define TWO_PI 6.28318530718

#define light_dir normalize(vec3(0.741,1.000,0.580))

#define initial_particle_density 1u
#define dt (isKeyPressed(KEY_P) ? 0.0 : 1.0)
#define rest_density 0.45
#define gravity 0.01

#define KERNEL_RADIUS 1.0
#define RENDER_KERNEL_RADIUS 1.0
#define PRESSURE 2.5
#define PRESSURE_RAD 0.8
#define VISCOSITY 0.65
#define SPIKE_KERNEL 0.5
#define SPIKE_RAD 0.75

#define force_mouse 0.005
#define force_mouse_rad 40.0
#define force_boundary 5.0
#define boundary_h 5.0
#define max_velocity 1.0
#define cooling 0.0

#define MAX_DIST 1e10
#define BLOCK_SIZE 4

#define IsoValue 0.25

//antialiasing
#define AA 1

#define RADIUS 0.3
#define NORMAL_SMOOTHNESS 0.0
#define FOV 2.5
#define MaxBounces 6
#define IOR 1.25
#define F0 0.005
#define Roughness 0.02

static const vec3 absorb = vec3(0.9,0.94,1.0);

//3d slice aspect ratio 
#define ar vec2(1.0,0.5)
static vec2 SCALE;
static vec2 size2d;
static vec3 size3d;

void InitGrid(vec2 iR)
{
    size2d = iR;
    SCALE = floor(ar*pow(iR.x*iR.y,0.1666666));
    size3d = vec3(floor(size2d.xy/SCALE)-2.0, SCALE.x*SCALE.y);
}

vec2 dim2from3(vec3 p3d)
{
    //p3d = clamp(p3d, vec3(0.0), size3d);
    float ny = float(uint(floor(p3d.z)) / uint(SCALE.x));
    float nx = floor(p3d.z) - ny*SCALE.x;
    return vec2(nx, ny)*vec2(size3d.xy) + p3d.xy;
}

vec3 dim3from2(vec2 p2d)
{
    vec2 tile = floor(p2d/size3d.xy);
    return vec3(p2d - size3d.xy*tile, tile.x + SCALE.x*tile.y);
}

bool InsideSimDomain(vec3 pos)
{
    return all(greaterThan(pos, vec3(0.0))) && all(lessThan(pos, size3d));
}

float sqr(float x)
{
    return x * x;
}

float cub(float x)
{
    return x*x*x;
}

float Pressure(float rho)
{
    return (rho/rest_density - 1.0)/max(rho*rho, 0.001);
}

float Gaussian(float r, float d)
{
    float norm = 1.0/(cub(d)*sqrt(cub(TWO_PI)));
    return norm * exp(-0.5*sqr(r/d));
}

vec3 GaussianGrad(vec3 dx, float d)
{
    float norm = 1.0/(cub(d)*sqrt(cub(TWO_PI)));
    float r = length(dx);
    return - (norm/sqr(d)) * exp(-0.5*sqr(r/d)) * dx;
}

#define GD(x, R) Gaussian(length(x),R)
#define GGRAD(x, R) GaussianGrad(x, R)
#define GS(x) exp(-dot(x,x))
#define DIR(phi) vec2(cos(TWO_PI*phi),sin(TWO_PI*phi))

#define loop(i,x) for(int i = 0; i < x; i++)
#define range(i,a,b) for(int i = a; i <= b; i++)

#define pixel(a, p, s) texture(a, (p+0.5)/vec2(s))

vec4 voxel(int ch, vec3 p3d)
{
    return pixel(ch, dim2from3(p3d), size2d);
}

//trilinearSample interpolation = linear interp between layers
vec4 trilinearSample(int ch, vec3 p3d)
{
    return mix(voxel(ch, vec3(p3d.xy, floor(p3d.z))),voxel(ch, vec3(p3d.xy, ceil(p3d.z))), fract(p3d.z));
}

float sdBox( vec3 p, vec3 b )
{
    vec3 d = abs(p) - b;
    return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}

vec2 hash21(float p)
{
	vec3 p3 = fract(vec3(p) * vec3(.1031, .1030, .0973));
	p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx+p3.yz)*p3.zy);
}

vec2 hash23(vec3 p3)
{
	p3 = fract(p3 * vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx+33.33);
    return fract((p3.xx+p3.yz)*p3.zy);
}

vec3 udir(vec2 rng)
{
    float phi = 2.*PI*rng.x;
    float ctheta = 2.*rng.y-1.;
    float stheta = sqrt(1.0-ctheta*ctheta);
    return vec3(cos(phi)*stheta, sin(phi)*stheta, ctheta);
}

struct Particle 
{
    uint mass;
    vec3 pos;
    vec3 vel;
    vec3 force;
    float density;
};

//5 bits for shared exponent, 9 bits for each component
uint packvec3(vec3 v)
{
    //get the exponent
    float maxv = max(abs(v.x), max(abs(v.y), abs(v.z)));
    int exp = clamp(int(ceil(log2(maxv))), -15, 15);
    float scale = exp2(-float(exp));
    uvec3 sv = uvec3(round(clamp(v*scale, -1.0, 1.0) * 255.0) + 255.0);
    uint packed = uint(exp + 15) | (sv.x << 5) | (sv.y << 14) | (sv.z << 23);
    return packed;
}

vec3 unpackvec3(uint packed)
{
    int exp = int(packed & 0x1Fu) - 15;
    vec3 sv = vec3((packed >> 5) & 0x1FFu, (packed >> 14) & 0x1FFu, (packed >> 23) & 0x1FFu);
    vec3 v = (sv - 255.0) / 255.0;
    v *= exp2(float(exp));
    return v;
}

uint packmasspos(uint mass, vec3 p0)
{
    uvec3 pos0 = uvec3(clamp(p0, 0.0, 1.0) * 255.0);
    return mass | (pos0.x << 8) | (pos0.y << 16) | (pos0.z << 24);
}

uint packMassPos(uint mass, vec3 pos)
{
    uvec3 pos0 = uvec3(clamp(pos, 0.0, 1.0) * 255.0); // Assuming pos range [0, 1] in a cell
    uint data1 = mass | (pos0.x << 8) | (pos0.y << 16) | (pos0.z << 24);
    return data1;
}

void unpackMassPos(uint packed, out uint mass, out vec3 pos)
{
    mass = packed & 0xFFu;
    uvec3 pos0 = uvec3((packed >> 8) & 0xFFu, (packed >> 16) & 0xFFu, (packed >> 24) & 0xFFu);
    pos = vec3(pos0) / 255.0;
}

vec4 packParticles(Particle p0, Particle p1, vec3 pos)
{
    p0.pos -= pos;
    p1.pos -= pos;

    uvec4 data = uvec4(
        packMassPos(p0.mass, p0.pos),
        packMassPos(p1.mass, p1.pos),
        packvec3(p0.vel),
        packvec3(p1.vel)
    );

    return uintBitsToFloat(data);
}

void unpackParticles(vec4 packed, vec3 pos, out Particle p0, out Particle p1)
{
    uvec4 data = floatBitsToUint(packed);
    
    unpackMassPos(data.x, p0.mass, p0.pos);
    unpackMassPos(data.y, p1.mass, p1.pos);

    p0.pos += pos;
    p1.pos += pos;

    p0.vel = unpackvec3(data.z);
    p1.vel = unpackvec3(data.w);
}

float sdBox( in vec2 p, in vec2 b )
{
    vec2 d = abs(p)-b;
    return length(max(d,0.0)) + min(max(d.x,d.y),0.0);
}

int ClosestCluster(Particle p0, Particle p1, Particle incoming)
{
    //first try to choose the particle with significantly smaller mass
    if(float(p0.mass) < 0.01*float(p1.mass) || float(p1.mass) < 0.01*float(p0.mass))
    {
        return p0.mass < p1.mass ? 0 : 1;
    }

    //otherwise choose the closest one
    float d0 = length(p0.pos - incoming.pos);
    float d1 = length(p1.pos - incoming.pos);
    return d0 < d1 ? 0 : 1;
}

void BlendParticle(inout Particle p, in Particle incoming)
{
    uint newMass = p.mass + incoming.mass;
    vec2 weight = vec2(p.mass, incoming.mass) / float(newMass);
    p.pos = p.pos*weight.x + incoming.pos*weight.y;
    p.vel = p.vel*weight.x + incoming.vel*weight.y;
    p.mass = newMass;
}

void Clusterize(inout Particle p0, inout Particle p1, in Particle incoming, vec3 pos, float timeStep)
{
    if(incoming.mass == 0u) return;
    //move incoming particle forwards
    incoming.pos += incoming.vel*timeStep;
    //check if the incoming particle is in the cell
    if(!all(equal(pos, floor(incoming.pos)))) return;
    
    int closest = ClosestCluster(p0, p1, incoming);
    if(closest == 0)
    {
        BlendParticle(p0, incoming);
    }
    else
    {
        BlendParticle(p1, incoming);
    }
}

void SplitParticle(inout Particle p1, inout Particle p2)
{
    vec3 pos = p1.pos;
    uint newMass = p1.mass;
    p1.mass = newMass/2u;
    p2.mass = newMass - p1.mass;
    vec3 dir = udir(hash23(pos));
    p1.pos = pos - dir*5e-3;
    p2.pos = pos + dir*5e-3;
    p2.vel = p1.vel;
}

void ApplyForce(inout Particle p, in Particle incoming)
{
    if(incoming.mass == 0u) return;

    vec3 dx = incoming.pos - p.pos;
    vec3 dv = incoming.vel - p.vel;
    float d = length(dx);
    vec3 dir = dx / max(d, 1e-3);
    
    float rho0 = p.density;
    float rho1 = incoming.density;
    float mass0 = float(p.mass);
    float mass1 = float(incoming.mass);
  
    vec3 ggrad = GGRAD(dx, PRESSURE_RAD);
    float pressure = 0.5*p.density*(Pressure(p.density) + Pressure(incoming.density));
    vec3 F_SPH = - PRESSURE * pressure * ggrad;
    vec3 F_VISC = VISCOSITY * dot(dir, dv) * ggrad;
    vec3 F_SPIKE = SPIKE_KERNEL * GD(d, SPIKE_RAD) * dir;
   
    p.force += - (F_SPH + F_VISC + F_SPIKE) * mass1;
}

float minv(vec3 a)
{
    return min(min(a.x, a.y),a.z);
}

float maxv(vec3 a)
{
    return max(max(a.x, a.y),a.z);
}

float distance2border(vec3 p)
{
    vec3 a = vec3(size3d - 1.) - p;
    return min(minv(p),minv(a)) + 1.;
}

vec4 border_grad(vec3 p)
{
    const float dx = 0.001;
    const vec3 k = vec3(1,-1,0);
	return  (k.xyyx*distance2border(p + k.xyy*dx) +
			 k.yyxx*distance2border(p + k.yyx*dx) +
			 k.yxyx*distance2border(p + k.yxy*dx) +
			 k.xxxx*distance2border(p + k.xxx*dx))/vec4(4.*dx,4.*dx,4.*dx,4.);
}

void IntegrateParticle(inout Particle p, vec3 pos, float time, float timeStep)
{
    p.force = p.force;/// max(0.0001, float(p.mass));
    p.force += gravity*vec3(0.4*sin(1.5*time), 0.2*cos(0.75*time), -1.0); //gravity
    

    vec4 border = border_grad(p.pos);
    vec3 bound =1.*normalize(border.xyz)*exp(-0.4*border.w*border.w);
    p.force += force_boundary * bound;
    p.vel += p.force * timeStep;

    //velocity limit
    float v = length(p.vel)/max_velocity;
    p.vel /= (v > 1.)?v:1.;
}

vec3 hsv2rgb( in vec3 c )
{
    vec3 rgb = clamp( abs(mod(c.x*6.0+vec3(0.0,4.0,2.0),6.0)-3.0)-1.0, 0.0, 1.0 );

	rgb = rgb*rgb*(3.0-2.0*rgb); // cubic smoothing	

	return c.z * mix( vec3(1.0), rgb, c.y);
}

//Undefine to only use 2 newton steps
//#define FULL_PRECISION

float EvalCubic(float x, vec4 k)
{
    return ((k.x * x + k.y) * x + k.z) * x + k.w;
}

float EvalQuadratic(float x, vec3 k)
{
    return (k.x * x + k.y) * x + k.z;
}

float NewtonIteration(float x, vec4 k3, vec3 k2)
{
    return x - EvalCubic(x, k3) / EvalQuadratic(x, k2);
}

float LinearSolve(float x1, float x2, float y1, float y2)
{
    return x1 + (x2 - x1) * (0.0 - y1) / (y2 - y1);
}

vec2 SolveSegment(vec4 k3, vec3 k2, float mint, float maxt, float c0, float c1)
{
    float x = LinearSolve(mint, maxt, c0, c1);
    
#ifdef FULL_PRECISION
    //usually you dont need more than 1 newton step for actual surface inside the voxel, but still can get weird artifacts
    for(int i = 0; i < 16; i++) 
    {
        float xprev = x;
        x = clamp(NewtonIteration(x, k3, k2), mint, maxt);
        x = NewtonIteration(x, k3, k2);
        if(abs(x-xprev) < 1e-2) break;
    }
#else
    x = clamp(NewtonIteration(x, k3, k2), mint, maxt);
    x = NewtonIteration(x, k3, k2);
    //x = NewtonIteration(x, k3, k2);
#endif
    
    return vec2(x, 1.0);
}

vec2 solveCubicFirstRootApprox(vec4 k3, float mint, float maxt, float cmin, float cmax)
{
    vec3 k2 = vec3(3.0 * k3.x, 2.0 * k3.y, k3.z); // 3ax^2 + 2bx + c, derivative of cubic
    float D = k2.y * k2.y - 4.0 * k2.x * k2.z;
    vec4 segment = vec4(0.0);

    if(D <= 0.0) { // one potential segment with root
        if(cmin * cmax > 0.0) return vec2(0.0); // no root in this segment
        segment = vec4(mint, maxt, cmin, cmax);
    } else { // 3 potential segments with roots
        float r = 1.0 / (2.0 * k2.x);
        float b = sqrt(D) * r;
        float xa = - k2.y * r - b;
        float xb = - k2.y * r + b;
        float xmin = clamp(min(xa, xb), mint, maxt);
        float xmax = clamp(max(xa, xb), mint, maxt);

        float cmin0 = EvalCubic(xmin, k3);
        float cmax0 = EvalCubic(xmax, k3);
        
        if(cmin * cmin0 < 0.0) segment = vec4(mint, xmin, cmin, cmin0); //root in first segment
        else if(cmin0 * cmax0 < 0.0) segment = vec4(xmin, xmax, cmin0, cmax0); //root in second segment
        else if(cmax0 * cmax < 0.0) segment = vec4(xmax, maxt, cmax0, cmax); //root in third segment
        else return vec2(0.0); //no roots in any segment
    }

    return SolveSegment(k3, k2, segment.x, segment.y, segment.z, segment.w);
}

vec2 solveCubicFirstRootApprox(vec4 k3, float mint, float maxt)
{
    float cmin = EvalCubic(mint, k3);
    float cmax = EvalCubic(maxt, k3);
    return solveCubicFirstRootApprox(k3, mint, maxt, cmin, cmax);
}

bool isSameSign(vec4 ys)
{
    return ys.x*ys.w > 0.0;
    bvec4 sgn = greaterThan(ys, vec4(0.0));
    return (sgn.x == sgn.w) && (sgn.x == sgn.z) && (sgn.x == sgn.w);
}

// Intersection from 4 trilinearSample samples along the ray
vec2 iIsoSurf4Samples(float mint, float deltat, vec4 ys)
{
    if(isSameSign(ys)) return vec2(0.0);
    //get cubic from 4 samples along ray (at equal intervals)
    float t1 = 3.0 * ys.y + ys.w;
    float t2 = 3.0 * ys.z + ys.x;
    float c3 = (t1 - t2) / 6.0;
    float c2 = (ys.z + ys.x) / 2.0 - ys.y;
    float c1 = (ys.z - ys.x) / 2.0 - c3;
    float c0 = ys.y;
    vec2 sol = solveCubicFirstRootApprox(vec4(c3, c2, c1, c0), -1.0, 2.0, ys.x, ys.w);
    sol.x = (sol.x + 1.0) * deltat + mint; //remap sol.x to [mint, maxt]
    return sol;
}


mat3 getCamera(vec2 angles)
{
   mat3 theta_rot = mat3(1,   0,              0,
                          0,  cos(angles.y),  -sin(angles.y),
                          0,  sin(angles.y),  cos(angles.y)); 
        
   mat3 phi_rot = mat3(cos(angles.x),   sin(angles.x), 0.,
        		       -sin(angles.x),   cos(angles.x), 0.,
        		        0.,              0.,            1.); 
        
   return theta_rot * phi_rot;
}

vec2 iBox( in vec3 ro, in vec3 rd, in vec3 boxSize ) 
{
    vec3 m = sign(rd)/max(abs(rd), 1e-8);
    vec3 n = m*ro;
    vec3 k = abs(m)*boxSize;
	
    vec3 t1 = -n - k;
    vec3 t2 = -n + k;

	float tN = max( max( t1.x, t1.y ), t1.z );
	float tF = min( min( t2.x, t2.y ), t2.z );
	
    if (tN > tF || tF <= 0.) {
        return vec2(MAX_DIST);
    } else {
        return vec2(tN, tF);
    }
}

struct VoxelRayProps
{
    vec3 rayInv;
    vec3 rayMask;
    vec3 rayStep;
};

VoxelRayProps CreateVoxelRayProps(vec3 rd)
{
    VoxelRayProps props;
    props.rayInv = sign(rd) / max(abs(rd), vec3(1e-6));
    props.rayMask = step(vec3(0.0), rd);
    props.rayStep = props.rayMask * 2.0 - 1.0;
    return props;
}

struct VoxelRay
{
    vec3 voxelPos;
    vec3 offset;
    float curTraveled;
    VoxelRayProps props;
};

VoxelRay CreateVoxelRay(vec3 ro, VoxelRayProps props)
{
    VoxelRay ray;
    ray.voxelPos = floor(ro);
    ray.offset = (props.rayMask - ro - props.rayStep*3e-6) * props.rayInv;
    ray.curTraveled = 0.0;
    ray.props = props;
    return ray;
}

vec4 ComputeNextVoxel(in VoxelRay ray)
{
    vec3 planeDistances = ray.voxelPos * ray.props.rayInv + ray.offset;
    float nextDistance = min(min(planeDistances.x, planeDistances.y), planeDistances.z); 
    vec3 nextVoxel = ray.voxelPos + vec3(equal(planeDistances, vec3(nextDistance))) * ray.props.rayStep;
    return vec4(nextVoxel, nextDistance);
}

void StepVoxelRay(inout VoxelRay ray, vec4 next)
{
    ray.voxelPos = next.xyz;
    ray.curTraveled = next.w;
}

float fresnelFull(vec3 Refl, vec3 Refr, vec3 Norm, float n1, float n2)
{
    float cosRefl = abs(dot(Refl, Norm));
    float cosRefr = abs(dot(Refr, Norm));
    float Rpar = (n1*cosRefl - n2*cosRefr) / (n1*cosRefl + n2*cosRefr);
    float Rper = (n1*cosRefr - n2*cosRefl) / (n1*cosRefr + n2*cosRefl);
    return 0.5*(Rpar*Rpar + Rper*Rper);
}

float NDF_ggx(vec3 m, vec3 n, float alpha)
{
    float alpha2 = alpha*alpha; 
    return alpha2/(PI*sqr( sqr(max(dot(n,m), 0.)) * (alpha2 - 1.0) + 1.0 ));
}

float G_ggx(float NdotV, float alpha)
{
    float alpha2 = alpha*alpha;
    return 2.0*NdotV/(NdotV + sqrt( mix(NdotV*NdotV, 1.0, alpha2) ));
}

static const int KEY_SPACE = 32;
static const int KEY_LEFT  = 37;
static const int KEY_UP    = 38;
static const int KEY_RIGHT = 39;
static const int KEY_DOWN  = 40;
static const int KEY_P = 80;

bool isKeyPressed(int KEY)
{
	return keyDown(KEY);
}

void mainImageA( out vec4 fragColor, in vec2 fragCoord )
{
    InitGrid(iResolution.xy);
    fragCoord = floor(fragCoord);
    vec3 pos = dim3from2(fragCoord);
    
    Particle p0, p1;

    //advect neighbors and accumulate + clusterize density if they fall into this cell
    range(i, -1, 1) range(j, -1, 1) range(k, -1, 1)
    {
        //load the particles 
        vec3 pos1 = pos + vec3(i, j, k);
        if(!all(lessThanEqual(pos1, size3d)) || !all(greaterThanEqual(pos1, vec3(0.0))))
        {
            continue;
        }
        Particle p0_, p1_;
        unpackParticles(LOAD3D(ChBufferC, pos1), pos1, p0_, p1_);
        
        Clusterize(p0, p1, p0_, pos, dt);
        Clusterize(p0, p1, p1_, pos, dt);
    }
    
    if(p1.mass == 0u && p0.mass > 0u)
    {
        SplitParticle(p0, p1);
    }

    if(p0.mass == 0u && p1.mass > 0u)
    {
        SplitParticle(p1, p0);
    }
    
    vec4 packed = packParticles(p0, p1, pos);
    fragColor = packed;
}

void AddDensity(inout Particle p, in Particle incoming, float rad)
{
    if(incoming.mass == 0u) return;
    float d = distance(p.pos, incoming.pos);
    float mass = float(incoming.mass);
    p.density += mass*GD(d,rad);
}

//compute particle SPH densities
void mainImageB( out vec4 fragColor, in vec2 fragCoord )
{
    InitGrid(iResolution.xy);
    fragCoord = floor(fragCoord);
    vec3 pos = dim3from2(fragCoord);
    
    Particle p0, p1, pV;
    pV.pos = pos + 0.5;
    
    //load the particles
    vec4 packed = LOAD3D(ChBufferA, pos);
    unpackParticles(packed, pos, p0, p1);
    
    range(i, -2, 2) range(j, -2, 2) range(k, -2, 2)
    {
        int dist = i*i + j*j + k*k;
        if(dist == 0 || dist > 16) continue;
        vec3 pos1 = pos + vec3(i, j, k);
        Particle p0_, p1_;
        unpackParticles(LOAD3D(ChBufferA, pos1), pos1, p0_, p1_);

        if(p0.mass > 0u)
        {
            AddDensity(p0, p0_, KERNEL_RADIUS);
            AddDensity(p0, p1_, KERNEL_RADIUS);
        }
        if(p1.mass > 0u)
        {
            AddDensity(p1, p0_, KERNEL_RADIUS);
            AddDensity(p1, p1_, KERNEL_RADIUS);
        }
        
        AddDensity(pV, p0_, RENDER_KERNEL_RADIUS);
        AddDensity(pV, p1_, RENDER_KERNEL_RADIUS);
    }

    if(p0.mass > 0u)
    {
        AddDensity(p0, p0, KERNEL_RADIUS);
        AddDensity(p0, p1, KERNEL_RADIUS);
    }
    if(p1.mass > 0u)
    {
        AddDensity(p1, p0, KERNEL_RADIUS);
        AddDensity(p1, p1, KERNEL_RADIUS);
    }
    AddDensity(pV, p0, RENDER_KERNEL_RADIUS);
    AddDensity(pV, p1, RENDER_KERNEL_RADIUS);
    
    if(any(lessThan(pos, vec3(1.0))) || any(greaterThan(pos, size3d - 2.0))) pV.density = 0.0;

    fragColor = vec4(p0.density, p1.density, pV.density, 0.0);
}

#define EMITTER_POS vec3(0.1,0.5,0.5)
#define EMITTER_RAD 4.0
#define EMITTER_VEL vec3(1.0, 0.0, 0.0)
#define EMITTER_NUM 1

#define VOID_POS vec3(0.8,0.5,0.1)
#define VOID_RAD 12.0

void mainImageC( out vec4 fragColor, in vec2 fragCoord )
{
    InitGrid(iResolution.xy);
    fragCoord = floor(fragCoord);
    vec3 pos = dim3from2(fragCoord);

    Particle p0, p1;
    
    //load the particles
    vec4 packed = LOAD3D(ChBufferA, pos);
    unpackParticles(packed, pos, p0, p1);
    
    //load density
    vec2 densities = voxel(ChBufferB, pos).xy;
    p0.density = densities.x;
    p1.density = densities.y;
    
    if(p0.mass + p1.mass > 0u) 
    {
        range(i, -2, 2) range(j, -2, 2) range(k, -2, 2)
        {
            int dist = i*i + j*j + k*k;
            if(dist == 0 || dist >= 6) continue;
            vec3 pos1 = pos + vec3(i, j, k);
            Particle p0_, p1_;
            unpackParticles(LOAD3D(ChBufferA, pos1), pos1, p0_, p1_);
            
            vec2 densities_ = voxel(ChBufferB, pos1).xy;
            p0_.density = densities_.x;
            p1_.density = densities_.y;

            //apply the force
            ApplyForce(p0, p0_);
            ApplyForce(p0, p1_);
            ApplyForce(p1, p0_);
            ApplyForce(p1, p1_);
        }

        ApplyForce(p0, p1);
        ApplyForce(p1, p0);

        IntegrateParticle(p0, pos, iTime, dt);
        IntegrateParticle(p1, pos, iTime, dt);
        
        if(isKeyPressed(KEY_UP))
        {
            float void_d = distance(p0.pos, size3d*VOID_POS);
            if(void_d < VOID_RAD)
            {
                p0.mass = 0u;
            }
        }
    
        if(!isKeyPressed(KEY_LEFT))
        {
            vec3 dx = normalize(p0.pos - size3d*0.5);
            p0.vel += vec3(dx.y, -dx.x, 0.0)*0.003;
        }
        
        if(isKeyPressed(KEY_RIGHT))
        {
            vec3 dx = normalize(p0.pos - size3d*0.5);
            p0.vel += vec3(-dx.y, dx.x, 0.0)*0.003;
        }
    }
    
        
    if(iFrame < 10)
    {
        if(pos.x < 0.4*size3d.x && pos.x > 0.0*size3d.x && 
           pos.y < 0.85*size3d.y && pos.y > 0.15*size3d.y &&
           pos.z < 0.85*size3d.z && pos.z > 0.15*size3d.z)
        {
            p0.mass = initial_particle_density;
            p1.mass = 0u;
        }

        p0.pos = pos;
        p0.vel = vec3(0.0);
        p1.pos = pos;
        p1.vel = vec3(0.0);
    }

    if(all(equal(p0.pos, p1.pos)))
    {
        p1.pos += 1e-2;
    }
    
    // if(isKeyPressed(KEY_SPACE))
    if(isKeyPressed(KEY_DOWN))
    {
    	float emitter_d = distance(pos, size3d*EMITTER_POS);
        if(emitter_d < EMITTER_RAD && int(pos.y) % 2 == 0 && int(pos.z) % 2 == 0 && int(pos.x) % 2 == 0)
        {
            Particle emit;
            emit.pos = pos;
            emit.mass = 1u;
            emit.vel = EMITTER_VEL;
            
            BlendParticle(p0, emit);
        }
    }
    
        
    packed = packParticles(p0, p1, pos);
    fragColor = packed;
}

#define SHADOWS 
//#define REFRACTION

vec3 getRay(vec2 angles, vec2 pos)
{
    mat3 camera = getCamera(angles);
    return normalize(transpose(camera)*vec3(FOV*pos.x, 1., FOV*pos.y));
}


struct Ray 
{
    vec3 ro;
    vec3 rd;
    float td;
    vec3 normal;
    vec3 color;
};

void iSphere(inout Ray ray, vec4 sphere, vec3 color)
{
    vec3 ro = ray.ro - sphere.xyz;
    float b = dot(ro, ray.rd);
    float c = dot(ro, ro) - sphere.w*sphere.w;
    float h = b*b - c;
    if (h > 0.) 
    {
	    h = sqrt(h);
        float d1 = -b-h;
        float d2 = -b+h;
        if (d1 >= 0.0 && d1 <= ray.td) {
            ray.normal = normalize(ro + ray.rd*d1);
            ray.color = color;
            ray.td = d1;
        } else if (d2 >= 0.0 && d2 <= ray.td) { 
            ray.normal = normalize(ro + ray.rd*d2); 
            ray.color = color;
            ray.td = d2;
        }
    }
}

void TraceCell(inout Ray ray, vec3 p)
{
    //load the particles 
    vec4 packed = LOAD3D(ChBufferA, p);
    Particle p0, p1;
    unpackParticles(packed, p, p0, p1);

    if(p0.mass > 0u) iSphere(ray, vec4(p0.pos, RADIUS*1.5), vec3(1.000,1.000,1.000));
    if(p1.mass > 0u) iSphere(ray, vec4(p1.pos, RADIUS*1.5), vec3(1.000,1.000,1.000));
}

void TraceCells(inout Ray ray, vec3 p)
{
    vec3 p0 = floor(p);
    vec4 rho = voxel(ChBufferB, p);
    if(rho.z < 1e-3) return;
    range(i, -1, 1) range(j, -1, 1) range(k, -1, 1)
    {
        //load the particles 
        vec3 p1 = p0 + vec3(i, j, k);
        TraceCell(ray, p1);
    }
}


float Density(vec3 p)
{
    return trilinearSample(ChBufferB, p).z;
}

// float Shadow(vec3 p)
// {
//     float optical_density = trilinearSample(ch2, p).x;
//     return exp(-optical_density)+0.05;
// }

vec4 calcNormal(vec3 p, float dx) {
	const vec3 k = vec3(1,-1,0);
	return   (k.xyyx*Density(p + k.xyy*dx) +
			 k.yyxx*Density(p + k.yyx*dx) +
			 k.yxyx*Density(p + k.yxy*dx) +
			 k.xxxx*Density(p + k.xxx*dx))/vec4(4.*dx,4.*dx,4.*dx,4.);
}

float TraceDensity(vec3 ro, vec3 rd)
{
    float dens = 0.0;
    float td = 0.0;
    for(int i = 0; i < 100; i++)
    {
        vec3 p = ro + rd * td;
        if(any(lessThan(p, vec3(1.0))) || any(greaterThan(p, size3d - 1.0))) return dens;
        float d = Density(p);
        dens += d * 2.0;
        td += 2.0;
    }
    return dens;
}


#define ISO_VALUE 0.5
float ParticleDensity(vec3 p)
{
    vec3 p0 = floor(p);
    float rho = voxel(ChBufferB, p).z;
    //if(rho < 0.001) return rho;
    //rho = 0.0;
    ////if larger then compute accurate density from particles
    //range(i, -1, 1) range(j, -1, 1) range(k, -1, 1)
    //{
    //    //load the particles 
    //    vec3 p1 = p0 + vec3(i, j, k);
    //    //load the particles
    //    vec4 packed = LOAD3D(ch0, p1);
    //    Particle p0_, p1_;
    //    unpackParticles(packed, p1, p0_, p1_);
    //    if(p0_.mass > 0u) rho += float(p0_.mass)*GD(length(p0_.pos - p), RADIUS);
    //    if(p1_.mass > 0u) rho += float(p1_.mass)*GD(length(p1_.pos - p), RADIUS);
    //}

    return rho;
}

vec3 ParticleGradient(vec3 p)
{
    vec3 p0 = floor(p);
    vec3 grad = vec3(0.0);

    range(i, -1, 1) range(j, -1, 1) range(k, -1, 1)
    {
        //load the particles 
        vec3 p1 = p0 + vec3(i, j, k);

        //load the particles
        vec4 packed = LOAD3D(ChBufferA, p1);
        Particle p0_, p1_;

        unpackParticles(packed, p1, p0_, p1_);

        if(p0_.mass > 0u) grad += float(p0_.mass)*GGRAD(p0_.pos - p, RADIUS);
        if(p1_.mass > 0u) grad += float(p1_.mass)*GGRAD(p1_.pos - p, RADIUS);
    }

    return grad;
}


float DE(vec3 p)
{
    return ISO_VALUE - ParticleDensity(p);
}

float TraceIsoSurface(Ray ray, float mint, float inside)
{
    const int step_count = 300;
    float td = mint;
    for(int i = 0; i < step_count; i++)
    {
        vec3 p = ray.ro + ray.rd * td;
        if(!all(lessThanEqual(p, size3d)) || !all(greaterThanEqual(p, vec3(0.))))
        {
            return ray.td;
        }
        float d = inside*4.0*DE(p);
        if(d < 0.0)
        {
            return td;
        }
        td += d;
    }
    return td;
}

float2 dirToLatLongUV(float3 d)
{
    float3 n = normalize(d);
    float  u = atan2(n.y, n.x) * (1.0 / (2.0 * PI)) + 0.5;
    float  v = 0.5 - asin(n.z) / PI;
    return float2(u, v);
}

float3 sampleSky(float3 dir, float lod)
{
    float2 uv = dirToLatLongUV(dir);
    return channel0.SampleLevel(trilinear, uv, clamp(lod, 0.0, 5.0)).rgb;
}

vec3 Background(vec3 rd)
{
    return sampleSky(rd, 0.0).xyz;
}

void mainImage( out vec4 col, in vec2 fragCoord )
{    
    InitGrid(iResolution.xy);
    
    vec2 uv = (fragCoord - 0.5*R)/max(R.x, R.y);

    vec2 angles = vec2(2.*PI, PI)*(iMouse.xy/iResolution.xy - 0.5);

    if(iMouse.z <= 0.)
    {
        angles = vec2(0.04, -0.5);
    }
    vec3 rd = getRay(angles, uv);
    vec3 center_rd = getRay(angles, vec2(0.));
 
    float d = sqrt(dot(vec3(size3d), vec3(size3d)))*0.5;
    vec3 ro = vec3(size3d)*vec3(0.5, 0.5, 0.5) - center_rd*d;
    
    

    vec2 tdBox = iBox(ro - vec3(size3d)*0.5, rd, 0.5*vec3(size3d));
    col.xyz =Background(rd);
    if(tdBox.x < MAX_DIST)
    {
        float td = max(tdBox.x+0.5, 0.0);
        Ray ray;
        ray.ro = ro;
        ray.rd = rd;
        ray.td = tdBox.y;
        int i = 0;
        for(; i < 200; i++)
        {
            vec3 p = ro + rd*td;
            TraceCells(ray, p);
            td += 2.5;
            if(td > tdBox.y-1.0)
            {
                break;
            }
            if(ray.td < tdBox.y-1.0)
            {
                break;
            }
        }
        
        //col.xyz = vec3(i)/200.0;
        //return;
        
        //float liq_td = TraceIsoSurface(ray, td);
        //ray.td = min(liq_td, ray.td);
        
        if(ray.td < tdBox.y)
        {
            vec3 p0 = ray.ro + ray.rd*ray.td;
            vec3 normal = normalize(-ParticleGradient(p0));
            vec3 normal1 = normalize(calcNormal(p0, 0.5).xyz);
            normal = -normalize(mix(normal, normal1, NORMAL_SMOOTHNESS));
            vec3 albedo = vec3(0.039,0.153,1.000);
            float LdotN = 0.5*dot(normal, light_dir)+0.5;
            #ifdef SHADOWS
                float shadow_d = TraceDensity(p0+light_dir*1.0, light_dir);
                float shadow = exp(-shadow_d) + 0.3*exp(-0.1*shadow_d);
            #else
                float shadow = 1.0;
            #endif
            vec3 refl_d = reflect(ray.rd, normal);
            vec3 refl = Background(refl_d);
            float K = 1. - pow(max(dot(normal,refl_d),0.), 2.);
            K = mix(0.0, K, 0.1);
            
            #ifdef REFRACTION
            vec3 refr_d = refract(ray.rd, normal, 1.0/1.33);
            
            float liquid_density = TraceDensity(p0, refr_d);
            
            vec3 liquid_color = exp(-0.5*liquid_density*vec3(0.953,0.353,0.247));
            vec3 refr_color = Background(refr_d) * liquid_color;
            col.xyz = 2.5*shadow*refr_color*(1.0 - K) + 0.*ray.color + 0.75*shadow*refl*K;
            #else
            col.xyz = 2.5*shadow*albedo*LdotN*(1.0 - K) + 0.*ray.color + 0.75*shadow*refl*K;
            #endif
        }
        
        //col.xyz = 0.01*vec3(1,1,1)*TraceDensity(ro + rd*max(tdBox.x+0.001,0.0), rd);
    }
    
    col.xyz = 1.0 - exp(-2.5*pow(col.xyz,vec3(1.0/1.4)));
}

////////////////////////////////////////////////////
////////////////////////////////////////////////////
////////////////////////////////////////////////////

[shader("compute")]
[numthreads(16, 16, 1)]
void BufferA(uint3 id : SV_DispatchThreadID) {
    vec2 fragCoord = vec2(id.xy) + 0.5;
    vec4 fragColor;
    mainImageA(fragColor, fragCoord);
    pass_out[int3(id.xy, 0)] = fragColor;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void BufferB(uint3 id : SV_DispatchThreadID) {
    vec2 fragCoord = vec2(id.xy) + 0.5;
    vec4 fragColor;
    mainImageB(fragColor, fragCoord);
    pass_out[int3(id.xy, 1)] = fragColor;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void BufferC(uint3 id : SV_DispatchThreadID) {
    vec2 fragCoord = vec2(id.xy) + 0.5;
    vec4 fragColor;
    mainImageC(fragColor, fragCoord);
    pass_out[int3(id.xy, 2)] = fragColor;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void Image(uint3 id : SV_DispatchThreadID) {
    vec2 fragCoord = vec2(id.xy) + 0.5;
    vec4 fragColor;
    mainImage(fragColor, fragCoord);
    screen[ivec2(id.x, SCREEN_SIZE.y - 1 - id.y)] = fragColor;
    // screen[ivec2(id.xy)] = fragColor;
}