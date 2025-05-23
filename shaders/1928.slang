import std;

//Kerr params
static const float a = 2.5;
static const float m = 1.0;
static const float Q = 0.0;
static const float PI = 3.14159265f;
static const float FOV = 2.5;
static const float Radius = 15.0;
static const float deltaTimeMin = 0.01;
static const float deltaTimeMax = 0.20;
static const float eps = 1e-3;

[Differentiable]
float sqr(float x) { return x*x; }

[Differentiable]
float4x4 diag(float4 a) {
    return float4x4(a.x,0,0,0,  0,a.y,0,0,  0,0,a.z,0,  0,0,0,a.w);
}

[Differentiable]
float4 KerrParams(float4 q) {
    float3 p = q.yzw;
    float rho = dot(p,p) - a*a;
    float r2 = 0.5*(rho + sqrt(sqr(rho) + 4.0*sqr(a*p.z)));
    float r = eps + sqrt(r2);
    float invr2 = 1.0/(eps + r2 + a*a); 
    float3 k = float3((r*p.x + a*p.y)*invr2, (r*p.y - a*p.x)*invr2, p.z/r);
    float f = r2*(2.0*m*r - Q*Q)/(eps + sqr(r2) + sqr(a*p.z));
    return float4(k, f);
}

[Differentiable]
float4x4 G(float4 q) { //Kerr metric in Kerr-Schild coordinates 
    float4 k = KerrParams(q);
    float4 kf0 = k.w*float4(1, k.xyz);
    return diag(float4(-1,1,1,1)) + float4x4(kf0, k.x*kf0, k.y*kf0, k.z*kf0);    
}

[Differentiable]
float4x4 Ginv(float4 q) { //inverse of Kerr metric in Kerr-Schild coordinates 
    float4 k = KerrParams(q);
    float4 kf1 = k.w*float4(1, -k.xyz);
    return diag(float4(-1,1,1,1)) + float4x4(-kf1, k.x*kf1, k.y*kf1, k.z*kf1); 
}

float4 vec2P(float3 v, float3 x) { //vector to momentum at point q
    return 2.0*mul(G(float4(0.0, x)),float4(1.0, v)); 
}

float4 P2Vec(float4 p, float4 q) {
    return mul(Ginv(q),p);
}

[Differentiable]
float L(float4 qt, float4x4 g) {
    return dot(mul(g,qt),qt);
}

[Differentiable]
float L(float4 qt, float4 q) {
    return L(qt, G(q));
}

[Differentiable]
float H(float4 p, float4x4 ginv) {
    return dot(mul(ginv,p),p);
}

[Differentiable]
float H(float4 p, float4 q) {
    return H(p, Ginv(q));
}

float4 dHdq(float4 p, float4 q) {
    DifferentialPair<float4> dH_dp = diffPair(p); 
    DifferentialPair<float4> dH_dq = diffPair(q); 
    bwd_diff(H)(dH_dp, dH_dq, 1.0);
    return dH_dq.d;
}

void step(inout float4 p, inout float4 q, float dt) {
    float4x4 ginv = Ginv(q);
    float4 qt = mul(ginv, p);
    float4 grad = dHdq(p, q);
    dt /= length(qt)+0.1*length(grad);
    p -= grad*dt;
    q += mul(ginv, p)*dt;
}

float3x3 getCamera(float2 angles) {
   float3x3 theta_rot = float3x3(1,   0,              0,
                          0,  cos(angles.y),  -sin(angles.y),
                          0,  sin(angles.y),  cos(angles.y)); 
        
   float3x3 phi_rot = float3x3(cos(angles.x),   sin(angles.x), 0.,
        		       -sin(angles.x),   cos(angles.x), 0.,
        		        0.,              0.,            1.); 
        
   return mul(theta_rot,phi_rot);
}

float3 getRay(float2 angles, float2 pos) {
    float3x3 camera = getCamera(angles);
    return normalize(mul(transpose(camera),float3(FOV*pos.x, 1., FOV*pos.y)));
}

float2 dirToLatLongUV(float3 d)
{
    float3 n = normalize(d);
    float  u = atan2(n.y, n.x) * (1.0 / (2.0 * PI)) + 0.5;
    float  v = 0.5 - asin(n.z) / PI;
    return float2(u, v);
}

float3 sampleSky(float3 dir)
{
    float2 uv = dirToLatLongUV(dir);
    uint width, height;
    channel0.GetDimensions(width, height);
    return channel0[int2(uv * float2(width, height))].xyz;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main_image(uint3 id : SV_DispatchThreadID) {
    // Viewport resolution (in pixels)
    uint width, height;
    screen.GetDimensions(width, height);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= width || id.y >= height)
        return;

    // Pixel coordinates (centre of pixel, origin at bottom left)
    float2 R = float2(width, height); 
    float2 fragCoord = float2(float(id.x) + 0.5, float(height - id.y) - 0.5);
    float2 uv = (fragCoord - 0.5*R)/max(R.x, R.y);
    float2 angles = float2(2.0*PI, PI)*(mouse.pos/R - 0.5);

    if(mouse.click == 0)
    {
        angles = float2(0.04 + 0.2*time.elapsed, 0.45);
    }

    float3 rd = getRay(angles, uv);
    float3 center_rd = getRay(angles, float2(0.));
    float3 ro = float3(1.0,0,0)-center_rd*Radius;

    float4 Q = float4(0., ro);
    float4 P = vec2P(rd, ro);

    bool hitEH = false;
        
    for(int i = 0; i < 512; i++)
    {
        float r = length(Q.yzw);
        float dt = lerp(deltaTimeMin, deltaTimeMax, r/10.0);
        step(P, Q, dt);
        //if(r < 1.0) {hitEH = true;  break;}
        if(r > 40.0) {break;}
    }
    
    float4 qt = P2Vec(P,Q);
    rd = normalize(qt.yzw);

    float l = length(P);

    float3 col = sampleSky(rd);
    if(hitEH) col = 0.0;

    col = col/(l*0.10);

    // col = l*0.01;
    // col += float3(0.2, 0.0, 0.0)*smoothstep(40.0,45.0,l);

    screen[id.xy] = float4(tanh(col), 1.0);
}
