// Kerr projector (compute.toys, Slang) - every image of points and meshes around a spinning black hole. Rays
// are traced once per camera pose into a table; points are then found in the table by Newton iteration (no per-
// point tracing, no frame-to-frame state), and everything is drawn with atomics (no hardware rasteriser).
//   sliders (0..1): dist (>1 = distance in M), spin, fov, speed, psize, exposure
//   drag = orbit, wheel / W / S = distance, click the buttons (top left) to toggle parts, H hides them
// Needs "Float32 Textures" on (settings): the per-point images live in the pass textures. Up to 3840 x 2160.
// Light travel time: every image shows its point where it was when the light left (toggle DELAY).
import std;
// ============================================================================ sizes
// ray table: rays are laid out on a log-polar chart around the critical curve
static const int PSI_COUNT = 192;                     // angular bins around the curve
static const int ELL_COUNT = 88;                      // radial rows on each side of the curve
static const int ROW_COUNT = 2*ELL_COUNT;             // outside rows, then inside rows
static const int STEP_COUNT = 128;                    // samples stored along each ray
static const int RAY_COUNT = PSI_COUNT*ROW_COUNT;
static const float STEP_ARC = 0.215;                  // log-polar arclength between samples
static const float INNER_LOG_ROWS = 55.0;             // inside rows up to here are log-spaced ...
static const float INNER_LOG_END = 0.08;              // ... up to depth e = 0.08, then linear to the centre
static const float INNER_LIN_STEP = (0.99995 - INNER_LOG_END)/(ELL_COUNT - 1 - INNER_LOG_ROWS);
static const int CRIT_BINS = 720;                     // resolution of rho_c(psi)
static const float PSI_WARP = 0.66;                   // psi bins are denser near the x axis
// inverse grid (section 2): nodes in (log r, cos theta, phi) per swept-angle band, psi half and slot
static const int BAND_COUNT = 10;                     // swept angle bands of pi/2, up to 5 pi
static const float BAND_WIDTH = 0.5*3.14159265;
static const float BAND_OVERLAP = 0.15;
static const int DIRECT_BANDS = 2;                    // swept angle < pi: the finer grid
static const int3 GRID_DIRECT = int3(40, 48, 96);
static const int3 GRID_LOOPS = int3(24, 32, 64);
static const int SECTOR_COUNT = 2;                    // psi halves: images of one band on either side
static const int SECTOR_OVERLAP = 8;
static const int SLOT_COUNT = 2*SECTOR_COUNT;         // per half: the containing cells of lowest, highest index
static const int GRID_SIZE = SLOT_COUNT*(DIRECT_BANDS*40*48*96 + (BAND_COUNT - DIRECT_BANDS)*24*32*64);
static const float GRID_R_MAX = 160;
static const int CELL_ROWS = 2*(ELL_COUNT - 1);
static const int CELL_COUNT = PSI_COUNT*CELL_ROWS*(STEP_COUNT - 1);
static const uint EMPTY_NODE = 0xffffffff;
static const int DELAY_PASSES = 10;                   // light delay search, see solvePoint
static const float WORLDLINE_STEP = 16.0;
static const int MAX_IMAGES = 10;                     // per point while solving
static const int PARTICLE_IMAGES = 8;                 // kept per particle (visible orders only)
// scene: particles + small spheres (lat-long meshes)
static const int PARTICLE_COUNT = 65536;               // accretion disk
static const int OBJECT_COUNT = 6;
static const int SPHERE_SEGS = 16, SPHERE_RINGS = 10;
static const int OBJECT_VERTS = (SPHERE_SEGS + 1)*(SPHERE_RINGS + 1);
static const int OBJECT_TRIS = 2*SPHERE_SEGS*SPHERE_RINGS;
static const int POINT_COUNT = PARTICLE_COUNT + OBJECT_COUNT*OBJECT_VERTS;
static const int SUBTRI_CAPACITY = 49152;
static const int MAX_PIXELS = 3840*2160;              // depth buffer: screens up to 4K
static const float DISK_BRIGHTNESS = 0.4;
static const int GLOW_PIXELS = 1280*720;              // additive disk buffer, 1/f resolution (see glowScale)
static const int SOLVE_THREADS = ((POINT_COUNT + 63)/64)*64;
static const float PI = 3.14159265;
static const int HORIZON_RECORDS = 0;                 // offsets into rayRecords
static const int SKY_RECORDS = RAY_COUNT;
// pass texture layers: background cache, image chart points, image screen data, raster colour
static const int BACKGROUND_LAYER = 0, CHART_LAYER = 1, SCREEN_LAYER = 2, COLOUR_LAYER = 3;
// ============================================================================ storage
[StorageBuffer(STEP_COUNT*RAY_COUNT)]        RWStructuredBuffer<float4> rayTable;        // [row][psi][k]: outgoing position, delay
[StorageBuffer(STEP_COUNT*RAY_COUNT/2)]      RWStructuredBuffer<uint> rayAngles;         // swept angles, 2 x f16 per uint
[StorageBuffer(2*RAY_COUNT)]                 RWStructuredBuffer<float4> rayRecords;      // horizon landing | sky direction
[StorageBuffer(GRID_SIZE)]                   RWStructuredBuffer<Atomic<uint>> inverseGrid;     // section 2
[StorageBuffer(SUBTRI_CAPACITY*6)]           RWStructuredBuffer<float4> subTriangles;    // 3 x (pixel, depth, g), 3 x (normal, extra)
[StorageBuffer(MAX_PIXELS)]                  RWStructuredBuffer<Atomic<uint>> depthBuffer;     // ~depth bits, 0 = empty
[StorageBuffer(GLOW_PIXELS)]                 RWStructuredBuffer<Atomic<uint>> glowBuffer;      // disk light, RGB 11/11/10
[StorageBuffer(CRIT_BINS)]                   RWStructuredBuffer<float> critRadius;       // critical curve rho_c(psi)
[StorageBuffer(8)]                           RWStructuredBuffer<float4> controlState;    // persistent camera / UI state
[StorageBuffer(4)]                           RWStructuredBuffer<Atomic<uint>> counters;
struct Params {
    float spin, horizonRadius;                              // a, r+
    float cameraRadius, cameraTheta, cameraYaw;             // yaw: world -> table frame rotation
    float cameraSigma, cameraDelta, cameraA, cameraOmega;   // metric at the camera, frame dragging rate
    float lapse;                                            // ZAMO lapse sqrt(Sigma Delta / A)
    float outsideEllStart, outsideEllStep;                  // outside rows: ell = start + row*step
    float insideLogStart, insideLogStep;                    // inside log rows: log e = start + row*step
    float2 projection;                                      // screen x, y scale
    float sceneTime, beaming, exposure, pointRadius;
    float pixelScale; int uiScale, glowScale;               // screen height / 1080 (>= 1), UI zoom, glow pixel size
    float innerEdge;                                        // disk inner edge: the prograde ISCO
    float cameraTimeShift;                                  // hTime(cameraRadius), see lightDelay
    uint tableDirty, viewDirty, uiHidden;                   // rebuild the table / the background cache
    uint toggles;          // bits: 0 particles, 1 objects, 2 stars, 3 sky grid, 4 horizon grid, 5-7 orders, 8 delay
    float orbitRadiusMin, orbitRadiusMax;                   // photon orbits seen on the critical curve
};
[StorageBuffer(1)] RWStructuredBuffer<Params> params;
#define P params[0]
float sqr(float x) { return x*x; }
// ============================================================================ Kerr geometry
// World and table positions are Cartesian with the spin along y; the table frame has the camera at yaw 0.
float spin() { return P.spin; }
float delta(float r) { return r*r - 2*r + spin()*spin(); }
float blRadius(float3 X) {
    float a2 = spin()*spin(); float b = dot(X, X) - a2;
    return sqrt(max(0.5*(b + sqrt(b*b + 4*a2*X.y*X.y)), 1e-12));
}
// outgoing azimuth phi_out = phi_BL - h(r): regular across the horizon, so the table stays smooth there
float outgoingShift(float r) {
    float rPlus = P.horizonRadius; float rMinus = 2 - rPlus;
    return spin()/(rPlus - rMinus)*log(abs((r - rPlus)/(r - rMinus)));
}
float3 toCartesian(float r, float cosTheta, float phi) {
    float R = sqrt(r*r + spin()*spin()); float sinTheta = sqrt(max(1 - cosTheta*cosTheta, 0.));
    return float3(R*sinTheta*cos(phi), r*cosTheta, -R*sinTheta*sin(phi));
}
float3 toTableFrame(float3 X) {
    float c = cos(P.cameraYaw), s = sin(P.cameraYaw);
    return float3(X.x*c - X.z*s, X.y, X.z*c + X.x*s);
}
float polarAngle(float3 X, float r) { return atan2(length(X.xz)/sqrt(r*r + spin()*spin()), X.y/r); }   // precise at poles
// rotation about the spin axis: phi -> phi + angle
float3 rotatePhi(float3 X, float angle) {
    float c = cos(angle), s = sin(angle);
    return float3(X.x*c + X.z*s, X.y, X.z*c - X.x*s);
}
// world point -> table frame outgoing position (w = r): the outgoing azimuth is a rotation by -h(r). Points
// inside the horizon are pushed out.
float4 targetOf(float3 worldPos) {
    float3 X = toTableFrame(worldPos); float r = blRadius(X);
    float minR = P.horizonRadius + 0.08;
    if (r < minR) { X *= minR/r; r = blRadius(X); }
    return float4(rotatePhi(X, -outgoingShift(r)), r);
}
// table-frame velocity of a target moving with world velocity V (the change of h(r) with r is left out)
float3 targetVelocity(float3 worldPos, float3 worldVelocity) { return rotatePhi(toTableFrame(worldVelocity), -outgoingShift(blRadius(worldPos))); }
// ============================================================================ screen chart
// Log-polar chart around the critical curve rho_c(psi); a chart point is (psi index, row, k, side).
//   outside (side +1): rho = rho_c + e^ell
//   inside  (side -1): rho = rho_c (1 - e), e log-spaced near the curve, then linear to the centre
float psiOf(float psiIndex) {
    float t = -PI + 2*PI*psiIndex/PSI_COUNT;
    return t + 0.5*PSI_WARP*sin(2*t);
}
float psiIndexOf(float psi) {
    float t = psi;
    for (int i = 0; i < 5; i++) t -= (t + 0.5*PSI_WARP*sin(2*t) - psi)/(1 + PSI_WARP*cos(2*t));
    return (t + PI)/(2*PI)*PSI_COUNT;
}
float wrapPsi(float x) { return x - PSI_COUNT*floor(x/PSI_COUNT); }
float psiDifference(float from, float to) {
    float d = to - from;
    return d - PSI_COUNT*floor(d/PSI_COUNT + 0.5);
}
float critRho(float psi) {
    float f = (psi + PI)/(2*PI)*CRIT_BINS - 0.5; int i = int(floor(f));
    float r0 = critRadius[(i%CRIT_BINS + CRIT_BINS)%CRIT_BINS];
    float r1 = critRadius[((i + 1)%CRIT_BINS + CRIT_BINS)%CRIT_BINS];
    return lerp(r0, r1, f - floor(f));
}
float insideDepth(float row) {
    return row <= INNER_LOG_ROWS ? exp(P.insideLogStart + row*P.insideLogStep) : INNER_LOG_END + (row - INNER_LOG_ROWS)*INNER_LIN_STEP;
}
float insideRowOf(float depth) {
    if (depth <= INNER_LOG_END) return (log(max(depth, 1e-30)) - P.insideLogStart)/P.insideLogStep;
    return INNER_LOG_ROWS + (depth - INNER_LOG_END)/INNER_LIN_STEP;
}
float outsideRowOf(float distance) { return (log(max(distance, 1e-30)) - P.outsideEllStart)/P.outsideEllStep; }
float outsideDistance(float row) { return exp(P.outsideEllStart + row*P.outsideEllStep); }
float ellPerRow(float row, float side) {   // d ell / d row
    return side > 0 ? P.outsideEllStep : row <= INNER_LOG_ROWS ? P.insideLogStep : INNER_LIN_STEP/insideDepth(row);
}
int rowOffset(float side) { return side > 0 ? 0 : ELL_COUNT; }
float2 chartToScreen(float4 chartPoint) {
    float psi = psiOf(chartPoint.x); float rhoC = critRho(psi);
    float rho = chartPoint.w > 0 ? rhoC + outsideDistance(chartPoint.y) : rhoC*(1 - insideDepth(chartPoint.y));
    return rho*float2(cos(psi), sin(psi));
}
float4 screenToChart(float2 xy) {
    float psi = atan2(xy.y, xy.x); float rho = length(xy);
    float rhoC = critRho(psi);
    if (rho > rhoC) return float4(psiIndexOf(psi), outsideRowOf(rho - rhoC), 0, 1);
    return float4(psiIndexOf(psi), insideRowOf(1 - rho/rhoC), 0, -1);
}
// signed distance from the curve, sigma = rho/rho_c - 1: linear across it (used to cross sides)
float sigmaOf(float4 chartPoint) {
    return chartPoint.w > 0 ? outsideDistance(chartPoint.y)/critRho(psiOf(chartPoint.x)) : -insideDepth(chartPoint.y);
}
float2 rowFromSigma(float sigma, float psiIndex) {   // -> (row, side)
    if (sigma >= 0) return float2(clamp(outsideRowOf(sigma*critRho(psiOf(psiIndex))), 0., ELL_COUNT - 1.), 1);
    return float2(clamp(insideRowOf(-sigma), 0., ELL_COUNT - 1.), -1);
}
// The inside chart is polar around the shadow centre, where psi degenerates. Near it (the linear rows) points
// are also handled in Cartesian form c = (rho/rho_c)(cos psi, sin psi).
bool nearCentre(float4 chartPoint) { return chartPoint.w < 0 && chartPoint.y > INNER_LOG_ROWS; }
float2 toCentreXY(float4 chartPoint) {
    float psi = psiOf(chartPoint.x);
    return (1 - insideDepth(chartPoint.y))*float2(cos(psi), sin(psi));
}
float2 fromCentreXY(float2 c) {   // -> (psi index, row)
    float psiIndex = wrapPsi(psiIndexOf(atan2(c.y, c.x))); float row = insideRowOf(1 - max(length(c), 5e-5));
    return float2(psiIndex, clamp(row, 0., ELL_COUNT - 1.));
}
// ============================================================================ rays
// Kerr-Schild Cartesian coordinates (spin along z): g = eta + f l l, so g^-1 = eta - f l l. With p_t = -1 the
// Hamiltonian is H = (|p|^2 - 1 - f (1 + l.p)^2)/2, and bwd_diff gives both dH/dp (the velocity) and dH/dx
// (minus the force).
[Differentiable]
float ksRadius(float3 X) {
    float a = no_diff spin(); float b = dot(X, X) - a*a;
    float d = sqrt(b*b + 4*a*a*X.z*X.z);
    float r2 = b >= 0 ? 0.5*(b + d) : 2*a*a*X.z*X.z/max(d - b, 1e-30);   // 2nd form: stable on the ring's disk
    return sqrt(max(r2, 1e-24));
}
[Differentiable]
float hamiltonian(float3 p, float3 X) {
    float a = no_diff spin(); float r = ksRadius(X);
    float W = r*r + a*a;
    float3 l = float3((r*X.x + a*X.y)/W, (r*X.y - a*X.x)/W, X.z/r);
    float f = 2*r*r*r/(r*r*r*r + a*a*X.z*X.z); float L = 1 + dot(l, p);
    return 0.5*(dot(p, p) - 1 - f*L*L);
}
struct Ray { float3 x, p; };
Ray flow(Ray ray) {
    var dp = diffPair(ray.p); var dx = diffPair(ray.x);
    bwd_diff(hamiltonian)(dp, dx, 1.0);
    return { dp.d, -dx.d };
}
Ray rayAdd(Ray ray, Ray d, float h) { return { ray.x + h*d.x, ray.p + h*d.p }; }
// Kerr-Schild time along the ray: dt/dlambda = dH/dp_t with the full H = (g^ab p_a p_b)/2, g^tt = -1 - f and
// g^ti = -f l^i, which at p_t = -1 is 1 + f (1 + l.p). This is the light travel time (see emissionTime).
float timeRate(Ray ray) {
    float a = spin(); float r = ksRadius(ray.x); float3 X = ray.x;
    float W = r*r + a*a;
    float3 l = float3((r*X.x + a*X.y)/W, (r*X.y - a*X.x)/W, X.z/r);
    return 1 + 2*r*r*r/(r*r*r*r + a*a*X.z*X.z)*(1 + dot(l, ray.p));
}
// RK4 step of stepArc log-polar arclength (|dx|/|x|): fine near the hole, long far out. Also returns the angle
// swept around the hole and the time elapsed during the step.
Ray rk4Step(Ray ray, float stepArc, out float sweptAngle, out float timeStep) {
    float rho = length(ray.x); Ray k1 = flow(ray);
    float h = stepArc/max(length(k1.x)/rho, 1e-6);
    Ray s2 = rayAdd(ray, k1, 0.5*h); Ray k2 = flow(s2);
    Ray s3 = rayAdd(ray, k2, 0.5*h); Ray k3 = flow(s3);
    Ray s4 = rayAdd(ray, k3, h); Ray k4 = flow(s4);
    float3 dX = h/6*(k1.x + 2*k2.x + 2*k3.x + k4.x); float3 dP = h/6*(k1.p + 2*k2.p + 2*k3.p + k4.p);
    sweptAngle = length(dX - dot(dX, ray.x)*ray.x/(rho*rho))/rho;
    timeStep = h/6*(timeRate(ray) + 2*timeRate(s2) + 2*timeRate(s3) + timeRate(s4));
    return { ray.x + dX, ray.p + dP };
}
// Constants of motion of the light reaching screen xy of the ZAMO camera: (lambda, eta, BL radial momentum, BL
// polar momentum)
float4 rayConstants(float2 xy) {
    float3 n = float3(-1, -xy.y, -xy.x)/sqrt(1 + dot(xy, xy));
    float Sigma = P.cameraSigma, Delta = P.cameraDelta, A = P.cameraA, omega = P.cameraOmega;
    float a = spin(), s = sin(P.cameraTheta), c = cos(P.cameraTheta);
    float lambda = n.z/(Sigma*sqrt(Delta)/(A*s) + omega*n.z);
    float energy = (1 - omega*lambda)*sqrt(A/(Sigma*Delta)); float thetaRate = sqrt(Sigma)*n.y*energy;
    float eta = thetaRate*thetaRate - a*a*c*c + lambda*lambda*c*c/(s*s);
    float radialMomentum = n.x*energy*sqrt(Sigma*Delta);
    return float4(lambda, eta, radialMomentum, -s*thetaRate);
}
// The ray arriving at pixel (x, y) is traced as the outgoing ray from (-x, y), the (t, phi) -> (-t, -phi)
// mirror of the arriving one. BL momentum -> Kerr-Schild position + covariant momentum with p_t = -1.
Ray initialRay(float2 xy) {
    float4 constants = rayConstants(xy); float lambda = constants.x;
    float a = spin(), r = P.cameraRadius, s = sin(P.cameraTheta), c = cos(P.cameraTheta);
    float Sigma = r*r + a*a*c*c; float Delta = delta(r);
    float W = r*r + a*a; float Pr = W - a*lambda;
    float kr = constants.z/Sigma;
    float ktOut = (W*Pr/Delta + a*(lambda - a*s*s))/Sigma + 2*r/Delta*kr;
    float kphiOut = (a*Pr/Delta - a + lambda/(s*s))/Sigma + a/Delta*kr;
    float3 X = float3(r*s, a*s, r*c);
    float3 kx = float3(s, 0, c)*kr + float3(r*c, a*c, -r*s)*(-constants.w/s)/Sigma + float3(-a*s, r*s, 0)*kphiOut;
    float3 l = float3((r*X.x + a*X.y)/W, (r*X.y - a*X.x)/W, X.z/r);
    float f = 2*r*r*r/(r*r*r*r + a*a*X.z*X.z); float lk = ktOut + dot(l, kx);
    return { X, (kx + f*l*lk)/(ktOut - f*lk) };
}
// Kerr-Schild position -> stored (mirrored, outgoing) table position, with the delay
float4 tableEntry(float3 X, float delay) {
    float r = ksRadius(X);
    float phi = atan2(X.y, X.x) - atan2(spin(), r) + outgoingShift(P.cameraRadius);
    return float4(toCartesian(r, clamp(X.z/r, -1., 1.), -phi), delay);
}
// table position -> direction in the world (theta, phi_BL) at the horizon shell
float3 horizonDirection(float3 X, float rShell) {
    float r = blRadius(X); float theta = polarAngle(X, r);
    float phi = atan2(-X.z, X.x) + outgoingShift(rShell);
    return float3(sin(theta)*cos(phi), cos(theta), -sin(theta)*sin(phi));
}
// ============================================================================ 0. camera, params
// screen point (x, y, ok) of the spherical photon orbit at radius orbitR
float3 photonOrbitPoint(float orbitR) {
    float a = spin(), theta = P.cameraTheta, omega = P.cameraOmega;
    float orbitDelta = delta(orbitR); float dDelta = 2*(orbitR - 1);
    if (dDelta <= 0 || orbitDelta <= 0) return 0;
    float lambda = (orbitR*orbitR + a*a - 4*orbitR*orbitDelta/dDelta)/a;
    float eta = 16*orbitR*orbitR*orbitDelta/(dDelta*dDelta) - sqr(lambda - a);
    float energy = (1 - omega*lambda)*sqrt(P.cameraA/(P.cameraSigma*P.cameraDelta));
    float nPhi = P.cameraSigma*sqrt(P.cameraDelta)/(P.cameraA*sin(theta))*lambda/(1 - omega*lambda);
    float polarPotential = eta + a*a*cos(theta)*cos(theta) - lambda*lambda/sqr(tan(theta));
    if (polarPotential < 0) return 0;
    float nTheta = sqrt(polarPotential)/(sqrt(P.cameraSigma)*energy);
    float nr2 = 1 - nPhi*nPhi - nTheta*nTheta;
    if (nr2 <= 0) return 0;
    return float3(-nPhi, -nTheta, sqrt(nr2))/sqrt(nr2);
}
bool tableDirty() { return P.tableDirty != 0; }
// controls -> camera state and the frame's parameters (thread 0 of setup)
void updateCamera() {
    float2 screenSize = float2(SCREEN_SIZE);
    float4 orbitState = controlState[0];      // yaw, pitch, yaw and pitch at press
    float4 clickState = controlState[1];      // previous click, press started on a button
    float4 distanceState = controlState[4];   // distance, wheel zoom, slider distance
    float4 uiState = controlState[5];         // toggles, previous H key, UI hidden
    int uiScale = max(1, (int(screenSize.y) + 270)/1080);
    counters[0].store(0);                     // sub-triangle count
    if (time.frame == 0) {
        orbitState = float4(0.6, 0.12, 0.6, 0.12); clickState = 0;
        distanceState = 0; uiState = float4(511, 0, 0, 0);
    }
    // press: on a button toggles it, elsewhere starts an orbit drag
    uint toggles = uint(uiState.x); float click = float(mouse.click);
    if (click > 0 && clickState.x == 0) {
        orbitState.zw = orbitState.xy;
        int button = uiButton(float2(mouse.pos) + 0.5, uiState.z, uiScale);
        clickState.y = button >= 0 ? 1 : 0;
        if (button >= 0) toggles ^= 1u << button;
    }
    if (click > 0 && clickState.y == 0) {
        float2 drag = float2(mouse.pos - mouse.start)/screenSize.y;
        orbitState.x = orbitState.z + 3*drag.x;
        orbitState.y = clamp(orbitState.w + 2*drag.y, -1.45, 1.45);
    }
    clickState.x = click; bool hideKey = keyDown(72);
    if (hideKey && uiState.y == 0) uiState.z = 1 - uiState.z;
    uiState.y = hideKey ? 1 : 0; uiState.x = toggles;
    // distance: changed by the slider (when it moves), the wheel (as a ratio) and W / S
    float sliderDistance = custom.dist > 1 ? custom.dist : 4 + 36*sqr(clamp(custom.dist, 0., 1.));
    float zoom = mouse.zoom > 0 ? mouse.zoom : 1;
    if (distanceState.x <= 0) distanceState = float4(sliderDistance, zoom, sliderDistance, 0);
    if (abs(sliderDistance - distanceState.z) > 1e-5) distanceState.xz = sliderDistance;
    distanceState.x *= distanceState.y/zoom; distanceState.y = zoom;
    float keyZoom = (keyDown(87) ? 0.985 : 1)/(keyDown(83) ? 0.985 : 1);
    distanceState.x = clamp(distanceState.x*keyZoom, 2.2, 80.);
    controlState[0] = orbitState; controlState[1] = clickState;
    controlState[4] = distanceState; controlState[5] = uiState;
    float a = max(0.999*clamp(custom.spin, 0., 1.), 1e-3); float r = distanceState.x;
    float theta = 0.5*PI - orbitState.y;
    float fov = lerp(20., 110., clamp(custom.fov, 0., 1.))*PI/180;
    float rPlus = 1 + sqrt(1 - a*a); float focal = 1/tan(0.5*fov);
    // the table depends on (a, r, theta) only; yaw is applied to the targets
    float4 tableKey = float4(a, r, theta, 0);
    bool rebuild = time.frame == 0 || any(abs(tableKey - controlState[2]) > float4(1e-6, 1e-4*r, 2e-4, 1));
    // the background cache also depends on the fov and the screen size
    float4 viewKey = float4(fov, screenSize, 0);
    bool viewChanged = rebuild || any(abs(viewKey - controlState[3]) > 1e-6);
    if (rebuild) controlState[2] = tableKey;
    if (viewChanged) controlState[3] = viewKey;
    float Sigma = r*r + a*a*cos(theta)*cos(theta); float Delta = r*r - 2*r + a*a;
    float A = sqr(r*r + a*a) - a*a*Delta*sin(theta)*sin(theta);
    P.spin = a; P.horizonRadius = rPlus;
    P.cameraRadius = r; P.cameraTheta = theta;
    P.cameraYaw = -orbitState.x; P.cameraSigma = Sigma;
    P.cameraDelta = Delta; P.cameraA = A;
    P.cameraOmega = a*(r*r + a*a - Delta)/A; P.lapse = sqrt(Sigma*Delta/A);
    P.outsideEllStart = -9; P.outsideEllStep = (log(2.8) + 9)/(ELL_COUNT - 1);
    P.insideLogStart = -9;
    P.insideLogStep = (log(INNER_LOG_END) + 9)/INNER_LOG_ROWS;
    P.projection = float2(focal*screenSize.y/screenSize.x, focal);
    P.sceneTime = time.elapsed*40*custom.speed; P.beaming = 1;
    P.exposure = lerp(0.2, 5., sqr(custom.exposure)); P.pixelScale = max(1., screenSize.y/1080);
    P.pointRadius = lerp(0.6, 4., custom.psize)*P.pixelScale; P.uiScale = uiScale;
    P.glowScale = max(1, max((int(screenSize.x) + 1279)/1280, (int(screenSize.y) + 719)/720));
    float z1 = 1 + pow(1 - a*a, 1/3.)*(pow(1 + a, 1/3.) + pow(1 - a, 1/3.)), z2 = sqrt(3*a*a + z1*z1);
    P.innerEdge = 3 + z2 - sqrt((3 - z1)*(3 + z1 + 2*z2));
    P.cameraTimeShift = hTime(r);
    P.tableDirty = rebuild ? 1 : 0; P.viewDirty = viewChanged ? 1 : 0;
    P.uiHidden = uiState.z > 0.5 ? 1 : 0; P.toggles = toggles;
}
// rho_c(psi): the curve is symmetric in y and its polar angle runs monotonically from one x-axis crossing
// (orbitRadiusMin) to the other (orbitRadiusMax) along the photon orbits: bisect per bin
float critRhoOfBin(int bin) {
    float targetAngle = abs(-PI + 2*PI*(bin + 0.5)/CRIT_BINS);
    float lo = P.orbitRadiusMin, hi = P.orbitRadiusMax;
    if (lo <= 0) return sqrt(27.)/P.cameraRadius;
    float3 pointLo = photonOrbitPoint(lo), pointHi = photonOrbitPoint(hi);
    bool increasing = atan2(abs(pointHi.y), pointHi.x) > atan2(abs(pointLo.y), pointLo.x);
    for (int i = 0; i < 40; i++) {
        float mid = 0.5*(lo + hi);
        float3 p = photonOrbitPoint(mid);
        if ((atan2(abs(p.y), p.x) < targetAngle) == increasing) lo = mid;
        else hi = mid;
    }
    float3 p = photonOrbitPoint(0.5*(lo + hi));
    return p.z > 0 ? length(p.xy) : sqrt(27.)/P.cameraRadius;
}
groupshared Atomic<uint> firstVisible;
groupshared Atomic<uint> lastVisible;
// One workgroup: thread 0 updates the camera, then (after a camera change) all threads find the radial range
// of the photon orbits visible on the critical curve and bisect rho_c(psi). The barriers stay outside the
// branches (WGSL wants them in uniform control flow).
[shader("compute")]
[WorkgroupCount(1, 1, 1)] [numthreads(256, 1, 1)]
void setup(uint thread : SV_GroupIndex) {
    if (thread == 0) {
        updateCamera();
        firstVisible.store(0xffffffff);
        lastVisible.store(0);
    }
    AllMemoryBarrierWithGroupSync();
    bool rebuild = tableDirty();
    float rPlus = P.horizonRadius;
    float scanStep = (6 - rPlus)/4000;
    for (uint i = thread; i <= 4000 && rebuild; i += 256) {
        if (photonOrbitPoint(rPlus + 1e-4 + scanStep*i).z <= 0) continue;
        firstVisible.min(i);
        lastVisible.max(i);
    }
    AllMemoryBarrierWithGroupSync();
    if (thread == 0 && rebuild) {   // refine both ends by halving steps
        uint first = firstVisible.load(), last = lastVisible.load();
        float lo = first <= 4000 ? rPlus + 1e-4 + scanStep*first : -1;
        float hi = first <= 4000 ? rPlus + 1e-4 + scanStep*last : -1;
        for (int i = 0; i < 40; i++) {
            float nudge = scanStep*exp2(-i - 1.);
            if (photonOrbitPoint(lo - nudge).z > 0) lo -= nudge;
            if (photonOrbitPoint(hi + nudge).z > 0) hi += nudge;
        }
        P.orbitRadiusMin = lo; P.orbitRadiusMax = hi;
    }
    AllMemoryBarrierWithGroupSync();
    for (int bin = int(thread); bin < CRIT_BINS && rebuild; bin += 256) critRadius[bin] = critRhoOfBin(bin);
}
// ============================================================================ 1. ray table
// 192 psi x 176 rows (outside / inside the critical curve) x 128 samples of 0.215 log-polar arclength. Each ray
// also records its exact horizon-shell crossing and its direction at infinity.
[shader("compute")]
[WorkgroupCount(RAY_COUNT/64, 1, 1)] [numthreads(64, 1, 1)]
void buildTable(uint3 id : SV_DispatchThreadID) {
    if (!tableDirty()) return;
    int ray = int(id.x); int row = ray/PSI_COUNT;
    float side = row < ELL_COUNT ? 1 : -1;
    Ray state = initialRay(chartToScreen(float4(ray%PSI_COUNT, row - rowOffset(side), 0, side)));
    float sweptAngle = 0, delay = 0, previousAngle = 0; float rShell = P.horizonRadius + 0.02;
    float a = spin(); bool frozen = false, hitHorizon = false;
    rayTable[ray*STEP_COUNT] = float4(tableEntry(state.x, 0).xyz, 0);
    rayRecords[HORIZON_RECORDS + ray] = float4(0, 0, 0, -1);
    rayRecords[SKY_RECORDS + ray] = float4(0, 0, 0, -1);
    for (int i = ray; i < GRID_SIZE; i += RAY_COUNT) inverseGrid[i].store(EMPTY_NODE);   // clear (section 2)
    for (int k = 1; k < STEP_COUNT; k++) {
        // sub-steps: shorter near the ring singularity and when approaching the horizon shell
        float arcLeft = STEP_ARC;
        for (int n = 0; n < 48 && !frozen && arcLeft > 1e-6; n++) {
            float r0 = ksRadius(state.x);
            float ringDistance = length(float2(length(state.x.xy) - a, state.x.z));
            float stepArc = min(arcLeft, min(STEP_ARC/6, 0.2*ringDistance/length(state.x) + 1e-4));
            if (!hitHorizon) stepArc = min(stepArc, max(0.5*(r0 - rShell)/r0, 0.004));
            float stepAngle, stepTime; Ray next = rk4Step(state, stepArc, stepAngle, stepTime);
            float r1 = ksRadius(next.x);
            if (!hitHorizon && r1 <= rShell) {
                float f = (r0 - rShell)/max(r0 - r1, 1e-9);
                float3 landing = tableEntry(lerp(state.x, next.x, f), sweptAngle).xyz;
                rayRecords[HORIZON_RECORDS + ray] = float4(horizonDirection(landing, rShell), sweptAngle + f*stepAngle);
                hitHorizon = true;
            }
            sweptAngle += stepAngle; delay += stepTime; arcLeft -= stepArc;
            state = next;
            if (length(state.x) > 420) {
                // escaped: the direction of motion is the direction on the sky
                float3 d = normalize(flow(state).x);
                float phi = -(atan2(d.y, d.x) - atan2(a, r1) + outgoingShift(P.cameraRadius));
                float sinTheta = sqrt(1 - d.z*d.z);
                rayRecords[SKY_RECORDS + ray] = float4(sinTheta*cos(phi), d.z, -sinTheta*sin(phi), sweptAngle);
                frozen = true;
            }
            frozen = frozen || r1 < P.horizonRadius - 0.05;
        }
        rayTable[ray*STEP_COUNT + k] = tableEntry(state.x, delay);
        if (k%2 == 1) rayAngles[(ray*STEP_COUNT + k)/2] = f32tof16(previousAngle) | (f32tof16(sweptAngle) << 16);
        previousAngle = sweptAngle;
    }
}
// samples of one ray are adjacent: the trilinear lookups in section 4 touch 4 short runs instead of 8 planes
float4 sampleAt(int ray, int k) { return rayTable[ray*STEP_COUNT + k]; }
int rayOf(int psiIndex, int row) { return row*PSI_COUNT + (psiIndex%PSI_COUNT + PSI_COUNT)%PSI_COUNT; }
float4 tableAt(int psiIndex, int row, int k) { return sampleAt(rayOf(psiIndex, row), k); }
float angleAt(int psiIndex, int row, int k) {
    uint index = rayOf(psiIndex, row)*STEP_COUNT + k;
    return f16tof32((rayAngles[index/2] >> (16*(index%2))) & 0xffff);
}
// ============================================================================ 2. inverse grid
// World position -> chart point, as the Newton start for every point (no history, so points may be anywhere).
// Split by swept angle into bands of pi/2 and by psi into halves, the rays cover space nearly once, so the
// inverse is almost a function: sampled on nodes in table-frame (log r, cos theta, phi_out) and interpolated.
// Near caustics a node can still lie in several sheets, so each keeps two: the containing cells with the lowest
// and the highest index (different k, as a rule). Built by rasterising every table cell (a curved hexahedron
// between 2 x 2 rays and 2 samples) into the nodes it contains. Direct images change fastest: finer grid.
int3 gridSize(int band) { return band < DIRECT_BANDS ? GRID_DIRECT : GRID_LOOPS; }
int nodeIndex(int band, int slot, int3 node) {
    int3 n = gridSize(band);
    int directNodes = GRID_DIRECT.x*GRID_DIRECT.y*GRID_DIRECT.z;
    int base = band < DIRECT_BANDS ? band*directNodes : DIRECT_BANDS*directNodes + (band - DIRECT_BANDS)*n.x*n.y*n.z;
    return SLOT_COUNT*base + ((slot*n.z + (node.z%n.z + n.z)%n.z)*n.y + node.y)*n.x + node.x;
}
float gridMinRadius() { return P.horizonRadius + 0.05; }
// table-frame position -> grid coordinates of the band's nodes (phi periodic)
float3 gridCoords(float3 X, int band) {
    int3 n = gridSize(band);
    float r = blRadius(X);
    float3 unit = float3(log(r/gridMinRadius())/log(GRID_R_MAX/gridMinRadius()), 0.5 + 0.5*cos(polarAngle(X, r)), atan2(-X.z, X.x)/(2*PI) + 0.5);
    return unit*float3(n.x - 1, n.y - 1, n.z);
}
float3 nodePosition(int band, int3 node) {
    int3 n = gridSize(band);
    float r = gridMinRadius()*pow(GRID_R_MAX/gridMinRadius(), node.x/float(n.x - 1));
    return toCartesian(r, -1 + 2*node.y/float(n.y - 1), -PI + 2*PI*node.z/n.z);
}
// table cell index <-> (psi, row, k) of its first corner; rows skip the seam between the two sides
int4 cellOrigin(int cell) {
    int cellRow = (cell/PSI_COUNT)%CELL_ROWS;
    int side = cellRow < ELL_COUNT - 1 ? 1 : -1;
    return int4(cell%PSI_COUNT, cellRow + (side > 0 ? 0 : 1), cell/(PSI_COUNT*CELL_ROWS), side);
}
float4 decodeNode(uint stored, int slot) {   // -> chart point (psi, row, k, side), side 0 = empty
    if (stored == EMPTY_NODE) return 0;
    uint key = slot%2 == 0 ? stored : ~stored;
    int4 origin = cellOrigin(int(key >> 3) - 1);
    float3 f = 0.25 + 0.5*float3((key >> 2) & 1, (key >> 1) & 1, key & 1);
    return float4(origin.x + f.x, origin.y - rowOffset(origin.w) + f.y, origin.z + f.z, origin.w);
}
void rasteriseCell(int cell) {
    int4 origin = cellOrigin(cell);
    float4 c[8];
    float angles[8];
    for (int n = 0; n < 8; n++) {
        int3 corner = origin.xyz + int3(n & 1, (n >> 1) & 1, n >> 2);
        c[n] = tableAt(corner.x, corner.y, corner.z); angles[n] = angleAt(corner.x, corner.y, corner.z);
    }
    // skip incoherent cells (the rays have spread apart, the interpolant is not a ray bundle): Newton would
    // reject solutions there anyway, and they are large, so they would take over many nodes
    for (int plane = 0; plane < 8; plane += 4) {
        float3 d0 = normalize(c[plane].xyz), d1 = normalize(c[plane + 1].xyz), d2 = normalize(c[plane + 2].xyz), d3 = normalize(c[plane + 3].xyz);
        if (min(min(dot(d0, d1), dot(d0, d2)), min(dot(d0, d3), dot(d1, d2))) < 0.866) return;
    }
    int bandMin = 99, bandMax = -1;
    for (int n = 0; n < 8; n++) {
        bandMin = min(bandMin, int(max(angles[n] - BAND_OVERLAP, 0.)/BAND_WIDTH));
        bandMax = max(bandMax, int((angles[n] + BAND_OVERLAP)/BAND_WIDTH));
    }
    for (int band = bandMin; band <= min(bandMax, BAND_COUNT - 1); band++) {
        // bounding box of the corners in node space (phi unwrapped next to corner 0)
        int3 size = gridSize(band);
        float3 lo = 1e9, hi = -1e9, g0 = gridCoords(c[0].xyz, band);
        for (int n = 0; n < 8; n++) {
            float3 g = gridCoords(c[n].xyz, band);
            g.z -= size.z*round((g.z - g0.z)/size.z);
            lo = min(lo, g); hi = max(hi, g);
        }
        int3 first = int3(ceil(lo)), last = int3(floor(hi));
        first.xy = max(first.xy, 0); last.xy = min(last.xy, size.xy - 1);
        int3 count = last - first + 1;
        if (any(count <= 0) || count.x*count.y*count.z > 1024) continue;
        for (int i = 0; i < count.x*count.y*count.z; i++) {
            int3 node = first + int3(i%count.x, (i/count.x)%count.y, i/(count.x*count.y));
            float3 X = nodePosition(band, node);
            // local coordinates of the node inside the cell: Newton on the trilinear interpolant
            float3 f = 0.5;
            TableSample s = interpolateCell(c, angles, f);
            for (int it = 0; it < 6; it++) {
                f -= solve3(s.dPsi, s.dRow, s.dK, s.position - X);
                s = interpolateCell(c, angles, clamp(f, -0.5, 1.5));
            }
            bool inside = all(f >= -0.01) && all(f <= 1.01) && length(s.position - X) < 1e-3*length(X) + 1e-4;
            // bands overlap too (by BAND_OVERLAP radians of swept angle)
            if (!inside || any(isnan(f)) || abs(s.sweptAngle - (band + 0.5)*BAND_WIDTH) > 0.5*BAND_WIDTH + BAND_OVERLAP) continue;
            // smaller cells (in node units) win; the cell's octant containing the node rides along
            // key: cell + 1 and the octant the node is in; slot pair: lowest and highest key (via ~key)
            uint3 half = uint3(f > 0.5);
            uint key = (uint(cell + 1) << 3) | (half.x << 2) | (half.y << 1) | half.z;
            // the sectors overlap by SECTOR_OVERLAP psi bins, so sheets crossing a sector edge stay whole in one
            for (int edge = -1; edge <= 1; edge += 2) {
                int sector = ((origin.x + edge*SECTOR_OVERLAP + PSI_COUNT)%PSI_COUNT)*SECTOR_COUNT/PSI_COUNT;
                if (edge == 1 && sector == ((origin.x - SECTOR_OVERLAP + PSI_COUNT)%PSI_COUNT)*SECTOR_COUNT/PSI_COUNT) break;
                inverseGrid[nodeIndex(band, 2*sector, node)].min(key);
                inverseGrid[nodeIndex(band, 2*sector + 1, node)].min(~key);
            }
        }
    }
}
// Newton starts for table-frame position X from a band's grid: the blend of the 8 surrounding nodes and the
// nearest filled node (for where the blend is meaningless: folds, the edge of the band's sheet)
// only the node nearest to X (one load): enough to follow the light delay
float4 nearestSeed(int band, int slot, float3 X) {
    int3 size = gridSize(band);
    float3 g = gridCoords(X, band);
    if (g.x < -0.5 || g.x > size.x - 0.5) return 0;
    int3 node = int3(round(g));
    node.xy = clamp(node.xy, 0, size.xy - 1);
    return decodeNode(inverseGrid[nodeIndex(band, slot, node)].load(), slot);
}
void gridSeeds(int band, int slot, float3 X, out float4 blended, out float4 nearest) {
    int3 size = gridSize(band);
    float3 g = gridCoords(X, band);
    blended = 0; nearest = 0;
    if (g.x < 0 || g.x > size.x - 1) return;
    int3 g0 = int3(floor(g));
    g0.xy = min(g0.xy, size.xy - 2);
    float3 f = g - g0;
    float4 q[8];
    float w[8];
    float best = 0;
    for (int n = 0; n < 8; n++) {
        int3 corner = int3(n & 1, (n >> 1) & 1, n >> 2);
        q[n] = decodeNode(inverseGrid[nodeIndex(band, slot, g0 + corner)].load(), slot);
        w[n] = (corner.x == 1 ? f.x : 1 - f.x)*(corner.y == 1 ? f.y : 1 - f.y)*(corner.z == 1 ? f.z : 1 - f.z);
        if (q[n].w != 0 && w[n] >= best) { best = w[n]; nearest = q[n]; }
    }
    blended = blendChartPoints(q, w);
}
[shader("compute")]
[WorkgroupCount(1024, 1, 1)] [numthreads(64, 1, 1)]
void buildInverseGrid(uint3 id : SV_DispatchThreadID) {
    if (!tableDirty()) return;
    for (int cell = int(id.x); cell < CELL_COUNT; cell += 1024*64) rasteriseCell(cell);
}
// barycentric weights of p in triangle (A, B, C)
float3 barycentric(float2 A, float2 B, float2 C, float2 p) {
    float area = (B.x - A.x)*(C.y - A.y) - (B.y - A.y)*(C.x - A.x);
    float wa = ((B.x - p.x)*(C.y - p.y) - (B.y - p.y)*(C.x - p.x))/area;
    float wb = ((C.x - p.x)*(A.y - p.y) - (C.y - p.y)*(A.x - p.x))/area;
    return float3(wa, wb, 1 - wa - wb);
}
// ============================================================================ 3. background
// The sky direction / horizon point of every pixel is a bicubic (Catmull-Rom) interpolation of the per-ray
// landing records over the chart (~0.02 px from a per-pixel trace), cached in pass layer 0 as (theta, phi, T,
// kind) and redone only when the view changes. kind: 0 none, 1 sky, 2 horizon, stored as kind + 1 + 2^-12 so
// that main_image can tell whether the pass textures are float32.
float4 catmullRomWeights(float t) {
    return float4(-0.5, 1.5, -1.5, 0.5)*t*t*t + float4(1, -2.5, 2, -0.5)*t*t + float4(-0.5, 0, 0.5, 0)*t + float4(0, 1, 0, 0);
}
// (direction, T) interpolated from the records starting at recordBase; w < 0 = no data
float4 chartInterpolate(float4 chartPoint, int recordBase) {
    float px = wrapPsi(chartPoint.x); float py = clamp(chartPoint.y, 0., ELL_COUNT - 1.);
    int i0 = int(px); int j0 = min(int(py), ELL_COUNT - 2);
    float2 f = float2(px - i0, py - j0); int rowBase = rowOffset(chartPoint.w);
    float4 wx = catmullRomWeights(f.x), wy = catmullRomWeights(f.y);
    float4 sum = 0; bool missing = false;
    for (int n = 0; n < 16; n++) {
        int row = rowBase + clamp(j0 - 1 + n/4, 0, ELL_COUNT - 1);
        float4 v = rayRecords[recordBase + row*PSI_COUNT + (i0 - 1 + n%4 + PSI_COUNT)%PSI_COUNT];
        missing = missing || v.w < 0; sum += wx[n%4]*wy[n/4]*v;
    }
    if (missing) {
        // next to missing rays: bilinear over the valid corners
        sum = 0; float weightSum = 0;
        for (int n = 0; n < 4; n++) {
            float4 v = rayRecords[recordBase + (rowBase + j0 + n/2)*PSI_COUNT + (i0 + n%2)%PSI_COUNT];
            float w = (n%2 == 1 ? f.x : 1 - f.x)*(n/2 == 1 ? f.y : 1 - f.y);
            if (v.w < 0) continue;
            sum += w*v; weightSum += w;
        }
        if (weightSum < 0.5) return float4(0, 0, 0, -1);
        sum /= weightSum;
    }
    return float4(normalize(sum.xyz), sum.w);
}
static const int KIND_NONE = 0, KIND_SKY = 1, KIND_HORIZON = 2;
float encodeKind(int kind) { return kind + 1 + exp2(-12.); }
int decodeKind(float code) { return int(round(code)) - 1; }
float2 pixelToScreen(float2 pixel) {
    return float2(pixel.x/SCREEN_SIZE.x*2 - 1, 1 - pixel.y/SCREEN_SIZE.y*2)/P.projection;
}
// one pixel of the cache; run by the solve dispatch (section 6) when the view changed
void cacheBackground(int2 pixel) {
    float4 chartPoint = screenToChart(pixelToScreen(pixel + 0.5));
    bool outside = chartPoint.w > 0;
    float4 landing = chartInterpolate(chartPoint, outside ? SKY_RECORDS : HORIZON_RECORDS);
    int kind = landing.w < 0 ? KIND_NONE : outside ? KIND_SKY : KIND_HORIZON;
    float theta = atan2(length(landing.xz), landing.y); float phi = atan2(-landing.z, landing.x);
    pass_out[int3(pixel, BACKGROUND_LAYER)] = float4(theta, phi, landing.w, encodeKind(kind));
}
// ============================================================================ 4. inverting the table
// Trilinear in (psi, row, k); the derivatives are exact for the interpolant.
struct TableSample {
    float3 position, dPsi, dRow, dK;   // dK is also the ray tangent
    float sweptAngle, delay;
    float3 dDelay;                     // d delay / d (psi, row, k)
};
// trilinear interpolation of 8 corners (bit 0 = psi + 1, bit 1 = row + 1, bit 2 = k + 1) at local f
TableSample interpolateCell(float4 c[8], float angles[8], float3 f) {
    float4 c00 = lerp(c[0], c[1], f.x), c10 = lerp(c[2], c[3], f.x);
    float4 c01 = lerp(c[4], c[5], f.x), c11 = lerp(c[6], c[7], f.x);
    float4 c0 = lerp(c00, c10, f.y), c1 = lerp(c01, c11, f.y); float4 value = lerp(c0, c1, f.z);
    float4 dPsi = lerp(lerp(c[1] - c[0], c[3] - c[2], f.y), lerp(c[5] - c[4], c[7] - c[6], f.y), f.z);
    float4 dRow = lerp(c10 - c00, c11 - c01, f.z), dK = c1 - c0;
    float a0 = lerp(lerp(angles[0], angles[1], f.x), lerp(angles[2], angles[3], f.x), f.y);
    float a1 = lerp(lerp(angles[4], angles[5], f.x), lerp(angles[6], angles[7], f.x), f.y);
    return { value.xyz, dPsi.xyz, dRow.xyz, dK.xyz, lerp(a0, a1, f.z), value.w, float3(dPsi.w, dRow.w, dK.w) };
}
TableSample evaluateTable(float4 chartPoint, bool withAngle = false) {   // the swept angle costs 8 more loads
    float px = wrapPsi(chartPoint.x); float py = clamp(chartPoint.y, 0., ELL_COUNT - 1.);
    float pz = clamp(chartPoint.z, 0., STEP_COUNT - 1.); int i0 = min(int(px), PSI_COUNT - 1);
    int rowLocal = min(int(py), ELL_COUNT - 2); int k0 = min(int(pz), STEP_COUNT - 2);
    float4 c[8];
    float angles[8];
    for (int n = 0; n < 8; n++) {
        int3 corner = int3(i0 + (n & 1), rowLocal + rowOffset(chartPoint.w) + ((n >> 1) & 1), k0 + (n >> 2));
        c[n] = tableAt(corner.x, corner.y, corner.z); angles[n] = withAngle ? angleAt(corner.x, corner.y, corner.z) : 0;
    }
    return interpolateCell(c, angles, float3(px - i0, py - rowLocal, pz - k0));
}
// Light travel time. The table's rays are the time-mirrored arriving rays, so its Kerr-Schild time is the outgoing
// time t_out = t_BL - hTime(r) of the arriving light, and the light reaching the camera now (BL time sceneTime)
// left the table position of radius r at t_BL = sceneTime - delay - hTime(r_camera) + hTime(r). hTime diverges at
// the horizon: light from close to it takes ever longer (objects falling in freeze and fade).
float hTime(float r) {
    float rPlus = P.horizonRadius, rMinus = 2 - rPlus;
    return (2*rPlus*log(abs(r - rPlus)) - 2*rMinus*log(abs(r - rMinus)))/(rPlus - rMinus);
}
float lightDelay(float delay, float r) { return toggleOn(8) ? delay + P.cameraTimeShift - hTime(r) : 0; }
float3 radiusGradient(float3 X) {   // of blRadius: from r^4 - (|X|^2 - a^2) r^2 - a^2 y^2 = 0
    float a2 = spin()*spin(), r = blRadius(X);
    return (r*r*X + float3(0, a2*X.y, 0))/(2*r*r*r - (dot(X, X) - a2)*r);
}
float3 solve3(float3 a, float3 b, float3 c, float3 F) {   // solves [a b c] x = F
    float3 bc = cross(b, c);
    return float3(dot(bc, F), dot(cross(c, a), F), dot(cross(a, b), F))/dot(a, bc);
}
// scale for tolerances: distance from the camera
float cameraDistance(float3 X, float r) { return clamp(length(X - rayTable[0].xyz), 0.25, max(r, 2.)); }
struct NewtonResult { float4 chartPoint; bool ok; float sweptAngle; float3 rayDirection; float emitTime; };
// Residual of chart point q for an emitter: its table position minus where the emitter was when the light left.
struct Residual { TableSample s; float3 F; float emitTime; float3 velocity; };
Residual residualAt(float4 q, Emitter e) {
    TableSample s = evaluateTable(q);
    float emitTime = P.sceneTime - lightDelay(s.delay, blRadius(s.position));
    Body b = emitterBody(e, emitTime);
    return { s, s.position - targetOf(b.position).xyz, emitTime, targetVelocity(b.position, b.velocity) };
}
// Newton in (psi, row, k) for the retarded position: F(q) = X(q) - Y(t_emit(q)), so the Jacobian gains the
// rank-one term dY/dt (x) d delay/dq (the emitter velocity times the delay gradient, incl. hTime). It stays
// invertible while the emitter is slower than light. Each iteration tries three candidates, keeps the best:
//   0: the plain step
//   1: the same step as a linear move in sigma (right near the camera, crosses the critical curve)
//   2: near the shadow centre, a Cartesian step (passes through the centre)
// Gives up early when the error stalls, or (with known images passed in) when it heads for one of them.
NewtonResult newton(float4 chartPoint, Emitter e, float tolerance, int maxIterations, float minCos = 0.866, int knownCount = 0,
                    float4 known[MAX_IMAGES] = {}, float maxStartError = 1e30) {
    float4 q = chartPoint; Residual current = residualAt(q, e);
    float error2 = dot(current.F, current.F), earlierError2 = error2;
    bool converged = false;
    NewtonResult failed = { q, false, 0, 0, 0 };
    if (error2 > maxStartError*maxStartError) return failed;   // a seed this far off rarely converges
    for (int it = 0; it < maxIterations; it++) {
        if (it == 2) for (int n = 0; n < knownCount; n++) if (sameImage(known[n], q, 2)) return failed;
        if (it%3 == 2) {
            if (it > 3 && error2 > 0.5*earlierError2 && error2 > 400*tolerance*tolerance) return failed;   // stalled
            earlierError2 = error2;
        }
        // stop one iteration after reaching the tolerance (or right away well below it)
        if (error2 < tolerance*tolerance) {
            if (converged || error2 < 0.0025*tolerance*tolerance) break;
            converged = true;
        }
        TableSample s = current.s;
        if (toggleOn(8)) {
            float3 velocity = current.velocity;
            float r = blRadius(s.position);
            float3 gradient = radiusGradient(s.position)*2*r/delta(r);
            float3 dDelay = s.dDelay - float3(dot(gradient, s.dPsi), dot(gradient, s.dRow), dot(gradient, s.dK));
            s.dPsi += velocity*dDelay.x; s.dRow += velocity*dDelay.y; s.dK += velocity*dDelay.z;
        }
        float3 F = current.F;
        float3 newtonStep = solve3(s.dPsi, s.dRow, s.dK, F);
        if (any(isnan(newtonStep))) break;
        newtonStep.xz /= max(1, max(abs(newtonStep.x)*0.5, abs(newtonStep.z)*0.25));
        float maxRowStep = 3/ellPerRow(q.y, q.w);
        newtonStep.y = clamp(newtonStep.y, -maxRowStep, maxRowStep);
        float kMin = max(0.25*q.z, 1e-3);
        float4 candidates[3];   // w = 2 marks "no candidate"
        candidates[0] = float4(q.x - newtonStep.x, clamp(q.y - newtonStep.y, 0., ELL_COUNT - 1.), clamp(q.z - newtonStep.z, kMin, STEP_COUNT - 1.), q.w);
        candidates[1] = candidates[2] = float4(0, 0, 0, 2);
        float ellStep = -newtonStep.y*ellPerRow(q.y, q.w);
        if (abs(ellStep) > 0.2) {
            float2 rowSide = rowFromSigma(sigmaOf(q)*(1 + ellStep), candidates[0].x);
            candidates[1] = float4(candidates[0].x, rowSide.x, candidates[0].z, rowSide.y);
        }
        if (nearCentre(q)) {
            // chain rule from (psi index, row) to the Cartesian c = (1 - e)(cos psi, sin psi)
            float t = -PI + 2*PI*q.x/PSI_COUNT; float psi = t + 0.5*PSI_WARP*sin(2*t);
            float radius = 1 - insideDepth(q.y);
            float3 dRadius = -s.dRow/INNER_LIN_STEP;
            float3 dAngle = s.dPsi/(2*PI/PSI_COUNT*(1 + PSI_WARP*cos(2*t)));
            float3 dCx = cos(psi)*dRadius - sin(psi)/radius*dAngle;
            float3 dCy = sin(psi)*dRadius + cos(psi)/radius*dAngle;
            float3 d = solve3(dCx, dCy, s.dK, F); d /= max(1, max(length(d.xy)/0.25, abs(d.z)*0.25));
            float2 psiRow = fromCentreXY(radius*float2(cos(psi), sin(psi)) - d.xy);
            if (!any(isnan(d))) candidates[2] = float4(psiRow, clamp(q.z - d.z, kMin, STEP_COUNT - 1.), -1);
        }
        float bestError = 1e30;
        for (int n = 0; n < 3; n++) {
            if (candidates[n].w > 1) continue;
            Residual candidate = residualAt(candidates[n], e);
            float candidateError = dot(candidate.F, candidate.F);
            if (candidateError >= bestError) continue;
            bestError = candidateError; q = candidates[n];
            current = candidate;
        }
        error2 = bestError;
    }
    q.x = wrapPsi(q.x);
    NewtonResult result = { q, false, 0, current.s.dK, current.emitTime };
    if (error2 >= tolerance*tolerance) return result;
    result.sweptAngle = evaluateTable(q, true).sweptAngle;
    // reject cells whose rays land far apart (a fold: the interpolated bundle is not a ray)
    int i0 = int(q.x); int j0 = min(int(q.y), ELL_COUNT - 2) + rowOffset(q.w);
    int k0 = min(int(q.z), STEP_COUNT - 2); float3 d[4];
    for (int n = 0; n < 4; n++) d[n] = normalize(tableAt(i0 + (n & 1), j0 + (n >> 1), k0).xyz);
    result.ok = min(min(dot(d[0], d[1]), dot(d[0], d[2])), min(dot(d[0], d[3]), dot(d[1], d[2]))) > minCos;
    return result;
}
// image order from the swept angle: direct (< pi), 1st loop, 2nd loop, else -1
int orderOf(float sweptAngle) {
    int band = int(sweptAngle/PI);
    return band == 0 ? 0 : band <= 2 ? 1 : band <= 4 ? 2 : -1;
}
bool toggleOn(uint bit) { return ((P.toggles >> bit) & 1) == 1; }
bool orderOn(int order) { return order >= 0 && toggleOn(5 + order); }
float orderGain(int order) { return order == 0 ? 1 : order == 1 ? 0.85 : 0.7; }
// ============================================================================ 5. scene
float hash1(float n) { return frac(sin(n)*43758.5453); }
float hashUint(uint x) {
    x = (x ^ (x >> 16))*0x7feb352d; x = (x ^ (x >> 15))*0x846ca68b;
    x ^= x >> 16;
    return (x >> 8)/16777216.0;
}
struct Body { float3 position, velocity, normal, centre; };
// Every emitter moves on an analytic path in BL time t, slower than light: the retarded-time solve has one root
// per ray branch only then (a faster emitter could be seen several times along one ray, or not at all). Speeds
// count relative to the local ZAMO, which is dragged around at omega: in the ergosphere nothing can stand still.
float3 equatorFrame(float r) {   // (omega, local speed per unit of (Omega - omega), local speed per coord speed)
    float a = spin(), D = max(delta(r), 1e-6), A = sqr(r*r + a*a) - a*a*D;
    float lapse = sqrt(r*r*D/A);
    return float3(2*a*r/A, sqrt(A)/(r*lapse), max(sqrt(A)/(r*r), r/sqrt(D))/lapse);
}
// prograde Keplerian angular velocity, kept within 0.95 c of the ZAMO
float orbitRate(float radius) {
    float3 frame = equatorFrame(radius);
    return clamp(1/(pow(radius, 1.5) + spin()), frame.x - 0.95/frame.y, frame.x + 0.95/frame.y);
}
// a circle in a tilted plane: set up once, evaluated at any time
struct Orbit { float3 u, v, n; float radius, rate, phase; };
Orbit makeOrbit(float radius, float phase, float tilt, float node) {
    float3 n = float3(sin(tilt)*cos(node), cos(tilt), sin(tilt)*sin(node));
    float3 u = normalize(cross(abs(n.y) < 0.99 ? float3(0, 1, 0) : float3(1, 0, 0), n));
    return { u, cross(n, u), n, radius, orbitRate(radius), phase };
}
Body orbitBody(Orbit o, float t) {
    float angle = o.phase + t*o.rate;
    float3 position = o.radius*(cos(angle)*o.u + sin(angle)*o.v);
    return { position, o.radius*o.rate*(-sin(angle)*o.u + cos(angle)*o.v), o.n, position };
}
Body orbit(float radius, float phase, float tilt, float node, float t) { return orbitBody(makeOrbit(radius, phase, tilt, node), t); }
// 2D value noise and 4 octaves of it, in [0, 1]
float valueNoise(float2 p) {
    float2 i = floor(p), f = p - i;
    f = f*f*(3 - 2*f);
    float2 h = float2(dot(i, float2(127.1, 311.7)), dot(i + float2(1, 0), float2(127.1, 311.7)));
    float a = hash1(h.x), b = hash1(h.y), c = hash1(dot(i + float2(0, 1), float2(127.1, 311.7))), d = hash1(dot(i + 1, float2(127.1, 311.7)));
    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
}
float fbm(float2 p) {
    float sum = 0, amplitude = 0.5;
    for (int octave = 0; octave < 4; octave++) { sum += amplitude*valueNoise(p); p = p*2.03 + 17.1; amplitude *= 0.5; }
    return sum/0.9375;
}
// Disk particles: surface density falling off with radius from the ISCO out to 18 M, modulated by noise at t = 0
// (rejection sampling of the angle, a few tries per particle). Keplerian shear then winds the clumps into spirals.
Orbit particleOrbit(int index) {
    uint i = uint(index);
    float u = (index + hashUint(8*i + 1))/PARTICLE_COUNT;   // radius stratified by index: neighbouring threads alike
    float radius = lerp(P.innerEdge, 18., u*u), phase = 0;
    for (uint attempt = 0; attempt < 6; attempt++) {
        uint h = 8*i + 3*attempt;
        phase = 2*PI*hashUint(h + 2);
        float density = smoothstep(0.35, 0.7, fbm(radius*float2(cos(phase), sin(phase))*0.35 + 3.7));
        if (hashUint(h + 3) < density) break;
    }
    float tilt = (hashUint(8*i + 5) - 0.5)*0.11*(1 + 0.08*radius);
    return makeOrbit(radius, phase, tilt, 2*PI*hashUint(8*i + 6));
}
Body particle(int index, float t) { return orbitBody(particleOrbit(index), t); }
float3 rotateAround(float3 v, float3 axis, float angle) {
    return v*cos(angle) + cross(axis, v)*sin(angle) + axis*dot(axis, v)*(1 - cos(angle));
}
// small spinning spheres on inclined orbits; the spin gets what the speed limit leaves (at the closest point)
Body objectVertex(int object, int vertex, float t) {
    float radius = lerp(6., 11., hash1(object*3.1 + 0.7)), size = 0.55 + 0.35*hash1(object*13.);
    float phase = 2*PI*object/OBJECT_COUNT + 0.4*hash1(object + 5.);
    Body body = orbit(radius, phase, (hash1(object*7.3) - 0.5)*1.1, 2*PI*hash1(object*11.1), t);
    float theta = PI*(vertex/(SPHERE_SEGS + 1))/SPHERE_RINGS;
    float phi = 2*PI*(vertex%(SPHERE_SEGS + 1))/SPHERE_SEGS;
    float3 axis = normalize(float3(hash1(object*17.), hash1(object*29.), hash1(object*41.)) - 0.5 + 1e-3);
    float3 localNormal = float3(sin(theta)*cos(phi), cos(theta), sin(theta)*sin(phi));
    float3 frame = equatorFrame(radius);
    float orbitSpeed = abs(orbitRate(radius) - frame.x)*frame.y;   // what the orbit uses of the speed limit
    float spinRate = min(0.6, max(0.95 - orbitSpeed, 0.)/(equatorFrame(radius - size).z*size));
    body.normal = rotateAround(localNormal, axis, object*37 + t*spinRate);
    body.position += body.normal*size;
    body.velocity += spinRate*cross(axis, body.normal*size);
    return body;
}
Body pointBody(int pointId, float t) {
    if (pointId < PARTICLE_COUNT) return particle(pointId, t);
    int local = pointId - PARTICLE_COUNT;
    return objectVertex(local/OBJECT_VERTS, local%OBJECT_VERTS, t);
}
// An emitter: one point, or a blend of up to 3 (along mesh edges and faces), all taken at the same time. A
// particle carries its orbit, set up once (Newton evaluates emitters at many times).
struct Emitter { int3 ids; float3 weights; Orbit orbit; };
Emitter pointEmitter(int pointId) {
    Orbit none = { 0, 0, 0, 0, 0, 0 };
    return { int3(pointId), float3(1, 0, 0), pointId < PARTICLE_COUNT ? particleOrbit(pointId) : none };
}
Emitter meshEmitter(int3 ids, float3 weights) {
    Orbit none = { 0, 0, 0, 0, 0, 0 };
    return { ids, weights, none };
}
Body emitterBody(Emitter e, float t) {
    if (e.orbit.radius > 0) return orbitBody(e.orbit, t);
    Body sum = { 0, 0, 0, 0 };
    for (int n = 0; n < 3; n++) {
        if (e.weights[n] == 0) continue;
        Body b = pointBody(e.ids[n], t);
        sum.position += e.weights[n]*b.position; sum.velocity += e.weights[n]*b.velocity;
        sum.normal += e.weights[n]*b.normal; sum.centre += e.weights[n]*b.centre;
    }
    return sum;
}
// ============================================================================ 6. per-point solve
// Up to 8 images per point, found from scratch every frame: Newton from each (band, slot) of the inverse grid,
// plus a straight line-of-sight seed for points near the camera. Stored in the pass textures for the raster.
bool sameImage(float4 a, float4 b, float slack = 1) {
    if (a.w == 0 || a.w != b.w) return false;
    if (nearCentre(a) && nearCentre(b)) return length(toCentreXY(a) - toCentreXY(b)) < 0.02*slack && abs(a.z - b.z) < slack;
    return abs(psiDifference(a.x, b.x)) < 0.6*slack && abs(a.y - b.y) < 0.6*slack && abs(a.z - b.z) < slack;
}
// emitter velocity in BL, v = (dr, dtheta, dphi)/dt from a finite difference, and g_ab u^a u^b/(u^t)^2 in w:
// negative for a timelike worldline (slower than light): the exact check behind orbitRate's limit
float4 blVelocity(float3 X, float3 velocity) {
    float a = spin(); float3 Xnext = X + velocity*1e-3; float r = blRadius(X), rNext = blRadius(Xnext);
    float theta = polarAngle(X, r), thetaNext = polarAngle(Xnext, rNext);
    float dPhi = atan2(-Xnext.z, Xnext.x) - atan2(-X.z, X.x);
    dPhi -= 2*PI*floor((dPhi + PI)/(2*PI));
    float3 v = float3(rNext - r, thetaNext - theta, dPhi)/1e-3;
    float sin2 = sin(theta)*sin(theta); float Sigma = r*r + a*a*cos(theta)*cos(theta);
    float Delta = delta(r); float A = sqr(r*r + a*a) - a*a*Delta*sin2;
    float norm = -(Delta - a*a*sin2)/Sigma - 2*a*sin2*(r*r + a*a - Delta)/Sigma*v.z;
    norm += sin2*A/Sigma*v.z*v.z + Sigma/Delta*v.x*v.x + Sigma*v.y*v.y;
    return float4(v, norm);
}
// g = (p.u_camera)/(p.u_emitter), exact for the ray through screen xy
float redshift(float2 screenXY, float3 X, float3 velocity, float3 rayDirection) {
    float4 constants = rayConstants(screenXY);
    float lambda = constants.x, eta = constants.y, a = spin();
    float4 v4 = blVelocity(X, velocity);
    float3 v = v4.xyz; float norm = v4.w;
    if (norm >= -1e-4) return 1;
    float r = blRadius(X), theta = polarAngle(X, r), phi = atan2(-X.z, X.x);
    float Delta = delta(r);
    float radialPotential = sqr(r*r + a*a - a*lambda) - Delta*(eta + sqr(lambda - a));
    float polarPotential = eta + a*a*cos(theta)*cos(theta) - lambda*lambda/sqr(tan(theta));
    // signs of the ray's radial and polar motion from its direction
    float3 thetaDirection = float3(cos(theta)*cos(phi), -sin(theta), -cos(theta)*sin(phi));
    float radialSign = sign(dot(rayDirection, X));
    float polarSign = sign(dot(rayDirection, thetaDirection));
    float pu = -1 - radialSign*sqrt(max(radialPotential, 0.))/Delta*v.x - polarSign*sqrt(max(polarPotential, 0.))*v.y + lambda*v.z;
    return clamp((-1 + P.cameraOmega*lambda)/P.lapse/(pu*rsqrt(-norm)), 0.02, 20.);
}
// Image storage in the pass textures (written to pass_out, read from pass_in, which compute.toys refreshes after
// every dispatch). Records run row-major through SCREEN_LAYER, then CHART_LAYER. A particle image is one texel:
// (x, y, g, order*256 + k), w < 0 = none. A mesh vertex image is two: its chart point and (x, y, g, T).
int3 recordTexel(int record) {
    int width = int(SCREEN_SIZE.x), layerSize = width*int(SCREEN_SIZE.y);
    int layer = record < layerSize ? SCREEN_LAYER : CHART_LAYER;
    record %= layerSize;
    return int3(record%width, record/width, layer);
}
int vertexRecord(int pointId, int slot) { return PARTICLE_COUNT*PARTICLE_IMAGES + 2*((pointId - PARTICLE_COUNT)*MAX_IMAGES + slot); }
bool imagesFit(int pointId) {
    int end = pointId < PARTICLE_COUNT ? (pointId + 1)*PARTICLE_IMAGES : vertexRecord(pointId, MAX_IMAGES);
    return end <= 2*int(SCREEN_SIZE.x*SCREEN_SIZE.y);
}
float4 loadParticleImage(int pointId, int slot) {
    return imagesFit(pointId) ? pass_in[recordTexel(pointId*PARTICLE_IMAGES + slot)] : float4(0, 0, 0, -1);
}
float4 loadChartPoint(int pointId, int slot) { return imagesFit(pointId) ? pass_in[recordTexel(vertexRecord(pointId, slot))] : 0; }
float4 loadScreenData(int pointId, int slot) { return pass_in[recordTexel(vertexRecord(pointId, slot) + 1)]; }
struct ImageList {
    float4 chartPoint[MAX_IMAGES]; float sweptAngle[MAX_IMAGES];
    float3 rayDirection[MAX_IMAGES]; float emitTime[MAX_IMAGES];
    bool contains(float4 q, int count = MAX_IMAGES, float slack = 1) {
        bool found = false;
        for (int s = 0; s < count; s++) found = found || sameImage(chartPoint[s], q, slack);
        return found;
    }
    [mutating] void put(int s, NewtonResult result) {
        chartPoint[s] = result.chartPoint; sweptAngle[s] = result.sweptAngle;
        rayDirection[s] = result.rayDirection; emitTime[s] = result.emitTime;
    }
    [mutating] void add(NewtonResult result) {   // into the first free slot
        int s = 0;
        while (s < MAX_IMAGES && chartPoint[s].w != 0) s++;
        if (s < MAX_IMAGES) put(s, result);
    }
};
void solvePoint(int pointId) {
    if (pointId >= POINT_COUNT || !imagesFit(pointId) || !toggleOn(pointId < PARTICLE_COUNT ? 0 : 1)) return;
    Emitter e = pointEmitter(pointId);
    float4 target = targetOf(emitterBody(e, P.sceneTime).position);   // where it is now
    float tolerance = 3e-4*cameraDistance(target.xyz, target.w);
    ImageList list;
    for (int s = 0; s < MAX_IMAGES; s++) list.put(s, { 0, false, 0, 0, 0 });
    // Retarded images, per grid layer: look up the point's position `back` M ago, read the light delay of that
    // seed, move `back` to it and repeat (converges for emitters slower than light, like the retarded time
    // itself). Where the layer has no seed, step further back. Then Newton from the seed for the moving target.
    float3 probes[6];   // positions WORLDLINE_STEP apart back in time, shared by all layers
    for (int n = 0; n < 6; n++) probes[n] = n == 0 ? target.xyz : targetOf(emitterBody(e, P.sceneTime - n*WORLDLINE_STEP).position).xyz;
    for (int layer = 0; layer < BAND_COUNT*SLOT_COUNT; layer++) {
        float back = 0;
        int probe = 0;
        float3 X = target.xyz;
        bool anySeed = !toggleOn(8);
        for (int pass = 0; pass < DELAY_PASSES && toggleOn(8); pass++) {
            X = probe >= 0 && probe < 6 ? probes[probe] : targetOf(emitterBody(e, P.sceneTime - back).position).xyz;
            float4 seed = nearestSeed(layer/SLOT_COUNT, layer%SLOT_COUNT, X);   // one load per pass
            if (seed.w == 0 && pass >= 5) break;
            if (seed.w == 0) { back += WORLDLINE_STEP; probe = probe >= 0 ? probe + 1 : -1; continue; }
            anySeed = true;
            float4 corner = tableAt(int(seed.x), int(seed.y) + rowOffset(seed.w), int(seed.z));
            float delay = lightDelay(corner.w, blRadius(corner.xyz));
            if (abs(delay - back) < 0.5) break;
            back = delay; probe = -1;
        }
        if (!anySeed) continue;   // nothing in this layer anywhere along the worldline searched
        float4 seeds[2] = { nearestSeed(layer/SLOT_COUNT, layer%SLOT_COUNT, X), float4(0) };
        if (seeds[0].w == 0) gridSeeds(layer/SLOT_COUNT, layer%SLOT_COUNT, X, seeds[0], seeds[1]);
        for (int attempt = 0; attempt < 2; attempt++) {
            if (seeds[attempt].w == 0 || list.contains(seeds[attempt], MAX_IMAGES, 3)) continue;   // found already
            NewtonResult result = newton(seeds[attempt], e, tolerance, 16, 0.866, MAX_IMAGES, list.chartPoint, 1.2*target.w);
            if (!result.ok || list.contains(result.chartPoint)) continue;
            list.add(result);
            break;
        }
    }
    // very close to the camera the grid is coarse: also seed from the straight line of sight
    float3 cameraPos = rayTable[0].xyz; float3 toTarget = target.xyz - cameraPos;
    float3 radialDir = normalize(cameraPos);
    float3 phiDir = normalize(cross(float3(0, 1, 0), radialDir));
    float3 thetaDir = cross(phiDir, radialDir); float depth = dot(toTarget, radialDir);
    float targetDistance = length(toTarget);
    if (targetDistance < 0.6*length(cameraPos) && depth < -0.03*targetDistance) {
        float4 chartPoint = screenToChart(float2(dot(toTarget, phiDir)/(-depth), dot(toTarget, thetaDir)/depth));
        chartPoint.y = clamp(chartPoint.y, 0., ELL_COUNT - 1.);
        // walk the ray to the sample pair that brackets the target's distance
        for (int k = 1; k < 64; k++) {
            float d0 = length(evaluateTable(float4(chartPoint.xy, k - 1, chartPoint.w)).position - cameraPos);
            float d1 = length(evaluateTable(float4(chartPoint.xy, k, chartPoint.w)).position - cameraPos);
            if (d1 < targetDistance) continue;
            float kStart = k - 1 + clamp((targetDistance - d0)/max(d1 - d0, 1e-6), 0., 1.);
            NewtonResult result = newton(float4(chartPoint.xy, kStart, chartPoint.w), e, tolerance, 24);
            if (result.ok && int(result.sweptAngle/PI) == 0 && !list.contains(result.chartPoint)) list.add(result);
            break;
        }
    }
    // colour shift from the velocity at emission; images of a (wrongly defined) faster-than-light emitter dropped
    int kept = 0;
    for (int s = 0; s < MAX_IMAGES; s++) {
        float4 q = list.chartPoint[s]; float2 xy = chartToScreen(q);
        Body then = emitterBody(e, list.emitTime[s]);
        float3 position = toTableFrame(then.position), velocity = toTableFrame(then.velocity);
        if (q.w != 0 && blVelocity(position, velocity).w >= -1e-4) q = 0;
        float g = q.w == 0 ? 1 : redshift(xy, position, velocity, list.rayDirection[s]);
        if (pointId >= PARTICLE_COUNT) {
            pass_out[recordTexel(vertexRecord(pointId, s))] = q;
            pass_out[recordTexel(vertexRecord(pointId, s) + 1)] = q.w == 0 ? 0 : float4(xy, g, list.sweptAngle[s]);
        } else if (q.w != 0 && orderOf(list.sweptAngle[s]) >= 0 && kept < PARTICLE_IMAGES) {
            pass_out[recordTexel(pointId*PARTICLE_IMAGES + kept)] = float4(xy, g, orderOf(list.sweptAngle[s])*256 + q.z);
            kept++;
        }
    }
    for (int s = kept; s < PARTICLE_IMAGES && pointId < PARTICLE_COUNT; s++) pass_out[recordTexel(pointId*PARTICLE_IMAGES + s)] = float4(0, 0, 0, -1);
}
// One dispatch for all per-frame work before the raster: the background cache (only after a view change), the
// glow buffer cleared, then one point per thread.
[shader("compute")]
[WorkgroupCount(SOLVE_THREADS/64, 1, 1)] [numthreads(64, 1, 1)]
void solve(uint3 id : SV_DispatchThreadID) {
    int pixelCount = int(SCREEN_SIZE.x*SCREEN_SIZE.y);
    for (int pixel = int(id.x); pixel < pixelCount && P.viewDirty != 0; pixel += SOLVE_THREADS)
        cacheBackground(int2(pixel%int(SCREEN_SIZE.x), pixel/int(SCREEN_SIZE.x)));
    for (int pixel = int(id.x); pixel < GLOW_PIXELS; pixel += SOLVE_THREADS) glowBuffer[pixel].store(0);
    if (int(id.x) < POINT_COUNT) solvePoint(int(id.x));
}
// ============================================================================ 7. atomic raster
// Pass 0: atomicMax of the complemented depth (k/STEP_COUNT as float bits, so 0 = empty and the nearest wins);
// pass 1: fragments whose depth won write their colour into COLOUR_LAYER. main_image clears the depth buffer
// after reading it, ready for the next frame. Objects: each triangle image is followed from an anchor corner by
// continuation, split into an n x n chart-space grid (vertices Newton-refined), and the sub-triangles are scan-
// converted by whole workgroups.
float2 toPixel(float2 xy) { return float2(xy.x*P.projection.x + 1, 1 - xy.y*P.projection.y)*0.5*float2(SCREEN_SIZE); }
void fragment(int2 pixel, float depth, float3 colour) {
    if (any(pixel < 0) || any(pixel >= SCREEN_SIZE)) return;
    uint depthBits = ~asuint(clamp(depth, 1e-6, 1.)); uint index = pixel.y*SCREEN_SIZE.x + pixel.x;
    if (index >= MAX_PIXELS) return;
    if (dispatch.id == 0) depthBuffer[index].max(depthBits);
    else if (depthBuffer[index].load() == depthBits) pass_out[int3(pixel, COLOUR_LAYER)] = float4(colour, 1);
}
float3 dopplerTint(float g) {
    float t = clamp((g - 1)*2.2, -1., 1.)*P.beaming;
    float3 tint = t > 0 ? lerp(1., float3(0.42, 0.62, 1.45), t) : lerp(1., float3(1.5, 0.42, 0.18), -t);
    return tint*lerp(1., clamp(g*g*g, 0.15, 5.), 0.7*P.beaming);
}
float3 blackbody(float kelvin) {
    float t = clamp(kelvin, 1000., 40000.)/100;
    float red = t <= 66 ? 1 : 1.29294*pow(t - 60, -0.1332);
    float green = t <= 66 ? 0.39008*log(t) - 0.63184 : 1.12989*pow(t - 60, -0.0755);
    float blue = t >= 66 ? 1 : t <= 19 ? 0 : 0.54321*log(t - 10) - 1.19625;
    return saturate(float3(red, green, blue));
}
// Disk light is additive: packed RGB (11/11/10 bits, 1/64 units) added with a saturating compare-exchange loop,
// into a buffer at 1/glowScale resolution. Only in pass 1, and only in front of the meshes (their depth is done).
static const float3 GLOW_UNITS = float3(64, 64, 64);
void addGlow(int2 glowPixel, float depth, float3 colour) {
    int2 glowSize = int2(SCREEN_SIZE)/P.glowScale;
    if (any(glowPixel < 0) || any(glowPixel >= glowSize)) return;
    int2 pixel = glowPixel*P.glowScale + P.glowScale/2;
    uint meshDepth = depthBuffer[pixel.y*SCREEN_SIZE.x + pixel.x].load();
    if (meshDepth != 0 && ~asuint(clamp(depth, 1e-6, 1.)) < meshDepth) return;   // behind a mesh
    uint3 add = uint3(min(colour*GLOW_UNITS + 0.5, float3(2047, 2047, 1023)));
    uint index = glowPixel.y*glowSize.x + glowPixel.x;
    uint old = glowBuffer[index].load();
    for (int attempt = 0; attempt < 16; attempt++) {
        uint3 sum = min(uint3(old & 2047, (old >> 11) & 2047, old >> 22) + add, uint3(2047, 2047, 1023));
        uint previous = glowBuffer[index].compareExchange(old, sum.x | (sum.y << 11) | (sum.z << 22));
        if (previous == old) return;
        old = previous;
    }
}
float3 glowAt(float2 glowPixel) {   // bilinear
    int2 glowSize = int2(SCREEN_SIZE)/P.glowScale;
    float2 p = glowPixel - 0.5;
    int2 p0 = int2(floor(p));
    float2 f = p - p0;
    float3 sum = 0;
    for (int n = 0; n < 4; n++) {
        int2 q = clamp(p0 + int2(n & 1, n >> 1), 0, glowSize - 1);
        uint v = glowBuffer[q.y*glowSize.x + q.x].load();
        sum += float3(v & 2047, (v >> 11) & 2047, v >> 22)/GLOW_UNITS*((n & 1) == 1 ? f.x : 1 - f.x)*((n >> 1) == 1 ? f.y : 1 - f.y);
    }
    return sum;
}
void rasterParticle(int i) {   // all images of particle i as soft additive dots
    float heat = pow(P.innerEdge/particleOrbit(i).radius, 0.75);
    float radius = P.pointRadius/P.glowScale;
    for (int s = 0; s < PARTICLE_IMAGES; s++) {
        float4 image = loadParticleImage(i, s);
        if (image.w < 0) break;
        int order = int(image.w/256);
        if (!orderOn(order)) continue;
        float g = image.z;
        float3 colour = blackbody(3500 + 9000*heat*g)*(0.35 + 1.4*heat)*DISK_BRIGHTNESS;
        colour *= dopplerTint(g)*orderGain(order)*(0.8 + 0.4*hashUint(uint(7*i)));
        float2 centre = toPixel(image.xy)/P.glowScale; int R = min(int(ceil(radius + 0.5)), 8);
        for (int dy = -R; dy <= R; dy++) for (int dx = -R; dx <= R; dx++) {
            float2 pixel = floor(centre) + float2(dx, dy) + 0.5;
            float weight = sqr(saturate(1 - length(pixel - centre)/(radius + 0.5)));
            if (weight > 0) addGlow(int2(pixel), (image.w - order*256)/STEP_COUNT, colour*weight);
        }
    }
}
// weighted blend of chart points (Cartesian near the centre, via sigma across the curve); empty ones skipped
float4 blendChartPoints(float4 q[8], float w[8]) {
    int first = -1;
    float total = 0;
    bool allCentre = true, sameSide = true;
    for (int n = 0; n < 8; n++) {
        if (q[n].w == 0 || w[n] <= 0) continue;
        if (first < 0) first = n;
        total += w[n];
        allCentre = allCentre && nearCentre(q[n]);
        sameSide = sameSide && q[n].w == q[first].w;
    }
    if (first < 0) return 0;
    float2 centre = 0;
    float psi = 0, row = 0, sigma = 0, k = 0;
    for (int n = 0; n < 8; n++) {
        if (q[n].w == 0 || w[n] <= 0) continue;
        float v = w[n]/total;
        psi += v*psiDifference(q[first].x, q[n].x); row += v*q[n].y; k += v*q[n].z;
        if (allCentre) centre += v*toCentreXY(q[n]);          // only when needed: these cost exp, sin, cos
        else if (!sameSide) sigma += v*sigmaOf(q[n]);
    }
    if (allCentre) return float4(fromCentreXY(centre), k, -1);
    psi = wrapPsi(q[first].x + psi);
    if (sameSide) return float4(psi, row, k, q[first].w);
    float2 rowSide = rowFromSigma(sigma, psi);
    return float4(psi, rowSide.x, k, rowSide.y);
}
// continue one image along a world segment, with adaptive steps
// continue one image along a mesh edge from vertex idFrom to idTo (points on it at the same time), adaptive steps
bool follow(inout float4 chartPoint, int idFrom, int idTo) {
    float t = 0, h = 0.5;
    for (int it = 0; it < 28 && h >= 1/128.; it++) {
        float tNext = min(t + h, 1.);
        Emitter e = meshEmitter(int3(idFrom, idTo, idTo), float3(1 - tNext, tNext, 0));
        float4 now = targetOf(emitterBody(e, P.sceneTime).position);
        float tolerance = (tNext >= 1 ? 7e-4 : 4e-3)*cameraDistance(now.xyz, now.w);
        NewtonResult result = newton(chartPoint, e, tolerance, 10, 0.25);
        if (!result.ok || abs(result.chartPoint.z - chartPoint.z) >= 12) { h *= 0.5; continue; }   // retry shorter
        chartPoint = result.chartPoint; t = tNext;
        if (t >= 1) return true;
        h = min(2*h, 0.5);
    }
    return false;
}
float3 objectAlbedo(int object) { return lerp(0.85, float3(hash1(object*7.), hash1(object*11.), hash1(object*13.)), 0.55) + 0.15; }
// one thread per (object, triangle, order)
[shader("compute")]
[WorkgroupCount(OBJECT_COUNT*OBJECT_TRIS*3/64, 1, 1)] [numthreads(64, 1, 1)]
void triGen(uint3 id : SV_DispatchThreadID) {
    int item = int(id.x);
    int order = item%3, tri = item/3%OBJECT_TRIS, object = item/3/OBJECT_TRIS;
    if (!toggleOn(1) || !orderOn(order)) return;
    int quad = (tri/2/SPHERE_SEGS)*(SPHERE_SEGS + 1) + (tri/2)%SPHERE_SEGS;
    int3 vertexIds = tri%2 == 0 ? int3(quad, quad + SPHERE_SEGS + 1, quad + 1) : int3(quad + 1, quad + SPHERE_SEGS + 1, quad + SPHERE_SEGS + 2);
    int3 pointIds = PARTICLE_COUNT + object*OBJECT_VERTS + vertexIds;
    // anchor: the first corner with an image of this order
    int anchor = -1;
    for (int c = 0; c < 3 && anchor < 0; c++) for (int s = 0; s < MAX_IMAGES && anchor < 0; s++) {
        if (loadChartPoint(pointIds[c], s).w != 0 && orderOf(loadScreenData(pointIds[c], s).w) == order) anchor = c;
    }
    if (anchor < 0) return;
    int imageCount = 0;
    for (int s0 = 0; s0 < MAX_IMAGES && imageCount < 3; s0++) {
        float4 anchorPoint = loadChartPoint(pointIds[anchor], s0);
        if (anchorPoint.w == 0 || orderOf(loadScreenData(pointIds[anchor], s0).w) != order) continue;
        imageCount++;
        // the same image at the other two corners
        float4 q[3] = { anchorPoint, anchorPoint, anchorPoint };
        bool ok = true;
        for (int c = 0; c < 3 && ok; c++)
            if (c != anchor) ok = follow(q[c], pointIds[anchor], pointIds[c]);
        if (!ok) continue;
        float2 x0 = toPixel(chartToScreen(q[0])), x1 = toPixel(chartToScreen(q[1])), x2 = toPixel(chartToScreen(q[2]));
        // size in 1080p pixels: the subdivision does not depend on the resolution
        float extent = max(length(x1 - x0), max(length(x2 - x1), length(x0 - x2)))/P.pixelScale;
        if (extent > 600) continue;
        int n = clamp(int(ceil(extent/16)), 1, 6);
        if (counters[0].load() + n*n > SUBTRI_CAPACITY*3/4) n = min(n, 2);   // running out: coarser
        uint base = counters[0].add(uint(n*n));
        if (base + n*n > SUBTRI_CAPACITY) return;
        // grid vertices (i, j), i + j <= n, row by row; inner ones are Newton-refined. Each shows the surface where
        // it was when its light left (normal and velocity too)
        float4 grid[28]; float3 normals[28];
        int idx = 0;
        for (int j = 0; j <= n; j++) for (int i = 0; i <= n - j; i++) {
            float a = i/float(n), b = j/float(n);
            Emitter e = meshEmitter(pointIds, float3(1 - a - b, a, b));
            float4 corners8[8] = { q[0], q[1], q[2], 0, 0, 0, 0, 0 };
            float weights[8] = { 1 - a - b, a, b, 0, 0, 0, 0, 0 };
            float4 qq = i + j == 0 ? q[0] : i == n ? q[1] : j == n ? q[2] : blendChartPoints(corners8, weights);
            if (i + j > 0 && i < n && j < n) {
                float4 now = targetOf(emitterBody(e, P.sceneTime).position);
                NewtonResult result = newton(qq, e, 7e-4*cameraDistance(now.xyz, now.w), 8, 0.25);
                if (result.ok && abs(result.chartPoint.z - qq.z) < 6) qq = result.chartPoint;
            }
            float2 xy = chartToScreen(qq);
            TableSample s = evaluateTable(qq);
            Body then = emitterBody(e, P.sceneTime - lightDelay(s.delay, blRadius(s.position)));
            float g = redshift(xy, toTableFrame(then.position), toTableFrame(then.velocity), s.dK);
            grid[idx] = float4(toPixel(xy), qq.z/STEP_COUNT, g);
            normals[idx] = normalize(then.normal + 1e-6);
            idx++;
        }
        // n*n sub-triangles: (object, order) ride along in the normals' w
        int outIndex = int(base);
        for (int j = 0; j < n; j++) for (int i = 0; i < n - j; i++) for (int up = 0; up < 2; up++) {
            if (up == 1 && i + j >= n - 1) continue;
            int row0 = j*(n + 1) - j*(j - 1)/2; int row1 = (j + 1)*(n + 1) - (j + 1)*j/2;
            int3 v = up == 0 ? int3(row0 + i, row0 + i + 1, row1 + i) : int3(row0 + i + 1, row1 + i + 1, row1 + i);
            subTriangles[outIndex*6] = grid[v.x]; subTriangles[outIndex*6 + 1] = grid[v.y];
            subTriangles[outIndex*6 + 2] = grid[v.z];
            subTriangles[outIndex*6 + 3] = float4(normals[v.x], object);
            subTriangles[outIndex*6 + 4] = float4(normals[v.y], order);
            subTriangles[outIndex*6 + 5] = float4(normals[v.z], 0); outIndex++;
        }
    }
}
// Both raster passes: in pass 1, thread i < PARTICLE_COUNT adds particle i's light; then each workgroup takes
// every 1024th sub-triangle, its 64 threads sharing the triangle's bounding box.
[shader("compute")]
[DispatchCount(2)] [WorkgroupCount(1024, 1, 1)] [numthreads(64, 1, 1)]
void rasterScene(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID, uint localIndex : SV_GroupIndex) {
    if (id.x < PARTICLE_COUNT && toggleOn(0) && dispatch.id == 1) rasterParticle(int(id.x));
    uint triCount = min(counters[0].load(), uint(SUBTRI_CAPACITY));
    for (uint t = group.x; t < triCount; t += 1024) {
        float4 pa = subTriangles[t*6], pb = subTriangles[t*6 + 1], pc = subTriangles[t*6 + 2];
        float area = (pb.x - pa.x)*(pc.y - pa.y) - (pb.y - pa.y)*(pc.x - pa.x);
        int2 lo = int2(max(floor(min(pa.xy, min(pb.xy, pc.xy))), 0.));
        int2 hi = int2(min(ceil(max(pa.xy, max(pb.xy, pc.xy))), float2(SCREEN_SIZE)));
        int2 size = hi - lo;
        if (abs(area) < 1e-6 || any(size <= 0) || any(size > int(192*P.pixelScale))) continue;
        float4 na = subTriangles[t*6 + 3], nb = subTriangles[t*6 + 4], nc = subTriangles[t*6 + 5];
        int object = int(na.w), order = int(nb.w); float3 objectCentre = objectVertex(object, 0, P.sceneTime).centre;
        for (int pixelIndex = int(localIndex); pixelIndex < size.x*size.y; pixelIndex += 64) {
            float2 c = lo + int2(pixelIndex%size.x, pixelIndex/size.x) + 0.5;
            float3 w = barycentric(pa.xy, pb.xy, pc.xy, c);
            if (min(w.x, min(w.y, w.z)) < -1e-4) continue;
            float g = w.x*pa.w + w.y*pb.w + w.z*pc.w;
            float3 normal = normalize(w.x*na.xyz + w.y*nb.xyz + w.z*nc.xyz);
            float keyLight = max(dot(normal, normalize(float3(0.35, 0.9, 0.2))), 0.);
            float holeGlow = max(dot(normal, normalize(-objectCentre)), 0.)*saturate(12/length(objectCentre));
            float3 colour = objectAlbedo(object)*(0.3 + 0.7*keyLight) + float3(1, 0.55, 0.25)*0.35*holeGlow;
            fragment(int2(c), w.x*pa.z + w.y*pb.z + w.z*pc.z, colour*dopplerTint(g)*orderGain(order));
        }
    }
}
// ============================================================================ 8. grids, UI, composite
float4 backgroundAt(int2 pixel) { return pass_in[int3(clamp(pixel, 0, SCREEN_SIZE - 1), BACKGROUND_LAYER)]; }
// cached (theta, table phi) -> world (theta, phi)
float2 worldAngles(float4 cached) { return float2(cached.x, cached.y + P.cameraYaw); }
// 1 px line; fades to the mean line density where the lines crowd
float lineCoverage(float lineDist, float gradient) {
    if (gradient <= 1e-7 || gradient > 0.5) return 0;
    float sharp = saturate(1 - 1.3*lineDist/gradient); float mean = min(1., 1.5*gradient/(PI/12));
    return lerp(sharp, mean, smoothstep(0.03, 0.1, gradient));
}
// 15 degree graticule on the sky or the horizon, anti-aliased from neighbour differences
float gridCoverage(int2 pixel, int kind) {
    float4 cached = backgroundAt(pixel);
    if (decodeKind(cached.w) != kind) return 0;
    float2 angles = worldAngles(cached); float2 gradient[2];
    for (int axis = 0; axis < 2; axis++) {
        int2 e = axis == 0 ? int2(1, 0) : int2(0, 1);
        float4 next = backgroundAt(pixel + e), prev = backgroundAt(pixel - e);
        float2 d = 0;
        if (decodeKind(next.w) == kind && all(pixel + e < SCREEN_SIZE)) d = worldAngles(next) - angles;
        else if (decodeKind(prev.w) == kind) d = angles - worldAngles(prev);
        gradient[axis] = float2(d.x, d.y - 2*PI*round(d.y/(2*PI)));
    }
    float2 steps = float2(0.5*PI - angles.x, angles.y)/(PI/12);
    float2 lineDistance = abs(steps - round(steps))*(PI/12);
    float latitude = abs(round(steps.x)) < 5.5 ? lineCoverage(lineDistance.x, length(float2(gradient[0].x, gradient[1].x))) : 0;
    float longitude = sin(angles.x) > 0.02 ? lineCoverage(lineDistance.y, length(float2(gradient[0].y, gradient[1].y))) : 0;
    return max(latitude, longitude);
}
// cube-map cells, three layers of point stars, and a faint band
float3 stars(float3 d) {
    float3 ad = abs(d);
    uint face = ad.x >= max(ad.y, ad.z) ? (d.x > 0 ? 0 : 1) : ad.y >= ad.z ? (d.y > 0 ? 2 : 3) : (d.z > 0 ? 4 : 5);
    float2 uv = face < 2 ? d.yz/ad.x : face < 4 ? d.xz/ad.y : d.xy/ad.z;
    float3 colour = 0;
    for (uint layer = 0; layer < 3; layer++) {
        float scale = 40*(layer + 1); float2 u = (uv*0.5 + 0.5)*scale;
        uint h = uint(u.x) + 1013*uint(u.y) + 1048573*face + 7919*layer;
        float2 offset = float2(hashUint(h), hashUint(h + 1)) - 0.5;
        float brightness = hashUint(h + 2);
        if (brightness <= 0.55) continue;
        float3 tint = lerp(float3(0.6, 0.7, 1), float3(1, 0.8, 0.6), hashUint(h + 3));
        float spot = 1 - smoothstep(0., 0.1, length(frac(u) - 0.5 - 0.7*offset)*80/scale);
        colour += tint*pow(brightness, 8.)*3*spot;
    }
    return colour + float3(0.035, 0.03, 0.045)*exp(-8*sqr(0.9*d.y + 0.3*d.x));
}
// text: 3x5 font drawn at 2x (8 x 10 cells), strings packed 4 characters per uint
static const uint FONT[42] = { 0, 11415, 25255, 29391, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11245, 27566, 14627, 27502,   // '0'..'Y'
    31143, 31140, 14699, 23533, 29847, 4714, 23469, 18727, 0, 27501, 11114, 27556, 0, 27565, 14478, 29842, 23407, 0, 0, 23213, 23186 };
static const uint2 LABELS[9] = { uint2(1313427280u, 538989396u), uint2(1162494543u, 542331971u), uint2(1380013139u, 538976339u),   // POINTS OBJECTS STARS
    uint2(542722899u, 1145655879u), uint2(1193297989u, 541346130u), uint2(1163020612u, 538989635u),    // SKY GRID, EH GRID, DIRECT
    uint2(1347374924u, 538980640u), uint2(1347374924u, 538980896u), uint2(1095517508u, 538976345u) };   // LOOP 1, LOOP 2, DELAY
static const uint WARNING[6] = { 1111576133u, 1176520012u, 1413566284u, 1411396147u, 1431590981u, 542328146u };   // ENABLE FLOAT32 TEXTURES
static const float3 SWATCH[9] = { float3(0.8, 0.86, 1), float3(0.85, 0.6, 0.45), float3(1), float3(0.35, 0.75, 0.72),
    float3(1, 0.62, 0.22), float3(0.62, 0.76, 0.91), float3(0.37, 0.85, 0.77), float3(0.88, 0.48, 0.84), float3(0.95, 0.9, 0.5) };
uint charOf(uint packed, int x) { return (packed >> (8*((x/8)%4))) & 255; }
// is position p inside an 8 x 10 character cell lit?
bool glyphLit(uint character, int2 p) {
    int gx = (p.x%8)/2, gy = p.y/2;
    return gx < 3 && gy < 5 && character >= 48 && character < 90 && ((FONT[character - 48] >> (14 - gy*3 - gx)) & 1) == 1;
}
// button under a pixel, or -1 (the UI is drawn at an integer zoom)
int uiButton(float2 pixel, float hidden, int uiScale) {
    int2 q = int2(pixel)/uiScale - int2(8, 26); int button = q.y/19;
    if (hidden > 0.5 || any(q < 0) || q.x >= 100 || button >= 9 || q.y%19 >= 16) return -1;
    return button;
}
float3 uiPixel(int2 pixel, float3 colour) {
    int button = uiButton(pixel + 0.5, P.uiHidden, P.uiScale);
    if (button < 0) return colour;
    bool on = toggleOn(button); int2 q = pixel/P.uiScale - int2(8, 26 + 19*button);
    int2 textPos = q - int2(14, 3);
    float3 c = on ? float3(0.16, 0.18, 0.22) : float3(0.07, 0.08, 0.1);
    if (any(q == 0) || q.x == 99 || q.y == 15) c = float3(on ? 0.45 : 0.2);         // border
    if (all(q >= 4) && all(q < int2(10, 11))) c = SWATCH[button]*(on ? 1 : 0.25);   // colour swatch
    if (all(textPos >= 0) && textPos.x < 64 && textPos.y < 10 && glyphLit(charOf(LABELS[button][textPos.x/32], textPos.x), textPos))
        c = float3(on ? 0.95 : 0.45);
    return lerp(colour, c, 0.92);
}
// banner shown when the pass textures are only 16 bit (the per-point images need float32)
float3 warningPixel(int2 pixel, float3 colour) {
    int2 q = pixel/P.uiScale - int2(8, 205);
    if (any(q < 0) || q.x >= 200 || q.y >= 16) return colour;
    int2 textPos = q - int2(4, 3);
    bool lit = all(textPos >= 0) && textPos.x < 192 && textPos.y < 10 && glyphLit(charOf(WARNING[textPos.x/32], textPos.x), textPos);
    return lit ? float3(1) : float3(0.6, 0.08, 0.05);
}
[shader("compute")]
[numthreads(16, 16, 1)]
void main_image(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= SCREEN_SIZE)) return;
    int2 pixel = int2(id.xy); float4 background = backgroundAt(pixel);
    int kind = decodeKind(background.w);
    // background image order (> 2 loops draw with the 2nd loop toggle, dimmed)
    int order = orderOf(background.z); bool orderVisible = orderOn(order < 0 ? 2 : order);
    float gain = order >= 0 ? orderGain(order) : 0.55; float3 colour = 0;
    if (kind == KIND_SKY && orderVisible) {
        float2 angles = worldAngles(background);
        if (toggleOn(2)) colour = stars(float3(sin(angles.x)*cos(angles.y), cos(angles.x), -sin(angles.x)*sin(angles.y)));
        if (toggleOn(3)) colour = lerp(colour, float3(0.35, 0.75, 0.72)*0.55*gain, gridCoverage(pixel, KIND_SKY));
    }
    if (kind == KIND_HORIZON && orderVisible && toggleOn(4))
        colour = lerp(colour, float3(1, 0.62, 0.22)*gain, gridCoverage(pixel, KIND_HORIZON));
    // rasterised meshes; the depth buffer is cleared here for the next frame
    uint index = id.y*SCREEN_SIZE.x + id.x;
    if (index < MAX_PIXELS) {
        if (depthBuffer[index].load() != 0) colour = pass_in[int3(pixel, COLOUR_LAYER)].rgb;
        depthBuffer[index].store(0);
    }
    colour += glowAt((pixel + 0.5)/P.glowScale);   // the disk, added on top
    colour = 1 - exp(colour*(-P.exposure));   // (not -colour: Slang emitted 0 - a + b for -(a + b) here)
    int2 uiPos = pixel/P.uiScale;
    if (P.uiHidden == 0) {
        // distance bars (log scale 2..80 M): asked for by the slider, in use
        float f = (uiPos.x - 8)/200.0;
        float sliderDistance = custom.dist > 1 ? custom.dist : 4 + 36*sqr(saturate(custom.dist));
        bool inBar = uiPos.x >= 8 && uiPos.x < 208;
        if (inBar && uiPos.y >= 8 && uiPos.y < 13) colour = f < log(sliderDistance/2)/log(40.) ? float3(0.3, 0.6, 1) : 0.12;
        if (inBar && uiPos.y >= 15 && uiPos.y < 20) colour = f < log(P.cameraRadius/2)/log(40.) ? float3(1, 0.6, 0.2) : 0.12;
        colour = uiPixel(pixel, colour);
    }
    // the kind code carries a 2^-12 fraction that a 16 bit texture rounds away
    if (abs(background.w - round(background.w)) < 1e-5) colour = warningPixel(pixel, colour);
    screen[id.xy] = float4(colour, 1);
}
