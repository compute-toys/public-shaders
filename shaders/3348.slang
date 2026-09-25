// Kerr projector (compute.toys, Slang) - every image of points and meshes around a spinning black hole. Rays
// are traced once per camera pose into a table; points are then found in the table by Newton iteration (no per-
// point tracing), and everything is drawn with atomics (no hardware rasteriser).
//   sliders (0..1): dist (>1 = distance in M), spin, fov, speed, psize, exposure
//   drag = orbit, wheel / W / S = distance, click the buttons (top left) to toggle parts, H hides them
// Needs "Float32 Textures" on (settings): the per-point images live in the pass textures. Up to 3840 x 2160.
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
// seed raster grid: per shell, map, layer and swept-angle band (width pi) a small (phi, theta) map
static const int SHELL_COUNT = 16, PEEL_LAYERS = 3, BAND_COUNT = 5;
static const int GRID_W = 112, GRID_H = 56;
static const int GRID_SIZE = 2*SHELL_COUNT*(PEEL_LAYERS + 1)*BAND_COUNT*GRID_H*GRID_W;
static const int MAX_IMAGES = 8;                      // per point
// scene: particles + small spheres (lat-long meshes)
static const int PARTICLE_COUNT = 16384;
static const int OBJECT_COUNT = 6;
static const int SPHERE_SEGS = 16, SPHERE_RINGS = 10;
static const int OBJECT_VERTS = (SPHERE_SEGS + 1)*(SPHERE_RINGS + 1);
static const int OBJECT_TRIS = 2*SPHERE_SEGS*SPHERE_RINGS;
static const int POINT_COUNT = PARTICLE_COUNT + OBJECT_COUNT*OBJECT_VERTS;
static const int SUBTRI_CAPACITY = 49152;
static const int MAX_PIXELS = 3840*2160;              // depth buffer: screens up to 4K
static const float PI = 3.14159265;
static const int HORIZON_RECORDS = RAY_COUNT;         // offsets into rayRecords
static const int SKY_RECORDS = 2*RAY_COUNT;
// pass texture layers: background cache, image chart points, image screen data, raster colour
static const int BACKGROUND_LAYER = 0, CHART_LAYER = 1, SCREEN_LAYER = 2, COLOUR_LAYER = 3;
// ============================================================================ storage
[StorageBuffer(STEP_COUNT*RAY_COUNT)]        RWStructuredBuffer<float4> rayTable;        // [row][psi][k]: outgoing position, swept angle
[StorageBuffer(SHELL_COUNT*RAY_COUNT)]       RWStructuredBuffer<float4> shellCrossings;  // [shell][ray]: (k, T) of 1st, 2nd crossing
[StorageBuffer(3*RAY_COUNT)]                 RWStructuredBuffer<float4> rayRecords;      // periastron | horizon landing | sky direction
[StorageBuffer(GRID_SIZE)]                   RWStructuredBuffer<Atomic<uint>> rasterGrid;      // seeds, see section 2
[StorageBuffer(SUBTRI_CAPACITY*6)]           RWStructuredBuffer<float4> subTriangles;    // 3 x (pixel, depth, g), 3 x (normal, extra)
[StorageBuffer(MAX_PIXELS)]                  RWStructuredBuffer<Atomic<uint>> depthBuffer;     // ~depth bits, 0 = empty
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
    float pixelScale; int uiScale;                          // screen height / 1080 (at least 1), UI zoom
    uint tableDirty, viewDirty, uiHidden;                   // rebuild the table / the background cache
    uint toggles;          // bits: 0 particles, 1 objects, 2 stars, 3 sky grid, 4 horizon grid, 5-7 image orders
    float orbitRadiusMin, orbitRadiusMax;                   // photon orbits seen on the critical curve
    float shellRadius[SHELL_COUNT];
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
// world point -> table frame outgoing position (w = r); points inside the horizon are pushed out
float4 targetOf(float3 worldPos) {
    float3 X = toTableFrame(worldPos); float r = blRadius(X);
    float minR = P.horizonRadius + 0.08;
    if (r < minR) { X *= minR/r; r = blRadius(X); }
    float phiOut = atan2(-X.z, X.x) - outgoingShift(r);
    return float4(toCartesian(r, cos(polarAngle(X, r)), phiOut), r);
}
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
// RK4 step of stepArc log-polar arclength (|dx|/|x|): fine near the hole, long far out. Also returns the angle
// swept around the hole during the step.
Ray rk4Step(Ray ray, float stepArc, out float sweptAngle) {
    float rho = length(ray.x); Ray k1 = flow(ray);
    float h = stepArc/max(length(k1.x)/rho, 1e-6); Ray k2 = flow(rayAdd(ray, k1, 0.5*h));
    Ray k3 = flow(rayAdd(ray, k2, 0.5*h)); Ray k4 = flow(rayAdd(ray, k3, h));
    float3 dX = h/6*(k1.x + 2*k2.x + 2*k3.x + k4.x); float3 dP = h/6*(k1.p + 2*k2.p + 2*k3.p + k4.p);
    sweptAngle = length(dX - dot(dX, ray.x)*ray.x/(rho*rho))/rho;
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
// Kerr-Schild position -> stored (mirrored, outgoing) table position
float4 tableEntry(float3 X, float sweptAngle) {
    float r = ksRadius(X);
    float phi = atan2(X.y, X.x) - atan2(spin(), r) + outgoingShift(P.cameraRadius);
    return float4(toCartesian(r, clamp(X.z/r, -1., 1.), -phi), max(sweptAngle, 1e-7));
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
        distanceState = 0; uiState = float4(255, 0, 0, 0);
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
    P.tableDirty = rebuild ? 1 : 0; P.viewDirty = viewChanged ? 1 : 0;
    P.uiHidden = uiState.z > 0.5 ? 1 : 0; P.toggles = toggles;
    // shells: horizon, 14 log-spaced up to 64 M, far sky
    for (int j = 0; j < SHELL_COUNT; j++) {
        float inner = rPlus + 0.35;
        float radius = inner*pow(64/inner, (j - 1)/float(SHELL_COUNT - 3));
        P.shellRadius[j] = j == 0 ? rPlus + 0.02 : j == SHELL_COUNT - 1 ? 350 : radius;
    }
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
    float sweptAngle = 0; float rShell = P.shellRadius[0];
    float a = spin(); bool frozen = false, hitHorizon = false;
    rayTable[ray*STEP_COUNT] = float4(tableEntry(state.x, 0).xyz, 0);
    rayRecords[HORIZON_RECORDS + ray] = float4(0, 0, 0, -1);
    rayRecords[SKY_RECORDS + ray] = float4(0, 0, 0, -1);
    for (int i = ray; i < GRID_SIZE; i += RAY_COUNT) rasterGrid[i].store(0);   // clear the seed grid (section 2)
    for (int k = 1; k < STEP_COUNT; k++) {
        // sub-steps: shorter near the ring singularity and when approaching the horizon shell
        float arcLeft = STEP_ARC;
        for (int n = 0; n < 48 && !frozen && arcLeft > 1e-6; n++) {
            float r0 = ksRadius(state.x);
            float ringDistance = length(float2(length(state.x.xy) - a, state.x.z));
            float stepArc = min(arcLeft, min(STEP_ARC/6, 0.2*ringDistance/length(state.x) + 1e-4));
            if (!hitHorizon) stepArc = min(stepArc, max(0.5*(r0 - rShell)/r0, 0.004));
            float stepAngle; Ray next = rk4Step(state, stepArc, stepAngle);
            float r1 = ksRadius(next.x);
            if (!hitHorizon && r1 <= rShell) {
                float f = (r0 - rShell)/max(r0 - r1, 1e-9);
                float3 landing = tableEntry(lerp(state.x, next.x, f), sweptAngle).xyz;
                rayRecords[HORIZON_RECORDS + ray] = float4(horizonDirection(landing, rShell), sweptAngle + f*stepAngle);
                hitHorizon = true;
            }
            sweptAngle += stepAngle; arcLeft -= stepArc;
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
        rayTable[ray*STEP_COUNT + k] = tableEntry(state.x, sweptAngle);
    }
    findCrossings(ray);
}
// samples of one ray are adjacent: the trilinear lookups in section 4 touch 4 short runs instead of 8 planes
float4 sampleAt(int ray, int k) { return rayTable[ray*STEP_COUNT + k]; }
float4 tableAt(int psiIndex, int row, int k) { return sampleAt(row*PSI_COUNT + (psiIndex%PSI_COUNT + PSI_COUNT)%PSI_COUNT, k); }
// direction from the hole of a ray at fractional sample k
float3 directionAt(int psiIndex, int row, float k) {
    int k0 = min(int(k), STEP_COUNT - 2);
    float3 X = lerp(tableAt(psiIndex, row, k0).xyz, tableAt(psiIndex, row, k0 + 1).xyz, k - k0);
    return X/blRadius(X);
}
// crossings of the 16 reference shells (the seed sources), and the ray's closest approach
void findCrossings(int ray) {
    for (int shell = 0; shell < SHELL_COUNT; shell++) {
        float4 result = float4(-1, 0, -1, 0); float4 prev = sampleAt(ray, 0);
        float prevDist = blRadius(prev.xyz) - P.shellRadius[shell];
        int found = 0;
        for (int k = 1; k < STEP_COUNT && found < 2; k++) {
            float4 cur = sampleAt(ray, k);
            float curDist = blRadius(cur.xyz) - P.shellRadius[shell];
            if (prevDist != 0 && prevDist*curDist <= 0 && curDist != prevDist) {
                float f = prevDist/(prevDist - curDist);
                float2 crossing = float2(k - 1 + f, lerp(prev.w, cur.w, f));
                if (found == 0) result.xy = crossing; else result.zw = crossing;
                found++;
            }
            prev = cur; prevDist = curDist;
        }
        shellCrossings[shell*RAY_COUNT + ray] = result;
    }
    // periastron: the minimum sample, refined by a parabola
    float best = 1e9; int kBest = -1;
    for (int k = 1; k < STEP_COUNT - 1; k++) {
        float r = blRadius(sampleAt(ray, k).xyz);
        if (r >= best) continue;
        best = r; kBest = k;
    }
    bool valid = kBest >= 2 && kBest < STEP_COUNT - 2 && best >= P.horizonRadius + 0.05;
    rayRecords[ray] = -1;
    if (!valid) return;
    float rBefore = blRadius(sampleAt(ray, kBest - 1).xyz);
    float rAfter = blRadius(sampleAt(ray, kBest + 1).xyz);
    float curvature = rBefore - 2*best + rAfter;
    float kMin = kBest + (curvature > 1e-9 ? clamp(0.5*(rBefore - rAfter)/curvature, -0.5, 0.5) : 0.);
    int k0 = int(kMin);
    float4 A = sampleAt(ray, k0), B = sampleAt(ray, k0 + 1);
    rayRecords[ray] = float4(kMin, blRadius(lerp(A.xyz, B.xyz, kMin - k0)), lerp(A.w, B.w, kMin - k0), 1);
}
// ============================================================================ 2. seed raster grid
// Per shell, the crossings are rasterised into (phi, theta) maps, one per swept-angle band (image order). A
// texel keeps the earliest ray: atomicMax of the complement of packed (k, row, psi). Layers 1, 2 are depth-
// peeled crossings behind the previous layer, layer 3 is the periastron. Map 1 is rotated (pole on x) and
// serves targets within 45 degrees of the spin axis.
int rasterGridIndex(int map, int shell, int layer, int band, int ty, int tx) {
    return ((((map*SHELL_COUNT + shell)*(PEEL_LAYERS + 1) + layer)*BAND_COUNT + band)*GRID_H + ty)*GRID_W + tx;
}
float2 mapAngles(float3 d, int map) {
    return map == 0 ? float2(atan2(d.z, d.x), acos(clamp(d.y, -1., 1.))) : float2(atan2(d.y, d.x), acos(clamp(d.z, -1., 1.)));
}
uint packSeed(float k, float row, float psiIndex) {
    uint kBits = uint(clamp(round(k*4), 0., 1023.)); uint rowBits = uint(clamp(round(row*4), 0., 1023.));
    uint psiBits = uint(round(wrapPsi(psiIndex)*4))%(4*PSI_COUNT);
    return (kBits << 20) | (rowBits << 10) | psiBits;
}
// barycentric weights of p in triangle (A, B, C)
float3 barycentric(float2 A, float2 B, float2 C, float2 p) {
    float area = (B.x - A.x)*(C.y - A.y) - (B.y - A.y)*(C.x - A.x);
    float wa = ((B.x - p.x)*(C.y - p.y) - (B.y - p.y)*(C.x - p.x))/area;
    float wb = ((C.x - p.x)*(A.y - p.y) - (C.y - p.y)*(A.x - p.x))/area;
    return float3(wa, wb, 1 - wa - wb);
}
// one thread per (shell, map, crossing, side, chart cell); each cell is two triangles in the map
[shader("compute")]
[DispatchCount(PEEL_LAYERS + 1)] [WorkgroupCount(PSI_COUNT*(ELL_COUNT - 1)*8*SHELL_COUNT/64, 1, 1)] [numthreads(64, 1, 1)]
void buildRasterGrid(uint3 id : SV_DispatchThreadID) {
    if (!tableDirty()) return;
    int layer = int(dispatch.id); int cell = int(id.x)%(PSI_COUNT*(ELL_COUNT - 1));
    int rest = int(id.x)/(PSI_COUNT*(ELL_COUNT - 1));
    int sideIndex = rest%2, crossing = (rest/2)%2, map = (rest/4)%2, shell = rest/8;
    int cellPsi = cell%PSI_COUNT, cellRow = cell/PSI_COUNT; bool periastron = layer == PEEL_LAYERS;
    if (periastron && crossing == 1) return;
    // periastra are binned into the shell whose radius they are closest to (log scale)
    float binLo = shell > 0 ? sqrt(P.shellRadius[shell - 1]*P.shellRadius[shell])/1.08 : 0;
    float binHi = shell < SHELL_COUNT - 1 ? sqrt(P.shellRadius[shell]*P.shellRadius[shell + 1])*1.08 : 1e9;
    float2 texel[4]; float kAt[4];
    float3 dirs[4]; int bandMin = 99, bandMax = -1;
    float phi0 = 0, phiMin = 1e9, phiMax = -1e9;
    for (int corner = 0; corner < 4; corner++) {
        int psiIndex = (cellPsi + (corner & 1))%PSI_COUNT;
        int row = sideIndex*ELL_COUNT + cellRow + (corner >> 1); int ray = row*PSI_COUNT + psiIndex;
        float4 peri = rayRecords[ray];
        float4 crossings = shellCrossings[shell*RAY_COUNT + ray];
        float k = periastron ? peri.x : crossing == 0 ? crossings.x : crossings.z;
        float T = periastron ? peri.z : crossing == 0 ? crossings.y : crossings.w;
        if (periastron && (peri.w < 0 || peri.y < binLo || peri.y > binHi)) return;
        if (k < 0) return;
        dirs[corner] = directionAt(psiIndex, row, k);
        bandMin = min(bandMin, int(T/PI)); bandMax = max(bandMax, int(T/PI));
        float2 angles = mapAngles(dirs[corner], map);
        float phi = angles.x;
        if (corner == 0) phi0 = phi = phi - 2*PI*floor(phi/(2*PI));
        else phi = phi0 + phi - phi0 - 2*PI*floor((phi - phi0 + PI)/(2*PI));   // unwrap next to corner 0
        phiMin = min(phiMin, phi); phiMax = max(phiMax, phi);
        texel[corner] = float2(phi/(2*PI)*GRID_W, angles.y/PI*GRID_H);
        kAt[corner] = k;
    }
    // cells spanning a fold or two bands would paint seeds where no ray goes
    if (bandMax - bandMin > 1 || phiMax - phiMin > 1.6) return;
    if (dot(dirs[0], dirs[1]) < 0.64 || dot(dirs[0], dirs[2]) < 0.64 || dot(dirs[0], dirs[3]) < 0.64) return;
    for (int bandSlot = 0; bandSlot < 2; bandSlot++) {
        if (bandSlot == 1 && bandMax == bandMin) break;
        int band = bandSlot == 0 ? bandMin : bandMax;
        if (band >= BAND_COUNT) continue;
        for (int tri = 0; tri < 2; tri++) {
            int3 v = tri == 0 ? int3(0, 1, 2) : int3(1, 3, 2);
            float2 A = texel[v.x], B = texel[v.y], C = texel[v.z];
            if (abs((B.x - A.x)*(C.y - A.y) - (B.y - A.y)*(C.x - A.x)) < 1e-9) continue;
            int2 t0 = int2(ceil(min(A, min(B, C)) - 0.5)); int2 t1 = int2(floor(max(A, max(B, C)) - 0.5));
            t0.y = max(t0.y, 0); t1.y = min(t1.y, GRID_H - 1);
            if (any(t1 - t0 > 48)) continue;
            for (int ty = t0.y; ty <= t1.y; ty++) for (int tx = t0.x; tx <= t1.x; tx++) {
                float3 w = barycentric(A, B, C, float2(tx, ty) + 0.5);
                if (min(w.x, min(w.y, w.z)) < -1e-5) continue;
                float k = w.x*kAt[v.x] + w.y*kAt[v.y] + w.z*kAt[v.z];
                float psiIndex = cellPsi + w.x*(v.x & 1) + w.y*(v.y & 1) + w.z*(v.z & 1);
                float row = sideIndex*ELL_COUNT + cellRow + w.x*(v.x >> 1) + w.y*(v.y >> 1) + w.z*(v.z >> 1);
                int txWrapped = (tx%GRID_W + GRID_W)%GRID_W;
                if (layer >= 1 && !periastron) {
                    // depth peel: only crossings clearly behind the previous layer's seed
                    uint previous = rasterGrid[rasterGridIndex(map, shell, layer - 1, band, ty, txWrapped)].load();
                    if (previous == 0 || k <= ((~previous) >> 20)*0.25 + 0.75) continue;
                }
                rasterGrid[rasterGridIndex(map, shell, layer, band, ty, txWrapped)].max(~packSeed(k, row, psiIndex));
            }
        }
    }
}
// seed chart point (psi, row, k, side) for direction d; side 0 = none
float4 seedAt(int shell, int layer, int band, float3 d, int2 offset) {
    int map = abs(d.y) > 0.7071 ? 1 : 0; float2 angles = mapAngles(d, map);
    int ty = clamp(int(angles.y/PI*GRID_H) + offset.y, 0, GRID_H - 1);
    int tx = (int(frac(angles.x/(2*PI))*GRID_W) + offset.x + GRID_W)%GRID_W;
    uint stored = rasterGrid[rasterGridIndex(map, shell, layer, band, ty, tx)].load();
    if (stored == 0) return 0;
    uint packed = ~stored; float row = ((packed >> 10) & 1023)*0.25;
    float side = row < ELL_COUNT - 0.5 ? 1 : -1;
    return float4((packed & 1023)*0.25, row - rowOffset(side), (packed >> 20)*0.25, side);
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
struct TableSample { float3 position, dPsi, dRow, dK; float sweptAngle; };   // dK is also the ray tangent
TableSample evaluateTable(float4 chartPoint) {
    float px = wrapPsi(chartPoint.x); float py = clamp(chartPoint.y, 0., ELL_COUNT - 1.);
    float pz = clamp(chartPoint.z, 0., STEP_COUNT - 1.); int i0 = min(int(px), PSI_COUNT - 1);
    int rowLocal = min(int(py), ELL_COUNT - 2); int k0 = min(int(pz), STEP_COUNT - 2);
    float fx = px - i0, fy = py - rowLocal, fz = pz - k0;
    // corner n: bit 0 = psi + 1, bit 1 = row + 1, bit 2 = k + 1
    float4 c[8];
    for (int n = 0; n < 8; n++)
        c[n] = tableAt(i0 + (n & 1), rowLocal + rowOffset(chartPoint.w) + ((n >> 1) & 1), k0 + (n >> 2));
    float4 c00 = lerp(c[0], c[1], fx), c10 = lerp(c[2], c[3], fx);
    float4 c01 = lerp(c[4], c[5], fx), c11 = lerp(c[6], c[7], fx);
    float4 c0 = lerp(c00, c10, fy), c1 = lerp(c01, c11, fy); float4 value = lerp(c0, c1, fz);
    float4 dPsi = lerp(lerp(c[1] - c[0], c[3] - c[2], fy), lerp(c[5] - c[4], c[7] - c[6], fy), fz);
    return { value.xyz, dPsi.xyz, lerp(c10 - c00, c11 - c01, fz).xyz, (c1 - c0).xyz, value.w };
}
float3 solve3(float3 a, float3 b, float3 c, float3 F) {   // solves [a b c] x = F
    float3 bc = cross(b, c);
    return float3(dot(bc, F), dot(cross(c, a), F), dot(cross(a, b), F))/dot(a, bc);
}
// scale for tolerances: distance from the camera
float cameraDistance(float3 X, float r) { return clamp(length(X - rayTable[0].xyz), 0.25, max(r, 2.)); }
struct NewtonResult { float4 chartPoint; bool ok; float sweptAngle; float3 rayDirection; };
// Newton in (psi, row, k). Each iteration tries three candidates and keeps the smallest residual:
//   0: the plain step
//   1: the same step as a linear move in sigma (right near the camera, crosses the critical curve)
//   2: near the shadow centre, a Cartesian step (passes through the centre)
NewtonResult newton(float4 chartPoint, float3 target, float tolerance, int maxIterations, float minCos = 0.866) {
    float4 q = chartPoint; TableSample current = evaluateTable(q);
    float3 F = current.position - target; float error2 = dot(F, F);
    bool converged = false;
    for (int it = 0; it < maxIterations; it++) {
        // stop one iteration after reaching the tolerance (or right away well below it)
        if (error2 < tolerance*tolerance) {
            if (converged || error2 < 0.0025*tolerance*tolerance) break;
            converged = true;
        }
        float3 newtonStep = solve3(current.dPsi, current.dRow, current.dK, F);
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
            float3 dRadius = -current.dRow/INNER_LIN_STEP;
            float3 dAngle = current.dPsi/(2*PI/PSI_COUNT*(1 + PSI_WARP*cos(2*t)));
            float3 dCx = cos(psi)*dRadius - sin(psi)/radius*dAngle;
            float3 dCy = sin(psi)*dRadius + cos(psi)/radius*dAngle;
            float3 d = solve3(dCx, dCy, current.dK, F); d /= max(1, max(length(d.xy)/0.25, abs(d.z)*0.25));
            float2 psiRow = fromCentreXY(radius*float2(cos(psi), sin(psi)) - d.xy);
            if (!any(isnan(d))) candidates[2] = float4(psiRow, clamp(q.z - d.z, kMin, STEP_COUNT - 1.), -1);
        }
        float bestError = 1e30;
        for (int n = 0; n < 3; n++) {
            if (candidates[n].w > 1) continue;
            TableSample s = evaluateTable(candidates[n]);
            float e = dot(s.position - target, s.position - target);
            if (e >= bestError) continue;
            bestError = e; q = candidates[n];
            current = s;
        }
        F = current.position - target; error2 = bestError;
    }
    q.x = wrapPsi(q.x);
    NewtonResult result = { q, false, current.sweptAngle, current.dK };
    if (error2 >= tolerance*tolerance) return result;
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
// prograde Keplerian circle in a tilted plane
Body orbit(float radius, float phase, float tilt, float node) {
    float angularRate = 1/(pow(radius, 1.5) + spin()); float angle = phase + P.sceneTime*angularRate;
    float3 n = float3(sin(tilt)*cos(node), cos(tilt), sin(tilt)*sin(node));
    float3 u = normalize(cross(abs(n.y) < 0.99 ? float3(0, 1, 0) : float3(1, 0, 0), n));
    float3 v = cross(n, u); float3 position = radius*(cos(angle)*u + sin(angle)*v);
    return { position, radius*angularRate*(-sin(angle)*u + cos(angle)*v), n, position };
}
Body particle(int index) {
    uint i = uint(index); float u = hashUint(4*i + 1);
    float radius = lerp(4.5, 18., u*u);
    float tilt = (hashUint(4*i + 3) - 0.5)*0.11*(1 + 0.08*radius);
    return orbit(radius, 2*PI*hashUint(4*i + 2), tilt, 2*PI*hashUint(4*i + 4));
}
float3 rotateAround(float3 v, float3 axis, float angle) {
    return v*cos(angle) + cross(axis, v)*sin(angle) + axis*dot(axis, v)*(1 - cos(angle));
}
Body objectVertex(int object, int vertex) {   // small spinning spheres on inclined orbits
    float radius = lerp(6., 11., hash1(object*3.1 + 0.7));
    float phase = 2*PI*object/OBJECT_COUNT + 0.4*hash1(object + 5.);
    Body body = orbit(radius, phase, (hash1(object*7.3) - 0.5)*1.1, 2*PI*hash1(object*11.1));
    float theta = PI*(vertex/(SPHERE_SEGS + 1))/SPHERE_RINGS;
    float phi = 2*PI*(vertex%(SPHERE_SEGS + 1))/SPHERE_SEGS;
    float3 axis = normalize(float3(hash1(object*17.), hash1(object*29.), hash1(object*41.)) - 0.5 + 1e-3);
    float3 localNormal = float3(sin(theta)*cos(phi), cos(theta), sin(theta)*sin(phi));
    body.normal = rotateAround(localNormal, axis, object*37 + P.sceneTime*0.6);
    body.position += body.normal*(0.55 + 0.35*hash1(object*13.));
    return body;
}
Body pointBody(int pointId) {
    if (pointId < PARTICLE_COUNT) return particle(pointId);
    int local = pointId - PARTICLE_COUNT;
    return objectVertex(local/OBJECT_VERTS, local%OBJECT_VERTS);
}
// ============================================================================ 6. per-point solve
// Up to 8 images per point, kept frame to frame (warm-started Newton). New images come from the raster grid
// (nearest shell, neighbour texels, adjacent shells, deeper layers) every 4th frame per point, for all points
// after a rebuild, plus a straight line-of-sight seed for points near the camera.
bool sameImage(float4 a, float4 b) {
    if (a.w == 0 || a.w != b.w) return false;
    if (nearCentre(a) && nearCentre(b)) return length(toCentreXY(a) - toCentreXY(b)) < 0.02 && abs(a.z - b.z) < 1;
    return abs(psiDifference(a.x, b.x)) < 0.6 && abs(a.y - b.y) < 0.6 && abs(a.z - b.z) < 1;
}
// g = (p.u_camera)/(p.u_emitter), exact for the ray through screen xy
float redshift(float2 screenXY, float3 X, float3 velocity, float3 rayDirection) {
    float4 constants = rayConstants(screenXY);
    float lambda = constants.x, eta = constants.y, a = spin();
    // emitter 4-velocity from a finite difference of its BL coordinates: v = (dr, dtheta, dphi)/dt
    float3 Xnext = X + velocity*1e-3; float r = blRadius(X), rNext = blRadius(Xnext);
    float theta = polarAngle(X, r), thetaNext = polarAngle(Xnext, rNext);
    float phi = atan2(-X.z, X.x); float dPhi = atan2(-Xnext.z, Xnext.x) - phi;
    dPhi -= 2*PI*floor((dPhi + PI)/(2*PI));
    float3 v = float3(rNext - r, thetaNext - theta, dPhi)/1e-3;
    float sin2 = sin(theta)*sin(theta); float Sigma = r*r + a*a*cos(theta)*cos(theta);
    float Delta = delta(r); float A = sqr(r*r + a*a) - a*a*Delta*sin2;
    // g_ab u^a u^b / (u^t)^2
    float norm = -(Delta - a*a*sin2)/Sigma - 2*a*sin2*(r*r + a*a - Delta)/Sigma*v.z;
    norm += sin2*A/Sigma*v.z*v.z + Sigma/Delta*v.x*v.x + Sigma*v.y*v.y;
    if (norm >= -1e-4) return 1;
    float radialPotential = sqr(r*r + a*a - a*lambda) - Delta*(eta + sqr(lambda - a));
    float polarPotential = eta + a*a*cos(theta)*cos(theta) - lambda*lambda/sqr(tan(theta));
    // signs of the ray's radial and polar motion from its direction
    float3 thetaDirection = float3(cos(theta)*cos(phi), -sin(theta), -cos(theta)*sin(phi));
    float radialSign = sign(dot(rayDirection, X));
    float polarSign = sign(dot(rayDirection, thetaDirection));
    float pu = -1 - radialSign*sqrt(max(radialPotential, 0.))/Delta*v.x - polarSign*sqrt(max(polarPotential, 0.))*v.y + lambda*v.z;
    return clamp((-1 + P.cameraOmega*lambda)/P.lapse/(pu*rsqrt(-norm)), 0.02, 20.);
}
// Image storage: slot s of point p is texel p*MAX_IMAGES + s of the pass textures, row-major across the screen:
// the chart point in CHART_LAYER, (x, y, g, T) in SCREEN_LAYER. Written to pass_out and read from pass_in,
// which compute.toys refreshes after every dispatch (so it persists across frames).
int2 imageTexel(int pointId, int slot) {
    int index = pointId*MAX_IMAGES + slot;
    return int2(index%int(SCREEN_SIZE.x), index/int(SCREEN_SIZE.x));
}
bool imagesFit(int pointId) { return (pointId + 1)*MAX_IMAGES <= int(SCREEN_SIZE.x*SCREEN_SIZE.y); }
float4 loadChartPoint(int pointId, int slot) { return imagesFit(pointId) ? pass_in[int3(imageTexel(pointId, slot), CHART_LAYER)] : 0; }
float4 loadScreenData(int pointId, int slot) { return pass_in[int3(imageTexel(pointId, slot), SCREEN_LAYER)]; }
void storeImage(int pointId, int slot, float4 chartPoint, float4 screenData) {
    pass_out[int3(imageTexel(pointId, slot), CHART_LAYER)] = chartPoint;
    pass_out[int3(imageTexel(pointId, slot), SCREEN_LAYER)] = screenData;
}
struct ImageList {
    float4 chartPoint[MAX_IMAGES]; float sweptAngle[MAX_IMAGES];
    float3 rayDirection[MAX_IMAGES];
    bool contains(float4 q, int count = MAX_IMAGES) {
        bool found = false;
        for (int s = 0; s < count; s++) found = found || sameImage(chartPoint[s], q);
        return found;
    }
    [mutating] void put(int s, NewtonResult result) {
        chartPoint[s] = result.chartPoint; sweptAngle[s] = result.sweptAngle;
        rayDirection[s] = result.rayDirection;
    }
    [mutating] void add(NewtonResult result) {   // into the first free slot
        int s = 0;
        while (s < MAX_IMAGES && chartPoint[s].w != 0) s++;
        if (s < MAX_IMAGES) put(s, result);
    }
};
void solvePoint(int pointId) {
    if (pointId >= POINT_COUNT || !imagesFit(pointId) || !toggleOn(pointId < PARTICLE_COUNT ? 0 : 1)) return;
    Body body = pointBody(pointId); float4 target = targetOf(body.position);
    float r = target.w; float tolerance = 3e-4*cameraDistance(target.xyz, r);
    ImageList list; int kept = 0;
    for (int s = 0; s < MAX_IMAGES; s++) {   // follow last frame's images
        list.put(s, { 0, false, 0, 0 });
        float4 previous = loadChartPoint(pointId, s);
        if (time.frame < 1 || previous.w == 0) continue;
        NewtonResult result = newton(previous, target.xyz, tolerance, 10);
        if (!result.ok || list.contains(result.chartPoint, s)) continue;
        list.put(s, result); kept++;
    }
    // full search: after a rebuild, when all images were lost, and every 4th frame per workgroup (whole
    // workgroups at once, so its threads don't wait on the few that search)
    if (tableDirty() || kept == 0 || (uint(pointId)/64 + time.frame)%4 == 0) {
        int nearest = 0;
        for (int j = 1; j < SHELL_COUNT; j++)
            if (abs(log(P.shellRadius[j]/r)) < abs(log(P.shellRadius[nearest]/r))) nearest = j;
        int towards = r > P.shellRadius[nearest] ? 1 : -1;
        float3 direction = target.xyz/r;   // same (theta, phi_out) encoding as the raster grid
        for (int band = 0; band < BAND_COUNT; band++) {
            // up to 20 seeds in blocks of 4: the nearest shell's layers, its neighbour texels, the next shell
            // outward, inward, two out; stop after 4 new images, or after 12 seeds if any found
            int newImages = 0;
            for (int q = 0; q < 20; q++) {
                if (newImages >= 4 || (q >= 12 && newImages > 0)) break;
                int block = q/4, l = q%4;
                int shellShift = block == 2 ? 1 : block == 3 ? -1 : block == 4 ? 2 : 0;
                int shell = clamp(nearest + shellShift*towards, 0, SHELL_COUNT - 1);
                int2 offset = int2(l == 0 ? 1 : l == 1 ? -1 : 0, l == 2 ? 1 : l == 3 ? -1 : 0);
                float4 seed = block == 1 ? seedAt(nearest, 0, band, direction, offset) : seedAt(shell, l, band, direction, 0);
                if (seed.w == 0) continue;
                NewtonResult result = newton(seed, target.xyz, tolerance, 20);
                if (!result.ok || int(result.sweptAngle/PI) != band || list.contains(result.chartPoint)) continue;
                newImages++; list.add(result);
            }
        }
        // near the camera the shells are sparse: seed from the straight line of sight
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
                NewtonResult result = newton(float4(chartPoint.xy, kStart, chartPoint.w), target.xyz, tolerance, 24);
                if (result.ok && int(result.sweptAngle/PI) == 0 && !list.contains(result.chartPoint)) list.add(result);
                break;
            }
        }
    }
    float3 tableVelocity = toTableFrame(body.velocity); float3 tablePosition = toTableFrame(body.position);
    for (int s = 0; s < MAX_IMAGES; s++) {
        float4 q = list.chartPoint[s]; float2 xy = chartToScreen(q);
        float g = q.w == 0 ? 1 : redshift(xy, tablePosition, tableVelocity, list.rayDirection[s]);
        storeImage(pointId, s, q, q.w == 0 ? 0 : float4(xy, g, list.sweptAngle[s]));
    }
}
// One dispatch for all per-frame work before the raster: the background cache (only after a view change,
// each thread taking every 65536th pixel), then one point per thread.
[shader("compute")]
[WorkgroupCount(1024, 1, 1)] [numthreads(64, 1, 1)]
void solve(uint3 id : SV_DispatchThreadID) {
    int pixelCount = int(SCREEN_SIZE.x*SCREEN_SIZE.y);
    for (int pixel = int(id.x); pixel < pixelCount && P.viewDirty != 0; pixel += 1024*64)
        cacheBackground(int2(pixel%int(SCREEN_SIZE.x), pixel/int(SCREEN_SIZE.x)));
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
void rasterParticle(int i) {   // all images of particle i as round dots
    float heat = pow(4.5/length(particle(i).position), 0.75);
    float radius = P.pointRadius;
    for (int s = 0; s < MAX_IMAGES; s++) {
        float4 chartPoint = loadChartPoint(i, s); float4 screenData = loadScreenData(i, s);
        int order = orderOf(screenData.w);
        if (chartPoint.w == 0 || !orderOn(order)) continue;
        float g = screenData.z;
        float3 colour = blackbody(3500 + 9000*heat*g)*(0.35 + 1.4*heat);
        colour *= dopplerTint(g)*orderGain(order)*(0.8 + 0.4*hashUint(uint(7*i)));
        float2 centre = toPixel(screenData.xy); int R = min(int(ceil(radius)), 8);
        for (int dy = -R; dy <= R; dy++) for (int dx = -R; dx <= R; dx++) {
            float2 pixel = floor(centre) + float2(dx, dy) + 0.5;
            if (length(pixel - centre) <= radius) fragment(int2(pixel), chartPoint.z/STEP_COUNT, colour);
        }
    }
}
// barycentric blend of three chart points (Cartesian near the centre, via sigma across the curve)
float4 blendChart(float4 q0, float4 q1, float4 q2, float a, float b) {
    float w0 = 1 - a - b; float k = w0*q0.z + a*q1.z + b*q2.z;
    if (nearCentre(q0) && nearCentre(q1) && nearCentre(q2))
        return float4(fromCentreXY(w0*toCentreXY(q0) + a*toCentreXY(q1) + b*toCentreXY(q2)), k, -1);
    float psiIndex = wrapPsi(q0.x + a*psiDifference(q0.x, q1.x) + b*psiDifference(q0.x, q2.x));
    if (q0.w == q1.w && q1.w == q2.w) return float4(psiIndex, w0*q0.y + a*q1.y + b*q2.y, k, q0.w);
    float2 rowSide = rowFromSigma(w0*sigmaOf(q0) + a*sigmaOf(q1) + b*sigmaOf(q2), psiIndex);
    return float4(psiIndex, rowSide.x, k, rowSide.y);
}
// continue one image along a world segment, with adaptive steps
bool follow(inout float4 chartPoint, float3 worldFrom, float3 worldTo) {
    float t = 0, h = 0.5;
    for (int it = 0; it < 28 && h >= 1/128.; it++) {
        float tNext = min(t + h, 1.);
        float4 target = targetOf(lerp(worldFrom, worldTo, tNext));
        float tolerance = (tNext >= 1 ? 7e-4 : 4e-3)*cameraDistance(target.xyz, target.w);
        NewtonResult result = newton(chartPoint, target.xyz, tolerance, 10, 0.25);
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
    Body corners[3] = { objectVertex(object, vertexIds.x), objectVertex(object, vertexIds.y), objectVertex(object, vertexIds.z) };
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
            if (c != anchor) ok = follow(q[c], corners[anchor].position, corners[c].position);
        if (!ok) continue;
        float2 x0 = toPixel(chartToScreen(q[0])), x1 = toPixel(chartToScreen(q[1])), x2 = toPixel(chartToScreen(q[2]));
        // size in 1080p pixels: the subdivision does not depend on the resolution
        float extent = max(length(x1 - x0), max(length(x2 - x1), length(x0 - x2)))/P.pixelScale;
        if (extent > 600) continue;
        int n = clamp(int(ceil(extent/16)), 1, 6);
        if (counters[0].load() + n*n > SUBTRI_CAPACITY*3/4) n = min(n, 2);   // running out: coarser
        uint base = counters[0].add(uint(n*n));
        if (base + n*n > SUBTRI_CAPACITY) return;
        // grid vertices (i, j), i + j <= n, row by row; inner ones are Newton-refined
        float4 grid[28]; float3 normals[28];
        float3 tableVelocity = toTableFrame(corners[0].velocity);
        int idx = 0;
        for (int j = 0; j <= n; j++) for (int i = 0; i <= n - j; i++) {
            float a = i/float(n), b = j/float(n);
            float3 worldPos = corners[0].position + a*(corners[1].position - corners[0].position) + b*(corners[2].position - corners[0].position);
            float4 qq = i + j == 0 ? q[0] : i == n ? q[1] : j == n ? q[2] : blendChart(q[0], q[1], q[2], a, b);
            if (i + j > 0 && i < n && j < n) {
                float4 target = targetOf(worldPos);
                NewtonResult result = newton(qq, target.xyz, 7e-4*cameraDistance(target.xyz, target.w), 8, 0.25);
                if (result.ok && abs(result.chartPoint.z - qq.z) < 6) qq = result.chartPoint;
            }
            float2 xy = chartToScreen(qq);
            float g = redshift(xy, toTableFrame(worldPos), tableVelocity, evaluateTable(qq).dK);
            grid[idx] = float4(toPixel(xy), qq.z/STEP_COUNT, g);
            normals[idx] = normalize((1 - a - b)*corners[0].normal + a*corners[1].normal + b*corners[2].normal + 1e-6);
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
// Both raster passes, for everything: thread i < PARTICLE_COUNT draws particle i, then each workgroup takes
// every 1024th sub-triangle, its 64 threads sharing the triangle's bounding box.
[shader("compute")]
[DispatchCount(2)] [WorkgroupCount(1024, 1, 1)] [numthreads(64, 1, 1)]
void rasterScene(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID, uint localIndex : SV_GroupIndex) {
    if (id.x < PARTICLE_COUNT && toggleOn(0)) rasterParticle(int(id.x));
    uint triCount = min(counters[0].load(), uint(SUBTRI_CAPACITY));
    for (uint t = group.x; t < triCount; t += 1024) {
        float4 pa = subTriangles[t*6], pb = subTriangles[t*6 + 1], pc = subTriangles[t*6 + 2];
        float area = (pb.x - pa.x)*(pc.y - pa.y) - (pb.y - pa.y)*(pc.x - pa.x);
        int2 lo = int2(max(floor(min(pa.xy, min(pb.xy, pc.xy))), 0.));
        int2 hi = int2(min(ceil(max(pa.xy, max(pb.xy, pc.xy))), float2(SCREEN_SIZE)));
        int2 size = hi - lo;
        if (abs(area) < 1e-6 || any(size <= 0) || any(size > int(192*P.pixelScale))) continue;
        float4 na = subTriangles[t*6 + 3], nb = subTriangles[t*6 + 4], nc = subTriangles[t*6 + 5];
        int object = int(na.w), order = int(nb.w); float3 objectCentre = objectVertex(object, 0).centre;
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
static const uint2 LABELS[8] = { uint2(1313427280u, 538989396u), uint2(1162494543u, 542331971u), uint2(1380013139u, 538976339u),   // POINTS OBJECTS STARS
    uint2(542722899u, 1145655879u), uint2(1193297989u, 541346130u), uint2(1163020612u, 538989635u),    // SKY GRID, EH GRID, DIRECT
    uint2(1347374924u, 538980640u), uint2(1347374924u, 538980896u) };                                   // LOOP 1, LOOP 2
static const uint WARNING[6] = { 1111576133u, 1176520012u, 1413566284u, 1411396147u, 1431590981u, 542328146u };   // ENABLE FLOAT32 TEXTURES
static const float3 SWATCH[8] = { float3(0.8, 0.86, 1), float3(0.85, 0.6, 0.45), float3(1), float3(0.35, 0.75, 0.72),
    float3(1, 0.62, 0.22), float3(0.62, 0.76, 0.91), float3(0.37, 0.85, 0.77), float3(0.88, 0.48, 0.84) };
uint charOf(uint packed, int x) { return (packed >> (8*((x/8)%4))) & 255; }
// is position p inside an 8 x 10 character cell lit?
bool glyphLit(uint character, int2 p) {
    int gx = (p.x%8)/2, gy = p.y/2;
    return gx < 3 && gy < 5 && character >= 48 && character < 90 && ((FONT[character - 48] >> (14 - gy*3 - gx)) & 1) == 1;
}
// button under a pixel, or -1 (the UI is drawn at an integer zoom)
int uiButton(float2 pixel, float hidden, int uiScale) {
    int2 q = int2(pixel)/uiScale - int2(8, 26); int button = q.y/19;
    if (hidden > 0.5 || any(q < 0) || q.x >= 100 || button >= 8 || q.y%19 >= 16) return -1;
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
    int2 q = pixel/P.uiScale - int2(8, 186);
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
    // rasterised points and objects; the depth buffer is cleared here for the next frame
    uint index = id.y*SCREEN_SIZE.x + id.x;
    if (index < MAX_PIXELS) {
        if (depthBuffer[index].load() != 0) colour = pass_in[int3(pixel, COLOUR_LAYER)].rgb;
        depthBuffer[index].store(0);
    }
    colour = 1 - exp(-colour*P.exposure);
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
