const PI = 3.141592653589793;
const TAU = 2. * PI;
const RESOLUTION = vec2f(SCREEN_WIDTH,SCREEN_HEIGHT);
const ASPECT = max(vec2f(RESOLUTION.x/RESOLUTION.y,RESOLUTION.y/RESOLUTION.x),vec2f(1));

fn eotf(x:vec3f)->vec3f{return pow(x,vec3f(2.2));}
fn fragcoord(p:vec2f,size:vec2f)->vec2f{return vec2f(p.x+.5,size.y-p.y-.5);}
fn texcoord(id:vec2u,size:vec2f)->vec2f{return fragcoord(vec2f(id),size)/size;}
fn quintic(x:f32)->f32{return x*x*x*(x*(x*6-15)+10);}
fn cro(a:vec2f,b:vec2f)->f32{return a.x*b.y-a.y*b.x;}
fn side(a:vec2f,b:vec2f)->f32{return select(0.,1.,cro(a,b)>0);}

fn frand(x:u32) -> f32 {
    const S = 1./1073741824.;
    var s = (x << 13u) ^ x;
    s = (s * (s * s * 15731u + 789221u) + 1376312589u) & 0x7fffffffu;
    return 1. - f32(s) * S;
}

fn rand(s:u32) -> f32 {
    return frand(s) * .5 + .5;
}

fn hash13(x:u32) -> vec3f {
    const S = 1. / 0x7fffffff;
    var n = (x << 13u) ^ x;
    n = n * (n * n * 15731u + 789221u) + 1376312589u;
    let k = n * vec3u(n,n*16807u,n*48271u);
    return vec3f(k & vec3u(0x7fffffffu)) * S;
}

fn rand_disk(s:u32) -> vec2f {
    let theta = TAU * rand(s);
    return vec2f(cos(theta), sin(theta));
}

struct Ray { o: vec2f, d: vec2f }
struct BezierQuad { p0: vec2f, p1: vec2f, p2: vec2f }
struct BezierCubic { p0:vec2f,p1:vec2f,p2:vec2f,p3:vec2f }
fn bezier_quad(B:BezierQuad, t: f32) -> vec2f {
    let s = 1 - t;
    return s*s*B.p0 + 2*t*s*B.p1 + t*t*B.p2;
}
fn bezier_quad_d1(B:BezierQuad,t:f32) -> vec2f {
    return 2*(B.p0-2*B.p1+B.p2)*t - 2*(B.p0-B.p1);
}
fn bezier_cubic(B:BezierCubic,t:f32) -> vec2f {
    let s = 1 - t;
    return s*s*s*B.p0 + 3*s*s*t*B.p1 + 3*s*t*t*B.p2 + t*t*t*B.p3;
}
fn bezier_cubic_d1(B:BezierCubic,t:f32) -> vec2f {
    let s = 1 - t;
    return 3*s*s*(B.p1-B.p0) + 6*s*t*(B.p2-B.p1) + 3*t*t*(B.p3-B.p2);
}
fn bezier_cubic_d2(B:BezierCubic,t:f32) -> vec2f {
    return 6*(1-t)*(B.p2-2*B.p1+B.p0) + 6*t*(B.p3-2*B.p2+B.p1);
}
fn cu2quc1(B:BezierCubic,gamma:f32) -> array<BezierQuad,2> {
    let q0 = B.p0; let q4 = B.p3;
    let q1 = B.p0 + 1.5 * gamma * (B.p1 - B.p0);
    let q3 = B.p3 + 1.5 * (1.-gamma) * (B.p2 - B.p3);
    let q2 = mix(q1, q3, gamma);
    return array<BezierQuad,2>(BezierQuad(q0,q1,q2), BezierQuad(q2,q3,q4));
}

fn isec_ray_bezier_quad(ray:Ray,B:BezierQuad) -> vec4f {
    let a = cro(ray.d,B.p0-2*B.p1+B.p2);
    let b = cro(ray.d,-2*(B.p0-B.p1)); 
    let c = cro(ray.d,B.p0-ray.o);
    if (abs(a) < 1e-8) {
        if (abs(b) < 1e-8) { return vec4f(-1); }
        let u = -c / b;
        let t = select(-1,dot(bezier_quad(B,u)-ray.o,ray.d),0<=u&&u<=1);
        return vec4f(t,t,u,u);
    }

    let det = b * b - 4. * a * c;
    if (det < 0.) { return vec4f(-1); }

    let t = (-b + sqrt(det)*vec2(1,-1))/(2*a);
    let t0 = select(-1,dot(bezier_quad(B,t.x)-ray.o,ray.d),0<=t.x&&t.x<=1);
    let t1 = select(-1,dot(bezier_quad(B,t.y)-ray.o,ray.d),0<=t.y&&t.y<=1);

    return vec4f(t0,t1,t);
}

#define N_POINTS 5
#define N_CURVES 15
#define DURATION 1200

#define N_CURVES_S 30
const CURVES_S = array<BezierQuad,N_CURVES_S>(
BezierQuad(vec2f(0.5247,0.5630),vec2f(0.3523,0.6821),vec2f(0.2406,0.6594)),
BezierQuad(vec2f(0.0447,0.6196),vec2f(-0.0987,0.5904),vec2f(-0.1375,0.4742)),
BezierQuad(vec2f(-0.1211,0.0095),vec2f(-0.1090,-0.0235),vec2f(0.0560,-0.0609)),
BezierQuad(vec2f(0.1846,-0.0930),vec2f(0.2794,-0.1187),vec2f(0.2986,-0.1474)),
BezierQuad(vec2f(0.3065,-0.3420),vec2f(0.2681,-0.3938),vec2f(0.2143,-0.3909)),
BezierQuad(vec2f(-0.1622,-0.5389),vec2f(-0.1255,-0.5785),vec2f(-0.0352,-0.6045)),
BezierQuad(vec2f(-0.1449,-0.3174),vec2f(-0.1692,-0.3508),vec2f(-0.2116,-0.4091)),
BezierQuad(vec2f(0.0434,-0.6272),vec2f(0.1430,-0.6559),vec2f(0.2274,-0.6411)),
BezierQuad(vec2f(0.3400,-0.6241),vec2f(0.3831,-0.6181),vec2f(0.4416,-0.5786)),
BezierQuad(vec2f(0.5007,-0.5387),vec2f(0.5632,-0.4964),vec2f(0.5824,-0.4415)),
BezierQuad(vec2f(0.6192,-0.3442),vec2f(0.6538,-0.2546),vec2f(0.6311,-0.1592)),
BezierQuad(vec2f(0.3569,0.1265),vec2f(0.3104,0.1507),vec2f(0.2402,0.1675)),
BezierQuad(vec2f(0.1308,0.1936),vec2f(0.0938,0.2024),vec2f(0.1011,0.2763)),
BezierQuad(vec2f(0.1125,0.3672),vec2f(0.1162,0.3946),vec2f(0.1827,0.3998)),
BezierQuad(vec2f(0.3208,0.4106),vec2f(0.3528,0.4131),vec2f(0.4243,0.3479)),
BezierQuad(vec2f(0.5053,0.3915),vec2f(0.5219,0.4186),vec2f(0.5564,0.4750)),
BezierQuad(vec2f(-0.3078,0.5687),vec2f(-0.3078,0.5831),vec2f(-0.3078,0.6112)),
BezierQuad(vec2f(-0.3436,0.6480),vec2f(-0.3698,0.6480),vec2f(-0.4015,0.6480)),
BezierQuad(vec2f(-0.4219,0.4404),vec2f(-0.4695,0.4875),vec2f(-0.5505,0.5676)),
BezierQuad(vec2f(-0.6318,0.5945),vec2f(-0.6318,0.5519),vec2f(-0.6318,0.4960)),
BezierQuad(vec2f(-0.5800,0.4041),vec2f(-0.4701,0.2955),vec2f(-0.3787,0.2051)),
BezierQuad(vec2f(-0.3078,0.1544),vec2f(-0.3078,0.2311),vec2f(-0.3078,0.3060)),
BezierQuad(vec2f(-0.3910,0.0228),vec2f(-0.4695,0.1005),vec2f(-0.5723,0.2022)),
BezierQuad(vec2f(-0.5074,-0.0546),vec2f(-0.4701,-0.0915),vec2f(-0.3420,-0.2182)),
BezierQuad(vec2f(-0.3078,-0.2255),vec2f(-0.3078,-0.1559),vec2f(-0.3078,-0.1124)),
BezierQuad(vec2f(-0.3429,-0.4116),vec2f(-0.4695,-0.2865),vec2f(-0.5669,-0.1902)),
BezierQuad(vec2f(-0.5114,-0.4377),vec2f(-0.4701,-0.4785),vec2f(-0.3906,-0.5571)),
BezierQuad(vec2f(-0.3078,-0.5907),vec2f(-0.3078,-0.5429),vec2f(-0.3078,-0.4839)),
BezierQuad(vec2f(-0.6318,-0.5337),vec2f(-0.6318,-0.5813),vec2f(-0.6318,-0.6193)),
BezierQuad(vec2f(-0.5524,-0.5960),vec2f(-0.5669,-0.5815),vec2f(-0.5933,-0.5551)),
);


struct Curve {
    pts: array<vec2f,N_POINTS>,
    t: array<vec2f,2>,
    u: array<f32,N_POINTS>,
    A: array<vec4f,N_POINTS>,

    p: BezierCubic,
    n: u32,
    err: f32,
}
#storage curves array<Curve,N_CURVES>

fn get_t() -> f32 {
    return quintic(saturate(min(f32(time.frame), DURATION)/DURATION));
}

#workgroup_count init 1 1 1 
@compute @workgroup_size(N_CURVES)
fn init(@builtin(global_invocation_id) id: vec3u) {
    if (curves[id.x].err > 1e-1 && curves[id.x].n < 60) { return; }

    curves[id.x].err = 0.;
    curves[id.x].n = 0;
    let seed = time.frame * N_CURVES + id.x;

    var pts: array<vec2f,N_POINTS>;
    for (var i = 0u; i < N_POINTS; i++) {
        pts[i] = hash13(seed + i).xy*2-1;
    }
    curves[id.x].pts = pts;

    for (var i = 1; i < N_POINTS; i++) {
        curves[id.x].u[i] = curves[id.x].u[i-1] + length(pts[i]-pts[i-1]);
    }
    for (var i = 1; i < N_POINTS; i++) {
        curves[id.x].u[i] /= curves[id.x].u[N_POINTS-1];
    }
    let t1 = normalize(pts[1] - pts[0]);
    let t2 = normalize(pts[N_POINTS-2] - pts[N_POINTS-1]);
    curves[id.x].t = array<vec2f,2>(t1,t2);
    for (var i = 0; i < N_POINTS; i++) {
        let u = curves[id.x].u[i];
        curves[id.x].A[i] = vec4f(t1*3*(1-u)*(1-u)*u,t2*3*(1-u)*u*u);
    }
}

#workgroup_count fit 1 1 1 
@compute @workgroup_size(N_CURVES)
fn fit(@builtin(global_invocation_id) id: vec3u) {
    let c = curves[id.x];
    let pts = c.pts;

    var C: vec4f; var X: vec2f;
    let curve = BezierCubic(pts[0],pts[0],pts[N_POINTS-1],pts[N_POINTS-1]);
    for (var i = 0; i < N_POINTS; i++) {
        let A = c.A[i];
        C.x += dot(A.xy,A.xy);
        let yz = dot(A.xy,A.zw);
        C.y += yz;
        C.z += yz;
        C.w += dot(A.zw,A.zw);
        let tmp = pts[i] - bezier_cubic(curve, c.u[i]);
        X.x += dot(A.xy, tmp);
        X.y += dot(A.zw, tmp);
    }
    let det_c0_c1 = C.x * C.w - C.z * C.y;
    let det_c0_x = C.x * X.y - C.z * X.x;
    let det_x_c1 = X.x * C.w - X.y * C.y;
    let a_1 = select(det_x_c1 / det_c0_c1, 0, det_c0_c1 == 0.);
    let a_2 = select(det_c0_x / det_c0_c1, 0, det_c0_c1 == 0.);

    let t1 = c.t[0]; let t2 = c.t[1];
    let p = BezierCubic(pts[0],pts[0]+t1*a_1,pts[N_POINTS-1]+t2*a_2,pts[N_POINTS-1]);
    curves[id.x].p = p;

    var err = 0.;
    for (var i = 0; i < N_POINTS; i++) {
        err = max(err, length(bezier_cubic(p, c.u[i]) - pts[i]));
    }
    curves[id.x].err = err;
    curves[id.x].n++;
}

@compute @workgroup_size(16,16)
fn mc(@builtin(global_invocation_id) id: vec3u) {
    if (any(id.xy>=textureDimensions(screen))) { return; }

    let p = (texcoord(id.xy,RESOLUTION)*2-1)*ASPECT;
    let seed = id.x + id.y * SCREEN_WIDTH + time.frame*SCREEN_WIDTH*SCREEN_HEIGHT;

    var u = 0.;
    let ray = Ray(p,rand_disk(seed));
    let t = get_t();

    for (var i = 0; i < N_CURVES;i++) {
        let qs = cu2quc1(curves[i].p,.5);
        for (var j = 0; j < 2; j++) {
            let k = i * 2 + j;
            var q = CURVES_S[k];
            if (k % 4 == 0) { q = BezierQuad(q.p2,q.p1,q.p0); }
            let b = BezierQuad(
                mix(qs[j].p0,q.p0,t),
                mix(qs[j].p1,q.p1,t),
                mix(qs[j].p2,q.p2,t),
            );
            let isec = isec_ray_bezier_quad(ray,b);
            if (isec.x > 0) { u += mix(-1.,1.,side(bezier_quad_d1(b,isec.z),ray.d)); }
            if (isec.y > 0) { u += mix(-1.,1.,side(bezier_quad_d1(b,isec.w),ray.d)); }
        }
    }

    let val = vec4f(u,0,0,0);

    let k = 1/mix(mix(1,10,t),f32(min(time.frame,300)),rand(seed+1));
    let coord = vec2i(id.xy);
    passStore(0,coord,mix(passLoad(0,coord,0),val,k));
}

@compute @workgroup_size(16, 16)
fn render(@builtin(global_invocation_id) id: vec3u) {
    if (any(id.xy>=textureDimensions(screen))) { return; }

    let data = passLoad(0,vec2i(id.xy),0);
    let val = data.x * mix(2./N_CURVES,1,get_t());
    let col = vec3f(val*.5+.5);
    textureStore(screen, id.xy, vec4f(eotf(col), 1));
}

#workgroup_count upate N_CURVES 1 1
@compute @workgroup_size(N_POINTS)
fn update(
    @builtin(workgroup_id) wid: vec3u,
    @builtin(local_invocation_id) id: vec3u,
) {
    let curve = curves[wid.x];
    if (curves[id.x].err > mix(.8,.5,get_t())) { return; }

    let u = curve.u[id.x];

    let d = bezier_cubic(curve.p, u) - curve.pts[id.x];
    let d1 = bezier_cubic_d1(curve.p, u);
    let denom = (d1*d1 + d * bezier_cubic_d2(curve.p,u));
    let denom1 = denom.x + denom.y;

    let u1 = u - select(dot(d,d1)/denom1,0,denom1 == 0.);
    curves[wid.x].u[id.x] = u1;

    let t1 = curve.t[0]; let t2 = curve.t[1];
    curves[wid.x].A[id.x] = vec4f(t1*3*(1-u1)*(1-u1)*u1,t2*3*(1-u1)*u1*u1);
}