import std;

typealias vec2f = float2;
typealias vec3f = float3;
typealias vec4f = float4;

typealias mat2f = float2x2;
typealias mat3f = float3x3;
typealias mat4f = float4x4;

typealias vec8f = float[8];

interface IDot<T, S> { static S dot(T, T); }

extension<T: IArithmetic, let N: int> T[N]: IDot<T[N], T> {
    static T dot(T[N] x, T[N] y) {
        var z = T(0);
        [ForceUnroll]
        for (var i = 0; i<N; i++) { z = z + x[i]*y[i]; }
        return z;
    }
}

interface IJoin {
    static This bottom(); // Identity for join
    static This join(This x, This y);
}

interface IMeet {
    static This top(); // Identity for meet
    static This meet(This x, This y);
}

interface ILattice: IJoin, IMeet {}

T operator|<T>(T x, T y) where T : IJoin { return T.join(x, y); }
T operator&<T>(T x, T y) where T : IMeet { return T.meet(x, y); }

[ForceInline]
internal void exterior3_bottom(out vec8f z) {
    z[0] = 1.;
    z[1] = 0.;
    z[2] = 0.;
    z[3] = 0.;
    z[4] = 0.;
    z[5] = 0.;
    z[6] = 0.;
    z[7] = 0.;
}

[ForceInline]
internal void exterior3_join(in vec8f x, in vec8f y, out vec8f z) {
    z[0] = x[0]*y[0];
    z[1] = x[0]*y[1] + x[1]*y[0];
    z[2] = x[0]*y[2] + x[2]*y[0];
    z[3] = x[0]*y[3] + x[1]*y[2] - x[2]*y[1] + x[3]*y[0];
    z[4] = x[0]*y[4] + x[4]*y[0];
    z[5] = x[0]*y[5] + x[1]*y[4] - x[4]*y[1] + x[5]*y[0];
    z[6] = x[0]*y[6] + x[2]*y[4] - x[4]*y[2] + x[6]*y[0];
    z[7] = x[0]*y[7] + x[1]*y[6] - x[2]*y[5] + x[3]*y[4] + x[4]*y[3] - x[5]*y[2] + x[6]*y[1] + x[7]*y[0];
}

[ForceInline]
internal void exterior3_top(out vec8f z) {
    z[0] = 0.;
    z[1] = 0.;
    z[2] = 0.;
    z[3] = 0.;
    z[4] = 0.;
    z[5] = 0.;
    z[6] = 0.;
    z[7] = 1.;
}

[ForceInline]
internal void exterior3_meet(in vec8f x, in vec8f y, out vec8f z) {
    z[0] = x[0]*y[7] + x[1]*y[6] - x[2]*y[5] + x[3]*y[4] + x[4]*y[3] - x[5]*y[2] + x[6]*y[1] + x[7]*y[0];
    z[1] = x[1]*y[7] + x[3]*y[5] - x[5]*y[3] + x[7]*y[1];
    z[2] = x[2]*y[7] + x[3]*y[6] - x[6]*y[3] + x[7]*y[2];
    z[3] = x[3]*y[7] + x[7]*y[3];
    z[4] = x[4]*y[7] + x[5]*y[6] - x[6]*y[5] + x[7]*y[4];
    z[5] = x[5]*y[7] + x[7]*y[5];
    z[6] = x[6]*y[7] + x[7]*y[6];
    z[7] = x[7]*y[7];
}

[ForceInline]
internal void exterior3_rdual(in vec8f x, out vec8f z) {
    z[0] = x[7];
    z[1] = x[6];
    z[2] = -x[5];
    z[3] = x[4];
    z[4] = x[3];
    z[5] = -x[2];
    z[6] = x[1];
    z[7] = x[0];
}

[ForceInline]
internal void exterior3_ldual(in vec8f x, out vec8f z) {
    z[0] = x[7];
    z[1] = x[6];
    z[2] = -x[5];
    z[3] = x[4];
    z[4] = x[3];
    z[5] = -x[2];
    z[6] = x[1];
    z[7] = x[0];
}

struct Exterior3Float: ILattice {
    typealias TScalar = float;
    typealias TVector = vec8f;

    static This bottom() {
        This z;
        exterior3_bottom(z.data);
        return z;
    }

    static This join(This x, This y) {
        This z;
        exterior3_join(x.data, y.data, z.data);
        return z;
    }

    static This top() {
        This z;
        exterior3_top(z.data);
        return z;
    }

    static This meet(This x, This y) {
        This z;
        exterior3_meet(x.data, y.data, z.data);
        return z;
    }

    static This rdual(This x) {
        This z;
        exterior3_rdual(x.data, z.data);
        return z;
    }

    static This ldual(This x) {
        This z;
        exterior3_ldual(x.data, z.data);
        return z;
    }

    static TScalar norm2(This x) {
        return vec8f.dot(x.data, x.data);
    }

    static This fromGrade0(float x) {
        This z;
        z.data[0] = x;
        return z;
    }

    static This fromGrade1(vec3f x) {
        This z;
        z.data[1] = x[0];
        z.data[2] = x[1];
        z.data[4] = x[2];
        return z;
    }

    static This fromGrade2(vec3f x) {
        This z;
        z.data[3] = x[0];
        z.data[5] = x[1];
        z.data[6] = x[2];
        return z;
    }

    static This fromGrade3(float x) {
        This z;
        z.data[7] = x;
        return z;
    }

    TVector data;
}

typealias ext3f = Exterior3Float;

static let PI: float = 3.1415926535897932384626;
static let TAU: float = 2.*PI;
static let EPSILON: float = 0.001;

static let BLUR: float = 0.001;

ext3f up(vec2f x) {
    // a*x^2 + b*y + c
    return ext3f.rdual(ext3f.fromGrade1(vec3f(x[0]*x[0], x[1], 1.)));
}

float dist(vec2f p, ext3f o) {
    return sqrt(abs(ext3f.norm2(o & up(p))));
}

void plot(in out vec3f outColor, vec2f p, ext3f o, float width, vec3f color) {
    let d = dist(p, o);

    // TODO: Use automatic differentiation for gradient
    let e = vec2f(EPSILON, 0.);
    let g = vec2f(d-dist(p+e.xy, o), d-dist(p+e.yx, o))/e.xx;

    let s = 1. - smoothstep(width-BLUR, width+BLUR, abs(d/length(g)));
    outColor = lerp(outColor, color, s);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main_image(uint3 id : SV_DispatchThreadID)
{
    let p = (2.*vec2f(id.xy) - vec2f(SCREEN_SIZE))/SCREEN_SIZE.y;

    var color = vec3f(1.);
    
    let t = time.elapsed;

    let parabola = ext3f.fromGrade1(vec3f(1., 1., -.5));
    let line = ext3f.fromGrade1(vec3f(0, 1., sin(TAU*t/10.)));

    let points = parabola | line;

    plot(color, p, parabola, .01, vec3f(1.,0.,0.));
    plot(color, p, line, .01, vec3f(0.,0.,1.));
    plot(color, p, points, .02, vec3f(0.,1.,0.));

    screen[id.xy] = vec4f(color, 1.);
}
