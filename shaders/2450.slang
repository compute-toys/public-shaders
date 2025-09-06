import std;

typealias vec2f = float2;
typealias vec3f = float3;
typealias vec4f = float4;

typealias mat2f = float2x2;
typealias mat3f = float3x3;
typealias mat4f = float4x4;

typealias vec6f = float[6];
typealias vec8f = float[8];
typealias vec16f = float[16];

interface IDot<V, S> { static S dot(V, V); }

T operator+=<T: IArithmetic>(out T x, T y) { x = x + y; return x; }
T operator*=<T: IArithmetic>(out T x, T y) { x = x * y; return x; }

extension<T: IArithmetic, let N: int> T[N]: IDot<T[N], T> {
    static T dot(T[N] x, T[N] y) {
        var z = T(0);
        [ForceUnroll]
        for (var i = 0; i<N; i++) { z += x[i]*y[i]; }
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

T operator|<T: IJoin>(T x, T y) { return T.join(x, y); }
T operator&<T: IMeet>(T x, T y) { return T.meet(x, y); }

[ForceInline]
internal void exterior4_bottom(out vec16f z) {
    z[ 0] = 1.;
    z[ 1] = 0.;
    z[ 2] = 0.;
    z[ 3] = 0.;
    z[ 4] = 0.;
    z[ 5] = 0.;
    z[ 6] = 0.;
    z[ 7] = 0.;
    z[ 8] = 0.;
    z[ 9] = 0.;
    z[10] = 0.;
    z[11] = 0.;
    z[12] = 0.;
    z[13] = 0.;
    z[14] = 0.;
    z[15] = 0.;
}

[ForceInline]
internal void exterior4_join(in vec16f x, in vec16f y, out vec16f z) {
    z[ 0] = x[ 0]*y[ 0];
    z[ 1] = x[ 0]*y[ 1] + x[ 1]*y[ 0];
    z[ 2] = x[ 0]*y[ 2] + x[ 2]*y[ 0];
    z[ 3] = x[ 0]*y[ 3] + x[ 1]*y[ 2] - x[ 2]*y[ 1] + x[ 3]*y[ 0];
    z[ 4] = x[ 0]*y[ 4] + x[ 4]*y[ 0];
    z[ 5] = x[ 0]*y[ 5] + x[ 1]*y[ 4] - x[ 4]*y[ 1] + x[ 5]*y[ 0];
    z[ 6] = x[ 0]*y[ 6] + x[ 2]*y[ 4] - x[ 4]*y[ 2] + x[ 6]*y[ 0];
    z[ 7] = x[ 0]*y[ 7] + x[ 1]*y[ 6] - x[ 2]*y[ 5] + x[ 3]*y[ 4] + x[ 4]*y[ 3] - x[ 5]*y[ 2] + x[ 6]*y[ 1] + x[ 7]*y[ 0];
    z[ 8] = x[ 0]*y[ 8] + x[ 8]*y[ 0];
    z[ 9] = x[ 0]*y[ 9] + x[ 1]*y[ 8] - x[ 8]*y[ 1] + x[ 9]*y[ 0];
    z[10] = x[ 0]*y[10] + x[ 2]*y[ 8] - x[ 8]*y[ 2] + x[10]*y[ 0];
    z[11] = x[ 0]*y[11] + x[ 1]*y[10] - x[ 2]*y[ 9] + x[ 3]*y[ 8] + x[ 8]*y[ 3] - x[ 9]*y[ 2] + x[10]*y[ 1] + x[11]*y[ 0];
    z[12] = x[ 0]*y[12] + x[ 4]*y[ 8] - x[ 8]*y[ 4] + x[12]*y[ 0];
    z[13] = x[ 0]*y[13] + x[ 1]*y[12] - x[ 4]*y[ 9] + x[ 5]*y[ 8] + x[ 8]*y[ 5] - x[ 9]*y[ 4] + x[12]*y[ 1] + x[13]*y[ 0];
    z[14] = x[ 0]*y[14] + x[ 2]*y[12] - x[ 4]*y[10] + x[ 6]*y[ 8] + x[ 8]*y[ 6] - x[10]*y[ 4] + x[12]*y[ 2] + x[14]*y[ 0];
    z[15] = x[ 0]*y[15] + x[ 1]*y[14] - x[ 2]*y[13] + x[ 3]*y[12] + x[ 4]*y[11] - x[ 5]*y[10] + x[ 6]*y[ 9] + x[ 7]*y[ 8] - x[ 8]*y[ 7] + x[ 9]*y[ 6] - x[10]*y[ 5] - x[11]*y[ 4] + x[12]*y[ 3] + x[13]*y[ 2] - x[14]*y[ 1] + x[15]*y[ 0];
}

[ForceInline]
internal void exterior4_top(out vec16f z) {
    z[ 0] = 0.;
    z[ 1] = 0.;
    z[ 2] = 0.;
    z[ 3] = 0.;
    z[ 4] = 0.;
    z[ 5] = 0.;
    z[ 6] = 0.;
    z[ 7] = 0.;
    z[ 8] = 0.;
    z[ 9] = 0.;
    z[10] = 0.;
    z[11] = 0.;
    z[12] = 0.;
    z[13] = 0.;
    z[14] = 0.;
    z[15] = 1.;
}

[ForceInline]
internal void exterior4_meet(in vec16f x, in vec16f y, out vec16f z) {
    z[ 0] = x[ 0]*y[15] + x[ 1]*y[14] - x[ 2]*y[13] + x[ 3]*y[12] + x[ 4]*y[11] - x[ 5]*y[10] + x[ 6]*y[ 9] + x[ 7]*y[ 8] - x[ 8]*y[ 7] + x[ 9]*y[ 6] - x[10]*y[ 5] - x[11]*y[ 4] + x[12]*y[ 3] + x[13]*y[ 2] - x[14]*y[ 1] + x[15]*y[ 0];
    z[ 1] = x[ 1]*y[15] + x[ 3]*y[13] - x[ 5]*y[11] + x[ 7]*y[ 9] + x[ 9]*y[ 7] - x[11]*y[ 5] + x[13]*y[ 3] + x[15]*y[ 1];
    z[ 2] = x[ 2]*y[15] + x[ 3]*y[14] - x[ 6]*y[11] + x[ 7]*y[10] + x[10]*y[ 7] - x[11]*y[ 6] + x[14]*y[ 3] + x[15]*y[ 2];
    z[ 3] = x[ 3]*y[15] + x[ 7]*y[11] - x[11]*y[ 7] + x[15]*y[ 3];
    z[ 4] = x[ 4]*y[15] + x[ 5]*y[14] - x[ 6]*y[13] + x[ 7]*y[12] + x[12]*y[ 7] - x[13]*y[ 6] + x[14]*y[ 5] + x[15]*y[ 4];
    z[ 5] = x[ 5]*y[15] + x[ 7]*y[13] - x[13]*y[ 7] + x[15]*y[ 5];
    z[ 6] = x[ 6]*y[15] + x[ 7]*y[14] - x[14]*y[ 7] + x[15]*y[ 6];
    z[ 7] = x[ 7]*y[15] + x[15]*y[ 7];
    z[ 8] = x[ 8]*y[15] + x[ 9]*y[14] - x[10]*y[13] + x[11]*y[12] + x[12]*y[11] - x[13]*y[10] + x[14]*y[ 9] + x[15]*y[ 8];
    z[ 9] = x[ 9]*y[15] + x[11]*y[13] - x[13]*y[11] + x[15]*y[ 9];
    z[10] = x[10]*y[15] + x[11]*y[14] - x[14]*y[11] + x[15]*y[10];
    z[11] = x[11]*y[15] + x[15]*y[11];
    z[12] = x[12]*y[15] + x[13]*y[14] - x[14]*y[13] + x[15]*y[12];
    z[13] = x[13]*y[15] + x[15]*y[13];
    z[14] = x[14]*y[15] + x[15]*y[14];
    z[15] = x[15]*y[15];
}

[ForceInline]
internal void exterior4_rdual(in vec16f x, out vec16f z) {
    z[ 0] = x[15];
    z[ 1] = x[14];
    z[ 2] = -x[13];
    z[ 3] = x[12];
    z[ 4] = x[11];
    z[ 5] = -x[10];
    z[ 6] = x[ 9];
    z[ 7] = x[ 8];
    z[ 8] = -x[ 7];
    z[ 9] = x[ 6];
    z[10] = -x[ 5];
    z[11] = -x[ 4];
    z[12] = x[ 3];
    z[13] = x[ 2];
    z[14] = -x[ 1];
    z[15] = x[ 0];
}

[ForceInline]
internal void exterior4_ldual(in vec16f x, out vec16f z) {
    z[ 0] = x[15];
    z[ 1] = -x[14];
    z[ 2] = x[13];
    z[ 3] = x[12];
    z[ 4] = -x[11];
    z[ 5] = -x[10];
    z[ 6] = x[ 9];
    z[ 7] = -x[ 8];
    z[ 8] = x[ 7];
    z[ 9] = x[ 6];
    z[10] = -x[ 5];
    z[11] = x[ 4];
    z[12] = x[ 3];
    z[13] = -x[ 2];
    z[14] = x[ 1];
    z[15] = x[ 0];
}

struct Exterior4Float: ILattice {
    typealias TScalar = float;
    typealias TVector = vec16f;

    static This bottom() {
        This z;
        exterior4_bottom(z.data);
        return z;
    }

    static This join(This x, This y) {
        This z;
        exterior4_join(x.data, y.data, z.data);
        return z;
    }

    static This top() {
        This z;
        exterior4_top(z.data);
        return z;
    }

    static This meet(This x, This y) {
        This z;
        exterior4_meet(x.data, y.data, z.data);
        return z;
    }

    static This rdual(This x) {
        This z;
        exterior4_rdual(x.data, z.data);
        return z;
    }

    static This ldual(This x) {
        This z;
        exterior4_ldual(x.data, z.data);
        return z;
    }

    static TScalar norm2(This x) {
        return vec16f.dot(x.data, x.data);
    }

    static This fromGrade0(float x) {
        This z;
        z.data[0] = x;
        return z;
    }

    static This fromGrade1(vec4f x) {
        This z;
        z.data[ 1] = x[0];
        z.data[ 2] = x[1];
        z.data[ 4] = x[2];
        z.data[ 8] = x[3];
        return z;
    }

    static This fromGrade2(vec6f x) {
        This z;
        z.data[ 3] = x[0];
        z.data[ 5] = x[1];
        z.data[ 6] = x[2];
        z.data[ 9] = x[3];
        z.data[10] = x[4];
        z.data[12] = x[5];
        return z;
    }

    static This fromGrade3(vec4f x) {
        This z;
        z.data[ 7] = x[0];
        z.data[11] = x[1];
        z.data[13] = x[2];
        z.data[14] = x[2];
        return z;
    }

    static This fromGrade4(float x) {
        This z;
        z.data[15] = x;
        return z;
    }

    TVector data;
}

typealias ext4f = Exterior4Float;

static let PI: float = 3.1415926535897932384626;
static let TAU: float = 2.*PI;
static let EPSILON: float = 0.001;

static let BLUR: float = 0.001;

ext4f up(vec2f x) {
    // a + b*x + c*y + d*(x^2 + y^2)
    return ext4f.rdual(ext4f.fromGrade1(vec4f(1., x[0], x[1], x[0]*x[0]+x[1]*x[1])));
}

float dist(vec2f p, ext4f o) {
    return sqrt(abs(ext4f.norm2(o & up(p))));
}

void plot(in out vec3f outColor, vec2f p, ext4f o, float width, vec3f color) {
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

    let a = up(vec2f(-0.6, 0));
    let b = up(vec2f(0, 0.5));
    let c = up(vec2f(0.6, sin(t)));
    
    let circle = a & b & c;

    let line = ext4f.fromGrade1(vec4f(sin(t), 0., 1., 0.));

    plot(color, p, circle, .01, vec3f(1.,0.,0.));
    plot(color, p, line, .01, vec3f(0.,0.,1.));
    plot(color, p, line | circle, .02, vec3f(1.,1.,0.));
    plot(color, p, a, .02, vec3f(0.,1.,0.));
    plot(color, p, b, .02, vec3f(0.,1.,0.));
    plot(color, p, c, .02, vec3f(0.,1.,0.));
    
    screen[id.xy] = vec4f(color, 1.);
}
