
import std;

typealias vec2 = float2;
typealias vec3 = float3;
typealias vec4 = float4;

// ==== SDF2D ===

interface SDF2D {
    float sdf2d(vec2 p);
}

struct SDF2DCircle : SDF2D {
    float size;

    __init(float s){
        size = s;
    }

    float sdf2d(vec2 p){
        return length(p)-size;
    }
}

struct SDF2DBox : SDF2D{
    vec2 size;

    __init(vec2 s){
        size = s;
    }

    float sdf2d(vec2 p){
        vec2 d = abs(p)-size;
        return length(max(d,0.0)) + min(max(d.x,d.y),0.0);
    }
}

struct SDF2DUnion<S1: SDF2D, S2: SDF2D> : SDF2D {
    S1 sdfa;
    S2 sdfb;
    float k;

    __init(S1 a, S2 b, float blend) {
        sdfa = a;
        sdfb = b;
        k = blend;
    }

    float sdf2d(vec2 p) {
        float a = sdfa.sdf2d(p);
        float b = sdfb.sdf2d(p);
        float h = clamp( 0.5 + 0.5*(b-a)/k, 0.0, 1.0 );
        return lerp( b, a, h ) - k*h*(1.0-h);
    }
}

struct SDF2DTranslate<S: SDF2D> : SDF2D {
    S sdf;
    vec2 offset;

    __init(S shape, vec2 translation) {
        sdf = shape;
        offset = translation;
    }

    float sdf2d(vec2 p) {
        return sdf.sdf2d(p - offset);
    }
}

struct SDF2DRotate<S: SDF2D> : SDF2D {
    S sdf;
    float angle;

    __init(S shape, float rotation) {
        sdf = shape;
        angle = rotation;
    }

    float sdf2d(vec2 p) {
        float c = cos(angle);
        float s = sin(angle);
        vec2 rotated = vec2(
            p.x * c - p.y * s,
            p.x * s + p.y * c
        );
        return sdf.sdf2d(rotated);
    }
}

struct SDF2DScale<S: SDF2D> : SDF2D {
    S sdf;
    vec2 scale;

    __init(S shape, vec2 scaling) {
        sdf = shape;
        scale = scaling;
    }

    float sdf2d(vec2 p) {
        vec2 scaled = p / scale;
        return sdf.sdf2d(scaled) * min(scale.x, scale.y);
    }
}

namespace sdf2d {

    func circle(float size) -> SDF2DCircle {
        return SDF2DCircle(size);
    }

    func box(vec2 size) -> SDF2DBox {
        return SDF2DBox(size);
    }

    func union<S1: SDF2D, S2: SDF2D>(S1 sdf1, S2 sdf2, float k = 0.0) -> SDF2DUnion<S1, S2> {
        return SDF2DUnion<S1, S2>(sdf1, sdf2, k);
    }

    func translate<S: SDF2D>(S shape, vec2 offset) -> SDF2DTranslate<S> {
        return SDF2DTranslate<S>(shape, offset);
    }

    func rotate<S: SDF2D>(S shape, float angle) -> SDF2DRotate<S> {
        return SDF2DRotate<S>(shape, angle);
    }

    func scale<S: SDF2D>(S shape, vec2 scale) -> SDF2DScale<S> {
        return SDF2DScale<S>(shape, scale);
    }

}

// ==== ImageRGB ===

interface ImageRGB {
    vec3 render(vec2 p);
}

struct RenderSDF2D<T: SDF2D> : ImageRGB{
    T sdf2d;

    vec3 render(vec2 p){
        float c=sdf2d.sdf2d(p);
        vec3 col = vec3(1.0) - sign(c)*vec3(0.1,0.4,0.7);
        col *= 1.0 - exp(-2.0*abs(c));
        col *= 0.8 + 0.2*cos(128.0*abs(c));
        col = lerp( col, vec3(1.0), 1.0-smoothstep(0.0,0.015,abs(c)) );

        return col;
    
    }
}

namespace imagergb {

    func rendersdf2d<S: SDF2D>(S sdf) -> RenderSDF2D<S> {
        return RenderSDF2D<S>(sdf);
    }

}


[shader("compute")]
[numthreads(16, 16, 1)]
void main_image(uint3 id : SV_DispatchThreadID)
{
    uint width, height;
    screen.GetDimensions(width, height);
    if (id.x >= width || id.y >= height) return;
    vec2 xy = (float2(float(id.x) , float(id.y)) * 2.0f - vec2(width, height)) / height;

    vec3 pixelcolor =
    imagergb::rendersdf2d(
        sdf2d::union(
            sdf2d::translate(
                sdf2d::rotate(
                    sdf2d::scale(
                        sdf2d::box(vec2(custom.BoxSize,custom.BoxSize)),
                        vec2(sin(time.elapsed*10.0)*0.2+1.0)
                    ),
                    time.elapsed
                ),
                vec2(sin(time.elapsed),0.0)),
            sdf2d::circle(custom.CircleSize),
            0.5
        )
    ).render(xy);

    screen[id.xy] = vec4(pixelcolor, 1.0);

}
