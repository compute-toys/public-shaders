import std;
import shadertoy;

mat4 setRotation( float x, float y, float z )
{
    float a = sin(x); float b = cos(x); 
    float c = sin(y); float d = cos(y); 
    float e = sin(z); float f = cos(z); 

    float ac = a*c;
    float bc = b*c;

    return mat4( d*f,      d*e,       -c, 0.0,
                 ac*f-b*e, ac*e+b*f, a*d, 0.0,
                 bc*f+a*e, bc*e-a*f, b*d, 0.0,
                 0.0,      0.0,      0.0, 1.0 );
}

mat4 setTranslation( float x, float y, float z )
{
    return mat4( 1.0, 0.0, 0.0, 0.0,
                 0.0, 1.0, 0.0, 0.0,
                 0.0, 0.0, 1.0, 0.0,
                 x,     y,   z, 1.0 );
}

struct Point {
    vec3 pos;
    vec2 uv;
};

struct Triangle
{
    Point a;
    Point b;
    Point c;
    vec3 n;
};

static const vec3 vertsCube[8] = {
    vec3( -1.0, -1.0, -1.0 ),
    vec3( -1.0, -1.0,  1.0 ),
    vec3( -1.0,  1.0, -1.0 ),
    vec3( -1.0,  1.0,  1.0 ),
    vec3(  1.0, -1.0, -1.0 ),
    vec3(  1.0, -1.0,  1.0 ),
    vec3(  1.0,  1.0, -1.0 ),
    vec3(  1.0,  1.0,  1.0 ) ,
};
static const vec3 vertsFloor[4] = {
    vec3( -3.0, -1.0, -3.0 ),
    vec3( -3.0, -1.0,  3.0 ),
    vec3(  3.0, -1.0,  3.0 ),
    vec3(  3.0, -1.0, -3.0 ),
};

void createCube( inout Array<Triangle, 14> triangles )
{
    triangles[0].a = { vertsCube[1], vec2(0.0,0.0) };
    triangles[0].b = { vertsCube[5], vec2(1.0,0.0) };
    triangles[0].c = { vertsCube[7], vec2(1.0,1.0) };
    triangles[0].n = vec3( 0.0, 0.0, 1.0 );
    triangles[1].a = { vertsCube[1], vec2(0.0,0.0) };
    triangles[1].b = { vertsCube[7], vec2(1.0,1.0) };
    triangles[1].c = { vertsCube[3], vec2(0.0,1.0) };
    triangles[1].n = vec3( 0.0, 0.0, 1.0 );

    triangles[2].a = { vertsCube[5], vec2(0.0,0.0) };
    triangles[2].b = { vertsCube[4], vec2(1.0,0.0) };
    triangles[2].c = { vertsCube[6], vec2(1.0,1.0) };
    triangles[2].n = vec3( 1.0, 0.0, 0.0 );
    triangles[3].a = { vertsCube[5], vec2(0.0,0.0) };
    triangles[3].b = { vertsCube[6], vec2(1.0,1.0) };
    triangles[3].c = { vertsCube[7], vec2(0.0,1.0) };
    triangles[3].n = vec3( 1.0, 0.0, 0.0 );

    triangles[4].a = { vertsCube[3], vec2(0.0,0.0) };
    triangles[4].b = { vertsCube[7], vec2(1.0,0.0) };
    triangles[4].c = { vertsCube[6], vec2(1.0,1.0) };
    triangles[4].n = vec3( 0.0, 1.0, 0.0 );
    triangles[5].a = { vertsCube[3], vec2(0.0,0.0) };
    triangles[5].b = { vertsCube[6], vec2(1.0,1.0) };
    triangles[5].c = { vertsCube[2], vec2(0.0,1.0) };
    triangles[5].n = vec3( 0.0, 1.0, 0.0 );

    triangles[6].a = { vertsCube[0], vec2(1.0,0.0) };
    triangles[6].b = { vertsCube[6], vec2(0.0,1.0) };
    triangles[6].c = { vertsCube[4], vec2(0.0,0.0) };
    triangles[6].n = vec3( 0.0, 0.0, -1.0 );
    triangles[7].a = { vertsCube[0], vec2(1.0,0.0) };
    triangles[7].b = { vertsCube[2], vec2(1.0,1.0) };
    triangles[7].c = { vertsCube[6], vec2(0.0,1.0) };
    triangles[7].n = vec3( 0.0, 0.0, -1.0 );

    triangles[8].a = { vertsCube[1], vec2(1.0,0.0) };
    triangles[8].b = { vertsCube[2], vec2(0.0,1.0) };
    triangles[8].c = { vertsCube[0], vec2(0.0,0.0) };
    triangles[8].n = vec3( -1.0, 0.0, 0.0 );
    triangles[9].a = { vertsCube[1], vec2(1.0,0.0) };
    triangles[9].b = { vertsCube[3], vec2(1.0,1.0) };
    triangles[9].c = { vertsCube[2], vec2(0.0,1.0) };
    triangles[9].n = vec3( -1.0, 0.0, 0.0 );

    triangles[10].a = { vertsCube[1], vec2(0.0,0.0) };
    triangles[10].b = { vertsCube[0], vec2(0.0,1.0) };
    triangles[10].c = { vertsCube[4], vec2(1.0,1.0) };
    triangles[10].n = vec3( 0.0, -1.0, 0.0 );
    triangles[11].a = { vertsCube[1], vec2(0.0,0.0) };
    triangles[11].b = { vertsCube[4], vec2(1.0,1.0) };
    triangles[11].c = { vertsCube[5], vec2(1.0,0.0) };
    triangles[11].n = vec3( 0.0, -1.0, 0.0 );
}

void createFloor( inout Array<Triangle, 14> triangles )
{
    triangles[12].a = { vertsFloor[0], vec2(0.0,0.0) };
    triangles[12].b = { vertsFloor[1], vec2(0.0,3.0) };
    triangles[12].c = { vertsFloor[3], vec2(3.0,0.0) };
    triangles[12].n = vec3( 0.0, 1.0, 0.0 );
    triangles[13].a = { vertsFloor[1], vec2(0.0,3.0) };
    triangles[13].b = { vertsFloor[2], vec2(3.0,3.0) };
    triangles[13].c = { vertsFloor[3], vec2(3.0,0.0) };
    triangles[13].n = vec3( 0.0, 1.0, 0.0 );
}

float cross2d( in vec2 a, in vec2 b )
{
    return a.x*b.y - a.y*b.x;
}

const static vec3 lig = normalize( vec3( 0.3,0.7,0.5) );

vec3 pixelShader( in vec3 nor, in vec2 uv, in float z, in vec3 wnor )
{
    float dif = clamp( dot( nor, lig ), 0.0, 1.0 );
    float brdf = 0.5 + 0.8*dif;
    brdf *= 6.0*exp2( -0.7*abs(z) );
            
    vec3 mate = channel0.SampleLevel( bilinear_repeat, uv, 0 ).xyz;
    
    vec3 col = brdf * mate;    
    
    return sqrt( col );
}

void mainImageCube( out vec4 fragColor, in vec2 fragCoord )
{
    mat4 mdv = setTranslation( 0.0, 0.5, -5.0 ) * 
               setRotation( 0.6, 0.0, 0.0 ) * 
               setRotation( 0.0, 0.5 * iTime, 0.0 );

    vec2 px = (2.0*fragCoord-iResolution.xy) / iResolution.y;

    Array<Triangle, 14> triangles;

    createCube( triangles );
    createFloor( triangles );

    vec3 color = vec3( 0.1, 0.2, 0.3 );

    // clear zbuffer
    float mindist = -1000000.0;

    // render triangles
    for( int i=0; i<14; i++ )
    {
        // transform to eye space
        vec3 ep0 = (mdv * vec4(triangles[i].a.pos,1.0)).xyz;
        vec3 ep1 = (mdv * vec4(triangles[i].b.pos,1.0)).xyz;
        vec3 ep2 = (mdv * vec4(triangles[i].c.pos,1.0)).xyz;
        vec3 nor = (mdv * vec4(triangles[i].n,0.0)).xyz;

        // transform to clip space
        float w0 = 1.0/ep0.z;
        float w1 = 1.0/ep1.z;
        float w2 = 1.0/ep2.z;

        vec2 cp0 = 2.0*ep0.xy * -w0;
        vec2 cp1 = 2.0*ep1.xy * -w1;
        vec2 cp2 = 2.0*ep2.xy * -w2;

        // fetch vertex attributes, and divide by z
        vec2 u0 = triangles[i].a.uv * w0;
        vec2 u1 = triangles[i].b.uv * w1;
        vec2 u2 = triangles[i].c.uv * w2;

        //-----------------------------------
        // rasterize
        //-----------------------------------

        // calculate areas for subtriangles
        vec3 di = vec3( cross2d( cp1 - cp0, px - cp0 ), 
                        cross2d( cp2 - cp1, px - cp1 ), 
                        cross2d( cp0 - cp2, px - cp2 ) );
        
        // if all positive, point is inside triangle
        if( all(greaterThan(di,vec3(0.0))) )
        {
            // calc barycentric coordinates
            vec3 ba = di.yzx / (di.x+di.y+di.z);

            // barycentric interpolation of attributes and 1/z
            float iz = ba.x*w0 + ba.y*w1 + ba.z*w2;
            vec2  uv = ba.x*u0 + ba.y*u1 + ba.z*u2;

            // recover interpolated attributes
            float z = 1.0/iz;
            uv *= z;

            // depth (-1/z) buffer test
            if( z>mindist )
            {
                mindist = z;

                // perform lighting/shading
                color = pixelShader( nor, uv, z, triangles[i].n );
            }
        }
    }

    fragColor = vec4(color,1.0);
}

struct Point2 {
    vec3 pos;
    vec3 col;
};

struct Triangle2
{
    Point2 a;
    Point2 b;
    Point2 c;
};

void mainImageTriangle( out vec4 fragColor, in vec2 fragCoord, in bool enableBackFaceCulling )
{
    mat4 mdv = setTranslation( 0.0, 0.5, -5.0 ) * 
               setRotation( 0.6, 0.0, 0.0 ) * 
               setRotation( 0.0, 0.5 * iTime, 0.0 );

    vec2 px = (2.0*fragCoord-iResolution.xy) / iResolution.y;

    Triangle2 single_triangle;

    single_triangle.a = { vec3( 0.0,  1.0,0.0), vec3(1.0,0.0,0.0) };
    single_triangle.b = { vec3(-1.0, -1.0,0.0), vec3(0.0,1.0,0.0) };
    single_triangle.c = { vec3( 1.0, -1.0,0.0), vec3(0.0,0.0,1.0) };

    vec3 color = vec3( 0.1, 0.2, 0.3 );

    // clear zbuffer
    float mindist = -1000000.0;

    // render triangle
    {
        // transform to eye space
        vec3 ep0 = (mdv * vec4(single_triangle.a.pos,1.0)).xyz;
        vec3 ep1 = (mdv * vec4(single_triangle.b.pos,1.0)).xyz;
        vec3 ep2 = (mdv * vec4(single_triangle.c.pos,1.0)).xyz;

        // transform to clip space
        float w0 = 1.0/ep0.z;
        float w1 = 1.0/ep1.z;
        float w2 = 1.0/ep2.z;

        vec2 cp0 = 2.0*ep0.xy * -w0;
        vec2 cp1 = 2.0*ep1.xy * -w1;
        vec2 cp2 = 2.0*ep2.xy * -w2;

        // fetch vertex attributes, and divide by z
        vec3 col0 = single_triangle.a.col * w0;
        vec3 col1 = single_triangle.b.col * w1;
        vec3 col2 = single_triangle.c.col * w2;

        //-----------------------------------
        // rasterize
        //-----------------------------------

        // calculate areas for subtriangles
        vec3 di = vec3( cross2d( cp1 - cp0, px - cp0 ), 
                        cross2d( cp2 - cp1, px - cp1 ), 
                        cross2d( cp0 - cp2, px - cp2 ) );
        
        // if all positive, point is inside triangle
        bool can_seen = all(greaterThan(di,vec3(0.0)));
        if (!enableBackFaceCulling)
            can_seen = can_seen || all(lessThan(di,vec3(0.0)));
        if( can_seen  )
        {
            // calc barycentric coordinates
            vec3 ba = di.yzx / (di.x+di.y+di.z);

            // barycentric interpolation of attributes and 1/z
            float iz  = ba.x*w0   + ba.y*w1   + ba.z*w2;
            vec3  col = ba.x*col0 + ba.y*col1 + ba.z*col2;

            // recover interpolated attributes
            float z = 1.0/iz;
            col *= z;

            // depth (-1/z) buffer test
            if( z>mindist )
            {
                mindist = z;

                // perform shading
                color = col;
            }
        }
    }

    fragColor = vec4(color,1.0);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main_image(uint3 id : SV_DispatchThreadID)
{
    uint width, height;
    screen.GetDimensions(width, height);

    if (id.x >= width || id.y >= height)
        return;

    float2 fragCoord = float2(float(id.x) + 0.5, float(height - id.y) - 0.5);
    float2 uv = fragCoord / float2(width, height);


    float4 color;
    mainImageCube(color, fragCoord);
    //mainImageTriangle(color, fragCoord, false);

    screen[id.xy] = color;
}
