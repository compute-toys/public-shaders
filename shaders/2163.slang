#define PI 3.1415926535
#define HALF_PI (PI / 2.0)
#define FAR_DISTANCE 1000000.0

#define MAX_DEPTH 8
#define SPHERE_COUNT 3
#define BOX_COUNT 8
#define N_IN 0.99
#define N_OUT 1.0

import std;

static const int SIZE = 1024;
static const int ELEMENTS = SIZE*SIZE;
[StorageBuffer(ELEMENTS)]
RWStructuredBuffer<float4> image;
[StorageBuffer(ELEMENTS)]
RWStructuredBuffer<int> totalSamples;

uint linearIndex(uint2 id) {
    return id.x + id.y * SIZE;
}

struct Material
{
    float3 emmitance;
    float3 reflectance;
    float roughness;
    float opacity;
};

struct Box
{
    Material material;
    float3 halfSize;
    float3x3 rotation;
    float3 position;
};

struct Sphere
{
    Material material;
    float3 position;
    float radius;
};

static Array<Sphere, 3> spheres;
static Array<Box, 8>    boxes;

void InitializeScene()
{
    spheres[0].position = float3(2.5, 1.5, -1.5);
    spheres[1].position = float3(-2.5, 2.5, -1.0);
    spheres[2].position = float3(0.5, -4.0, 3.0);
    spheres[0].radius = 1.5;
    spheres[1].radius = 1.0;
    spheres[2].radius = 1.0;
    spheres[0].material.roughness = 1.0;
    spheres[1].material.roughness = 0.8;
    spheres[2].material.roughness = 1.0;
    spheres[0].material.opacity = 0.0;
    spheres[1].material.opacity = 0.0;
    spheres[2].material.opacity = 0.8;
    spheres[0].material.reflectance = float3(1.0, 0.0, 0.0);
    spheres[1].material.reflectance = float3(1.0, 0.4, 0.0);
    spheres[2].material.reflectance = float3(1.0, 1.0, 1.0);
    spheres[0].material.emmitance = float3(0.0, 0.0, 0.0);
    spheres[1].material.emmitance = float3(0.0, 0.0, 0.0);
    spheres[2].material.emmitance = float3(0.0, 0.0, 0.0);

    // up
    boxes[0].material.roughness = 0.0;
    boxes[0].material.emmitance = float3(0.0, 0.0, 0.0);
    boxes[0].material.reflectance = float3(1.0, 1.0, 1.0);
    boxes[0].halfSize = float3(5.0, 0.5, 5.0);
    boxes[0].position = float3(0.0, 5.5, 0.0);
    boxes[0].rotation = float3x3(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
    );

    // down
    boxes[1].material.roughness = 0.3;
    boxes[1].material.opacity = 0.0;
    boxes[1].material.emmitance = float3(0.0, 0.0, 0.0);
    boxes[1].material.reflectance = float3(1.0, 1.0, 1.0);
    boxes[1].halfSize = float3(5.0, 0.5, 5.0);
    boxes[1].position = float3(0.0, -5.5, 0.0);
    boxes[1].rotation = float3x3(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
    );

    // right
    boxes[2].material.roughness = 0.0;
    boxes[2].material.opacity = 0.0;
    boxes[2].material.emmitance = float3(0.0, 0.0, 0.0);
    boxes[2].material.reflectance = float3(0.0, 1.0, 0.0);
    boxes[2].halfSize = float3(5.0, 0.5, 5.0);
    boxes[2].position = float3(5.5, 0.0, 0.0);
    boxes[2].rotation = float3x3(
            0.0, 1.0, 0.0,
            -1.0, 0.0, 0.0,
            0.0, 0.0, 1.0
    );

    // left
    boxes[3].material.roughness = 0.0;
    boxes[3].material.opacity = 0.0;
    boxes[3].material.emmitance = float3(0.0, 0.0, 0.0);
    boxes[3].material.reflectance = float3(1.0, 0.0, 0.0);
    boxes[3].halfSize = float3(5.0, 0.5, 5.0);
    boxes[3].position = float3(-5.5, 0.0, 0.0);
    boxes[3].rotation = float3x3(
            0.0, 1.0, 0.0,
            -1.0, 0.0, 0.0,
            0.0, 0.0, 1.0
    );

    // back
    boxes[4].material.roughness = 0.0;
    boxes[4].material.opacity = 0.0;
    boxes[4].material.emmitance = float3(0.0, 0.0, 0.0);
    boxes[4].material.reflectance = float3(1.0, 1.0, 1.0);
    boxes[4].halfSize = float3(5.0, 0.5, 5.0);
    boxes[4].position = float3(0.0, 0.0, -5.5);
    boxes[4].rotation = float3x3(
            1.0, 0.0, 0.0,
            0.0, 0.0, 1.0,
            0.0, 1.0, 0.0
    );

    // light source
    boxes[5].material.roughness = 0.0;
    boxes[5].material.opacity = 0.0;
    boxes[5].material.emmitance = float3(6.0, 6.0, 6.0);
    boxes[5].material.reflectance = float3(1.0, 1.0, 1.0);
    boxes[5].halfSize = float3(2.5, 0.2, 2.5);
    boxes[5].position = float3(0.0, 4.8, 0.0);
    boxes[5].rotation = float3x3(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
    );

    // boxes
    boxes[6].material.roughness = 0.0;
    boxes[6].material.opacity = 0.0;
    boxes[6].material.emmitance = float3(0.0, 0.0, 0.0);
    boxes[6].material.reflectance = float3(1.0, 1.0, 1.0);
    boxes[6].halfSize = float3(1.5, 3.0, 1.5);
    boxes[6].position = float3(-2.0, -2.0, -0.0);
    boxes[6].rotation = float3x3(
            0.7, 0.0, 0.7,
            0.0, 1.0, 0.0,
            -0.7, 0.0, 0.7
    );
    // boxes
    boxes[7].material.roughness = 0.0;
    boxes[7].material.opacity = 0.0;
    boxes[7].material.emmitance = float3(0.0, 0.0, 0.0);
    boxes[7].material.reflectance = float3(1.0, 1.0, 1.0);
    boxes[7].halfSize = float3(1.0, 1.5, 1.0);
    boxes[7].position = float3(2.5, -3.5, -0.0);
    boxes[7].rotation = float3x3(
            0.7, 0.0, 0.7,
            0.0, 1.0, 0.0,
            -0.7, 0.0, 0.7
    );
}

float RandomNoise(float2 co)
{
    co *= fract(time.elapsed * 12.343);
    return fract(sin(dot(co.xy, float2(12.9898, 78.233))) * 43758.5453);
}

float3 RandomHemispherePoint(float2 rand)
{
    float cosTheta = sqrt(1.0 - rand.x);
    float sinTheta = sqrt(rand.x);
    float phi = 2.0 * PI * rand.y;
    return float3(
            cos(phi) * sinTheta,
            sin(phi) * sinTheta,
            cosTheta
    );
}

float3 NormalOrientedHemispherePoint(float2 rand, float3 n)
{
    float3 v = RandomHemispherePoint(rand);
    return dot(v, n) < 0.0 ? -v : v;
}

float FresnelSchlick(float nIn, float nOut, float3 direction, float3 normal)
{
    float R0 = ((nOut - nIn) * (nOut - nIn)) / ((nOut + nIn) * (nOut + nIn));
    float fresnel = R0 + (1.0 - R0) * pow((1.0 - abs(dot(direction, normal))), 5.0);
    return fresnel;
}

float3 IdealRefract(float3 direction, float3 normal, float nIn, float nOut)
{
    bool fromOutside = dot(normal, direction) < 0.0;
    float ratio = fromOutside ? nOut / nIn : nIn / nOut;

    float3 refraction, reflection;

    refraction = fromOutside ? refract(direction, normal, ratio) : -refract(-direction, normal, ratio);
    reflection = reflect(direction, normal);

    // return refraction == float3(0.0, 0.0, 0.0) ? reflection : refraction;
    return select(refraction == float3(0.0, 0.0, 0.0), reflection, refraction);
}

bool IntersectRaySphere(float3 origin, float3 direction, Sphere sphere, out float fraction, out float3 normal)
{
    float3 L = origin - sphere.position;
    float a = dot(direction, direction);
    float b = 2.0 * dot(L, direction);
    float c = dot(L, L) - sphere.radius * sphere.radius;
    float D = b * b - 4 * a * c;

    if (D < 0.0) return false;

    float r1 = (-b - sqrt(D)) / (2.0 * a);
    float r2 = (-b + sqrt(D)) / (2.0 * a);

    if (r1 > 0.0)
        fraction = r1;
    else if (r2 > 0.0)
        fraction = r2;
    else
        return false;

    normal = normalize(direction * fraction + L);

    return true;
}

bool IntersectRayBox(float3 origin, float3 direction, Box box, out float fraction, out float3 normal)
{
    float3 rd = mul(direction, box.rotation);
    float3 ro = mul((origin - box.position), box.rotation);

    float3 m = float3(1.0, 1.0, 1.0) / rd;

    float3 s = float3((rd.x < 0.0) ? 1.0 : -1.0,
                      (rd.y < 0.0) ? 1.0 : -1.0,
                      (rd.z < 0.0) ? 1.0 : -1.0);
    float3 t1 = m * (-ro + s * box.halfSize);
    float3 t2 = m * (-ro - s * box.halfSize);

    float tN = max(max(t1.x, t1.y), t1.z);
    float tF = min(min(t2.x, t2.y), t2.z);

    if (tN > tF || tF < 0.0) return false;

    float3x3 txi = transpose(box.rotation);

    if (t1.x > t1.y && t1.x > t1.z)
        normal = txi[0] * s.x;
    else if (t1.y > t1.z)
        normal = txi[1] * s.y;
    else
        normal = txi[2] * s.z;

    fraction = tN;

    return true;
}

bool CastRay(float3 rayOrigin, float3 rayDirection, out float fraction, out float3 normal, out Material material)
{
    float minDistance = FAR_DISTANCE;

    for (int i = 0; i < SPHERE_COUNT; i++)
    {
        float F;
        float3 N;
        if (IntersectRaySphere(rayOrigin, rayDirection, spheres[i], F, N) && F < minDistance)
        {
            minDistance = F;
            normal = N;
            material = spheres[i].material;
        }
    }

    for (int i = 0; i < BOX_COUNT; i++)
    {
        float F;
        float3 N;
        if (IntersectRayBox(rayOrigin, rayDirection, boxes[i], F, N) && F < minDistance)
        {
            minDistance = F;
            normal = N;
            material = boxes[i].material;
        }
    }

    fraction = minDistance;
    return minDistance != FAR_DISTANCE;
}

bool IsRefracted(float rand, float3 direction, float3 normal, float opacity, float nIn, float nOut)
{
    float fresnel = FresnelSchlick(nIn, nOut, direction, normal);
    return opacity > rand && fresnel < rand;
}

float3 TracePath(float3 rayOrigin, float3 rayDirection, float seed, float2 TexCoord)
{
    float3 L = float3(0.0, 0.0, 0.0);
    float3 F = float3(1.0, 1.0, 1.0);
    for (int i = 0; i < MAX_DEPTH; i++)
    {
        float fraction;
        float3 normal;
        Material material;
        bool hit = CastRay(rayOrigin, rayDirection, fraction, normal, material);
        if (hit)
        {
            float3 newRayOrigin = rayOrigin + fraction * rayDirection;

            float2 rand = float2(RandomNoise(seed * TexCoord.xy), seed * RandomNoise(TexCoord.yx));
            float3 hemisphereDistributedDirection = NormalOrientedHemispherePoint(rand, normal);

            float3 randomVec = float3(
                    RandomNoise(sin(seed * TexCoord.xy)),
                    RandomNoise(cos(seed * TexCoord.xy)),
                    RandomNoise(sin(seed * TexCoord.yx))
            );
            randomVec = normalize(2.0 * randomVec - 1.0);

            float3 tangent = cross(randomVec, normal);
            float3 bitangent = cross(normal, tangent);
            float3x3 transform = float3x3(tangent, bitangent, normal);

            float3 newRayDirection = mul(hemisphereDistributedDirection, transform);

            float refractRand = RandomNoise(cos(seed * TexCoord.yx));
            bool refracted = IsRefracted(refractRand, rayDirection, normal, material.opacity, N_IN, N_OUT);
            if (refracted)
            {
                float3 idealRefraction = IdealRefract(rayDirection, normal, N_IN, N_OUT);
                newRayDirection = normalize(lerp(-newRayDirection, idealRefraction, material.roughness));
                newRayOrigin += normal * (dot(newRayDirection, normal) < 0.0 ? -0.8 : 0.8);
            }
            else
            {
                float3 idealReflection = reflect(rayDirection, normal);
                newRayDirection = normalize(lerp(newRayDirection, idealReflection, material.roughness));
                newRayOrigin += normal * 0.8;
            }

            rayDirection = newRayDirection;
            rayOrigin = newRayOrigin;

            L += F * material.emmitance;
            F *= material.reflectance;
        }
        else
        {
            F = float3(1.0, 1.0, 1.0);
        }
    }

    return L;
}

[DispatchOnce]
[shader("compute")]
[numthreads(16, 16, 1)]
void first_frame(uint3 id : SV_DispatchThreadID)
{
    uint width, height;
    screen.GetDimensions(width, height);

    if (id.x >= width || id.y >= height)
        return;

    int uSamples = max(10, int(custom.Samples));
    float3 uPosition = float3(0.0, 0.0, 30.0);

    float2 fragCoord = float2(float(id.x) + 0.5, float(height - id.y) - 0.5);
    float2 uv = fragCoord / float2(width, height);

    InitializeScene();

    const float fovVerticalSlope = 1.0 / 5.0;
    const float2 randomPixelCenter = float2(id.xy);
    const float2 screenUV          = float2((2.0 * randomPixelCenter.x - width)  / height,    //
                                           -(2.0 * randomPixelCenter.y - height) / height);
    float3 rayDirection = float3(fovVerticalSlope * screenUV.x, fovVerticalSlope * screenUV.y, -1.0);
    rayDirection = normalize(rayDirection);
    float3 direction = rayDirection;

    float3 totalColor = float3(0.0, 0.0, 0.0);
    for (int i = 0; i < uSamples; i++)
    {
        float seed = sin(float(i) * time.elapsed);
        float3 sampleColor = TracePath(uPosition, direction, seed, uv);
        totalColor += sampleColor;
    }

    float3 color = totalColor / float(uSamples);
    color = color / (color + float3(1.0, 1.0, 1.0));
    float tmp = 1.0 / 2.2;
    color = pow(color, float3(tmp, tmp, tmp));

    image[linearIndex(id.xy)] = float4(color, 1.0);
    totalSamples[linearIndex(id.xy)] = 0;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main_image(uint3 id : SV_DispatchThreadID)
{
    uint width, height;
    screen.GetDimensions(width, height);

    if (id.x >= width || id.y >= height)
        return;

    int uSamples = max(10, int(custom.Samples));
    float3 uPosition = float3(0.0, 0.0, 30.0);

    float2 fragCoord = float2(float(id.x) + 0.5, float(height - id.y) - 0.5);
    float2 uv = fragCoord / float2(width, height);

    InitializeScene();

    const float fovVerticalSlope = 1.0 / 5.0;
    const float2 randomPixelCenter = float2(id.xy) + float2(1.5) * RandomNoise(id.xy);
    const float2 screenUV          = float2((2.0 * randomPixelCenter.x - width)  / height,    //
                                           -(2.0 * randomPixelCenter.y - height) / height);
    float3 rayDirection = float3(fovVerticalSlope * screenUV.x, fovVerticalSlope * screenUV.y, -1.0);
    rayDirection = normalize(rayDirection);
    float3 direction = rayDirection;

    float3 totalColor = float3(0.0, 0.0, 0.0);
    for (int i = 0; i < uSamples; i++)
    {
        float seed = sin(float(i) * time.elapsed);
        float3 sampleColor = TracePath(uPosition, direction, seed, uv);
        totalColor += sampleColor;
    }

    float3 color = totalColor / float(uSamples);
    color = color / (color + float3(1.0, 1.0, 1.0));
    float tmp = 1.0 / 2.2;
    color = pow(color, float3(tmp, tmp, tmp));

    float4 total_color = image[linearIndex(id.xy)] * totalSamples[linearIndex(id.xy)] + float4(color, 1.0) * uSamples;
    total_color = total_color / float(totalSamples[linearIndex(id.xy)]+uSamples);

    screen[id.xy] = total_color;

    image[linearIndex(id.xy)] = total_color;
    totalSamples[linearIndex(id.xy)] += uSamples;
}
