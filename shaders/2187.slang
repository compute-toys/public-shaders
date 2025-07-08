import std;

static const int SIZE = 1024;
static const int ELEMENTS = SIZE*SIZE;
[StorageBuffer(ELEMENTS)]
RWStructuredBuffer<int> firePixels;

uint linearIndex(uint2 id) {
    return id.x + id.y * SIZE;
}

static const float3 fireColor[37] = {
  float3(0.027451,0.027451,0.027451),
  float3(0.121569,0.027451,0.027451),
  float3(0.184314,0.0588235,0.027451),
  float3(0.278431,0.0588235,0.027451),
  float3(0.341176,0.0901961,0.027451),
  float3(0.403922,0.121569,0.027451),
  float3(0.466667,0.121569,0.027451),
  float3(0.560784,0.152941,0.027451),
  float3(0.623529,0.184314,0.027451),
  float3(0.686275,0.247059,0.027451),
  float3(0.74902,0.278431,0.027451),
  float3(0.780392,0.278431,0.027451),
  float3(0.87451,0.309804,0.027451),
  float3(0.87451,0.341176,0.027451),
  float3(0.87451,0.341176,0.027451),
  float3(0.843137,0.372549,0.027451),
  float3(0.843137,0.372549,0.027451),
  float3(0.843137,0.403922,0.0588235),
  float3(0.811765,0.435294,0.0588235),
  float3(0.811765,0.466667,0.0588235),
  float3(0.811765,0.498039,0.0588235),
  float3(0.811765,0.529412,0.0901961),
  float3(0.780392,0.529412,0.0901961),
  float3(0.780392,0.560784,0.0901961),
  float3(0.780392,0.592157,0.121569),
  float3(0.74902,0.623529,0.121569),
  float3(0.74902,0.623529,0.121569),
  float3(0.74902,0.654902,0.152941),
  float3(0.74902,0.654902,0.152941),
  float3(0.74902,0.686275,0.184314),
  float3(0.717647,0.686275,0.184314),
  float3(0.717647,0.717647,0.184314),
  float3(0.717647,0.717647,0.215686),
  float3(0.811765,0.811765,0.435294),
  float3(0.87451,0.87451,0.623529),
  float3(0.937255,0.937255,0.780392),
  float3(1,1,1),
};

float RandomNoise(float2 co)
{
    co *= fract(time.elapsed * 12.343);
    return fract(sin(dot(co.xy, float2(12.9898, 78.233))) * 43758.5453);
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

    firePixels[linearIndex(id.xy)] = 0;

    if ( id.y == (height-1) )
        firePixels[linearIndex(id.xy)] = 36;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void spread_fire(uint3 id : SV_DispatchThreadID)
{
    uint width, height;
    screen.GetDimensions(width, height);

    if (id.x >= width || id.y >= height)
        return;

    if ( id.y < (height-1) )
    {
        uint2 dstId = id.xy;
        uint2 srcId = id.xy;
        srcId.y += 1;
        uint32_t rand = uint32_t(round(RandomNoise(float2(srcId)) * 3.0)) & 3;
        firePixels[linearIndex(dstId)] = firePixels[linearIndex(srcId)] - (rand & 1);
    }

    if (firePixels[linearIndex(id.xy)] < 0)
        firePixels[linearIndex(id.xy)] = 0;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main_image(uint3 id : SV_DispatchThreadID)
{
    uint width, height;
    screen.GetDimensions(width, height);

    if (id.x >= width || id.y >= height)
        return;

    screen[id.xy] = float4(fireColor[firePixels[linearIndex(id.xy)]], 1.0);
}
