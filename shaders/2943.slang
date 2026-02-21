import std;

static const float FLT_MAX = 3.402823466e+38;
static const float SQRT_2 = 1.4142135623730951l;

static const float sigma = 1;

static const uint width  = 64;
static const uint height = 64;

static const float initCoverage   = 0.1;
static const uint2 gridSize       = uint2(round(float2(width, height) * sqrt(initCoverage)));
static const uint  initPixelCount = gridSize.x * gridSize.y;

[StorageBuffer(1)]
RWStructuredBuffer<uint> phase;

// used in phase 1 + 2
[StorageBuffer(width * height)]
RWStructuredBuffer<uint> tempMask;

[StorageBuffer(width * height)]
RWStructuredBuffer<uint> rank;

static const uint lutWidth  = width / 2 + 1;
static const uint lutHeight = height / 2 + 1;

[StorageBuffer(lutWidth * lutHeight)]
RWStructuredBuffer<float> gaussLUT;

[StorageBuffer(width * height)]
RWStructuredBuffer<float> energy;

// used in phase 2
[StorageBuffer(width * height)]
RWStructuredBuffer<float> tempEnergy;

[StorageBuffer(1)]
RWStructuredBuffer<uint> currentRank;

// lowest energy, highest energy
[StorageBuffer(2)]
RWStructuredBuffer<uint2> energyExtrema;

[StorageBuffer(2)]
RWStructuredBuffer<uint2> prevEnergyExtrema;

uint getIdx(uint2 pos) { return pos.x + pos.y * width; }
uint2 getPos(uint idx) { return uint2(idx % width, idx / width); }

uint2 pcg2d(uint2 v)
{
    v = v * 1664525u + 1013904223u;

    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;

    v = v ^ (v >> 16u);

    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;

    v = v ^ (v >> 16u);

    return v;
}

float erf(float x)
{
    float t = 1 / (1 + 0.5 * abs(x));
    return sign(x) * (1 - t * exp(
        -x*x-1.26551223
        +t*(+1.00002368
        +t*(+0.37409196
        +t*(+0.09678418
        +t*(-0.18628806
        +t*(+0.27886807
        +t*(-1.13520398
        +t*(+1.48851587
        +t*(-0.82215223
        +t*(+0.17087277)))))))))));
}

float filteredGaussian(int2 pos)
{
    float den = sigma * SQRT_2;
    float2 p = float2(pos);
    return 0.25 * sigma * (erf((p.x + 0.5) / den) - erf((p.x - 0.5) / den))
                        * (erf((p.y + 0.5) / den) - erf((p.y - 0.5) / den));
}

[shader("compute")]
[DispatchOnce]
[numthreads(16, 16, 1)]
[WorkgroupCount((lutWidth + 15) / 16, (lutHeight + 15) / 16, 1)]
void initLUT(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= lutWidth || id.y >= lutHeight) return;
    float res = 0;

    int R = 4;

    for (int i = -R; i <= R; i++)
    {
        for (int j = -R; j <= R; j++)
        {
            res += filteredGaussian(int2(int(id.x) + int(width)  * i,
                                         int(id.y) + int(height) * j));
        }
    }

    gaussLUT[id.x + id.y * lutWidth] = res;
}

float kernel(uint2 pos, uint2 mean)
{
    uint2 p = uint2(abs(int2(pos) - int2(mean)));
          p = min(p, uint2(width, height) - p);
    
    return gaussLUT[p.x + p.y * lutWidth];
}

[shader("compute")]
[DispatchOnce]
[numthreads(16, 16, 1)]
[WorkgroupCount((width + 15) / 16, (height + 15) / 16, 1)]
void init(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= width || id.y >= height) return;
    
    uint2 seed = uint2(69);
    
    rank[getIdx(id.xy)] = 0;
    tempMask[getIdx(id.xy)] = 0;
    energy[getIdx(id.xy)] = 0;

    for (uint i = 0; i < gridSize.x; i++)
    {
        for (uint j = 0; j < gridSize.y; j++)
        {
            seed = pcg2d(seed);
            uint2 bound1 = uint2(round(float2(i,     j    ) * float2(width, height) / float2(gridSize)));
            uint2 bound2 = uint2(round(float2(i + 1, j + 1) * float2(width, height) / float2(gridSize)));
            uint2 pos = bound1 + (seed % (bound2 - bound1));
            if (all(id.xy == pos)) tempMask[getIdx(id.xy)] = 1;
            energy[getIdx(id.xy)] += kernel(id.xy, pos);
        }
    }
    
    if (all(id.xy == uint2(0))) phase[0] = 1;
}

[shader("compute")]
[numthreads(1, 1, 1)]
[WorkgroupCount(1, 1, 1)]
void findPixel()
{
    if (currentRank[0] == width * height) return;

    prevEnergyExtrema[0] = energyExtrema[0];
    prevEnergyExtrema[1] = energyExtrema[1];
    
    float best  = FLT_MAX;
    float worst = 0;

    for (uint i = 0; i < width; i++)
    {
        for (uint j = 0; j < height; j++)
        {
            uint2 pos = uint2(i, j);
            uint idx = getIdx(pos);
            float temp = phase[0] == 2 ? tempEnergy[idx] : energy[idx];
            
            uint mask = phase[0] < 3 ? tempMask[idx] : rank[idx];

            if (mask > 0)
            {
                if (temp > worst)
                {
                    worst = temp;
                    energyExtrema[1] = pos;
                }
            }
            else
            {
                if (temp < best)
                {
                    best = temp;
                    energyExtrema[0] = pos;
                }
            }
        }
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
[WorkgroupCount((width + 15) / 16, (height + 15) / 16, 1)]
void update(uint3 id : SV_DispatchThreadID)
{
    if (currentRank[0] == width * height) return;
    if (id.x >= width || id.y >= height) return;

    if (phase[0] == 1 && all(id.xy == uint2(0)) &&
        all(energyExtrema[0] == prevEnergyExtrema[1]) &&
        all(energyExtrema[1] == prevEnergyExtrema[0]))
    {
        phase[0] = 2;
        currentRank[0] = initPixelCount;
    }

    uint2 pos = id.xy;
    uint  idx = getIdx(pos);

    uint2 lowPos  = energyExtrema[0];
    uint2 highPos = energyExtrema[1];

    bool isLow  = all(pos == lowPos);
    bool isHigh = all(pos == highPos);

    if (isLow)
    {
        if (phase[0] == 1)
            tempMask[idx] = 1;

        if (phase[0] == 3)
        {
            currentRank[0]++;
            rank[idx] = currentRank[0];
        }
    }

    if (phase[0] != 2)
        energy[idx] += kernel(pos, lowPos);

    if (phase[0] < 3 && isHigh)
        tempMask[idx] = 0;

    if (phase[0] == 1)
    {
        energy[idx] -= kernel(pos, highPos);
        tempEnergy[idx] = energy[idx];
    }

    if (phase[0] == 2)
    {
        if (isHigh)
        {
            rank[idx] = currentRank[0];
            currentRank[0]--;
        }

        tempEnergy[idx] -= kernel(pos, highPos);

        if (currentRank[0] == 0)
        {
            phase[0] = 3;
            currentRank[0] = initPixelCount;
        }
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main_image(uint3 id : SV_DispatchThreadID)
{
    uint w, h;
    screen.GetDimensions(w, h);

    if (id.x >= w || id.y >= h) return;

    float2 uv = (float2(id.xy) + .5) / float(h);

    float3 col = float3(0);

    uint2 p = uint2(uv * float(height));

    if (p.x < width)
    {
        col = phase[0] == 1 ? float3(tempMask[getIdx(p)])
            : phase[0] == 2 ? float3(rank[getIdx(p)]) / float(initPixelCount)
                            : float3(rank[getIdx(p)]) / float(currentRank[0]);
    }

    screen[id.xy] = float4(col, 1);
}
