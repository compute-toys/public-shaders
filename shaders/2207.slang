import std;

static const float LEARNING_RATE = 0.1;
static const uint EPOCHS_PER_FRAME = 1;
static const int DATA_POINT_RADIUS = 10;
static const uint EXAMPLE_COUNT = 32;
static const float3 ID0 = {0.0, 0.0, 1.0};
static const float3 ID1 = {0.5, 0.0, 0.5};
static const float3 ID2 = {0.2, 0.8, 0.2};
static const float3 ID3 = {1.0, 0.5, 0.0};
static const Tuple<float2, float3> EXAMPLES[] = {
    {{0.181259, 0.600128}, ID0},
    {{0.209161, 0.686168}, ID0},
    {{0.272544, 0.760272}, ID0},
    {{0.210224, 0.875845}, ID0},
    {{0.045871, 0.691806}, ID0},
    {{0.107407, 0.879798}, ID0},
    {{0.096386, 0.464752}, ID0},
    {{0.705028, 0.667955}, ID1},
    {{0.561318, 0.717127}, ID1},
    {{0.551332, 0.755783}, ID1},
    {{0.642029, 0.859787}, ID1},
    {{0.578756, 0.661293}, ID1},
    {{0.625287, 0.561804}, ID1},
    {{0.993646, 0.878986}, ID1},
    {{0.905712, 0.376894}, ID2},
    {{0.828445, 0.289356}, ID2},
    {{0.900651, 0.442489}, ID2},
    {{0.979855, 0.145529}, ID2},
    {{0.535972, 0.125601}, ID3},
    {{0.535338, 0.286151}, ID3},
    {{0.422854, 0.193598}, ID3},
    {{0.596321, 0.269657}, ID3},
    {{0.616100, 0.290497}, ID3},
    {{0.540227, 0.336999}, ID3},
    {{0.632601, 0.092070}, ID3},
    {{0.386532, 0.083337}, ID3},
    {{0.707578, 0.041996}, ID3},
    {{0.308483, 0.217245}, ID3},
    {{0.380038, 0.107003}, ID3},
    {{0.325411, 0.257011}, ID3},
    {{0.592546, 0.425580}, ID3},
    {{0.415377, 0.017189}, ID3}
};

// Based on lowbias32 from https://nullprogram.com/blog/2018/07/31/
float rand01(inout uint seed) {
    seed ^= seed >> 16;
    seed *= 0x7FEB352Du;
    seed ^= seed >> 15;
    seed *= 0x846CA68Bu;
    seed ^= seed >> 16;
    return float(seed) / 4294967296.0;
}

vector<float, N> rand01<int N>(inout uint seed) {
    vector<float, N> result;
    for (var i = 0; i < N; i++) {
        result[i] = rand01(seed);
    }

    return result;
}

matrix<float, R, C> rand01<int R, int C>(inout uint seed) {
    matrix<float, R, C> result;
    for (var i = 0; i < R; i++) {
        for (var j = 0; j < C; j++) {
            result[i][j] = rand01(seed);
        }
    }

    return result;
}

struct Activation {
    [Differentiable]
    static float sigmoid(float x) {
        return 1.0 / (1.0 + exp(-x));
    }

    [Differentiable]
    static vector<float, N> sigmoid<int N>(vector<float, N> x) {
        return 1.0 / (1.0 + exp(-x));
    }
};

struct NNParams : IDifferentiable {
    float4x2 w0 = {}; float4 b0 = {};
    float4x4 w1 = {}; float4 b1 = {};
    float3x4 w2 = {}; float3 b2 = {};

    static NNParams new(uint seed = 332243498u) {
        return NNParams(
            (rand01<4, 2>(seed) * 2.0 - 1.0) * sqrt(2.0 / 2.0), {},
            (rand01<4, 4>(seed) * 2.0 - 1.0) * sqrt(2.0 / 4.0), {},
            (rand01<3, 4>(seed) * 2.0 - 1.0) * sqrt(2.0 / 4.0), {}
        );
    }

    [Differentiable]
    float3 eval(no_diff float2 x) {
        let x1 = Activation::sigmoid(mul(w0, x ) + b0);
        let x2 = Activation::sigmoid(mul(w1, x1) + b1);
        let x3 = Activation::sigmoid(mul(w2, x2) + b2);
        return x3;
    }

    [Differentiable]
    static float cost(NNParams nn, no_diff float2 x, no_diff float3 y) {
        let err = y - nn.eval(x);
        return dot(err, err);
    }
};

NNParams operator+(NNParams lhs, NNParams rhs) {
    return NNParams(
        lhs.w0 + rhs.w0, lhs.b0 + rhs.b0,
        lhs.w1 + rhs.w1, lhs.b1 + rhs.b1,
        lhs.w2 + rhs.w2, lhs.b2 + rhs.b2
    );
}

NNParams operator*(NNParams lhs, float rhs) {
    return NNParams(
        lhs.w0 * rhs, lhs.b0 * rhs,
        lhs.w1 * rhs, lhs.b1 * rhs,
        lhs.w2 * rhs, lhs.b2 * rhs
    );
}

[StorageBuffer(1)] RWStructuredBuffer<NNParams> params;

[shader("compute")]
[numthreads(1, 1, 1)]
[WorkgroupCount(1, 1, 1)]
//[DispatchOnce] // Bugged
void init_training() {
    if (time.frame == 0) {
        params[0] = NNParams::new(reinterpret<uint>(time.elapsed) ^ 1357880991u);
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
[WorkgroupCount(1, 1, 1)]
[DispatchCount(EPOCHS_PER_FRAME)]
void training_round() {
    NNParams accum = {};
    for (var i = 0; i < EXAMPLES.getCount(); i++) {
        var grad = diffPair(params[0]);
        bwd_diff(NNParams::cost)(grad, EXAMPLES[i]._0, EXAMPLES[i]._1, 1.0);
        accum = accum + grad.d;
    }

    accum = accum * (-LEARNING_RATE / float(EXAMPLES.getCount()));
    params[0] = params[0] + accum;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void show_output(uint3 id : SV_DispatchThreadID) {
    uint2 size;
    screen.GetDimensions(size.x, size.y);
    if (any(id.xy >= size)) { return; }
    screen[id.xy] = float4(params[0].eval(float2(id.xy) / float2(size)), 1.0);
}

[shader("compute")]
[numthreads(EXAMPLE_COUNT, 1, 1)]
[WorkgroupCount(1, 1, 1)]
void splat_data_points(uint3 id : SV_DispatchThreadID) {
    if (id.x >= EXAMPLE_COUNT) return;

    uint2 size;
    screen.GetDimensions(size.x, size.y);

    let pos = int2(EXAMPLES[id.x]._0 * float2(size));
    let color = EXAMPLES[id.x]._1;

    int2 offs;
    for (offs.y = -DATA_POINT_RADIUS; offs.y < DATA_POINT_RADIUS; offs.y++) {
        for (offs.x = -DATA_POINT_RADIUS; offs.x < DATA_POINT_RADIUS; offs.x++) {
            let uv = float2(offs) / float(DATA_POINT_RADIUS);
            let square_dist = dot(uv, uv);
            if (square_dist < 1.0) {
                screen[pos + offs] = float4(color * (1.0 - square_dist), 1.0);
            }
        }
    }
}