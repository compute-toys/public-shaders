import std;

static const float LEARNING_RATE = 0.1;
static const uint EPOCHS_PER_FRAME = 1;
static const int DATA_POINT_RADIUS = 10;
static const uint EXAMPLE_COUNT = 32;
static const Tuple<float2, float> EXAMPLES[] = {
    {{0.488845, 0.367371}, {1.0}},
    {{0.313280, 0.261770}, {0.0}},
    {{0.895153, 0.780527}, {0.0}},
    {{0.515393, 0.652667}, {1.0}},
    {{0.927011, 0.752037}, {0.0}},
    {{0.910428, 0.570217}, {0.0}},
    {{0.507262, 0.693612}, {1.0}},
    {{0.813340, 0.279186}, {0.0}},
    {{0.345013, 0.213742}, {0.0}},
    {{0.733453, 0.149250}, {0.0}},
    {{0.215776, 0.508310}, {0.0}},
    {{0.922050, 0.773392}, {0.0}},
    {{0.889062, 0.352426}, {0.0}},
    {{0.507663, 0.332348}, {1.0}},
    {{0.901346, 0.423011}, {0.0}},
    {{0.683697, 0.493722}, {1.0}},
    {{0.928846, 0.994238}, {0.0}},
    {{0.492549, 0.502561}, {1.0}},
    {{0.741904, 0.368015}, {0.0}},
    {{0.663560, 0.587033}, {1.0}},
    {{0.943349, 0.426079}, {0.0}},
    {{0.895796, 0.678477}, {0.0}},
    {{0.733398, 0.601963}, {0.0}},
    {{0.120375, 0.058295}, {0.0}},
    {{0.262164, 0.756264}, {0.0}},
    {{0.366633, 0.104583}, {0.0}},
    {{0.133297, 0.155483}, {0.0}},
    {{0.420718, 0.660074}, {1.0}},
    {{0.305777, 0.098515}, {0.0}},
    {{0.620712, 0.649312}, {1.0}},
    {{0.419432, 0.579353}, {1.0}},
    {{0.448368, 0.760281}, {0.0}}
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
    float4   w2 = {}; float  b2 = {};

    static NNParams new(uint seed = 332243498u) {
        return NNParams(
            (rand01<4, 2>(seed) * 2.0 - 1.0) * sqrt(2.0 / 2.0), {},
            (rand01<4, 4>(seed) * 2.0 - 1.0) * sqrt(2.0 / 4.0), {},
            (rand01<   4>(seed) * 2.0 - 1.0) * sqrt(2.0 / 4.0), {}
        );
    }

    [Differentiable]
    float eval(no_diff float2 x) {
        let x1 = Activation::sigmoid(mul(w0, x ) + b0);
        let x2 = Activation::sigmoid(mul(w1, x1) + b1);
        let x3 = Activation::sigmoid(dot(w2, x2) + b2);
        return x3;
    }

    [Differentiable]
    static float cost(NNParams nn, no_diff float2 x, no_diff float y) {
        let err = y - nn.eval(x);
        return err * err;
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
        params[0] = NNParams::new(reinterpret<uint>(time.elapsed));
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
    let value = params[0].eval(float2(id.xy) / float2(size)) * 2.0 - 1.0;
    screen[id.xy] = float4(max(0.0, value), 0.0, max(-value, 0.0), 1.0);
}

[shader("compute")]
[numthreads(EXAMPLE_COUNT, 1, 1)]
void splat_data_points(uint3 id : SV_DispatchThreadID) {
    if (id.x >= EXAMPLE_COUNT) return;

    uint2 size;
    screen.GetDimensions(size.x, size.y);

    let pos = int2(EXAMPLES[id.x]._0 * float2(size));
    let cls = EXAMPLES[id.x]._1 >= 0.5;
    let color = cls ? float4(1.0, 0.8, 0.0, 1.0) : float4(0.0, 0.5, 1.0, 1.0);

    int2 offs;
    for (offs.y = -DATA_POINT_RADIUS; offs.y < DATA_POINT_RADIUS; offs.y++) {
        for (offs.x = -DATA_POINT_RADIUS; offs.x < DATA_POINT_RADIUS; offs.x++) {
            if (offs.x * offs.x + offs.y * offs.y < DATA_POINT_RADIUS * DATA_POINT_RADIUS) {
                screen[pos + offs] = color;
            }
        }
    }
}