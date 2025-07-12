import std;

enum Char : uint {
    END = 0u,
    NEWLINE = 10u,
    SPACE = 32u,
    EXCLAMATION = 33u,
    QUOTE2 = 34u,
    HASTAG = 35u,
    DOLLAR = 36u,
    MOD = 37u,
    AMP = 38u,
    QUOTE1 = 39u,
    OPEN_PAREN = 40u,
    CLOSE_PAREN = 41u,
    STAR = 42u,
    PLUS = 43u,
    COMMA = 44u,
    DASH = 45u,
    DOT = 46u,
    SLASH = 47u,
    D0 = 48u,
    D1 = 49u,
    D2 = 50u,
    D3 = 51u,
    D4 = 52u,
    D5 = 53u,
    D6 = 54u,
    D7 = 55u,
    D8 = 56u,
    D9 = 57u,
    COLON = 58u,
    SEMICOLON = 59u,
    LT = 60u,
    EQ = 61u,
    GT = 62u,
    QUESTION = 63u,
    AT = 64u,
    A = 65u,
    B = 66u,
    C = 67u,
    D = 68u,
    E = 69u,
    F = 70u,
    G = 71u,
    H = 72u,
    I = 73u,
    J = 74u,
    K = 75u,
    L = 76u,
    M = 77u,
    N = 78u,
    O = 79u,
    P = 80u,
    Q = 81u,
    R = 82u,
    S = 83u,
    T = 84u,
    U = 85u,
    V = 86u,
    W = 87u,
    X = 88u,
    Y = 89u,
    Z = 90u,
    OPEN_BRACK = 91u,
    BACKSLASH = 92u,
    CLOSE_BRACK = 93u,
    CARET = 94u,
    UNDERSCORE = 95u,
    BACKTICK = 96u,
    a = 97u,
    b = 98u,
    c = 99u,
    d = 100u,
    e = 101u,
    f = 102u,
    g = 103u,
    h = 104u,
    i = 105u,
    j = 106u,
    k = 107u,
    l = 108u,
    m = 109u,
    n = 110u,
    o = 111u,
    p = 112u,
    q = 113u,
    r = 114u,
    s = 115u,
    t = 116u,
    u = 117u,
    v = 118u,
    w = 119u,
    x = 120u,
    y = 121u,
    z = 122u,
    OPEN_CURLY = 123u,
    PIPE = 124u,
    CLOSE_CURLY = 125u,
    TILDE = 126u,
    SPLAT = ~0u
};

namespace Debug {
    static const float2 CHAR_SIZE = float2(20.0, 40.0);
    static const float4 CHAR_COLOR = float4(0.0, 0.0, 0.0, 1.0);
    static const uint str0[] = {Char::OPEN_PAREN, Char::SPLAT, Char::COMMA, Char::SPACE, Char::SPLAT, Char::CLOSE_PAREN, Char::NEWLINE, Char::END};
    static const uint str1[] = {Char::QUOTE2, Char::SPLAT, Char::SPACE, Char::SPLAT, Char::SPACE, Char::SPLAT, Char::QUOTE2, Char::NEWLINE, Char::END};
    static const uint str2[] = {Char::NEWLINE, Char::END};

    [StorageBuffer(16777216)] RWStructuredBuffer<uint> buffer;
    [StorageBuffer(1)] RWStructuredBuffer<Atomic<uint>> atomics;

    void clear() {
        atomics[0].store(0u, MemoryOrder::Relaxed);
    }

    //uint repr_len<T>(T val) { return 0u; }
    //void print_repr<T>(T val, inout uint buf_pos) {}

    uint repr_len(uint val) {
        var len = 1u;
        while (true) {
            val /= 10;
            if (val == 0) return len;
            len++;
        }
    }

    void print_repr(uint val, inout uint buf_pos) {
        let len = repr_len(val);
        buf_pos += len;
        for (var i = 1u; i <= len; i++) {
            buffer[buf_pos - i] = val % 10 + Char::D0;
            val /= 10;
        }
    }

    void printf_fmt_seg(uint fmt[], inout uint fmt_pos, inout uint buf_pos) {
        while (true) {
            let ch = fmt[fmt_pos];
            if (ch == Char::END) return;

            fmt_pos++;
            if (ch == Char::SPLAT) return;

            buffer[buf_pos++] = ch;
        }
    }

    void printf<each T>(uint fmt[], expand each T args) where T == uint {
        var total_len = 0u;
        for (var i = 0u; fmt[i] != Char::END; i++) {
            if (fmt[i] != Char::SPLAT) total_len++;
        }

        expand total_len += repr_len(each args);
        var buf_pos = atomics[0].add(total_len, MemoryOrder::AcquireRelease);

        var fmt_pos = 0u;
        expand (printf_fmt_seg(fmt, fmt_pos, buf_pos), print_repr(each args, buf_pos));
        printf_fmt_seg(fmt, fmt_pos, buf_pos);
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
[WorkgroupCount(1, 1, 1)]
void reset() {
    Debug::clear();
}

[shader("compute")]
[numthreads(16, 16, 1)]
void test(uint3 id : SV_DispatchThreadID) {
    uint2 size;
    screen.GetDimensions(size.x, size.y);
    if (any(id.xy >= size)) return;

    if (all(id.xy == uint2(10, 84))) {
        Debug::printf(Debug::str0, id.x, id.y);
    }

    if (all(id.xy == uint2(17, 1))) {
        Debug::printf(Debug::str0, id.x, id.y);
        Debug::printf(Debug::str2);
        Debug::printf(Debug::str1, ~0u, 45u, 1u);
    }

    if (all(id.xy == uint2(171, 2))) {
        Debug::printf(Debug::str0, id.x, id.y);
    }

    var uv = float2(id.xy) / float2(size);
    uv.y = 1.0 - uv.y;
    pass_out[id] = float4(uv, 0.0, 1.0);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void print(uint3 id : SV_DispatchThreadID) {
    uint2 size;
    screen.GetDimensions(size.x, size.y);
    if (any(id.xy >= size)) return;
    let buffer_len = Debug::atomics[0].load(MemoryOrder::Relaxed);
    screen[id.xy] = pass_in[id];

    let uv = float2(id.xy) / Debug::CHAR_SIZE;
    var char_index = 0u;
    uint ch;

    for (var row = 0u; row < uint(uv.y); row++) {
        do {
            if (char_index >= buffer_len) return;
            ch = Debug::buffer[char_index++];
        } while (ch != Char::NEWLINE);
    }

    for (var col = 0u; col < uint(uv.x); col++) {
        if (char_index >= buffer_len) return;
        ch = Debug::buffer[char_index++];
        if (ch == Char::NEWLINE) return;
    }

    if (char_index >= buffer_len) return;
    ch = Debug::buffer[char_index];
    if (ch == Char::NEWLINE) return;

    let char_offs = float2(ch % 16, ch / 16);
    let char_uv = (fract(uv) - 0.5) * float2(Debug::CHAR_SIZE.x / Debug::CHAR_SIZE.y, 1.0) + 0.5;
    if (char_uv.x >= 0.0 && char_uv.x < 1.0) {
        let value = channel0.SampleLevel(bilinear, (char_offs + char_uv) / 16.0, 0.0);
        screen[id.xy] = lerp(pass_in[id], Debug::CHAR_COLOR, smoothstep(0.0, 1.0 / Debug::CHAR_SIZE.y, 0.5 - value.w));
    }
}