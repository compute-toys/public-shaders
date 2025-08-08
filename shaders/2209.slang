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
    static const uint ATOMIC_CURSOR = 0u;
    static const uint2 TERMINAL_SIZE = uint2(32u, 32u);

    [StorageBuffer(1048576)] RWStructuredBuffer<uint> terminal;
    [StorageBuffer(1)] RWStructuredBuffer<Atomic<uint>> atomics;

    // WARNING: this is a critical section
    uint get_cursor() {
        return atomics[ATOMIC_CURSOR].load();
    }

    // WARNING: this is a critical section
    uint move_cursor(uint delta) {
        return atomics[ATOMIC_CURSOR].add(delta);
    }

    void reset() {
        let clear_count = min(TERMINAL_SIZE.x * TERMINAL_SIZE.y, get_cursor());
        for (var i = 0u; i < clear_count; i++) terminal[i] = 0u;
        atomics[ATOMIC_CURSOR].store(0u);
    }

    void preput_char(uint ch, inout uint cursor) {
        cursor = ch != Char::NEWLINE ? (cursor + 1) : (cursor - cursor % TERMINAL_SIZE.x + TERMINAL_SIZE.x);
    }

    void put_char(uint ch, inout uint cursor) {
        if (ch != Char::NEWLINE) terminal[cursor] = ch;
        cursor = ch != Char::NEWLINE ? (cursor + 1) : (cursor - cursor % TERMINAL_SIZE.x + TERMINAL_SIZE.x);
    }

    interface IPrintable {
        void preprint(inout uint cursor);
        void print(inout uint cursor);
    }

    struct Uint : IPrintable {
        uint value;
        uint digit_count;

        __init(uint value) {
            this.value = value;
            this.digit_count = 1u;
            while (true) {
                value /= 10;
                if (value == 0) break;
                this.digit_count++;
            }
        }

        void preprint(inout uint cursor) {
            cursor += this.digit_count;
        }

        void print(inout uint cursor) {
            cursor += this.digit_count;
            var val = this.value;
            for (var i = 1u; i <= this.digit_count; i++) {
                terminal[cursor - i] = val % 10 + Char::D0;
                val /= 10;
            }
        }
    }

    class stdout {
        private static void _preprint_helper(uint fmt[], inout uint fmtptr, inout uint cursor) {
            while (true) {
                let ch = fmt[fmtptr];
                if (ch == Char::END) return;

                fmtptr++;
                if (ch == Char::SPLAT) return;

                preput_char(ch, cursor);
            }
        }

        private static void _print_helper(uint fmt[], inout uint fmtptr, inout uint cursor) {
            while (true) {
                let ch = fmt[fmtptr];
                if (ch == Char::END) return;

                fmtptr++;
                if (ch == Char::SPLAT) return;

                put_char(ch, cursor);
            }
        }

        static void printf<each T : IPrintable>(uint fmt[], expand each T args) {
            var cursor_motion = 0u;
            var fmtptr = 0u;
            expand (
                _preprint_helper(fmt, fmtptr, cursor_motion),
                (each args).preprint(cursor_motion)
            );
            _preprint_helper(fmt, fmtptr, cursor_motion);
            preput_char(Char::NEWLINE, cursor_motion); // needed to break dependency on initial cursor position

            var cursor = move_cursor(cursor_motion);
            fmtptr = 0u;
            expand (
                _print_helper(fmt, fmtptr, cursor),
                (each args).print(cursor)
            );
            _print_helper(fmt, fmtptr, cursor);
        }
    }
}

static const uint str0[] = {Char::OPEN_PAREN, Char::SPLAT, Char::COMMA, Char::SPACE, Char::SPLAT, Char::CLOSE_PAREN, Char::END};
static const uint str1[] = {Char::QUOTE2, Char::SPLAT, Char::SPACE, Char::SPLAT, Char::SPACE, Char::SPLAT, Char::QUOTE2, Char::END};
static const uint str2[] = {Char::END};

[shader("compute")]
[numthreads(1, 1, 1)]
[WorkgroupCount(1, 1, 1)]
void reset() {
    Debug::reset();
}

[shader("compute")]
[numthreads(16, 16, 1)]
void test(uint3 id : SV_DispatchThreadID) {
    uint2 size;
    screen.GetDimensions(size.x, size.y);
    if (any(id.xy >= size)) return;

    typealias stdout = Debug::stdout;
    typealias Uint = Debug::Uint;

    if (all(id.xy == uint2(10, 84))) {
        stdout.printf(str0, Uint(id.x), Uint(id.y));
    }

    if (all(id.xy == uint2(17, 1))) {
        stdout.printf(str0, Uint(id.x), Uint(id.y));
        stdout.printf(str2);
        stdout.printf(str1, Uint(~0), Uint(45), Uint(1));
    }

    if (all(id.xy == uint2(171, 2))) {
        stdout.printf(str0, Uint(id.x), Uint(id.y));
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
    screen[id.xy] = pass_in[id];

    let uv = float2(id.xy) / Debug::CHAR_SIZE;

    let grid_pos = uint2(floor(uv));
    if (any(grid_pos >= Debug::TERMINAL_SIZE)) return;

    let char_index = grid_pos.y * Debug::TERMINAL_SIZE.x + grid_pos.x;
    if (char_index >= Debug::get_cursor()) return;

    let ch = Debug::terminal[char_index];
    if (ch == 0) return;

    let char_offs = float2(ch % 16, ch / 16);
    let char_uv = (fract(uv) - 0.5) * float2(Debug::CHAR_SIZE.x / Debug::CHAR_SIZE.y, 1.0) + 0.5;
    if (char_uv.x >= 0.0 && char_uv.x < 1.0) {
        let value = channel0.SampleLevel(bilinear, (char_offs + char_uv) / 16.0, 0.0);
        screen[id.xy] = lerp(pass_in[id], Debug::CHAR_COLOR, smoothstep(0.0, 1.0 / Debug::CHAR_SIZE.y, 0.5 - value.w));
    }
}