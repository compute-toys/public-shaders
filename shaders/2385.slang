import std;

static const float LINE_WIDTH = 0.04;
static const float CELL_RADIUS = 0.3;
static const int FRAMES_PER_UPDATE = 40;
static const int STEPS_PER_UPDATE = 1;
static const int STEPS_UNTIL_RESET = 100;


float sdf_sphere(float2 p, float r) {
    return length(p) - r;
}

float sdf_line(float2 p, float2 a, float2 b) {
    float2 ab = b-a, ap = p-a;
    float2 on_line = saturate(dot(ab, ap) / dot(ab, ab)) * ab;
    return length(on_line - ap);
}

float sdf_box(float2 p, float2 b) {
    float2 d = abs(p)-b;
    return length(max(d,0.0)) + min(max(d.x,d.y),0.0);
}

float4 alpha_over(float4 behind, float4 above) {
    if (behind.a == 0)
        above.a = 1;
    float4 r = lerp(float4(behind.rgb, 1) * behind.a, float4(above.rgb, 1), above.a);
    r.rgb /= r.a;
    return r;
}

float2 sort2(float2 v) {
    return (v.x < v.y) ? v.xy : v.yx;
}

uint murmur3(inout uint x) {
    // Murmur3 finalize step as hashing
	x ^= x >> 16;
	x *= 0x85ebca6b;
	x ^= x >> 13;
	x *= 0xc2b2ae35;
	x ^= x >> 16;
    return x;
}

enum State {
    NONE     = 0,
    PLAYER_1 = 1,
    PLAYER_2 = 2,
    INVALID  = 3
}

float4 stateColor(State s) {
    switch (s) {
        default:
        case State.NONE:
            return float4(.8,.8,.8,1);
        case State.PLAYER_1:
            return float4(.8,.4,.4,1);
        case State.PLAYER_2:
            return float4(.4,.4,.8,1);
        case State.INVALID:
            return float4(1,.4,1,1); 
    }
}

struct Board {
    uint32_t mask_player_1;
    uint32_t mask_player_2;

    State getState(uint field_idx) {
        // Fields are numbered in clockwise fashion starting from 12 o'clock and the innermost ring.
        uint p1 = (mask_player_1 >> field_idx) & 1;
        uint p2 = ((mask_player_2 >> field_idx) << 1) & 2;
        return State(p1 | p2);
    }

    static Board randomState(uint seed) {
        uint set_mask = murmur3(seed) & ((1 << 24) - 1);
        uint is_player_1 = murmur3(seed);
        return Board(set_mask & is_player_1, set_mask & ~is_player_1);
    }

    float4 render(float2 px) {
        uint2 dim;
        screen.GetDimensions(dim.x, dim.y);

        // uv in [-1, 1] range
        float2 uv = (2*px - dim.xy) / min(dim.x, dim.y);
        uv *= 4;

        // Cell id
        int field;
        float field_dist = 1.#INF;
        float2 field_uv;
        for (int ring = 1; ring <= 3; ring ++) {
            float2 l_uv = uv / ring;
            int2 cell = int2(clamp(round(l_uv), -1, 1));
            if (all(cell == 0)) {
                continue;
            }
            
            float2 c_uv = l_uv - cell;
            c_uv *= ring;

            float d = sdf_sphere(c_uv, CELL_RADIUS);
            if (d < field_dist) {
                field_dist = d;
                field_uv = c_uv;
                //field = (cell.x*cell.y + 1) + (1-cell.y)*2 + (cell.x < 0 && cell.y == 0) * 4;
                field = int(round(atan2(-cell.y, cell.x) / 3.1415926 * 4 + 3));
                field += 8 * (ring-1);
            }
        }

        float4 col = float4(1);

        // Connections
        float line_mask = abs(sdf_box(uv, 3));
        line_mask = min(line_mask, abs(sdf_box(uv, 2)));
        line_mask = min(line_mask, abs(sdf_box(uv, 1)));
        line_mask = min(line_mask, sdf_line(sort2(abs(uv)), float2(0,1), float2(0,3)));
        line_mask = 1 - smoothstep(0, .01, line_mask - LINE_WIDTH);
        col = alpha_over(col, float4(float3(0.2), line_mask));

        // Field colors
        if (field_dist < 0) {
            State field_state = getState(field);
            uint mills = ~in_mill_mask(true) & ~in_mill_mask(false) & ~free_fields_mask();
            uint movable = movable_mask(true) | movable_mask(false);
            float4 field_col = stateColor(field_state);
            field_col.rgb *= 1 + ((mills >> field) & 1) * .4;
            col = alpha_over(col, field_col);
        }
        
        // Field border color
        float field_border_mask = 1 - smoothstep(0, .01, abs(field_dist) - LINE_WIDTH);
        col = alpha_over(col, float4(float3(0.2), field_border_mask));

        return col;
    }

    uint32_t player_mask(bool is_player_1) {
        return is_player_1 ? mask_player_1 : mask_player_2;
    }

    static uint32_t rotate_mask_ccw_90(uint32_t mask) {
        const uint32_t spill_mask = 0x03_0303;
        return ((mask & ~spill_mask) >> 2) | ((mask & spill_mask) << 6);
    }

    static uint32_t rotate_mask_cw_90(uint32_t mask) {
        const uint32_t spill_mask = 0xc0_c0c0;
        return ((mask & ~spill_mask) << 2) | ((mask & spill_mask) >> 6);  
    }
    
    static uint32_t rotate_mask_ccw_45(uint32_t mask) {
        const uint32_t spill_mask = 0x01_0101;
        return ((mask & ~spill_mask) >> 1) | ((mask & spill_mask) << 7);
    }

    static uint32_t rotate_mask_cw_45(uint32_t mask) {
        const uint32_t spill_mask = 0x80_8080;
        return ((mask & ~spill_mask) << 1) | ((mask & spill_mask) >> 7);  
    }

    uint32_t free_fields_mask() {
        return ~occupied_fields_mask() & 0xFF_FFFF;
    }

    uint32_t occupied_fields_mask() {
        return mask_player_1 | mask_player_2;
    }

    uint32_t movable_mask(bool player) {
        static const uint FLYING = 1 << 31;
        uint fields = player_mask(player);
        uint pieces = countbits(fields);
        // Lost
        if (pieces <= 2)
            return 0;
        // Flying
        if (pieces == 3)
            return fields | FLYING;
        // Normal movement
        const uint ring_mask = 0b10101010_10101010_10101010;
        uint free = free_fields_mask();
        uint can_move = rotate_mask_cw_45(free) & fields;// Move in-ring to the left
        can_move |= rotate_mask_ccw_45(free) & fields;   // Move in-ring to the right
        can_move |= ((free >> 8) & ring_mask) & fields;  // Move outwards
        can_move |= ((free << 8) & ring_mask) & fields;  // Move inwards
        // If boxed in, able to fly
        if (can_move == 0)
            return fields | FLYING;
        return can_move;
    }

    uint32_t target_position_mask(uint field_idx, bool can_fly) {
        // Given a field index and whether the piece can fly return the valid target positions
        // Does not check that the given field is a piece, just the adjacency information.
        uint free = free_fields_mask();
        if (can_fly)
            return free;
        uint field = 1 << field_idx;
        const uint ring_mask = 0b10101010_10101010_10101010;
        uint in_ring = rotate_mask_ccw_45(field) | rotate_mask_cw_45(field);
        uint cross_ring = ((field << 8) | (field >> 8)) & ring_mask;
        uint dirs = in_ring | cross_ring;
        
        return dirs & free;
    }

    uint32_t in_mill_mask(bool player_1) {
        uint fields = player_mask(player_1);
        
        // Mask out only the horizonals, and check if all of a horizontal are set 
        // (only in case of 111 will it propagate the carry)
        const uint h_mask = 0b01110111_01110111_01110111;
        uint in_horizontal_mill = (h_mask & fields) + 0x11_1111;
        // Extract the corresponding carry results of the max 3 consecutive 111 fields
        // and spread it back to the bits *7 triplicates, shift to original positions
        in_horizontal_mill = ((~h_mask & in_horizontal_mill) * 7) >> 3;
        
        // Vertical mills, similar to horizontal mills, but we roll the lowest top left corners
        // up to the next ring to perform the same trick as for the horizontals...
        const uint v_mask = 0b1_11011101_11011101_11011101;
        const uint v_mask_corner = 0x101_0101;
        const uint v_mask_rest = v_mask & ~v_mask_corner;
        uint in_vertical_mill = (fields & v_mask_rest) | ((fields & v_mask_corner) << 8);
        in_vertical_mill += 0x44_4444;
        in_vertical_mill = ((~v_mask & in_vertical_mill) * 7) >> 3;
        in_vertical_mill = (v_mask_rest & in_vertical_mill) | ((v_mask_corner & in_vertical_mill) >> 8);

        // Crossover rings, collect the masks in the lowest ring, then spread back across the rings
        const uint ring_mask = 0b10101010;
        uint in_ring_mill = (fields >> 16) & (fields >> 8) & fields & ring_mask;
        in_ring_mill *= 0x01_0101;

        return in_horizontal_mill | in_vertical_mill | in_ring_mill;
    }

    uint32_t takeable_mask(bool player_1) {
        // Get all the fields from the given player which can be taken
        return player_mask(player_1) & ~in_mill_mask(player_1);
    }

    Board apply_move(bool is_player_1, Move move) {
        // Takes a given move summary and generates the next board state from it -- non strict variant
        Board new_board = {mask_player_1, mask_player_2};
        // Remove the source piece and place it on the target field
        uint new_fields = player_mask(is_player_1);
        new_fields &= ~((move.source < 24) ? (1 << move.source) : 0);
        new_fields |= (1 << move.target);
        if (is_player_1) {
            new_board.mask_player_1 = new_fields;
        } else {
            new_board.mask_player_2 = new_fields;
        }
        // Remove the taken piece
        if (move.take < 24){
            if (is_player_1) {
                new_board.mask_player_2 &= ~(1 << move.take);
            } else {
                new_board.mask_player_1 &= ~(1 << move.take);
            }
        }
        return new_board;
    }
}



struct Move {
    uint packed;
    
    // Which piece to take (-1 = 31 if none, during beginning of game)
    property uint source { 
        get { return packed & 0x1F; }
        set { packed ^= uint((packed ^ newValue) & 0x1F);}
    }
    // Where to place it
    property uint target {
        get { return (packed >> 5) & 0x1F; }
        set { packed ^= uint((packed ^ (newValue<<5)) & (0x1F<<5));}
    }
    // Which opponent to take (-1 = 31 if none)
    property uint take {
        get { return (packed >> 10) & 0x1F; }
        set { packed ^= uint((packed ^ (newValue<<10)) & (0x1F << 10));}
    }

    static const Move empty() {
        Move m;
        m.source = -1;
        m.target = -1;
        m.take = -1;
        return m;
    }
}

// RNG helpers
struct RNG {
    uint state = 0xDEADBEEF;

    __init(uint seed) {
        this.seed(seed);
    }

    static uint xorshift32(uint x) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        return x;
    }

    [mutating]
    void seed(uint s) {
        state = s ^ 0x6E6E7718;
    }

    [mutating]
    uint next_raw() {
        state = xorshift32(state);
        return state;
    }

    [mutating]
    float next() {
        return next_raw() / float(0xFFFFFFFF);
    }   

    [mutating]
    vector<float, N> next<int N>() {
        vector<float, N> result;
        for (var i = 0; i < N; i++)
            result[i] = next();
        return result;
    }
};



// Bit helper
uint nth_bit_idx(uint mask, uint n) {
    uint found = 0;
    uint pos = 0;
    for (uint i = 0; (i < 32) && (mask != 0); i++) {
        pos = firstbitlow(mask);
        if (found == n)
            return pos;
        mask -= 1 << pos;
        found++;
    }
    return -1;
}

uint wrapping_bit_idx(uint mask, uint n) {
    uint nth_bit = n % countbits(mask);
    return nth_bit_idx(mask, nth_bit);
}

// TODO: Implement an agent below
// To create a valid move do the following:
//   1. Query for all movable tiles
//      - Skipped for the first 2*9 = 18 turns, placement phase
//   2. Select a tile and check where it can get moved to
//   3. Check if a mill was created, and if any opponent piece can be taken.
//      - If so, create a new move with a takable piece taken

interface IAgent {
    property bool is_player_1 {get;}
    [mutating]
    Move get_move(Board board, uint turn);
}

struct RandomAgent : IAgent {
    bool is_player_1;
    RNG rng;
    __init(bool is_player_1, uint seed) {
        this.is_player_1 = is_player_1;
        this.rng = RNG(seed);
    }

    [mutating]
    Move get_move(Board board, uint turn) {
        Move m = Move.empty();

        uint targets;
        if (turn < 2*9) {
            targets = board.free_fields_mask();
        } else {
            uint movable = board.movable_mask(is_player_1);
            bool can_fly = (movable >> 31) != 0;
            // Note: We bias the rng with the current turn, otherwise due to the weak rng it easily gets caught in cycles.
            m.source = wrapping_bit_idx(movable & 0xFFFFFF, turn + rng.next_raw());  
            targets = board.target_position_mask(m.source, can_fly);
        }

        m.target = wrapping_bit_idx(targets, turn + rng.next_raw());

        // Check if we move into a mill, if so we can take
        Board proposal = board.apply_move(is_player_1, m);
        if ((proposal.in_mill_mask(is_player_1) & (1 << m.target)) != 0) {
            uint takeable = board.takeable_mask(!is_player_1);
            m.take = wrapping_bit_idx(takeable, turn + rng.next_raw());
        }
        return m;
    }
}

struct TreeSearchAgent : IAgent {
    bool is_player_1;
    __init(bool is_player_1) {
        this.is_player_1 = is_player_1;
    }

    Move get_move(Board board, uint turn) {

    }
}

// Test runner
struct TestRunner {
    IAgent player_1;
    IAgent player_2;
    Board board_state;
    uint turn;

    __init(IAgent p1, IAgent p2) {
        this.player_1 = p1;
        this.player_2 = p2;
        this.board_state = {0, 0};
        this.turn = 0;
    }

    [mutating]
    void step() {
        bool is_player_1 = (turn & 1) == 0;
        IAgent player;
        if (is_player_1) {
            player = player_1;
        } else {
            player = player_2;
        }
        Move move = player.get_move(board_state, turn);
        this.turn += 1;
        this.board_state = board_state.apply_move(is_player_1, move);
    }

    uint2 get_score() {
        return uint2(countbits(board_state.mask_player_1), countbits(board_state.mask_player_2));
    }

    bool is_gameover() {
        // if (this.turn > 100)
        //     return true;
        if (this.turn < 2*9)
            return false;
        uint2 score = get_score();
        if (min(score.x, score.y) < 3)
            return true;
        return false;
    }
}


[StorageBuffer(1)]
RWStructuredBuffer<TestRunner> runners;

[shader("compute")]
[numthreads(1)]
[WorkgroupCount(1, 1, 1)]
void step_run(uint3 id : SV_DispatchThreadID) {
    TestRunner runner;
    bool do_reset = true;
    if (time.frame > 0) {
        runner = runners[0];
        do_reset = time.frame % (STEPS_UNTIL_RESET * FRAMES_PER_UPDATE / STEPS_PER_UPDATE + 100) == 0;
    }
    if (do_reset) {
        runner = TestRunner(RandomAgent(true, time.frame), RandomAgent(false, time.frame + 456));
    }
    if (time.frame % FRAMES_PER_UPDATE == 0) {
        for (int i = 0; i < STEPS_PER_UPDATE && !runner.is_gameover(); i++)
            runner.step();
    }
    runners[0] = runner;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main_image(uint3 id : SV_DispatchThreadID)
{
    // Viewport resolution (in pixels)
    uint width, height;
    screen.GetDimensions(width, height);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= width || id.y >= height)
        return;

    // Pixel coordinates (centre of pixel, origin at bottom left)
    float2 fragCoord = float2(float(id.x) + 0.5, float(height - id.y) - 0.5);


    TestRunner runner = runners[0];
    Board board = runner.board_state;

    float4 col = float4(.8, .8, .8, 1);
    // Board board = {1+14*0x010101, 20};
    // board = Board.randomState(int(time.elapsed));
    col = alpha_over(col, board.render(fragCoord));

    // Convert from gamma-encoded to linear colour space
    float3 res = pow(col.rgb, 2.2);

    // Output to screen (linear colour space)
    screen[id.xy] = float4(res.rgb, 1.0);
}
