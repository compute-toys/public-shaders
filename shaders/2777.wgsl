// 32x32 Grid
const GRID_SIZE: u32 = 32u;
const NUM_CELLS: u32 = 1024u;

// ISA
const OP_HALT: u32 = 0u; const OP_LIT: u32 = 1u; const OP_ADD: u32 = 2u; const OP_SUB: u32 = 3u; 
const OP_GT: u32 = 4u; const OP_SET_COLOR: u32 = 7u; const OP_JMP: u32 = 8u; const OP_JZE: u32 = 9u;
const OP_DUP: u32 = 10u; const OP_LOAD_REG: u32 = 13u; const OP_STORE_REG: u32 = 14u; 
const OP_SENSE_NEIGHBORS: u32 = 15u; 

struct VMContext {
    stack: array<u32, 8>, pc: u32, sp: u32, color_reg: u32, done: u32, gpr: u32,
    pad1: u32, pad2: u32, pad3: u32, 
}
struct Board { vms: array<VMContext, NUM_CELLS>, }
@group(0) @binding(0) var<storage, read_write> board : Board;

fn get_color(id: u32) -> vec3f {
    if (id == 1u) { return vec3f(0.2, 1.0, 0.4); } 
    return vec3f(0.05, 0.05, 0.08); 
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    let g_id = vec2u(floor((vec2f(id.xy) / vec2f(screen_size)) * 32.0)); 
    let idx = g_id.y * GRID_SIZE + g_id.x;

    // --- 双缓存核心逻辑 ---
    // 我们利用 u32 的 32 位空间。
    // frame_parity = 0: 读低16位，写高16位
    // frame_parity = 1: 读高16位，写低16位
    // 这样永远不会读到"正在写"的数据
    let parity = time.frame % 2u;
    let read_shift  = parity * 16u;        // 当前帧读取的位置
    let write_shift = (1u - parity) * 16u; // 当前帧写入的位置

    // --- ROM ---
    var rom = array<u32, 34>(
        OP_SENSE_NEIGHBORS, OP_LOAD_REG, OP_JZE, 14u,
        OP_LIT, 2u, OP_SUB, OP_LIT, 1u, OP_GT, OP_JZE, 21u, OP_JMP, 27u,
        OP_LIT, 3u, OP_SUB, OP_JZE, 21u, OP_JMP, 27u,
        OP_LIT, 1u, OP_DUP, OP_STORE_REG, OP_SET_COLOR, OP_HALT, 
        OP_LIT, 0u, OP_DUP, OP_STORE_REG, OP_SET_COLOR, OP_HALT, 
        0u 
    );

    var vm = board.vms[idx];

    // --- 阶段 1: 经典图案初始化 ---
    if (time.frame < 2u) {
        var alive = 0u;
        let x = g_id.x;
        let y = g_id.y;

        // 1. 闪烁灯 (Blinker)
        if (y == 5u && x >= 4u && x <= 6u) { alive = 1u; }
        // 2. 滑翔机 (Glider)
        if ((x == 26u && y == 4u) || (x == 27u && y == 5u) || (x >= 25u && x <= 27u && y == 6u)) { alive = 1u; }
        // 3. 蜂巢 (Beehive)
        if ((y == 25u && (x == 5u || x == 6u)) || (y == 26u && (x == 4u || x == 7u)) || (y == 27u && (x == 5u || x == 6u))) { alive = 1u; }
        // 4. R-五格骨牌 (R-Pentomino)
        if ((x == 16u && y == 15u) || (x == 17u && y == 15u) || (x == 15u && y == 16u) || (x == 16u && y == 16u) || (x == 16u && y == 17u)) { alive = 1u; }

        // 【关键】初始化时，把低16位和高16位都写成 alive
        // 这样无论下一帧读哪一边，都是有数据的！
        board.vms[idx].gpr = alive | (alive << 16u);
        board.vms[idx].color_reg = alive;
        
        let col = get_color(alive);
        textureStore(screen, id.xy, vec4f(col, 1.0));
        return;
    }

    // --- 阶段 2: 演化 ---
    // 每 5 帧跑一次
    if (time.frame % 5 == 0u) {
        vm.sp = 0u; vm.pc = 0u; vm.done = 0u;
        
        // 1. 从 Global 读取状态到 Local
        // 注意：这里我们做了解码，VM 内部只看到 0 或 1
        vm.gpr = (board.vms[idx].gpr >> read_shift) & 0xFFFFu;

        for (var gas = 0; gas < 60; gas++) {
            if (vm.done == 1u) { break; }
            if (vm.pc >= 33u) { vm.done = 1u; break; }

            let instr = rom[vm.pc]; vm.pc++; 
            switch (instr) {
                case OP_HALT: { vm.done = 1u; }
                case OP_LIT: { vm.stack[vm.sp] = rom[vm.pc]; vm.pc++; vm.sp++; }
                case OP_ADD: { vm.sp-=2; vm.stack[vm.sp] = vm.stack[vm.sp] + vm.stack[vm.sp+1]; vm.sp++; }
                case OP_SUB: { vm.sp-=2; vm.stack[vm.sp] = vm.stack[vm.sp] - vm.stack[vm.sp+1]; vm.sp++; }
                case OP_GT:  { vm.sp-=2; vm.stack[vm.sp] = select(0u,1u,vm.stack[vm.sp]>vm.stack[vm.sp+1]); vm.sp++; }
                case OP_JMP: { vm.pc = rom[vm.pc]; }
                case OP_JZE: { vm.sp--; if(vm.stack[vm.sp]==0u){vm.pc=rom[vm.pc];}else{vm.pc++;} }
                case OP_DUP: { vm.stack[vm.sp] = vm.stack[vm.sp-1]; vm.sp++; }
                
                // 本地寄存器操作 (不涉及 global)
                case OP_LOAD_REG: { vm.stack[vm.sp] = vm.gpr; vm.sp++; }
                case OP_STORE_REG: { vm.sp--; vm.gpr = vm.stack[vm.sp]; } 
                case OP_SET_COLOR: { vm.sp--; vm.color_reg = vm.stack[vm.sp]; }
                
                // 邻居感知 (关键修正：必须带 Shift 读取)
                case OP_SENSE_NEIGHBORS: {
                    var sum = 0u;
                    let igx = i32(g_id.x); let igy = i32(g_id.y);
                    for (var i = -1; i <= 1; i++) {
                        for (var j = -1; j <= 1; j++) {
                            if (i == 0 && j == 0) { continue; }
                            let nx = igx + i; let ny = igy + j;
                            if (nx >= 0 && nx < 32 && ny >= 0 && ny < 32) {
                                let n_idx = u32(ny) * GRID_SIZE + u32(nx);
                                
                                // 【核心逻辑】读取邻居时，必须按照 read_shift 解码
                                // 这样我们读到的永远是"上一帧"的稳定状态
                                let n_val = (board.vms[n_idx].gpr >> read_shift) & 0xFFFFu;
                                if (n_val > 0u) { sum++; }
                            }
                        }
                    }
                    vm.stack[vm.sp] = sum; vm.sp++;
                }
                default: { vm.done = 1u; }
            }
        }
        
        // 2. 计算完毕，写回 Global
        // 我们需要保留 Read 部分的数据 (给还没跑完的兄弟看)
        // 只更新 Write 部分的数据 (给下一帧用)
        let old_data = board.vms[idx].gpr & (0xFFFFu << read_shift); // 保留旧的
        let new_data = (vm.gpr & 0xFFFFu) << write_shift;            // 写入新的
        
        board.vms[idx].gpr = old_data | new_data;
        board.vms[idx].color_reg = vm.color_reg; // 颜色无所谓，直接更
    }

    // 渲染
    let uv = vec2f(id.xy) / vec2f(screen_size);
    let cell_uv = fract(uv * 32.0) - 0.5;
    let d = length(cell_uv);
    let mask = smoothstep(0.48, 0.4, d);
    let col = get_color(board.vms[idx].color_reg);
    let glow = smoothstep(0.45, 0.1, d) * (0.2 + f32(board.vms[idx].color_reg)*0.6);
    textureStore(screen, id.xy, vec4f((col + glow) * mask, 1.0));
}