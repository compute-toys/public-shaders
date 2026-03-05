// ============================================================================
// Conway's Game of Life — WGSL Compute Shader for compute.toys
// ============================================================================
//
// Architecture overview:
//
// The simulation grid is divided into 16×16 tiles, each processed by one
// workgroup. Every tile uses an 18×18 shared memory region ("halo") that
// includes a 1-pixel border from neighboring tiles. This border is needed
// so each cell can read its 8 neighbors without cross-workgroup access.
//
// Problem: workgroups can't synchronize with each other, so writing border
// pixels to the main buffer during the same pass that other workgroups read
// from it would cause a data race. Solution: border pixels are written to
// a separate `tile_edges` staging buffer, then copied to `buffer` in a
// second dispatch (`updateBorders`).
//
// Dispatch order:
//   1. initState      (once)  — initialize grid from texture
//   2. updateState             — simulate one Game of Life step per tile
//   3. updateBorders           — copy border results back to main buffer
//   4. main_image              — render buffer to screen
//
// ============================================================================

// Main simulation grid: 1 = alive, 0 = dead
#storage buffer array<u32>

// Staging buffer for tile border pixels (avoids cross-workgroup races)
// Layout: [tile_index * 64 + edge * 16 + pixel_offset]
//   edge 0 = left   (offset = local_y)
//   edge 1 = right  (offset = local_y)
//   edge 2 = top    (offset = local_x)
//   edge 3 = bottom (offset = local_x)
#storage tile_edges array<u32>

// Shared memory tile: 18×18 to hold 16×16 cells + 1-pixel halo border
const TILE_SIZE = 16;
const TILE_WITH_HALO = TILE_SIZE + 2; // 18
var<workgroup> tile: array<u32, TILE_WITH_HALO * TILE_WITH_HALO>; // 324


// ============================================================================
// Index helpers
// ============================================================================

/// Flat index into the main `buffer` array
fn global_index(position: vec3u, screen_size: vec2u) -> u32 {
    return position.y * screen_size.x + position.x;
}

/// Flat index into the shared `tile` array (with +1 offset for the halo)
fn tile_local_index(local_position: vec3i) -> i32 {
    return (local_position.y + 1) * TILE_WITH_HALO + (local_position.x + 1);
}

/// Flat index into the `tile_edges` staging buffer
fn tile_edge_index(
    global_position: vec3u,
    local_position: vec3i,
    screen_size: vec2u,
    edge: u32
) -> i32 {
    let tiles_per_row = screen_size.x / u32(TILE_SIZE);
    let tile_x = global_position.x / u32(TILE_SIZE);
    let tile_y = global_position.y / u32(TILE_SIZE);
    let tile_id = tile_y * tiles_per_row + tile_x;

    // For left/right edges the offset along the edge is local_y;
    // for top/bottom edges the offset is local_x.
    let pixel_offset = select(local_position.y, local_position.x, edge >= 2u);

    return i32(tile_id) * (4 * TILE_SIZE) + i32(edge) * TILE_SIZE + pixel_offset;
}


// ============================================================================
// Tile edge staging (read / write border pixels to avoid cross-tile races)
// ============================================================================

/// If this pixel sits on a tile border, read its value from the staging buffer.
/// Returns 1000 for non-border pixels (sentinel value).
fn read_tile_edge(
    global_position: vec3u,
    local_position: vec3i,
    screen_size: vec2u,
) -> u32 {
    if (local_position.x == 0) {
        return tile_edges[tile_edge_index(global_position, local_position, screen_size, 0)];
    }
    if (local_position.x == TILE_SIZE - 1) {
        return tile_edges[tile_edge_index(global_position, local_position, screen_size, 1)];
    }
    if (local_position.y == 0) {
        return tile_edges[tile_edge_index(global_position, local_position, screen_size, 2)];
    }
    if (local_position.y == TILE_SIZE - 1) {
        return tile_edges[tile_edge_index(global_position, local_position, screen_size, 3)];
    }
    return 1000;
}

/// If this pixel sits on a tile border, write its value to the staging buffer.
/// Returns true if the pixel was on a border, false otherwise.
fn write_tile_edge(
    global_position: vec3u,
    local_position: vec3i,
    screen_size: vec2u,
    value: u32
) -> bool {
    if (local_position.x == 0) {
        tile_edges[tile_edge_index(global_position, local_position, screen_size, 0)] = value;
        return true;
    }
    if (local_position.x == TILE_SIZE - 1) {
        tile_edges[tile_edge_index(global_position, local_position, screen_size, 1)] = value;
        return true;
    }
    if (local_position.y == 0) {
        tile_edges[tile_edge_index(global_position, local_position, screen_size, 2)] = value;
        return true;
    }
    if (local_position.y == TILE_SIZE - 1) {
        tile_edges[tile_edge_index(global_position, local_position, screen_size, 3)] = value;
        return true;
    }
    return false;
}


// ============================================================================
// Tile loading — populate shared memory from global buffer
// ============================================================================

/// Helper: load a single halo pixel from the global buffer into shared memory.
/// If the global coordinate is out of bounds, writes 0 (dead border).
fn load_halo_pixel(
    halo_local: vec3i,
    halo_global: vec3u,
    screen_size: vec2u,
    is_valid: bool
) {
    let destination = tile_local_index(halo_local);
    if (is_valid) {
        tile[destination] = buffer[global_index(halo_global, screen_size)];
    } else {
        tile[destination] = 0;
    }
}

/// Load this cell and its halo neighbors into shared memory.
/// Each thread loads its own cell. Threads on the tile border also load
/// the adjacent halo pixel. Corner threads load diagonal halo pixels.
fn load_tile_into_shared_memory(
    global_position: vec3u,
    local_position: vec3i,
    screen_size: vec2u
) {
    // --- Load own cell ---
    let own_index = tile_local_index(local_position);
    tile[own_index] = buffer[global_index(global_position, screen_size)];

    let gx = global_position.x;
    let gy = global_position.y;
    let lx = local_position.x;
    let ly = local_position.y;
    let max_x = screen_size.x;
    let max_y = screen_size.y;

    let on_left   = lx == 0;
    let on_right  = lx == TILE_SIZE - 1;
    let on_top    = ly == 0;
    let on_bottom = ly == TILE_SIZE - 1;

    // --- Load edge halo pixels (4 edges) ---
    if (on_left) {
        load_halo_pixel(
            vec3i(lx - 1, ly, 0),
            vec3u(gx - 1, gy, 0),
            screen_size,
            gx > 0
        );
    }
    if (on_right) {
        load_halo_pixel(
            vec3i(lx + 1, ly, 0),
            vec3u(gx + 1, gy, 0),
            screen_size,
            gx + 1 < max_x
        );
    }
    if (on_top) {
        load_halo_pixel(
            vec3i(lx, ly - 1, 0),
            vec3u(gx, gy - 1, 0),
            screen_size,
            gy > 0
        );
    }
    if (on_bottom) {
        load_halo_pixel(
            vec3i(lx, ly + 1, 0),
            vec3u(gx, gy + 1, 0),
            screen_size,
            gy + 1 < max_y
        );
    }

    // --- Load corner halo pixels (4 corners) ---
    if (on_left && on_top) {
        load_halo_pixel(
            vec3i(lx - 1, ly - 1, 0),
            vec3u(gx - 1, gy - 1, 0),
            screen_size,
            gx > 0 && gy > 0
        );
    }
    if (on_right && on_top) {
        load_halo_pixel(
            vec3i(lx + 1, ly - 1, 0),
            vec3u(gx + 1, gy - 1, 0),
            screen_size,
            gx + 1 < max_x && gy > 0
        );
    }
    if (on_left && on_bottom) {
        load_halo_pixel(
            vec3i(lx - 1, ly + 1, 0),
            vec3u(gx - 1, gy + 1, 0),
            screen_size,
            gx > 0 && gy + 1 < max_y
        );
    }
    if (on_right && on_bottom) {
        load_halo_pixel(
            vec3i(lx + 1, ly + 1, 0),
            vec3u(gx + 1, gy + 1, 0),
            screen_size,
            gx + 1 < max_x && gy + 1 < max_y
        );
    }
}


// ============================================================================
// Game of Life rules
// ============================================================================

/// Count alive neighbors and apply the Game of Life rules:
///   - Alive cell with 2 or 3 neighbors → survives
///   - Dead cell with exactly 3 neighbors → becomes alive
///   - All other cells → die or stay dead
fn apply_game_of_life_rules(local_position: vec3i) -> u32 {
    let cell_value = tile[tile_local_index(local_position)];

    var alive_neighbors = 0u;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) { continue; }
            let neighbor = vec3i(local_position.x + dx, local_position.y + dy, 0);
            alive_neighbors += tile[tile_local_index(neighbor)];
        }
    }

    if (cell_value > 0) {
        // Alive: survive with 2 or 3 neighbors, die otherwise
        return select(0u, 1u, alive_neighbors == 2 || alive_neighbors == 3);
    } else {
        // Dead: come alive with exactly 3 neighbors
        return select(0u, 1u, alive_neighbors == 3);
    }
}


// ============================================================================
// Dispatch passes
// ============================================================================

/// Pass 0 (once): Initialize the grid from a texture
#dispatch_once initState
@compute @workgroup_size(16, 16)
fn initState(
    @builtin(global_invocation_id) global_position: vec3u,
    @builtin(local_invocation_id) local_position_u32: vec3u,
) {
    let local_position = vec3i(local_position_u32);
    let screen_size = textureDimensions(screen);

    if (global_position.x >= screen_size.x || global_position.y >= screen_size.y) {
        return;
    }

    // Sample input texture to decide initial alive/dead state
    let pixel_center = vec2f(
        f32(global_position.x) + 0.5,
        f32(screen_size.y - global_position.y) - 0.5
    );
    let uv = pixel_center / vec2f(screen_size);
    let color = textureSampleLevel(channel0, bilinear, uv, 0);
    let is_alive = u32(color.r > 0.5 || color.g > 0.5 || color.b > 0.5);

    buffer[global_index(global_position, screen_size)] = is_alive;

    // Clear tile edges staging buffer
    write_tile_edge(global_position, local_position, screen_size, 0);
}


/// Pass 1: Simulate one step of Game of Life
@compute @workgroup_size(16, 16)
fn updateState(
    @builtin(global_invocation_id) global_position: vec3u,
    @builtin(local_invocation_id) local_position_u32: vec3u,
) {
    let local_position = vec3i(local_position_u32);
    let screen_size = textureDimensions(screen);
    let in_bounds = global_position.x < screen_size.x
                 && global_position.y < screen_size.y;

    // Step 1: Load cell data + halo into shared memory
    if (in_bounds) {
        load_tile_into_shared_memory(global_position, local_position, screen_size);
    }

    // All threads must finish writing before any thread reads neighbors
    workgroupBarrier();

    // Step 2: Apply Game of Life rules using shared memory
    var new_value = 0u;
    if (in_bounds) {
        new_value = apply_game_of_life_rules(local_position);
    }

    workgroupBarrier();

    // Step 3: Write results back
    if (in_bounds) {
        let is_border = write_tile_edge(
            global_position,
            local_position,
            screen_size,
            new_value
        );

        // Border pixels go to the staging buffer (written above).
        // Interior pixels go directly to the main buffer.
        if (!is_border) {
            buffer[global_index(global_position, screen_size)] = new_value;
        }
    }
}


/// Pass 2: Copy border results from staging buffer to main buffer
@compute @workgroup_size(16, 16)
fn updateBorders(
    @builtin(global_invocation_id) global_position: vec3u,
    @builtin(local_invocation_id) local_position_u32: vec3u,
) {
    let local_position = vec3i(local_position_u32);
    let screen_size = textureDimensions(screen);

    if (global_position.x >= screen_size.x || global_position.y >= screen_size.y) {
        return;
    }

    let staged_value = read_tile_edge(global_position, local_position, screen_size);

    // Sentinel value 1000 means "not a border pixel" — skip it
    if (staged_value < 100) {
        buffer[global_index(global_position, screen_size)] = staged_value;
    }
}


/// Pass 3: Render the simulation buffer to screen
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) global_position: vec3u) {
    let screen_size = textureDimensions(screen);

    if (global_position.x >= screen_size.x || global_position.y >= screen_size.y) {
        return;
    }

    let cell_value = buffer[global_index(global_position, screen_size)];

    let color = select(
        vec4f(0.0, 0.0, 0.0, 1.0),  // dead: black
        vec4f(1.0, 1.0, 1.0, 1.0),  // alive: white
        cell_value > 0
    );

    textureStore(screen, global_position.xy, color);
}
