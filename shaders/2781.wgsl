// PNGine Logo Shader
// ===================
// A compute shader that renders an animated grid forming the letter "P".
// Each cell pulses with a wave effect, creating a living, breathing logo.
//
// Controls: custom.C (mouse X) - wave phase offset

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/// 2D transformation with position, rotation, scale, and anchor point.
/// All angles are in radians. I use this to perform a lot of 2D operations when
/// working with shapes in compute.toys.
struct Transform2D {
    pos: vec2f,       // World position (translation)
    angle: f32,       // Rotation in radians
    scale: vec2f,     // Scale factors (x, y)
    anchor: vec2f,    // Rotation anchor in local space
}

/// Result from SDF evaluation: distance to surface and color.
/// Useful to perform layers or other kind of simple compositions
struct SDFResult {
    dist: f32,
    color: vec3f,
}

// ============================================================================
// CONFIGURATION
// ============================================================================

// Grid dimensions: 13 columns x 15 rows
const GRID_WIDTH: f32 = 13.0;
const GRID_HEIGHT: f32 = 15.0;

// Grid bounds (centered at origin)
const GRID_HALF_X: f32 = (GRID_WIDTH - 1.0) / 2.0;
const GRID_HALF_Y: f32 = (GRID_HEIGHT - 1.0) / 2.0;
const GRID_MIN: vec2f = vec2f(-GRID_HALF_X, -GRID_HALF_Y);
const GRID_MAX: vec2f = vec2f(GRID_HALF_X, GRID_HALF_Y);

// Visual settings
const BACKGROUND_COLOR: vec3f = vec3f(0.9, 0.8, 0.7);
const MAIN_SCALE: f32 = 0.1;

// ============================================================================
// MATH UTILITIES
// ============================================================================

/// Maps a value from one range to another.
/// Example: map_range(0.5, 0.0, 1.0, 10.0, 20.0) returns 15.0
fn map_range(value: f32, in_min: f32, in_max: f32, out_min: f32, out_max: f32) -> f32 {
    return out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min);
}

/// Attempt at the iconic 'cosine palette' by Inigo Quilez.
/// Creates smooth color gradients based on a parameter t.
/// See: https://iquilezles.org/articles/palettes/
fn palette(t: f32) -> vec3f {
    let a = vec3f(0.5, 0.5, 0.5);
    let b = vec3f(0.5, 0.5, 0.5);
    let c = vec3f(1.0, 1.0, 1.0);
    let d = vec3f(0.263, 0.416, 0.557);
    return a + b * cos(6.28318 * (c * t + d));
}

// ============================================================================
// 2D TRANSFORMATIONS
// ============================================================================

/// Transforms a world-space point into local space of the given transform.
/// Applies inverse operations: translation -> rotation -> scale.
fn transform_to_local(world_pos: vec2f, xform: Transform2D) -> vec2f {
    // Step 1: Remove world translation
    var p = world_pos - xform.pos;

    // Step 2: Apply inverse rotation
    let c = cos(xform.angle);
    let s = sin(xform.angle);
    p = vec2f(
        c * p.x + s * p.y,
        -s * p.x + c * p.y
    );

    // Step 3: Remove anchor offset (now in rotated space)
    p -= xform.anchor;

    // Step 4: Apply inverse scale
    p /= xform.scale;

    return p;
}

/// Corrects SDF distance after non-uniform scaling.
/// For uniform scale: exact. For mild anisotropy: approximated.
fn scale_sdf_distance(dist: f32, xform: Transform2D) -> f32 {
    let scale_diff = abs(xform.scale.x - xform.scale.y);

    // Uniform scale: exact correction
    if (scale_diff < 0.001) {
        return dist * xform.scale.x;
    }

    // Mild anisotropy (< 2:1 ratio): use harmonic mean
    let ratio = max(xform.scale.x, xform.scale.y) / min(xform.scale.x, xform.scale.y);
    if (ratio < 2.0) {
        let harmonic_scale = 2.0 / (1.0 / xform.scale.x + 1.0 / xform.scale.y);
        return dist * harmonic_scale;
    }

    // Extreme anisotropy: conservative estimate (use smaller scale)
    return dist * min(xform.scale.x, xform.scale.y);
}

// ============================================================================
// SIGNED DISTANCE FUNCTIONS (SDFs)
// ============================================================================

/// SDF for an axis-aligned box centered at origin.
/// Parameter b: half-extents (width/2, height/2).
fn sdf_box(p: vec2f, half_size: vec2f) -> f32 {
    let d = abs(p) - half_size;
    let outside_dist = length(max(d, vec2f(0.0)));
    let inside_dist = min(max(d.x, d.y), 0.0);
    return outside_dist + inside_dist;
}

/// SDF for a box with transformation applied.
fn sdf_box_transformed(p: vec2f, half_size: vec2f, xform: Transform2D) -> f32 {
    let local_p = transform_to_local(p, xform);
    let raw_dist = sdf_box(local_p, half_size);
    return scale_sdf_distance(raw_dist, xform);
}

// ============================================================================
// LETTER "P" MASK
// ============================================================================

/// Returns true if the cell at (col, row) should be HIDDEN.
/// The visible cells form the letter "P" in a 13x15 grid.
///
/// Grid coordinates: (0,0) is top-left, x increases right, y increases down.
/// The letter P occupies columns 3-9, rows 3-11.
fn is_cell_hidden(col: f32, row: f32) -> bool {
    // Row 3: Top of the P (horizontal bar)
    if (row == 3.0 && col >= 3.0 && col <= 8.0) { return true; }

    // Rows 4-7: Upper portion (vertical stem + right edge of bowl)
    if (row == 4.0 && ((col >= 3.0 && col <= 4.0) || (col >= 8.0 && col <= 9.0))) { return true; }
    if (row == 5.0 && ((col >= 3.0 && col <= 4.0) || (col >= 8.0 && col <= 9.0))) { return true; }
    if (row == 6.0 && ((col >= 3.0 && col <= 4.0) || (col >= 8.0 && col <= 9.0))) { return true; }
    if (row == 7.0 && ((col >= 3.0 && col <= 4.0) || col == 8.0)) { return true; }

    // Row 8: Middle bar (closes the bowl)
    if (row == 8.0 && col >= 3.0 && col <= 7.0) { return true; }

    // Rows 9-11: Lower stem
    if (row == 9.0 && col >= 3.0 && col <= 4.0) { return true; }
    if (row == 10.0 && col >= 3.0 && col <= 4.0) { return true; }
    if (row == 11.0 && col >= 3.0 && col <= 4.0) { return true; }

    return false;
}

// ============================================================================
// RENDERING
// ============================================================================

/// Renders the animated grid at the given UV coordinates.
fn render(uv: vec2f, xform: Transform2D, screen_size: vec2u) -> vec3f {
    // Transform to grid space
    let grid_pos = transform_to_local(uv, xform);

    // Determine which cell we're in
    let cell_id = floor(grid_pos);
    let cell_uv = fract(grid_pos) - 0.5;  // Center UV within cell (-0.5 to 0.5)

    // Check if we're inside the grid bounds
    let inside_grid = cell_id.x >= GRID_MIN.x && cell_id.x <= GRID_MAX.x
                   && cell_id.y >= GRID_MIN.y && cell_id.y <= GRID_MAX.y;

    if (!inside_grid) {
        return BACKGROUND_COLOR;
    }

    // Convert to grid coordinates (0,0 at top-left)
    let grid_col = cell_id.x - GRID_MIN.x;
    let grid_row = GRID_MAX.y - cell_id.y;

    // Skip cells that form the letter "P"
    if (is_cell_hidden(grid_col, grid_row)) {
        return BACKGROUND_COLOR;
    }

    // === Animation ===
    // custom.C is a compute.toys uniform controlled by mouse X position
    let time_offset = (custom.C - 2.0) * 4.0;

    // Wave travels diagonally across the grid
    let wave = sin(grid_col * 0.5 + grid_row * 0.3 + time_offset * 2.0);

    // Box size pulses with the wave (smaller = more margin)
    let box_scale = map_range(wave, -1.0, 1.0, 0.1, 0.5);
    let margin = 1.0 - box_scale;

    // Color shifts based on distance from origin
    let color_param = length(vec2f(grid_col, grid_row)) * 0.8 - time_offset * 0.1;
    let box_color = palette(color_param);

    // === Draw the box ===
    let box_xform = Transform2D(vec2f(0.0), 0.0, vec2f(1.0 - margin), vec2f(0.0));
    let dist = sdf_box_transformed(cell_uv, vec2f(1.0), box_xform);

    // Anti-aliased edge
    let aa_width = 1.5 / f32(screen_size.y);
    let blend = smoothstep(-aa_width, aa_width, dist);

    return mix(box_color, BACKGROUND_COLOR, blend);
}

// ============================================================================
// COMPUTE SHADER ENTRY POINT
// ============================================================================

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);

    // Skip pixels outside the viewport (edge workgroups may overdraw)
    if (id.x >= screen_size.x || id.y >= screen_size.y) {
        return;
    }

    // Convert pixel coordinates to normalized UV space
    // - Center of pixel (add 0.5)
    // - Flip Y so origin is bottom-left
    // - Normalize to [-aspect, aspect] x [-1, 1]
    let pixel_center = vec2f(f32(id.x) + 0.5, f32(screen_size.y - id.y) - 0.5);
    let uv = (pixel_center * 2.0 - vec2f(screen_size)) / f32(screen_size.y);

    // Render at scaled size (MAIN_SCALE controls zoom level)
    let main_xform = Transform2D(vec2f(0.0), 0.0, vec2f(MAIN_SCALE), vec2f(0.0));
    var color = render(uv, main_xform, screen_size);

    // Convert from sRGB to linear color space for correct display
    color = pow(color, vec3f(2.2));

    // Write to output texture
    textureStore(screen, id.xy, vec4f(color, 1.0));
}
