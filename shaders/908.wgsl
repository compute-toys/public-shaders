const PI: f32 = 3.1415926;
const GRID_SIZE: f32 = 72.0;
const GRID_THIC: f32 = 1.0;

const RECT_COL: vec3f = vec3f(1.0, 0.5, 0.0);

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = vec2f(textureDimensions(screen));

    // Pixel coordinates(left up)
    var fragCoord = vec2f(id.xy);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (fragCoord.x >= screen_size.x || fragCoord.y >= screen_size.y) { return; }

    // Pixel coordinates(centre)
    fragCoord -= 0.5 * screen_size;

    // Rotation
    let radian = PI * custom.rotate_rdn;
    let rot_mtx = mat2x2f(cos(radian), sin(radian), -sin(radian), cos(radian));
    fragCoord *= rot_mtx;

    // // Scale
    let ratio = 1 / mix(0.5 / custom.scale, 2.0 * custom.scale, 0.5 + fragCoord.y / screen_size.y);
    fragCoord *= ratio;

    // Transformation
    fragCoord -= (vec2f(custom.transform_x, custom.transform_y) - 0.5) * screen_size;
    fragCoord -= vec2f(0.0, sin(4 * time.elapsed) * screen_size.y);

    let d_grid = sdf_grid(fragCoord, 0.5 * GRID_SIZE);
    let smt = custom.grid_smooth * pow(2, ratio);
    var s_grid = smoothstep(GRID_THIC + smt, GRID_THIC, -d_grid);
    // Darken the further grid
    s_grid = clamp(s_grid / pow(ratio, 1.8), 0, 1);

    // Normalised pixel coordinates (from 0 to 1)
    let uv = fragCoord / screen_size.y;

    // Get distance to circle border
    let d_rect = sdf_box(uv, custom.side_len) - custom.box_round;
    let s_rect = smoothstep(0, clamp(0.008 * ratio, 0, 1), -d_rect);

    let grid_col = vec3f(s_grid);

    var col = grid_col * (1 - s_rect) + s_rect * RECT_COL;
    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn sdf_box(p: vec2f, s: f32) -> f32 {
    let d = abs(p) - vec2f(s);

    return length(max(d, vec2f(0))) + min(max(d.x, d.y), 0);
}

// p: coordinate
// hgs: half grid size
fn sdf_grid(p: vec2f, hgs: f32) -> f32 {
    let d = abs(abs(p % (2 * hgs)) - hgs) - hgs;

    return max(d.x, d.y);
}