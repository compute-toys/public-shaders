@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let frag_coord = vec2f(id.xy);

    let grid_uv = frag_coord / f32(screen_size.x) * 3.0;

    let cell_id = ceil(grid_uv);
    let cell_uv = fract(grid_uv - 0.5 * vec2f(cell_id.y * time.elapsed, 0.0)) - 0.5;

    let dist = length(cell_uv);
    let modifier = (dist - 0.4) * f32(screen_size.x) / 40.0;

    let shape = max(dist, modifier);
    let blob = sqrt(1.0 - smoothstep(0.2, 0.5, shape));
    let blob2 = blob * 2.0 * (1.0 - blob);

    let disorted_uv = grid_uv + cell_uv * blob2;
    let hash = cos(length(cos(disorted_uv) - grid_uv) * 3.0);

    let noisy_col = cos(4.0 * hash + vec3f(0., 4., 8.) * blob2 * blob2);
    let core_glow = pow(blob2 / 0.5, 9.0);
    let tex_term = -0.2 * tanh(noisy_col / 0.1 / (1.0 + blob / 0.1));

    var col = vec3f(tanh(core_glow + tex_term + 1.3));
    col = col * col;

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}
