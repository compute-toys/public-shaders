enable subgroups;

var<workgroup> wgmem: vec3f;

@compute @workgroup_size(16, 16)
fn main_image(
        @builtin(global_invocation_id) gid: vec3u,
        @builtin(subgroup_size) subgroupSize : u32,
        @builtin(subgroup_invocation_id) sgid : u32,
        @builtin(local_invocation_index) lid : u32
) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    // if (gid.x >= screen_size.x || gid.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = vec2f(f32(gid.x) + .5, f32(screen_size.y - gid.y) - .5);

    // Normalised pixel coordinates (from 0 to 1)
    let uv = fragCoord / vec2f(screen_size);

    // Time varying pixel colour
    var col = .5 + .5 * cos(time.elapsed + uv.xyx + vec3f(0.,2.,4.));

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // One thread per workgroup writes the value to workgroup memory.
    if (lid == 0) {
        wgmem = col;
    }
    workgroupBarrier();
    var v = vec3f(0);

    // One thread per subgroup reads the value from workgroup memory
    // and shares that value with every other thread in the subgroup
    // to reduce local memory bandwidth.
    if (sgid == 0) {
      v = wgmem;
    }
    v = subgroupBroadcast(v, 0);
    var res = v;

    // Output to screen (linear colour space)
    textureStore(screen, gid.xy, 10 * vec4f(res - col, 1.));
}
