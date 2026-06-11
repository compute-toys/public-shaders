const CELL_SIZE = 128u;

fn sdCircle(p: vec2f, r: f32) -> f32
{
    return length(p) - r;
}

fn hash3D(p: vec3<f32>) -> vec3<f32> {
  var p3 = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yxz + 33.33);
  return fract((p3.xxy + p3.yxx) * p3.zyx) * 2.0 - 1.0;
}

fn hash22(p: vec2f) -> vec2f {
  var p3 = fract(vec3f(p.xyx) * vec3f(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.xx + p3.yz) * p3.zy);
}

fn get_cell_center(cell: vec2u) -> vec2f {
    return vec2f(cell * CELL_SIZE) + mix(vec2(0.5), hash22(vec2f(cell)),custom.distort) * f32(CELL_SIZE);
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    var closest_cell = vec2u(0);
    var closest_cell_dist = 1000000000.0;

    for (var x = -1; x <= 1; x++) {
        for (var y = -1; y <= 1; y++) {
            let cell = vec2i(id.xy / CELL_SIZE) + vec2i(x, y);
            let cell_center = get_cell_center(vec2u(cell));
            let dist = distance(vec2f(id.xy), cell_center);

            if (dist < closest_cell_dist) {
                closest_cell = vec2u(cell);
                closest_cell_dist = dist;
            }
        }
    }

    let cell_center = get_cell_center(closest_cell);

    // Time varying pixel colour
    var col = hash3D(vec3(f32(closest_cell.x), f32(closest_cell.y), -f32(closest_cell.y)));

    if(sdCircle(vec2f(id.xy) - cell_center, 8.0) < 0.0){
        col = vec3(1.0);
    }

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}
