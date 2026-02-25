@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    // Viewport resolution (in pixels)
    let screen_size: vec2<u32> = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) {
        return;
    }

    let pos = vec3<f32>(
        f32(id.x),
        f32(id.y),
        time.elapsed * 10.
    );

    let color: f32 = voronoi_3d(pos);

    // Output to screen (linear colour space)
    textureStore(
        screen,
        vec2<i32>(i32(id.x), i32(id.y)),
        vec4<f32>(color, color, color, 0.2)
    );
}

const cell_size = 40.;

fn voronoi_3d(loc: vec3<f32>) -> f32 {
    let cell = get_current_cell(loc);
    
    var winning_distance = 3.402823466e+38;
    var winning_point = vec3<i32>(0, 0, 0);

    for (var x = -1; x <= 1; x += 1)
    {
        for (var y = -1; y <= 1; y += 1)
        {
            for (var z = -1; z <= 1; z += 1)
            {
                let middle_point = get_cell_point(cell + vec3<i32>(x, y, z));
                let dist = distance(loc, middle_point);

                let new_point = cell + vec3<i32>(x, y, z);
                let is_closer = dist < winning_distance;

                winning_distance = select(winning_distance, dist, is_closer);
                winning_point = select(winning_point, new_point, is_closer);
            }
        }    
    }


    return winning_distance / cell_size;
}

fn randomVec3(x: i32, y: i32, z: i32) -> vec3<f32> {
    let h = hash3D(x, y, z);

    return vec3<f32>(
        f32( h        & 0xFFu) / 255.0,
        f32((h >> 8u) & 0xFFu) / 255.0,
        f32((h >>16u) & 0xFFu) / 255.0
    );
}

fn hash3D(x: i32, y: i32, z: i32) -> u32 {
    var h: u32 =
          u32(x) * 374761393u
        + u32(y) * 668265263u
        + u32(z) * 2147483647u;

    h = (h ^ (h >> 13u)) * 1274126177u;
    h = h ^ (h >> 16u);

    return h;
}

fn get_current_cell(loc: vec3<f32>) -> vec3<i32> {
    let cell = floor(loc / cell_size);

    return vec3<i32>(cell);
}

fn get_cell_point(cell: vec3<i32>) -> vec3<f32> {
    var middle_point = vec3<f32>(cell) * cell_size + vec3<f32>(cell_size * 0.5);
    middle_point += randomVec3(cell.x, cell.y, cell.z) * 19.;

    return middle_point;
}