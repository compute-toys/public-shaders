fn get_cell_state(position: vec2i) -> i32 {
    let color = textureLoad(pass_in, position, 0, 0);

    if color.r > 0.05 {
        return 1;
    } else {
        return 0;
    }
}

fn get_neighbor_count(position: vec2i) -> i32 {
    let n1 = get_cell_state(position.xy + vec2i(0, -1));
    let n2 = get_cell_state(position.xy + vec2i(0, 1));
    let n3 = get_cell_state(position.xy + vec2i(-1, 0));
    let n4 = get_cell_state(position.xy + vec2i(1, 0));

    let n5 = get_cell_state(position.xy + vec2i(-1, -1));
    let n6 = get_cell_state(position.xy + vec2i(1, -1));
    let n7 = get_cell_state(position.xy + vec2i(-1, 1));
    let n8 = get_cell_state(position.xy + vec2i(1, 1));


    return n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8;
}

fn read_cell(position: vec2i) -> vec4f {
    return textureLoad(pass_in, position, 0, 0);
}

fn write_cell(position: vec2i, color: vec4f) {
    textureStore(pass_out, position, 0, color);
    textureStore(screen, position, color);
}

fn cell_progress(cell_alive: bool, neighbor_count: i32) -> bool {
    if (cell_alive) {
		if (!(neighbor_count == 2 || neighbor_count == 3)) {
            return false;
        }
    }
    if neighbor_count == 3 {
        return true;
    }
    return cell_alive;
}


@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Boiler
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);
    let uv = fragCoord / vec2f(screen_size);
    let uv_int: vec2i = vec2i(i32(id.x), i32(id.y));

    // Static
    let on_color = vec4f(0.10, 1.0, 1.0, 1.0);
    let off_color = vec4f(0.0, 0.0, 0.0, 1.0);

    // Handle starting state
    if time.elapsed < 0.5 {  
        let noise_color = textureSampleLevel(channel0, bilinear, vec2f(uv.x, 1.0 - uv.y), 0);
        if noise_color.r > 0.5 {
            write_cell(uv_int.xy, on_color);
        } else {
            write_cell(uv_int.xy, off_color);
        }
        return;
    }


    // Read persistent texture
    let current_color = textureLoad(pass_in, uv_int.xy, 0, 0);
        
    // Do Stuff
    let neighbor_count = get_neighbor_count(uv_int);
    let cell_alive = current_color.r > 0.05;
    
    let new_cell = cell_progress(cell_alive, neighbor_count);
    
    if new_cell {
        let color = vec4f(0.10, 1.0, 1.0, 1.0);
        write_cell(uv_int.xy, color);
    } else {
        let color = vec4f(0.0, 0.0, current_color.b - 0.01, 1.0);
        write_cell(uv_int.xy, color);
    }
    
}
