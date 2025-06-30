#storage positions array<vec2f>
#storage velocities array<vec2f>


const NUM_PARTICLES = 200u;
const STATE_IDX = NUM_PARTICLES - 1u;

const uvmult = 1.6;


fn hash(n_in: uint) -> uint {
    var n = n_in * 1337u;
    n = (n ^ 61u) ^ (n >> 16u); n *= 9u;
    n = (n ^ (n >> 4u)) * 69069u; n ^= n >> 15u;
    return n;
}


fn wrapped_distance(p1: vec2f, p2: vec2f) -> f32 {
    var delta = abs(p1 - p2);
    if (delta.x > 0.5) { delta.x = 1.0 - delta.x; }
    if (delta.y > 0.5) { delta.y = 1.0 - delta.y; }
    return length(delta);
}


fn wrapped_delta(p1: vec2f, p2: vec2f) -> vec2f {
    var delta = p1 - p2;
    if (delta.x > 0.5) { delta.x -= 1.0; } else if (delta.x < -0.5) { delta.x += 1.0; }
    if (delta.y > 0.5) { delta.y -= 1.0; } else if (delta.y < -0.5) { delta.y += 1.0; }
    return delta;
}



@compute @workgroup_size(30)
fn pass1(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= NUM_PARTICLES) { return; }


    if (time.frame == 0u) {
        if (i == STATE_IDX) {
            positions[STATE_IDX] = vec2f(2.0, 12345.0);
            velocities[STATE_IDX] = vec2f(0.0);
        } else {
            let n = hash(i);
            positions[i] = vec2f(float(n % 1000u), float((n/1000u)%1000u)) / 1000.0;
            velocities[i] = vec2f(0.0);
        }
        return;
    }


    if (i == 0u) {
        let num_reds = u32(positions[STATE_IDX].x);
        var random_seed = u32(positions[STATE_IDX].y);

        if (num_reds > 1u && num_reds < STATE_IDX) {
            var visited: array<bool, STATE_IDX>;
            var queue: array<u32, STATE_IDX>;
            for(var j=0u; j<num_reds; j=j+1u) { visited[j] = false; }

            var queue_head = 0u;
            var queue_tail = 0u;
            var visited_count = 0u;

            queue[queue_tail] = 0u;
            queue_tail += 1u;
            visited[0] = true;
            visited_count = 1u;

            while (queue_head < queue_tail) {
                let current_idx = queue[queue_head];
                queue_head += 1u;
                for (var neighbor_idx = 0u; neighbor_idx < num_reds; neighbor_idx = neighbor_idx + 1u) {
                    if (!visited[neighbor_idx]) {
                        
                        let p_current = positions[current_idx];
                        let p_neighbor = positions[neighbor_idx];
                        var is_adjacent = false;
                        
                        let delta_cn = wrapped_delta(p_current, p_neighbor);
                        let r_sq = dot(delta_cn, delta_cn) * 0.25;

                        if (r_sq > 0.0000001) {
                            is_adjacent = true;
                            
                            let m = fract(p_neighbor + delta_cn * 0.5);

                          
                            for (var k = 0u; k < STATE_IDX; k = k + 1u) { 
                                if (k == current_idx || k == neighbor_idx) { continue; }
                                
                                let p_k = positions[k];
                                let delta_km = wrapped_delta(p_k, m);
                                let dist_km_sq = dot(delta_km, delta_km);
                                
                                if (dist_km_sq < r_sq - 0.000001) {
                                    is_adjacent = false;
                                    break;
                                }
                            }
                        }

                        if (is_adjacent) {
                            visited[neighbor_idx] = true;
                            visited_count += 1u;
                            queue[queue_tail] = neighbor_idx;
                            queue_tail += 1u;
                        }
                    }
                }
            }

            if (visited_count == num_reds) {
                let next_num_reds = num_reds + 1u;
                let pool_size = STATE_IDX - num_reds;

                if (pool_size > 0u) {
                    let random_unred_offset = hash(random_seed) % pool_size;
                    let idx_to_swap = num_reds + random_unred_offset;

                    let temp_pos = positions[num_reds];
                    let temp_vel = velocities[num_reds];
                    positions[num_reds] = positions[idx_to_swap];
                    velocities[num_reds] = velocities[idx_to_swap];
                    positions[idx_to_swap] = temp_pos;
                    velocities[idx_to_swap] = temp_vel;
                    
                    velocities[STATE_IDX] = vec2f(f32(num_reds), f32(time.frame));
                    
                    positions[STATE_IDX].x = f32(next_num_reds);
                    positions[STATE_IDX].y = f32(hash(random_seed + 1u));
                }
            }
        }
    }


    if (i >= STATE_IDX) { return; }

    var pos = positions[i];
    var vel = velocities[i];
    var force = vec2f(0.0);

    for (var j = 0u; j < STATE_IDX; j = j + 1u) {
        if (i == j) { continue; }
        let delta = wrapped_delta(pos, positions[j]);
        let dist_sq = max(0.0001, dot(delta, delta));
        force += delta / dist_sq * 0.00008;
    }

    if (mouse.pos.x >= 0) {
        let mouse_uv = uvmult*float2(mouse.pos) / float2(textureDimensions(screen));
        let mouse_delta = wrapped_delta(pos, mouse_uv);
        let mouse_dist_sq = max(0.0001, dot(mouse_delta, mouse_delta));
        var mouse_force_dir = 1.0;
        if (mouse.click > 0) { mouse_force_dir = 1.8; }
        force += mouse_force_dir * mouse_delta / mouse_dist_sq * 0.0003;
    }

    vel += force;
    vel *= 0.9982;
    pos += vel * 0.0003;
    pos = fract(pos);

    positions[i] = pos;
    velocities[i] = vel;
}


fn rect(uv: vec2f, x: f32, y: f32, w: f32, h: f32) -> f32 {
    if (uv.x > x && uv.x < x + w && uv.y > y && uv.y < y + h) {
        return 1.0;
    }
    return 0.0;
}


fn draw_digit(uv_in: vec2f, digit: u32) -> f32 {
    var uv = uv_in;
    uv.y = 1.0 - uv.y; 


    let thick = 0.18;
    let h_w = 0.6; 
    let v_h = 0.4; 


    let a = rect(uv, 0.2, 1.0 - thick, h_w, thick);
    let b = rect(uv, 1.0 - thick, 0.5, thick, v_h);
    let c = rect(uv, 1.0 - thick, 0.1, thick, v_h);
    let d = rect(uv, 0.2, 0.0, h_w, thick);
    let e = rect(uv, 0.0, 0.1, thick, v_h);
    let f = rect(uv, 0.0, 0.5, thick, v_h);
    let g = rect(uv, 0.2, 0.5 - thick / 2.0, h_w, thick);

    var s = 0.0;
    switch (digit) {
        case 0u: { s = a + b + c + d + e + f; }
        case 1u: { s = b + c; }
        case 2u: { s = a + b + g + e + d; }
        case 3u: { s = a + b + g + c + d; }
        case 4u: { s = f + g + b + c; }
        case 5u: { s = a + f + g + c + d; }
        case 6u: { s = a + f + g + e + c + d; }
        case 7u: { s = a + b + c; }
        case 8u: { s = a + b + c + d + e + f + g; }
        case 9u: { s = a + b + c + d + f + g; }
        default: {}
    }
    return min(s, 1.0); 
}


@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    let screen_dims = float2(textureDimensions(screen));
    var uv = uvmult * float2(id.xy) / screen_dims;
    let screen_uv = float2(id.xy) / screen_dims;


    let ripple_data = velocities[STATE_IDX];
    let ripple_start_frame = u32(ripple_data.y);

    if (ripple_start_frame > 0u) {
        let ripple_age = f32(time.frame - ripple_start_frame);
        let ripple_max_age = 120.0; 

        if (ripple_age < ripple_max_age) {
            let ripple_idx = u32(ripple_data.x);
            let ripple_origin = positions[ripple_idx];
            
            let wrapped_origin_to_uv = wrapped_delta(uv, ripple_origin);
            let dist_to_origin = length(wrapped_origin_to_uv);


            let ripple_speed = 0.01;
            let ripple_width = 0.15;
            let ripple_strength = 0.01;

            let current_radius = ripple_age * ripple_speed;
            
            let wave_dist = abs(dist_to_origin - current_radius);
            var wave_val = smoothstep(ripple_width, 0.0, wave_dist);

            wave_val *= smoothstep(ripple_max_age, ripple_max_age * 0.1, ripple_age);

            if (dist_to_origin > 0.0001) {
                 let perturbation_dir = wrapped_origin_to_uv / dist_to_origin;
                 uv += perturbation_dir * wave_val * ripple_strength;
                 uv = fract(uv);
            }
        }
    }


    var dist1 = 1000.0;
    var dist2 = 1000.0;
    var idx1 = 0u;
    for (var i = 0u; i < STATE_IDX; i = i + 1u) {
        let p = positions[i];
        let d = length(wrapped_delta(uv, p));
        if (d < dist1) {
            dist2 = dist1;
            dist1 = d;
            idx1 = i;
        } else if (d < dist2) {
            dist2 = d;
        }
    }


    let boundary = pow(smoothstep(0.00, 0.024, dist2 - dist1), 12.5);
    var baseColor = vec3f(boundary);


    let num_reds = u32(positions[STATE_IDX].x);
    if (idx1 < num_reds) {
        let fill = 1. - smoothstep(0.0, 0.7, dist1);

        let hue = (f32(idx1) / f32(num_reds)) * 3.2831;
        let red_color = vec3f(
            0.5 + 0.5 * cos(hue + 0.0),
            0.5 + 0.5 * cos(hue + 1.094),
            0.5 + 0.5 * cos(hue +2.188)
        );
        baseColor = mix(baseColor, red_color, fill*1.4);
    }

    let lightDir = normalize(vec2f(sin(uv.x*22.), -0.8));

    let cellSeed = positions[idx1];
    let cellDelta = wrapped_delta(uv, cellSeed);
    let cellNormal = normalize(cellDelta);

    let diff = 1-pow(clamp(sin(2.1*dot(cellNormal, lightDir)), 0.0, 1.0),11.1);
    let lighting = 0.07+ .93 * diff;


    var color = baseColor * lighting;

    var score: u32;
    if (num_reds >= 2u) {
        score = num_reds - 2u;
    } else {
        score = 0u;
    }
    
    let text_pos = vec2f(0.05, 0.50);
    let char_size = vec2f(0.02, 0.05);
    let char_spacing = 0.035;
    let padding = 0.03;
    
    var score_width: f32;
    if (score >= 100u) {
        score_width = char_size.x + 2.0 * char_spacing;
    } else if (score >= 10u) {
        score_width = char_size.x + 1.0 * char_spacing;
    } else {
        score_width = char_size.x;
    }

    let bg_pos = text_pos - padding;
    let bg_size = vec2f(score_width + 2.0 * padding, char_size.y + 2.0 * padding);
    let bg_mask = rect(screen_uv, bg_pos.x-0.01, bg_pos.y-0.014, bg_size.x, bg_size.y*1.3);
    let bg_color = vec3f(0);
    
    color = mix(color, bg_color, bg_mask * 1);

    var text_mask = 0.0;
    
    let d100 = score / 100u;
    let d10 = (score / 10u) % 10u;
    let d1 = score % 10u;

    if (score >= 100u) {
        var digit_uv = (screen_uv - text_pos) / char_size;
        text_mask = max(text_mask, draw_digit(digit_uv, d100));
        digit_uv = (screen_uv - (text_pos + vec2f(char_spacing, 0.0))) / char_size;
        text_mask = max(text_mask, draw_digit(digit_uv, d10));
        digit_uv = (screen_uv - (text_pos + vec2f(char_spacing * 2.0, 0.0))) / char_size;
        text_mask = max(text_mask, draw_digit(digit_uv, d1));
    } else if (score >= 10u) {
        var digit_uv = (screen_uv - text_pos) / char_size;
        text_mask = max(text_mask, draw_digit(digit_uv, d10));
        digit_uv = (screen_uv - (text_pos + vec2f(char_spacing, 0.0))) / char_size;
        text_mask = max(text_mask, draw_digit(digit_uv, d1));
    } else {
        let digit_uv = (screen_uv - text_pos) / char_size;
        text_mask = max(text_mask, draw_digit(digit_uv, d1));
    }

    let text_color = vec3f(1.0, 0.8, 0.4);
    color = mix(color, text_color, text_mask);

    textureStore(screen, int2(id.xy), vec4f(color, 1.0));
}