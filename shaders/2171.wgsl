fn hash(p: float2) -> float {
    let p2 = fract(p * 0.3183099 + 0.1);
    return fract((p2.x + p2.y) * (p2.x * p2.y) * 43758.5453);
}

fn generate_bandwidth_data(x: float, channel: float, t: float) -> float {
    // Realistic bandwidth patterns
    let base_load = 0.3 + channel * 0.1;
    let daily_pattern = sin(x * 0.5 + channel) * 0.2;
    let traffic_spikes = smoothstep(0.8, 1.0, sin(x * 2.0 + t + channel * 2.0)) * 0.3;
    let noise = sin(x * 10.0 + channel * 3.0) * 0.05;
    return clamp(base_load + daily_pattern + traffic_spikes + noise, 0.0, 1.0);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: uint3) {
    let screen_size = uint2(textureDimensions(screen));
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    
    let resolution = float2(screen_size);
    let uv = float2(id.xy) / resolution;
    let t = float(time.frame) / 60.;
    
    // Dark modern background
    let bg_grad = mix(
        float3(0.02, 0.02, 0.03),
        float3(0.0, 0.01, 0.02),
        uv.y
    );
    var col = float4(bg_grad, 0.9);
    
    // === CYBER BACKGROUND ELEMENTS === //
    
    // PERSPECTIVE GRID FLOOR - Skewed and receding with 3D angle
    let grid_horizon = 0.4;  // Where the grid disappears
    let grid_nearest = 0.67; // Nearest point (2/3 down)
    
    // Vanishing point offset for skew (left side smaller)
    let vanishing_x = 0.3; // Shifted left for perspective
    let skew_strength = 0.4; // How much skew to apply
    
    // Calculate perspective-corrected Y position
    if (uv.y > grid_horizon) {
        let grid_y = (uv.y - grid_horizon) / (grid_nearest - grid_horizon);
        let z_depth = 1.0 / (grid_y + 0.1); // Perspective division
        
        // Calculate X skew based on depth
        let x_offset = (vanishing_x - uv.x) * (1.0 - grid_y) * skew_strength;
        let skewed_x = uv.x + x_offset;
        
        // Grid line spacing increases with distance
        let grid_spacing = 30.0;
        let grid_x_spacing = grid_spacing / z_depth;
        let grid_z_spacing = grid_spacing;
        
        // Vertical lines (receding into distance with skew)
        let vert_lines = 30;
        for (var vl = -vert_lines/2; vl < vert_lines/2; vl = vl + 1) {
            // Lines converge toward vanishing point
            let line_base_x = 0.5 + float(vl) / float(vert_lines) * 3.0;
            let line_x = mix(line_base_x, vanishing_x, 1.0 - grid_y);
            
            // Adjust line thickness based on distance and position
            let thickness = 0.001 / z_depth * (1.0 + abs(float(vl)) / float(vert_lines) * 0.5);
            
            if (abs(skewed_x - line_x) < thickness) {
                let grid_fade = exp(-z_depth * 0.2);
                // Fade more on the left side (smaller in distance)
                let side_fade = 1.0 - (vanishing_x - line_x + 0.5) * 0.3;
                col = float4(col.rgb + float3(0.0, 0.3, 0.5) * 0.2 * grid_fade * side_fade, col.a);
                
                // Tracer check - rare colored pulses traveling UP the lines
                let tracer_id = float(vl) * 13.37 + floor(t * 0.2);
                let tracer_chance = hash(float2(tracer_id, 0.0));
                if (tracer_chance > 0.95) {
                    // Tracer position (moving from near to far)
                    let tracer_progress = fract(t * 0.5 + tracer_chance);
                    let tracer_z = tracer_progress * 10.0;
                    let tracer_y = grid_horizon + (1.0 - 1.0 / (tracer_z + 0.1)) * (grid_nearest - grid_horizon);
                    
                    let dist_to_tracer = abs(uv.y - tracer_y);
                    
                    // Create skip effect - only show certain segments
                    let segment_length = 0.03;
                    let skip_pattern = sin(tracer_progress * 30.0) > 0.3; // Creates on/off pattern
                    
                    if (dist_to_tracer < 0.02 / (tracer_z + 1.0) && skip_pattern) {
                        // Much more subtle alpha
                        let base_intensity = (1.0 - dist_to_tracer / (0.02 / (tracer_z + 1.0)));
                        let fade_intensity = (1.0 - tracer_progress) * 0.3; // Max 30% opacity
                        let ghost_fade = sin(tracer_progress * 50.0) * 0.5 + 0.5; // Flicker effect
                        let tracer_intensity = base_intensity * fade_intensity * ghost_fade;
                        
                        // Different colors for different lines
                        var tracer_color = float3(0.0, 1.0, 1.0); // Cyan
                        let color_pick = int(abs(float(vl)) % 4);
                        if (color_pick == 1) { tracer_color = float3(1.0, 0.2, 0.5); } // Pink
                        else if (color_pick == 2) { tracer_color = float3(0.5, 1.0, 0.3); } // Green
                        else if (color_pick == 3) { tracer_color = float3(1.0, 0.5, 0.0); } // Orange
                        
                        // Add trailing ghost effect
                        let ghost_trail = (1.0 - fract(tracer_progress * 20.0)) * 0.5;
                        
                        col = float4(col.rgb + tracer_color * tracer_intensity * side_fade, col.a);
                        
                        // Add subtle afterimage ghosts
                        for (var g = 1; g < 3; g = g + 1) {
                            let ghost_offset = float(g) * 0.02;
                            let ghost_y = tracer_y + ghost_offset;
                            if (abs(uv.y - ghost_y) < 0.01 / (tracer_z + 1.0)) {
                                let ghost_intensity = tracer_intensity * 0.2 / float(g + 1);
                                col = float4(col.rgb + tracer_color * ghost_intensity * side_fade, col.a);
                            }
                        }
                    }
                }
            }
        }
        
        // Horizontal lines (parallel to horizon but curved for perspective)
        let horiz_lines = 30;
        for (var hl = 0; hl < horiz_lines; hl = hl + 1) {
            let line_z = float(hl) / float(horiz_lines) * 10.0;
            let line_y = grid_horizon + (1.0 - 1.0 / (line_z + 0.1)) * (grid_nearest - grid_horizon);
            
            // Curve the horizontal lines for 3D effect
            let curve = sin((skewed_x - vanishing_x) * 2.0) * 0.02 * (1.0 - grid_y);
            
            if (abs(uv.y - (line_y + curve)) < 0.001) {
                let grid_fade = exp(-line_z * 0.3);
                // Stronger fade on left side
                let side_fade = 0.5 + skewed_x * 0.5;
                col = float4(col.rgb + float3(0.0, 0.3, 0.5) * 0.15 * grid_fade * side_fade, col.a);
            }
        }
        
        // Add subtle fog/haze that's stronger on the left (distant) side
        let fog_density = (1.0 - grid_y) * 0.1 * (1.0 - skewed_x);
        col = float4(col.rgb + float3(0.0, 0.1, 0.2) * fog_density, col.a);
    }
    
    // Scrolling digital rain in background
    let rain_x = floor(uv.x * 40.0);
    let rain_offset = hash(float2(rain_x, 0.0)) * 10.0;
    let rain_y = fract(uv.y * 2.0 + t * 0.3 + rain_offset);
    if (rain_y < 0.1 && fract(uv.x * 40.0) < 0.3) {
        let rain_alpha = (1.0 - rain_y * 10.0) * 0.15;
        col = float4(col.rgb + float3(0.0, 0.3, 0.5) * rain_alpha, col.a);
    }
    
    // Circuit board patterns
    let circuit_scale = 20.0;
    let cx = floor(uv.x * circuit_scale);
    let cy = floor(uv.y * circuit_scale);
    let circuit_pattern = hash(float2(cx, cy));
    
    // Horizontal circuit traces
    if (fract(uv.y * circuit_scale) < 0.05 && circuit_pattern > 0.3) {
        col = float4(col.rgb + float3(0.0, 0.2, 0.3) * 0.1, col.a);
    }
    
    // Vertical circuit traces
    if (fract(uv.x * circuit_scale) < 0.05 && circuit_pattern > 0.5 && circuit_pattern < 0.8) {
        col = float4(col.rgb + float3(0.0, 0.2, 0.3) * 0.1, col.a);
    }
    
    // Circuit nodes
    let node_dist = length(fract(uv * circuit_scale) - 0.5);
    if (node_dist < 0.1 && circuit_pattern > 0.7) {
        let node_glow = (1.0 - node_dist * 10.0);
        col = float4(col.rgb + float3(0.0, 0.5, 0.8) * node_glow * 0.2, col.a);
    }
    
    // Hexagonal grid overlay
    let hex_size = 0.05;
    let hx = uv.x * 17.32 / hex_size;
    let hy = uv.y * 20.0 / hex_size;
    let hex_x = floor(hx);
    let hex_y = floor(hy);
    
    let hex_fx = fract(hx);
    let hex_fy = fract(hy);
    
    // Hexagon edges
    let hex_edge = min(
        min(abs(hex_fx - 0.5), abs(hex_fy - 0.5)),
        abs(hex_fx + hex_fy - 1.0)
    );
    
    if (hex_edge < 0.02) {
        let hex_pulse = sin(t * 2.0 + hex_x * 0.1 + hex_y * 0.1) * 0.5 + 0.5;
        col = float4(col.rgb + float3(0.1, 0.3, 0.5) * 0.05 * hex_pulse, col.a);
    }
    
    // Floating data particles in far background
    for (var i = 0; i < 20; i = i + 1) {
        let particle_seed = float(i) * 1.618;
        let px = fract(sin(particle_seed) * 43758.5453);
        let py = fract(sin(particle_seed + 1.0) * 43758.5453);
        let particle_speed = 0.5 + fract(particle_seed * 2.0) * 0.5;
        
        let particle_x = px;
        let particle_y = fract(py + t * particle_speed * 0.1);
        
        let dist = length(uv - float2(particle_x, particle_y));
        if (dist < 0.002) {
            col = float4(col.rgb + float3(0.2, 0.5, 1.0) * 0.5, col.a);
        }
        
        // Particle trail
        let trail_dist = abs(uv.x - particle_x);
        if (trail_dist < 0.001 && uv.y > particle_y && uv.y < particle_y + 0.1) {
            let trail_alpha = (1.0 - (uv.y - particle_y) * 10.0) * 0.2;
            col = float4(col.rgb + float3(0.1, 0.3, 0.8) * trail_alpha, col.a);
        }
    }
    
    // Scanning beam effect
    let scan_pos = fract(t * 0.2);
    let scan_dist = abs(uv.y - scan_pos);
    if (scan_dist < 0.003) {
        let scan_alpha = (1.0 - scan_dist / 0.003) * 0.3;
        col = float4(col.rgb + float3(0.0, 1.0, 1.0) * scan_alpha, col.a);
    }
    
    // Geometric data flow lines
    let flow_lines = 8;
    for (var fl = 0; fl < flow_lines; fl = fl + 1) {
        let line_y = float(fl) / float(flow_lines);
        let line_offset = sin(float(fl) * 1.618) * 0.3;
        let flow_x = fract(uv.x + t * 0.3 + line_offset);
        
        let line_dist = abs(uv.y - line_y);
        if (line_dist < 0.001 && flow_x < 0.1) {
            let pulse = flow_x * 10.0;
            col = float4(col.rgb + float3(0.0, 0.5, 0.8) * pulse * 0.2, col.a);
        }
    }
    
    // Matrix-style cascading data in the far background
    let matrix_cols = 60;
    let col_id = floor(uv.x * float(matrix_cols));
    let col_offset = hash(float2(col_id, 0.0)) * 2.0;
    let matrix_y = fract(uv.y * 3.0 + t * 0.5 + col_offset);
    
    if (fract(uv.x * float(matrix_cols)) < 0.1) {
        let char_y = floor(matrix_y * 20.0);
        let char_brightness = hash(float2(col_id, char_y + t));
        if (char_brightness > 0.7) {
            let fade = 1.0 - matrix_y;
            col = float4(col.rgb + float3(0.0, 0.2, 0.3) * fade * 0.15, col.a);
        }
    }
    
    // Floating wireframe cubes in background
    for (var cube = 0; cube < 3; cube = cube + 1) {
        let cube_t = t * 0.1 + float(cube) * 2.094;
        let cube_x = 0.2 + float(cube) * 0.3;
        let cube_y = 0.3 + sin(cube_t) * 0.1;
        let cube_size = 0.05;
        
        // Simple wireframe square (2D projection of cube)
        let dx = abs(uv.x - cube_x);
        let dy = abs(uv.y - cube_y);
        
        if ((dx < cube_size && dy < cube_size) && 
            (dx > cube_size - 0.002 || dy > cube_size - 0.002)) {
            let cube_alpha = sin(cube_t * 3.0) * 0.2 + 0.3;
            col = float4(col.rgb + float3(0.2, 0.5, 1.0) * cube_alpha * 0.3, col.a);
        }
    }
    
    // === END CYBER BACKGROUND === //
    
    // 3D perspective transform
    let perspective = 0.5;
    let tilt = 0.3;
    
    // Graph dimensions in 3D space
    let graph_width = 0.8;
    let graph_height = 0.5;
    let graph_depth = 0.3;
    
    // Number of bandwidth channels to display
    let num_channels = 4;
    
    // Draw grid floor
    let grid_y = 0.7;
    for (var gx = 0; gx < 20; gx = gx + 1) {
        let x = float(gx) / 20.0;
        let world_x = (x - 0.5) * graph_width;
        
        // Apply perspective
        let z = graph_depth;
        let screen_y = grid_y + z * tilt;
        let screen_x = 0.5 + world_x * (1.0 - z * perspective);
        
        if (abs(uv.x - screen_x) < 0.001) {
            col = float4(col.rgb + float3(0.1, 0.2, 0.3) * 0.2, col.a);
        }
    }
    
    // Draw bandwidth data for each channel
    for (var channel = 0; channel < num_channels; channel = channel + 1) {
        let z = float(channel) / float(num_channels - 1) * graph_depth;
        
        // Channel colors
        var channel_color = float3(0.0, 0.8, 1.0); // Cyan
        if (channel == 1) { channel_color = float3(1.0, 0.2, 0.5); } // Pink
        else if (channel == 2) { channel_color = float3(0.5, 1.0, 0.3); } // Green
        else if (channel == 3) { channel_color = float3(1.0, 0.5, 0.0); } // Orange
        
        // Fade channels in the back
        let depth_fade = 1.0 - z / graph_depth * 0.5;
        channel_color = channel_color * depth_fade;
        
        // Sample bandwidth data
        let samples = 150;
        var closest_dist = 1000.0;
        var on_line = false;
        
        for (var i = 0; i < samples - 1; i = i + 1) {
            let x1 = float(i) / float(samples);
            let x2 = float(i + 1) / float(samples);
            
            // Get bandwidth values
            let time_offset = t * 0.5;
            let bw1 = generate_bandwidth_data(x1 * 10.0 - time_offset, float(channel), t);
            let bw2 = generate_bandwidth_data(x2 * 10.0 - time_offset, float(channel), t);
            
            // Transform to 3D positions
            let world_x1 = (x1 - 0.5) * graph_width;
            let world_x2 = (x2 - 0.5) * graph_width;
            let world_y1 = 0.7 - bw1 * graph_height;
            let world_y2 = 0.7 - bw2 * graph_height;
            
            // Apply perspective transformation
            let screen_x1 = 0.5 + world_x1 * (1.0 - z * perspective);
            let screen_x2 = 0.5 + world_x2 * (1.0 - z * perspective);
            let screen_y1 = world_y1 + z * tilt;
            let screen_y2 = world_y2 + z * tilt;
            
            // Check if we're near this line segment
            if (uv.x >= screen_x1 && uv.x <= screen_x2) {
                let t_seg = (uv.x - screen_x1) / (screen_x2 - screen_x1);
                let y_interp = mix(screen_y1, screen_y2, t_seg);
                let dist = abs(uv.y - y_interp);
                
                if (dist < closest_dist) {
                    closest_dist = dist;
                    on_line = dist < 0.003;
                }
            }
        }
        
        // Draw the line
        if (on_line) {
            col = float4(channel_color, 1.0);
        }
        
        // Add glow
        let glow_intensity = exp(-closest_dist * 200.0);
        col = float4(col.rgb + channel_color * glow_intensity * 0.5, col.a);
        
        // Fill area under the curve (with transparency)
        let fill_samples = 50;
        for (var fx = 0; fx < fill_samples; fx = fx + 1) {
            let fx_norm = float(fx) / float(fill_samples);
            let world_fx = (fx_norm - 0.5) * graph_width;
            let screen_fx = 0.5 + world_fx * (1.0 - z * perspective);
            
            if (abs(uv.x - screen_fx) < 0.002) {
                let bw = generate_bandwidth_data(fx_norm * 10.0 - t * 0.5, float(channel), t);
                let world_fy = 0.7 - bw * graph_height;
                let screen_fy = world_fy + z * tilt;
                
                if (uv.y > screen_fy && uv.y < 0.7 + z * tilt) {
                    let fill_alpha = (1.0 - (uv.y - screen_fy) / (0.7 + z * tilt - screen_fy)) * 0.1;
                    col = float4(col.rgb + channel_color * fill_alpha, col.a);
                }
            }
        }
    }
    
    // Y-axis labels (bandwidth scale)
    if (uv.x < 0.12) {
        for (var i = 0; i <= 5; i = i + 1) {
            let label_y = 0.7 - float(i) / 5.0 * graph_height;
            if (abs(uv.y - label_y) < 0.002) {
                col = float4(col.rgb + float3(0.5), col.a);
            }
        }
    }
    
    // Time axis
    if (uv.y > 0.68 && uv.y < 0.72) {
        let time_marks = sin(uv.x * 20.0 - t * 2.0) > 0.0;
        if (time_marks) {
            col = float4(col.rgb + float3(0.2), col.a);
        }
    }
    
    // Modern UI overlay elements
    // Peak indicator
    let peak_indicator = smoothstep(0.9, 1.0, sin(t * 2.0));
    if (uv.x > 0.85 && uv.x < 0.87 && uv.y > 0.1 && uv.y < 0.15) {
        col = float4(col.rgb + float3(1.0, 0.2, 0.2) * peak_indicator, col.a);
    }
    
    // Subtle vignette
    let vignette = 1.0 - length(uv - 0.5) * 0.5;
    col = float4(col.rgb * vignette, col.a);
    
    // === FOREGROUND CYBER ELEMENTS (very subtle) === //
    
    // HUD corner brackets
    let corner_size = 0.05;
    let corner_thickness = 0.002;
    
    // Top-left corner
    if ((uv.x < corner_size && abs(uv.y - corner_size) < corner_thickness) ||
        (uv.y < corner_size && abs(uv.x - corner_size) < corner_thickness)) {
        col = float4(col.rgb + float3(0.0, 0.8, 1.0) * 0.3, col.a);
    }
    
    // Bottom-right corner
    if ((uv.x > 1.0 - corner_size && abs(uv.y - (1.0 - corner_size)) < corner_thickness) ||
        (uv.y > 1.0 - corner_size && abs(uv.x - (1.0 - corner_size)) < corner_thickness)) {
        col = float4(col.rgb + float3(0.0, 0.8, 1.0) * 0.3, col.a);
    }
    
    // Tech border frame
    let border_width = 0.003;
    if (uv.x < border_width || uv.x > 1.0 - border_width || 
        uv.y < border_width || uv.y > 1.0 - border_width) {
        col = float4(col.rgb + float3(0.0, 0.5, 0.8) * 0.2, col.a);
    }
    
    // Holographic interference pattern (very subtle)
    let interference = sin(uv.x * 100.0) * sin(uv.y * 100.0 + t * 2.0);
    col = float4(col.rgb + float3(0.0, 0.3, 0.5) * interference * 0.02, col.a);
    
    // Data readout display (top left)
    if (uv.x > 0.02 && uv.x < 0.2 && uv.y > 0.02 && uv.y < 0.08) {
        // Readout background
        col = float4(col.rgb + float3(0.0, 0.1, 0.2) * 0.1, col.a);
        
        // Animated bars
        let bar_y = (uv.y - 0.02) / 0.06;
        let bar_x = (uv.x - 0.02) / 0.18;
        let bar_val = sin(bar_x * 10.0 + t * 3.0) * 0.5 + 0.5;
        
        if (bar_y < bar_val && fract(bar_x * 10.0) > 0.2) {
            col = float4(col.rgb + float3(0.0, 1.0, 0.5) * 0.3, col.a);
        }
    }
    
    // === END FOREGROUND CYBER === //
    
    textureStore(screen, int2(id.xy), float4(col.rgb, col.a));
}
