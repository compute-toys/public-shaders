// Adaptive Radiance Cascades Rasterization with Sky Integration

// implementation of 
// Radiance Cascades: A Novel Approach to Calculating Global Illumination
// https://drive.google.com/file/d/1L6v1_7HY2X-LV3Ofb6oyTIxgEaP4LOI6/view
// by Alexander Sannikov

// Based on the OG shadertoy implementation of Radiance Cascades by 
// fad: https://www.shadertoy.com/view/mtlBzX

// Sky integral formula taken from
// Analytic Direct Illumination - Mathis
// https://www.shadertoy.com/view/NttSW7

// This implementation provides radiance probe
// direct rasterization using per-pixel adaptive kernels.

#define MAX_UINT 0xFFFFFFFF
#define PI 3.14159265359
#define GRID_SIZE 1024
#define UNIT_SCALE 1e5
#define C0_LENGTH 0.5
#define NUM_CASCADES 5u
#define SUN_AND_SKY 0

struct radiance_buffer {
    r: atomic<u32>,
    g: atomic<u32>,
    b: atomic<u32>,
    a: atomic<u32>,
    z: atomic<u32>,
}

#storage CANVAS  array<vec4f, GRID_SIZE * GRID_SIZE>;
#storage FLATLAND  array<radiance_buffer, GRID_SIZE * GRID_SIZE * NUM_CASCADES>;

fn store_radiance(id: u32, new_value: vec3f, new_depth: f32) {
    let new_depth_u = u32(new_depth * UNIT_SCALE);
    let old_depth_u = atomicMin(&FLATLAND[id].z, new_depth_u);
    if (new_depth_u <= old_depth_u) 
    {
        atomicStore(&FLATLAND[id].r, u32(new_value.r * f32(UNIT_SCALE)));
        atomicStore(&FLATLAND[id].g, u32(new_value.g * f32(UNIT_SCALE)));
        atomicStore(&FLATLAND[id].b, u32(new_value.b * f32(UNIT_SCALE)));
        atomicStore(&FLATLAND[id].a, u32(UNIT_SCALE));
        atomicStore(&FLATLAND[id].z, new_depth_u);
    }
}

fn merge_radiance(id: u32, new_value: vec4f) {
    atomicStore(&FLATLAND[id].r, u32(new_value.r * f32(UNIT_SCALE)));
    atomicStore(&FLATLAND[id].g, u32(new_value.g * f32(UNIT_SCALE)));
    atomicStore(&FLATLAND[id].b, u32(new_value.b * f32(UNIT_SCALE)));
    atomicStore(&FLATLAND[id].a, u32(new_value.a * f32(UNIT_SCALE)));
}

fn reset_radiance(id: u32) {
    atomicStore(&FLATLAND[id].r, 0);
    atomicStore(&FLATLAND[id].g, 0);
    atomicStore(&FLATLAND[id].b, 0);
    atomicStore(&FLATLAND[id].a, 0);
    atomicStore(&FLATLAND[id].z, MAX_UINT);
}

fn get_radiance(id: u32) -> vec4f {
    var radiance = vec4f(
        f32(atomicLoad(&FLATLAND[id].r)),
        f32(atomicLoad(&FLATLAND[id].g)),
        f32(atomicLoad(&FLATLAND[id].b)),
        f32(atomicLoad(&FLATLAND[id].a))
    );
    return radiance / f32(UNIT_SCALE);
}

fn integrate_sky_radiance_(angle: vec2f) -> vec3f {
    // Sky radiance helper function
    let a1 = angle.y;
    let a0 = angle.x;
    let SkyColor = vec3f(0.2,0.5,1.);
    let SunColor = vec3f(1.,0.5,0.1) * 20.;
    let SunA = 5.0;
    let SunS = 64.0;
    let SSunS = sqrt(SunS);
    let ISSunS = 1./SSunS;
    var SI = SkyColor*(a1-a0-0.5*(cos(a1)-cos(a0)));
    SI += SunColor*(atan(SSunS*(SunA-a0))-atan(SSunS*(SunA-a1)))*ISSunS;
    return SI / 6.0; 
}

fn integrate_sky_radiance(angle: vec2f) -> vec3f {
    // Integrate the radiance from the sky over an interval of directions
    if (angle.y < 2.0 * PI) {
        return integrate_sky_radiance_(angle);
    }
    return
        integrate_sky_radiance_(vec2f(angle.x, 2.0 * PI)) +
        integrate_sky_radiance_(vec2f(0.0, angle.y - 2.0 * PI));
}

// Returns x offset in FLATLAND for cascade 'c' 
fn offset_cascade(c: u32) -> u32 {
    return c * GRID_SIZE * GRID_SIZE;
}

fn get_angle_num(c: u32) -> u32 {
    return 4u << (c * 2);
}

fn get_probe_size(c: u32) -> u32 {
    return 2u << c;
}

fn get_angle_id(v: vec2f, c: u32) -> u32 {
    let num_angles = get_angle_num(c);
    let normalized_v = normalize(v);
    var angle = atan2(normalized_v.y, normalized_v.x);
    angle = angle + (2.0 * PI) * step(0.0, -angle); 
    let normalized_angle = angle / (2.0 * PI);
    let id = u32(floor(normalized_angle * f32(num_angles)));
    return id;
}

fn get_direction_from_id(id: u32, c: u32) -> vec2f {
    let num_angles = get_angle_num(c);
    let angle = (f32(id) + 0.5) * (2.0 * PI) / f32(num_angles);
    return vec2f(cos(angle), sin(angle));
}

fn cascade_interval(base: f32, c: u32) -> f32 {
    return base * f32(1u << (2u * c));
}

fn flatten_2d(coord: vec2i, phi: u32, cascade: u32, grid_size: u32) -> u32 {
    let probe_size = get_probe_size(cascade);
    let phi_rel = vec2u(
        phi % probe_size,
        phi / probe_size);
    let phi_abs = vec2u(
       u32(coord.x) + phi_rel.x,
       u32(coord.y) + phi_rel.y,
    );
    return u32(phi_abs.y * grid_size + phi_abs.x);
}

fn unflatten_2d(pixel_coord: vec2u, cascade: u32, grid_size: u32) -> u32 {
    let probe_size = get_probe_size(cascade);
    let phi_rel = vec2u(
        pixel_coord.x % probe_size,
        pixel_coord.y % probe_size
    );
    let phi = phi_rel.y * probe_size + phi_rel.x;
    return phi;
}

fn sample_cascade(cascade: u32, coord: vec2i, phi: u32) -> vec4f {
    return get_radiance(offset_cascade(cascade) + flatten_2d(coord, phi, cascade, GRID_SIZE));
}

#workgroup_count paint 64 64 1
@compute @workgroup_size(16, 16)
fn paint(@builtin(global_invocation_id) id: vec3u) {     
   //reset cache 
    let r_id = (id.y * GRID_SIZE + id.x);
    for (var i: u32 = 0; i < NUM_CASCADES; i++) { 
         reset_radiance(r_id + offset_cascade(i));
    }
   
    let size = custom.brush_size;
    let dist = distance(vec2f(id.xy), vec2f(mouse.pos.xy));
    let t = f32(time.frame) / 40.0;
    if (dist < size && mouse.click > 0) { // inside
        var cr = vec4f(sin(t), cos(t), sin(t + PI), 1.0);
        var v = custom.value;
        cr = vec4f(cr.rgb * v, 1.0);
        CANVAS[id.y * GRID_SIZE  + id.x] = cr;
    }
}

#workgroup_count radiance 64 64 NUM_CASCADES
@compute @workgroup_size(16, 16)
fn radiance(@builtin(global_invocation_id) id: vec3u) {  
    let pixel = CANVAS[id.y * GRID_SIZE  + id.x];
    let position = vec2f(id.xy) + 0.5;

    if (pixel.a != 0) 
    {
        let cascade = id.z;
        let kernel_size = i32(u32(round(C0_LENGTH * 2)) << cascade);
        let probe_size = get_probe_size(cascade);
        let interval_start = cascade_interval(C0_LENGTH, cascade);
        let interval_end = cascade_interval(C0_LENGTH, cascade + 1);
        var base_cell_pos = vec2i(floor(position / f32(probe_size)) * f32(probe_size));

        for (var x: i32 = -kernel_size; x <= kernel_size; x++) {
            for (var y: i32 = -kernel_size; y <= kernel_size; y++) {
                let offset = vec2i(x, y);
                var cell_pos = base_cell_pos + offset * i32(probe_size);
                if (cell_pos.x < 0 || cell_pos.x >= i32(GRID_SIZE) || cell_pos.y < 0 || cell_pos.y >= i32(GRID_SIZE)) {
                    continue;  // Skip out-of-bounds cells
                }
                let cell_center = vec2f(cell_pos) + vec2f(0.5f * f32(probe_size));
                let direction = normalize(position - cell_center);
                let phi = get_angle_id(direction, cascade);
                let dist = distance(position, cell_center);
                if (dist > interval_start && dist < interval_end) 
                { 
                    let flat_coords = offset_cascade(cascade) + flatten_2d(cell_pos, phi, cascade, GRID_SIZE);
                    store_radiance(flat_coords, pixel.rgb, dist);
                }
            }
        }  
    }
}

#dispatch_count merge NUM_CASCADES
#workgroup_count merge 64 64 1
@compute @workgroup_size(16, 16)
fn merge(@builtin(global_invocation_id) id: vec3u) {
    let position = vec2f(id.xy) + 0.5;
    let r_id = (id.y * GRID_SIZE + id.x);
    
    let cascade = NUM_CASCADES - dispatch.id - 1;

    let phi = unflatten_2d(id.xy, cascade, GRID_SIZE);
    let phi_num = get_angle_num(cascade);
    
    let probe_size = get_probe_size(cascade);
    let next_probe_size = get_probe_size(cascade + 1);
    
    let base_cell_pos = vec2i(floor(position / f32(probe_size)) * f32(probe_size));
    let base_cell_center = vec2f(base_cell_pos) + vec2f(0.5f * f32(probe_size));
    
    var s = vec4f(0.0);
    let num_samples = 4u;
    var si = get_radiance(r_id + offset_cascade(cascade));
    
    if (si.a == 0) {
        for (var i = 0u; i < num_samples; i++) {
            if (cascade == NUM_CASCADES - 1) { 
                 if(bool(SUN_AND_SKY)) {
                    // If we are the top-level cascade, then there's no other
                    // cascade to merge with, so instead merge with the sky radiance
                    let angle = vec2f(f32(i + phi), f32(i + phi + 1)) / f32(phi_num) * 2.0 * PI;
                    let sky_radiance = integrate_sky_radiance(angle);
                    let normalized_radiance = sky_radiance / (angle.y - angle.x);
                    si += vec4f(normalized_radiance, 0);
                }
            }
            else {
                let shift_pos = vec2f(base_cell_pos) - 1.0;
                let next_top_left_pos = vec2i(floor(shift_pos / f32(next_probe_size)) * f32(next_probe_size));
                let next_top_left_center = vec2f(next_top_left_pos) + 0.5f * vec2f(f32(next_probe_size));
                let fraction = abs(base_cell_center - next_top_left_center) / f32(next_probe_size);
                let phi_next = phi * 4u + i;
                let S00 = sample_cascade(cascade + 1, next_top_left_pos + vec2i(0, 0) * i32(next_probe_size), phi_next);
                let S10 = sample_cascade(cascade + 1, next_top_left_pos + vec2i(1, 0) * i32(next_probe_size), phi_next);
                let S01 = sample_cascade(cascade + 1, next_top_left_pos + vec2i(0, 1) * i32(next_probe_size), phi_next);
                let S11 = sample_cascade(cascade + 1, next_top_left_pos + vec2i(1, 1) * i32(next_probe_size), phi_next);
                
                s += mix(
                    mix(S00, S10, fraction.x),
                    mix(S01, S11, fraction.x),
                    fraction.y
                );
                }
            }
        si += s;
    }
    si /= f32(num_samples);
    
    merge_radiance(r_id + offset_cascade(cascade), si);
}

#workgroup_count display 64 64 1
@compute @workgroup_size(16, 16)
fn display(@builtin(global_invocation_id) id: vec3u) {
    let position = vec2f(id.xy) + 0.5;
    var total_radiance = vec3f(0.0, 0.0, 0.0);
    var total_weight = 0.0;

    let cascade: u32 = 0;
    let probe_size = get_probe_size(cascade);
    let cell_pos = vec2i(floor(position / f32(probe_size)) * f32(probe_size));
            
    let cell_center = vec2f(cell_pos) + vec2f(0.5f * f32(probe_size));
    let phi_num = get_angle_num(cascade);
    
    let fraction = (position - vec2f(cell_pos)) / f32(probe_size);

    for (var phi: u32 = 0; phi < phi_num; phi++) {
        let S00 = sample_cascade(cascade, cell_pos + vec2i(0, 0) * i32(probe_size), phi);
        let S10 = sample_cascade(cascade, cell_pos + vec2i(1, 0) * i32(probe_size), phi);
        let S01 = sample_cascade(cascade, cell_pos + vec2i(0, 1) * i32(probe_size), phi);
        let S11 = sample_cascade(cascade, cell_pos + vec2i(1, 1) * i32(probe_size), phi);
        
        // Bilinear interpolation
        let interpolated = mix(
            mix(S00, S10, fraction.x),
            mix(S01, S11, fraction.x),
            fraction.y
        );
        
        total_radiance += interpolated.rgb;
        total_weight += 1.0;        
    }

    if (total_weight > 0.0) {
        total_radiance /= total_weight;
    }

    let c_id = (id.y * GRID_SIZE + id.x);
    var out_color = CANVAS[c_id] + vec4f(total_radiance * 2 * PI, 1.0);
    out_color = vec4f(1.0 - 1.0 / pow(1.0 + out_color.rgb, vec3f(2.5)), 1.0);

    textureStore(screen, id.xy, out_color);
}