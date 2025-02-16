struct LBMFields {
    // Density fields
    rho_surf: array<array<f32, SCREEN_HEIGHT>, SCREEN_WIDTH>,
    rho_old: array<array<f32, SCREEN_HEIGHT>, SCREEN_WIDTH>,
    rho_new: array<array<f32, SCREEN_HEIGHT>, SCREEN_WIDTH>,
    
    // Pigment fields
    pigment_surf: array<array<f32, SCREEN_HEIGHT>, SCREEN_WIDTH>,
    pigment_old: array<array<f32, SCREEN_HEIGHT>, SCREEN_WIDTH>,
    pigment_new: array<array<f32, SCREEN_HEIGHT>, SCREEN_WIDTH>,
    fixture: array<array<f32, SCREEN_HEIGHT>, SCREEN_WIDTH>,
    
    // Pinning and permeability
    blk_txt: array<array<f32, SCREEN_HEIGHT>, SCREEN_WIDTH>,
    perm: array<array<array<f32, 9>, SCREEN_HEIGHT>, SCREEN_WIDTH>,
}

struct LBMFields2 {
    // LBM distribution functions
    f_old: array<array<array<f32, 9>, SCREEN_HEIGHT>, SCREEN_WIDTH>,
    f_new: array<array<array<f32, 9>, SCREEN_HEIGHT>, SCREEN_WIDTH>,
    vel: array<array<vec2<f32>, SCREEN_HEIGHT>, SCREEN_WIDTH>,
}

#storage fields LBMFields
#storage fields2 LBMFields2

// LBM weights and directions
const W = array<f32, 9>(
    4.0 / 9.0,  // center
    1.0 / 9.0,  // cardinal directions
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 36.0, // diagonal directions
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0
);

const E = array<vec2<i32>, 9>(
    vec2(0, 0),   // center
    vec2(1, 0),   // down
    vec2(0, 1),   // right
    vec2(-1, 0),  // up
    vec2(0, -1),  // left
    vec2(1, 1),   // down-right
    vec2(-1, 1),  // up-right
    vec2(-1, -1), // up-left
    vec2(1, -1)   // down-left
);

// Helper function to compute equilibrium distribution
fn f_eq(rho: f32, vel: vec2<f32>, k: i32) -> f32 {
   let eu = dot(vec2<f32>(f32(E[k].x), f32(E[k].y)), vel);
   let uv = dot(vel, vel);
   let advect = smoothstep(0.0, 0.8, rho);
   return W[k] * (rho + advect * (3.0 * eu + 4.5 * eu * eu - 1.5 * uv));
}

fn lerp(vl: f32, vr: f32, frac: f32) -> f32 {
   return vl + clamp(frac, 0.0, 1.0) * (vr - vl);
}

// Initialize simulation
#dispatch_once init1
@compute @workgroup_size(16, 16)
fn init1(@builtin(global_invocation_id) id: vec3<u32>) {
   let i = id.x;
   let j = id.y;
   if (i >= SCREEN_WIDTH || j >= SCREEN_HEIGHT) { return; }
   
   fields.rho_surf[i][j] = 2.5; // Dampen the page
   fields.rho_old[i][j] = 0.0;
   fields.rho_new[i][j] = 0.0;
   fields.pigment_surf[i][j] = 0.0;
   fields.pigment_old[i][j] = 0.0;
   fields.pigment_new[i][j] = 0.0;
   fields.fixture[i][j] = 0.0;
   fields2.vel[i][j] = vec2(0.0);
   fields.blk_txt[i][j] = 0.0;
}
#dispatch_once init2
@compute @workgroup_size(16, 16)
fn init2(@builtin(global_invocation_id) id: vec3<u32>) {
   let i = id.x;
   let j = id.y;
   if (i >= SCREEN_WIDTH || j >= SCREEN_HEIGHT) { return; }
   
   // Initialize distribution functions to equilibrium
   for (var k = 0; k < 9; k++) {
       let feq = f_eq(fields.rho_new[i][j], fields2.vel[i][j], k);
       fields2.f_new[i][j][k] = feq;
       fields2.f_old[i][j][k] = feq;
       fields.perm[i][j][k] = 1.0;
   }
}

// Update blocking/permeability based on paper texture
@compute @workgroup_size(16, 16)
fn update_block1(@builtin(global_invocation_id) id: vec3<u32>) {
   let i = id.x;
   let j = id.y;
   if (i == 0 || i >= SCREEN_WIDTH-1 || j == 0 || j >= SCREEN_HEIGHT-1) { return; }
   
   let xuan = textureSampleLevel(channel0, bilinear_repeat, vec2<f32>(f32(i) / 256, f32(j) / 256), 0.0).r;
   let lines = textureSampleLevel(channel1, bilinear_repeat, vec2<f32>(f32(i) / 156, f32(j) / 152), 0.0).r;
   
   // Check pinning conditions
   var pinning = fields.rho_new[i][j] == 0.0;
   let pinning_threshold = 0.05 + 0.2 * fields.fixture[i][j] + 0.3 * lines;
       
   // Check neighbors for pinning
   for (var k = 1; k < 5; k++) {
       let ip = i32(i) - E[k].x;
       let jp = i32(j) - E[k].y;
       if (ip >= 0 && ip < i32(SCREEN_WIDTH) && jp >= 0 && jp < i32(SCREEN_HEIGHT)) {
           pinning = pinning && fields2.f_old[u32(ip)][u32(jp)][k] < pinning_threshold;
       }
   }
   for (var k = 5; k < 9; k++) {
       let ip = i32(i) - E[k].x;
       let jp = i32(j) - E[k].y;
       if (ip >= 0 && ip < i32(SCREEN_WIDTH) && jp >= 0 && jp < i32(SCREEN_HEIGHT)) {
           pinning = pinning && fields2.f_old[u32(ip)][u32(jp)][k] < pinning_threshold * 1.414213562;
       }
   }
   
   if (pinning) {
       fields.blk_txt[i][j] = 100.0;
   } else {
       fields.blk_txt[i][j] = xuan + 0.3 * fields.fixture[i][j];
   }
}

@compute @workgroup_size(16, 16)
fn update_block2(@builtin(global_invocation_id) id: vec3<u32>) {
   let i = id.x;
   let j = id.y;
   if (i == 0 || i >= SCREEN_WIDTH-1 || j == 0 || j >= SCREEN_HEIGHT-1) { return; }

   // Update permeability
   for (var k = 0; k < 9; k++) {
       let ip = i32(i) - E[k].x;
       let jp = i32(j) - E[k].y;
       if (ip >= 0 && ip < i32(SCREEN_WIDTH) && jp >= 0 && jp < i32(SCREEN_HEIGHT)) {
           fields.perm[i][j][k] = 1.0 - clamp(0.5 * fields.blk_txt[u32(ip)][u32(jp)] + 
                                             0.5 * fields.blk_txt[i][j], 0.0, 1.0);
       }
   }
}

// Collision and streaming
@compute @workgroup_size(16, 16)
fn collide_and_stream1(@builtin(global_invocation_id) id: vec3<u32>) {
   let i = id.x;
   let j = id.y;
   if (i == 0 || i >= SCREEN_WIDTH-1 || j == 0 || j >= SCREEN_HEIGHT-1) { return; }
   
   // First collision step
   for (var k = 0; k < 9; k++) {
       let ip = i32(i) - E[k].x;
       let jp = i32(j) - E[k].y;
       if (ip >= 0 && ip < i32(SCREEN_WIDTH) && jp >= 0 && jp < i32(SCREEN_HEIGHT)) {
           fields2.f_new[i][j][k] = 0.5 * fields2.f_old[u32(ip)][u32(jp)][k] + 
                                  0.5 * f_eq(fields.rho_new[u32(ip)][u32(jp)], 
                                           fields2.vel[u32(ip)][u32(jp)], k);
       }
   }
}

@compute @workgroup_size(16, 16)
fn collide_and_stream2(@builtin(global_invocation_id) id: vec3<u32>) {
   let i = id.x;
   let j = id.y;
   if (i == 0 || i >= SCREEN_WIDTH-1 || j == 0 || j >= SCREEN_HEIGHT-1) { return; }
   
   // Copy to old
   for (var k = 0; k < 9; k++) {
       fields2.f_old[i][j][k] = fields2.f_new[i][j][k];
   }
}

@compute @workgroup_size(16, 16)
fn collide_and_stream3(@builtin(global_invocation_id) id: vec3<u32>) {
   let i = id.x;
   let j = id.y;
   if (i == 0 || i >= SCREEN_WIDTH-1 || j == 0 || j >= SCREEN_HEIGHT-1) { return; }
   
   // Streaming with permeability
   fields2.f_new[i][j][1] = lerp(fields2.f_old[i][j][3], 
                               fields2.f_old[i-1][j][1], fields.perm[i][j][1]);
   fields2.f_new[i][j][2] = lerp(fields2.f_old[i][j][4], 
                               fields2.f_old[i][j-1][2], fields.perm[i][j][2]);
   fields2.f_new[i][j][3] = lerp(fields2.f_old[i][j][1], 
                               fields2.f_old[i+1][j][3], fields.perm[i][j][3]);
   fields2.f_new[i][j][4] = lerp(fields2.f_old[i][j][2], 
                               fields2.f_old[i][j+1][4], fields.perm[i][j][4]);
   
   fields2.f_new[i][j][5] = lerp(fields2.f_old[i][j][7], 
                               fields2.f_old[i-1][j-1][5], fields.perm[i][j][5]);
   fields2.f_new[i][j][6] = lerp(fields2.f_old[i][j][8], 
                               fields2.f_old[i+1][j-1][6], fields.perm[i][j][6]);
   fields2.f_new[i][j][7] = lerp(fields2.f_old[i][j][5], 
                               fields2.f_old[i+1][j+1][7], fields.perm[i][j][7]);
   fields2.f_new[i][j][8] = lerp(fields2.f_old[i][j][6], 
                               fields2.f_old[i-1][j+1][8], fields.perm[i][j][8]);
   
   // Additional evaporation at pinning edges
   let evap_b = 0.00005 * 0.1;
   for (var k = 0; k < 9; k++) {
       if (fields.perm[i][j][k] > 0.999) {
           fields2.f_new[i][j][k] = max(0.0, fields2.f_new[i][j][k] - evap_b);
       }
   }
}

// Update macroscopic variables
@compute @workgroup_size(16, 16)
fn update_macro_var(@builtin(global_invocation_id) id: vec3<u32>) {
   let i = id.x;
   let j = id.y;
   if (i == 0 || i >= SCREEN_WIDTH-1 || j == 0 || j >= SCREEN_HEIGHT-1) { return; }
   
   fields.rho_old[i][j] = fields.rho_new[i][j];
   fields.rho_new[i][j] = 0.0;
   fields2.vel[i][j] = vec2(0.0);
   
   for (var k = 0; k < 9; k++) {
       fields2.f_old[i][j][k] = fields2.f_new[i][j][k];
       fields.rho_new[i][j] += fields2.f_new[i][j][k];
       fields2.vel[i][j] += vec2<f32>(f32(E[k].x), f32(E[k].y)) * fields2.f_new[i][j][k];
   }
   
   // Global evaporation
   let evap_s = 0.00015 * 0.1;
   fields.rho_new[i][j] = max(0.0, fields.rho_new[i][j] - evap_s);
}

// Handle seeping from surface
@compute @workgroup_size(16, 16)
fn seep(@builtin(global_invocation_id) id: vec3<u32>) {
   let i = id.x;
   let j = id.y;
   if (i == 0 || i >= SCREEN_WIDTH-1 || j == 0 || j >= SCREEN_HEIGHT-1) { return; }
   
   let rho_seep_amt = min(0.01, fields.rho_surf[i][j]);
   fields.rho_new[i][j] += rho_seep_amt;
   fields.rho_surf[i][j] -= rho_seep_amt;
   
   let pig_seep_amt = min(0.01, fields.pigment_surf[i][j]);
   fields.pigment_new[i][j] += pig_seep_amt;
   fields.pigment_surf[i][j] -= pig_seep_amt;
}

@compute @workgroup_size(16, 16)
fn deposit_pigment(@builtin(global_invocation_id) id: vec3<u32>) {
   let i = id.x;
   let j = id.y;
   if (i == 0 || i >= SCREEN_WIDTH-1 || j == 0 || j >= SCREEN_HEIGHT-1) { return; }
   
   // Handle deposition first
   let water_loss = fields.rho_old[i][j] - fields.rho_new[i][j];
   if (fields.rho_old[i][j] > 0.0 && water_loss > 0.0) {
       let fix_factor = water_loss / fields.rho_old[i][j];
       fields.fixture[i][j] += fix_factor * fields.pigment_new[i][j];
       fields.pigment_new[i][j] -= fix_factor * fields.pigment_new[i][j];
   }
}

@compute @workgroup_size(16, 16)
fn advect_pigment(@builtin(global_invocation_id) id: vec3<u32>) {
   let i = id.x;
   let j = id.y;
   if (i == 0 || i >= SCREEN_WIDTH-1 || j == 0 || j >= SCREEN_HEIGHT-1) { return; }
   
   // Then handle advection
   fields.pigment_old[i][j] = fields.pigment_new[i][j];
   if (fields.rho_new[i][j] <= 0.00001) {
       fields.pigment_new[i][j] = 0.0;
   } else if (fields.rho_old[i][j] <= 0.00001) {
       fields.pigment_new[i][j] = 0.0;
       for (var k = 1; k < 9; k++) {
           let ip = i32(i) - E[k].x;
           let jp = i32(j) - E[k].y;
           fields.pigment_new[i][j] += fields2.f_new[i][j][k] * fields.pigment_old[ip][jp];
       }
       fields.pigment_new[i][j] /= fields.rho_new[i][j];
   } else {
       // Semi-Lagrangian advection
       let pos = vec2<f32>(f32(i), f32(j)) - fields2.vel[i][j];
       let pos_i = floor(pos);
       let frac = pos - pos_i;
       let i0 = u32(pos_i.x);
       let j0 = u32(pos_i.y);
       
       if (i0 > 0 && i0 < SCREEN_WIDTH-1 && j0 > 0 && j0 < SCREEN_HEIGHT-1) {
           let p00 = fields.pigment_old[i0][j0];
           let p10 = fields.pigment_old[i0+1][j0];
           let p01 = fields.pigment_old[i0][j0+1];
           let p11 = fields.pigment_old[i0+1][j0+1];
           
           let p0 = mix(p00, p10, frac.x);
           let p1 = mix(p01, p11, frac.x);
           fields.pigment_new[i][j] = mix(p0, p1, frac.y);
       }
   }
}

// Handle mouse interaction for painting
@compute @workgroup_size(16, 16)
fn paint(@builtin(global_invocation_id) id: vec3<u32>) {
   let i = id.x;
   let j = id.y;
   if (i >= SCREEN_WIDTH || j >= SCREEN_HEIGHT) { return; }
   
   if (mouse.click == 1) {
       let mouse_pos = vec2<f32>(mouse.pos);
       let pos = vec2<f32>(f32(i), f32(j));
       let dist = distance(mouse_pos, pos);
       
       if (dist < custom.brush_size) {
           fields.rho_surf[i][j] = custom.paint_density;
           fields.pigment_surf[i][j] = custom.paint_pigment;
       }
   }
}

// Final rendering
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
   let pos = vec2<i32>(id.xy);
   if (pos.x >= SCREEN_WIDTH || pos.y >= SCREEN_HEIGHT) { return; }
   
   let i = u32(pos.x);
   let j = u32(pos.y);
   
   var color = vec4<f32>(
       exp(-4*(clamp(fields.pigment_new[i][j], 0, 1))),
       exp(-4*(clamp(fields.fixture[i][j], 0, 1))),
       exp(-4*(clamp(fields.pigment_surf[i][j], 0, 1))),
       1.0
   );
   
   textureStore(screen, pos, color);
}
