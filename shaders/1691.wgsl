/* References:
 * https://hal.science/hal-01791898/file/hthincBVD_final.pdf
 * https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/euler.py
 */

#define GRID_WIDTH min(SCREEN_WIDTH, 1280)
#define GRID_HEIGHT min(SCREEN_HEIGHT, 720)

struct SimFields {
    // Conservative variables [rho, rho*u, rho*v, rho*E]
    Q: array<array<vec4<f32>, GRID_HEIGHT>, GRID_WIDTH>,
    Q_old: array<array<vec4<f32>, GRID_HEIGHT>, GRID_WIDTH>,
    
    // Primitive variables [rho, u, v, p]
    W: array<array<vec4<f32>, GRID_HEIGHT>, GRID_WIDTH>,
    
    // Fluxes
    F_x: array<array<vec4<f32>, GRID_HEIGHT>, GRID_WIDTH>,
    F_y: array<array<vec4<f32>, GRID_HEIGHT>, GRID_WIDTH>,
    
    // Adaptive timestep
    dt_atomic: atomic<u32>,
    dt: f32,

    // Stuff this field in here to balance the storage buffer sizes
    W_xl: array<array<array<vec4<f32>, 3>, GRID_HEIGHT>, GRID_WIDTH>,
}

// Face reconstructions for THINC
struct THINCFields {
    // W_xl: array<array<array<vec4<f32>, 3>, GRID_HEIGHT>, GRID_WIDTH>,
    W_xr: array<array<array<vec4<f32>, 3>, GRID_HEIGHT>, GRID_WIDTH>,
    W_yl: array<array<array<vec4<f32>, 3>, GRID_HEIGHT>, GRID_WIDTH>,
    W_yr: array<array<array<vec4<f32>, 3>, GRID_HEIGHT>, GRID_WIDTH>,
}

#storage fields SimFields
#storage thinc_fields THINCFields

#define config custom

// Utility functions
fn is_interior_cell(i: u32, j: u32) -> bool {
    return i > 0u && i < GRID_WIDTH - 1u && 
           j > 0u && j < GRID_HEIGHT - 1u;
}

fn is_interior_x_face(i: u32, j: u32) -> bool {
    return i > 1u && i < GRID_WIDTH - 1u && 
           j > 0u && j < GRID_HEIGHT - 1u;
}

fn is_boundary_x_face(i: u32, j: u32) -> bool {
    return (i == 1u || i == GRID_WIDTH - 1u) && 
           j > 0u && j < GRID_HEIGHT - 1u;
}

fn is_interior_y_face(i: u32, j: u32) -> bool {
    return i > 0u && i < GRID_WIDTH - 1u && 
           j > 1u && j < GRID_HEIGHT - 1u;
}

fn is_boundary_y_face(i: u32, j: u32) -> bool {
    return i > 0u && i < GRID_WIDTH - 1u && 
           (j == 1u || j == GRID_HEIGHT - 1u);
}

fn w_to_q(w: vec4<f32>) -> vec4<f32> {
    let rho = w.x;
    let u = w.y;
    let v = w.z;
    let p = w.w;
    
    let kinetic = 0.5 * rho * (u * u + v * v);
    let internal = p / (config.gamma - 1.0);
    let E = kinetic + internal;
    
    return vec4<f32>(rho, rho * u, rho * v, E);
}

fn q_to_w(q: vec4<f32>) -> vec4<f32> {
    let rho = q.x;
    let u = q.y / rho;
    let v = q.z / rho;

    // Energy calculations
    let E = q.w;
    let kinetic = 0.5 * (u * u + v * v) * rho;
    let internal = E - kinetic;
    let p = (config.gamma - 1.0) * internal;
    
    return vec4<f32>(rho, u, v, p);
}

fn mc_lim(r: f32) -> f32 {
    return max(0.0, min(2.0 * r, min(0.5 * (r + 1.0), 2.0)));
}

fn thinc(wl: f32, wc: f32, wr: f32, beta: f32) -> vec2<f32> {
    var w0 = wc;
    var w1 = wc;
    
    if ((wr - wc) * (wc - wl) > 0.0) {
        let wmin = min(wr, wl);
        let wmax = max(wr, wl);
        let wdelta = wmax - wmin + 1e-6;  // Avoid pure zero
        let theta = sign(wr - wl);
        
        let C = (wc - wmin + 1e-6) / wdelta;
        
        let B = exp(theta * beta * (2.0 * C - 1.0));
        
        let A = (B / cosh(beta) - 1.0) / tanh(beta);
        
        w0 = wmin + wdelta * 0.5 * (1.0 + theta * A);
        w1 = wmin + wdelta * 0.5 * (1.0 + theta * (tanh(beta) + A) / (1.0 + A * tanh(beta)));
    }
    
    return vec2<f32>(w0, w1);
}

fn HLLC_flux(qL: vec4<f32>, qR: vec4<f32>, n: vec2<f32>) -> vec4<f32> {
    let nx = n.x;
    let ny = n.y;
    
    // Left state
    let rL = qL.x;
    let uL = qL.y / qL.x;
    let vL = qL.z / qL.x;
    let pL = (config.gamma - 1.0) * (qL.w - 0.5 * (qL.y * qL.y + qL.z * qL.z) / qL.x);
    let vnL = uL * nx + vL * ny;
    let vtL = -uL * ny + vL * nx;
    let aL = sqrt(config.gamma * pL / rL);
    let HL = (qL.w + pL) / rL;
    
    // Right state
    let rR = qR.x;
    let uR = qR.y / qR.x;
    let vR = qR.z / qR.x;
    let pR = (config.gamma - 1.0) * (qR.w - 0.5 * (qR.y * qR.y + qR.z * qR.z) / qR.x);
    let vnR = uR * nx + vR * ny;
    let vtR = -uR * ny + vR * nx;
    let aR = sqrt(config.gamma * pR / rR);
    let HR = (qR.w + pR) / rR;
    
    // Roe averages
    let rt = sqrt(rR / rL);
    let u = (uL + rt * uR) / (1.0 + rt);
    let v = (vL + rt * vR) / (1.0 + rt);
    let H = (HL + rt * HR) / (1.0 + rt);
    let a = sqrt((config.gamma - 1.0) * (H - (u * u + v * v) / 2.0));
    let vn = u * nx + v * ny;
    
    // Wave speeds
    let sL = min(vnL - aL, vn - a);
    let sR = max(vnR + aR, vn + a);
    let sM = (pL - pR + rR * vnR * (sR - vnR) - rL * vnL * (sL - vnL)) / 
             (rR * (sR - vnR) - rL * (sL - vnL));
    
    // HLLC flux
    var flux = vec4<f32>(0.0);
    
    if (sL >= 0.0) {
        let fL = vec4<f32>(
            rL * vnL,
            rL * vnL * uL + pL * nx,
            rL * vnL * vL + pL * ny,
            rL * vnL * HL
        );
        flux = fL;
    } else if (sM >= 0.0) {
        let fL = vec4<f32>(
            rL * vnL,
            rL * vnL * uL + pL * nx,
            rL * vnL * vL + pL * ny,
            rL * vnL * HL
        );
        let qsL = rL * (sL - vnL) / (sL - sM) * vec4<f32>(
            1.0,
            sM * nx - vtL * ny,
            sM * ny + vtL * nx,
            qL.w / rL + (sM - vnL) * (sM + pL / (rL * (sL - vnL)))
        );
        flux = fL + sL * (qsL - qL);
    } else if (sR >= 0.0) {
        let fR = vec4<f32>(
            rR * vnR,
            rR * vnR * uR + pR * nx,
            rR * vnR * vR + pR * ny,
            rR * vnR * HR
        );
        let qsR = rR * (sR - vnR) / (sR - sM) * vec4<f32>(
            1.0,
            sM * nx - vtR * ny,
            sM * ny + vtR * nx,
            qR.w / rR + (sM - vnR) * (sM + pR / (rR * (sR - vnR)))
        );
        flux = fR + sR * (qsR - qR);
    } else {
        let fR = vec4<f32>(
            rR * vnR,
            rR * vnR * uR + pR * nx,
            rR * vnR * vR + pR * ny,
            rR * vnR * HR
        );
        flux = fR;
    }
    
    return flux;
}

fn backup_state_for_cell(i: u32, j: u32) {
    fields.Q_old[i][j] = fields.Q[i][j];
}

fn compute_primitives_for_cell(i: u32, j: u32) {
    fields.W[i][j] = q_to_w(fields.Q[i][j]);
}

fn compute_thinc_reconstruction_for_cell(i: u32, j: u32) {
    if (!is_interior_cell(i, j)) { return; }
    
    for (var f = 0; f < 4; f++) {
        // X-direction reconstruction
        let x_smooth = thinc(
            fields.W[i-1u][j][f],
            fields.W[i][j][f],
            fields.W[i+1u][j][f],
            1.2
        );
        thinc_fields.W_xr[i][j][0][f] = x_smooth.x;
        fields.W_xl[i+1u][j][0][f] = x_smooth.y;

        let x_sharp = thinc(
            fields.W[i-1u][j][f],
            fields.W[i][j][f],
            fields.W[i+1u][j][f],
            2.0
        );
        thinc_fields.W_xr[i][j][1][f] = x_sharp.x;
        fields.W_xl[i+1u][j][1][f] = x_sharp.y;

        // Y-direction reconstruction
        let y_smooth = thinc(
            fields.W[i][j-1u][f],
            fields.W[i][j][f],
            fields.W[i][j+1u][f],
            1.2
        );
        thinc_fields.W_yr[i][j][0][f] = y_smooth.x;
        thinc_fields.W_yl[i][j+1u][0][f] = y_smooth.y;

        let y_sharp = thinc(
            fields.W[i][j-1u][f],
            fields.W[i][j][f],
            fields.W[i][j+1u][f],
            2.0
        );
        thinc_fields.W_yr[i][j][1][f] = y_sharp.x;
        thinc_fields.W_yl[i][j+1u][1][f] = y_sharp.y;
    }
}

fn compute_thinc_bvd_for_cell(i: u32, j: u32) {
    if (!is_interior_cell(i, j)) { return; }

    for (var f = 0; f < 4; f++) {
        // X-direction BVD
        let x_TBV_smooth = abs(fields.W_xl[i][j][0][f] - thinc_fields.W_xr[i][j][0][f]) + 
                          abs(fields.W_xl[i+1u][j][0][f] - thinc_fields.W_xr[i+1u][j][0][f]);
        let x_TBV_sharp = abs(fields.W_xl[i][j][1][f] - thinc_fields.W_xr[i][j][1][f]) + 
                         abs(fields.W_xl[i+1u][j][1][f] - thinc_fields.W_xr[i+1u][j][1][f]);

        if (x_TBV_smooth < x_TBV_sharp) {
            thinc_fields.W_xr[i][j][2][f] = thinc_fields.W_xr[i][j][0][f];
            fields.W_xl[i+1u][j][2][f] = fields.W_xl[i+1u][j][0][f];
        } else {
            thinc_fields.W_xr[i][j][2][f] = thinc_fields.W_xr[i][j][1][f];
            fields.W_xl[i+1u][j][2][f] = fields.W_xl[i+1u][j][1][f];
        }

        // Y-direction BVD
        let y_TBV_smooth = abs(thinc_fields.W_yl[i][j][0][f] - thinc_fields.W_yr[i][j][0][f]) + 
                          abs(thinc_fields.W_yl[i][j+1u][0][f] - thinc_fields.W_yr[i][j+1u][0][f]);
        let y_TBV_sharp = abs(thinc_fields.W_yl[i][j][1][f] - thinc_fields.W_yr[i][j][1][f]) + 
                         abs(thinc_fields.W_yl[i][j+1u][1][f] - thinc_fields.W_yr[i][j+1u][1][f]);

        if (y_TBV_smooth < y_TBV_sharp) {
            thinc_fields.W_yr[i][j][2][f] = thinc_fields.W_yr[i][j][0][f];
            thinc_fields.W_yl[i][j+1u][2][f] = thinc_fields.W_yl[i][j+1u][0][f];
        } else {
            thinc_fields.W_yr[i][j][2][f] = thinc_fields.W_yr[i][j][1][f];
            thinc_fields.W_yl[i][j+1u][2][f] = thinc_fields.W_yl[i][j+1u][1][f];
        }
    }
}

fn compute_thinc_fluxes_for_cell(i: u32, j: u32) {
    if (is_interior_x_face(i, j)) {
        fields.F_x[i][j] = HLLC_flux(
            w_to_q(fields.W_xl[i][j][2]),
            w_to_q(thinc_fields.W_xr[i][j][2]),
            vec2<f32>(1.0, 0.0)
        );
    } else if (is_boundary_x_face(i, j)) {
        fields.F_x[i][j] = HLLC_flux(
            fields.Q[i-1u][j],
            fields.Q[i][j],
            vec2<f32>(1.0, 0.0)
        );
    }

    if (is_interior_y_face(i, j)) {
        fields.F_y[i][j] = HLLC_flux(
            w_to_q(thinc_fields.W_yl[i][j][2]),
            w_to_q(thinc_fields.W_yr[i][j][2]),
            vec2<f32>(0.0, 1.0)
        );
    } else if (is_boundary_y_face(i, j)) {
        fields.F_y[i][j] = HLLC_flux(
            fields.Q[i][j-1u],
            fields.Q[i][j],
            vec2<f32>(0.0, 1.0)
        );
    }
}

fn update_solution_for_cell(i: u32, j: u32, is_final: bool) {
    if (!is_interior_cell(i, j)) { return; }
    
    let h = 1.0 / f32(min(GRID_WIDTH, GRID_HEIGHT) - 2u);
    
    if (is_final) {
        fields.Q[i][j] = (fields.Q[i][j] + fields.Q_old[i][j]) / 2.0 + 
            fields.dt * (
                fields.F_x[i][j] - fields.F_x[i+1u][j] +
                fields.F_y[i][j] - fields.F_y[i][j+1u]
            ) / h;
    } else {
        fields.Q[i][j] = fields.Q[i][j] + 
            fields.dt * (
                fields.F_x[i][j] - fields.F_x[i+1u][j] +
                fields.F_y[i][j] - fields.F_y[i][j+1u]
            ) / h;
    }
}

fn apply_boundary_conditions_for_cell(i: u32, j: u32) {
    if (is_interior_cell(i, j)) { return; }
    
    // Wall boundary conditions
    if (i == 0u) {
        fields.Q[i][j] = fields.Q[i + 1u][j];
        fields.Q[i][j].y = -fields.Q[i + 1u][j].y;  // Reflect x-velocity
    }
    if (i == GRID_WIDTH - 1u) {
        fields.Q[i][j] = fields.Q[i - 1u][j];
        fields.Q[i][j].y = -fields.Q[i - 1u][j].y;
    }
    if (j == 0u) {
        fields.Q[i][j] = fields.Q[i][j + 1u];
        fields.Q[i][j].z = -fields.Q[i][j + 1u].z;  // Reflect y-velocity
    }
    if (j == GRID_HEIGHT - 1u) {
        fields.Q[i][j] = fields.Q[i][j - 1u];
        fields.Q[i][j].z = -fields.Q[i][j - 1u].z;
    }
}


/// Entrypoints ///

#dispatch_once init
@compute @workgroup_size(8, 8)
fn init(@builtin(global_invocation_id) id: vec3<u32>, 
        @builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    
    if (i >= GRID_WIDTH || j >= GRID_HEIGHT) { return; }
    
    let center = vec2<f32>(f32(GRID_WIDTH) / 3.0, f32(GRID_HEIGHT) / 3.0);
    let pos = vec2<f32>(f32(i), f32(j));
    let dist = distance(pos, center);
    let radius = 0.25 * f32(min(GRID_WIDTH, GRID_HEIGHT));
    
    let rho = select(0.125, 10.0, dist < radius);
    let u = 0.0;
    let v = 0.0;
    let p = select(0.1, 10.0, dist < radius);
    
    let kinetic = 0.5 * rho * (u * u + v * v);
    let internal = p / (config.gamma - 1.0);
    let E = kinetic + internal;
    
    let q = vec4<f32>(rho, rho * u, rho * v, E);
    
    fields.Q[i][j] = q;
    fields.Q_old[i][j] = q;
}

@compute @workgroup_size(1)
#workgroup_count set_dt_atomic 1 1 1
fn set_dt_atomic(@builtin(global_invocation_id) id: vec3<u32>) {
    atomicStore(&fields.dt_atomic, u32(1e9));
}

@compute @workgroup_size(8, 8)
fn calc_dt(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    if (i >= GRID_WIDTH || j >= GRID_HEIGHT) { return; }
    
    if (is_interior_cell(i, j)) {
        let w = q_to_w(fields.Q[i][j]);
        let a = sqrt(config.gamma * w.w / w.x);  // sound speed
        let vel = sqrt(w.y * w.y + w.z * w.z);  // velocity magnitude
        let ws = a + vel;  // characteristic speed
        
        let h = 1.0 / f32(min(GRID_WIDTH, GRID_HEIGHT) - 2u);
        let dt_local = config.CFL * h / ws / 2.0;
        
        atomicMin(&fields.dt_atomic, u32(dt_local * 1e6));
    }
}

@compute @workgroup_size(1)
#workgroup_count set_dt 1 1 1
fn set_dt(@builtin(global_invocation_id) id: vec3<u32>) {
    fields.dt = f32(atomicLoad(&fields.dt_atomic)) / 1e6;
}

@compute @workgroup_size(8, 8)
fn backup_state(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    if (i >= GRID_WIDTH || j >= GRID_HEIGHT) { return; }
    backup_state_for_cell(i, j);
}

// First stage computations
@compute @workgroup_size(8, 8)
fn compute_primitives_1(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    if (i >= GRID_WIDTH || j >= GRID_HEIGHT) { return; }
    compute_primitives_for_cell(i, j);
}

@compute @workgroup_size(8, 8)
fn compute_thinc_reconstruction_1(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    if (i >= GRID_WIDTH || j >= GRID_HEIGHT) { return; }
    compute_thinc_reconstruction_for_cell(i, j);
}

@compute @workgroup_size(8, 8)
fn compute_thinc_bvd_1(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    if (i >= GRID_WIDTH || j >= GRID_HEIGHT) { return; }
    compute_thinc_bvd_for_cell(i, j);
}

@compute @workgroup_size(8, 8)
fn compute_thinc_fluxes_1(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    if (i >= GRID_WIDTH || j >= GRID_HEIGHT) { return; }
    compute_thinc_fluxes_for_cell(i, j);
}

@compute @workgroup_size(8, 8)
fn update_solution_1(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    if (i >= GRID_WIDTH || j >= GRID_HEIGHT) { return; }
    update_solution_for_cell(i, j, false);
}

// Second stage computations
@compute @workgroup_size(8, 8)
fn compute_primitives_2(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    if (i >= GRID_WIDTH || j >= GRID_HEIGHT) { return; }
    compute_primitives_for_cell(i, j);
}

@compute @workgroup_size(8, 8)
fn compute_thinc_reconstruction_2(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    if (i >= GRID_WIDTH || j >= GRID_HEIGHT) { return; }
    compute_thinc_reconstruction_for_cell(i, j);
}

@compute @workgroup_size(8, 8)
fn compute_thinc_bvd_2(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    if (i >= GRID_WIDTH || j >= GRID_HEIGHT) { return; }
    compute_thinc_bvd_for_cell(i, j);
}

@compute @workgroup_size(8, 8)
fn compute_thinc_fluxes_2(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    if (i >= GRID_WIDTH || j >= GRID_HEIGHT) { return; }
    compute_thinc_fluxes_for_cell(i, j);
}

@compute @workgroup_size(8, 8)
fn update_solution_2(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    if (i >= GRID_WIDTH || j >= GRID_HEIGHT) { return; }
    update_solution_for_cell(i, j, true);
}

@compute @workgroup_size(8, 8)
fn apply_boundary_conditions(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    if (i >= GRID_WIDTH || j >= GRID_HEIGHT) { return; }
    apply_boundary_conditions_for_cell(i, j);
}

@compute @workgroup_size(8, 8)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.y > SCREEN_HEIGHT - 5) {
        // Red bar at bottom to show adaptive timestep
        textureStore(screen, id.xy, select(
            vec4f(0, 0, 0, 1), vec4f(1, 0, 0, 1),
            f32(id.x) < f32(GRID_WIDTH) * (fields.dt / 1e-3)));
        return;
    }

    let i = i32(id.x) - i32(SCREEN_WIDTH - GRID_WIDTH) / 2;
    let j = i32(id.y) - i32(SCREEN_HEIGHT - GRID_HEIGHT) / 2;
    if (i <= 0 || i >= GRID_WIDTH-1 || j <= 0 || j >= GRID_HEIGHT-1) {
        textureStore(screen, id.xy, vec4(1));
        return;
    }

    // Numerical schlieren (density gradient magnitude)
    let h = 1.0 / f32(min(GRID_WIDTH, GRID_HEIGHT) - 2);
    let schlieren = length(vec2(
        (fields.Q[i+1][j].x - fields.Q[i-1][j].x) / (2.0 * h),
        (fields.Q[i][j+1].x - fields.Q[i][j-1].x) / (2.0 * h),
    ));

    textureStore(screen, id.xy, vec4(exp(-schlieren / 30)));
}
