struct DendriteFields {
    phi: array<array<f32, SCREEN_HEIGHT>, SCREEN_WIDTH>,              // phase field
    temperature: array<array<f32, SCREEN_HEIGHT>, SCREEN_WIDTH>,      // temperature field
    phi_old: array<array<f32, SCREEN_HEIGHT>, SCREEN_WIDTH>,         
    temperature_old: array<array<f32, SCREEN_HEIGHT>, SCREEN_WIDTH>,
    dEnergy_dGrad_term1: array<array<vec2<f32>, SCREEN_HEIGHT>, SCREEN_WIDTH>,
    epsilons: array<array<f32, SCREEN_HEIGHT>, SCREEN_WIDTH>,         // anisotropic gradient energy coefficient
    phiRate: array<array<vec4<f32>, SCREEN_HEIGHT>, SCREEN_WIDTH>,    // RK4 rates for phi
    temperatureRate: array<array<vec4<f32>, SCREEN_HEIGHT>, SCREEN_WIDTH>, // RK4 rates for temperature
}

#storage fields DendriteFields

#define config custom

const RK4_DT_RATIOS = array<f32, 4>(0.0, 0.5, 0.5, 1.0);
const RK4_WEIGHTS = array<f32, 4>(0.166666667, 0.333333333, 0.333333333, 0.166666667);

#dispatch_once init
@compute @workgroup_size(8, 8)
fn init(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let j = global_id.y;
    if (i >= SCREEN_WIDTH || j >= SCREEN_HEIGHT) { return; }
    
    let center = vec2<f32>(SCREEN_WIDTH / 2.0, SCREEN_HEIGHT / 2.0);
    let pos = vec2<f32>(f32(i), f32(j));
    let radius = 4.0;
    
    // Initialize phi field based on distance from center
    if (distance(pos, center) < radius) {
        fields.phi[i][j] = 1.0;
    } else {
        fields.phi[i][j] = 0.0;
    }
    
    fields.phi_old[i][j] = fields.phi[i][j];
    fields.temperature[i][j] = 0.0;
    fields.temperature_old[i][j] = 0.0;
}

// Helper function to get neighboring indices with periodic boundary conditions
fn neighbor_index(i: u32, j: u32) -> vec4<u32> {
    let im = select(i - 1u, SCREEN_WIDTH - 1u, i == 0u);
    let jm = select(j - 1u, SCREEN_HEIGHT - 1u, j == 0u);
    let ip = select(i + 1u, 0u, i == SCREEN_WIDTH - 1u);
    let jp = select(j + 1u, 0u, j == SCREEN_HEIGHT - 1u);
    return vec4<u32>(im, jm, ip, jp);
}

// Helper function to calculate divergence of dEnergy_dGrad_term1
fn divergence_dEnergy_dGrad_term1(i: u32, j: u32) -> f32 {
    let neighbors = neighbor_index(i, j);
    let im = neighbors.x;
    let jm = neighbors.y;
    let ip = neighbors.z;
    let jp = neighbors.w;
    
    return (fields.dEnergy_dGrad_term1[ip][j].x - 
            fields.dEnergy_dGrad_term1[im][j].x) / (2.0 * config.dx) + 
           (fields.dEnergy_dGrad_term1[i][jp].y - 
            fields.dEnergy_dGrad_term1[i][jm].y) / (2.0 * config.dx);
}

// Compute rates for RK4 integration
fn rate_a(global_id: vec3<u32>, rk_loop: u32) {
    let i = global_id.x;
    let j = global_id.y;
    if (i >= SCREEN_WIDTH || j >= SCREEN_HEIGHT) { return; }
    
    let neighbors = neighbor_index(i, j);
    let im = neighbors.x;
    let jm = neighbors.y;
    let ip = neighbors.z;
    let jp = neighbors.w;
    
    let grad = vec2<f32>(
        (fields.phi[ip][j] - fields.phi[im][j]) / (2.0 * config.dx),
        (fields.phi[i][jp] - fields.phi[i][jm]) / (2.0 * config.dx)
    );
    
    let gradNorm = dot(grad, grad);
    
    if (gradNorm < 1.0e-8) {
        fields.dEnergy_dGrad_term1[i][j] = vec2<f32>(0.0);
        let angle = atan2(grad.y, grad.x);
        fields.epsilons[i][j] = config.grad_energy_coef * 
            (1.0 + config.aniso_magnitude * cos(floor(config.n_fold_symmetry) * (angle - config.angle0)));
    } else {
        let angle = atan2(grad.y, grad.x);
        let epsilon = config.grad_energy_coef * 
            (1.0 + config.aniso_magnitude * cos(floor(config.n_fold_symmetry) * (angle - config.angle0)));
        fields.epsilons[i][j] = epsilon;
        
        let dAngle_dGrad = vec2<f32>(-grad.y / gradNorm, grad.x / gradNorm);
        let tmp = config.grad_energy_coef * config.aniso_magnitude * 
                 -sin(floor(config.n_fold_symmetry) * (angle - config.angle0)) * 
                 floor(config.n_fold_symmetry);
        let depsilon_dGrad = tmp * dAngle_dGrad;
        fields.dEnergy_dGrad_term1[i][j] = epsilon * depsilon_dGrad * gradNorm;
    }
}

fn rate_b(global_id: vec3<u32>, rk_loop: u32) {
    let i = global_id.x;
    let j = global_id.y;
    if (i >= SCREEN_WIDTH || j >= SCREEN_HEIGHT) { return; }
    
    let neighbors = neighbor_index(i, j);
    let im = neighbors.x;
    let jm = neighbors.y;
    let ip = neighbors.z;
    let jp = neighbors.w;
    
    // Compute Laplacians
    let lapla_phi = (
        2.0 * (fields.phi[im][j] + fields.phi[i][jm] + 
               fields.phi[ip][j] + fields.phi[i][jp]) +
        (fields.phi[im][jm] + fields.phi[im][jp] + 
         fields.phi[ip][jm] + fields.phi[ip][jp]) -
        12.0 * fields.phi[i][j]
    ) / (3.0 * config.dx * config.dx);
    
    let lapla_tp = (
        2.0 * (fields.temperature[im][j] + fields.temperature[i][jm] + 
               fields.temperature[ip][j] + fields.temperature[i][jp]) +
        (fields.temperature[im][jm] + fields.temperature[im][jp] + 
         fields.temperature[ip][jm] + fields.temperature[ip][jp]) -
        12.0 * fields.temperature[i][j]
    ) / (3.0 * config.dx * config.dx);
    
    // Compute rates
    let m_chem = config.alpha * 
                 atan2(config.gamma * (config.temperature_equi - fields.temperature[i][j]), 1.0);
    let phi_current = fields.phi[i][j];
    let chemicalForce = phi_current * (1.0 - phi_current) * (phi_current - 0.5 + m_chem);
    let gradForce_term1 = divergence_dEnergy_dGrad_term1(i, j);
    
    let grad_epsilon2 = vec2<f32>(
        (pow(fields.epsilons[ip][j], 2.0) - 
         pow(fields.epsilons[im][j], 2.0)) / (2.0 * config.dx),
        (pow(fields.epsilons[i][jp], 2.0) - 
         pow(fields.epsilons[i][jm], 2.0)) / (2.0 * config.dx)
    );
    
    let grad = vec2<f32>(
        (fields.phi[ip][j] - fields.phi[im][j]) / (2.0 * config.dx),
        (fields.phi[i][jp] - fields.phi[i][jm]) / (2.0 * config.dx)
    );

    let gradForce_term2 = dot(grad_epsilon2, grad) + 
                         pow(fields.epsilons[i][j], 2.0) * lapla_phi;
    
    let phi_rate = config.mobility * 1e5 * (chemicalForce + gradForce_term1 + gradForce_term2);
    fields.phiRate[i][j][rk_loop] = phi_rate;
    fields.temperatureRate[i][j][rk_loop] = lapla_tp + config.latent_heat_coef * phi_rate;
}

fn rk4_intermediate_update(global_id: vec3<u32>, rk_loop: u32) {
    let i = global_id.x;
    let j = global_id.y;
    if (i >= SCREEN_WIDTH || j >= SCREEN_HEIGHT) { return; }
    
    fields.phi[i][j] = fields.phi_old[i][j] + 
                             RK4_DT_RATIOS[rk_loop] * config.dt * 1e-3 * 
                             fields.phiRate[i][j][rk_loop - 1u];
    fields.temperature[i][j] = fields.temperature_old[i][j] + 
                                    RK4_DT_RATIOS[rk_loop] * config.dt * 1e-3 * 
                                    fields.temperatureRate[i][j][rk_loop - 1u];
}

@compute @workgroup_size(16, 16)
fn rate0a(@builtin(global_invocation_id) global_id: vec3<u32>) {
    rate_a(global_id, 0);
}

@compute @workgroup_size(16, 16)
fn rate0b(@builtin(global_invocation_id) global_id: vec3<u32>) {
    rate_b(global_id, 0);
}

@compute @workgroup_size(16, 16)
fn rk4_intermediate_update_1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    rk4_intermediate_update(global_id, 1);
}

@compute @workgroup_size(16, 16)
fn rate1a(@builtin(global_invocation_id) global_id: vec3<u32>) {
    rate_a(global_id, 1);
}

@compute @workgroup_size(16, 16)
fn rate1b(@builtin(global_invocation_id) global_id: vec3<u32>) {
    rate_b(global_id, 1);
}

@compute @workgroup_size(16, 16)
fn rk4_intermediate_update_2(@builtin(global_invocation_id) global_id: vec3<u32>) {
    rk4_intermediate_update(global_id, 2);
}

@compute @workgroup_size(16, 16)
fn rate2a(@builtin(global_invocation_id) global_id: vec3<u32>) {
    rate_a(global_id, 2);
}

@compute @workgroup_size(16, 16)
fn rate2b(@builtin(global_invocation_id) global_id: vec3<u32>) {
    rate_b(global_id, 2);
}

@compute @workgroup_size(16, 16)
fn rk4_intermediate_update_3(@builtin(global_invocation_id) global_id: vec3<u32>) {
    rk4_intermediate_update(global_id, 3);
}

@compute @workgroup_size(16, 16)
fn rate3a(@builtin(global_invocation_id) global_id: vec3<u32>) {
    rate_a(global_id, 3);
}

@compute @workgroup_size(16, 16)
fn rate3b(@builtin(global_invocation_id) global_id: vec3<u32>) {
    rate_b(global_id, 3);
}

@compute @workgroup_size(16, 16)
fn rk4_total_update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let j = global_id.y;
    if (i >= SCREEN_WIDTH || j >= SCREEN_HEIGHT) { return; }
    
    var phi_update = fields.phi_old[i][j];
    var temp_update = fields.temperature_old[i][j];
    
    for (var k = 0u; k < 4u; k++) {
        phi_update += RK4_WEIGHTS[k] * config.dt * 1e-3 * fields.phiRate[i][j][k];
        temp_update += RK4_WEIGHTS[k] * config.dt * 1e-3 * fields.temperatureRate[i][j][k];
    }
    
    fields.phi_old[i][j] = phi_update;
    fields.temperature_old[i][j] = temp_update;
    fields.phi[i][j] = phi_update;
    fields.temperature[i][j] = temp_update;
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) global_id: vec3u) {
    let i = global_id.x;
    let j = global_id.y;
    if (i >= SCREEN_WIDTH || j >= SCREEN_HEIGHT) { return; }

    var bg = vec3(0.0, 0.1, 0.2);
    bg *= 1 - 0.7 * fields.phi[i-10][j-10];

    var col = mix(bg, vec3(0.7, 0.8, 0.9), fields.phi[i][j]);
    col *= 1 - 0.7 * f32(j) / SCREEN_HEIGHT;

    textureStore(screen, vec2(i, j), vec4(col, 1));
}
