/*
    MLS-MPM/APIC fluid system using Tait EoS. 
    
    Uses density-based pressure in addition to a jacobi solve for better particle
    separation, but it'd be preferable to do it another way. The particle count
    is very high here so MLS-MPM struggles to maintain volume.

    If performance is bad on your system, lower particle count (N_PART) or the
    number of pressure solve steps (JACOBI_STEPS).
    
    Uses quantized mass and momentum to avoid using CAS for particle splatting.
    I attempt to cancel out quantization error here so that even using a low
    quantization factor is fairly stable. This was mostly done to debug instability
    caused by other issues, so it can probably be removed for improved performance.

    The buffer packing is pretty aggressive in order to support SDF generation for
    boundaries and rendering, which is not in this shader yet.
*/

#include "Dave_Hoskins/hash"

const ATOM_SCALE_F : float = pow(2.,16.); // quantization factor
const SDF_THRESH : float = 0.02;          // density threshold for “inside”
const BIG        : float = 1e20;          // sentinel for “no seed”
const BOUND_PAD  : float = 10;            // boundary size
const MASS_EPS   : float = 1e-6;          // minimum mass
const FLUID_T    : float = 0.005;         // mass threshold to mark fluid

#define GRAVITY      float2(0.0, custom.gravity)
#define DT           custom.timestep
#define PRESSURE_K   custom.pressure_density
#define MASS_SMOOTH  custom.mass_smooth
#define VISC_NU      custom.viscosity
#define APIC_SCALE   custom.apic_scale
#define PRESSURE_J   custom.pressure_jacobi

// particle counts
#define N_PART 3000000u

// pressure solver steps
#define JACOBI_STEPS 30

// particle workgroup count
#calcdefine N_PART_WG (((N_PART)+255)/256)

struct Particle {
  x   : float2,       // pixel space
  v   : float2,
  C00 : float, C01 : float,
  C10 : float, C11 : float,  // APIC affine matrix
  J   : float,
  _p  : float,
};
#storage particles array<Particle, N_PART>

// atomic_storage[x][y][0] = mass_i
// atomic_storage[x][y][1] = mvx_i
// atomic_storage[x][y][2] = mvy_i
#storage atomic_storage array<array<array<atomic<i32>, 3>, SCREEN_HEIGHT>, SCREEN_WIDTH>

fn in_bounds(i:int, j:int) -> bool {
  return (i >= 0 && j >= 0 && i < SCREEN_WIDTH && j < SCREEN_HEIGHT);
}

fn particle_mass() -> float {
  let cells = float(SCREEN_WIDTH * SCREEN_HEIGHT);
  let ppc   = float(N_PART) / max(cells, 1.0);
  // make the average grid density equal rest_density
  return custom.rest_density / max(ppc, 1e-6);
}

fn encodeFP(x: float) -> int { return int(round(x * ATOM_SCALE_F)); }
fn decodeFP(x: int) -> float { return float(x) / ATOM_SCALE_F; }


fn w3_centered(a: float) -> float3 {
  // a in [-0.5, 0.5]
  return float3(
    0.5 * (0.5 - a) * (0.5 - a),
    0.75 - a * a,
    0.5 * (0.5 + a) * (0.5 + a)
  );
}

fn loadC(p: Particle) -> float2x2 {
  let col0 = float2(p.C00, p.C01); // column 0: rows 0,1
  let col1 = float2(p.C10, p.C11); // column 1: rows 0,1
  return float2x2(col0, col1);
}

fn storeC(C: float2x2, p: ptr<function, Particle>) {
  (*p).C00 = C[0][0];  (*p).C01 = C[0][1];
  (*p).C10 = C[1][0];  (*p).C11 = C[1][1];
}

@compute @workgroup_size(256)
fn init(@builtin(global_invocation_id) gid: uint3) {
  if (gid.x >= N_PART) { return; }

  // hoskins hash performs best in 0..1 range
  let r = hash22(float2(gid.xy)/1000.);

  var p = Particle(
    float2(0.2 + 0.6*r.x, 0.65 + 0.3*r.y) * float2(SCREEN_WIDTH, SCREEN_HEIGHT),
    float2(0.0, 0.0),
    0.0, 0.0, 0.0, 0.0,
    1.0, 0.0
  );
  particles[gid.x] = p;
}
#workgroup_count init N_PART_WG 1 1
#dispatch_once init

// clear atomic storage
@compute @workgroup_size(16,16,1)
fn clear_grid_atom(@builtin(global_invocation_id) id: uint3) {
  if (id.x >= SCREEN_WIDTH || id.y >= SCREEN_HEIGHT) { return; }
  atomicStore(&atomic_storage[id.x][id.y][0], 0);
  atomicStore(&atomic_storage[id.x][id.y][1], 0);
  atomicStore(&atomic_storage[id.x][id.y][2], 0);
}

@compute @workgroup_size(256)
fn p2g(@builtin(global_invocation_id) gid: uint3) {
  if (gid.x >= N_PART) { return; }
  let p  = particles[gid.x];
  let C  = loadC(p);
  let mp = particle_mass();

  // centered stencil
  let cell = int2(floor(p.x)); // integer cell
  let c    = p.x - (float2(cell) + vec2(0.5, 0.5)); // in [-0.5, 0.5]

  let wx = w3_centered(c.x);
  let wy = w3_centered(c.y);

  // Build quantized taps first (mass & momentum), then balance.
  var ix : array<int, 9>;
  var iy : array<int, 9>;
  var qm : array<int, 9>;   // mass
  var qx : array<int, 9>;   // momentum x
  var qy : array<int, 9>;   // momentum y
  var sm : int = 0; var sx : int = 0; var sy : int = 0;
  var n  : uint = 0u;

  // target totals
  let M_t  = encodeFP(mp);
  let Mx_t = encodeFP(mp * p.v.x);
  let My_t = encodeFP(mp * p.v.y);

  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let node = cell + int2(dx, dy);
      let idx  = dx + 1;   // 0..2
      let idy  = dy + 1;   // 0..2
      let w    = wx[idx] * wy[idy];

      // dpos in grid cells
      let dpos   = float2(float(dx), float(dy)) - c;
      let v_apic = p.v + APIC_SCALE * (C * dpos);

      // quantize taps
      var qmi = encodeFP(mp * w);
      var qxi = encodeFP((mp * w) * v_apic.x);
      var qyi = encodeFP((mp * w) * v_apic.y);

      // near boundaries you can keep kernel consistent by writing zeros
      if (!in_bounds(node.x, node.y)) {
        // pick a safe in-bounds surrogate
        let safe = cell; // dx=0,dy=0
        ix[n] = safe.x; iy[n] = safe.y;
        qm[n] = 0; qx[n] = 0; qy[n] = 0;
      } else {
        ix[n] = node.x; iy[n] = node.y;
        qm[n] = qmi;    qx[n] = qxi;  qy[n] = qyi;
        sm   += qmi;    sx   += qxi;  sy   += qyi;
      }
      n++;
    }
  }

  // enforce exact totals on center tap (dy=0,dx=0 -> index 4)
  let cidx = 4u;
  qm[cidx] += (M_t  - sm);
  qx[cidx] += (Mx_t - sx);
  qy[cidx] += (My_t - sy);

  // accumulate
  for (var k = 0u; k < 9u; k++) {
    atomicAdd(&atomic_storage[ix[k]][iy[k]][0], qm[k]);
    atomicAdd(&atomic_storage[ix[k]][iy[k]][1], qx[k]);
    atomicAdd(&atomic_storage[ix[k]][iy[k]][2], qy[k]);
  }
}
#workgroup_count p2g N_PART_WG 1 1

@compute @workgroup_size(256)
fn p2g_stress(@builtin(global_invocation_id) gid: uint3) {
  if (gid.x >= N_PART) { return; }
  let p  = particles[gid.x];
  let C  = loadC(p);
  let mp = particle_mass();

  // centered stencil
  let cell = int2(floor(p.x));
  let c    = p.x - (float2(cell) + float2(0.5, 0.5));
  let wx   = w3_centered(c.x);
  let wy   = w3_centered(c.y);

  // density from already-splatted masses
  var rho = 0.0;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let idx  = dx + 1;
      let idy  = dy + 1;
      let node = cell + int2(dx, dy);
      if (!in_bounds(node.x, node.y)) { continue; }
      let w = wx[idx] * wy[idy];
      rho  += w * decodeFP(atomicLoad(&atomic_storage[node.x][node.y][0]));
    }
  }
  if (rho <= MASS_EPS) { return; }
  let Vp = mp / rho;

  // Tait EOS + viscous stress
  let rho0 = custom.rest_density;
  let rho_eff = clamp(rho, MASS_EPS, 2.0 * rho0);
  var press = custom.eos_stiffness * (pow(rho_eff / rho0, custom.eos_power) - 1.0);
  if (press < 0.0) { press = 0.0; }

  let S = 0.5 * (C + transpose(C));
  let I = float2x2(float2(1., 0.), float2(0., 1.));
  let sigma = (-press) * I + (2.0 * custom.stress_visc) * S;
  let fac   = -Vp * 4.0 * DT;

  // build, then enforce exact zero-sum impulse in integer space
  var ix : array<int, 9>;
  var iy : array<int, 9>;
  var qx : array<int, 9>;
  var qy : array<int, 9>;
  var sx : int = 0;
  var sy : int = 0;
  var n  : uint = 0u;

  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let idx  = dx + 1;
      let idy  = dy + 1;
      let node = cell + int2(dx, dy);

      if (!in_bounds(node.x, node.y)) {
        // consistent kernel length, contribute 0 to a safe in-bounds cell
        let safe = cell; // center
        ix[n] = safe.x; iy[n] = safe.y;
        qx[n] = 0; qy[n] = 0;
        n++;
        continue;
      }

      let w    = wx[idx] * wy[idy];
      let dpos = float2(float(dx), float(dy)) - c;
      let dv   = fac * w * (sigma * dpos);

      let qxi = encodeFP(dv.x);
      let qyi = encodeFP(dv.y);

      ix[n] = node.x; iy[n] = node.y;
      qx[n] = qxi;    qy[n] = qyi;
      sx   += qxi;    sy   += qyi;
      n++;
    }
  }

  // exact zero-sum (center index 4)
  let cidx = 4u;
  qx[cidx] -= sx;
  qy[cidx] -= sy;

  for (var k = 0u; k < 9u; k++) {
    atomicAdd(&atomic_storage[ix[k]][iy[k]][1], qx[k]);
    atomicAdd(&atomic_storage[ix[k]][iy[k]][2], qy[k]);
  }
}
#workgroup_count p2g_stress N_PART_WG 1 1

@compute @workgroup_size(16,16,1)
fn grid_update_atomic(@builtin(global_invocation_id) id: uint3) {
  if (id.x >= SCREEN_WIDTH || id.y >= SCREEN_HEIGHT) { return; }

  let m_i  = atomicLoad(&atomic_storage[id.x][id.y][0]);
  let mx_i = atomicLoad(&atomic_storage[id.x][id.y][1]);
  let my_i = atomicLoad(&atomic_storage[id.x][id.y][2]);

  var m = decodeFP(m_i);
  var mx = decodeFP(mx_i);
  var my = decodeFP(my_i);
  var v = float2(mx, my);

  // momentum to velocity
  if (m > MASS_EPS) { v /= m; } else { v = float2(0.0); m = 0.0; }

  // gravity
  // this needs to happen early for correct behavior on subsequent passes
  v += DT * GRAVITY;

  /*
  let i = i32(id.x);
  let j = i32(id.y);
  if (i < BOUND_PAD && v.x < 0.0)                       { v.x = 0.0; }
  if (i > (SCREEN_WIDTH  - 1 - BOUND_PAD) && v.x > 0.0) { v.x = 0.0; }
  if (j < BOUND_PAD && v.y < 0.0)                       { v.y = 0.0; }
  if (j > (SCREEN_HEIGHT - 1 - BOUND_PAD) && v.y > 0.0) { v.y = 0.0; }
  */

  // store current grid
  passStore(0, int2(id.xy), float4(v, m, 0.0));
}

// gaussian weights here are computed from erf derivative and normalized
@compute @workgroup_size(16,16,1)
fn blur_x(@builtin(global_invocation_id) id: uint3) {
  if (id.x >= SCREEN_WIDTH || id.y >= SCREEN_HEIGHT) { return; }

  let g_w = array<float,5>(0.0613595809,0.2447702197,0.3877403988,0.2447702197,0.0613595809);
  var sum = float4(0.);
  for (var i = -2; i <= 2; i += 1) {
    let idx = int2( clamp(int(id.x) + i, 0, SCREEN_WIDTH - 1), int(id.y) );
    sum += g_w[i+2] * passLoad(0, idx, 0);
  }

  passStore(1, int2(id.xy), sum);
}

// second blur pass. we can reuse pass 1 after this point
@compute @workgroup_size(16,16,1)
fn blur_y(@builtin(global_invocation_id) id: uint3) {
  if (id.x >= SCREEN_WIDTH || id.y >= SCREEN_HEIGHT) { return; }

  let g_w = array<float,5>(0.0613595809,0.2447702197,0.3877403988,0.2447702197,0.0613595809);
  var sum = float4(0.);
  for (var i = -2; i <= 2; i += 1) {
    let idx = int2( int(id.x), clamp(int(id.y) + i, 0, SCREEN_HEIGHT - 1) );
    sum += g_w[i+2] * passLoad(1, idx, 0);
  }

  passStore(2, int2(id.xy), sum);
}

@compute @workgroup_size(16,16,1)
fn grid_update(@builtin(global_invocation_id) id: uint3) {
  if (id.x >= SCREEN_WIDTH || id.y >= SCREEN_HEIGHT) { return; }

  let p = passLoad(0, int2(id.xy), 0);
  var m = p.z;
  var v = p.xy;

  let i = int(id.x);
  let j = int(id.y);
  let iL = max(i - 1, 0);
  let iR = min(i + 1, SCREEN_WIDTH  - 1);
  let jT = max(j - 1, 0);
  let jB = min(j + 1, SCREEN_HEIGHT - 1);

  let u_L = passLoad(2, int2(iL, j ), 0);
  let u_R = passLoad(2, int2(iR, j ), 0);
  let u_T = passLoad(2, int2(i , jT), 0);
  let u_B = passLoad(2, int2(i , jB), 0);

  let blur = passLoad(2, int2(id.xy), 0);
  m = mix(m, blur.z, MASS_SMOOTH);

  // gradient
  let grad_rho = float2(0.5 * (u_R.z - u_L.z), 0.5 * (u_B.z - u_T.z));

  // may need to scale pressure by mass here
  //var invm = 0.0;
  //if (m > MASS_EPS) { invm = 1.0 / m; }

  // density-based pressure
  v -= DT * PRESSURE_K * grad_rho;

  // viscosity with cheap laplacian approximation
  v += DT * VISC_NU * (blur.xy - v);

  /* bounce on walls
  if (i < BOUND_PAD && v.x < 0.0) { v.x = -v.x; }
  if (i > (SCREEN_WIDTH  - 1 - BOUND_PAD) && v.x > 0.0) { v.x = -v.x; }
  if (j < BOUND_PAD && v.y < 0.0) { v.y = -v.y; }
  if (j > (SCREEN_HEIGHT - 1 - BOUND_PAD) && v.y > 0.0) { v.y = -v.y; }
  */

  // write
  passStore(1, int2(id.xy), float4(v.x, v.y, m, 0.0));
}

@compute @workgroup_size(16,16,1)
fn copy_back(@builtin(global_invocation_id) id: uint3) {
  let p = passLoad(1, int2(id.xy), 0);
  passStore(0, int2(id.xy), p);
}

fn fluid_at(ij:int2)->bool { return passLoad(1, ij, 0).y > 0.5; }

@compute @workgroup_size(16,16,1)
fn divergence(@builtin(global_invocation_id) id: uint3) {
  if (id.x >= SCREEN_WIDTH || id.y >= SCREEN_HEIGHT) { return; }
  let ij = int2(id.xy);
  let m_blur = passLoad(2, int2(id.xy), 0).z;

  // if no fluid, divergence is 0, mark as air
  if (m_blur < FLUID_T) { passStore(1, ij, float4(0.0)); return; }

  let i = ij.x; let j = ij.y;
  let iL = max(i-1,0); let iR = min(i+1, SCREEN_WIDTH-1);
  let jT = max(j-1,0); let jB = min(j+1, SCREEN_HEIGHT-1);

  let uL = passLoad(0, int2(iL,j), 0).x;
  let uR = passLoad(0, int2(iR,j), 0).x;
  let vT = passLoad(0, int2(i, jT), 0).y;
  let vB = passLoad(0, int2(i, jB), 0).y;

  // divergence
  let div = 0.5 * ((uR - uL) + (vB - vT));

  // store divergence, second channel marks as fluid
  passStore(1, ij, float4(div, 1.0, 0.0, 0.0));
}

// reuse pass 1 here and pingpong between the 3rd and 4th channels
@compute @workgroup_size(16,16,1)
fn pressure_jacobi(@builtin(global_invocation_id) id: uint3) {
  if (id.x >= SCREEN_WIDTH || id.y >= SCREEN_HEIGHT) { return; }
  let ij = int2(id.xy);

  // pingpong src and dest
  let src = select(2, 3, bool(dispatch.id % 2));
  let dst = select(3, 2, bool(dispatch.id % 2));

  let D = passLoad(1, ij, 0);
  if (!fluid_at(ij)) { passStore(1, ij, float4(D.xy, 0., 0.)); return; }

  let i = ij.x; let j = ij.y;
  let iL = max(i-1,0); let iR = min(i+1, SCREEN_WIDTH-1);
  let jT = max(j-1,0); let jB = min(j+1, SCREEN_HEIGHT-1);

  var sum = 0.0;
  var Nf  = 0.0;

  if (fluid_at(int2(iL,j))) { sum += passLoad(1, int2(iL,j), 0)[src]; Nf += 1.0; }
  if (fluid_at(int2(iR,j))) { sum += passLoad(1, int2(iR,j), 0)[src]; Nf += 1.0; }
  if (fluid_at(int2(i, jT))) { sum += passLoad(1, int2(i, jT), 0)[src]; Nf += 1.0; }
  if (fluid_at(int2(i, jB))) { sum += passLoad(1, int2(i, jB), 0)[src]; Nf += 1.0; }

  let b = D.x; //divergence
  let p_new = (sum - b) / max(Nf, 1.0);
  var res = float4(D.xy, 0., 0.);
  res[dst] = p_new;
  passStore(1, ij, res);
}
#dispatch_count pressure_jacobi JACOBI_STEPS

@compute @workgroup_size(16,16,1)
fn pressure_apply(@builtin(global_invocation_id) id: uint3) {
  if (id.x >= SCREEN_WIDTH || id.y >= SCREEN_HEIGHT) { return; }
  let ij = int2(id.xy);
  var U = passLoad(0, ij, 0); // (vx,vy,m,_)

  if (U.z <= MASS_EPS) { passStore(0, ij, U); return; }

  // pick source channel based on step count
  let src = select(2, 3, bool(JACOBI_STEPS % 2));
  let i = ij.x; let j = ij.y;
  let iL = max(i-1,0); let iR = min(i+1, SCREEN_WIDTH-1);
  let jT = max(j-1,0); let jB = min(j+1, SCREEN_HEIGHT-1);

  var pL : f32; var pR : f32; var pT : f32; var pB : f32;

  if (fluid_at(int2(iL,j)))  { pL = passLoad(1, int2(iL,j), 0)[src]; } else { pL = 0.;}
  if (fluid_at(int2(iR,j)))  { pR = passLoad(1, int2(iR,j), 0)[src]; } else { pR = 0.;}
  if (fluid_at(int2(i, jT))) { pT = passLoad(1, int2(i,jT), 0)[src]; } else { pT = 0.;}
  if (fluid_at(int2(i, jB))) { pB = passLoad(1, int2(i,jB), 0)[src]; } else { pB = 0.;}

  let gradp = 0.5 * float2(pR - pL, pB - pT);

  // using blurred mass/density for rho here, not really correct
  let rho = max(passLoad(2, ij, 0).z, 1e-6);
  // apply
  U = float4(U.xy - DT * PRESSURE_J * gradp / rho, U.zw);

  // should maybe zero pressure normals on boundary here
  passStore(0, ij, U);
}

// APIC gather
@compute @workgroup_size(256)
fn g2p(@builtin(global_invocation_id) gid: uint3) {
  if (gid.x >= N_PART) { return; }
  var p = particles[gid.x];

  let cell = int2(floor(p.x));
  let c    = p.x - (float2(cell) + float2(0.5, 0.5));
  let wx   = w3_centered(c.x);
  let wy   = w3_centered(c.y);

  // velocity gather with renorm by occupied nodes
  var v_p  = float2(0.0);
  var wsum = 0.0;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let idx  = dx + 1;
      let idy  = dy + 1;
      let node = cell + int2(dx, dy);
      if (!in_bounds(node.x, node.y)) { continue; }
      let w = wx[idx] * wy[idy];
      let g = passLoad(0, node, 0);    // (vx, vy, m, _)
      if (g.z > MASS_EPS) {
        v_p  += w * g.xy;
        wsum += w;
      }
    }
  }
  if (wsum > 0.0) { v_p /= wsum; }

  // APIC B = Σ (w v) ⊗ dpos, then C = 4 * B
  var B     = float2x2();
  var wsumB = 0.0;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let idx  = dx + 1;
      let idy  = dy + 1;
      let node = cell + int2(dx, dy);
      if (!in_bounds(node.x, node.y)) { continue; }
      let w = wx[idx] * wy[idy];
      let g = passLoad(0, node, 0);
      if (g.z <= MASS_EPS) { continue; }
      let dpos = vec2<f32>(f32(dx), f32(dy)) - c;
      let wv   = w * g.xy;
      // column-wise constructor: [ [wv.x*dpos.x, wv.y*dpos.x], [wv.x*dpos.y, wv.y*dpos.y] ]
      let col0 = wv * dpos.x;
      let col1 = wv * dpos.y;
      B += float2x2(col0, col1);
      wsumB += w;
    }
  }
  if (wsumB > 0.0) { B *= (1.0 / wsumB); }

  p.v = v_p;
  storeC(4.0 * B, &p);

  // wall pre-impulse + advect
  // could be done for arbitrary boundaries once we build an sdf using JFA
  let k              = 3.0;
  let wall_stiffness = 0.30;
  let wall_min = float2(float(BOUND_PAD) + 1.0, float(BOUND_PAD) + 1.0);
  let wall_max = float2(float(SCREEN_WIDTH  - 1 - BOUND_PAD) - 1.0,
                           float(SCREEN_HEIGHT - 1 - BOUND_PAD) - 1.0);
  let x_n = p.x + p.v * (DT * k);
  if (x_n.x < wall_min.x) { p.v.x += wall_stiffness * (wall_min.x - x_n.x); }
  if (x_n.x > wall_max.x) { p.v.x += wall_stiffness * (wall_max.x - x_n.x); }
  if (x_n.y < wall_min.y) { p.v.y += wall_stiffness * (wall_min.y - x_n.y); }
  if (x_n.y > wall_max.y) { p.v.y += wall_stiffness * (wall_max.y - x_n.y); }

  p.x += DT * p.v;

  let rng = 0.1*(hash22(float2(float(gid.x), time.elapsed))-0.5);

  // jitter a little bit to prevent overlapping particles
  p.x += rng;

  if (mouse.click > 0) {
    p.v += 2.0 * exp(-0.05 * length(float2(mouse.pos) - p.x)) * float2(mouse.delta);
  }

  let lo = float2(float(BOUND_PAD), float(BOUND_PAD));
  let hi = float2(float(SCREEN_WIDTH - 1 - BOUND_PAD), float(SCREEN_HEIGHT - 1 - BOUND_PAD));
  p.x = clamp(p.x, lo, hi);

  particles[gid.x] = p;
}
#workgroup_count g2p N_PART_WG 1 1

@compute @workgroup_size(16,16,1)
fn draw(@builtin(global_invocation_id) id: uint3) {
  if (id.x >= SCREEN_WIDTH || id.y >= SCREEN_HEIGHT) { return; }
  let g = passLoad(1, int2(id.xy), 0);
  let m = passLoad(0, int2(id.xy), 0);

  //textureStore(screen, int2(id.xy), 0.1*float4(g.z,0,0,0) + 0.01*m.zzzz);
  textureStore(screen, int2(id.xy), 0.01*m.zzzz);
}