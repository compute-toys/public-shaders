#include "nikat/noise_simplex_3d"
#include "iq/noise_simplex_2d"
#include "Dave_Hoskins/hash"

/*


!!CLICK AND DRAG TO PAINT!!
Check the uniforms for a few parameters to mess with.


Continued ideation on scatter/gather particles with atomics.
Since our particles are spatially localized, we can treat them
as an Eulerian fluid, and perform divergence minimization directly
on the particles' velocity. This is similar to the approach in
Reintegration Tracking, but with no inherent limit on particle
velocity.
*/

/* 
Visualize occupancy of the scatter buffer.
Occupancy 0 = blue
Occupancy 1 = green, occurs in most cases.
Occupancy 2 = yellow
Occupancy 3 = red, occurs very rarely. This is maximum occupancy,
              and forward advection breaks over this threshold.
*/
const VISUALIZE_OCCUPANCY = false;

struct MouseHistory {
    mouse_prev: uint2,
    mouse_delta: float2
}
// two screen size * 4 atomic buffers
#storage atomic_storage array<array<array<array<atomic<i32>,4>,SCREEN_HEIGHT>,SCREEN_WIDTH>,2>
#storage mouse_history MouseHistory
var<private> screen_size: int2;

fn noise_simplex_2d_hash(u: float2) -> float2 {
    return hash22(u);
}

fn noise_simplex_3d_hash(u: float3) -> float3 {
    return hash33(u);
}

fn rand2(u: uint2) -> float2 {
    return 2. * hash22(float2(u) + time.elapsed) - 1.;
}

fn atomicAddToIDArray(pass_id: uint, pos: uint2, id: int) -> bool {
    for (var i = 0; i < 4; i++) {
        loop {
            let p = &atomic_storage[pass_id][pos.x][pos.y][i];
            let res = atomicCompareExchangeWeak(p, -1, id);
            // success, exit
            if res.exchanged { return true; }
            // didn't exchange, slot is occupied, continue
            if (res.old_value != -1) { break; }
        }

    }
    return false;
}

fn atomicLoadInt4(pass_id: uint, id: uint2) -> int4 {
    let px = &atomic_storage[pass_id][id.x][id.y][0];
    let py = &atomic_storage[pass_id][id.x][id.y][1];
    let pz = &atomic_storage[pass_id][id.x][id.y][2];
    let pw = &atomic_storage[pass_id][id.x][id.y][3];
    let rx = atomicLoad(px);
    let ry = atomicLoad(py);
    let rz = atomicLoad(pz);
    let rw = atomicLoad(pw);
    return int4(rx,ry,rz,rw);
}

fn atomicStoreInt4(pass_id: uint, id: uint2, v: int4) {
    let px = &atomic_storage[pass_id][id.x][id.y][0];
    let py = &atomic_storage[pass_id][id.x][id.y][1];
    let pz = &atomic_storage[pass_id][id.x][id.y][2];
    let pw = &atomic_storage[pass_id][id.x][id.y][3];
    atomicStore(px, v.x);
    atomicStore(py, v.y);
    atomicStore(pz, v.z);
    atomicStore(pw, v.w);
}

fn pm(x: int2, n: int2) -> int2 {
    return ((x % n) + n) % n;
}

fn pmf(x: float2, n: float2) -> float2 {
    return n*fract(x/n);
    //return ((x % n) + n) % n;
}

fn simplex4(c: float2) -> float4 {
    return float4(noise_simplex_2d(c),noise_simplex_2d(c + float2(1.124124)),abs(noise_simplex_2d(c + float2(2.213623))),noise_simplex_2d(c + float2(3.12393)));      
}

fn simplex3_2(c: float3) -> float2 {
    return float2(noise_simplex_3d(c),noise_simplex_3d(c + float3(1.124124)));      
}

fn simplex2(c: float2) -> float2 {
    return float2(noise_simplex_2d(c),noise_simplex_2d(c + float2(1.124124)));      
}

fn pos_to_id(pos: int2) -> int {
    return pos.x + pos.y * screen_size.x;
}

fn id_to_pos(id: int) -> int2 {
    return int2(id % screen_size.x, id / screen_size.x);
}

fn wrap(pos: uint2, offset: int2) -> uint2 {
    return uint2(pm(int2(pos) + offset, screen_size));
}

fn curl_noise(id: uint2, off: float2) -> float2 {
    let c = float2(id.xy)/float2(screen_size);
    let t = time.elapsed;
    let r = 0.3*float2(-1,1)*(simplex3_2(float3(8.*(c+off),t))
                            - simplex3_2(float3(8.* c,     t))).yx;
    return r;  
    
}

fn pack(f: float2) -> float {
    return bitcast<float>(pack2x16float(f));
}

fn unpack(f: float) -> float2 {
    return unpack2x16float(bitcast<uint>(f));
}

const p_y3 = array<float, 13>(-1.0391083e-001, -3.4489894e-001, -4.8179728e-001, -3.1508410e-001,  1.1805352e-001,  1.1920299e-001, -1.4625093e-001,  1.1920299e-001,  1.1805352e-001, -3.1508410e-001, -4.8179728e-001, -3.4489894e-001, -1.0391083e-001);
const p_y2 = array<float, 13>(2.6484959e-003, -4.4125709e-003, -6.8390049e-002, -2.5511910e-001, -5.5398879e-001, -1.2936001e-001, 4.6167731e-001, -1.2936001e-001, -5.5398879e-001, -2.5511910e-001, -6.8390049e-002, -4.4125709e-003, 2.6484959e-003);
const p_y1 = array<float, 13>(1.9000778e-006, -2.0540590e-003, -1.3499238e-002, -5.1257182e-002, -1.5871959e-001, -4.7194022e-001, -7.0606907e-001, -4.7194022e-001, -1.5871959e-001, -5.1257182e-002, -1.3499238e-002, -2.0540590e-003,  1.9000778e-006);

const p_x3 = array<float, 13>(3.7276962e-001,  5.4995743e-001,  2.4023362e-001, -7.8265086e-004,  1.8311873e-002, -2.3270335e-002, -1.0109051e-055,  2.3270335e-002, -1.8311873e-002,  7.8265088e-004, -2.4023362e-001, -5.4995743e-001, -3.7276962e-001);
const p_x2 = array<float, 13>(5.2398670e-002,  4.2486224e-002, -1.0892533e-001, -3.3953004e-001, -5.0984393e-001,  3.2918550e-001,  0.0, -3.2918550e-001,  5.0984393e-001,  3.3953004e-001,  1.0892533e-001, -4.2486224e-002, -5.2398670e-002);
const p_x1 = array<float, 13>(6.2750203e-003, -1.6414278e-003, -4.3498466e-002, -1.3135171e-001, -3.0484343e-001, -6.2280256e-001, 0.0, 6.2280256e-001, 3.0484343e-001, 1.3135171e-001, 4.3498466e-002, 1.6414278e-003, -6.2750203e-003);

const s_i = array<float, 3>(  5.2045614e-001, 4.5787111e-002, 5.3607463e-003);

const g_x = array<float, 13>(1.8154960e-002, 5.1439053e-002, 1.1757498e-001, 2.2045309e-001, 3.4292702e-001, 4.4580513e-001, 
         4.8633287e-001, 4.4580513e-001, 3.4292702e-001, 2.2045309e-001, 1.1757498e-001, 5.1439053e-002, 1.8154960e-002);  
const RANGE = 6;

fn poisson_x(id: uint2) -> float4 {
    var P1 = float2(0);
    var P2 = float2(0);
    var P3 = float2(0);
    var G = 0.0;
    var Gw = 0.0;
    for (var i = -RANGE; i <= RANGE; i++) {
        let index = RANGE + i;
        let p = pm(int2(id.xy) + int2(i,0), int2(screen_size));
        
        let t = passLoad(0,p,0).xy;
        let g = passLoad(2,p,0).x;
        
        P1 += float2(p_x1[index], p_y1[index]) * t;
        P2 += float2(p_x2[index], p_y2[index]) * t;
        P3 += float2(p_x3[index], p_y3[index]) * t;
        
        Gw += g_x[index];
        G  += g_x[index] * g;
    }
    
    G /= Gw;

    return float4(pack(P1),pack(P2),pack(P3), G);
}

fn poisson_y(id: uint2) -> float4 {
    var P = float2(0);
    var G = 0.0;
    var Gw = 0.0;
    for (var i = -RANGE; i <= RANGE; i++) {
        let index = RANGE + i;
        let p = pm(int2(id) + int2(0,i), int2(screen_size));
        
        let tx = passLoad(1,p,0);
        let t1 = unpack(tx.x);
        let t2 = unpack(tx.y);
        let t3 = unpack(tx.z);

        let g = tx.w;
        
        P += s_i[0] * float2(p_x1[index], p_y1[index]).yx * t1;
        P += s_i[1] * float2(p_x2[index], p_y2[index]).yx * t2;
        P += s_i[2] * float2(p_x3[index], p_y3[index]).yx * t3;
        Gw += g_x[index];
        G  += g_x[index] * g;
    }
    
    G /= Gw;

    return float4((P.x + P.y) + G);
}

fn laplacian(pass_id: int, id: uint2) -> float4 {
    let sw = passLoad(pass_id, int2(wrap(id,int2(-1,-1))), 0);
    let s =  passLoad(pass_id, int2(wrap(id,int2( 0,-1))), 0);
    let se = passLoad(pass_id, int2(wrap(id,int2( 1,-1))), 0);
    let e  = passLoad(pass_id, int2(wrap(id,int2( 1, 0))), 0);
    let ne = passLoad(pass_id, int2(wrap(id,int2( 1, 1))), 0);
    let n  = passLoad(pass_id, int2(wrap(id,int2( 0, 1))), 0);
    let nw = passLoad(pass_id, int2(wrap(id,int2(-1, 1))), 0);
    let w  = passLoad(pass_id, int2(wrap(id,int2(-1, 0))), 0);
    let c  = passLoad(pass_id, int2(wrap(id,int2( 0, 0))), 0);

    return (-20.*c + 4.*(n+e+s+w) + (ne+nw+se+sw))/6.;
}

fn grad(pass_id: int, id: uint2) -> float2 {
    let sw = passLoad(pass_id, int2(wrap(id,int2(-1,-1))), 0).x;
    let s =  passLoad(pass_id, int2(wrap(id,int2( 0,-1))), 0).x;
    let se = passLoad(pass_id, int2(wrap(id,int2( 1,-1))), 0).x;
    let e  = passLoad(pass_id, int2(wrap(id,int2( 1, 0))), 0).x;
    let ne = passLoad(pass_id, int2(wrap(id,int2( 1, 1))), 0).x;
    let n  = passLoad(pass_id, int2(wrap(id,int2( 0, 1))), 0).x;
    let nw = passLoad(pass_id, int2(wrap(id,int2(-1, 1))), 0).x;
    let w  = passLoad(pass_id, int2(wrap(id,int2(-1, 0))), 0).x;

    return 0.25*float2((e + 0.5 * (ne + se)) - (w + 0.5 * (nw + sw)),
                       (n + 0.5 * (ne + nw)) - (s + 0.5 * (se + sw)));
    //return 0.5*float2(e-w,n-s);
}

fn initialize(id: uint2) -> bool {
    screen_size = int2(SCREEN_WIDTH, SCREEN_HEIGHT);
    if (id.x >= SCREEN_WIDTH || id.y >= SCREEN_HEIGHT) {
        return false;
    }
    return true;
}

@compute @workgroup_size(16, 16)
fn init(@builtin(global_invocation_id) id : uint3)
{
    if (!initialize(id.xy)) {return;}
    let pass_id = time.frame % 2;
    atomicStoreInt4(pass_id, id.xy, int4(-1));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: uint3) {
    if (!initialize(id.xy)) {return;}
    let read_pass_id = (time.frame + 1) % 2;

    var U_sum = float4(0);
    // Gather
    for(var i = -1; i <= 1; i++) {
        for(var j = -1; j <= 1; j++) {
            let ids = atomicLoadInt4(read_pass_id, wrap(id.xy,int2(i,j)));
            for(var n = 0; n < 4; n++) {
                if (ids[n] >= 0) {
                    let pos = id_to_pos(ids[n]);
                    let V = passLoad(0, pos, 0);
                    // wrap into our bounding box
                    let p = pmf(float2(pos) + V.xy, float2(screen_size));
                    let d = p - float2(id.xy);
                    // if particle neighbors cell
                    if (d.x <= 1.0 && 
                        d.x >= -1.0 &&
                        d.y <= 1.0 &&
                        d.y >= -1.0) 
                    {
                        //construct bilinear interpolation weights
                        let m = fract(d);
                        var w: float;
                        if (d.x >= 0) {
                            if (d.y >= 0) {
                                w = mix(mix(1.,0.,m.x),0.,m.y);
                            } else {
                                w = mix(0.,mix(1.,0.,m.x),m.y);
                            }
                        } else if (d.x < 0) {
                            if (d.y >= 0) {
                                w = mix(mix(0.,1.,m.x),0.,m.y);
                            } else {
                                w = mix(0.,mix(0.,1.,m.x),m.y);
                            } 
                        } 
                        U_sum += w*V;
                    }
                }
            }
        }
    }
    

    U_sum += float4(custom.noise*curl_noise(id.xy, U_sum.xy),0,0);

    let local_ids = atomicLoadInt4(read_pass_id, id.xy);
    

    var mouse_vel = mouse_history.mouse_delta;

    if (mouse.pos.x != 0 && mouse.pos.y != 0){
        if (id.x == 0 && id.y == 0 && mouse.click > 0) {
            mouse_history.mouse_delta = mix(float2(mouse.pos) - float2(mouse_history.mouse_prev),mouse_history.mouse_delta, 0.1);
            mouse_history.mouse_prev = mouse.pos;
        }

        let mouse_distance = distance(float2(mouse.pos),float2(id.xy));
        if (mouse.click > 0 && mouse_distance < 30.) {
            U_sum += 1.0*float4(mouse_vel / (mouse_distance + 1.), 1 / (mouse_distance + 1.), 0);
        }
    }

    passStore(0, int2(id.xy), U_sum);
}

@compute @workgroup_size(16, 16)
fn blur(@builtin(global_invocation_id) id: uint3) {
    if (!initialize(id.xy)) {return;}
    var U = passLoad(0, int2(id.xy), 0);
    U += float4(custom.blur/3.3 * laplacian(0, id.xy).xy,0,0);
    passStore(0, int2(id.xy), U);
}

@compute @workgroup_size(16, 16)
fn poisson_x_pass(@builtin(global_invocation_id) id: uint3) {
    if (!initialize(id.xy)) {return;}
    passStore(1, int2(id.xy), poisson_x(id.xy));
}

@compute @workgroup_size(16, 16)
fn poisson_y_pass(@builtin(global_invocation_id) id: uint3) {
    if (!initialize(id.xy)) {return;}
    passStore(2, int2(id.xy), poisson_y(id.xy));
}

@compute @workgroup_size(16, 16)
fn minimize(@builtin(global_invocation_id) id: uint3) {
    if (!initialize(id.xy)) {return;}
    var U = passLoad(0, int2(id.xy), 0);
    U -= float4(custom.divmin * grad(2, id.xy),0,0);
    passStore(0, int2(id.xy), U);
}

@compute @workgroup_size(16, 16)
fn scatter(@builtin(global_invocation_id) id: uint3) {
    if (!initialize(id.xy)) {return;}
    let write_pass_id = (time.frame) % 2;
    let local_id = pos_to_id(int2(id.xy));
    var U = passLoad(0, int2(id.xy), 0);
    //Scatter to atomic buffer
    atomicAddToIDArray(write_pass_id, wrap(id.xy,int2(round(U.xy))), local_id);
}

fn rgb2hsv(c: float3) -> float3
{
    let K = float4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    let p = mix(float4(c.bg, K.wz), float4(c.gb, K.xy), step(c.b, c.g));
    let q = mix(float4(p.xyw, c.r), float4(c.r, p.yzx), step(p.x, c.r));

    let d = q.x - min(q.w, q.y);
    let e = 1.0e-10;
    return float3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

fn hsv2rgb(c: float3) -> float3
{
    let K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    let p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, float3(0), float3(1)), c.y);
}

@compute @workgroup_size(16, 16)
fn image(@builtin(global_invocation_id) id: uint3) {
    if (!initialize(id.xy)) {return;}
    let read_pass_id = (time.frame + 1) % 2;
    var U = passLoad(0, int2(id.xy), 0);
    let i = atomicLoadInt4(read_pass_id, id.xy);
    let b = select(int4(0),int4(1),i != int4(-1));
    let c = float(b.x+b.y+b.z+b.w)/4.;
    //let col = mix(float4(0,0,1,1),float4(1,0,0,1),c);
    let col = pow(hsv2rgb(float3(0.5-0.5*c,1.,0.2+U.z)),float3(2));
    if(VISUALIZE_OCCUPANCY) {
        textureStore(screen, int2(id.xy), float4(col,1));
    } else {
        textureStore(screen, int2(id.xy), float4(U.z));
    }
}
