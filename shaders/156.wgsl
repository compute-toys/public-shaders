#include "nikat/noise_simplex_3d"
#include "iq/noise_simplex_2d"
#include "Dave_Hoskins/hash"

struct MouseHistory {
    mouse_prev: uint2,
    mouse_delta: float2
}
// two screen size * 4 atomic buffers
#storage atomic_storage array<array<array<array<atomic<i32>,4>,SCREEN_HEIGHT>,SCREEN_WIDTH>,2>
#storage mouse_history MouseHistory
var<private> screen_size: int2;


/* 
A scatter-gather particle technique with atomics. 
Particles are discovered by their IDs, which are scattered 
into the atomic buffer, and the velocity of the particle 
is placed in position in the gather pass using 
bilinear interpolation into four cells in the pass buffer. 
Up to four particles can be scattered into each cell 
using an array of atomics. 
Has a bug where particles with near-zero velocity can 
blow up in velocity.
*/
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
    return ((x % n) + n) % n;
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

@compute @workgroup_size(16, 16)
fn init(@builtin(global_invocation_id) id : uint3)
{
    let pass_id = time.frame % 2;
    atomicStoreInt4(pass_id, id.xy, int4(-1));
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

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: uint3) {
    screen_size = int2(SCREEN_WIDTH, SCREEN_HEIGHT);
    let write_pass_id = (time.frame) % 2;
    let read_pass_id = (time.frame + 1) % 2;
    if (id.x >= SCREEN_WIDTH || id.y >= SCREEN_HEIGHT) {
        return;
    }

    let local_id = pos_to_id(int2(id.xy));

    var U_sum = float4(0);
    // Gather
    for(var i = -1; i <= 1; i++) {
        for(var j = -1; j <= 1; j++) {
            let ids = atomicLoadInt4(read_pass_id, wrap(id.xy,int2(i,j)));
            for(var n = 0; n < 4; n++) {
                if (ids[n] >= 0) {
                    let pos = id_to_pos(ids[n]);
                    let V = passLoad(0, pos, 0);
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
    

    U_sum += float4(0.3*curl_noise(id.xy, U_sum.xy),0,0);

    let local_ids = atomicLoadInt4(read_pass_id, id.xy);
    

    var mouse_vel = mouse_history.mouse_delta;

    if (id.x == 0 && id.y == 0 && mouse.click > 0) {
        mouse_history.mouse_delta = mix(float2(mouse.pos) - float2(mouse_history.mouse_prev),mouse_history.mouse_delta, 0.5);
        mouse_history.mouse_prev = mouse.pos;
    }

    let mouse_distance = distance(float2(mouse.pos),float2(id.xy));
    if (mouse.click > 0 && mouse_distance < 30.) {
        U_sum += 1.0*float4(mouse_vel / (mouse_distance + 1.), 0, 0);
    }

    //Scatter
    atomicAddToIDArray(write_pass_id, wrap(id.xy,int2(round(U_sum.xy))), local_id);
    passStore(0, int2(id.xy), 0.99*U_sum);
    textureStore(screen, int2(id.xy), float4(0.25*length(U_sum.xy)));
}