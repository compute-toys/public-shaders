#include "iq/noise_simplex_2d"
#include "Dave_Hoskins/hash"

fn noise_simplex_2d_hash(u: float2) -> float2 {
    return hash22(u);
}

fn rand2(u: uint2) -> float2 {
    return 2. * hash22(float2(u) + time.elapsed) - 1.;
}

#storage atomic_storage array<array<atomic<u32>,512>,3840>

fn atomicAddFloat(id: uint2, v: float) -> float {
    let p = &atomic_storage[id.x][id.y];
    var old = atomicLoad(p);
    var new_val = 0u;
    loop {
        new_val = bitcast<uint>(bitcast<float>(old) + v);
        let res = atomicCompareExchangeWeak(p, old, new_val);
        if res.exchanged { break; }
        old = res.old_value;
    }
    return bitcast<float>(new_val);
}

fn atomicAddFloat4(id: uint2, v: float4) -> float4 {
    let sid = uint2(4,1) * id;
    let rx = atomicAddFloat(sid, v.x);
    let ry = atomicAddFloat(sid + uint2(1,0), v.y);
    let rz = atomicAddFloat(sid + uint2(2,0), v.z);
    let rw = atomicAddFloat(sid + uint2(3,0), v.w);
    return float4(rx,ry,rz,rw);
}

fn atomicLoadFloat4(id: uint2) -> float4 {
    let sid = uint2(4,1) * id;
    let px = &atomic_storage[sid.x][sid.y];
    let py = &atomic_storage[sid.x + 1][sid.y];
    let pz = &atomic_storage[sid.x + 2][sid.y];
    let pw = &atomic_storage[sid.x + 3][sid.y];
    let rx = bitcast<float>(atomicLoad(px));
    let ry = bitcast<float>(atomicLoad(py));
    let rz = bitcast<float>(atomicLoad(pz));
    let rw = bitcast<float>(atomicLoad(pw));
    return float4(rx,ry,rz,rw);
}

fn atomicStoreFloat4(id: uint2, v: float4) {
    let sid = uint2(4,1) * id;
    let px = &atomic_storage[sid.x][sid.y];
    let py = &atomic_storage[sid.x + 1][sid.y];
    let pz = &atomic_storage[sid.x + 2][sid.y];
    let pw = &atomic_storage[sid.x + 3][sid.y];
    atomicStore(px, bitcast<uint>(v.x));
    atomicStore(py, bitcast<uint>(v.y));
    atomicStore(pz, bitcast<uint>(v.z));
    atomicStore(pw, bitcast<uint>(v.w));
}

fn pm(x: int2, n: int2) -> int2 {
    return ((x % n) + n) % n;
}

fn simplex4(c: float2) -> float4 {
    return float4(noise_simplex_2d(c),noise_simplex_2d(c + float2(1.124124)),abs(noise_simplex_2d(c + float2(2.213623))),noise_simplex_2d(c + float2(3.12393)));      
}

fn simplex2(c: float2) -> float2 {
    return float2(noise_simplex_2d(c),noise_simplex_2d(c + float2(1.124124)));      
}

@compute @workgroup_size(16, 16)
#workgroup_count Init 60 32 1
fn init(@builtin(global_invocation_id) id : uint3)
{
    if (time.frame < 30) {
        let screen_size = textureDimensions(screen);
        let c = float2(2*id.xy)/float2(screen_size);
        let n = simplex4(c);
        atomicStoreFloat4(id.xy, n);
    }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: uint3) {
    let screen_size = textureDimensions(screen);
    let c = float2(id.xy)/float2(screen_size);
    if (id.x >= 960 || id.y >= 512) {
        return;
    }
    var n = atomicLoadFloat4(id.xy) * float4(1.0, 1.0, 1., 1.);
    let r = n.yx * float2(-1,1) * (simplex2(2.*c + 0.1*time.elapsed) - simplex2(2.*c + n.xy/float2(screen_size) + 0.1*time.elapsed)); 
        
    n += 100.0*float4(r,0.,0.);
    n = float4(n.z*normalize(n.xy),n.z,n.w);
    var r1 = float4(0);
    let vs = 1.25;
    let pos0 = uint2(pm(int2(id.xy) + int2(vs*n.xy), int2(960,512)));
    let pos1 = uint2(pm(int2(id.xy) + int2(-vs*n.xy), int2(960,512)));
    if (n.z > 4.0) {
        let n0 = n * float4( 0.5, 0.5,0.5,1);
        let n1 = n * float4(-0.5,-0.5,0.5,1);
        atomicAddFloat4(pos0, n0);
        atomicAddFloat4(pos1, n1);
        r1 = atomicAddFloat4(id.xy, -n);
    } else {
        atomicAddFloat4(pos0, n);
        r1 = atomicAddFloat4(id.xy, -n);
    }

    let prev = 0.8 * passSampleLevelBilinearRepeat(0, c, 0.) + 0.2*r1.zzzz;
    passStore(0, int2(id.xy), prev);

    var col = fract(r1);
    textureStore(screen, int2(id.xy), prev);
}