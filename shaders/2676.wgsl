// We sadly need to calculate some define values (hence I just calculated all)
// since the preprocessor for workgroups does only work with non-chained values.

#define CHUNK_SIZE 16 // Must be multiple of biggest subchunk dimension.
#define CHUNK_AMOUNT 4
#define WORLD_SIZE 64 // CHUNK_SIZE * CHUNK_AMOUNT

#define SUB_CHUNK_X 4
#define SUB_CHUNK_Y 4
#define SUB_CHUNK_Z 2

#define CELLS_PER_SUBCHUNK 32 // SUB_CHUNK_X * SUB_CHUNK_Y * SUB_CHUNK_Z (Must be 32)

#define SUBCHUNKS_PER_CHUNK_X 4 // CHUNK_SIZE / SUB_CHUNK_X
#define SUBCHUNKS_PER_CHUNK_Y 4 // CHUNK_SIZE / SUB_CHUNK_Y
#define SUBCHUNKS_PER_CHUNK_Z 8 // CHUNK_SIZE / SUB_CHUNK_Z

#define SUBCHUNKS_WITH_BORDER_X 6 // SUBCHUNKS_PER_CHUNK_X + 2
#define SUBCHUNKS_WITH_BORDER_Y 6 // SUBCHUNKS_PER_CHUNK_Y + 2
#define SUBCHUNKS_WITH_BORDER_Z 10 // SUBCHUNKS_PER_CHUNK_Z + 2
#define THREADS_PER_CHUNK 128 // SUBCHUNKS_PER_CHUNK_X * SUBCHUNKS_PER_CHUNK_Y * SUBCHUNKS_PER_CHUNK_Z

#define TOTAL_SUBCHUNKS_TO_LOAD 360 // SUBCHUNKS_WITH_BORDER_X * SUBCHUNKS_WITH_BORDER_Y * SUBCHUNKS_WITH_BORDER_Z

var<workgroup> localChunk: array<u32, TOTAL_SUBCHUNKS_TO_LOAD>;
var<workgroup> chunkHasActiveCells: atomic<u32>;

#define SUBCHUNKS_PER_WORLD_X 16 // WORLD_SIZE / SUB_CHUNK_X
#define SUBCHUNKS_PER_WORLD_Y 16 // WORLD_SIZE / SUB_CHUNK_Y
#define SUBCHUNKS_PER_WORLD_Z 32 // WORLD_SIZE / SUB_CHUNK_Z
#define TOTAL_WORLD_SUBCHUNKS 8192 // SUBCHUNKS_PER_WORLD_X * SUBCHUNKS_PER_WORLD_Y * SUBCHUNKS_PER_WORLD_Z
#define INIT_WG_X 33 // (TOTAL_WORLD_SUBCHUNKS+255)/256
#define CHUNK_OCCUPANCY_U32S ((CHUNK_AMOUNT * CHUNK_AMOUNT * CHUNK_AMOUNT) + 31) / 32

struct Store {
    BlocksA: array<u32, TOTAL_WORLD_SUBCHUNKS>,
    BlocksB: array<u32, TOTAL_WORLD_SUBCHUNKS>,
    Frame: u32,
    Cooldown: float,
    ChunkOccupancyA: array<atomic<u32>, CHUNK_OCCUPANCY_U32S>,
    ChunkOccupancyB: array<atomic<u32>, CHUNK_OCCUPANCY_U32S>,
}

#storage store Store

// Pattern 1: "445" - Stable, crystalline structures (default)
#define SURVIVAL_MASK_1 ((1u<<4u) | (1u<<5u))
#define BIRTH_MASK_1 (1u<<5u)

// Pattern 2: "Amoeba" - Chaotic, organic growth
#define SURVIVAL_MASK_2 ((1u<<9u) | (1u<<10u) | (1u<<11u) | (1u<<12u) | (1u<<13u) | (1u<<14u) | (1u<<15u) | (1u<<16u) | (1u<<17u) | (1u<<18u) | (1u<<19u) | (1u<<20u) | (1u<<21u) | (1u<<22u) | (1u<<23u) | (1u<<24u) | (1u<<25u) | (1u<<26u))
#define BIRTH_MASK_2 ((1u<<5u) | (1u<<6u) | (1u<<7u) | (1u<<12u) | (1u<<13u) | (1u<<15u))

// Pattern 3: "Clouds" - Wispy, cloud-like formations
#define SURVIVAL_MASK_3 ((1u<<13u) | (1u<<14u) | (1u<<15u) | (1u<<16u) | (1u<<17u) | (1u<<18u) | (1u<<19u) | (1u<<20u) | (1u<<21u) | (1u<<22u) | (1u<<23u) | (1u<<24u) | (1u<<25u) | (1u<<26u))
#define BIRTH_MASK_3 ((1u<<13u) | (1u<<14u) | (1u<<17u) | (1u<<18u) | (1u<<19u))

// Pattern 4: "Pyroclastic" - Explosive, expanding patterns
#define SURVIVAL_MASK_4 ((1u<<4u) | (1u<<5u) | (1u<<6u) | (1u<<7u))
#define BIRTH_MASK_4 ((1u<<6u) | (1u<<7u) | (1u<<8u))

// Pattern 5: "Builder 1" - Slow, stable construction
#define SURVIVAL_MASK_5 ((1u<<2u) | (1u<<6u) | (1u<<9u))
#define BIRTH_MASK_5 ((1u<<4u) | (1u<<6u) | (1u<<8u) | (1u<<9u))

// Pattern 6: "Diamoeba" - Diamond-like growth
#define SURVIVAL_MASK_6 ((1u<<5u) | (1u<<6u) | (1u<<7u) | (1u<<8u))
#define BIRTH_MASK_6 ((1u<<5u) | (1u<<6u) | (1u<<7u) | (1u<<8u))

// Pattern 7: "Slow Decay" - Structures slowly dissolve
#define SURVIVAL_MASK_7 ((1u<<5u) | (1u<<6u) | (1u<<7u) | (1u<<8u))
#define BIRTH_MASK_7 (1u<<5u)

// Pattern 8: "Oscillator" - Pulsing, oscillating structures
#define SURVIVAL_MASK_8 ((1u<<4u) | (1u<<5u) | (1u<<6u))
#define BIRTH_MASK_8 ((1u<<5u) | (1u<<6u))

// Pattern 9: "678/567" - Dense, blob-like growth
#define SURVIVAL_MASK_9 ((1u<<6u) | (1u<<7u) | (1u<<8u))
#define BIRTH_MASK_9 ((1u<<5u) | (1u<<6u) | (1u<<7u))

// Pattern 10: "Spiky" - Sharp, crystalline spikes
#define SURVIVAL_MASK_10 ((1u<<7u) | (1u<<8u) | (1u<<9u) | (1u<<10u))
#define BIRTH_MASK_10 ((1u<<4u) | (1u<<5u))

#define DENSITY custom.density
#define SEED 0u

#define MAX_DIST 200

#define EPSILON 0.0000001

#define SPARSE_COLOR vec3f(0.0, 0.4, 1.0)
#define MID_COLOR vec3f(0.1, 1.0, 0.4)
#define DENSE_COLOR vec3f(1.0, 0.1, 0.3)
#define nearColor vec3f(1.0, 0.63, 0.08)
#define farColor vec3f(0.69, 0.07, 0.46)

fn getGlobalSubchunkIndex(x: i32, y: i32, z: i32) -> u32 {
    let wx = ((x % WORLD_SIZE) + WORLD_SIZE) % WORLD_SIZE;
    let wy = ((y % WORLD_SIZE) + WORLD_SIZE) % WORLD_SIZE;
    let wz = ((z % WORLD_SIZE) + WORLD_SIZE) % WORLD_SIZE;
    
    let subChunkX = wx / SUB_CHUNK_X;
    let subChunkY = wy / SUB_CHUNK_Y;
    let subChunkZ = wz / SUB_CHUNK_Z;

    let subChunkIndex = subChunkX + 
                        subChunkY * SUBCHUNKS_PER_WORLD_X + 
                        subChunkZ * SUBCHUNKS_PER_WORLD_X * SUBCHUNKS_PER_WORLD_Y;
    
    return u32(subChunkIndex);
}

fn getLocalSubchunkIndex(lx: u32, ly: u32, lz: u32) -> u32 {
    let subChunkX = lx / SUB_CHUNK_X;
    let subChunkY = ly / SUB_CHUNK_Y;
    let subChunkZ = lz / SUB_CHUNK_Z;
    
    return subChunkX + 
           subChunkY * SUBCHUNKS_WITH_BORDER_X + 
           subChunkZ * SUBCHUNKS_WITH_BORDER_X * SUBCHUNKS_WITH_BORDER_Y;
}

fn getSubchunkBit(x: u32, y: u32, z: u32) -> u32 {
    let localX = x % SUB_CHUNK_X;
    let localY = y % SUB_CHUNK_Y;
    let localZ = z % SUB_CHUNK_Z;

    let bitIndex =  localX + 
                    localY * SUB_CHUNK_X + 
                    localZ * SUB_CHUNK_X * SUB_CHUNK_Y;
    
    return bitIndex;
}

fn applyRules(currentState: u32, neighbors: u32) -> u32 {
    let patternIndex = u32(custom.game_pattern);
    
    var survivalMask: u32;
    var birthMask: u32;
    
    switch (patternIndex) {
        case 0u: {
            survivalMask = SURVIVAL_MASK_1;
            birthMask = BIRTH_MASK_1;
        }
        case 1u: {
            survivalMask = SURVIVAL_MASK_2;
            birthMask = BIRTH_MASK_2;
        }
        case 2u: {
            survivalMask = SURVIVAL_MASK_3;
            birthMask = BIRTH_MASK_3;
        }
        case 3u: {
            survivalMask = SURVIVAL_MASK_4;
            birthMask = BIRTH_MASK_4;
        }
        case 4u: {
            survivalMask = SURVIVAL_MASK_5;
            birthMask = BIRTH_MASK_5;
        }
        case 5u: {
            survivalMask = SURVIVAL_MASK_6;
            birthMask = BIRTH_MASK_6;
        }
        case 6u: {
            survivalMask = SURVIVAL_MASK_7;
            birthMask = BIRTH_MASK_7;
        }
        case 7u: {
            survivalMask = SURVIVAL_MASK_8;
            birthMask = BIRTH_MASK_8;
        }
        case 8u: {
            survivalMask = SURVIVAL_MASK_9;
            birthMask = BIRTH_MASK_9;
        }
        default: {
            survivalMask = SURVIVAL_MASK_10;
            birthMask = BIRTH_MASK_10;
        }
    }
    
    let mask = select(birthMask, survivalMask, currentState == 1u);
    return (mask >> neighbors) & 1u;
}

fn getReadBuffer(index: u32) -> u32 {
    if ((store.Frame & 1u) == 0u) {
        return store.BlocksA[index];
    } else {
        return store.BlocksB[index];
    }
}

fn setWriteBuffer(index: u32, value: u32) {
    if ((store.Frame & 1u) == 0u) {
        store.BlocksB[index] = value;
    } else {
        store.BlocksA[index] = value;
    }
}

fn pcgHash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn random(seed: u32) -> f32 {
    return f32(pcgHash(seed)) / 4294967296.0;
}

fn randomBits(seed: u32, density: f32) -> u32 {
    var result = 0u;
    
    for (var bit = 0u; bit < 32u; bit++) {
        let bitSeed = seed * 31u + bit;
        if (random(bitSeed) < density) {
            result |= (1u << bit);
        }
    }
    
    return result;
}

fn getReadOccupancy(arrayIdx: u32) -> u32 {
    if ((store.Frame & 1u) == 0u) {
        return atomicLoad(&store.ChunkOccupancyA[arrayIdx]);
    } else {
        return atomicLoad(&store.ChunkOccupancyB[arrayIdx]);
    }
}

fn clearWriteOccupancy(arrayIdx: u32) {
    if ((store.Frame & 1u) == 0u) {
        atomicStore(&store.ChunkOccupancyB[arrayIdx], 0u);
    } else {
        atomicStore(&store.ChunkOccupancyA[arrayIdx], 0u);
    }
}

fn setWriteOccupancy(arrayIdx: u32, mask: u32) {
    if ((store.Frame & 1u) == 0u) {
        atomicOr(&store.ChunkOccupancyB[arrayIdx], mask);
    } else {
        atomicOr(&store.ChunkOccupancyA[arrayIdx], mask);
    }
}

#workgroup_count initialize INIT_WG_X 1 1
#dispatch_once initialize
@compute @workgroup_size(256, 1, 1)
fn initialize(@builtin(global_invocation_id) id: vec3u) {
    let index = id.x + SEED;
    if (index >= TOTAL_WORLD_SUBCHUNKS) { return; }
    
    let seed = index * 123456789u + 987654321u;
    let randomPattern = randomBits(seed, DENSITY);
    
    store.BlocksA[index] = randomPattern;
    store.BlocksB[index] = randomPattern;

    if (id.x == 0) {
        store.Cooldown = 0.;
        for (var i = 0u; i < CHUNK_OCCUPANCY_U32S; i++) {
            atomicStore(&store.ChunkOccupancyA[i], 0u);
            atomicStore(&store.ChunkOccupancyB[i], 0u);
        }
        }
}

#workgroup_count update CHUNK_AMOUNT CHUNK_AMOUNT CHUNK_AMOUNT
@compute @workgroup_size(SUBCHUNKS_PER_CHUNK_X, SUBCHUNKS_PER_CHUNK_Y, SUBCHUNKS_PER_CHUNK_Z)
fn update(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id) wid: vec3u
) {
    if (store.Cooldown > 0.) {return;}

    if (lid.x == 0u && lid.y == 0u && lid.z == 0u) {
        atomicStore(&chunkHasActiveCells, 0u);
    }

    let chunkBase = vec3i(wid * vec3u(CHUNK_SIZE));
    
    let threadIdx = lid.x + lid.y * SUBCHUNKS_PER_CHUNK_X + lid.z * SUBCHUNKS_PER_CHUNK_X * SUBCHUNKS_PER_CHUNK_Y;
    
    let loadsPerThread = u32((TOTAL_SUBCHUNKS_TO_LOAD + THREADS_PER_CHUNK - 1) / THREADS_PER_CHUNK);
    
    for (var i = 0u; i < loadsPerThread; i++) {
        let loadIdx = threadIdx + i * THREADS_PER_CHUNK;
        if (loadIdx < TOTAL_SUBCHUNKS_TO_LOAD) {
            let localSubX = loadIdx % SUBCHUNKS_WITH_BORDER_X;
            let localSubY = (loadIdx / SUBCHUNKS_WITH_BORDER_X) % SUBCHUNKS_WITH_BORDER_Y;
            let localSubZ = loadIdx / (SUBCHUNKS_WITH_BORDER_X * SUBCHUNKS_WITH_BORDER_Y);

            let worldSubX = (chunkBase.x / SUB_CHUNK_X) + i32(localSubX) - 1; // Note: -1 is the Border offset
            let worldSubY = (chunkBase.y / SUB_CHUNK_Y) + i32(localSubY) - 1;
            let worldSubZ = (chunkBase.z / SUB_CHUNK_Z) + i32(localSubZ) - 1;

            let wrappedSubX = (worldSubX + SUBCHUNKS_PER_WORLD_X) % SUBCHUNKS_PER_WORLD_X;
            let wrappedSubY = (worldSubY + SUBCHUNKS_PER_WORLD_Y) % SUBCHUNKS_PER_WORLD_Y;
            let wrappedSubZ = (worldSubZ + SUBCHUNKS_PER_WORLD_Z) % SUBCHUNKS_PER_WORLD_Z;
            
            let globalIdx = wrappedSubX + 
                            wrappedSubY * SUBCHUNKS_PER_WORLD_X + 
                            wrappedSubZ * SUBCHUNKS_PER_WORLD_X * SUBCHUNKS_PER_WORLD_Y;
            
            localChunk[loadIdx] = getReadBuffer(u32(globalIdx));
        }
    }

    var result = 0u;
    let lbx = (lid.x + 1u) * SUB_CHUNK_X;
    let lby = (lid.y + 1u) * SUB_CHUNK_Y;
    let lbz = (lid.z + 1u) * SUB_CHUNK_Z;
    let index = getGlobalSubchunkIndex(
        chunkBase.x + i32(lid.x) * SUB_CHUNK_X,
        chunkBase.y + i32(lid.y) * SUB_CHUNK_Y,
        chunkBase.z + i32(lid.z) * SUB_CHUNK_Z
    );
    
    workgroupBarrier();
    
    for (var i = 0u; i < CELLS_PER_SUBCHUNK; i++) {
        let lx = lbx + i % SUB_CHUNK_X;
        let ly = lby + (i / SUB_CHUNK_X) % SUB_CHUNK_Y;
        let lz = lbz + i / (SUB_CHUNK_X * SUB_CHUNK_Y);
        
        var neighbors = 0u;
        for (var dz = -1; dz <= 1; dz++) {
            for (var dy = -1; dy <= 1; dy++) {
                for (var dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0 && dz == 0) { continue; }

                    let cx = u32(i32(lx) + dx);
                    let cy = u32(i32(ly) + dy);
                    let cz = u32(i32(lz) + dz);
                    
                    let lIndex = getLocalSubchunkIndex(cx, cy, cz);
                    let bit = getSubchunkBit(cx, cy, cz);

                    neighbors += (localChunk[lIndex] >> bit) & 1u;
                }
            }
        }

        let lIndex = getLocalSubchunkIndex(lx, ly, lz);
        let bit = getSubchunkBit(lx, ly, lz);
        let currentState = (localChunk[lIndex] >> bit) & 1u;
        
        let newState = applyRules(currentState, neighbors);

        result |= newState << bit;
    }

    setWriteBuffer(index, result);
    
    if (result != 0u) {
        atomicOr(&chunkHasActiveCells, 1u);
    }

    workgroupBarrier();

    if (lid.x == 0u && lid.y == 0u && lid.z == 0u) {
        let hasActive = atomicLoad(&chunkHasActiveCells);
        let chunkIdx = wid.x + wid.y * CHUNK_AMOUNT + wid.z * CHUNK_AMOUNT * CHUNK_AMOUNT;
        let arrayIdx = chunkIdx / 32u;
        let bitIdx = chunkIdx % 32u;
        
        if (hasActive != 0u) {
            setWriteOccupancy(arrayIdx, 1u << bitIdx);
        } else {
            if ((store.Frame & 1u) == 0u) {
                atomicAnd(&store.ChunkOccupancyB[arrayIdx], ~(1u << bitIdx));
            } else {
                atomicAnd(&store.ChunkOccupancyA[arrayIdx], ~(1u << bitIdx));
            }
        }
    }
}

#workgroup_count incrementFrame 1 1 1
@compute @workgroup_size(1, 1, 1)
fn incrementFrame() {
    if (store.Cooldown <= 0.) {
        store.Frame = store.Frame + 1u;
        store.Cooldown = custom.update_every;
    } else {
        store.Cooldown -= time.delta;
    }
}

struct RayHit {
    hit: bool,
    dist: f32,
    pos: vec3i,
    normal: vec3f,
}

fn isChunkOccupied(chunkX: i32, chunkY: i32, chunkZ: i32) -> bool {
    let wx = ((chunkX % CHUNK_AMOUNT) + CHUNK_AMOUNT) % CHUNK_AMOUNT;
    let wy = ((chunkY % CHUNK_AMOUNT) + CHUNK_AMOUNT) % CHUNK_AMOUNT;
    let wz = ((chunkZ % CHUNK_AMOUNT) + CHUNK_AMOUNT) % CHUNK_AMOUNT;
    
    let chunkIdx = u32(wx + wy * CHUNK_AMOUNT + wz * CHUNK_AMOUNT * CHUNK_AMOUNT);
    
    let arrayIdx = chunkIdx / 32u;
    
    let bitIdx = chunkIdx % 32u;
    
    return bool((getReadOccupancy(arrayIdx) >> bitIdx) & 1u);
}

fn ddaChunks(ro: vec3f, rd: vec3f, origin: vec3f) -> RayHit {
    var result: RayHit;
    result.hit = false;
    
    let rayDir = normalize(rd);
    
    var chunkPos = vec3i(
        i32(floor(ro.x / f32(CHUNK_SIZE))),
        i32(floor(ro.y / f32(CHUNK_SIZE))),
        i32(floor(ro.z / f32(CHUNK_SIZE)))
    );
    
    let step = vec3i(
        select(-1, 1, rayDir.x >= 0.0),
        select(-1, 1, rayDir.y >= 0.0),
        select(-1, 1, rayDir.z >= 0.0)
    );
    
    let chunkBoundary = vec3f(
        f32(select(chunkPos.x, chunkPos.x + 1, rayDir.x >= 0.0) * CHUNK_SIZE),
        f32(select(chunkPos.y, chunkPos.y + 1, rayDir.y >= 0.0) * CHUNK_SIZE),
        f32(select(chunkPos.z, chunkPos.z + 1, rayDir.z >= 0.0) * CHUNK_SIZE)
    );
    
    var tMax = (chunkBoundary - ro) / (rayDir + EPSILON);
    
    let tDelta = vec3f(f32(CHUNK_SIZE)) / (abs(rayDir) + EPSILON);

    for (var i = 0; i < CHUNK_AMOUNT * 3; i++) {
        let wrappedChunk = vec3i(
            ((chunkPos.x % CHUNK_AMOUNT) + CHUNK_AMOUNT) % CHUNK_AMOUNT,
            ((chunkPos.y % CHUNK_AMOUNT) + CHUNK_AMOUNT) % CHUNK_AMOUNT,
            ((chunkPos.z % CHUNK_AMOUNT) + CHUNK_AMOUNT) % CHUNK_AMOUNT
        );
        
        if (isChunkOccupied(wrappedChunk.x, wrappedChunk.y, wrappedChunk.z)) {
            let voxelHit = ddaVoxels(ro, rayDir, chunkPos, origin);
            if (voxelHit.hit) {
                return voxelHit;
            }
        }
        
        if (tMax.x < tMax.y) {
            if (tMax.x < tMax.z) {
                chunkPos.x += step.x;
                tMax.x += tDelta.x;
            } else {
                chunkPos.z += step.z;
                tMax.z += tDelta.z;
            }
        } else {
            if (tMax.y < tMax.z) {
                chunkPos.y += step.y;
                tMax.y += tDelta.y;
            } else {
                chunkPos.z += step.z;
                tMax.z += tDelta.z;
            }
        }
        
        let currentDist = min(min(tMax.x, tMax.y), tMax.z);
        if (currentDist > MAX_DIST) {
            break;
        }
    }
    
    return result;
}

fn getCellState(x: i32, y: i32, z: i32) -> bool {
    let wx = ((x % WORLD_SIZE) + WORLD_SIZE) % WORLD_SIZE;
    let wy = ((y % WORLD_SIZE) + WORLD_SIZE) % WORLD_SIZE;
    let wz = ((z % WORLD_SIZE) + WORLD_SIZE) % WORLD_SIZE;
    
    let subChunkIdx = getGlobalSubchunkIndex(wx, wy, wz);
    
    let bitIdx = getSubchunkBit(u32(wx), u32(wy), u32(wz));
    
    let data = getReadBuffer(subChunkIdx);
    
    return bool((data >> bitIdx) & 1u);
}

fn ddaVoxels(ro: vec3f, rd: vec3f, chunk: vec3i, origin: vec3f) -> RayHit {
    var result: RayHit;
    result.hit = false;
    
    let chunkMin = vec3f(chunk * CHUNK_SIZE);
    let chunkMax = chunkMin + vec3f(f32(CHUNK_SIZE));
    
    let invDir = 1.0 / (rd + EPSILON);
    let t0 = (chunkMin - ro) * invDir;
    let t1 = (chunkMax - ro) * invDir;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    let tenter = max(max(max(tmin.x, tmin.y), tmin.z), 0.0);
    let texit = min(min(tmax.x, tmax.y), tmax.z);
    
    if (tenter >= texit) {
        return result;
    }
    
    let startPos = ro + rd * (tenter + 0.001);
    
    var voxelPos = vec3i(floor(startPos));
    
    let step = vec3i(
        select(-1, 1, rd.x >= 0.0),
        select(-1, 1, rd.y >= 0.0),
        select(-1, 1, rd.z >= 0.0)
    );
    
    let voxelBoundary = vec3f(
        f32(select(voxelPos.x, voxelPos.x + 1, rd.x >= 0.0)),
        f32(select(voxelPos.y, voxelPos.y + 1, rd.y >= 0.0)),
        f32(select(voxelPos.z, voxelPos.z + 1, rd.z >= 0.0))
    );

    var tMax = (voxelBoundary - startPos) / (rd + EPSILON);
    let tDelta = 1.0 / (abs(rd) + EPSILON);
    
    var normal = vec3f(0.0);
    let tol = 0.0001;
    if (abs(tenter - tmin.x) < tol) {
        normal = vec3f(-sign(rd.x), 0.0, 0.0);
    } else if (abs(tenter - tmin.y) < tol) {
        normal = vec3f(0.0, -sign(rd.y), 0.0);
    } else {
        normal = vec3f(0.0, 0.0, -sign(rd.z));
    }
    
    for (var i = 0; i < CHUNK_SIZE * 3; i++) {
        if (getCellState(voxelPos.x, voxelPos.y, voxelPos.z)) {
            let distToCamera = length(vec3f(voxelPos) - origin);
            let cameraSphereRadius = custom.camera_sphere * f32(CHUNK_SIZE);
        
            if (distToCamera > cameraSphereRadius) {
                result.hit = true;
                result.pos = voxelPos;
                result.normal = normal;

                let voxelCenter = vec3f(voxelPos) + vec3f(0.5);
                let fadeStart = cameraSphereRadius * 0.8;
                result.dist = length(vec3f(voxelCenter) - ro);

                return result;
            }
        }
        
        if (tMax.x < tMax.y) {
            if (tMax.x < tMax.z) {
                voxelPos.x += step.x;
                tMax.x += tDelta.x;
                normal = vec3f(-f32(step.x), 0.0, 0.0);
            } else {
                voxelPos.z += step.z;
                tMax.z += tDelta.z;
                normal = vec3f(0.0, 0.0, -f32(step.z));
            }
        } else {
            if (tMax.y < tMax.z) {
                voxelPos.y += step.y;
                tMax.y += tDelta.y;
                normal = vec3f(0.0, -f32(step.y), 0.0);
            } else {
                voxelPos.z += step.z;
                tMax.z += tDelta.z;
                normal = vec3f(0.0, 0.0, -f32(step.z));
            }
        }
        
        let localPos = voxelPos - chunk * CHUNK_SIZE;
        if (localPos.x < 0 || localPos.x >= CHUNK_SIZE ||
            localPos.y < 0 || localPos.y >= CHUNK_SIZE ||
            localPos.z < 0 || localPos.z >= CHUNK_SIZE) {
            break;
        }
    }
    
    return result;
}

fn getCellNeighborCount(x: i32, y: i32, z: i32) -> u32 {
    var count = 0u;
    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0 && dz == 0) { continue; }
                if (getCellState(x + dx, y + dy, z + dz)) {
                    count++;
                }
            }
        }
    }
    return count;
}

fn calculateLighting(pos: vec3i, normal: vec3f, viewDir: vec3f, distFromCamera: f32) -> vec3f {
    let sunAngle = time.elapsed * 0.05;
    let lightDir = normalize(vec3f(
        cos(sunAngle) * 0.8,
        0.7 + sin(sunAngle * 0.5) * 0.3,
        sin(sunAngle) * 0.8
    ));
    
    let diffuse = max(dot(normal, lightDir), 0.0);
    let ambient = 0.2;
    
    let halfDir = normalize(lightDir + viewDir);
    let specular = pow(max(dot(normal, halfDir), 0.0), 64.0) * 0.8;
    
    let neighborCount = getCellNeighborCount(pos.x, pos.y, pos.z);
    let densityFactor = f32(neighborCount) / 26.0;
    
    var baseColor: vec3f;
    if (densityFactor < 0.3) {
        baseColor = mix(SPARSE_COLOR, MID_COLOR, densityFactor / 0.2);
    } else {
        baseColor = mix(MID_COLOR, DENSE_COLOR, (densityFactor - 0.2) / 0.7);
    }
    
    let distanceFactor = saturate(distFromCamera / (MAX_DIST * 0.5));
    baseColor = mix(baseColor, mix(nearColor, farColor, distanceFactor), custom.color);
    
    let pulse = sin(f32(pos.x + pos.y + pos.z) * 0.1 + time.elapsed * 2.0) * 0.1 + 1.0;
    baseColor *= pulse;
    
    let rim = pow(1.0 - max(dot(viewDir, normal), 0.0), 1.5) * 0.5;
    
    let lighting = diffuse + ambient + specular * 1.5 + rim;
    return baseColor * lighting;
}

struct Camera {
    pos: vec3f,
    forward: vec3f,
    right: vec3f,
    up: vec3f,
}

fn createCamera(lookFrom: vec3f, lookAt: vec3f, vup: vec3f, fov: f32) -> Camera {
    var cam: Camera;
    cam.pos = lookFrom;
    
    cam.forward = normalize(lookAt - lookFrom);
    cam.right = normalize(cross(cam.forward, vup));
    cam.up = cross(cam.right, cam.forward);
    
    return cam;
}

fn getCameraRay(cam: Camera, uv: vec2f, aspectRatio: f32, fov: f32) -> vec3f {
    let fovScale = tan(fov * 0.5);
    
    return normalize(
        cam.forward + 
        cam.right * uv.x * fovScale * aspectRatio + 
        cam.up * uv.y * fovScale
    );
}

@compute @workgroup_size(16, 16, 1)
fn draw(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);

    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let fx = f32(id.x) + 0.5;
    let fy = f32(id.y) + 0.5;
    let sx = f32(screen_size.x);
    let sy = f32(screen_size.y);
    
    let uv = vec2f((fx - sx * 0.5) / sy, (sy - fy - sy * 0.5) / sy);

    let aspectRatio = sx / sy;
    
    let camDist = f32(WORLD_SIZE) * 1.2;
    let camAngle = time.elapsed * 0.12;
    let verticalAngle = sin(time.elapsed * 0.08) * 0.3;
    let camHeight = f32(WORLD_SIZE) * 0.5 + sin(time.elapsed * 0.1) * f32(WORLD_SIZE) * 0.25;
    
    let lookFrom = vec3f(
        f32(WORLD_SIZE) * 0.5 + cos(camAngle) * cos(verticalAngle) * camDist,
        camHeight,
        f32(WORLD_SIZE) * 0.5 + sin(camAngle) * cos(verticalAngle) * camDist
    );
    let lookAt = vec3f(f32(WORLD_SIZE) * 0.5);
    let vup = vec3f(0.0, 1.0, 0.0);
    
    let cam = createCamera(lookFrom, lookAt, vup, custom.fov);
    let rayDir = getCameraRay(cam, uv, aspectRatio, custom.fov);
    
    var hit = ddaChunks(cam.pos, rayDir, lookFrom);
        
    var col = vec3f(0.0);
    
    
    if (hit.hit) {
        let distToCamera = length(vec3f(hit.pos) - cam.pos);
        let viewDir = -rayDir;
        col = calculateLighting(hit.pos, hit.normal, viewDir, distToCamera);

        let fogStart = MAX_DIST * 0.3;
        let fogEnd = MAX_DIST * 0.8;
        let fogAmount = smoothstep(fogStart, fogEnd, hit.dist);
        
        let fogColor = mix(
            vec3f(0.08, 0.12, 0.20),
            vec3f(0.15, 0.18, 0.25),
            (sin(time.elapsed * 0.05) + 1.0) * 0.5
        );
        
        col = mix(col, fogColor, fogAmount * 0.7);
        
        let neighborCount = getCellNeighborCount(hit.pos.x, hit.pos.y, hit.pos.z);
        let ao = 1.0 - (f32(neighborCount) / 26.0) * 0.3;
        col *= ao;
    } else {
        let sunAngle = time.elapsed * 0.05;
        let sunDir = normalize(vec3f(cos(sunAngle) * 0.8, 0.7, sin(sunAngle) * 0.8));
        let sunDot = max(dot(rayDir, sunDir), 0.0);
        
        let skyTop = vec3f(0.1, 0.2, 0.4);
        let skyHorizon = vec3f(0.15, 0.25, 0.35);
        let skyBottom = vec3f(0.02, 0.05, 0.1);
        
        let upness = rayDir.y;
        var skyColor: vec3f;
        if (upness > 0.0) {
            skyColor = mix(skyHorizon, skyTop, upness);
        } else {
            skyColor = mix(skyHorizon, skyBottom, -upness);
        }
        
        let sunGlow = pow(sunDot, 256.0) * vec3f(1.0, 0.9, 0.7);
        let sunHalo = pow(sunDot, 8.0) * 0.1 * vec3f(1.0, 0.95, 0.8);
        
        col = skyColor + sunGlow + sunHalo;
    }

    col = col / (col + vec3f(0.8));
    
    col = pow(col, vec3f(0.9, 0.95, 1.0));
        
    let vignetteStrength = length(uv) * 1.5;
    col *= 1.0 - vignetteStrength * vignetteStrength * 0.3;
    
    let noise = random(u32(id.x + id.y * screen_size.x + u32(time.elapsed * 1000.0)));
    col += (noise - 0.5) * 0.02;

    textureStore(screen, int2(id.xy), vec4f(saturate(col), 1.0));
}