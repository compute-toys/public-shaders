#define TAU 6.28318530718
#define IDENTITY_4X4 mat4x4<f32>(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)

#define TORUS_RADIUS 1.5
#define TORUS_THICKNESS 0.5
#define TORUS_MESH_RES_U 128
#define TORUS_MESH_RES_V 64
#define TORUS_SUM_RES_U 2048
#define TORUS_SUM_RES_V 512
#define TORUS_SUM_RES_R 64

#define SPHERE_MESH_RES 8
#define SPHERE_RADIUS 0.25
#calcdefine SPHERE_VERTICES_OFFSET ((TORUS_MESH_RES_U + 1) * (TORUS_MESH_RES_V + 1))
#calcdefine SPHERE_INDICES_OFFSET (TORUS_MESH_RES_U * TORUS_MESH_RES_V * 2)

#calcdefine VERTEX_COUNT (SPHERE_VERTICES_OFFSET + 6 * (SPHERE_MESH_RES + 1)**2)
#calcdefine TRIANGLE_COUNT (SPHERE_INDICES_OFFSET + 12 * SPHERE_MESH_RES**2)

#storage memory Memory
#storage globalScratch array<vec3<f32>>

var<workgroup> localScratch: array<vec3<f32>, 256>;

struct Memory {
    moonPosition: vec3<f32>,
    moonVelocity: vec3<f32>,
    moonAcceleration: vec3<f32>,
    planetTransform: mat4x4<f32>,
    moonTransform: mat4x4<f32>,
    worldToClipTransform: mat4x4<f32>,
    vertices: array<Vertex, VERTEX_COUNT>,
    indices: array<vec3<u32>, TRIANGLE_COUNT>,
    fb: array<array<array<atomic<u32>, 3>, SCREEN_WIDTH>, SCREEN_HEIGHT>
}

struct Vertex {
    position: vec3<f32>,
    normal: vec3<f32>,
    uv: vec2<f32>
}

struct Triangle {
    vertices: mat3x3<f32>,
    normals: mat3x3<f32>,
    uvs: mat3x2<f32>
}

fn shadeFragment(fragCoord: vec2<i32>, tri: ptr<function, Triangle>, bary: vec3<f32>) -> vec3<f32> {
    let light = normalize(vec3<f32>(1.0, 0.5, 3.0));
    let normal = normalize((*tri).normals * bary);
    let albedo = textureSampleLevel(channel0, bilinear, (*tri).uvs * bary, 0).rgb;
    let shade = albedo * max(0.05, dot(light, normal));
    return pow(shade, vec3<f32>(1.0 / 2.2));
    //let bluenoise = textureLoad(channel1, fragCoord % vec2<i32>(textureDimensions(channel1)), 0).r * 2.0 - 1.0;
    //let dither = (0.5 + sign(bluenoise) * (1.0 - sqrt(1.0 - abs(bluenoise)))) / 255.0;
    //return clamp(shade + dither, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn sampleTorusGravity(uvr: vec3<f32>) -> vec3<f32> {
    const k = 1.0;

    let co = cos(uvr.xy);
    let si = sin(uvr.xy);
    let t = TORUS_RADIUS + uvr.z * co.y;
    let p = vec3<f32>(t * co.x, uvr.z * si.y, -t * si.x);

    let to = p - memory.moonPosition;
    let r2 = dot(to, to);
    let accel = k * to / (r2 * sqrt(r2));

    let dV = t * uvr.z * ((TAU / f32(TORUS_SUM_RES_U)) * (TAU / f32(TORUS_SUM_RES_V)) * (TORUS_THICKNESS / f32(TORUS_SUM_RES_R)));
    return accel * dV;
}

fn ndcToScreenSpace(p: vec2<f32>) -> vec2<i32> {
    return vec2<i32>(
        i32((0.5 + 0.5 * p.x) * f32(SCREEN_WIDTH)),
        i32((0.5 - 0.5 * p.y) * f32(SCREEN_HEIGHT))
    );
}

fn translate(offset: vec3<f32>) -> mat4x4<f32> {
    return mat4x4<f32>(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        offset.x, offset.y, offset.z, 1.0
    );
}

fn rotate(angle: f32, i: i32, j: i32) -> mat4x4<f32> {
    var m = IDENTITY_4X4;
    let co = cos(angle);
    let si = sin(angle);
    m[i][i] = co;
    m[i][j] = si;
    m[j][i] = -si;
    m[j][j] = co;
    return m;
}

fn perspective(focalLength: f32, aspectRatio: f32, zNear: f32, zFar: f32) -> mat4x4<f32> {
    let zScale = zFar / (zNear - zFar);
    return mat4x4<f32>(
        focalLength / aspectRatio, 0.0, 0.0, 0.0,
        0.0, focalLength, 0.0, 0.0,
        0.0, 0.0, zScale, -1.0,
        0.0, 0.0, zNear * zScale, 0.0
    );
}

#dispatch_once initSimulation
#workgroup_count initSimulation 1 1 1
@compute @workgroup_size(1)
fn initSimulation() {
    memory.moonPosition = vec3<f32>(3.0, 0.0, 0.0);
    memory.moonVelocity = vec3<f32>(0.2, 0.8, 0.3);
}

#workgroup_count calcTransforms 1 1 1
@compute @workgroup_size(1)
fn calcTransforms() {
    // NOTE: only rotations and translations work due to assumptions to simplify normal transformation
    memory.planetTransform = IDENTITY_4X4;
    memory.moonTransform = translate(memory.moonPosition);

    let cameraRotation = vec2<f32>(mouse.pos - vec2<i32>(SCREEN_WIDTH, SCREEN_HEIGHT) / 2) / f32(SCREEN_HEIGHT) * TAU / 2.0;
    let worldToView = translate(vec3<f32>(0.0, 0.0, -custom.cameraDistance)) *
        rotate(-cameraRotation.y, 2, 1) *
        rotate(-cameraRotation.x, 0, 2);

    let aspectRatio = f32(SCREEN_WIDTH) / f32(SCREEN_HEIGHT);
    let viewToClip = perspective(custom.focalLength, aspectRatio, custom.zNear, custom.zFar);
    memory.worldToClipTransform = viewToClip * worldToView;
}

//#dispatch_once genTorusVertices
#workgroup_count genTorusVertices 9 5 1
@compute @workgroup_size(16, 16)
fn genTorusVertices(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id.xy > vec2<u32>(TORUS_MESH_RES_U, TORUS_MESH_RES_V))) { return; }

    let st = vec2<f32>(id.xy) / vec2<f32>(TORUS_MESH_RES_U, TORUS_MESH_RES_V);
    let uv = st * TAU;
    let co = cos(uv);
    let si = sin(uv);
    let r = TORUS_RADIUS + TORUS_THICKNESS * co.y;
    let position = vec3<f32>(r * co.x, TORUS_THICKNESS * si.y, -r * si.x);
    let normal = vec3<f32>(co.x * co.y, si.y, -si.x * co.y);

    let index = id.y * (TORUS_MESH_RES_U + 1) + id.x;
    memory.vertices[index] = Vertex(
        (memory.planetTransform * vec4<f32>(position, 1.0)).xyz,
        (memory.planetTransform * vec4<f32>(normal, 0.0)).xyz,
        st.yx * vec2<f32>(1.0, 9.0 / 11.0 - 1.0 / 5632.0)
    );
}

#dispatch_once genTorusIndices
#workgroup_count genTorusIndices 8 4 1
@compute @workgroup_size(16, 16)
fn genTorusIndices(@builtin(global_invocation_id) id: vec3<u32>) {
    let corners = vec4<u32>(id.xy, id.xy + 1);
    let quadIndices = corners.yyww * (TORUS_MESH_RES_U + 1) + corners.xzzx;
    let indicesOffset = (id.y * TORUS_MESH_RES_U + id.x) * 2;
    memory.indices[indicesOffset + 0] = quadIndices.xyz;
    memory.indices[indicesOffset + 1] = quadIndices.zwx;
}

//#dispatch_once genSphereVertices
#workgroup_count genSphereVertices 1 1 6
@compute @workgroup_size(9, 9)
fn genSphereVertices(@builtin(global_invocation_id) id: vec3<u32>) {
    var st = vec2<f32>(id.xy) / f32(SPHERE_MESH_RES);
    var normal = normalize(vec3<f32>(st * 2.0 - 1.0, 1.0));

    switch (id.z) {
        case 0: {
            normal = vec3<f32>(normal.z, normal.y, -normal.x);
            //st += vec2<f32>(0.0, 0.0);
            break;
        }

        case 1: {
            normal = vec3<f32>(-normal.z, normal.y, normal.x);
            st += vec2<f32>(0.0, 1.0);
            break;
        }

        case 2: {
            normal = vec3<f32>(normal.x, normal.z, -normal.y);
            st += vec2<f32>(1.0, 0.0);
            break;
        }

        case 3: {
            normal = vec3<f32>(normal.x, -normal.z, normal.y);
            st += vec2<f32>(1.0, 1.0);
            break;
        }

        case 4: {
            //normal = vec3<f32>(normal.x, normal.y, normal.z);
            st += vec2<f32>(2.0, 0.0);
            break;
        }

        default: {
            normal = vec3<f32>(-normal.x, normal.y, -normal.z);
            st += vec2<f32>(2.0, 1.0);
            break;
        }
    }

    let position = normal * SPHERE_RADIUS;
    let index = id.z * (SPHERE_MESH_RES + 1) * (SPHERE_MESH_RES + 1) + id.y * (SPHERE_MESH_RES + 1) + id.x + SPHERE_VERTICES_OFFSET;
    memory.vertices[index] = Vertex(
        (memory.moonTransform * vec4<f32>(position, 1.0)).xyz,
        (memory.moonTransform * vec4<f32>(normal, 0.0)).xyz,
        fma(st, vec2<f32>(1.0 / 3.0, 1.0 / 11.0), vec2<f32>(0.0, 9.0 / 11.0))
    );
}

#dispatch_once genSphereIndices
#workgroup_count genSphereIndices 1 1 6
@compute @workgroup_size(8, 8)
fn genSphereIndices(@builtin(global_invocation_id) id: vec3<u32>) {
    let corners = vec4<u32>(id.xy, id.xy + 1);
    let quadIndices = id.z * (SPHERE_MESH_RES + 1) * (SPHERE_MESH_RES + 1) + corners.yyww * (SPHERE_MESH_RES + 1) + corners.xzzx + SPHERE_VERTICES_OFFSET;
    let indicesOffset = (id.z * SPHERE_MESH_RES * SPHERE_MESH_RES + id.y * SPHERE_MESH_RES + id.x) * 2 + SPHERE_INDICES_OFFSET;
    memory.indices[indicesOffset + 0] = quadIndices.xyz;
    memory.indices[indicesOffset + 1] = quadIndices.zwx;
}

@compute @workgroup_size(16, 16)
fn clearFramebuffer(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= SCREEN_WIDTH || id.y >= SCREEN_HEIGHT) { return; }
    atomicStore(&memory.fb[id.y][id.x][0], 0xffffff00);
    atomicStore(&memory.fb[id.y][id.x][1], 0xffffff00);
    atomicStore(&memory.fb[id.y][id.x][2], 0xffffff00);
}

#workgroup_count drawTriangles 67 1 1
@compute @workgroup_size(256)
fn drawTriangles(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= TRIANGLE_COUNT) { return; }

    let indices = memory.indices[id.x];
    let v0 = memory.vertices[indices.x];
    let v1 = memory.vertices[indices.y];
    let v2 = memory.vertices[indices.z];
    var tri = Triangle(
        mat3x3<f32>(v0.position, v1.position, v2.position),
        mat3x3<f32>(v0.normal, v1.normal, v2.normal),
        mat3x2<f32>(v0.uv, v1.uv, v2.uv)
    );

    // Transform to clip space
    // TODO: actually clip triangles against frustum
    let clipSpaceVerts = mat3x4<f32>(
        memory.worldToClipTransform * vec4<f32>(tri.vertices[0], 1.0),
        memory.worldToClipTransform * vec4<f32>(tri.vertices[1], 1.0),
        memory.worldToClipTransform * vec4<f32>(tri.vertices[2], 1.0)
    );

    let perspFactor = 1.0 / vec3<f32>(
        clipSpaceVerts[0].w,
        clipSpaceVerts[1].w,
        clipSpaceVerts[2].w
    );

    // Transform to NDC (Normalized Device Coordinates)
    let ndc = mat3x3<f32>(
        clipSpaceVerts[0].xyz * perspFactor[0],
        clipSpaceVerts[1].xyz * perspFactor[1],
        clipSpaceVerts[2].xyz * perspFactor[2]
    );

    // Map to screen space and hold onto the index of the vertex
    var screenA = vec3<i32>(ndcToScreenSpace(ndc[0].xy), 0);
    var screenB = vec3<i32>(ndcToScreenSpace(ndc[1].xy), 1);
    var screenC = vec3<i32>(ndcToScreenSpace(ndc[2].xy), 2);

    // Cull backfacing triangles
    let deltaBA = screenB.xy - screenA.xy;
    let deltaCA = screenC.xy - screenA.xy;
    if (deltaBA.x * deltaCA.y > deltaBA.y * deltaCA.x) { return; }

    // Sort vertices in screen space by y coordinate
    if (screenA.y > screenC.y) { let tmp = screenA; screenA = screenC; screenC = tmp; }
    if (screenA.y > screenB.y) { let tmp = screenA; screenA = screenB; screenB = tmp; }
    if (screenB.y > screenC.y) { let tmp = screenB; screenB = screenC; screenC = tmp; }

    // Prepare barycentric coordinates for perspective correct interpolation
    var baryA = vec4<f32>(0.0);
    var baryB = vec4<f32>(0.0);
    var baryC = vec4<f32>(0.0);
    baryA.w = perspFactor[screenA[2]];
    baryB.w = perspFactor[screenB[2]];
    baryC.w = perspFactor[screenC[2]];
    baryA[screenA[2]] = baryA.w;
    baryB[screenB[2]] = baryB.w;
    baryC[screenC[2]] = baryC.w;

    let depths = vec3<f32>(ndc[0].z, ndc[1].z, ndc[2].z);

    if (screenC.y > screenA.y) {
        let deltaBA = vec2<f32>(screenB.xy - screenA.xy);
        let deltaCB = vec2<f32>(screenC.xy - screenB.xy);
        let deltaCA = vec2<f32>(screenC.xy - screenA.xy);

        // Left edge
        var leftX = f32(screenA.x);
        var leftDeltaX = deltaCA.x / deltaCA.y;
        var leftBary = baryA;
        var leftDeltaBary = (baryC - baryA) / deltaCA.y;

        // Right edge
        var rightX: f32;
        var rightDeltaX: f32;
        var rightBary: vec4<f32>;
        var rightDeltaBary: vec4<f32>;

        // Sort first edge pair
        let swapEdgePair = i32(leftX + leftDeltaX * deltaBA.y) > screenB.x;
        if (swapEdgePair) {
            rightX = leftX;
            rightDeltaX = leftDeltaX;
            rightBary = leftBary;
            rightDeltaBary = leftDeltaBary;
        }

        if (screenB.y > screenA.y) {
            // Set short edge to edge A->B
            if (swapEdgePair) {
                leftX = f32(screenA.x);
                leftDeltaX = deltaBA.x / deltaBA.y;
                leftBary = baryA;
                leftDeltaBary = (baryB - baryA) / deltaBA.y;
            } else {
                rightX = f32(screenA.x);
                rightDeltaX = deltaBA.x / deltaBA.y;
                rightBary = baryA;
                rightDeltaBary = (baryB - baryA) / deltaBA.y;
            }

            // Draw upper half
            for (var y = screenA.y; y < screenB.y; y++) {
                var bary = leftBary;
                let deltaBary = (rightBary - leftBary) / (rightX - leftX);
                for (var x = i32(leftX); x < i32(rightX); x++) {
                    let worldBary = bary.xyz / bary.w;
                    let depthBits = u32(dot(depths, worldBary) * f32(0xffffff)) << 8;
                    let shade = shadeFragment(vec2<i32>(x, y), &tri, worldBary);
                    atomicMin(&memory.fb[y][x][0], depthBits | u32(shade.r * f32(0xff)));
                    atomicMin(&memory.fb[y][x][1], depthBits | u32(shade.g * f32(0xff)));
                    atomicMin(&memory.fb[y][x][2], depthBits | u32(shade.b * f32(0xff)));
                    bary += deltaBary;
                }

                leftX += leftDeltaX;
                rightX += rightDeltaX;
                leftBary += leftDeltaBary;
                rightBary += rightDeltaBary;
            }
        }

        if (screenC.y > screenB.y) {
            // Set short edge to edge B->C
            if (swapEdgePair) {
                leftX = f32(screenB.x);
                leftDeltaX = deltaCB.x / deltaCB.y;
                leftBary = baryB;
                leftDeltaBary = (baryC - baryB) / deltaCB.y;
            } else {
                rightX = f32(screenB.x);
                rightDeltaX = deltaCB.x / deltaCB.y;
                rightBary = baryB;
                rightDeltaBary = (baryC - baryB) / deltaCB.y;
            }

            // Draw lower half
            for (var y = screenB.y; y < screenC.y; y++) {
                var bary = leftBary;
                let deltaBary = (rightBary - leftBary) / (rightX - leftX);
                for (var x = i32(leftX); x < i32(rightX); x++) {
                    let worldBary = bary.xyz / bary.w;
                    let depthBits = u32(dot(depths, worldBary) * f32(0xffffff)) << 8;
                    let shade = shadeFragment(vec2<i32>(x, y), &tri, worldBary);
                    atomicMin(&memory.fb[y][x][0], depthBits | u32(shade.r * f32(0xff)));
                    atomicMin(&memory.fb[y][x][1], depthBits | u32(shade.g * f32(0xff)));
                    atomicMin(&memory.fb[y][x][2], depthBits | u32(shade.b * f32(0xff)));
                    bary += deltaBary;
                }

                leftX += leftDeltaX;
                rightX += rightDeltaX;
                leftBary += leftDeltaBary;
                rightBary += rightDeltaBary;
            }
        }
    }
}

@compute @workgroup_size(16, 16)
fn decodeImage(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= SCREEN_WIDTH || id.y >= SCREEN_HEIGHT) { return; }

    let color = vec3<f32>(vec3<u32>(
        atomicLoad(&memory.fb[id.y][id.x][0]),
        atomicLoad(&memory.fb[id.y][id.x][1]),
        atomicLoad(&memory.fb[id.y][id.x][2])
    ) & vec3<u32>(0xff)) / f32(0xff);

    textureStore(screen, id.xy, vec4<f32>(pow(color, vec3<f32>(2.2)), 1.0));
}

#workgroup_count integralReduction0 4 512 64
@compute @workgroup_size(256)
fn integralReduction0(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let scale = vec3<f32>(TAU, TAU, TORUS_THICKNESS) / vec3<f32>(TORUS_SUM_RES_U, TORUS_SUM_RES_V, TORUS_SUM_RES_R);
    let uvr0 = vec3<f32>(gid) * scale;
    let uvr1 = vec3<f32>(gid + vec3<u32>(TORUS_SUM_RES_U / 2, 0, 0)) * scale;
    localScratch[lid.x] = sampleTorusGravity(uvr0) + sampleTorusGravity(uvr1);
    workgroupBarrier();

    for (var n = 128u; n > 0; n >>= 1) {
        if (lid.x < n) { localScratch[lid.x] += localScratch[lid.x + n]; }
        workgroupBarrier();
    }

    if (lid.x == 0) {
        let index = (wgid.x * TORUS_SUM_RES_V + wgid.y) * TORUS_SUM_RES_R + wgid.z;
        globalScratch[index] = localScratch[0];
    }
}

#workgroup_count integralReduction1 256 1 1
@compute @workgroup_size(256)
fn integralReduction1(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    localScratch[lid.x] = globalScratch[gid.x] + globalScratch[gid.x + 65536];
    workgroupBarrier();

    for (var n = 128u; n > 0; n >>= 1) {
        if (lid.x < n) { localScratch[lid.x] += localScratch[lid.x + n]; }
        workgroupBarrier();
    }

    if (lid.x == 0) { globalScratch[wgid.x] = localScratch[0]; }
}

#workgroup_count integralReduction2 1 1 1
@compute @workgroup_size(128)
fn integralReduction2(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    localScratch[gid.x] = globalScratch[gid.x] + globalScratch[gid.x + 128];
    workgroupBarrier();

    for (var n = 64u; n > 0; n >>= 1) {
        if (gid.x < n) { localScratch[gid.x] += localScratch[gid.x + n]; }
        workgroupBarrier();
    }

    if (gid.x == 0) { memory.moonAcceleration = localScratch[0]; }
}

#workgroup_count stepSimulation 1 1 1
@compute @workgroup_size(1)
fn stepSimulation() {
    memory.moonVelocity += memory.moonAcceleration * time.delta;
    memory.moonPosition += memory.moonVelocity * time.delta;
}