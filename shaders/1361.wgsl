#define VERTEX_COUNT 2898
#define VERTEX_WGSIZE 16
#define VERTEX_WGCOUNT 182

#define TRIANGLE_COUNT 966
#define TRIANGLE_WGSIZE 16
#define TRIANGLE_WGCOUNT 61

#storage memory Memory

struct Memory {
    triangles: array<Triangle, TRIANGLE_COUNT>,
    fb: array<array<array<atomic<u32>, 3>, SCREEN_WIDTH>, SCREEN_HEIGHT>
}

struct Triangle {
    vertices: mat3x3<f32>,
    normals: mat3x3<f32>,
    uvs: mat3x2<f32>
}

fn worldToClipSpace(worldSpacePos: vec3<f32>) -> vec4<f32> {
    let viewSpacePos = worldSpacePos + vec3<f32>(0.0, 0.0, -3.0);
    let aspectRatio = f32(SCREEN_WIDTH) / f32(SCREEN_HEIGHT);
    return vec4<f32>(
        viewSpacePos.x / aspectRatio,
        viewSpacePos.y,
        (viewSpacePos.z + custom.zNear) * custom.zFar / (custom.zNear - custom.zFar),
        -viewSpacePos.z
    );
}

fn ndcToScreenSpace(ndc: vec2<f32>) -> vec2<i32> {
    return vec2<i32>(
        i32((0.5 + 0.5 * ndc.x) * f32(SCREEN_WIDTH)),
        i32((0.5 - 0.5 * ndc.y) * f32(SCREEN_HEIGHT))
    );
}

fn shadeFragment(tri: ptr<function, Triangle>, bary: vec3<f32>) -> vec3<f32> {
    let pos = (*tri).vertices * bary;
    let nor = normalize((*tri).normals * bary);
    let uv = (*tri).uvs * bary;
    let diffuse = max(0.1, dot(nor, normalize(vec3<f32>(custom.lightX, custom.lightY, custom.lightZ))));
    return textureSampleLevel(channel1, bilinear, uv, 0).rgb * diffuse;
}

#workgroup_count genGeometry VERTEX_WGCOUNT 1 1
@compute @workgroup_size(VERTEX_WGSIZE)
fn genGeometry(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= VERTEX_COUNT) { return; }

    let triIndex = id.x / 3;
    let vertIndex = id.x % 3;
    let dataOffs = (2 - vertIndex) * 3;
    var vertex = pow(textureLoad(channel0, vec2<u32>(triIndex, dataOffs), 0).xyz, vec3<f32>(1.0 / 2.2)) * 4.0 - 2.0;
    var normal = pow(textureLoad(channel0, vec2<u32>(triIndex, dataOffs + 1), 0).xyz, vec3<f32>(1.0 / 2.2)) * 2.0 - 1.0;
    var uv = pow(textureLoad(channel0, vec2<u32>(triIndex, dataOffs + 2), 0).xy, vec2<f32>(1.0 / 2.2));

    let co = cos(time.elapsed);
    let si = sin(time.elapsed);
    let rotation = mat3x3<f32>(co, 0.0, si, 0.0, 1.0, 0.0, -si, 0.0, co);
    vertex = rotation * vertex;
    normal = rotation * normal;

    memory.triangles[triIndex].vertices[vertIndex] = vertex;
    memory.triangles[triIndex].normals[vertIndex] = normal;
    memory.triangles[triIndex].uvs[vertIndex] = uv;
}

@compute @workgroup_size(16, 16)
fn clearFramebuffer(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= SCREEN_WIDTH || id.y >= SCREEN_HEIGHT) { return; }
    atomicStore(&memory.fb[id.y][id.x][0], 0xffffff00);
    atomicStore(&memory.fb[id.y][id.x][1], 0xffffff00);
    atomicStore(&memory.fb[id.y][id.x][2], 0xffffff00);
}

#workgroup_count drawTriangles TRIANGLE_WGCOUNT 1 1
@compute @workgroup_size(TRIANGLE_WGSIZE)
fn drawTriangles(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= TRIANGLE_COUNT) { return; }
    var tri = memory.triangles[id.x];

    // Transform to clip space
    // TODO: actually clip triangles against frustum
    let clipSpaceVerts = mat3x4<f32>(
        worldToClipSpace(tri.vertices[0]),
        worldToClipSpace(tri.vertices[1]),
        worldToClipSpace(tri.vertices[2])
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
                    let shade = shadeFragment(&tri, worldBary);
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
                    let shade = shadeFragment(&tri, worldBary);
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
    textureStore(screen, id.xy, vec4<f32>(
        f32(atomicLoad(&memory.fb[id.y][id.x][0]) & 0xff) / f32(0xff),
        f32(atomicLoad(&memory.fb[id.y][id.x][1]) & 0xff) / f32(0xff),
        f32(atomicLoad(&memory.fb[id.y][id.x][2]) & 0xff) / f32(0xff),
        1.0
    ));
}