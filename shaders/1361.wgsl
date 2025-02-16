#define TRIANGLE_COUNT 966
#define TRIANGLE_WGSIZE 16
#define TRIANGLE_WGCOUNT 61

#define VERTEX_COUNT 2898
#define VERTEX_WGSIZE 16
#define VERTEX_WGCOUNT 182

#storage triangles array<Triangle, TRIANGLE_COUNT>
#storage framebuf array<array<array<atomic<u32>, 3>, SCREEN_WIDTH>, SCREEN_HEIGHT>

struct Triangle {
    vertices: mat3x3<f32>,
    normals: mat3x3<f32>,
    uvs: mat3x2<f32>
}

fn worldToViewSpace(p: vec3<f32>) -> vec3<f32> {
    return p + vec3<f32>(0.0, 0.0, -3.0);
}

fn viewToClipSpace(p: vec3<f32>) -> vec3<f32> {
    var proj = vec3<f32>(p.xy, mix(-custom.zNear, custom.zFar, (-p.z - custom.zNear) / (custom.zFar - custom.zNear))) / -p.z;
    proj.x *= f32(SCREEN_HEIGHT) / f32(SCREEN_WIDTH);
    return proj;
}

fn ndcToScreenSpace(p: vec2<f32>) -> vec2<i32> {
    return vec2<i32>(
        i32((0.5 + 0.5 * p.x) * f32(SCREEN_WIDTH)),
        i32((0.5 - 0.5 * p.y) * f32(SCREEN_HEIGHT))
    );
}

fn shadePixel(tri: ptr<function, Triangle>, bary: vec3<f32>) -> vec3<f32> {
    var pos = (*tri).vertices * bary;
    var nor = normalize((*tri).normals * bary);
    var uv = (*tri).uvs * bary;
    var diffuse = max(0.1, dot(nor, normalize(vec3<f32>(custom.lightX, custom.lightY, custom.lightZ))));
    return textureSampleLevel(channel1, bilinear, uv, 0).rgb * diffuse;
}

#workgroup_count genGeometry VERTEX_WGCOUNT 1 1
@compute @workgroup_size(VERTEX_WGSIZE)
fn genGeometry(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= VERTEX_COUNT) { return; }

    var triIndex = id.x / 3;
    var vertIndex = id.x % 3;
    var dataOffs = (2 - vertIndex) * 3;
    var vertex = pow(textureLoad(channel0, vec2<u32>(triIndex, dataOffs), 0).xyz, vec3<f32>(1.0 / 2.2)) * 4.0 - 2.0;
    var normal = pow(textureLoad(channel0, vec2<u32>(triIndex, dataOffs + 1), 0).xyz, vec3<f32>(1.0 / 2.2)) * 2.0 - 1.0;
    var uv = pow(textureLoad(channel0, vec2<u32>(triIndex, dataOffs + 2), 0).xy, vec2<f32>(1.0 / 2.2));

    var co = cos(time.elapsed);
    var si = sin(time.elapsed);
    var rotation = mat3x3<f32>(co, 0.0, si, 0.0, 1.0, 0.0, -si, 0.0, co);
    vertex = rotation * vertex;
    normal = rotation * normal;

    triangles[triIndex].vertices[vertIndex] = vertex;
    triangles[triIndex].normals[vertIndex] = normal;
    triangles[triIndex].uvs[vertIndex] = uv;
}

@compute @workgroup_size(16, 16)
fn clearFramebuffer(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= SCREEN_WIDTH || id.y >= SCREEN_HEIGHT) { return; }
    atomicStore(&framebuf[id.y][id.x][0], 0xffffff00);
    atomicStore(&framebuf[id.y][id.x][1], 0xffffff00);
    atomicStore(&framebuf[id.y][id.x][2], 0xffffff00);
}

#workgroup_count drawTriangles TRIANGLE_WGCOUNT 1 1
@compute @workgroup_size(TRIANGLE_WGSIZE)
fn drawTriangles(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= TRIANGLE_COUNT) { return; }
    var tri = triangles[id.x];

    // Transform to view space
    var viewSpaceVerts: mat3x3<f32>;
    viewSpaceVerts[0] = worldToViewSpace(tri.vertices[0]);
    viewSpaceVerts[1] = worldToViewSpace(tri.vertices[1]);
    viewSpaceVerts[2] = worldToViewSpace(tri.vertices[2]);

    // Transform to clip space
    // TODO: actually clip triangles against frustum
    var clipSpaceVerts: mat3x3<f32>;
    clipSpaceVerts[0] = viewToClipSpace(viewSpaceVerts[0]);
    clipSpaceVerts[1] = viewToClipSpace(viewSpaceVerts[1]);
    clipSpaceVerts[2] = viewToClipSpace(viewSpaceVerts[2]);

    // Cull backfacing triangles
    var perp = cross(clipSpaceVerts[1] - clipSpaceVerts[0], clipSpaceVerts[2] - clipSpaceVerts[0]);
    if (perp.z < 0.0) { return; }

    // Map to screen space and hold onto the index of the vertex
    var screenA = vec3<i32>(ndcToScreenSpace(clipSpaceVerts[0].xy), 0);
    var screenB = vec3<i32>(ndcToScreenSpace(clipSpaceVerts[1].xy), 1);
    var screenC = vec3<i32>(ndcToScreenSpace(clipSpaceVerts[2].xy), 2);

    // Depth values remapped to [0...1] range
    var depths = 0.5 + 0.5 * vec3<f32>(clipSpaceVerts[0].z, clipSpaceVerts[1].z, clipSpaceVerts[2].z);

    // Sort vertices in screen space by y coordinate
    if (screenA.y > screenC.y) { var tmp = screenA; screenA = screenC; screenC = tmp; }
    if (screenA.y > screenB.y) { var tmp = screenA; screenA = screenB; screenB = tmp; }
    if (screenB.y > screenC.y) { var tmp = screenB; screenB = screenC; screenC = tmp; }

    // Prepare barycentric coordinates for perspective correct interpolation
    var baryA = vec4<f32>(0.0);
    var baryB = vec4<f32>(0.0);
    var baryC = vec4<f32>(0.0);
    baryA.w = 1.0 / viewSpaceVerts[screenA[2]].z;
    baryB.w = 1.0 / viewSpaceVerts[screenB[2]].z;
    baryC.w = 1.0 / viewSpaceVerts[screenC[2]].z;
    baryA[screenA[2]] = baryA.w;
    baryB[screenB[2]] = baryB.w;
    baryC[screenC[2]] = baryC.w;

    if (screenC.y > screenA.y) {
        var deltaBA = vec2<f32>(screenB.xy - screenA.xy);
        var deltaCB = vec2<f32>(screenC.xy - screenB.xy);
        var deltaCA = vec2<f32>(screenC.xy - screenA.xy);

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
        var swapEdgePair = i32(leftX + leftDeltaX * deltaBA.y) > screenB.x;
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
                var deltaBary = (rightBary - leftBary) / (rightX - leftX);
                for (var x = u32(leftX); x < u32(rightX); x++) {
                    var worldBary = bary.xyz / bary.w;
                    var depthBits = u32(dot(depths, worldBary) * f32(0xffffff)) << 8;
                    var shade = shadePixel(&tri, worldBary);
                    atomicMin(&framebuf[y][x][0], depthBits | u32(shade.r * f32(0xff)));
                    atomicMin(&framebuf[y][x][1], depthBits | u32(shade.g * f32(0xff)));
                    atomicMin(&framebuf[y][x][2], depthBits | u32(shade.b * f32(0xff)));
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
                var deltaBary = (rightBary - leftBary) / (rightX - leftX);
                for (var x = u32(leftX); x < u32(rightX); x++) {
                    var worldBary = bary.xyz / bary.w;
                    var depthBits = u32(dot(depths, worldBary) * f32(0xffffff)) << 8;
                    var shade = shadePixel(&tri, worldBary);
                    atomicMin(&framebuf[y][x][0], depthBits | u32(shade.r * f32(0xff)));
                    atomicMin(&framebuf[y][x][1], depthBits | u32(shade.g * f32(0xff)));
                    atomicMin(&framebuf[y][x][2], depthBits | u32(shade.b * f32(0xff)));
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
        f32(atomicLoad(&framebuf[id.y][id.x][0]) & 0xff) / f32(0xff),
        f32(atomicLoad(&framebuf[id.y][id.x][1]) & 0xff) / f32(0xff),
        f32(atomicLoad(&framebuf[id.y][id.x][2]) & 0xff) / f32(0xff),
        1.0
    ));
}