/// Simulation ///

// Grid resolution
const N = 128u;
const quad_size = 1.0 / f32(N);

// Storage for particle positions and velocities
struct ParticleData {
    positions: array<array<vec3<f32>, N>, N>,
    velocities: array<array<vec3<f32>, N>, N>,
}
#storage particles ParticleData

// Helper function for periodic boundary conditions
fn is_valid_index(i: u32, j: u32) -> bool {
    return i < N && j < N;
}

// Initialize the mass points
// #dispatch_once init
@compute @workgroup_size(8, 8)
fn init(@builtin(global_invocation_id) id: vec3<u32>) {
    if (time.frame % 1000 != 0) { return; }

    let i = id.x;
    let j = id.y;
    if (!is_valid_index(i, j)) { return; }
    
    particles.positions[i][j] = vec3<f32>(f32(i) * quad_size - 0.5, 0.6, f32(j) * quad_size - 0.5);
    particles.velocities[i][j] = vec3<f32>(0.0, 0.0, 0.0);
}

// Physics simulation step
#dispatch_count simulate 4
@compute @workgroup_size(8, 8)
fn simulate(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    if (!is_valid_index(i, j)) { return; }
    
    var pos = particles.positions[i][j].xyz;
    var vel = particles.velocities[i][j].xyz;

    let dt = custom.dt / f32(N);
    
    // Apply gravity
    vel.y -= 9.8 * dt;
    
    // Calculate spring forces
    var force = vec3<f32>(0.0);
    
    // Check neighboring particles within distance 2
    for (var oi = -2; oi <= 2; oi++) {
        for (var oj = -2; oj <= 2; oj++) {
            if (oi == 0 && oj == 0) { continue; }
            if (abs(oi) + abs(oj) > 2) { continue; }
            
            let ni = i32(i) + oi;
            let nj = i32(j) + oj;
            
            if (ni >= 0 && ni < i32(N) && nj >= 0 && nj < i32(N)) {
                let neighbor_pos = particles.positions[u32(ni)][u32(nj)].xyz;
                let neighbor_vel = particles.velocities[u32(ni)][u32(nj)].xyz;
                
                let x_ij = pos - neighbor_pos;
                let v_ij = vel - neighbor_vel;
                let dist = length(x_ij);
                if (dist > 0.0) {
                    let d = x_ij / dist;
                    let original_dist = quad_size * length(vec2<f32>(f32(oi), f32(oj)));
                    
                    // Spring force
                    force += -1e4 * custom.stiffness * d * (dist / original_dist - 1.0);
                    // Dashpot damping
                    force += -dot(v_ij, d) * d * 1e3 * quad_size;
                }
            }
        }
    }
    
    // Update velocity with forces
    vel += force * dt;
    
    // Apply drag damping
    vel *= exp(-dt);
    
    // Ball collision
    let offset_to_center = pos - vec3(0);
    let dist_to_ball = length(offset_to_center);
    if (dist_to_ball <= custom.ball_radius) {
        let normal = offset_to_center / dist_to_ball;
        let vel_normal = dot(vel, normal);
        if (vel_normal < 0.0) {
            vel -= vel_normal * normal;
        }
    }
    
    // Update position
    pos += vel * dt;
    
    // Store updated values
    particles.positions[i][j] = pos;
    particles.velocities[i][j] = vel;
}

/// Rasterisation ///

#storage framebuf array<array<array<atomic<u32>, 3>, SCREEN_WIDTH>, SCREEN_HEIGHT>

struct Triangle {
    vertices: mat3x3<f32>,
    normals: mat3x3<f32>,
    uvs: mat3x2<f32>
}

fn worldToViewSpace(p: vec3<f32>) -> vec3<f32> {
    return p + vec3<f32>(0.0, 0.0, -2.0);
}

fn viewToClipSpace(p: vec3<f32>) -> vec4<f32> {
    let fov = radians(30.0);  
    let aspect = f32(SCREEN_WIDTH) / f32(SCREEN_HEIGHT);
    let f = 1.0 / tan(fov * 0.5);
    
    // WebGPU perspective projection
    // Map [-zNear, -zFar] to [0, 1]
    return vec4<f32>(
        (f / aspect) * p.x,
        f * p.y,
        (-p.z - custom.zNear) / (custom.zFar - custom.zNear),
        -p.z  // W for perspective divide
    );
}

fn ndcToScreenSpace(p: vec2<f32>) -> vec2<i32> {
    return vec2<i32>(
        i32((0.5 + 0.5 * p.x) * f32(SCREEN_WIDTH)),
        i32((0.5 - 0.5 * p.y) * f32(SCREEN_HEIGHT))
    );
}

fn shadePixel(tri: Triangle, bary: vec3<f32>, backfacing: bool) -> vec3<f32> {
    var pos = tri.vertices * bary;
    var nor = normalize(tri.normals * bary);
    var uv = tri.uvs * bary;
    var diffuse = max(0.1, dot(nor, normalize(vec3<f32>(custom.lightX, custom.lightY, custom.lightZ))));
    var tex = textureSampleLevel(channel1, bilinear, uv, 0).rgb;
    if (backfacing) {
        tex = vec3(tex.r * 5 + 1);
    } else {
        tex = vec3(0, tex.r * 5, 0);
    }
    return tex * diffuse;
}

@compute @workgroup_size(16, 16)
fn clearFramebuffer(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= SCREEN_WIDTH || id.y >= SCREEN_HEIGHT) { return; }
    let depthBits = 0xffffff00u;
    atomicStore(&framebuf[id.y][id.x][0], depthBits | u32(custom.backgroundR * f32(0xff)));
    atomicStore(&framebuf[id.y][id.x][1], depthBits | u32(custom.backgroundG * f32(0xff)));
    atomicStore(&framebuf[id.y][id.x][2], depthBits | u32(custom.backgroundB * f32(0xff)));
}

fn drawTriangle(tri: Triangle) {
    // Transform to view space
    var viewSpaceVerts: mat3x3<f32>;
    viewSpaceVerts[0] = worldToViewSpace(tri.vertices[0]);
    viewSpaceVerts[1] = worldToViewSpace(tri.vertices[1]);
    viewSpaceVerts[2] = worldToViewSpace(tri.vertices[2]);

    // Cull if ANY vertex is behind near plane (remember view space Z is negative)
    if (viewSpaceVerts[0].z > -custom.zNear || 
        viewSpaceVerts[1].z > -custom.zNear || 
        viewSpaceVerts[2].z > -custom.zNear) {
        return;
    }

    // Transform to clip space
    var clipSpaceVerts: mat3x4<f32>;
    clipSpaceVerts[0] = viewToClipSpace(viewSpaceVerts[0]);
    clipSpaceVerts[1] = viewToClipSpace(viewSpaceVerts[1]);
    clipSpaceVerts[2] = viewToClipSpace(viewSpaceVerts[2]);

    // Perform perspective divide
    let v0 = clipSpaceVerts[0].xyz / clipSpaceVerts[0].w;
    let v1 = clipSpaceVerts[1].xyz / clipSpaceVerts[1].w;
    let v2 = clipSpaceVerts[2].xyz / clipSpaceVerts[2].w;

    // Cull backfacing triangles
    var perp = cross(v1 - v0, v2 - v0);
    let backfacing = perp.z < 0.0;
    // if (backfacing) { return; }

    // Map to screen space and hold onto the index of the vertex
    var screenA = vec3<i32>(ndcToScreenSpace(v0.xy), 0);
    var screenB = vec3<i32>(ndcToScreenSpace(v1.xy), 1);
    var screenC = vec3<i32>(ndcToScreenSpace(v2.xy), 2);

    var depths = vec3<f32>(v0.z, v1.z, v2.z);

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
                    var depth = dot(depths, worldBary);
                    var depthBits = u32(depth * f32(0xffffff)) << 8;
                    var shade = shadePixel(tri, worldBary, backfacing);
                    // if (backfacing) { shade = shade.bgr; }
                    // shade *= 1 - depth;
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
                    var depth = dot(depths, worldBary);
                    var depthBits = u32(depth * f32(0xffffff)) << 8;
                    var shade = shadePixel(tri, worldBary, backfacing);
                    // if (backfacing) { shade = shade.bgr; }
                    // shade *= 1 - depth;
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

/// Cloth rendering ///

fn compute_vertex_normal(i: u32, j: u32) -> vec3<f32> {
    // Get positions of neighboring vertices (when available)
    let right = select(particles.positions[i][j].xyz, particles.positions[i+1][j].xyz, is_valid_index(i+1, j));
    let left = select(particles.positions[i][j].xyz, particles.positions[i-1][j].xyz, is_valid_index(i-1, j));
    let up = select(particles.positions[i][j].xyz, particles.positions[i][j-1].xyz, is_valid_index(i, j-1));
    let down = select(particles.positions[i][j].xyz, particles.positions[i][j+1].xyz, is_valid_index(i, j+1));
    
    // Compute derivatives using central differences
    let dx = right - left;
    let dy = down - up;
    
    // Normal is cross product of derivatives
    return normalize(cross(dx, dy));
}

@compute @workgroup_size(8, 8)
fn render(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;
    if (i == 0 || i >= N-1 || j == 0 || j >= N-1) { return; }

    var co = cos(time.elapsed / 5);
    var si = sin(time.elapsed / 5);
    var rotation = mat3x3<f32>(co, 0.0, si, 0.0, 1.0, 0.0, -si, 0.0, co);

    // Get vertex positions
    let p00 = rotation * particles.positions[i][j].xyz;
    let p10 = rotation * particles.positions[i+1][j].xyz;
    let p01 = rotation * particles.positions[i][j+1].xyz;
    let p11 = rotation * particles.positions[i+1][j+1].xyz;

    // Compute smooth normals for each vertex
    let n00 = compute_vertex_normal(i, j);
    let n10 = compute_vertex_normal(i+1, j);
    let n01 = compute_vertex_normal(i, j+1);
    let n11 = compute_vertex_normal(i+1, j+1);

    // Calculate UV coordinates
    let uv_scale = 1.0 / f32(N);
    let uv00 = vec2f(f32(i), f32(j)) * uv_scale;
    let uv10 = vec2f(f32(i+1), f32(j)) * uv_scale;
    let uv01 = vec2f(f32(i), f32(j+1)) * uv_scale;
    let uv11 = vec2f(f32(i+1), f32(j+1)) * uv_scale;

    // Draw triangles with smooth normals
    var tri1: Triangle;
    tri1.vertices = mat3x3(p00, p01, p10);
    tri1.normals = mat3x3(n00, n01, n10);
    tri1.uvs = mat3x2(uv00, uv01, uv10);
    drawTriangle(tri1);

    var tri2: Triangle;
    tri2.vertices = mat3x3(p10, p01, p11);
    tri2.normals = mat3x3(n10, n01, n11);
    tri2.uvs = mat3x2(uv10, uv01, uv11);
    drawTriangle(tri2);
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
