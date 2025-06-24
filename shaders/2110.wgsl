
const TAU: f32 = 6.28318530718;


fn hash1D(p: vec2f) -> f32 {
    return fract(sin(dot(p, vec2f(12.9898, 78.233))) * 43758.5453);
}


fn hash2D(p: vec2f) -> vec2f {
    return fract(sin(vec2f(dot(p, vec2f(127.1, 311.7)),
                           dot(p, vec2f(269.5, 183.3)))) * 43758.5453);
}

fn palette(t: f32) -> vec3f {
    let base = vec3f(t) + vec3f(0.0, 0.33, 0.67);
    return vec3f(0.8) + 0.2 * cos(TAU * base);
}

fn sdfRoundedBox(p: vec2f, center: vec2f, size: vec2f, radius: f32) -> f32 {
    let q = abs(p - center) - size + vec2f(radius);
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2f(0.0))) - radius;
}

fn sdfBox(p: vec2f, center: vec2f, size: vec2f) -> f32 {
    let d = abs(p - center) - size;
    return length(max(d, vec2f(0.0))) + min(max(d.x, d.y), 0.0);
}


@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);

    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let frag_coord_pix = vec2f(f32(id.x), f32(screen_size.y - id.y));
    
    let uv = frag_coord_pix / f32(screen_size.y);
    let resolution = vec2f(screen_size) / f32(screen_size.y);
    
    var canvasColor = vec3f(0.95, 0.92, 0.88) - hash1D(uv * 2.0) * 0.05;

    let boxCenter = vec2f(0.5 * resolution.x, 0.5);
    let boxSize = vec2f(0.35);
    let boxCornerRadius = 0.01;

    let distToBox = sdfRoundedBox(uv, boxCenter, boxSize, boxCornerRadius);

    if (distToBox < 0.005) {
        
        var boxUV = (uv - boxCenter + boxSize) / (boxSize * 2.0);
        boxUV = boxUV * 3.0;
        let gridCellID = floor(boxUV);
        let fragCoordInCell = fract(boxUV);

        var minDistSqToPoint = 8.0;
        var vecToClosestPoint: vec2f;
        var closestPointCellOffset: vec2f;

        for (var j = -1; j <= 1; j = j + 1) {
            for (var i = -1; i <= 1; i = i + 1) {
                let cellOffset = vec2f(f32(i), f32(j));
                let voronoiPointID = gridCellID + cellOffset;
                let voronoiPointPosition = vec2f(0.5) + 0.5 * sin(0.4 * time.elapsed + TAU * hash2D(voronoiPointID));
                let vecToVoronoiPoint = cellOffset + voronoiPointPosition - fragCoordInCell;
                let distSqToPoint = dot(vecToVoronoiPoint, vecToVoronoiPoint);

                if (distSqToPoint < minDistSqToPoint) {
                    minDistSqToPoint = distSqToPoint;
                    vecToClosestPoint = vecToVoronoiPoint;
                    closestPointCellOffset = cellOffset;
                }
            }
        }
        
        var edgePotential = 0.0;
        for (var j = -2; j <= 2; j = j + 1) {
            for (var i = -2; i <= 2; i = i + 1) {
                let cellOffset = closestPointCellOffset + vec2f(f32(i), f32(j));
                let voronoiPointID = gridCellID + cellOffset;
                
                let voronoiPointPosition = vec2f(0.5) + 0.5 * sin(0.4 * time.elapsed + TAU * hash2D(voronoiPointID));
                let vecToVoronoiPoint = cellOffset + voronoiPointPosition - fragCoordInCell;
                
                // Exclude the cell's own point
                if (length(vecToClosestPoint - vecToVoronoiPoint) > 0.001) {
                    let midPointVec = 0.5 * (vecToClosestPoint + vecToVoronoiPoint);
                    let dirVec = normalize(vecToVoronoiPoint - vecToClosestPoint);
                    edgePotential = edgePotential + exp(-35.0 * dot(midPointVec, dirVec));
                }
            }
        }
        let distToGrout = -log(edgePotential) / 35.0;
        
        let cellHash = hash2D(gridCellID + closestPointCellOffset);
        let paletteColor = palette(cellHash.x / 0.25);
        var boxCellColor = select(vec3f(1.0), paletteColor, cellHash.x < 0.25);
        
        boxCellColor = boxCellColor * smoothstep(0.0, 0.022, distToGrout);
        
        let groutEffect = exp(-abs(distToGrout - 0.01) / 0.015);
        let groutTexture = hash1D((boxUV - gridCellID - closestPointCellOffset) * 10.0);
        boxCellColor = boxCellColor * (1.0 - 0.55 * groutTexture * groutEffect);

        canvasColor = mix(canvasColor, boxCellColor, smoothstep(0.005, 0.0, distToBox));
    }

    let paletteBarCenter = boxCenter + vec2f(0.4, 0.0);
    let paletteBarSize = vec2f(0.03, 0.35);
    let distToPaletteBar = sdfRoundedBox(uv, paletteBarCenter, paletteBarSize, 0.0);
    
    let paletteBarGradientPos = 1.0 - (uv.y - 0.15) / 0.7;
    let paletteBarColor = palette(paletteBarGradientPos);
    
    canvasColor = mix(canvasColor, paletteBarColor, 1.0 - smoothstep(0.0, 0.005, distToPaletteBar));

    let filmGrainEffect = (hash1D(vec2f(id.xy) + 25.0) - 0.5) * 0.08;
    canvasColor = canvasColor * (1.0 + filmGrainEffect);
    
    canvasColor = pow(canvasColor, vec3f(2.2 / 1.0));

    textureStore(screen, id.xy, vec4f(clamp(canvasColor, vec3f(0.0), vec3f(1.0)), 1.0));
}