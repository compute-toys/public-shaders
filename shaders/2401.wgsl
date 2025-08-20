fn random2(p: vec2<f32> ) -> vec2<f32> {
    return fract(sin(vec2(dot(p,vec2(127.1,311.7)),dot(p,vec2(269.5,183.3))))*43758.5453);
}

fn perlinNoise2(p: vec2f) -> f32 {
    let uv = abs((p * 0.01) + vec2(0.5)) % vec2(1.0);
    return textureSampleLevel(channel0, bilinear, uv, 0.0).r * 2.0;
}

fn distanceToLineSegment(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

const ROAD_COLOUR = vec3(0.1, 0.1, 0.1);

fn getCellPoint(gridIndex: vec2<f32>) -> vec2<f32> {
    var point = random2(gridIndex);
    point = 0.5 + 0.5*sin(6.2831*point);
    return mix(vec2(0.5), point, custom.distort);
}

// TODO: check if this is the largest of its neighbour when determining towns
fn getCellWeight(p: vec2<f32>) -> f32 {
    let node_noise = perlinNoise2(p * custom.node_perlin_scale) ;
    let terrain_noise = perlinNoise2(p * custom.terrain_perlin_scale);
    let terrain_range = custom.hill_level - custom.sea_level;
    let terrain_valley_noise =  max(1.0 - pow(((terrain_noise - custom.sea_level)/(0.5 * terrain_range)) - 1.0, 2.0),0.0);
    let t = terrain_valley_noise;
    let weight = node_noise * t;
    return weight;
}

fn drawLinesForCell(i_st: vec2<f32>, pixel: vec2<f32>, scaleFactor: f32) -> f32 {
    var pointInCell = getCellPoint(i_st);

    var p1 = (i_st + pointInCell) / scaleFactor;

    let weight = getCellWeight(i_st + pointInCell);

    if(weight < custom.node_weight_threshold){
        return 1000.0;
    }

    var largestNodePoint = vec2(0.0);
    var largestNodeWeight = -1.0;

    for(var x = -1; x <= 1; x ++){
        for(var y = -1; y <= 1; y ++){
            if (x == 0 && y == 0){
                continue;
            }
            let neighbor = vec2(f32(x),f32(y));
            let point = getCellPoint(i_st + neighbor);
            let offsetPoint = neighbor + point;
            let p2 = (i_st + offsetPoint) / scaleFactor;

            let weight = getCellWeight(i_st + offsetPoint);
           
            if(weight > largestNodeWeight && weight > custom.node_weight_threshold){
                largestNodePoint = p2;
                largestNodeWeight = weight;
            }
        }
    }
    let perturbX = (perlinNoise2(pixel * custom.road_perturb_freq * custom.scale) * 2.0 - 1.0) * custom.road_perturb_amp / custom.scale;
    let perturbY = (perlinNoise2(pixel.yx * custom.road_perturb_freq * custom.scale) * 2.0 - 1.0)  * custom.road_perturb_amp / custom.scale;


    let perturbedPixel = pixel + perturbX + perturbY;

    if(largestNodeWeight < custom.node_weight_threshold){
        return 1000.0;
    }

    return distanceToLineSegment(perturbedPixel, p1, largestNodePoint);
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    var screen_size = vec2<f32>(textureDimensions(screen));
    let scaleFactor = (screen_size.x/screen_size.y) * custom.scale * 0.02;
    let pixel = vec2(i32(textureDimensions(screen).x / 2 - id.x), i32(textureDimensions(screen).y / 2 - id.y));

    var st = vec2<f32>(pixel.xy);
    st *= scaleFactor;

    // Tile the space
    let i_st = floor(st);
    let f_st = fract(st);
    
    var pointInCell = getCellPoint(i_st);
    var color = vec3(0.0);

    // Draw cell center
    var weight = getCellWeight(i_st + pointInCell);

    var distanceToClosestRoad = 1000.0;
    for(var x = -1; x <= 1; x += 1){
        for(var y = -1; y <= 1; y += 1){
            let distanceToCellRoad = drawLinesForCell(i_st + vec2(f32(x), f32(y)), vec2<f32>(pixel), scaleFactor);
            distanceToClosestRoad = min(distanceToClosestRoad, distanceToCellRoad);
        }
    }

    let terrain_noise = perlinNoise2((i_st + f_st) * custom.terrain_perlin_scale);
    color = vec3(0.0,0.0,1.0);
    if (terrain_noise > custom.sea_level - 0.2){
        color = vec3(1.0,1.0,0.3);
    }
    if (terrain_noise > custom.sea_level - 0.1){
        color = vec3(0.0,1.0,0.0);
    }
    if(terrain_noise > custom.hill_level + 0.1){
        color = vec3(0.5);
    }
    
    
    //color *= vec3(distanceToClosestRoad * 0.02 * custom.scale);

    if(distanceToClosestRoad * custom.scale < 2.0){
        color = ROAD_COLOUR;
    }
    
    if(weight > custom.town_node_size && distance(f_st, pointInCell) < 0.3){
        color = vec3(1.0, 0.0, 0.0);
    }
    
    textureStore(screen, id.xy, vec4f(color, 1.));
}