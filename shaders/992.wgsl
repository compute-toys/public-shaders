fn random2(p: vec2<f32> ) -> vec2<f32> {
    return fract(sin(vec2(dot(p,vec2(127.1,311.7)),dot(p,vec2(269.5,183.3))))*43758.5453);
}

fn permute4(x: vec4f) -> vec4f { return ((x * 34. + 1.) * x) % vec4f(289.); }
fn fade2(t: vec2f) -> vec2f { return t * t * t * (t * (t * 6. - 15.) + 10.); }

fn perlinNoise2(P: vec2f) -> f32 {
    var Pi: vec4f = floor(P.xyxy) + vec4f(0., 0., 1., 1.);
    let Pf = fract(P.xyxy) - vec4f(0., 0., 1., 1.);
    Pi = Pi % vec4f(289.); // To avoid truncation effects in permutation
    let ix = Pi.xzxz;
    let iy = Pi.yyww;
    let fx = Pf.xzxz;
    let fy = Pf.yyww;
    let i = permute4(permute4(ix) + iy);
    var gx: vec4f = 2. * fract(i * 0.0243902439) - 1.; // 1/41 = 0.024...
    let gy = abs(gx) - 0.5;
    let tx = floor(gx + 0.5);
    gx = gx - tx;
    var g00: vec2f = vec2f(gx.x, gy.x);
    var g10: vec2f = vec2f(gx.y, gy.y);
    var g01: vec2f = vec2f(gx.z, gy.z);
    var g11: vec2f = vec2f(gx.w, gy.w);
    let norm = 1.79284291400159 - 0.85373472095314 *
        vec4f(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11));
    g00 = g00 * norm.x;
    g01 = g01 * norm.y;
    g10 = g10 * norm.z;
    g11 = g11 * norm.w;
    let n00 = dot(g00, vec2f(fx.x, fy.x));
    let n10 = dot(g10, vec2f(fx.y, fy.y));
    let n01 = dot(g01, vec2f(fx.z, fy.z));
    let n11 = dot(g11, vec2f(fx.w, fy.w));
    let fade_xy = fade2(Pf.xy);
    let n_x = mix(vec2f(n00, n01), vec2f(n10, n11), vec2f(fade_xy.x));
    let n_xy = mix(n_x.x, n_x.y, fade_xy.y);
    return 2.3 * n_xy;
}

// Function to check if a pixel position is on the line between start and end
fn isPixelOnLine(p0: vec2<f32>, p1: vec2<f32>, pixelPos: vec2<f32>) -> bool {
    let steps = 64;
    var p = p0;
    let d = p1 - p0;
    let inc = d / f32(steps);
    for (var i: i32 = 0; i < steps; i ++) {
        p += inc;

        let t = f32(i) / f32(steps);
        let distance_to_halfway = abs(t - 0.5) * 2.0;

        let noise = perlinNoise2(p * custom.road_perturb_freq * custom.scale) * custom.road_perturb_amp * (1.0 - distance_to_halfway);

        if(all(round(p + noise) == pixelPos)){
            return true;
        }
    }
    return false;
}

const ROAD_COLOUR = vec3(0.1, 0.1, 0.1);

fn getCellPoint(gridIndex: vec2<f32>) -> vec2<f32> {
    var point = random2(gridIndex);
    point = 0.5 + 0.5*sin(6.2831*point);
    return mix(vec2(0.5), point, custom.distort);
}

fn getCellWeight(p: vec2<f32>) -> f32 {
    let node_noise = perlinNoise2(p * custom.node_perlin_scale)  * 0.5 + 0.5;
    let terrain_noise = perlinNoise2(p * custom.terrain_perlin_scale) * 0.5 + 0.5;
    let terrain_range = custom.hill_level - custom.sea_level;
    let terrain_valley_noise =  max(1.0 - pow(((terrain_noise - custom.sea_level)/(0.5 * terrain_range)) - 1.0, 2.0),0.0);
    let t = terrain_valley_noise;
    return node_noise * t;
}

fn drawLinesForCell(i_st: vec2<f32>, pixel: vec2<f32>, scaleFactor: f32) -> bool {
    var pointInCell = getCellPoint(i_st);
    if(pointInCell.x < 0.0){
        return false;
    }

    let size = getCellWeight(i_st + pointInCell);

    if(size <= 0.0){
        return false;
    }

    var p1 = (i_st + pointInCell) / scaleFactor;

    var largestNodePoint = vec2(-2.0);
    var largestNodeSize = 0.0;

    for(var x = -1; x <= 1; x ++){
        for(var y = -1; y <= 1; y ++){
            if (x == 0 && y == 0){
                continue;
            }
            let neighbor = vec2(f32(x),f32(y));
            var point = getCellPoint(i_st + neighbor);
            var offsetPoint = neighbor + point;
            var p1 = (i_st + pointInCell) / scaleFactor;
            var p2 = (i_st + offsetPoint) / scaleFactor;


            var size = getCellWeight(i_st + offsetPoint);
           
            if(size > largestNodeSize){
                largestNodePoint = p2;
                largestNodeSize = size;
            }
        }
    }

    if(isPixelOnLine(p1,largestNodePoint, pixel) && largestNodeSize > 0.0){
        return true;
    }

    return false;
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

    // // Draw cell center
    var size = getCellWeight(i_st + pointInCell);

    let terrain_noise = perlinNoise2((i_st + f_st) * custom.terrain_perlin_scale) * 0.5 + 0.5;
    color = vec3(0.0,0.0,1.0);
    if (terrain_noise > custom.sea_level - 0.05){
        color = vec3(1.0,1.0,0.3);
    }
    if (terrain_noise > custom.sea_level){
        color = vec3(0.0,1.0,0.0);
    }
    if(terrain_noise > custom.hill_level){
        color = vec3(0.5);
    }

    if(distance(f_st, pointInCell) < size *0.4){
        color = vec3(1.0, 0.0, 0.0);
    }

    // Draw grid
    color *= 1.0 - (step(.98, f_st.x) * 0.25 + step(.98, f_st.y) * 0.25);
    
    textureStore(screen, id.xy, vec4f(color, 1.));


    for( var x = -1; x <= 1; x+= 1){
        for( var y = -1; y <= 1; y+= 1){
            if(drawLinesForCell(i_st + vec2(f32(x),f32(y)), vec2<f32>(pixel), scaleFactor)){
                for( var x = -1; x <= 1; x+= 1){
                    for( var y = -1; y <= 1; y+= 1){
                        textureStore(screen, vec2(i32(id.x) + x,i32(id.y) + y), vec4f(ROAD_COLOUR, 1.));
                    }
                }
                break;
            }
        }
    }
}
