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
    for (var i: i32 = 0; i < steps; i = i + 1) {
        p += inc;
        if(all(round(p) == pixelPos)){
            return true;
        }
    }
    return false;
}

const ROAD_COLOUR = vec3(0.1, 0.1, 0.1);
const BEACH_HEIGHT = 0.3;
const GRASS_HEIGHT = 0.4;
const MOUNTAIN_HEIGHT = 0.8;


// TODO: exclude edges with too many connections (junctions)
// TODO: exclude edges with no connections (just one strip)
fn isValidCell(gridIndex: vec2<f32>) -> bool {
    var isChecker = abs(gridIndex.x) % 2 == abs(gridIndex.y) % 2;
    var noiseValue = perlinNoise2(gridIndex * custom.perlin_scale) * 0.5 + 0.5;
    var isGrass = noiseValue > GRASS_HEIGHT + 0.1;
    var isMountain = noiseValue > MOUNTAIN_HEIGHT - 0.1;
    return isChecker && isGrass && !isMountain; 
}

fn getCellPoint(gridIndex: vec2<f32>) -> vec2<f32> {
    if(!isValidCell(gridIndex)){
        return vec2(-1.0);
    }
    var point = random2(gridIndex);
    point = 0.5 + 0.5*sin(custom.random_co * 10.0 + 6.2831*point);
    return mix(vec2(0.5), point, custom.distort);
}

fn drawLinesForCell(i_st: vec2<f32>, pixel: vec2<f32>, scaleFactor: f32) -> bool {
    var pointInCell = getCellPoint(i_st);
    if(pointInCell.x < 0.0){
        return false;
    }
    // Draw from this cells point to valid neighbours
    for(var i = 0; i < 9; i++){
        let x = (i % 3) - 1;
        let y = (i / 3) - 1;
        
        let neighbor = vec2(f32(x),f32(y));
        var point = getCellPoint(i_st + neighbor);
        if(point.x < 0.0){
            continue;
        }
        var offsetPoint = neighbor + point;
        var p1 = (i_st + pointInCell) / scaleFactor;
        var p2 = (i_st + offsetPoint) / scaleFactor;

        if(isPixelOnLine(p1,p2, pixel)){
            return true;
        }
    }
    return false;
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    var screen_size = vec2<f32>(textureDimensions(screen));
    let scaleFactor = (screen_size.x/screen_size.y) * custom.scale * 0.02;
    let offset =  vec2(i32((custom.x - 0.5) * 2000.0 - time.elapsed * 10.0) , i32((custom.y - 0.5) * 2000.0));
    let pixel = vec2(i32(id.x), i32(textureDimensions(screen).y - id.y)) - offset;

    var st = vec2<f32>(pixel.xy);
    st *= scaleFactor;

    // Tile the space
    let i_st = floor(st);
    let f_st = fract(st);
    
    var pointInCell = getCellPoint(i_st);
    var noiseValue = perlinNoise2(st * custom.perlin_scale) * 0.5 + 0.5;
    var color = vec3(noiseValue);

    // // Draw cell center
    if(distance(f_st, pointInCell) < 0.2){
        color = vec3(1.0, 0.0, 0.0);
    }

    // Draw grid
    color *= 1.0 - (step(.98, f_st.x) * 0.25 + step(.98, f_st.y) * 0.25);
    textureStore(screen, id.xy, vec4f(color, 1.));

    for(var i = 0; i < 9; i++){
        let x = (i % 3) - 1;
        let y = (i / 3) - 1;
        if(drawLinesForCell(i_st + vec2(f32(x),f32(y)), vec2<f32>(pixel), scaleFactor)){
            textureStore(screen, id.xy, vec4f(ROAD_COLOUR, 1.));
            textureStore(screen, vec2<i32>(id.xy) + vec2(-1,0), vec4f(ROAD_COLOUR, 1.));
            textureStore(screen, vec2<i32>(id.xy) + vec2(1,0), vec4f(ROAD_COLOUR, 1.));
            textureStore(screen, vec2<i32>(id.xy) + vec2(0,-1), vec4f(ROAD_COLOUR, 1.));
            textureStore(screen, vec2<i32>(id.xy) + vec2(0,1), vec4f(ROAD_COLOUR, 1.));
            break;
        }
    }
}
