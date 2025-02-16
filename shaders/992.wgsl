fn random2(p: vec2<f32> ) -> vec2<f32> {
    return fract(sin(vec2(dot(p,vec2(127.1,311.7)),dot(p,vec2(269.5,183.3))))*43758.5453);
}

fn rand22(n: vec2f) -> f32 { return fract(sin(dot(n, vec2f(12.9898, 4.1414))) * 43758.5453); }

fn noise2(n: vec2<f32>) -> f32 {
    let d = vec2<f32>(0., 1.);
    let b = floor(n);
    let f = smoothstep(vec2<f32>(0.), vec2<f32>(1.), fract(n));
    return mix(
        mix(rand22(b), rand22(b + d.yx), f.x), 
        mix(rand22(b + d.xy), rand22(b + d.yy), f.x), 
        f.y);
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

fn catmullRom2D(p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>, t: f32) -> vec2<f32> {
    let t2 = t * t;
    let t3 = t2 * t;
    let v0 = (p2 - p0) * 0.5;
    let v1 = (p3 - p1) * 0.5;
    let a = 2.0 * t3 - 3.0 * t2 + 1.0;
    let b = t3 - 2.0 * t2 + t;
    let c = -2.0 * t3 + 3.0 * t2;
    let d = t3 - t2;
    return a * p1 + b * v0 + c * p2 + d * v1;
}

// Function to check if a pixel position is on the line between start and end
fn isPixelOnLine(p0: vec2<f32>, p1: vec2<f32>, pixelPos: vec2<f32>) -> bool {
    let steps = 32;
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

fn isPixelOnCatmullRomSpline(p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>, pixelPos: vec2<f32>) -> bool {
    let steps = i32(clamp(length(abs(p2 - p1)), 0,128));
    let epsilon = 0.001; // Adjust epsilon based on your precision needs
    for (var i: i32 = 0; i < steps; i = i + 1) {
        let t = f32(i) / f32(steps - 1);
        let p = catmullRom2D(p0, p1, p2, p3, t);
        if (all(round(p) == round(pixelPos))) {
            return true;
        }
    }
    return false;
}


const GRASS_COLOUR = vec3(0,0.5,0);
const SEA_COLOUR = vec3(0,0,1);
const ROAD_COLOUR = vec3(0.1, 0.1, 0.1);
const MOUNTAIN_COLOUR = vec3(0.3, 0.3, 0.3);
const BEACH_COLOUR = vec3(0.8, 0.7, 0.5);
const BEACH_HEIGHT = 0.3;
const GRASS_HEIGHT = 0.4;
const MOUNTAIN_HEIGHT = 0.8;


// TODO: exclude edges with too many connections (junctions)
// TODO: exclude edges with no connections (just one strip)
fn isValidCell(gridIndex: vec2<f32>) -> bool {
    var isChecker = abs(gridIndex.x) % 2 == abs(gridIndex.y) % 2;
    // isChecker = true;
    var noiseValue = perlinNoise2(gridIndex * 0.1) * 0.5 + 0.5;
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

// check closest dot product to 1
fn closestValidNeighbor(gridIndex: vec2<f32>, normalizedDirection: vec2<f32>) -> vec2<f32> {
    var minDot = 999.0;
    var closestValidNeighbor = vec2(1.0);
    var angle = minDot;
    // closest to zero is best (dot - 1)
    var check = vec2(-1.0,-1.0);
    angle = abs(dot(normalize(check), normalizedDirection) - 1);
    if(angle < minDot && isValidCell(gridIndex + check)){
        closestValidNeighbor = check;
        minDot = angle;
    }
    check = vec2(-1.0,1.0);
    angle = abs(dot(normalize(check), normalizedDirection) - 1);
    if(angle < minDot && isValidCell(gridIndex + check)){
        closestValidNeighbor = check;
        minDot = angle;
    }
    check = vec2(1.0,1.0);
    angle = abs(dot(normalize(check), normalizedDirection) - 1);
    if(angle < minDot && isValidCell(gridIndex + check)){
        closestValidNeighbor = check;
        minDot = angle;
    }
    check = vec2(1.0,-1.0);
    angle = abs(dot(normalize(check), normalizedDirection) - 1);
    if(angle < minDot && isValidCell(gridIndex + check)){
        closestValidNeighbor = check;
        minDot = angle;
    }

    // Return the closest valid neighbor tile position based on the quadrant
    return closestValidNeighbor;
}

fn isValidEdge(p0: vec2<f32>, p1: vec2<f32>, scaleFactor: f32) -> bool{
    var dir = normalize(p1 - p0);
    let isDirectionUpwards = dir.y > 0;
    // let isFarEnough = distance(p0,p1) > 0.9 / scaleFactor;
    return isDirectionUpwards;
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
        if(x == 0 && y == 0){
            continue;
        }
        let neighbor = vec2(f32(x),f32(y));
        var point = getCellPoint(i_st + neighbor);
        if(point.x < 0.0){
            continue;
        }
        var offsetPoint = neighbor + point;
        var p1 = (i_st + pointInCell) / scaleFactor;
        var p2 = (i_st + offsetPoint) / scaleFactor;

        // only draw down, otherwise i get two variants
        if(!isValidEdge(p1,p2,scaleFactor)){
            continue;
        }

        let dir = normalize(p2 - p1);

        // Adjancency index in the range of vec2(-1) to vec2(1)
        let endOffset = closestValidNeighbor(i_st + neighbor, dir);
        let endPoint = endOffset + getCellPoint(i_st + endOffset);
        var p3 = (i_st + endPoint) / scaleFactor;

        let startOffset = closestValidNeighbor(i_st + neighbor, -dir);
        let startPoint = startOffset + getCellPoint(i_st + startOffset);
        var p0 = (i_st + startPoint) / scaleFactor;

        if(isPixelOnCatmullRomSpline(mix(p1,p0,custom.spline_amount),p1,p2,mix(p2,p3,custom.spline_amount), pixel)){
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
    var noiseValue = perlinNoise2(i_st * 0.1) * 0.5 + 0.5;
    var isBeach = noiseValue > BEACH_HEIGHT;
    var isGrass = noiseValue > GRASS_HEIGHT;
    var isMountain = noiseValue > MOUNTAIN_HEIGHT;
    var color = select(SEA_COLOUR, BEACH_COLOUR, isBeach);
    color = select(color, GRASS_COLOUR, isGrass);
    color = select(color, MOUNTAIN_COLOUR, isMountain);

    // // Draw cell center
    if(distance(f_st, pointInCell) < 0.03){
        // color = vec3(0.5);
    }

    // Draw grid
    //color *= 1.0 - (step(.98, f_st.x) * 0.25 + step(.98, f_st.y) * 0.25);
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

    // Output to screen (linear colour space)
    
}
