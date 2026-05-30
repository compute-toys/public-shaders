struct Ray {
    position: vec3f,
    direction: vec3f,
};

fn murmurHash11(source: u32) -> u32{
    var src = source;
    const M = 0x5bd1e995u;
    var h = 1190494759u;
    src *= M; src ^= src>>24u; src *= M;
    h *= M; h ^= src;
    h ^= h>>13u; h *= M; h ^= h>>15u;
    return h;
}

fn murmurHash13(source: vec3u) -> u32 {
    var src = source;
    const M = 0x5bd1e995u;
    var h = 1190494759u;
    src *= M;
    src ^= vec3u(src.x>>24u,src.y>>24u,src.z>>24u);
    src *= M;
    h *= M; h ^= src.x; h *= M; h ^= src.y; h *= M; h ^= src.z;
    h ^= h>>13u; h *= M; h ^= h>>15u;
    return h;
}

// PRNG Utility Functions
//--------------------------------------------------------------------

fn seedPRNG(seed: vec3f) -> u32 {
    return murmurHash13(bitcast<vec3u>(seed));
}

fn shuffle(prngState: ptr<function, u32>) {
    *prngState = murmurHash11(*prngState);
}

// Random in range [0, Max Uint = 0xFFFFFFFFu]
fn randomUint(prngState: ptr<function, u32>) -> u32 {
    shuffle(prngState);
    return *prngState;
}

fn randomFloat(prngState: ptr<function, u32>) -> f32 {
    return float(randomUint(prngState)) / float(0xFFFFFFFFu);
}

fn randomVec3f(state: ptr<function, u32>) -> vec3f {
    return vec3f(randomFloat(state),randomFloat(state),randomFloat(state));
}


fn whiteNoise3d(p: vec3f) -> vec3f {
    var seed = seedPRNG(p);
    //shuffle(&seed);
    return randomVec3f(&seed);
}

fn fade(t: f32) -> f32 {
	return ((6*t - 15)*t + 10)*t*t*t;
}

fn fade3(t: vec3f) -> vec3f {
    return vec3f(fade(t.x), fade(t.y), fade(t.z));
}

fn perlinNoise3d(p: vec3f) -> f32 {
    let lowerBound = floor(p);
    let dp = p - lowerBound;

    var dots: array<array<array<f32, 2>, 2>, 2>;

    for (var i = 0; i < 2; i++) {
        for (var j = 0; j < 2; j++) {
            for (var k = 0; k < 2; k++) {
                let corner = lowerBound + vec3f(f32(i), f32(j), f32(k));
                let gradient = normalize((whiteNoise3d(corner) - 0.5) * 2.0);
                let dist = p - corner;
                dots[i][j][k] = dot(dist, gradient);
            }
        }
    }

    var interpVals = fade3(fract(p));

    for (var i = 0; i < 2; i++) {
        for (var j = 0; j < 2; j++) {
            dots[i][j][0] = mix(dots[i][j][0], dots[i][j][1], interpVals.z);
        }
    }
    for (var i = 0; i < 2; i++) {
        dots[i][0][0] = mix(dots[i][0][0], dots[i][1][0], interpVals.y);
    }
    return mix(dots[0][0][0], dots[1][0][0], interpVals.x);
}

fn octaveNoise3d(p: vec3f) -> f32 {
    var amplitude = 1.0;
    var frequency = 1.0;
    var scale = 0.0;
    var noise = 0.0;
    for (var i = 0u; i < max(u32(custom.octaves), 1u); i++) {
        noise += amplitude * perlinNoise3d(p * frequency);
        amplitude *= custom.persistence;
        frequency *= custom.lacunarity;
        scale += amplitude;
    }
    noise /= scale;
    return noise;
}


fn sdScene(p1: vec3f) -> vec4f {
    let theta = time.elapsed / 30.0;
    let c = cos(theta);
    let s = sin(theta);
    let mat = mat3x3(c, s, 0.0,
                     -s, c, 0.0,
                     0.0, 0.0, 1.0);
    let p = mat * p1;
    
    let pos = vec3f(0.0, 0.0, 3.0);
    let radius = 2.0;
    
    let sphereDist = distance(p, pos) - radius;
    let terrainDist = max(0.0, custom.continentHeight * octaveNoise3d(custom.continentFreq * p + custom.continentSeed) + custom.waterLevel + custom.height * perlinNoise3d(p * custom.freq + custom.seed));
    let dist = sphereDist - terrainDist;

    let height = sphereDist;

    let water = vec3f(0.0, 0.3, 0.6);
    let sand = vec3f(0.9, 0.9, 0.6);
    let green1 = vec3f(0.2, 0.6, 0.3);
    let green2 = vec3f(0.3, 0.7, 0.3) * 0.8;
    let green3 = vec3f(0.3, 0.8, 0.3) * 0.6;
    let gray0 = green3 * 0.8;
    let gray1 = green3 * 0.6;//vec3f(0.65, 0.6, 0.6) * 0.5;
    let gray2 = vec3f(0.6, 0.55, 0.55) * 0.5;
    let gray3 = vec3f(0.55, 0.50, 0.50) * 0.6;
    let gray4 = vec3f(0.55, 0.45, 0.45) * 0.7;
    let gray5 = vec3f(0.50, 0.40, 0.40) * 0.8;
    let gray6 = vec3f(0.50, 0.35, 0.35) * 0.9;
    let gray7 = vec3f(0.45, 0.3, 0.3);
    let gray8 = vec3f(0.4, 0.25, 0.25);
    let gray9 = vec3f(0.35, 0.2, 0.2);
    let scale = custom.colorScale;

    var color = water;
    if (height > 0.0001) {
        if (height < 1.0 * scale) {
            color = sand;
        } else if (height < 2.0 * scale) {
            color = green1;
        } else if (height < 3.0 * scale) {
            color = green2;
        } else if (height < 4.0 * scale) {
            color = green3;
        } else if (height < 5.0 * scale) {
            color = gray0;
        } else if (height < 6.0 * scale) {
            color = gray1;
        } else if (height < 7.0 * scale) {
            color = gray2;
        } else if (height < 8.0 * scale) {
            color = gray3;
        } else if (height < 9.0 * scale) {
            color = gray4;
        } else if (height < 10.0 * scale) {
            color = gray5;
        } else if (height < 11.0 * scale) {
            color = gray6;
        } else if (height < 12.0 * scale) {
            color = gray7;
        } else if (height < 13.0 * scale) {
            color = gray8;
        } else if (height < 14.0 * scale) {
            color = gray9;
        } else {
            color = gray9;
        }
    }
    return vec4f(dist, color);
}

fn normal(p: vec3f) -> vec3f {
    const h = 0.0005; // replace by an appropriate value
    const k = vec2f(1,-1);
    return normalize( k.xyy*sdScene( p + k.xyy*h ).x + 
                      k.yyx*sdScene( p + k.yyx*h ).x + 
                      k.yxy*sdScene( p + k.yxy*h ).x + 
                      k.xxx*sdScene( p + k.xxx*h ).x );
}

#define MAX_DIST 100.0
#define SURF_DIST 0.00001
#define MAX_MARCHES 150u

fn rayMarch(ray: Ray) -> vec4f {
    
    //let ray = ray1;
    var t = 0.0f;
    var safeDist = 0.0f;
    var col = vec3f(0.0);
    var p = ray.position;
    for (var i = 0u; i < MAX_MARCHES; i++) {
        p = p + ray.direction * safeDist;// * 0.8;
        let sd = sdScene(p);
        safeDist = sd.x;
        let noiseBound = 2.0 * (custom.continentHeight + custom.height);
        if (safeDist < noiseBound) {
            safeDist *= 0.5;
        }
        if (safeDist < SURF_DIST || t > MAX_DIST) {
            return vec4f(t, col);
        }
        t += safeDist;
        col = sd.yzw;
    }
    return vec4f(t, col);
}


fn getLight(p: vec3f) -> vec3f {
    let intensity = 1.0;
    var color = vec3f(1.0) * intensity;

    var lightPos = vec3f(5.0 * sin(time.elapsed), 5., 6. + 5.*cos(time.elapsed));
    //lightPos = normalize(vec3f(0.0,0.0,50.0));
    var l = normalize(lightPos - p);
    l = normalize(vec3(0,1,-1.5));
    let n = normal(p);
    var diff = dot(n, l);
    diff = clamp(diff, 0., 1.);
    let d = rayMarch(Ray(p + n*SURF_DIST * 2.,l)).x;
    //if (d < length(lightPos - p)) {
    //    diff *= .1;
    //}
    return color * diff + vec3f(0.3);
}

//@group(0) @binding(18) var buffer: texture_storage_2d<rgba8unorm, read_write>;
#storage buffer array<vec3f, SCREEN_WIDTH * SCREEN_HEIGHT>
#storage thresh array<vec3f, SCREEN_WIDTH * SCREEN_HEIGHT>

#dispatch_count image 1
#workgroup_count image SCREEN_WIDTH SCREEN_HEIGHT 1
@compute @workgroup_size(16, 16)
fn image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);

    let aspect = f32(screen_size.x) / f32(screen_size.y);

    // Normalised pixel coordinates (from 0 to 1)
    let uv = fragCoord / vec2f(screen_size);
    
    let ndc = uv * 2.0 - 1.0;
    let rayDir = normalize(vec3f(ndc.x * aspect, ndc.y, 1.0)); 

    let ray = Ray(vec3f(0.0, 2.0, 2.0), rayDir);

    // Time varying pixel colour
    var hit = rayMarch(ray);
    var col = vec3f(0.0);
    var prng = seedPRNG(vec3f(uv, 0.0));
    if (randomFloat(&prng) < 0.0003) {
        col = vec3f(randomFloat(&prng));
    }
    //col = vec3f(octaveNoise3d(vec3f(fragCoord * 0.05, time.elapsed)) * 0.5 + 0.5);
    //if (noise < 0.5) {
    //    col = vec3f(1.0);
    //}
    if (hit.x < MAX_DIST) {
        col = hit.yzw * getLight(ray.position + ray.direction * hit.x);
    }
    // Test, check perlin noise implementation
    //col = vec3f(perlinNoise3d(vec3f(fragCoord / 30.0, time.elapsed))) * 0.5 + 0.5;
    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    //textureStore(buffer, vec2i(id.xy), vec4f(col, 1.));
    buffer[id.x + id.y * screen_size.x] = col;
}

#dispatch_count threshold 1
#workgroup_count threshold SCREEN_WIDTH SCREEN_HEIGHT 1
@compute @workgroup_size(16, 16)
fn threshold(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    var col = buffer[id.x + id.y * screen_size.x];
    if ((col.r + col.g + col.b) / 3.0 > custom.thresh) {
        thresh[id.x + id.y * screen_size.x] = col;//vec3f(1.0);// col;
    } else {
        thresh[id.x + id.y * screen_size.x] = vec3f(0.0);
    }
}

#dispatch_count gaussian_blur 1
#workgroup_count gaussian_blur SCREEN_WIDTH SCREEN_HEIGHT 1
@compute @workgroup_size(16, 16)
fn gaussian_blur(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);

    let kernel = i32(custom.kernel);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    var col = vec3f(0.0);
    let stdev = f32(kernel) / 6.0;
    for (var i = - kernel/2; i <= kernel/2; i++) {
        for (var j = - kernel/2; j <= kernel/2; j++) {
            let x = u32(clamp(i32(id.x) + i, 0, i32(SCREEN_WIDTH - 1)));
            let y = u32(clamp(i32(id.y) + j, 0, i32(SCREEN_HEIGHT - 1)));
            let fi = f32(i);
            let fj = f32(j);
            let weight = exp(-(fi*fi+fj*fj)/(2.0*stdev*stdev))/(2.0*3.14159*stdev*stdev);
            col += weight * thresh[x + y * screen_size.x];
        }
    }
    thresh[id.x + id.y * screen_size.x] = col;
}

#dispatch_count main_image 1
#workgroup_count main_image SCREEN_WIDTH SCREEN_HEIGHT 1
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    var col = custom.glow * thresh[id.x + id.y * screen_size.x] + buffer[id.x + id.y * screen_size.x];
    //var col = vec3f(1.0);
    textureStore(screen, vec2i(id.xy), vec4f(col, 1.));
}