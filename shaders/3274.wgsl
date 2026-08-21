fn hash3(n: f32) -> vec3<f32> {
    let v = vec3<f32>(
        fract(sin(n) * 43758.5453123),
        fract(sin(n + 1.0) * 43758.5453123),
        fract(sin(n + 2.0) * 43758.5453123)
    );
    return normalize(v * 2.0 - vec3<f32>(1.0));
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);

    // 1. Animated Quaternion & Rotation Matrix Setup
    let seed: f32 = 42.0;
    let axis = hash3(seed);       
    let angle = time.elapsed * 0.4;     
    let q = vec4<f32>(axis * sin(angle * 0.5), cos(angle * 0.5));
    
    let R = mat3x3<f32>(
        vec3<f32>(1.0 - 2.0*(q.y*q.y + q.z*q.z),       2.0*(q.x*q.y - q.z*q.w),       2.0*(q.x*q.z + q.y*q.w)),
        vec3<f32>(      2.0*(q.x*q.y + q.z*q.w), 1.0 - 2.0*(q.x*q.x + q.z*q.z),       2.0*(q.y*q.z - q.x*q.w)),
        vec3<f32>(      2.0*(q.x*q.z - q.y*q.w),       2.0*(q.y*q.z + q.x*q.w), 1.0 - 2.0*(q.x*q.x + q.y*q.y))
    );
    
    // 2. Define vertices and oscillate depth cleanly
    var a = R * vec3<f32>(-1.0, -0.7, 0.0);
    var b = R * vec3<f32>(1.0, -0.7, 0.0);
    var c = R * vec3<f32>(0.0, 0.9, 0.0);
    
    let offset = vec3<f32>(0.0, 0.0, 1.4 + cos(time.elapsed * 0.5) * 1.6); 
    a += offset; b += offset; c += offset;
    
    let uvA = vec2<f32>(0.0, 0.0); // 0,0
    let uvB = vec2<f32>(1.0, 0.0); // 4,0
    let uvC = vec2<f32>(0.5, 1.0); // 2,4
    
    let o = vec3<f32>(0.0, 0.0, -2.0);
    let iRes = vec2<i32>(textureDimensions(channel0));
    
    var colorAccum = vec4<f32>(0.0);
    var totalWeight = 0.0;
    
    // 3. High-Quality 4x4 Grid Super-Sampling (SSAA)
    for (var j = 0; j < 4; j++) {
        for (var i = 0; i < 4; i++) {
            let subPixelOffset = (vec2<f32>(f32(i), f32(j)) - 1.5) * 0.25;
            
            // F matches screen pixel positions (fragCoord.xy)
            let p = (fragCoord.xy + subPixelOffset - vec2<f32>(textureDimensions(screen)) * 0.5) / f32(textureDimensions(screen).y);
            let d = normalize(vec3<f32>(p, 1.0));
            
            let weight = (1.0 - abs(subPixelOffset.x)) * (1.0 - abs(subPixelOffset.y));
            
            // Ray-Triangle Plane Intersection
            var e1: vec3<f32> = b - a;
            var e2: vec3<f32> = c - a;
            var n: vec3<f32> = cross(e1, e2);
            var t: f32 = dot(a - o, n) / dot(d, n);
            var q: vec3<f32> = o + d * t;
            
            // Barycentric Coordinates (inside test)
            var area: f32 = dot(n, n);
            var u: f32 = dot(cross(c - b, q - b), n) / area;
            var v: f32 = dot(cross(a - c, q - c), n) / area;
            var w: f32 = 1.0 - u - v;

            if (t > 0.0 && u >= 0.0 && v >= 0.0 && w >= 0.0) {
                let sampleColor = textureLoad(channel0, vec2<i32>(fract(u * uvA + v * uvB + w * uvC) * vec2<f32>(textureDimensions(channel0))), 0);
                
                colorAccum += sampleColor * weight;
                totalWeight += weight;
            } else {
                colorAccum += vec4<f32>(0.0, 0.0, 0.0, 1.0) * weight;
                totalWeight += weight;
            }

        }
    }
    
    textureStore(screen, id.xy, colorAccum / totalWeight);
}
