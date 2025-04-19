// Classic Perlin noise 2D implementation
fn hash2(p: float2) -> float2 {
    let p2 = float2(dot(p, float2(127.1, 311.7)), dot(p, float2(269.5, 183.3)));
    return -1.0 + 2.0 * fract(sin(p2) * 43758.5453123);
}

/// Classic Perlin noise 2D implementation
/// Returns noise values in range [-1, 1]
/// 
/// Implementation steps:
/// 1. Determine grid cell coordinates (Pi) and fractional position (Pf)
/// 2. Calculate gradient vectors for each grid corner
/// 3. Compute dot products between gradients and distance vectors
/// 4. Interpolate using smooth quintic curve
fn perlin2d(p: float2) -> float {
    // Integer grid coordinates
    let Pi = floor(p);
    // Fractional position within grid cell
    let Pf = fract(p);

    // Get coordinates for 4 grid cell corners
    let i00 = Pi;              // (0,0)
    let i10 = Pi + float2(1, 0); // (1,0)
    let i01 = Pi + float2(0, 1); // (0,1)
    let i11 = Pi + float2(1, 1); // (1,1)
    
    // Generate pseudo-random gradient vectors for each grid corner
    // These determine the "slope" at each grid point
    let g00 = normalize(hash2(i00));
    let g10 = normalize(hash2(i10));
    let g01 = normalize(hash2(i01));
    let g11 = normalize(hash2(i11));
    
    // Distance vectors
    let d00 = Pf - float2(0, 0);
    let d10 = Pf - float2(1, 0);
    let d01 = Pf - float2(0, 1);
    let d11 = Pf - float2(1, 1);
    
    // Dot products
    let n00 = dot(g00, d00);
    let n10 = dot(g10, d10);
    let n01 = dot(g01, d01);
    let n11 = dot(g11, d11);
    
    // Quintic interpolation curve (6t^5 - 15t^4 + 10t^3)
    // Smoother than cubic interpolation, reduces grid artifacts
    let u = Pf * Pf * Pf * (Pf * (Pf * 6.0 - 15.0) + 10.0);
    
    // Bilinear interpolation of noise values
    let nx0 = mix(n00, n10, u.x); // Bottom edge
    let nx1 = mix(n01, n11, u.x); // Top edge
    let nxy = mix(nx0, nx1, u.y); // Final vertical mix
    
    // Scale result to approximate [-1,1] range
    return nxy * 1.4142;
}


/* 3D hash function matching our 2D implementation */
fn hash3(p: float3) -> float3 {
    let p2 = float3(
        dot(p, float3(127.1, 311.7, 241.1)),
        dot(p, float3(269.5, 183.3, 391.5)),
        dot(p, float3(419.2, 371.9, 149.7))
    );
    return -1.0 + 2.0 * fract(sin(p2) * 43758.5453123);
}

/// 3D Perlin noise implementation
/// Follows similar structure to 2D version but with 8 grid cell corners
/// Returns noise values in range [-1, 1]
fn perlin3d(p: float3) -> float {
    let Pi = floor(p);
    let Pf = fract(p);
    
    // Grid cell coordinates
    let i000 = Pi;
    let i100 = Pi + float3(1, 0, 0);
    let i010 = Pi + float3(0, 1, 0);
    let i110 = Pi + float3(1, 1, 0);
    let i001 = Pi + float3(0, 0, 1);
    let i101 = Pi + float3(1, 0, 1);
    let i011 = Pi + float3(0, 1, 1);
    let i111 = Pi + float3(1, 1, 1);
    
    // Gradient vectors
    let g000 = normalize(hash3(i000));
    let g100 = normalize(hash3(i100));
    let g010 = normalize(hash3(i010));
    let g110 = normalize(hash3(i110));
    let g001 = normalize(hash3(i001));
    let g101 = normalize(hash3(i101));
    let g011 = normalize(hash3(i011));
    let g111 = normalize(hash3(i111));
    
    // Distance vectors
    let d000 = Pf - float3(0, 0, 0);
    let d100 = Pf - float3(1, 0, 0);
    let d010 = Pf - float3(0, 1, 0);
    let d110 = Pf - float3(1, 1, 0);
    let d001 = Pf - float3(0, 0, 1);
    let d101 = Pf - float3(1, 0, 1);
    let d011 = Pf - float3(0, 1, 1);
    let d111 = Pf - float3(1, 1, 1);
    
    // Dot products
    let n000 = dot(g000, d000);
    let n100 = dot(g100, d100);
    let n010 = dot(g010, d010);
    let n110 = dot(g110, d110);
    let n001 = dot(g001, d001);
    let n101 = dot(g101, d101);
    let n011 = dot(g011, d011);
    let n111 = dot(g111, d111);
    
    // 3D quintic interpolation parameters
    let u = Pf * Pf * Pf * (Pf * (Pf * 6.0 - 15.0) + 10.0);
    return mix(
        mix(
            mix(n000, n100, u.x),
            mix(n010, n110, u.x),
            u.y
        ),
        mix(
            mix(n001, n101, u.x),
            mix(n011, n111, u.x),
            u.y
        ),
        u.z
    ) * 1.1547; // scale to [-1,1]
}

/// Fractal Brownian Motion (fBm) using Perlin noise
/// Combines multiple octaves of noise at increasing frequencies
/// - persistence: Amplitude multiplier between octaves (0.0-1.0)
/// - lacunarity: Frequency multiplier between octaves (typically 2.0)
/// Fractal Brownian Motion with enhanced controls
/// Applies to both 2D and 3D noise through shared parameters
fn perlin_fractal(m: float3) -> float {
    var amplitude = 1.0;
    var frequency = 1.0;
    var total = 0.0;
    var max_value = 0.0;
    
    // Main octave loop with flexible controls
    for (var i: u32 = 0; i < u32(custom.octaves); i++) {
        // Sample noise and accumulate
        total += perlin3d(m * frequency) * amplitude;
        max_value += amplitude;
        
        // Update amplitude and frequency using customizable rates
        amplitude *= custom.persistence * custom.gain;  // Combined persistence and gain
        frequency *= custom.lacunarity;  // Remove floor() for smooth transitions
        
        // Apply amplitude clamping to prevent overfloww
        amplitude = clamp(amplitude, 0.001, 100.0);
    }
    
    // Normalize and apply final transformations
    var value = total / max_value;
    value = pow(abs(value), custom.exponent) * sign(value); // Exponent for contrast control
    return mix(value, smoothstep(custom.range, custom.range, value), custom.smoothing);
}


@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    let screen_size = uint2(textureDimensions(screen));
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    let fragCoord = float2(id.xy) + .5;
    let resolution = float2(screen_size);
    let px = fragCoord / resolution.x;
    let p3 = float3(px, time.elapsed*0.025);

    var f = 0.;
    // Visualize different noise types side-by-side
    if (px.x < .5) { 
        // Left side: Raw 3D Perlin noise
        f = perlin3d(p3*custom.scale);
    } else { 
        // Right side: Fractal noise (4 octaves combined)
        f = perlin_fractal(p3*custom.scale);
    }
    
    // Remap noise from [-1,1] to [0,1] for display
    f = 0.5 + 0.5*f;
    
    // Smooth transition between left/right sections
    f *= smoothstep(0.0, 0.005, abs(px.x - 0.5));

    f = pow(f, 2.2); // perceptual gradient to linear colour space
    textureStore(screen, int2(id.xy), float4(f, f, f, 1.));
}
