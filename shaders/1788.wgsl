const PI = acos(-1.);

#define BufferA 0
#define BufferB 1
#define BufferC 2
#define BufferD 3

#define resolution vec2f(textureDimensions(screen).xy)

fn texelFetch(channel: int, coord: vec2i, lod: i32) -> vec4f {
    return passLoad(channel, coord, lod);
}

fn textureLod(channel: int, coord: vec2f, lod: f32) -> vec4f {
    return textureSampleLevel(pass_in, bilinear, coord, channel, lod);
}

fn texture(channel: int, coord: vec2f) -> vec4f {
    // this is not a fragment shader, so mip level estimation is impossible/hard
    // just use the 0th mip level
    return textureSampleLevel(pass_in, bilinear, coord, channel, 0.0);
}

fn gaussian(pos: vec2<f32>, sigma: f32) -> f32 {
	let left: f32 = 1. / (2. * PI * sigma * sigma);
	let right: f32 = exp(-dot(pos, pos) / (2. * sigma * sigma));
	return left * right;
} 

fn get_dir(dir: vec2f) -> vec2f {
    let d = normalize(texture(BufferC, dir).xy);
    return select(vec2f(1.0, 0.0), d, all(d == d)); // If NaN, return default direction
}

fn lum( v : vec3f) -> f32 {
	return dot(v, vec3(.299, .587, .114));
}

struct Camera {
    pos: vec2f,
    zoom : f32,
}
const ZOOM_SPEED = 0.0001;
const MOV_SPEED = 300.;

#storage state Camera

fn intensity(color: vec4f) -> f32 {
	return sqrt(color.x * color.x + color.y * color.y + color.z * color.z);
}

fn rgb2Luminance(rgb : vec3f) -> f32 {
    return 0.299*rgb.x + 0.587*rgb.y + 0.114*rgb.z ;
}

fn find_closest_vector(vectors: array<vec3f, 256>) -> vec3f {
    let directions = array<vec3f, 4>(
        vec3f(1.0, 0.0, 0.0),  // Vertical
        vec3f(0.0, 1.0, 0.0),  // Horizontal
        vec3f(1.0, 1.0, 0.0),  // Descending Diagonal
        vec3f(0.0, 1.0, 1.0)   // Ascending Diagonal
    );

    let weights = array<f32, 4>(
		1.1,
		1.1,
		0.8,
		0.8
		);

    var best_match: vec3f = vec3f(0.0, 0.0, 0.0); // Default to black
    var max_similarity: f32 = -1.0;
    var has_direction: bool = false;
    var black_pixel_count: i32 = 0;
    let black_threshold: i32 = int(custom.tolerance); // Adjust this threshold as needed

    for (var i: i32 = 0; i < 256; i++) {
        let v = vectors[i];

        if (all(v == vec3f(0.0, 0.0, 0.0))) {
            black_pixel_count += 1;
            continue; // Skip black pixels
        }

        has_direction = true;
        let norm_v = normalize(v);

        for (var j: i32 = 0; j < 4; j++) {
            let similarity = dot(norm_v, directions[j]) * weights[j]; // Apply bias
            if (similarity > max_similarity) {
                max_similarity = similarity;
                best_match = directions[j];
            }
        }
    }

    // If too many black pixels, return black
    if (black_pixel_count > black_threshold) {
        return vec3f(0.0, 0.0, 0.0);
    }

    return select(vec3f(0.0, 0.0, 0.0), best_match, has_direction);
}

fn char_(pos: vec2f, colour:vec3f, c: int) -> float4 {
    var p = pos % vec2f(1);

	let sdf = textureSampleLevel( channel0, trilinear, vec2f(p.x/16,p.y/49) + fract( vec2f(float(c)/16, float(c/16)/49) ), 0.).a;
    let col = mix(vec4f(0), vec4f(colour,1), sdf);
    return col;
}

fn map_outline (color:vec3f) -> i32 {
	if (all(color == vec3f(1,0,0))) {
		return 703;
	}
	if (all(color == vec3f(0,1,0))) {
		return 45;
	}
	if (all(color == vec3f(1,1,0))) {
		return 92;
	}
	if (all(color == vec3f(0,1,1))) {
		return 47;
	}
	return 0;
}

fn map_char(value: i32) -> i32 {

    let char_map: array<i32, 24> = array<i32, 24>(
        0,    // ' '
        151,  // '·'
        46,   // '∙'
        44,   // ','
        7,    // '•'
        43,   // '+'
        752,  // '■'
        591,  // 'ơ'
        687,  // '∞'
        42,   // '*'
        35,   // '#'
        763,  // '●'
        4,    // '♦'
        3,    // '♥'
        5,    // '♣'
        64,   // '@'
        2,    // '☻'
        8,    // '◘'
        10,   // '◙'
        750,  // '▓'
        745,  // '█'
        745,  // '█'
        745,  // '█'
        745   // '█'
    );

    if (value >= 0 && value < 24) {
        return char_map[value];
    } else {
        return 745; // Default value '█'
    }
}

fn sobel(channel: int, stepx: f32, stepy: f32, center: vec2f) -> vec3f {
    let offsets = array<vec2f, 8>(
        vec2f(-stepx, stepy), 	vec2f(-stepx, 0.),	 	vec2f(-stepx, -stepy),
        vec2f(0., stepy), 		vec2f(0., -stepy),
        vec2f(stepx, stepy), 	vec2f(stepx, 0.), 		vec2f(stepx, -stepy)
    );

    var intensities: array<f32, 8>;
    for (var i: i32 = 0; i < 8; i++) {
        intensities[i] = intensity(texture(channel, center + offsets[i]));
    }

    let x = intensities[0] + 2.0 * intensities[1] + intensities[2] -
            intensities[5] - 2.0 * intensities[6] - intensities[7];

    let y = -intensities[0] - 2.0 * intensities[3] - intensities[5] +
             intensities[2] + 2.0 * intensities[4] + intensities[7];

    return vec3f(x * x - x * y, y * y - x * y, x * y);
}


fn soft_threshold (lum: f32 , threshold: f32, soft: f32) -> f32 {
	let f: f32 = soft / 2.;
	let a: f32 = threshold - f;
	let b: f32 = threshold + f;
	let v: f32 = smoothstep(a, b, lum);
	return v;
}

fn is_nan(v: vec3f) -> bool {
    return any(v != v); // True if any component is NaN
}

@compute @workgroup_size(1, 1)
#dispatch_once initialization
fn initialization() {
    let screen_size = textureDimensions(screen);
    let texture_size = textureDimensions(channel1);
    state.pos = vec2f(screen_size / 2);
    state.zoom = 1;
}

@compute @workgroup_size(16, 16)
fn Pass_Threshold(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);
	let texture_size = textureDimensions(channel1);
    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    //update view
    if keyDown(83) {
        state.pos += (vec2f(0, MOV_SPEED) / vec2f(f32(screen_size.x)) ) ;
    }
	if keyDown(87) {
        state.pos += ( vec2f(0, - MOV_SPEED) / vec2f(f32(screen_size.x)) )  ;
    }
	if keyDown(68) {
        state.pos += ( vec2f(MOV_SPEED, 0 ) / vec2f(f32(screen_size.y)) )  ;
    }
	if keyDown(65) {
        state.pos += ( vec2f(- MOV_SPEED, 0 ) / vec2f(f32(screen_size.y)) )  ;
    }
    if keyDown(16) { // up arrow
        state.zoom += ZOOM_SPEED ;
    }
    if keyDown(17) { // down arrow
        state.zoom -= ZOOM_SPEED;
    }


    //navigating the image
    var uv = vec2f(id.xy);
    uv /= vec2f(texture_size) ;
    uv = uv  * state.zoom;
    uv = uv + state.pos/ vec2f(screen_size);

	var col = textureSampleLevel(channel1, trilinear,uv,0.).rgb;

	// soft threshold
	col = vec3f(soft_threshold ( col.x , custom.threshold, custom.soft));

    // Output to screen (linear colour space)
    textureStore(pass_out, int2(id.xy), BufferA, vec4f(col, 0.));
}

//make structure tensor from partial derivative using sobel operator
const sobel_mat: mat3x3f = mat3x3f
    (-1., 0., 1.,
     -2., 0., 2.,
     -1., 0., 1.);

@compute @workgroup_size(16, 16)
fn Pass_A(@builtin(global_invocation_id) id: uint3) {
    let fragCoord = vec2f(id.xy);
	let uv = fragCoord/resolution.xy;

    var gradX: vec3f = vec3f(0);
	var gradY: vec3f = gradX;

	for (var i: i32 = 0; i < 3; i++) {

		for (var j: i32 = 0; j < 3; j++) {
			let offsetUV: vec2f = vec2f(f32(i - 1), f32(j - 1)) / resolution.xy;
			let col: vec3f = texture(BufferA, uv + offsetUV).rgb;
			gradX += (sobel_mat[i][j] * col);
			gradY += (sobel_mat[j][2 - i] * col);
		}

	}

	gradX = gradX * (0.25);
	gradY = gradY * (0.25);
	let fragColor = vec4f(
		dot(gradX, gradX), 
		dot(gradX, gradY), 
		dot(gradY, gradY),
		1.
		);

    textureStore(pass_out, int2(id.xy), BufferB, fragColor);
}

#define K_1 10
#define SIGMA_C 2.9//[0, 5]

@compute @workgroup_size(16, 16)
fn Pass_B(@builtin(global_invocation_id) id: uint3) {
    //blur structure tensor, calc eigenvectors. this is the etf.
	let fragCoord = vec2f(id.xy);
	let uv = fragCoord/resolution.xy;
	var sum: vec4f;

	for (var i: i32 = -K_1; i <= K_1; i++) {
		for (var j: i32 = -K_1; j <= K_1; j++) {
			let offset: vec2f = vec2f(f32(i), f32(j));
			let weight: f32 = gaussian(offset, SIGMA_C);
			let offsetUV: vec2f = offset /resolution.xy;
			var efg: vec3f = texture(BufferB, uv + offsetUV).rgb;
			sum += (weight * vec4f(efg, 1.));
		}

	}

	let efg: vec3f = sum.rgb / sum.a;
	
	//calc eigenvalues
	let e: f32 = efg.x;
	let f: f32 = efg.y;
	let g: f32 = efg.z;

	let disc: f32 = sqrt(pow(e - g, 2.) + 4. * f * f);
	let l1: f32 = 0.5 * (e + g + disc);
	let l2: f32 = 0.5 * (e + g - disc);

	//calc eigenvectors
	let v1: vec2f = vec2f(f, l1 - e);
	let v2: vec2f = vec2f(l2 - g, f);

	let fragColor = vec4f(v1, v2);
    textureStore(pass_out, int2(id.xy), BufferC, fragColor);
}

#define K_2 10
#define SIGMA_E 3.5//[0, 5]
#define SIGMA_K 2.5//[1, 5]

#define TAU 1.4//?sharpness?

#define N 5//posterize palette size

@compute @workgroup_size(16, 16)
fn Pass_C(@builtin(global_invocation_id) id: uint3) {
        
	//difference of 1d gaussians
	let fragCoord = vec2f(id.xy);
	let uv: vec2f = fragCoord / resolution.xy;
	let dir: vec2f = get_dir(uv);

	var sumA: vec4f = vec4f(0.);
	var sumB: vec4f = sumA;

	for (var i: i32 = -K_2; i <= K_2; i++) {
		let offset: vec2f = dir * f32(i);
		let weightA: f32 = gaussian(offset, SIGMA_E);
		let weightB: f32 = gaussian(offset, SIGMA_E * SIGMA_K);
		let offsetUV: vec2f = offset / resolution.xy;
		let col: vec4f = texture(BufferA, uv + offsetUV);
		sumA = sumA + (weightA * vec4f(col.rgb, 1.));
		sumB = sumB + (weightB * vec4f(col.rgb, 1.));
	}

	let meanA: vec3f = sumA.rgb / (sumA.a);
	let meanB: vec3f = sumB.rgb / (sumB.a);
	let diff: vec3f = (1. + TAU) * meanA - TAU * meanB;
	var fragColor = vec4f(diff, 1.);

	// posterize
	fragColor = floor(fragColor * f32(N - 1.) + 0.5) / f32(N - 1.);

    textureStore(pass_out, int2(id.xy), BufferD, fragColor);
}

#define K_3 7//[3, 10]
#define SIGMA_M 2.8//[0, 5]

#define PHI 2.9//[0, 10]
#define EPSILON .7//[0, 1]

@compute @workgroup_size(16, 16)
fn Pass_D(@builtin(global_invocation_id) id: uint3) {
    //edge aligned blur, threshold
	
	let fragCoord = vec2f(id.xy);
	var fragColor: vec4f;

	let uv: vec2f = fragCoord / resolution.xy;

	var pt0: vec2f = fragCoord + 0.5;
	var pt: vec2f = pt0;
	let dir0: vec2f = get_dir(uv);
	var dir: vec2f = dir0;
	var sum: vec4f = vec4f(0.);

	for (var i: i32 = -K_3; i <= K_3; i++) {
		//middle: init for ahead
		if (i == 0) {		
			pt = pt0;
			dir = dir0;
		} else { 		
			pt = pt + (f32(sign(i)) * dir);
		}

		//sampling
		let weight: f32 = gaussian(pt - pt0, SIGMA_M);
		let ptUV: vec2f = pt / resolution.xy;
		let col: vec4f = texture(BufferD, ptUV);
		sum = sum + (weight * vec4f(col.rgb, 1.));
		dir = get_dir(ptUV);
		
	}

	let mean: vec3f = sum.rgb / sum.a;

	var diff: f32 = lum(mean);
	if (diff > EPSILON) {	
		fragColor = vec4f(1.);
	} else { 	
		fragColor = vec4f(1. + tanh(PHI * (diff - EPSILON)));
	}

    textureStore(pass_out, int2(id.xy), BufferA, fragColor);
}

#define K_4 6//[3, 10]
#define SIGMA_A 3.2//[0, 3]

@compute @workgroup_size(16, 16)
fn Pass_E(@builtin(global_invocation_id) id: uint3) {
	let fragCoord = vec2f(id.xy);
	var fragColor: vec4f;

	let uv: vec2f = fragCoord / resolution.xy;
	let pt0: vec2f = fragCoord + 0.5;
	var pt: vec2f = pt0;
	let dir0: vec2f = get_dir(uv);
	var dir: vec2f = dir0;
	var sum: vec2f = vec2f(0.);

	for (var i: i32 = -K_4; i <= K_4; i++) {
		//which way are we going?
		switch (sign(i)) {
			case -1 {
				//behind: step back
				pt = pt - (dir);
				break;
			}
			case 0 {
				//middle: init for ahead
				pt = pt0;
				dir = dir0;
				break;
			}
			case 1 {
				//ahead: step forward
				pt = pt + (dir);
				break;
			}
			default {}
		}
		//sampling
		let weight: f32 = gaussian(pt - pt0, SIGMA_A);
		let ptUV: vec2f = pt / resolution.xy;
		var col: vec4f = texture(BufferA, ptUV);
		sum = sum + (vec2f(col.r, 1.) * weight);
		dir = get_dir(ptUV);
	}

	let mean: f32 = sum.x / sum.y;
	var col: vec4f = texture(BufferA, uv);
	col = col / (max(col.r, max(col.g, col.b)));
	fragColor = col * mean;

    textureStore(pass_out, int2(id.xy), BufferA, fragColor);
}

@compute @workgroup_size(16, 16)
fn Pass_Threshold2(@builtin(global_invocation_id) id: uint3) {

    let fragCoord = vec2f(id.xy);
    let uv = fragCoord / resolution.xy;

	var col : vec3f;

	// pass 1 sobel filter
	col = texture(BufferA, uv).rgb;
	// soft threshold
	col = vec3f(soft_threshold ( col.x , custom.threshold2, custom.soft2));
	
    // Output to screen (linear colour space)
    textureStore(pass_out, int2(id.xy), BufferB, vec4f(col, 0.));
}

@compute @workgroup_size(16, 16)
fn Pass_Sobel(@builtin(global_invocation_id) id: uint3) {

    let fragCoord = vec2f(id.xy);
    let uv = fragCoord / resolution.xy;

	var col : vec3f;

    col = vec3f(sobel( BufferB ,custom.steps/ float(resolution.x),custom.steps/ float(resolution.y),uv));

    // Output to screen (linear colour space)
    textureStore(pass_out ,int2(id.xy), BufferA, vec4f(col, 0.));
}

var<workgroup> wgmem: array<vec3f,256>;
enable subgroups;

@compute @workgroup_size(10, 10)
fn Pass_ascii(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(workgroup_id) wid: vec3u,
    @builtin(local_invocation_index) lid: u32,
    @builtin(subgroup_invocation_id) sgid: u32
) {
    let screen_size = textureDimensions(screen);
	let texture_size = textureDimensions(channel1);

    // Compute UV coordinates
    let fragCoord = vec2f(gid.xy);
    let uv = fragCoord / vec2f(screen_size);

    // UV for character mapping (optimized)
    let uv_char = vec2f(f32(lid % 10), f32(lid / 10)) * 0.1;

    // Load color from texture (avoiding redundant operations)
    let col = textureSampleLevel(pass_in, bilinear, uv, BufferA, 0.0).rgb;

    // Store in shared memory
    wgmem[lid] = col;
    workgroupBarrier(); // Only needed before reading shared memory

    var v: vec3f;
    var c: i32;
	var col_ascii: vec3f;
	var char : i32;

    // Only first thread in the subgroup calculates
    if (lid == 0) {
        v = find_closest_vector(wgmem);
        c = map_outline(v);
		col_ascii = textureSampleLevel(channel1, bilinear,uv,0.).rgb;
		let l = rgb2Luminance(col_ascii);
		 char = map_char(int(l * 24));
    }

    // Efficient subgroup broadcasting
    c = subgroupBroadcastFirst(c);
	char = subgroupBroadcastFirst(char);

    // Compute final result
    let line = char_(uv_char, vec3f(1), c).rgb;
	// col_ascii = char_(uv_char, col_ascii, char).xyz;
    col_ascii = char_(uv_char, vec3f(1), char).xyz;

    // Store output
    textureStore(screen, gid.xy, vec4f(line, 0.));
	// textureStore(screen, gid.xy, vec4f(col_ascii, 0.));
}