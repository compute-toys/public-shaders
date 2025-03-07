
const PI = acos(-1);

// This template assumes that channels are in the following order both for inputs and outputs of the buffers:
#define PassA 0
#define PassB 1
#define PassC 2
#define PassD 3

#define resolution vec2f(textureDimensions(screen).xy)

fn is_nan(v: vec3f) -> bool {
    return any(v != v); // True if any component is NaN
}

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

fn get_dir(dir:vec2f) -> vec2f {
	var d = normalize(texture(PassB, dir).xy);

	if (d.x != d.x || d.y != d.y) {
        return vec2f(1.0, 0.0); // Debug: If NaN, return a valid direction
    }

	return d;
}

fn lum( v : vec3f) -> f32 {
	return dot(v, vec3(.299, .587, .114));
}
//make structure tensor from partial derivative using sobel operator
const sobel: mat3x3f = mat3x3f
    (-1., 0., 1.,
     -2., 0., 2.,
     -1., 0., 1.);

fn BufferA(fragCoord: vec2f) -> vec4f {
    // Normalized pixel coordinates (from 0 to 1)
    let uv = fragCoord/resolution.xy;

    var gradX: vec3f = vec3f(0);
	var gradY: vec3f = gradX;

	for (var i: i32 = 0; i < 3; i = i + 1) {

		for (var j: i32 = 0; j < 3; j = j + 1) {
			let offsetUV: vec2f = vec2f(f32(i - 1), f32(j - 1)) / resolution.xy;
			let col: vec3f = textureSampleLevel(channel0,bilinear, uv + offsetUV, 0).rgb;
			gradX += (sobel[i][j] * col);
			gradY += (sobel[j][2 - i] * col);
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

    return fragColor;
}

#define K_1 10
#define SIGMA_C 2.9//[0, 5]

fn BufferB(fragCoord: vec2f) -> vec4f {
	//blur structure tensor, calc eigenvectors. this is the etf.
	let uv = fragCoord/resolution.xy;
	var sum: vec4f;

	for (var i: i32 = -K_1; i <= K_1; i = i + 1) {
		for (var j: i32 = -K_1; j <= K_1; j = j + 1) {
			let offset: vec2f = vec2f(f32(i), f32(j));
			let weight: f32 = gaussian(offset, SIGMA_C);
			let offsetUV: vec2f = offset /resolution.xy;
			var efg: vec3f = texture(PassA, uv + offsetUV).rgb;
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
    return fragColor;
}

#define K_2 10
#define SIGMA_E 3.5//[0, 5]
#define SIGMA_K 2.5//[1, 5]

#define TAU 1.4//?sharpness?

#define N 5//posterize palette size

fn BufferC(fragCoord: vec2f) -> vec4f {

    //difference of 1d gaussians
	let uv: vec2f = fragCoord / resolution.xy;
	let dir: vec2f = get_dir(uv);

	if (dir.x != dir.x || dir.y != dir.y) { // Detect NaN in direction vector
    return vec4f(0.0, 1.0, 0.0, 1.0); // Green -> NaN in `dir`
	}

	var sumA: vec4f = vec4f(0.);
	var sumB: vec4f = sumA;

	for (var i: i32 = -K_2; i <= K_2; i = i + 1) {
		let offset: vec2f = dir * f32(i);
		let weightA: f32 = gaussian(offset, SIGMA_E);
		let weightB: f32 = gaussian(offset, SIGMA_E * SIGMA_K);
		let offsetUV: vec2f = offset / resolution.xy;
		let col: vec4f = textureSampleLevel(channel0,bilinear, uv + offsetUV, 0);
		sumA = sumA + (weightA * vec4f(col.rgb, 1.));
		sumB = sumB + (weightB * vec4f(col.rgb, 1.));
	}

	let meanA: vec3f = sumA.rgb / (sumA.a);
	let meanB: vec3f = sumB.rgb / (sumB.a);
	let diff: vec3f = (1. + TAU) * meanA - TAU * meanB;
	var fragColor = vec4f(diff, 1.);

	// posterize
	fragColor = floor(fragColor * f32(N - 1.) + 0.5) / f32(N - 1.);

    return fragColor;
}

#define K_3 7//[3, 10]
#define SIGMA_M 2.8//[0, 5]

#define PHI 2.9//[0, 10]
#define EPSILON .7//[0, 1]

fn BufferD(fragCoord: vec2f) -> vec4f {
	//edge aligned blur, threshold
   
	var fragColor: vec4f;

	let uv: vec2f = fragCoord / resolution.xy;

	var pt0: vec2f = fragCoord + 0.5;
	var pt: vec2f = pt0;
	let dir0: vec2f = get_dir(uv);
	var dir: vec2f = dir0;
	var sum: vec4f = vec4f(0.);

	for (var i: i32 = -K_3; i <= K_3; i = i + 1) {
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
		let col: vec4f = texture(PassC, ptUV);
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
	return fragColor;
}

#define K_4 6//[3, 10]
#define SIGMA_A 3.2//[0, 3]

fn Image(fragCoord: vec2f) -> vec4f {    
	//antialiasing 2nd edge aligned blur
	var fragColor: vec4f;

	let uv: vec2f = fragCoord / resolution.xy;
	let pt0: vec2f = fragCoord + 0.5;
	var pt: vec2f = pt0;
	let dir0: vec2f = get_dir(uv);
	var dir: vec2f = dir0;
	var sum: vec2f = vec2f(0.);

	for (var i: i32 = -K_4; i <= K_4; i = i + 1) {
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
		var col: vec4f = texture(PassD, ptUV);
		sum = sum + (vec2f(col.r, 1.) * weight);
		dir = get_dir(ptUV);
	}

	let mean: f32 = sum.x / sum.y;
	var col: vec4f = textureSampleLevel(channel0,bilinear, uv, 0.);
	col = col / (max(col.r, max(col.g, col.b)));
	fragColor = col * mean;
	return fragColor;
}

@compute @workgroup_size(16, 16)
fn BufferA_Pass(@builtin(global_invocation_id) id: uint3) {
    let col = BufferA(vec2f(id.xy));
    textureStore(pass_out, int2(id.xy), 0, col);
}

@compute @workgroup_size(16, 16)
fn BufferB_Pass(@builtin(global_invocation_id) id: uint3) {
    let col = BufferB(vec2f(id.xy));
    textureStore(pass_out, int2(id.xy), 1, col);
}

@compute @workgroup_size(16, 16)
fn BufferC_Pass(@builtin(global_invocation_id) id: uint3) {
    let col = BufferC(vec2f(id.xy));
    textureStore(pass_out, int2(id.xy), 2, col);
}

@compute @workgroup_size(16, 16)
fn BufferD_Pass(@builtin(global_invocation_id) id: uint3) {
    let col = BufferD(vec2f(id.xy));
    textureStore(pass_out, int2(id.xy), 3, col);
}

@compute @workgroup_size(16, 16)
fn Main_Pass(@builtin(global_invocation_id) id: uint3) {
    let col = Image(vec2f(id.xy));
    textureStore(screen, int2(id.xy), col);
}
