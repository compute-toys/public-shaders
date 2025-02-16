
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let fragCoord = vec2f(id.xy) - 0.5 * vec2f(screen_size);

    let uv = fragCoord / f32(screen_size.y);

    let d = max(sdf_circle(uv, custom.rad), 0.);

    let max_brightness_inv = f32(0.001);
    let brightness_adj = custom.brightness;

    var b = brightness_adj / (max_brightness_inv + d);

    var pulse = sin(time.elapsed * 2.0) * 0.5 + 0.5;
    b *= pulse;

    var col = vec3f(custom.r, custom.g, custom.b);
    col *= b;
    col = tonemap_aces(col);

    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn sdf_scene(p: vec2f) -> f32 {
    const CELL_SIZE: f32 = 100.;
    const BALL_R: f32 = 30.;
    
    let cell_coord = modulo(p, vec2f(CELL_SIZE));
    let d_balls = sdf_circle(cell_coord - vec2f(CELL_SIZE / 2.), BALL_R);

    return d_balls;
}

fn sdf_circle(p: vec2f, r: f32) -> f32 {
    return length(p) - r;
}

fn tonemap_aces(x: vec3f) -> vec3f {
    let a = x * (x + 0.0245786) - 0.000090537;
    let b = x * (0.983729 * x + 0.4329510) + 0.238081;

    return clamp(a / b, vec3f(0.), vec3f(1.));
}

fn tonemap_agx(x: vec3f) -> vec3f {
	let x2 = vec3f(x * x);
	let x4 = vec3f(x2 * x2);
	return (15.5 * x4 * x2
		- 40.14 * x4 * x
		+ 31.96 * x4
		- 6.868 * x2 * x
		+ 0.4298 * x2
		+ 0.1191 * x
		- 0.00232);
}

fn modulo(x: vec2f, y: vec2f) -> vec2f {
    return x - y * floor(x / y);
}