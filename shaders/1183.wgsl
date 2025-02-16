
// The moving thing
struct Tracer {
    x: float,
    y: float,
    angle: float,
    hue: float,
};

// The numbers
#define T 128
#define Z 4096 * 128
#define ZT 4096

#storage tracers array<Tracer,Z>
#storage current_index u32

// Hash function www.cs.ubc.ca/~rbridson/docs/schechter-sca08-turbulence.pdf
fn hash(ini: u32) -> u32
{
    var state = ini;
    state ^= 2747636419u;
    state *= 2654435769u;
    state ^= state >> 16;
    state *= 2654435769u;
    state ^= state >> 16;
    state *= 2654435769u;
    return state;
}

// ah yes
fn scaleToRange01(state: u32) -> f32
{
    return f32(state) / 4294967295.0;
}

#dispatch_once init
#workgroup_count init ZT 1 1
@compute @workgroup_size(T, 1, 1)
fn init(@builtin(global_invocation_id) id3: vec3u) {
 
    var screen_size = textureDimensions(screen);
    var i = id3.x;

    if (i >= Z) {
        return;
    }

    // so basically put the moving things in a circle by having a random angle + distance
    var angle = scaleToRange01(hash(i * 121)) * 2.0 * 3.14;
    var max_distance = f32(min(screen_size.x, screen_size.y) / 2);
    var distance = scaleToRange01(hash(1 + i * 1337)) * (max_distance - 90.) + 30.;
    var x = f32(screen_size.x / 2) + distance * sin(angle);
    var y = f32(screen_size.y / 2) + distance * cos(angle);
    tracers[i].x = x;
    tracers[i].y = y;
    tracers[i].angle = angle;
    tracers[i].hue = scaleToRange01(hash(i * 121));
}

// use the sensor to sense, using the configured direction
// distance between hue value is used to check compatibility
fn get_sample(x: f32, y: f32, angle: f32, hue: f32) -> f32 {
    var vx = int(x + sin(angle) * custom.sensor_distance);
    var vy = int(y + cos(angle) * custom.sensor_distance);
    var sensor_size = int(custom.sensor_size);
    var sum = 0.0;


	for (var offsetX = -sensor_size; offsetX <= sensor_size; offsetX ++) {
		for (var offsetY = -sensor_size; offsetY <= sensor_size; offsetY ++) {

            var value = passLoad(0, vec2i(vx + offsetX, vy + offsetY), 0);
            var distance = min(abs(value.y - hue), abs(value.y + 1.0 - hue));
            var distance2 = min(distance, abs(value.y - 1.0 - hue));

			sum += (custom.hue_tolerance - distance2) * passLoad(0, vec2i(vx + offsetX, vy + offsetY), 0).x;
		}
	}


    return sum;
}

// good ol conversion code i found on the web
fn hsv_to_rgb(c: vec3f) -> vec3f {
    var K = vec4f(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    var p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    var cc = p - K.xxx;
    cc.x = clamp(cc.x, 0.0, 1.0);
    cc.y = clamp(cc.y, 0.0, 1.0);
    cc.z = clamp(cc.z, 0.0, 1.0);
    return c.z * mix(K.xxx, cc, c.y);
}

// simulation:
// - move the tracer
// - find next travel angle by sensing
// - mark position for the diffusion pass
#workgroup_count simulate ZT 1 1
@compute @workgroup_size(T, 1, 1)
fn simulate(@builtin(global_invocation_id) id3: vec3u) {
    var i = id3.x;
    var t = tracers[i];

    // teleport the tracer in case of mouse click 
    if (i >= current_index && i < current_index + 30 && mouse.click != 0) {
        current_index = current_index + 30;
        if (current_index > Z) {
            current_index = 0;
        }
        t.x = float(mouse.pos.x);
        t.y = float(mouse.pos.y);
    }


    t.x += sin(t.angle) * custom.speed;
    t.y += cos(t.angle) * custom.speed;


	var r1 = scaleToRange01(hash(uint(t.x) * 3000 + uint(t.y) + hash(i * 7 + time.frame * 10))) * custom.entropy;
    var r2 = scaleToRange01(hash(5000 + uint(t.x) * 3000 + uint(t.y) + hash(i * 3 + time.frame * 10))) *  custom.entropy;
	var r3 = scaleToRange01(hash(10000 + uint(t.x) * 3000 + uint(t.y) + hash(i * 11 + time.frame * 10))) *  custom.entropy;

    var screen_size = textureDimensions(screen);
    if (t.x >= f32(screen_size.x) - 1 || t.x < 0) {
        t.angle = 2. * 3.14 - t.angle;
        t.x += sin(t.angle) * custom.speed;
    }
    if (t.y >= f32(screen_size.y) - 1 || t.y < 0) {
        t.angle = 3.14 - t.angle;
        t.y += cos(t.angle) * custom.speed;
    }

    var v_prev   = passLoad(0, vec2i(int(t.x), int(t.y)), 0);
    var v_left   = get_sample(t.x, t.y, t.angle - custom.sensor_angle, t.hue) + r1;
    var v_center = get_sample(t.x, t.y, t.angle, t.hue);
    var v_right  = get_sample(t.x, t.y, t.angle + custom.sensor_angle, t.hue) + r3;

    

    
    if (v_center >= v_left && v_center >= v_right) {
        // do nothing
    } else if (v_left >= v_right) {
        // turn left
        t.angle    -= custom.rotation_speed;
    } else {
        // turn right
        t.angle    += custom.rotation_speed;
    }

    tracers[i] = t;
    passStore(0, vec2i(int(t.x), int(t.y)), vec4f(1., t.hue, 1., 1.));
}

// blend two hsl values taking care of luminosity and the fact that
// hue is on a circle
fn blendHsl(a: vec2f, b: vec4f, mix: float) -> vec2f {
    var ay = a.y;
    
    if (a.y < (b.y - 0.5)) {
        ay = a.y + 1.0;
    } else if (a.y > b.y + 0.5) {
        ay = a.y - 1.0;
    }
    
    var avgHue = (ay * a.x + b.y * b.x * mix) / (0.0001 + a.x + b.x * mix);
    return vec2f((a.x + b.x * mix) / (1 + mix), fract(avgHue));
}

// diffuse the trail of the tracers on the 8 neighbors
@compute @workgroup_size(16, 16)
fn diffuse(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    var value = passLoad(0, vec2i(int(id.x), int(id.y)), 0);
    var t = passLoad(0, vec2i(int(id.x - 1), int(id.y - 1)), 0).xy;
    t = blendHsl(t, passLoad(0, vec2i(int(id.x - 1), int(id.y    )), 0), 1.);
    t = blendHsl(t, passLoad(0, vec2i(int(id.x - 1), int(id.y + 1)), 0), 1./2.);
    t = blendHsl(t, passLoad(0, vec2i(int(id.x    ), int(id.y - 1)), 0), 1./3.);
    t = blendHsl(t, passLoad(0, vec2i(int(id.x    ), int(id.y + 1)), 0), 1./4.);
    t = blendHsl(t, passLoad(0, vec2i(int(id.x + 1), int(id.y - 1)), 0), 1./5.);
    t = blendHsl(t, passLoad(0, vec2i(int(id.x + 1), int(id.y    )), 0), 1./6.);
    t = blendHsl(t, passLoad(0, vec2i(int(id.x + 1), int(id.y + 1)), 0), 1./7.);

    var result = blendHsl(t, value, 1 / (0.001 + custom.diffusion_ratio));
    var avgHue = result.y;
    var avgValue = result.x * custom.decay_ratio;
    passStore(0, vec2i(int(id.x), int(id.y)), vec4f(avgValue, avgHue, 1.0, 1.0)) ;
    

    var rgb = hsv_to_rgb(vec3f(avgHue, 0.9, value.x * 0.9));

    textureStore(screen, id.xy, vec4f(rgb, 1.0));

}
