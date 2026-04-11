
fn hash( p: vec2f ) -> vec2f // replace this by something better
{
	let a = vec2( dot(p,vec2(127.1,311.7)), dot(p,vec2(269.5,183.3)) );
	return -1.0 + 2.0*fract(sin(a)*43758.5453123);
}

const K1 = 0.366025404; // (sqrt(3)-1)/2;
const K2 = 0.211324865; // (3-sqrt(3))/6;

fn noise( p: vec2f ) -> f32
{
	let  i = floor( p + (p.x+p.y)*K1 );
    let  a = p - i + (i.x+i.y)*K2;
    let m = step(a.y,a.x); 
    let  o = vec2f(m,1.0-m);
    let  b = a - o + K2;
	let  c = a - 1.0 + 2.0*K2;
    let  h = max( 0.5-vec3f(dot(a,a), dot(b,b), dot(c,c) ), vec3(0.0) );
	let  n = h*h*h*h*vec3f( dot(a,hash(i+0.0)), dot(b,hash(i+o)), dot(c,hash(i+1.0)));
    return dot( n, vec3f(70.0) );
}

fn noise2( p: vec2f ) -> vec2f {
    return vec2(noise(p), noise(p.yx));
}


fn get_uv(id: vec2u) -> vec2f {
     // Viewport resolution (in pixels)
    let screen_size = vec2i(textureDimensions(screen));

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - i32(id.y)) - .5);

    // Normalised pixel coordinates (from 0 to 1)
    return fragCoord / vec2(f32(screen_size.x));
}

const OCTAVES = 2;
const LACUNARITY = 4.0;
const GAIN = 0.5;

fn get_height(id: vec2u) -> f32 {
    let uv = get_uv(id);

    var sum = 0.0;
    var amp = 1.0;
    var freq = custom.hill_frequency;
    var norm = 0.0;

    for (var i: i32 = 0; i < OCTAVES; i = i + 1) {
        let n = noise(uv * freq);
        sum += n * amp;
        norm += amp;
        freq *= LACUNARITY;
        amp *= GAIN;
    }

    return (sum / norm) * 0.5 + 0.5;
}

fn get_grad(id: vec2u) -> vec2f {
    let up = get_height(vec2u(vec2i(id) + vec2i(0,1)));
    let down = get_height(vec2u(vec2i(id) - vec2i(0,1)));
    let left = get_height(vec2u(vec2i(id) - vec2i(1,0)));
    let right = get_height(vec2u(vec2i(id) + vec2i(1,0)));

    return vec2f((right - left) * 0.5, (down - up) * 0.5);
}

fn get_angle_dir(theta: f32) -> vec2f {
    return vec2f(cos(theta), sin(theta));
}

fn get_sine_wave(id: vec2u, dir: vec2f) -> f32 {
    let uv = get_uv(id);
    let phase = dot(uv, dir) * custom.ridge_frequency;
    return sin(phase) * 0.5 + 0.5;
}

fn get_voronoi_cell(id: vec2u) -> vec2i {
    return vec2i(floor(vec2f(id) / custom.voronoi_scale));
}


fn get_point_in_cell(id: vec2u, cell: vec2i) -> vec2f {
    let cell_origin = vec2f(cell) * custom.voronoi_scale;
    return (vec2f(id) - cell_origin) / custom.voronoi_scale;
}

fn dir_to_rad(dir: vec2f) -> f32 {
    return atan2(dir.y, dir.x);
}

struct VoronoiOutput {
    closest_cell_center: vec2f,
    closest_cell_distance: f32,
}

fn get_closest_point_in_surrounding_cells(id: vec2u) -> vec2f {
    let cell = get_voronoi_cell(id);

    // vector realtive to pixel, length determines distance
    var closest_pos = vec2f(999.0);

    // Check 3x3 neighborhood around current cell
    for (var dy: i32 = -1; dy <= 1; dy++) {
        for (var dx: i32 = -1; dx <= 1; dx++) {
           let neighbour_cell = cell + vec2i(dx,dy);
           let local_pos = get_point_in_cell(id, neighbour_cell);
           let voronoi_center = hash(vec2f(neighbour_cell)) * 0.5 + 0.5;
           let to_center = voronoi_center - local_pos;

           if (length(to_center) < length(closest_pos)){
             closest_pos = to_center;
           }
        }
    }
    
    return closest_pos * custom.voronoi_scale;
}

// get surrounding gradients blended for smooth gullies 
fn get_blended_gradients(id: vec2u) -> vec2f {
    let cell = get_voronoi_cell(id);

    // vector realtive to pixel, length determines distance
    var closest_pos = vec2f(999.0);

    // Check 3x3 neighborhood around current cell
    for (var dy: i32 = -1; dy <= 1; dy++) {
        for (var dx: i32 = -1; dx <= 1; dx++) {
           let neighbour_cell = cell + vec2i(dx,dy);
           let local_pos = get_point_in_cell(id, neighbour_cell);
           let voronoi_center = hash(vec2f(neighbour_cell)) * 0.5 + 0.5;
           let to_center = voronoi_center - local_pos;

           if (length(to_center) < length(closest_pos)){
             closest_pos = to_center;
           }
        }
    }
    
    return closest_pos * custom.voronoi_scale;
}

fn rotate(p: vec2f, angle: f32) -> vec2f {
    return vec2f(
        p.x * cos(angle) - p.y * sin(angle),
        p.x * sin(angle) + p.y * cos(angle) 
    );
}

fn translate(p: vec2f, t: vec2f) -> vec2f {
    return p + t;
}


@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let base_height = get_height(id.xy);

    let voronoi_cell = get_voronoi_cell(id.xy);

    let closest = get_closest_point_in_surrounding_cells(id.xy);

    let closest_id = vec2u(vec2f(id.xy) + closest); 

    let grad = normalize(get_grad(id.xy));

    let perp_to_grad = rotate(grad, 3.1423  / 4.0);

    let colour = vec3f(get_sine_wave(id.xy, perp_to_grad));
    
    textureStore(screen, id.xy, vec4f(colour, 1.));
}
