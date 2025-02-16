// Floyd-Steinberg dithering parallelized across a single workgroup

// Enable to show execution order (need to restart shader)
const ANIMATE = false;
// Number of colors to quantize to (per channel)
const PALETTE_SIZE = 2;
// Enable to quantize luminance instead of color
const LUMINANCE = false;

fn to_pixel(color: vec4f) -> vec3f {
    if LUMINANCE {
        return vec3(saturate(dot(vec3(0.2126, 0.7152, 0.0722), color.rgb)));
    } else {
        return saturate(color.rgb);
    }
}

fn srgb_eotf(col: vec3f) -> vec3f {
    let c = col > vec3(0.04045);
    return select(col / 12.92, pow((col + 0.055) / 1.055, vec3(2.4)), c);
}

fn srgb_inverse_eotf(col: vec3f) -> vec3f {
    let c = col > vec3(0.0031308);                                                
    return select(col * 12.92, 1.055 * pow(col, vec3(1.0 / 2.4)) - 0.055, c);
}

fn find_closest_palette_color(pixel: vec3f) -> vec3f {
    let a = f32(PALETTE_SIZE) - 1.0;
    return srgb_eotf(round(a * srgb_inverse_eotf(saturate(pixel))) / a);
}

#storage pixels array<vec3f>

fn at(x: i32, y: i32) -> i32 {
    let dimensions = vec2i(textureDimensions(screen));

    if x < 0 || dimensions.x <= x || y < 0 || dimensions.y <= y {
        return dimensions.x * dimensions.y;
    }

    return x + y * dimensions.x;
}

@compute @workgroup_size(16, 16)
fn initialize_pixels(@builtin(global_invocation_id) id: vec3u) {
    let dimensions = textureDimensions(screen);

    if any(id.xy >= dimensions) {
        return;
    }

    let uv = (vec2f(id.xy) + 0.5) / vec2f(dimensions);
    let color = textureSampleLevel(channel0, bilinear, uv, 0);
    pixels[at(i32(id.x), i32(id.y))] = to_pixel(color);
}

const wgs = 256;
var<workgroup> dimensions_: vec2i;
#workgroup_count quantize 1 1 1
@compute @workgroup_size(wgs)
fn quantize(@builtin(local_invocation_index) id: u32) {
    // We gotta do this to make uniformity analysis happy
    if id == 0u {
        dimensions_ = vec2i(textureDimensions(screen));
    }
    let dimensions = workgroupUniformLoad(&dimensions_);

    let width = max(wgs * 3, dimensions.x);
    var i_max = (dimensions.y + wgs - 1) / wgs * width + (wgs - 1) * 3;

    if ANIMATE {
        i_max = min(i_max, i32(time.frame));
    }

    for (var i = 0; i < i_max; i++) {
        storageBarrier();
        let wi = i - i32(id) * 3;
        let y = wi / width * wgs + i32(id);
        let x = wi % width;
        
        if 0 <= wi && 0 <= x && x < dimensions.x && 0 <= y && y < dimensions.y {
            // https://en.wikipedia.org/wiki/Floyd–Steinberg_dithering
            let old_pixel = pixels[at(x, y)];
            let new_pixel = find_closest_palette_color(old_pixel);
            pixels[at(x, y)] = new_pixel;
            let quant_error = old_pixel - new_pixel;
            pixels[at(x + 1, y + 0)] += quant_error * 7.0 / 16.0;
            pixels[at(x - 1, y + 1)] += quant_error * 3.0 / 16.0;
            pixels[at(x + 0, y + 1)] += quant_error * 5.0 / 16.0;
            pixels[at(x + 1, y + 1)] += quant_error * 1.0 / 16.0;
        }
    }
}

// Single-threaded version (very slow)
// #workgroup_count quantize 1 1 1
// @compute @workgroup_size(1)
// fn quantize() {
//     let dimensions = vec2i(textureDimensions(screen));
//     for (var y = 0; y < dimensions.y; y++) {
//         for (var x = 0; x < dimensions.x; x++) {
//             // https://en.wikipedia.org/wiki/Floyd–Steinberg_dithering
//             let old_pixel = pixels[at(x, y)];
//             let new_pixel = find_closest_palette_color(old_pixel);
//             pixels[at(x, y)] = new_pixel;
//             let quant_error = old_pixel - new_pixel;
//             pixels[at(x + 1, y + 0)] += quant_error * 7.0 / 16.0;
//             pixels[at(x - 1, y + 1)] += quant_error * 3.0 / 16.0;
//             pixels[at(x + 0, y + 1)] += quant_error * 5.0 / 16.0;
//             pixels[at(x + 1, y + 1)] += quant_error * 1.0 / 16.0;
//         }
//     }
// }

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    if any(id.xy >= textureDimensions(screen)) {
        return;
    }

    textureStore(screen, id.xy, vec4(pixels[at(i32(id.x), i32(id.y))], 1.0));
}