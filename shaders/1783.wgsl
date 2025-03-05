struct Camera {
    pos: vec2f,
    zoom : f32,
}

const MOV_SPEED = 0.0004;
const ZOOM_SPEED = 0.00009;
const BLOCK_SPEED = 0.005;

#storage state Camera
#storage block_size f32

fn char_(pos: vec2f, colour:vec3f, c: int) -> float4 {
    var p = pos % vec2f(1);

	let sdf = textureSampleLevel( channel0, trilinear, vec2f(p.x/16,p.y/49) + fract( vec2f(float(c)/16, float(c/16)/49) ), 0.).a;
    let col = mix(vec4f(0), vec4f(colour,1), sdf);
    return col;
}

fn rgb2Luminance(rgb : vec3f) -> f32 {
    return 0.299*rgb.x + 0.587*rgb.y + 0.114*rgb.z ;
}

fn map_char(value : i32) -> i32 {
    //choose character per value
    switch value {
        case 0 {      
            return 0;   //
        }
        case 1 { 
            return 151; // ·
        }
        case 2 {  
            return 46;  // ∙
        }
        case 3 {  
            return 44;  // ,
        }
        case 4 {
            return 7;   // •
        }
        case 5 {
            return 43;  // +
        }
        case 6 {
            return 752; // ■
        }
        case 7 {
            return 591; // ơ
        }
        case 8 {
            return 687; // ∞
        }
        case 9 {
            return 42;  // *
        }
        case 10 {
            return 35;  // #
        }
        case 11 {
            return 763; // ●
        }
        case 12 {
            return 4;   // ♦
        }
        case 13 {
            return 3;   // ♥
        }
        case 14 {
            return 5;   // ♣
        }
        case 15 {
            return 64;  // @
        }
        case 16 {
            return 2;   // ☻
        }
        case 17 {
            return 8;   // ◘
        }
        case 18 {
            return 10;  // ◙
        }
        case 19 {
            return 750; // ▓
        }
        case 20 {
            return 745; // █
        }
        case 21 {
            return 745; // █
        }
        case 22 {
            return 745; // █
        }
        case 23 {
            return 745; // █
        }
        default {    
            return 745; // █
        }
    }
}
@compute @workgroup_size(1, 1)
#dispatch_once initialization
fn initialization() {
    let screen_size = textureDimensions(screen);
    state.pos = vec2f(screen_size / 2);
    state.zoom = 1;
    block_size = 100;
}

@compute @workgroup_size(16, 16)
fn fragment(@builtin(global_invocation_id) id: vec3u) {

    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);
    let texture_size = textureDimensions(channel1);
    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }


    block_size = abs(block_size);
    state.zoom = abs(state.zoom);

    //update view
    if mouse.click ==1 {
        state.pos += (vec2f(mouse.pos) / vec2f(screen_size) - vec2f(0.5) );
    }
    if keyDown(38) | keyDown(87) { // up arrow
        state.zoom += ZOOM_SPEED;
    }
    if keyDown(40) | keyDown(83) { // down arrow
        state.zoom -= ZOOM_SPEED;
    }

    //update block size
    if keyDown(39) | keyDown(68) { // right arrow
        block_size += BLOCK_SPEED;
    }
    if keyDown(37) | keyDown(65) { // left arrow
        block_size -= BLOCK_SPEED;
    }


    //navigating the image
    var uv = vec2f(id.xy);
    uv /= vec2f(texture_size) ;
    uv = uv  * state.zoom;
    uv = uv + state.pos/ vec2f(screen_size);

    //pixelate uv
    var uv_pix = uv * block_size;
    uv_pix = floor(uv_pix);
    uv_pix /= block_size;

    // texture
    var col = textureSampleLevel(channel1, bilinear, uv_pix, 0.).rgb;

    // convert to luminance [0 , 1]
    let l = rgb2Luminance(col);

    // maps calue to corresponding character
    let v = map_char(int(l * 24));
    
    //character uv
    let uv_char = uv * block_size;

    col = char_(uv_char, col, v).xyz;
    // col = char_(uv_char, vec3f(1), v).xyz;

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col,0.));
}
