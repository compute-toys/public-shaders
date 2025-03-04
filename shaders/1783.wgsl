
fn char_(pos: float2, colour:vec3f, c: int) -> float4 {
    var p = pos % vec2f(1);

	let sdf = textureSampleLevel( channel0, trilinear, float2(p.x,1.-p.y)/16. + fract( float2(float(c), float(c/16)) / 16. ), 0.).a;
    let texel = float(0.01);
    var col = float4(0);
    col = mix(col, float4(colour,1), sdf);
    return col;
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {

    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);
    let texture_size = textureDimensions(channel1);
    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    //navigating the image
    var uv = vec2f(id.xy) * vec2f(custom.zoom) + (vec2f(screen_size.xy) * vec2f(custom.x,custom.y)) ;
    uv /= vec2f(texture_size.xy) ;
    

    //pixelate
    var uv_pix = uv * custom.block_size * vec2f(texture_size) /3000;
    uv_pix = floor(uv_pix);
    uv_pix /= custom.block_size;

    // texture
    var col = textureSampleLevel(channel1, bilinear, uv_pix, 0.).rgb;

    //convert to luminance
    var v = int((0.299*col.x + 0.587*col.y + 0.114*col.z) *16);

    //choose character per value
    switch v {
        case 0 {      
            v = 46; //.
        }
        case 1 { 
            v = 33; // !
        }
        case 2 {
            v = 61; // =
        }
        case 3 {  
            v = 43; // +
        }
        case 4 {  
            v = 7; // big dot
        }
        case 5 {
            v = 42; // *
        }
        case 6 {
            v = 144; //>>
        }
        case 7 {
            v = 19; // !!
        }
        case 8 {
            v = 1; // face
        }
        case 9 {
            v = 38; // line break
        }
        case 10 {
            v = 20; // &
        }
        case 11 {
            v = 64; // @
        }
        case 13 {
            v = 145; // 1/4
        }
        case 12 {
            v = 4; // carreau
        }
        case 14 {
            v = 5; //treffle
        }
        case 15 {
            v = 3; //heart
        }
        case 16 {
            v = 2; //happy face
        }
        default {    
            v = 10;
        }
    }
    
    //character uv
    let uv_char = uv * custom.block_size * vec2f(texture_size) /3000;

    col = char_(uv_char, col, v).xyz;
    // col = char_(uv_char, vec3f(1), v).xyz;

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col,0.));
}
