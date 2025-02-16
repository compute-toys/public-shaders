// Include the following library to enable string literals
#include <string>

// Simple fast, textureless text for debugging purposes


// main inspiration
// https://poniesandlight.co.uk/reflect/debug_print_text/

// Font
//http://www.fial.com/~scott/tamsyn-font/

//variables to text logic by @davidar
//https://compute.toys/view/59

// How to use:
//    0. copy a FONT const
//    1. copy functions write_...
//    2. each function takes in a value, a start position and the (scaled) fragCoord
//    3.it returns 0 if the fragment is not inside and 1 if it is
//    4. this value can be used to mix between the current color and the font color
// col = mix(col, WHITE, f32(write_line("like this",  vec2f(0.,100.), fragCoord)));

// original font
/* 
const FONT_TAMSYN = array<vec4<i32>, 96>(
    //vec4i( 0x00000000, 0x08080808, 0x08080800, 0x08080000 ), // !
    //vec4i(  0x00000000, 0x1212127E, 0x24247E48, 0x48480000  ),
    //vec4i(  0x00000042, 0x44485060, 0x50484442, 0x00000000   ),
    vec4i( 0x00000000, 0x00000000, 0x00000000, 0x00000000 ),
    vec4i( 0x00242424, 0x24000000, 0x00000000, 0x00000000 ),
    vec4i( 0x00001010, 0x10101010, 0x00001010, 0x00000000 ),
    vec4i( 0x00000024, 0x247E2424, 0x247E2424, 0x00000000 ),
    vec4i( 0x00000808, 0x1E20201C, 0x02023C08, 0x08000000 ),
    vec4i( 0x00000030, 0x494A3408, 0x16294906, 0x00000000 ),
    vec4i( 0x00003048, 0x48483031, 0x49464639, 0x00000000 ),
    vec4i( 0x00101010, 0x10000000, 0x00000000, 0x00000000 ),
    vec4i( 0x00000408, 0x08101010, 0x10101008, 0x08040000 ),
    vec4i( 0x00002010, 0x10080808, 0x08080810, 0x10200000 ),
    vec4i( 0x00000000, 0x0024187E, 0x18240000, 0x00000000 ),
    vec4i( 0x00000000, 0x0808087F, 0x08080800, 0x00000000 ),
    vec4i( 0x00000000, 0x00000000, 0x00001818, 0x08081000 ),
    vec4i( 0x00000000, 0x0000007E, 0x00000000, 0x00000000 ),
    vec4i( 0x00000000, 0x00000000, 0x00001818, 0x00000000 ),
    vec4i( 0x00000202, 0x04040808, 0x10102020, 0x40400000 ),
    vec4i( 0x0000003C, 0x42464A52, 0x6242423C, 0x00000000 ),
    vec4i( 0x00000008, 0x18280808, 0x0808083E, 0x00000000 ),
    vec4i( 0x0000003C, 0x42020204, 0x0810207E, 0x00000000 ),
    vec4i( 0x0000007E, 0x04081C02, 0x0202423C, 0x00000000 ),
    vec4i( 0x00000004, 0x0C142444, 0x7E040404, 0x00000000 ),
    vec4i( 0x0000007E, 0x40407C02, 0x0202423C, 0x00000000 ),
    vec4i( 0x0000001C, 0x2040407C, 0x4242423C, 0x00000000 ),
    vec4i( 0x0000007E, 0x02040408, 0x08101010, 0x00000000 ),
    vec4i( 0x0000003C, 0x4242423C, 0x4242423C, 0x00000000 ),
    vec4i( 0x0000003C, 0x4242423E, 0x02020438, 0x00000000 ),
    vec4i( 0x00000000, 0x00181800, 0x00001818, 0x00000000 ),
    vec4i( 0x00000000, 0x00181800, 0x00001818, 0x08081000 ),
    vec4i( 0x00000004, 0x08102040, 0x20100804, 0x00000000 ),
    vec4i( 0x00000000, 0x00007E00, 0x007E0000, 0x00000000 ),
    vec4i( 0x00000020, 0x10080402, 0x04081020, 0x00000000 ),
    vec4i( 0x00003C42, 0x02040810, 0x00001010, 0x00000000 ),
    vec4i( 0x00001C22, 0x414F5151, 0x51534D40, 0x201F0000 ),
    vec4i( 0x00000018, 0x24424242, 0x7E424242, 0x00000000 ),
    vec4i( 0x0000007C, 0x4242427C, 0x4242427C, 0x00000000 ),
    vec4i( 0x0000001E, 0x20404040, 0x4040201E, 0x00000000 ),
    vec4i( 0x00000078, 0x44424242, 0x42424478, 0x00000000 ),
    vec4i( 0x0000007E, 0x4040407C, 0x4040407E, 0x00000000 ),
    vec4i( 0x0000007E, 0x4040407C, 0x40404040, 0x00000000 ),
    vec4i( 0x0000001E, 0x20404046, 0x4242221E, 0x00000000 ),
    vec4i( 0x00000042, 0x4242427E, 0x42424242, 0x00000000 ),
    vec4i( 0x0000003E, 0x08080808, 0x0808083E, 0x00000000 ),
    vec4i( 0x00000002, 0x02020202, 0x0242423C, 0x00000000 ),
    vec4i( 0x00000042, 0x44485060, 0x50484442, 0x00000000 ),
    vec4i( 0x00000040, 0x40404040, 0x4040407E, 0x00000000 ),
    vec4i( 0x00000041, 0x63554949, 0x41414141, 0x00000000 ),
    vec4i( 0x00000042, 0x62524A46, 0x42424242, 0x00000000 ),
    vec4i( 0x0000003C, 0x42424242, 0x4242423C, 0x00000000 ),
    vec4i( 0x0000007C, 0x4242427C, 0x40404040, 0x00000000 ),
    vec4i( 0x0000003C, 0x42424242, 0x4242423C, 0x04020000 ),
    vec4i( 0x0000007C, 0x4242427C, 0x48444242, 0x00000000 ),
    vec4i( 0x0000003E, 0x40402018, 0x0402027C, 0x00000000 ),
    vec4i( 0x0000007F, 0x08080808, 0x08080808, 0x00000000 ),
    vec4i( 0x00000042, 0x42424242, 0x4242423C, 0x00000000 ),
    vec4i( 0x00000042, 0x42424242, 0x24241818, 0x00000000 ),
    vec4i( 0x00000041, 0x41414149, 0x49495563, 0x00000000 ),
    vec4i( 0x00000041, 0x41221408, 0x14224141, 0x00000000 ),
    vec4i( 0x00000041, 0x41221408, 0x08080808, 0x00000000 ),
    vec4i( 0x0000007E, 0x04080810, 0x1020207E, 0x00000000 ),
    vec4i( 0x00001E10, 0x10101010, 0x10101010, 0x101E0000 ),
    vec4i( 0x00004040, 0x20201010, 0x08080404, 0x02020000 ),
    vec4i( 0x00007808, 0x08080808, 0x08080808, 0x08780000 ),
    vec4i( 0x00001028, 0x44000000, 0x00000000, 0x00000000 ),
    vec4i( 0x00000000, 0x00000000, 0x00000000, 0x00FF0000 ),
    vec4i( 0x00201008, 0x04000000, 0x00000000, 0x00000000 ),
    vec4i( 0x00000000, 0x003C0202, 0x3E42423E, 0x00000000 ),
    vec4i( 0x00004040, 0x407C4242, 0x4242427C, 0x00000000 ),
    vec4i( 0x00000000, 0x003C4240, 0x4040423C, 0x00000000 ),
    vec4i( 0x00000202, 0x023E4242, 0x4242423E, 0x00000000 ),
    vec4i( 0x00000000, 0x003C4242, 0x7E40403E, 0x00000000 ),
    vec4i( 0x00000E10, 0x107E1010, 0x10101010, 0x00000000 ),
    vec4i( 0x00000000, 0x003E4242, 0x4242423E, 0x02023C00 ),
    vec4i( 0x00004040, 0x407C4242, 0x42424242, 0x00000000 ),
    vec4i( 0x00000808, 0x00380808, 0x0808083E, 0x00000000 ),
    vec4i( 0x00000404, 0x001C0404, 0x04040404, 0x04043800 ),
    vec4i( 0x00004040, 0x40444850, 0x70484442, 0x00000000 ),
    vec4i( 0x00003808, 0x08080808, 0x0808083E, 0x00000000 ),
    vec4i( 0x00000000, 0x00774949, 0x49494949, 0x00000000 ),
    vec4i( 0x00000000, 0x007C4242, 0x42424242, 0x00000000 ),
    vec4i( 0x00000000, 0x003C4242, 0x4242423C, 0x00000000 ),
    vec4i( 0x00000000, 0x007C4242, 0x4242427C, 0x40404000 ),
    vec4i( 0x00000000, 0x003E4242, 0x4242423E, 0x02020200 ),
    vec4i( 0x00000000, 0x002E3020, 0x20202020, 0x00000000 ),
    vec4i( 0x00000000, 0x003E4020, 0x1804027C, 0x00000000 ),
    vec4i( 0x00000010, 0x107E1010, 0x1010100E, 0x00000000 ),
    vec4i( 0x00000000, 0x00424242, 0x4242423E, 0x00000000 ),
    vec4i( 0x00000000, 0x00424242, 0x24241818, 0x00000000 ),
    vec4i( 0x00000000, 0x00414141, 0x49495563, 0x00000000 ),
    vec4i( 0x00000000, 0x00412214, 0x08142241, 0x00000000 ),
    vec4i( 0x00000000, 0x00424242, 0x4242423E, 0x02023C00 ),
    vec4i( 0x00000000, 0x007E0408, 0x1020407E, 0x00000000 ),
    vec4i( 0x000E1010, 0x101010E0, 0x10101010, 0x100E0000 ),
    vec4i( 0x00080808, 0x08080808, 0x08080808, 0x08080000 ),
    vec4i( 0x00700808, 0x08080807, 0x08080808, 0x08700000 ),
    vec4i( 0x00003149, 0x46000000, 0x00000000, 0x00000000 ),
    vec4i( 0x00000000, 0x00000000, 0x00000000, 0x00000000 ),
); */

const MY_FONT = array<vec4<i32>, 97>(
    vec4i( 0x00000000, 0x00000000, 0x00000000, 0x00000000 ), //Space
    vec4i( 0x00001010, 0x10101010, 0x00001010, 0x00000000 ), //!
    vec4i( 0x00242424, 0x24000000, 0x00000000, 0x00000000 ), //"
    vec4i( 0x00000024, 0x247E2424, 0x247E2424, 0x00000000 ), //#
    vec4i( 0x00000808, 0x1E20201C, 0x02023C08, 0x08000000 ), // $
    vec4i( 0x00000030, 0x494A3408, 0x16294906, 0x00000000 ), // %
    vec4i( 0x00003C40, 0x40403F42, 0x4242423C, 0x00000000 ), // & (different)
    vec4i( 0x00101010, 0x10000000, 0x00000000, 0x00000000 ), // ...
    vec4i( 0x00000408, 0x08101010, 0x10101008, 0x08040000 ),
    vec4i( 0x00002010, 0x10080808, 0x08080810, 0x10200000 ),
    vec4i( 0x00000000, 0x0024187E, 0x18240000, 0x00000000 ),
    vec4i( 0x00000000, 0x0808087F, 0x08080800, 0x00000000 ),
    vec4i( 0x00000000, 0x00000000, 0x00001818, 0x08081000 ),
    vec4i( 0x00000000, 0x0000007E, 0x00000000, 0x00000000 ),
    vec4i( 0x00000000, 0x00000000, 0x00001818, 0x00000000 ),
    vec4i( 0x00000202, 0x04040808, 0x10102020, 0x40400000 ),
    vec4i( 0x0000003C, 0x42464A52, 0x6242423C, 0x00000000 ),
    vec4i( 0x00000008, 0x18280808, 0x0808083E, 0x00000000 ),
    vec4i( 0x0000003C, 0x42020204, 0x0810207E, 0x00000000 ),
    vec4i( 0x0000007E, 0x04081C02, 0x0202423C, 0x00000000 ),
    vec4i( 0x00000004, 0x0C142444, 0x7E040404, 0x00000000 ),
    vec4i( 0x0000007E, 0x40407C02, 0x0202423C, 0x00000000 ),
    vec4i( 0x0000001C, 0x2040407C, 0x4242423C, 0x00000000 ),
    vec4i( 0x0000007E, 0x02040408, 0x08101010, 0x00000000 ),
    vec4i( 0x0000003C, 0x4242423C, 0x4242423C, 0x00000000 ),
    vec4i( 0x0000003C, 0x4242423E, 0x02020438, 0x00000000 ),
    vec4i( 0x00000000, 0x00181800, 0x00001818, 0x00000000 ),
    vec4i( 0x00000000, 0x00181800, 0x00001818, 0x08081000 ),
    vec4i( 0x00000004, 0x08102040, 0x20100804, 0x00000000 ),
    vec4i( 0x00000000, 0x00007E00, 0x007E0000, 0x00000000 ),
    vec4i( 0x00000020, 0x10080402, 0x04081020, 0x00000000 ),
    vec4i( 0x00003C42, 0x02040810, 0x00001010, 0x00000000 ),
    vec4i( 0x00001C22, 0x414F5151, 0x51534D40, 0x201F0000 ),
    vec4i( 0x00000018, 0x24424242, 0x7E424242, 0x00000000 ),
    vec4i( 0x0000007C, 0x4242427C, 0x4242427C, 0x00000000 ),
    vec4i( 0x0000001E, 0x20404040, 0x4040201E, 0x00000000 ),
    vec4i( 0x00000078, 0x44424242, 0x42424478, 0x00000000 ),
    vec4i( 0x0000007E, 0x4040407C, 0x4040407E, 0x00000000 ),
    vec4i( 0x0000007E, 0x4040407C, 0x40404040, 0x00000000 ),
    vec4i( 0x0000001E, 0x20404046, 0x4242221E, 0x00000000 ),
    vec4i( 0x00000042, 0x4242427E, 0x42424242, 0x00000000 ),
    vec4i( 0x0000003E, 0x08080808, 0x0808083E, 0x00000000 ),
    vec4i( 0x00000002, 0x02020202, 0x0242423C, 0x00000000 ),
    vec4i( 0x00000042, 0x44485060, 0x50484442, 0x00000000 ),
    vec4i( 0x00000040, 0x40404040, 0x4040407E, 0x00000000 ),
    vec4i( 0x00000041, 0x63554949, 0x41414141, 0x00000000 ),
    vec4i( 0x00000042, 0x62524A46, 0x42424242, 0x00000000 ),
    vec4i( 0x0000003C, 0x42424242, 0x4242423C, 0x00000000 ),
    vec4i( 0x0000007C, 0x4242427C, 0x40404040, 0x00000000 ),
    vec4i( 0x0000003C, 0x42424242, 0x4242423C, 0x04020000 ),
    vec4i( 0x0000007C, 0x4242427C, 0x48444242, 0x00000000 ),
    vec4i( 0x0000003E, 0x40402018, 0x0402027C, 0x00000000 ),
    vec4i( 0x0000007F, 0x08080808, 0x08080808, 0x00000000 ),
    vec4i( 0x00000042, 0x42424242, 0x4242423C, 0x00000000 ),
    vec4i( 0x00000042, 0x42424242, 0x24241818, 0x00000000 ),
    vec4i( 0x00000041, 0x41414149, 0x49495563, 0x00000000 ),
    vec4i( 0x00000041, 0x41221408, 0x14224141, 0x00000000 ),
    vec4i( 0x00000041, 0x41221408, 0x08080808, 0x00000000 ),
    vec4i( 0x0000007E, 0x04080810, 0x1020207E, 0x00000000 ),
    vec4i( 0x00001E10, 0x10101010, 0x10101010, 0x101E0000 ),
    vec4i( 0x00004040, 0x20201010, 0x08080404, 0x02020000 ),
    vec4i( 0x00007808, 0x08080808, 0x08080808, 0x08780000 ),
    vec4i( 0x00001028, 0x44000000, 0x00000000, 0x00000000 ),
    vec4i( 0x00000000, 0x00000000, 0x00000000, 0x00FF0000 ),
    vec4i( 0x00201008, 0x04000000, 0x00000000, 0x00000000 ),
    vec4i( 0x00000000, 0x003C0202, 0x3E42423E, 0x00000000 ),
    vec4i( 0x00004040, 0x407C4242, 0x4242427C, 0x00000000 ),
    vec4i( 0x00000000, 0x003C4240, 0x4040423C, 0x00000000 ),
    vec4i( 0x00000202, 0x023E4242, 0x4242423E, 0x00000000 ),
    vec4i( 0x00000000, 0x003C4242, 0x7E40403E, 0x00000000 ),
    vec4i( 0x00000E10, 0x107E1010, 0x10101010, 0x00000000 ),
    vec4i( 0x00000000, 0x003E4242, 0x4242423E, 0x02023C00 ),
    vec4i( 0x00004040, 0x407C4242, 0x42424242, 0x00000000 ),
    vec4i( 0x00000808, 0x00380808, 0x0808083E, 0x00000000 ),
    vec4i( 0x00000404, 0x001C0404, 0x04040404, 0x04043800 ),
    vec4i( 0x00004040, 0x40444850, 0x70484442, 0x00000000 ),
    vec4i( 0x00003808, 0x08080808, 0x0808083E, 0x00000000 ),
    vec4i( 0x00000000, 0x00774949, 0x49494949, 0x00000000 ),
    vec4i( 0x00000000, 0x007C4242, 0x42424242, 0x00000000 ),
    vec4i( 0x00000000, 0x003C4242, 0x4242423C, 0x00000000 ),
    vec4i( 0x00000000, 0x007C4242, 0x4242427C, 0x40404000 ),
    vec4i( 0x00000000, 0x003E4242, 0x4242423E, 0x02020200 ),
    vec4i( 0x00000000, 0x002E3020, 0x20202020, 0x00000000 ),
    vec4i( 0x00000000, 0x003E4020, 0x1804027C, 0x00000000 ),
    vec4i( 0x00000010, 0x107E1010, 0x1010100E, 0x00000000 ),
    vec4i( 0x00000000, 0x00424242, 0x4242423E, 0x00000000 ),
    vec4i( 0x00000000, 0x00424242, 0x24241818, 0x00000000 ),
    vec4i( 0x00000000, 0x00414141, 0x49495563, 0x00000000 ),
    vec4i( 0x00000000, 0x00412214, 0x08142241, 0x00000000 ),
    vec4i( 0x00000000, 0x00424242, 0x4242423E, 0x02023C00 ),
    vec4i( 0x00000000, 0x007E0408, 0x1020407E, 0x00000000 ),
    vec4i( 0x000E1010, 0x101010E0, 0x10101010, 0x100E0000 ),
    vec4i( 0x00080808, 0x08080808, 0x08080808, 0x08080000 ),
    vec4i( 0x00700808, 0x08080807, 0x08080808, 0x08700000 ),
    vec4i( 0x00003149, 0x46000000, 0x00000000, 0x00000000 ),
    vec4i( 0x7E42425A, 0x4A4A4A4A, 0x5A524252, 0x5242427E ), // added .notdef character
    vec4i( 0x00000000, 0x00000000, 0x00000000, 0x00000000 ),
);

const BLACK = vec3f(0.);
const WHITE = vec3f(1.);

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);
    let uv = fragCoord / vec2f(screen_size);
    var col = .5 + .5 * cos(time.elapsed + uv.xyx + vec3f(0.,2.,4.));

    let lf = fragCoord / 2;
    col = mix(col, WHITE, f32(write_line_jumpy("Hello @compute.toys",  vec2f(170.,150.), lf)));
    col = mix(col, WHITE, f32(write_line("?!\"#$%&{}~⌂á",  vec2f(100.,30.), lf)));
    col = mix(col, WHITE, f32(writef(time.elapsed, 6,  vec2f(20.,200.), lf)));
    col = mix(col, WHITE, f32(writei_auto(i32(time.frame),  vec2f(20.,180.), lf)));
    col = mix(col, WHITE, f32(write_i(mouse.click, 3,  vec2f(20., 160.), lf)));
    
    col = mix(col, BLACK, f32(write_i(i32(mouse.pos.x), 3,  vec2f(f32(mouse.pos.x), f32(screen_size.y)-f32(mouse.pos.y)) , fragCoord)));
    col = mix(col, BLACK, f32(write_i(i32(mouse.pos.y), 3,  vec2f(f32(mouse.pos.x), f32(screen_size.y)-f32(mouse.pos.y) - 18.) , fragCoord)));
    
    col = pow(col, vec3f(2.2));
    textureStore(screen, id.xy, vec4f(col, 1.));
}

fn write_line(s: String, pos: vec2f, fc: vec2f) -> i32{
    var out = 0;

    for (var i = 0u; i < s.len; i++) {
        let j = f32(i);
        var ch = s.chars[i] - 32;
        //print .notdef if ascii code is unknown
        if ch < 0 || ch > 94 {
            ch = 95;
        }
        out |= i32(write_char( ch , pos + vec2f(j*8., 0.), fc));
    }
    return out;
}

fn write_line_jumpy(s: String, pos: vec2f, fc: vec2f) -> i32{
    var out = 0;

    for (var i = 0u; i < s.len; i++) {
        let j = f32(i);
        var ch = s.chars[i] - 32;
        if ch < 0 || ch > 94 {
            ch = 95;
        }
        out |= i32(write_char( ch , pos + vec2f(j*8., 5. * sin(time.elapsed * 10. + f32(i))), fc));
    }
    return out;
}

fn write_char(i: u32, pos: vec2f, fc1: vec2f) -> i32 {
    var col = 0;
    var fc = fc1 - pos;
    if fc.x < 8 && fc.y < 16 && fc.x > 0. && fc.y > 0{
        //fc.x = 16. - fc.x;
        let accessor =  3 - i32(floor(fc.y / 4.));
        //there is probably an easier way to do this but it works
        let b = extractBits(MY_FONT[i][accessor], u32((i32(fc.y) - (3-accessor) * 4) * 8) + 7 - u32(fc.x ) , 1);
        if b != 0 {
            col = 1;
        }
    }
    return col;
}

fn write_i(x_: int, d: int, pos: vec2f, fc: vec2f) -> i32 {   
    var col = 0;
    let x = float(x_);
    var d0 = pow(10, float(d-1));
    for(var i = 0; i < d; i++)
    {
        let digit = uint(abs(x)/d0); 
        col |= write_char(0x30 + (digit % 10) - 32, pos + vec2f(f32(i)*8.,0.), fc);
        
        d0 /= 10.0;
    }
    return col;
}

fn writei_auto(x_: int, pos: vec2f, fc: vec2f) -> i32{
    var col = 0;   
    let x = float(x_);
    if(x<0.0)
    {
        col |= i32(write_char(0x2d-32, pos, fc)); //minus sign
    }

    let d = max(int(ceil(log(abs(x))/log(10.0))), 1)+1;
    var d0 = pow(10, float(d-1));
    for(var i = 0; i < d; i++)
    {
        let digit = uint(abs(x)/d0); 
        col |= i32(write_char(0x30 + (digit % 10) -32, pos + vec2f(f32(i)*8., 0.), fc));
        d0 /= 10.0;
    }
    return col;
}


fn writef(x: float, fraction_digits: int, pos: vec2f, fc: vec2f) -> i32 {
    let m = pow(10.0, float(fraction_digits));
    var out = 0;
    out |= write_i(int(x), 6, pos, fc);
    out |= write_line(".",pos + vec2f(48.,0.), fc);
    out |= write_i(int(fract(abs(x)) * m), fraction_digits, pos + vec2f(54.,0.), fc);
    return out;
}


// The code I didn't steal is released under the MIT License

/*
Copyright 2024 @quilde

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

*/