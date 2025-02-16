// Include the following library to enable string literals
#include <string>

#define TERMINAL_ROWS 10
#define TERMINAL_COLS 32

var<private> terminal_cursor: uint2;

#storage terminal_grid array<array<uint,TERMINAL_COLS>,TERMINAL_ROWS>

fn terminal_write_char(ascii: uint)
{
    if (ascii == 0) { // NULL
    } else if (ascii == 0x0a) { // '\n'
        terminal_cursor.x = 0u;
        terminal_cursor.y += 1u;
    } else {
        terminal_grid[terminal_cursor.y][terminal_cursor.x] = ascii;
        terminal_cursor.x += 1u;
    }
}

fn terminal_write(s: String) {
    for (var i = 0u; i < s.len; i++) {
        let ascii = s.chars[i];
        terminal_write_char(ascii);
    }
}

fn terminal_writei_auto(x_: int)
{   
    let x = float(x_);
    if(x<0.0)
    {
        terminal_write_char(0x2d); //minus sign
    }

    let d = max(int(ceil(log(abs(x))/log(10.0))), 1);
    var d0 = pow(10, float(d-1));
    for(var i = 0; i < d; i++)
    {
        let digit = uint(abs(x)/d0); 
        terminal_write_char(0x30 + (digit % 10));
        d0 /= 10.0;
    }
}

fn terminal_writei(x_: int, d: int)
{   
    let x = float(x_);
    var d0 = pow(10, float(d-1));
    for(var i = 0; i < d; i++)
    {
        let digit = uint(abs(x)/d0); 
        terminal_write_char(0x30 + (digit % 10));
        d0 /= 10.0;
    }
}

fn terminal_writef(x: float, fraction_digits: int)
{
    let m = pow(10.0, float(fraction_digits));
    terminal_writei_auto(int(x));
    terminal_write(".");
    terminal_writei(int(fract(abs(x)) * m), fraction_digits);
}

fn terminal_write2f(x: float2, fd: int)
{
    terminal_write("(");
    terminal_writef(x.x,fd);
    terminal_write(", ");
    terminal_writef(x.y,fd);
    terminal_write(")");
}

fn terminal_write3f(x: float3, fd: int)
{
    terminal_write("(");
    terminal_writef(x.x,fd);
    terminal_write(", ");
    terminal_writef(x.y,fd);
    terminal_write(", ");
    terminal_writef(x.z,fd);
    terminal_write(")");
}

fn terminal_write4f(x: float4, fd: int)
{
    terminal_write("(");
    terminal_writef(x.x,fd);
    terminal_write(", ");
    terminal_writef(x.y,fd);
    terminal_write(", ");
    terminal_writef(x.z,fd);
    terminal_write(", ");
    terminal_writef(x.w,fd);
    terminal_write(")");
}

fn terminal_clear() {
    for (var i = 0; i < TERMINAL_ROWS; i += 1) {
        for (var j = 0; j < TERMINAL_COLS; j += 1) {
            terminal_grid[i][j] = 0;
        }
    }
}

fn terminal_render(pos: uint2) -> float4 {
    let screen_size = uint2(textureDimensions(screen));
    let aspect = float(screen_size.y) / float(screen_size.x) * float(TERMINAL_COLS) / float(TERMINAL_ROWS);
    let texel = float(TERMINAL_ROWS) / float(screen_size.y);
    var uv = float2(pos) * float2(aspect, 1.) * texel;
    let ascii = terminal_grid[int(uv.y)][int(uv.x)];

    if (0x20 < ascii && ascii < 0x80) { // printable character
        uv = fract(uv);
        uv.x = (uv.x - .5) / aspect + .5; // aspect ratio correction
        uv += float2(uint2(ascii % 16u, ascii / 16u)); // character lookup
        let sdf = textureSampleLevel(channel1, trilinear, uv / 16., 0.).a;

        var col = float4(0);
        col = mix(col, float4(0,0,0,1), smoothstep(.525 + texel, .525 - texel, sdf));
        col = mix(col, float4(1,1,1,1), smoothstep(.490 + texel, .490 - texel, sdf));
        return col;
    }
    return float4(0);
}

// Precompute the text in a single thread and write it to a storage buffer,
// so that the rendering pass only needs two cheap lookups.
// String literals are desugared by the preprocessor into WGSL arrays,
// and wrapped in a String struct.
@compute @workgroup_size(1)
#workgroup_count singlethreaded 1 1 1
fn singlethreaded() {
    terminal_clear();
    terminal_write("Hello,\nworld!\n");
    terminal_write("time.frame = ");
    terminal_writei_auto(int(time.frame));
    terminal_write("\n");
    terminal_write("time.elapsed = ");
    terminal_writef(time.elapsed, 3);
    terminal_write("\n");
    terminal_write("mouse: ");
    terminal_write2f(float2(mouse.pos) / float2(textureDimensions(screen)), 3);
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    // Viewport resolution (in pixels)
    let screen_size = uint2(textureDimensions(screen));

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = float2(float(id.x) + .5, float(screen_size.y - id.y) - .5);

    // Normalised pixel coordinates (from 0 to 1)
    let uv = fragCoord / float2(screen_size);

    // Time varying pixel colour
    var col = .5 + .5 * cos(time.elapsed + uv.xyx + float3(0.,2.,4.));

    // Convert from gamma-encoded to linear colour space
    col = pow(col, float3(2.2));

    // Render terminal overlay
    let text = terminal_render(id.xy);
    col = mix(col, text.rgb, text.a);

    // Output to screen (linear colour space)
    textureStore(screen, int2(id.xy), float4(col, 1.));
}
