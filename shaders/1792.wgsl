////////////////////////////////////////////////////
///// SHADERTOY TO COMPUTE.TOYS TEMPLATE v0.1 //////
////////////////////////////////////////////////////

// This template assumes that channels are in the following order both for inputs and outputs of the buffers:
#define iChannel0 0
#define iChannel1 1
#define iChannel2 2
#define iChannel3 3

// The template is not perfect, as you can't the channels used, 
// but it should be a good starting point for porting shaders from shadertoy to compute.toys

#define iResolution vec2f(textureDimensions(screen).xy)
#define iTime (time.elapsed)
#define iTimeDelta (time.delta)
#define iFrameRate (time.frame)
#define iFrame (time.frame)
#define iMouse vec4f(vec2f(mouse.pos), vec2f(mouse.click))

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

fn BufferA(fragCoord: vec2f) -> vec4f {
    // Standard Shadertoy example code
    var fragColor = vec4f(0.0);

    // Normalized pixel coordinates (from 0 to 1)
    let uv = fragCoord/iResolution.xy;

    // Time varying pixel color
    let col = 0.5 + 0.5*cos(iTime+uv.xyx+vec3(0,2,4));

    // Output to screen
    fragColor = vec4(col,1.0);

    return fragColor;
}

fn BufferB(fragCoord: vec2f) -> vec4f {
    // Example sampling code from Buffer A
    let col = texture(iChannel0, fragCoord / iResolution.xy);

    return col;
}

fn BufferC(fragCoord: vec2f) -> vec4f {
    // Texel fetch example from Buffer B
    let col = texelFetch(iChannel1, vec2i(fragCoord), 0);

    return col;
}

fn BufferD(fragCoord: vec2f) -> vec4f {
    // Texture lod example from Buffer C
    let col = textureLod(iChannel2, fragCoord / iResolution.xy, 0);

    return col;
}

fn Image(fragCoord: vec2f) -> vec4f {
    // Just sample from Buffer D
    let col = texture(iChannel3, fragCoord / iResolution.xy);

    return col;
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