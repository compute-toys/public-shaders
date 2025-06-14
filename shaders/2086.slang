import std;
import shadertoy;

// There are, however, some differences from Shadertoy:
// 1. All samples from iChannels that are mapped to Buffer A-D
//    must be replaced with ChBufferA-D
//    channel0/channel1 can be used normally, 
//    but note they are called channel0 not channel0
// 2. Globals must be static
// 3. Compute shaders don't support screen derivatives, so dFdx/dFdy/fwidth won't work
//    They must be emulated, either analytically, or doing multiple passes
//    This also means that texture(ch, p) always samples the 0th MIP level
//    Technically can be fully emulated using subgroup operations and clever mapping,
//    but not supported here at the moment.
// 4. There are no sampler2D or other GLSL texture types
//    Those must be either replaced by an int (for the buffer index)
//    Or replaced with Texture2D<float4> which is the type of the input channel0/1

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.xy;
    
    //CHANGED TO: iMouse.xy -> vec2(iMouse.x, iResolution.y-iMouse.y)
    vec2 mouse = vec2(iMouse.x, iResolution.y-iMouse.y);
    if (length(mouse)<1.0) {
        mouse = iResolution.xy/2.0;
    } vec2 m2 = (uv-mouse/iResolution.xy);
    
    float roundedBox = pow(abs(m2.x*iResolution.x/iResolution.y),8.0)+pow(abs(m2.y),8.0);    
    float rb1 = clamp((1.0-roundedBox*10000.0)*8.0, 0.,1.); // rounded box
    float rb2 = clamp((0.95-roundedBox*9500.0)*16.0, 0.,1.)-clamp(pow(0.9-roundedBox*9500.0, 1.0)*16.0, 0.,1.); // borders
    float rb3 = (clamp((1.5-roundedBox*11000.0)*2.0, 0.,1.)-clamp(pow(1.0-roundedBox*11000.0, 1.0)*2.0, 0.,1.)); // shadow gradient

    fragColor = vec4(0);
    
    float transition = smoothstep(0.0, 1.0, rb1+rb2);
    
    if (transition>0.0) {
    
        vec2 lens;
        
        // Sample 0
        lens = ((uv-0.5)*1.0*(1.0-roundedBox*5000.0)+0.5);
        
        // Sample 1
        // lens = ((uv-0.5)*1.0*(1.0-roundedBox*1000.0)+0.5);
        
        // Sample 2
        // lens = ((uv-0.5)*(roundedBox*1000.0+0.9)+0.5);
        
        // Sample 3
        // lens = ((uv-0.5)*0.9*(roundedBox*5000.0+0.9)+0.5);
       
        // Blur
        float total = 0.0;
        for (float x = -4.0; x <= 4.0; x++) {
            for (float y = -4.0; y <= 4.0; y++) {
                vec2 offset = vec2(x, y) * 0.5 / iResolution.xy;
                fragColor += texture(channel0, offset+lens);
                total += 1.0;
            }
        } fragColor/=total;
        
        // Lighting
        float gradient = clamp( (clamp(m2.y,0.0,0.2)+0.1)/2.0, 0.,1.) + clamp( (clamp(-m2.y,-1000.0,0.2)*rb3+0.1)/2.0, 0.,1.);
        vec4 lighting = clamp(fragColor+vec4(rb1)*gradient+vec4(rb2)*0.3, 0.,1.);
        
        // Antialiasing
        fragColor = mix(texture(channel0, uv), lighting, transition);
        
    } else { 
        fragColor = texture(channel0, uv);
    }
}

////////////////////////////////////////////////////////////
//Compute dispatches emulating fullscreen fragment shaders//
////////////////////////////////////////////////////////////

// 5. Unlike fragment shaders here you actually can control how the GPU spawns
//    the "pixel" threads.  Depending on the algorithm it might be preferrable 
//    to change the size of the workgroups

static const int WG_X = 16;
static const int WG_Y = 16;

#define ComputeFragColor(bfunc, id) \
    if(any(id.xy >= SCREEN_SIZE)) return; \
    vec4 fragColor; \
    bfunc(fragColor, vec2(id.xy) + 0.5);

#define BufferPass(bid, bfunc, id)  \
    ComputeFragColor(bfunc, id) \
    pass_out[int3(id.xy, bid)] = fragColor; 

[shader("compute")] [numthreads(WG_X, WG_Y, 1)]
void Image(uint3 id : SV_DispatchThreadID) {
    ComputeFragColor(mainImage, id)

    //CHANGED TO: [ivec2(id.x, SCREEN_SIZE.y - 1 - id.y)] => screen[id.xy]
    screen[ivec2(id.x, id.y)] = pow(fragColor, 2.2);
}