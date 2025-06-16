import std;
import shadertoy;

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.xy;
    
    //CHANGED TO: iMouse.xy -> vec2(iMouse.x, iResolution.y-iMouse.y)
    vec2 mouse = vec2(iMouse.x, iResolution.y-iMouse.y);
    if (length(mouse)<1.0) {
        mouse = iResolution.xy/2.0;
    } vec2 m2 = (uv-mouse/iResolution.xy);
    
    float roundedBox = pow(abs(m2.x*iResolution.x/iResolution.y),8.0)+pow(abs(m2.y),8.0);    
    float rb1 = clamp((1.0-roundedBox*10000.0)*8.0, 0.,1.);
    float rb2 = clamp((0.95-roundedBox*9500.0)*16.0, 0.,1.)-clamp(pow(0.9-roundedBox*9500.0, 1.0)*16.0, 0.,1.); // borders
    float rb3 = (clamp((1.5-roundedBox*11000.0)*2.0, 0.,1.)-clamp(pow(1.0-roundedBox*11000.0, 1.0)*2.0, 0.,1.)); // shadow gradient

    fragColor = vec4(0);
    
    float transition = smoothstep(0.0, 1.0, rb1+rb2);
    
    if (transition>0.0) {
    
        vec2 lens;

    #define TYPE 0

    #if (TYPE == 0)
        lens = ((uv-0.5)*1.0*(1.0-roundedBox*5000.0)+0.5);
    #elif (TYPE == 1)
        lens = ((uv-0.5)*1.0*(1.0-roundedBox*1000.0)+0.5);
    #elif (TYPE == 2)
        lens = ((uv-0.5)*(roundedBox*1000.0+0.9)+0.5);
    #elif (TYPE == 3)
        lens = ((uv-0.5)*0.9*(roundedBox*5000.0+0.9)+0.5);
    #endif
       
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