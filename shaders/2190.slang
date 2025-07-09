import std;
import shadertoy;

//remove after: https://github.com/compute-toys/compute.toys/pull/352
public static vec4 iDate = vec4(2022., 3., 28., iTime + 31. + 60.*56. + 60.*60.*14.);

// by Nikos Papadopoulos, 4rknova / 2014
// Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

#define DISCREET_SECONDS
#define AA	4.

#define PI  3.14159265359
#define EPS .01

float df_disk(in vec2 p, in vec2 c, in float r)
{
    return clamp(length(p - c) - r, 0., 1.);
}

float df_circ(in vec2 p, in vec2 c, in float r)
{
    return abs(r - length(p - c));
}

float df_line(in vec2 p, in vec2 a, in vec2 b)
{
    vec2 pa = p - a, ba = b - a;
	float h = clamp(dot(pa,ba) / dot(ba,ba), 0., 1.);	
	return length(pa - ba * h);
}

float sharpen(in float d, in float w)
{
    float e = 1. / min(iResolution.y , iResolution.x);
    return 1. - smoothstep(-e, e, d - w);
}

vec2 rotate(in vec2 p, in float t)
{
    t = t * 2. * PI;
    return vec2(p.x * cos(t) - p.y * sin(t),
                p.y * cos(t) + p.x * sin(t));
}

float df_scene(vec2 uv)
{    
	float thrs = iDate.w / 3600.;
	float tmin = mod(iDate.w, 3600.) / 60.;
    float tsec = mod(mod(iDate.w, 3600.), 60.);
    
    #ifdef DISCREET_SECONDS
    	tsec = floor(tsec);
    #endif
    
    vec2 c = vec2(0), u = vec2(0,1);
    float c1 = sharpen(df_circ(uv, c, .90), EPS * 1.5);
    float c2 = sharpen(df_circ(uv, c, .04), EPS * 0.5);
    float d1 = sharpen(df_disk(uv, c, .01), EPS * 1.5);
    float l1 = sharpen(df_line(uv, c, rotate(u,-thrs / 12.) * .60), EPS * 1.7);
    float l2 = sharpen(df_line(uv, c, rotate(u,-tmin / 60.) * .80), EPS * 1.0);
    float l3 = sharpen(df_line(uv, c, rotate(u,-tsec / 60.) * .85), EPS * 0.5);
    return max(max(max(max(max(l1, l2), l3), c1), c2), d1);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
	vec2 uv = (fragCoord.xy / iResolution.xy * 2. - 1.);
    uv.x *= iResolution.x / iResolution.y;
    vec3 col = vec3(0);
    
#ifdef AA
    // Antialiasing via supersampling
    float e = 1. / min(iResolution.y , iResolution.x);    
    for (float i = -AA; i < AA; ++i) {
        for (float j = -AA; j < AA; ++j) {
    		col += df_scene(uv + vec2(i, j) * (e/AA)) / (4.*AA*AA);
        }
    }
#else
    col += df_scene(uv);
#endif /* AA */
    
	fragColor = vec4(col, 1);
}

// Workgroup size 
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
    screen[ivec2(id.x, SCREEN_SIZE.y - 1 - id.y)] = pow(fragColor, 2.2);
}