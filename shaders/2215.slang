import std;
import shadertoy;

#define PI 3.14159
#define MAXDEPTH 7

mat3 Rot (float angle)
{
    float c = cos(angle);
    float s = sin(angle);
    
    return  mat3(
            vec3(c, s, 0),
            vec3(-s, c, 0),
            vec3(0, 0, 1)
    ); 
}

mat3 Disp (vec2 displacement)
{
    return  mat3(
            vec3(1, 0, 0),
            vec3(0, 1, 0),
            vec3(displacement, 1)
    ); 
}

float sdCappedCylinder( vec2 p, vec2 h )
{
    p -= vec2(0.,h.y);
    vec2 d = abs(vec2(length(p.x),p.y)) - h;
    return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float center(vec2 pt)
{
    mat3 posR = Rot(-(custom.Angle*60./360.)*2.*PI);
    mat3 negR = Rot(custom.Angle*60./360.*2.*PI);
    
    const int depth = 7;
    const int branches = 3; 
    int maxDepth = int(pow(float(branches) , float(depth )));
    float len = 1.7*mouse.zoom;
    float wid = .01;
    pt.xy += ((-mouse.pos+mouse.start) / iResolution.xy) * vec2(1.0,-1.0) * vec2(16.0,9.0);
    pt = pt + vec2(0,2);  
    mat3 m1 = posR*Disp(vec2(0,-len));
    
    float trunk = sdCappedCylinder(pt-vec2(0.,0.), vec2(wid,len));
    float d = 500.;
    
    int c = 0; // Running count for optimizations
    for (int count = 0; count <= 100; ++count){
        int off = int(pow(float(branches), float(depth)));
        vec2 pt_n = pt;
        for (int i = 1; i <= depth; ++i)
        {
            float l = len/pow(2.,float(i));
            
            off /= branches; 
            int dec = c / off;
            int path = dec - branches*(dec/branches); //  dec % kBranches
            
            mat3 mx;
            if(path == 0){
                mx = posR*Disp(vec2(0,-2.*l));
            }
            else if(path == 1){
                mat3 wind = Rot(0.2*sin(iTime+6.2));
                mx = wind*negR*Disp(vec2(0,-2.*l));
            }
            else if(path == 2){
                mat3 wind = Rot(0.2*sin(iTime+1.));
                mx = wind*Disp(vec2(0,-4.*l)) ;
            }
            
            pt_n = (mx * vec3(pt_n,1)).xy;
            float y = sdCappedCylinder(pt_n, vec2(wid,l));   
            
            // Early bail out. Bounding volume is a noodle of radius
            // 2. * l around line segment.
            if( y-2.0*l > 0.0 ) { c += off-1; break; }
                d = min( d, y );
        }
        
        ++c;
        if (c > maxDepth) break;
    }
    return min(d,trunk); 
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord.xy / iResolution.xy;
    uv = uv * 2.0 - 1.0;		
    uv.x *= iResolution.x / iResolution.y;
    uv *= 5.;
    
    fragColor = vec4(0);
    float c = center(uv);
    
    float t = clamp(c, 0.0, .04) * 2.*12.5;
    vec4 bg = vec4(0);
    vec4 fg = vec4(.8);
    fragColor = mix(bg, fg, 1.-t);  
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main_image(uint3 id : SV_DispatchThreadID)
{
    uint width, height;
    screen.GetDimensions(width, height);

    if (id.x >= width || id.y >= height)
        return;

    float2 fragCoord = float2(float(id.x) + 0.5, float(height - id.y) - 0.5);
    float2 uv = fragCoord / float2(width, height);

    // screen[id.xy] = float4(uv, 0.0, 1.0);

    float4 color;
    mainImage(color, fragCoord);

    screen[id.xy] = color;
}
