import std;

float smoothMin(float a, float b, float k){
    float h = saturate(0.5+0.5 * (b-a)/k);
    return lerp(b,a,h) - k*h * (1.0 - h);
}

float map(float3 p){
    float3 c1 = float3(-1.0 + sin(time.elapsed), 0.0, 0.0);
    float3 c2 = float3( 1.0 - sin(time.elapsed), 0.0, 0.0);

    float d1 = length(p - c1) - 1.0;
    float d2 = length(p- c2) - 1.0;

    return smoothMin(d1, d2, 0.5);
}

float3 calcNormal(float3 p){
    float e = 0.001;
    float dx = map(p + float3(e,0,0)) - map(p - float3(e,0,0));
    float dy = map(p + float3(0,e,0)) - map(p - float3(0,e,0));
    float dz = map(p + float3(0,0,e)) - map(p - float3(0,0,e));
    return normalize(float3(dx,dy,dz));
}

[shader("compute")]
[numthreads(16,16,1)]
void main_image(uint3 id : SV_DispatchThreadID)
{
    uint width,height;
    screen.GetDimensions(width,height);
    if(id.x>=width || id.y>=height) return;
    float2 uv=(float2(id.xy)-0.5*float2(width,height))/height;
   
    float3 ro=float3(0,0,-4);  // Bare minimum camera XD
    float3 rd=normalize(float3(uv,1));

    // Raymarching
    float t=0.0;
    bool hit=false;
    for(int i=0; i<128; i++){
        float3 p = ro + rd*t;
        float d = map(p);
        if(d < 0.001){
            hit=true;
            break;
        }
        t += d;
        if(t > 50.0) break;
    }

    float3 col = float3(0.6,0.8,1.0);
    if(hit){
        float3 p = ro+rd*t;
        float3 N = calcNormal(p);
        float3 L = normalize(float3(1,-2,-1));
        float diff = max(dot(N,L),0.0);

        col = diff*float3(0.2,0.7,1.0);
    }

    screen[id.xy] = float4(col,1);
}