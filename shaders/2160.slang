import std;

float iPlane(in float3 ro, in float3 rd) {
  return -ro.y/rd.y;
}

float3 nPlane(in float3 pos) {
  return float3(0.0f, 1.0f, 0.0f);
}

float iSphere(in float3 ro, in float3 rd, in float4 sph) {
  float r = sph.w;
  float3 oc = ro - sph.xyz;
  float b = 2.0f * dot(oc, rd);
  float c = dot(oc, oc) - r*r;
  float h = b*b - 4.0*c;
  if(h < 0.0) return -1;
  float t = (-b - sqrt(h)) / 2.0f;
  return t;
}

float3 nSphere(in float3 pos, in float4 sph) {
  return (pos - sph.xyz)/sph.w;
}

static float4 sph1 = float4(0.0f, 1.0f, 0.0f, 1.0f);
float intersect(in float3 ro, in float3 rd, out float resT) {
  resT = 1000.0f;
  float id = -1.0f;
  float tsph = iSphere(ro, rd, sph1);
  float tpla = iPlane(ro, rd);
  if (tsph>0.0f) {
    id = 1.0f;
    resT = tsph;
  }
  if (tpla > 0.0f && tpla < resT){
    id = 2.0f;
    resT = tpla;
  }
  return id;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main_image(uint3 id : SV_DispatchThreadID)
{
  uint width, height;
  screen.GetDimensions(width, height);
  
  if (id.x >= width || id.y >= height) {
    return;
  }

  float3 light = normalize(float3(0.57f, 0.57f, 0.57f));
  sph1.x = 0.5f * cos(time.elapsed);
  sph1.z = 0.5f * sin(time.elapsed);

  float2 fragCoord = float2(float(id.x) + 0.5, float(height - id.y) - 0.5);

  float2 uv = fragCoord / float2(width, height);
  float2 uv_fix = float2((float)width/(float)height, 1.0f);

  float3 ro = float3(0.0f, 0.5f, 3.0f);
  float3 rd = normalize(float3((-1.0f + 2.0f * uv) * uv_fix, -1.0f));

  float3 color = float3(0.0f, 0.0f, 0.0f);

  float t;
  float obj_id = intersect(ro, rd, t);
  if (obj_id > 0.0f && obj_id < 1.5f) {
    float3 pos = ro + t*rd;
    float3 nor = nSphere(pos, sph1);
    float dif = clamp(dot(nor, light), 0.0f, 1.0f);
    float ao = 0.5 + 0.5*nor.y;
    color = float3(1.0f, 0.8f, 0.6f)*dif*ao + float3(0.5f, 0.6f, 0.7f)*ao;
  } else if (obj_id > 1.5f) {
    float3 pos = ro + t*rd;
    float3 nor = nPlane(pos);
    float dif = clamp(dot(nor, light), 0.0f, 1.0f);
    float amb = smoothstep(0.0f, 2.0f*sph1.w, length(pos.xz-sph1.xz));
    color = float3(1.0f, 0.8f, 0.6f)*dif + amb*float3(0.5f, 0.6f, 0.7f);
    color = float3(amb, amb, amb) * 0.7f;
  }

  screen[id.xy] = float4(color, 1.0);
}
