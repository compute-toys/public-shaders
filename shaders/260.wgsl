#define PI  3.1415926535897932384f
#define thrds 32        //GPU threads per wrap
#define srf .1f         //distance considered surface
#define rml 50          //raymarch loop
#define G 64.f         //size of fractal

fn objec(p2: float3) -> float
{
    var p = p2;
    var rt = cos(45.9f*.25f*.5f-float2(.0,.5)*PI);
    var st = sin(45.9f* 1.f*.5f-float2(.0,.5)*PI);
    var a = 999999.f;
    var s = 1.f;
    for(var i=0; i<12; i++)
    {
        p = abs(p)-.5f*G;
        p = float3( p.x*rt.x - p.z*rt.y, p.y,
                    p.x*rt.y + p.z*rt.x );
        p = float3( p.x*st.x - p.y*st.y,
                    p.x*st.y + p.y*st.x, p.z );
        p *= 1.7f;
        s *= 1.7f;    
        a = min(a,(length(float2(max(length(p*float3(1,0,1))-1.5f*G,0.f),p.y+.1f*G))-.1f*G)/s);
    }
    return a;
}
fn getCol(pos: float3) -> float3
{
    var p = pos;
    var rt = cos(45.9f*.25f*.5f-float2(.0,.5)*PI);
    var st = sin(45.9f* 1.f*.5f-float2(.0,.5)*PI);
    var px = dot(p,float3(2.f/G));
    var s = 1.f;
    var col = float3(0);
    for(var i = 0; i < 12; i++)
    {
        p = abs(p)-.5f*G;
        p = float3( p.x*rt.x - p.z*rt.y, p.y,
                    p.x*rt.y + p.z*rt.x );
        p = float3( p.x*st.x - p.y*st.y,
                    p.x*st.y + p.y*st.x, p.z );
        p *= 1.5f+.4083f*.5f;
        s *= 1.5f+.4083f*.5f;   
        if(i>2){col += (cos((float(i)+.5f)*64.f*.1546f+float3(1,2,3)*PI*2.f/3.f)*.5f+.4f)*
                min(length(float2(max(length(p*float3(1,0,1))-1.f*G,0.f),p.y))-.6f*G,G*.8f);}
    }
    return col/G*.32f;
}
#storage camo array<float3,3>;
#workgroup_count camera 1 1 1
@compute @workgroup_size(1,1,1)
fn camera(@builtin(global_invocation_id) id: uint3)
{
    var keyW = float(keyDown(87u));
    var keyS = float(keyDown(83u));
    var keyA = float(keyDown(65u));
    var keyD = float(keyDown(68u));
    var camDir = camo[0];  if(time.frame==0u){camDir = float3(0,0,1);}
    var camPos = camo[1];  if(time.frame==0u){camPos = float3(9,66,-22);}
    var mouPre = camo[2];
    var mouNow = float3(float2(mouse.pos),float(mouse.click));
    var mouDif = mouNow-mouPre;
    let res    = float2(textureDimensions(screen));
    //mod camDir data
    {
        var cd = camDir;
        var m = mouPre.z*mouDif*-4.f/res.y;
        var y = asin(camDir.y) + m.y;
        if(-1.56f > y){y =-1.56f;};
        if( 1.56f < y){y = 1.56f;};
        var a = cos(m.x-float2(0,PI*.5f));
        var n = cos(y)*normalize(float2(cd.x*a.x - cd.z*a.y,
                                        cd.x*a.y + cd.z*a.x));
        if(any(m.xy!=float2(0))){cd = float3(n.x,sin(y),n.y);}
        camo[0] = cd;
    }
    //mod camPos data
    {
        var s = objec(camPos)*.04f; //*time.elapsed;
        camo[1] = camPos + (keyW-keyS)*s*camDir +
                           (keyD-keyA)*s*normalize(float3(camDir.z,0,-camDir.x));
    }
    //mod mouse data
    {
        camo[2] = float3(mouNow.x,mouNow.y,float(mouNow.z!=0.f && (any(mouDif.xy!=float2(0)) || mouPre.z!=0.f)));
    }
}
@compute @workgroup_size(thrds,1,1)
fn main_image(@builtin(global_invocation_id) id: uint3, @builtin(local_invocation_index) id2: uint)
{
    var screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x) { return; }
    var res = float2(screen_size);
    var uv  = float2(id.xy)+ .5f;
        uv  = float2(1,-1)*(2.f*uv.xy-res.xy)/res.y;
    var camDir = camo[0];
    var camPos = camo[1];
    var mtx0 = normalize(float3(camDir.z,0,-camDir.x));
    var mtx  = mat3x3<f32>(mtx0, cross(camDir,mtx0), camDir);
    var ray  = mtx*normalize(float3(uv,1));//direction of ray from camera
    var p = camPos;
    var l = 0.f;                    //distance to surface
    for(var i = 0; i < rml; i++)    //ray march
    {
        l = objec(p);  p += l*ray;
    }
    var c = pow(getCol(p),float3(2));
    textureStore(screen, id.xy, float4(c,1));
}
