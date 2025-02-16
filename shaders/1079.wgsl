// basic raymarching

fn sphereMap(p: float3) -> float
{
    return length(p) - 1.; // distance to a sphere of radius 1
}

fn rayMarchSphere(ro: float3, rd: float3, tmin: float, tmax: float) -> float
{
    var t = tmin; 
    for(var i=0; i<400; i++ )
    {
        let  pos = ro + t*rd;
        let  d = sphereMap( pos );
        t += d;
        if( t>tmax ) { break; };
    }

    return t;
}


@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    let screen_size = uint2(textureDimensions(screen));
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    let fragCoord = float2(id.xy) + .5;
    let resolution = float2(screen_size);

    var uv = (fragCoord * 2. - resolution.xy) / resolution.y;

    // camera
    let ro = vec3(0.0, 0, -3.0);
    let rd = normalize(vec3(uv, 1.));

    //----------------------------------
    // raycast terrain and tree envelope
    //----------------------------------
    const tmax = 2000.0;
    let t = rayMarchSphere(ro, rd, 0, tmax);
    let col = vec3(t * .2);
    textureStore(screen, int2(id.xy), float4(col, 1.));
}
