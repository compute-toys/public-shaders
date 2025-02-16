

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);

    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }


    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);

    var uv = (fragCoord * 2.0 - vec2f(screen_size.xy)) / f32(screen_size.y);
    var uv0 = uv;
    var finalcol = vec3f(0.0);

    for (var i = 0; i < i32(custom.iterations * 10.); i++) {
        uv = fract(uv * 1.5) - .5;
 
        var d = length(uv) * exp(-length(uv0));

        var col = hcl2rgb(
            vec3f( 
            length(uv0) + f32(i) * .4 + time.elapsed, 
            custom.chroma, 
            custom.lightness)
        );

        d = sin(d * 8. + time.elapsed * custom.speed * 10.) / 8.;
        d = abs(d);
    
        //d = smoothstep(.0, .1 , d);
        d = (.01 / d);
        
        finalcol += col * d * custom.neon * 10.;
        
    }
    // Convert from gamma-encoded to linear colour space
    finalcol = pow(finalcol, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(finalcol, 1.));
}

fn hcl2rgb(hcl1: vec3f) -> vec3f{
    const PI = 3.14159265359;
    var hcl = hcl1;
    hcl.y *= 0.33;
    
    var lab = vec3(
        hcl.z,
        hcl.y * cos(hcl.x * PI*2.0),
        hcl.y * sin(hcl.x * PI*2.0)
    );
    
    var lms = vec3(
        lab.x + 0.3963377774f * lab.y + 0.2158037573f * lab.z,
        lab.x - 0.1055613458f * lab.y - 0.0638541728f * lab.z,
        lab.x - 0.0894841775f * lab.y - 1.2914855480f * lab.z
    );
    
    lms = pow(max(lms, vec3(0.0)), vec3(3.0));
    
    var rgb = vec3f(
    4.0767416621 * lms.x - 3.3077115913 * lms.y + 0.2309699292 * lms.z,
    -1.2684380046f * lms.x + 2.6097574011f * lms.y - 0.3413193965f * lms.z,
    -0.0041960863f * lms.x - 0.7034186147f * lms.y + 1.7076147010f * lms.z
    );
     
    rgb = transfer(rgb);
    
    if (any( rgb < vec3(0.0)) || any(rgb > vec3(1.0))) {
        rgb = vec3(0.5);
    }

    return rgb;
}

fn transferf(v: f32) -> f32 {
    
    var r = 0.0;
    if (v <= 0.0031308) {
            r = 12.92 * v;
        }else {
            r = 1.055 * pow(v, 0.4166666666666667) - 0.055;
        };
    return r;
}

fn transfer(v: vec3f)->vec3f {
    return vec3f(transferf(v.x), transferf(v.y), transferf(v.z));
}