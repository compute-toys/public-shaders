const BACKGROUND_TEST_PLATE = 0;

// BT.709 weights
const Wr: f32 = 0.2126;
const Wb: f32 = 0.0722;
const Wg: f32 = 1.-Wr-Wb;

// conversion matrices derived from
//  Y = R*Wr +  G*Wg +  B*Wb
// Cb = 0.5  * (B-Y) / (1-Wb)
// Cr = 0.5  * (R-Y) / (1-Wr)
// I've omitted the chroma scaling as the keying algorithm below requires them
// to be in the range from -1.0 to 1.0

const toycbcr = mat3x3<f32> (
    Wr, -Wr/(1.-Wb),  1.,
    Wg, -Wg/(1.-Wb), -Wg/(1.-Wr),
    Wb,  1.,         -Wb/(1.-Wr)
);

const torgb = mat3x3<f32> (
    1,      1,             1,
    0,     -Wb/Wg*(1.-Wb), 1.-Wb,
    1.-Wr, -Wr/Wg*(1.-Wr), 0
);

// to/from srgb
fn ftosrgb(val: f32) -> f32
{ 
    var v = val;
    if( v < 0.0031308f ) {
        v *= 12.92f;
    }
    else {
        v = 1.055 * pow(v, 1./2.4) - 0.055;
    }
    return v;
}

fn v3tosrgb(val: vec3f) -> vec3f {
    return vec3f(ftosrgb(val.x),ftosrgb(val.y),ftosrgb(val.z));
}

fn ftolinear (val: f32) -> f32 {
    if( val <= 0.04045) {
        return val / 12.92;
    }
    else {
        return pow((val+0.055)/1.055f,2.4);
    }
}

fn v3tolinear(val: vec3f) -> vec3f {
    return vec3f(ftolinear(val.x),ftolinear(val.y),ftolinear(val.z));
}


fn rotate(v: vec2f, r: f32) -> vec2f {    
    let s = sin(r);
    let c = cos(r);
    let mat = mat2x2<f32>(
        c,-s,
        s, c
    );

    return mat*v;  
}

fn is_in_rect(u: vec2f, r: vec4f) -> f32 {
    return step(r.x,u.x)*step(u.x,r.z)*step(r.y,u.y)*step(u.y,r.z);
}

fn test_plate(uv: vec2f) -> vec3f {
    return vec3f(.5,(uv*2.-1.)*vec2f(1.,-1.));
}

fn overlay_testplate(fg: vec3f, uv_in: vec2f) -> vec3f {
    let uv = (uv_in-vec2f(0.75,0))*4.;
    return mix(fg,test_plate(uv),is_in_rect(uv,vec4f(0,0,1,1)));
}

fn get_foreground_picture(uv: vec2f) -> vec3f {
    var foreground = toycbcr*v3tosrgb(textureSampleLevel(channel0,bilinear,fract(uv*vec2(2,1)-vec2(0.5,0)),0.).rgb);
    foreground = overlay_testplate(foreground, uv);
    return foreground;
}

fn get_background_picture(uv: vec2f) -> vec3f {
    var background = vec3f(0);
    if(BACKGROUND_TEST_PLATE == 1)
    {
        background = test_plate(uv);
    }
    else 
    {
        background = toycbcr*v3tosrgb(textureSampleLevel(channel1,bilinear,uv,0.).rgb);
    }

    return background;
}

fn key_generator(chroma: vec2f, key_angle: f32, acceptance_angle: f32) -> f32 {    
    // rotate the chroma plane so the key color aligns with the X axis
    let xy = rotate(chroma, key_angle);
    
    // The acceptance angle defines a slice in the colorwheel around the key color.
    // This will result in 0 on the key color, and increasing linearly to 1 when
    // reaching the border of the defined slice.
    let d = abs(xy.y) / tan(acceptance_angle / 2.);
    
    // x, clamped to positive values, defines the key, subtracting d to limit the range
    // of colors to within the acceptance angle.
    let key = clamp(xy.x - d, 0., 1.);
    
    return key;
}

fn foreground_suppressor(
    fg: vec3f,
    key: f32,
    angle: f32,
    y_suppression_scale: f32
) -> vec3f {
    // rotate the key ccw back onto the key color
    let chroma_key=vec2f(key*cos(angle),key*sin(angle));

    let suppressed_foreground = vec3f(
        fg.x-y_suppression_scale*key,
        fg.yz-chroma_key
    );
    
    return suppressed_foreground;
}

fn key_processor(key: f32, key_color_saturation: f32) -> f32 {
    let key_lift = 0.;
    let key_gain = 1./key_color_saturation;
    let background_key = clamp(key-key_lift,0.,1.)*key_gain;
    
    return background_key;
}

fn get_key_color(id: vec2<u32>, uv: vec2f) -> vec3f {
    var key_color = toycbcr*vec3f(0.40, 0.91, 0.49);
    if(time.frame==0) {
        textureStore(pass_out, id.xy, 0, vec4(key_color,1));
    }
    if(mouse.click!=0)
    {
        key_color = get_foreground_picture(vec2f(mouse.pos)/vec2f(textureDimensions(screen)));
        textureStore(pass_out, id.xy, 0, vec4(key_color,1));
    }
    else {
        key_color = textureLoad(pass_in, id.xy, 0, 0).rgb;
    }
    return key_color;
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    let fragCoord = vec2f(f32(id.x) + .5, f32(id.y) + .5);
    let uv = fragCoord / vec2f(screen_size);

    //////////////////////////////////

    let key_color = get_key_color(id.xy,uv);
    let chroma_angle = atan2(key_color.z, key_color.y);

    // Calculate required scale to suppress Y to 0 at key color luminance
    // Foreground areas with higher or lower luminance will brighten or darken the background image
    let y_suppression_scale = key_color.x / length(key_color.yz);
    let acceptance_angle = radians(90.0);

    let foreground = get_foreground_picture(uv);
    var background = get_background_picture(uv);

    let key_fg = key_generator(foreground.yz, chroma_angle, acceptance_angle);
    let key_bg = key_processor(key_fg,length(key_color.yz));

    let suppressed_foreground = foreground_suppressor(foreground, key_fg,chroma_angle, y_suppression_scale);
    
    background *= key_bg;
    
    var col = (suppressed_foreground + background);

    textureStore(screen, id.xy, vec4f(v3tolinear(torgb*mix(col, foreground, smoothstep(0.48, 0.52 , sin(time.elapsed)*.5+.5 - uv.x))), 1.0));
}
