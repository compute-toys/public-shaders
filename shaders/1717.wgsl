#storage plot array<atomic<u32>>;

const sRGB_to_XYZ = mat3x3<f32>(
    0.4124564,  0.2126729, 0.0193339,
    0.3575761, 0.7151522, 0.1191920,
    0.1804375, 0.0721750, 0.9503041
);

fn XYZ_to_xyY(p: vec3f) -> vec3f {
    return vec3f(
        p.x / (p.x + p.y + p.z),
        p.y / (p.x + p.y + p.z),
        p.y
    );
}

const PI=acos(-1);

fn spherical(X: vec3f) -> vec2f
{
    let th = atan2(X.x,X.z);
    let phi = asin(X.y);
    return 1./vec2f(PI,PI/2.)*vec2f(th,phi)*0.5+0.5;
}

fn rotcw_x(X: vec3f, r: f32) -> vec3f
{
    let rmat = mat3x3<f32>(cos(r),0,-sin(r),0,1,0,-sin(r),0,-cos(r));
    
    return X * rmat;
}

fn sample_pano(uv_in: vec2f, fscreen_size: vec2f) -> vec3f {
    const FOV = 90.;
    let zf = 0.5*fscreen_size.x/fscreen_size.y/-tan(radians(FOV*0.5));
    let uv = uv_in *2.-1.;
    let ray  = normalize(vec3f(uv,zf));
    let col = textureSampleLevel(channel0,bilinear,fract(spherical(rotcw_x(ray,PI*sin(time.elapsed*0.1)))),0.).xyz;
    return col;
}

fn sample_XYZ(uv: vec2f, fscreen_size: vec2f) -> vec3f {
    var sample = XYZ_to_xyY(sRGB_to_XYZ*sample_pano(uv,fscreen_size));
    return sample;
}

@compute @workgroup_size(16, 16)
fn init_plot(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = int2(textureDimensions(screen));
    let idx = int(id.x) + int(screen_size.x * int(id.y));

    atomicStore(&plot[idx], 0u);
}

// multi-pass example,
// converts the input image to xyY
@compute @workgroup_size(16, 16)
fn to_xyY(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);

    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let fragCoord = vec2f(f32(id.x) + .5, f32(id.y) - .5);
    let fscreen_size=vec2f(screen_size);

    let uv = fragCoord / fscreen_size;

    let col = sample_XYZ(uv,fscreen_size);

    textureStore(pass_out, int2(id.xy), 0, vec4f(col,1));
}

// for every pixel, take the xy part from the xyY space and increment a counter
// at the x,y coordinates in the plot output
// for better performance, a memory block per workgroup could be used to avoid
// collisions between workgroups with a post step to accumulate group outputs
@compute @workgroup_size(16, 16)
fn plot_gamut(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    let fscreen_size = vec2f(screen_size);

    if (id.x >= screen_size.x || id.y >= screen_size.y) {
        return;
    }

    let col = textureLoad(pass_in, int2(id.xy), 0, 0);

    let loc = clamp(
        vec2u(0),
        screen_size,
        vec2u(u32(col.x*fscreen_size.x),u32((1.-col.y)*fscreen_size.y)));

    atomicAdd(&plot[loc.x+loc.y*screen_size.x],1);
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    let fragCoord = vec2f(f32(id.x) + .5, f32(id.y) + .5);
    var uv = fragCoord / vec2f(screen_size);

    // align background image sourced from wikimedia to plotted result
    uv = uv * vec2f(972.0 / 870.0,1) * 102./97.2;
    uv = uv + vec2f(52.5 / 870.0, -102. / 972.0);

    let plot = pow(clamp(0.,1.,f32(atomicLoad(&plot[id.x+id.y*screen_size.x]))*0.05),0.5);
    let diagram = textureSampleLevel(channel1,bilinear,uv,0.).xyz;
    var col = mix(diagram,vec3f(1),plot);
    let back = sample_pano(uv,vec2f(screen_size));
    col = mix(back,col,select(length(col)*10.,1.,length(col)>=0.1));

    textureStore(screen, id.xy, vec4f(col,1.));
}
