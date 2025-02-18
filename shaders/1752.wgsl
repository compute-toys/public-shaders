#storage render array<atomic<u32>>

const PI = acos(-1.);

// bt.709 weights
const Wr=0.2126;
const Wb=0.0722;

// bt.2020 weights
// const Wr=0.2627;
// const Wb=0.0593;
 const Wg=1.-Wr-Wb;

const to_ycbcr = mat3x3(
    Wr, -0.5*Wr/(1.-Wb),  0.5,
    Wg, -0.5*Wg/(1.-Wb), -0.5*Wg/(1.-Wr),
    Wb,  0.5,            -0.5*Wb/(1.-Wr)
);

fn hash2(x_in: u32) -> u32 {
    var x = x_in^(x_in >> 16);
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

fn rand(x_in: u32) -> f32 {
    let f = bitcast<f32>((hash2(x_in) & 0x007fffff) | 0x3f800000) - 1.0;
    return f;
}

fn randf(x_in: f32) -> f32 {
    let f = bitcast<f32>((hash2(bitcast<u32>(x_in)) & 0x007fffff) | 0x3f800000) - 1.0;
    return f;
}


// One-time generation of the RGB cube point coordinates
@compute @workgroup_size(16, 16)
#dispatch_once build_rgb_cube
fn build_rgb_cube(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    let total = screen_size.x*screen_size.y;
    let block_size = u32(pow(f32(total),1./3.));

    let idx = id.x+id.y*screen_size.x;
    if(idx >= block_size*block_size*block_size) {return;}

    let r = vec3f(rand(idx),rand((idx<<8)^(idx>>24)),rand((idx<<16)^(idx>>16)));

    let mid = id%block_size;

    let col = vec3f(
        vec3u(
            idx%block_size,
            (idx/block_size)%block_size,
            idx/(block_size*block_size)
        )
    ) / vec3f(f32(block_size)) * vec3f(1.-r*0.03);
    textureStore(pass_out, int2(id.xy), 0, vec4f(col,1));
}


@compute @workgroup_size(16, 16)
fn clear_render_storage(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    let idx = id.x + screen_size.x * id.y;

    atomicStore(&render[idx], 0);
}


fn roty(u: vec3f, r: f32) -> vec3f
{
    let s = sin(r);
    let c = cos(r);
    
    let m = mat3x3(
        c,0,-s,
        0,1,0,
        s,0,c
    );
    
    return m*u;
}

@compute @workgroup_size(16, 16)
fn render_point_cloud(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    let fscreen_size=vec2f(screen_size);

    let ar = fscreen_size.y / fscreen_size.x;

    let uv = vec2f(id.xy) / fscreen_size;

    // swap comments tags below to use texture in channel0 as point cloud source
    var rgb = textureLoad(pass_in, int2(id.xy), 0, 0).xyz;
//    var rgb = textureSampleLevel(channel0,bilinear,uv,0.).xyz;

    let r = randf(rgb.x)*randf(rgb.y)*randf(rgb.z);

    let col = mix(
        rgb,
        to_ycbcr * rgb + vec3(0,0.5,0.5),
        pow( sin(sin((time.elapsed+rgb.x+r)*0.2*PI/2.)*PI/2.), 4.)
    );

    var pos = roty((col - 0.5), time.elapsed*0.2);
    pos = pos*vec3(ar,1,1) / (pos.z+1.8)+0.5; // hacky perspective rendering

    let idx = u32(pos.x * fscreen_size.x) + u32(pos.y*fscreen_size.y)*screen_size.x;

    // pack the components in an u32 for atomic storage
    // put the depth in the most significant bits to allow depth-testing
    // with atomicMax
    let output = (u32(clamp(0.,1.,1.-pos.z)*255.) << 24) |
    (u32(clamp(0.,1.,rgb.x)*255.)<<16) |
    (u32(clamp(0.,1.,rgb.y)*255.)<<8) |
    u32(clamp(0.,1.,rgb.z)*255.);

    atomicMax(&render[idx], output);
}


@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    let idx = (id.x+screen_size.x*id.y);
    let packed = atomicLoad(&render[idx]);

    let c = vec3u(
        (packed >> 16) & 0xff,
        (packed >> 8) & 0xff,
        packed &0xff
        );

    var col = vec3f(c)/255.;

    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    col = pow(col, vec3f(2.2));

    textureStore(screen, id.xy, vec4f(col, 1.));
}