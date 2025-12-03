enable subgroups;

#storage atomic_storage array<atomic<u32>>;

#define TERMINAL_ROWS 10
#define TERMINAL_COLS 32

fn slot() -> u32
{
    return time.frame % 10;
}

fn terminal_render(pos: uint2) -> float4 {
    let screen_size = uint2(textureDimensions(screen));
    let aspect = float(screen_size.y) / float(screen_size.x) * float(TERMINAL_COLS) / float(TERMINAL_ROWS);
    let texel = float(TERMINAL_ROWS) / float(screen_size.y);
    var uv = float2(pos) * float2(aspect, 1.) * texel;

    var number = atomicLoad(&atomic_storage[u32(uv.y)]);
    
    let digits = 10;
    let col = digits-1 - i32(uv.x);
    if (col < 0 || col > digits-1)
    {
        return float4(0);
    }
    for (var i=0; i<col; i++)
    {
        number /= 10u;
    }
    let digit = number % 10u;
    let ascii = 0x30u + u32(digit);

    if (0x20 < ascii && ascii < 0x80) { // printable character
        uv = fract(uv);
        uv.x = (uv.x - .5) / aspect + .5; // aspect ratio correction
        uv += float2(uint2(ascii % 16u, ascii / 16u)); // character lookup
        let sdf = textureSampleLevel(channel1, trilinear, uv / 16., 0.).a;

        var col = float4(0);
        col = mix(col, float4(0,0,0,1), smoothstep(.525 + texel, .525 - texel, sdf));
        col = mix(col, float4(1,1,1,1), smoothstep(.490 + texel, .490 - texel, sdf));
        return col;
    }
    return float4(0);
}

#workgroup_count reset 1 1 1

@compute @workgroup_size(1, 1)
fn reset()
{
    atomicStore(&atomic_storage[slot()], 0);
}

//random number generator
//https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
//https://jcgt.org/published/0009/03/02/

fn rand_pcg(rng_state : ptr<function,u32>) -> u32
{
    *rng_state = *rng_state * 747796405u + 2891336453u;
    let state = *rng_state;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_xqo(rng_state : ptr<function,u32>) -> uint
{
    let input = *rng_state;
    let state = (input | 1u) ^ (input * input);
    *rng_state = state;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn unsignedFloat(r : u32) -> f32    //[0,+1]
{
    let mantissa = r & ((1u<<23)-1);
    let frand_plus_one = mantissa | (127<<23);  //127 exponent of 2^0 = 1 and some fractional bits....
    return bitcast<f32>(frand_plus_one)-1.;
}

fn signedFloat(r : u32) -> f32      //[-1,+1]
{
    let frand  = unsignedFloat(r);
    let signed = bitcast<u32>(frand) | ((r << 8) & (1u<<31));   //use an extra bit of randomness for sign
    return bitcast<f32>(signed);
}

#storage noise array<vec2f>

var<workgroup> sharedmem : atomic<u32>;

const w = 1000u;

//one beellion samples mwuahahahaha
#workgroup_count monte 250 250 250

@compute @workgroup_size(4, 4, 4)
fn monte(@builtin(global_invocation_id) id:vec3u,
        @builtin(local_invocation_index) flat : u32,
        @builtin(subgroup_invocation_id) sgid : u32 )
{
    if (flat == 0u)
    {
        atomicStore(&sharedmem, 0);
    }
    workgroupBarrier();

    var sum = 0u;

    let flatid = id.z * w * w + id.y * w + id.x;

    var rand = flatid + (time.frame*739)%7919;                

    //four beellion samples mwuahahahahahahaha
    for (var i=0; i<4; i++)
    {
        let randx = rand_xqo(&rand);
        let randy = rand_xqo(&rand);

        let uv = vec2f(unsignedFloat(randx),unsignedFloat(randy));

        let inside = dot(uv,uv) < 1 + 2.327e-6;    //yeah there's some error :(

        //just avoids hitting the atomic so much
        let sgcount = countOneBits(subgroupBallot(inside));        
        sum += sgcount.x + sgcount.y + sgcount.z + sgcount.w;

        let screen_size = textureDimensions(screen);
        if (flatid < screen_size.x * screen_size.y && i == 0) 
        {
            noise[flatid] = uv;
        }
    }

    //sum the subgroups to shared memory
    if (subgroupElect())
    {
        atomicAdd(&sharedmem, sum);
    }

    //accumulate final result in global memory
    workgroupBarrier();

    if (flat == 0u)
    {
        let tot = atomicLoad(&sharedmem);
        if (tot > 0)
        {
            atomicAdd(&atomic_storage[slot()], tot);
        }
    }
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);

    // Normalised pixel coordinates (from 0 to 1)
    let uv = fragCoord / vec2f(screen_size);

    // Time varying pixel colour
    var col = .5 + .5 * cos(time.elapsed + uv.xyx + vec3f(0.,2.,4.));

    let flatid = id.y * screen_size.x + id.x;
    col = vec3f( noise[flatid], 0 );

    col += terminal_render(id.xy).xyz;

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}
