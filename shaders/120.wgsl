#include "Dave_Hoskins/hash"

const pi: f32 = acos(-1.0);

// Random number generator (rv = "random vector")
var<private> rv: vec4f;
fn shuffle() { rv = fract(hash44(rv) + rv.wxyz); }

// This is the accumulator for the main image.
//
// For every pixel on the screen, there is one atomic counter for how many
// points landed on that pixel. The `draw` pass writes to this buffer in random
// order, and the `present` pass writes its content to the screen and resets
// all entries to zero.
//
// The elements in this array need to be atomic since multiple threads may try
// to write to the same element at the same time. It is a uint rather than a
// float because atomic floats aren't supported (you can build your own using
// `bitcast` and `atomicCompareExchangeWeak` but it's slow and complicated).
//
// As of writing this, each storage buffers on this website is backed by a
// massive 128MB memory allocation. Since a 1920x1080 RGBA f32 texture is only
// about 33.18 MB, we can be sure the array is big enough for our use cases.
#storage accumulator array<atomic<u32>>

// Chooses one of the attractor points randomly.
//
// You can adjust the number of points using the `n_points` slider.
fn get_random_corner_point() -> vec2f {
    let N = round(custom.n_points);
    let k = floor(rv.w * N);
    let a = 2.0 * pi * k / N;
    return vec2f(sin(a), -cos(a));
}

// The `draw` stage splats points into the `accumulator`.
//
// This stage is dispatched in 32*32 workgroups, and each workgroup consists
// of 16*16 threads, meaning this function runs 262144 times.
#workgroup_count draw 32 32 1
@compute @workgroup_size(16, 16)
fn draw(@builtin(global_invocation_id) id: vec3u) {
    let R = vec2i(textureDimensions(screen));

    // Initialize random state
    rv = vec4f(id.xyzz + time.frame);
    shuffle();

    // Set the starting point of the chaos game.
    //
    // The distribution you choose here may be visible in the final image,
    // e.g. if you choose the starting point uniformly from a large square,
    // you might get some square shaped artefacts in the noise.
    //
    // Here, I'm using the Box-Muller transfrom to generate random points from
    // a standard Gaussian distribution. (This is probably overkill, but idc)
    let r = sqrt(-2.0 * log(rv.xy));
    let t = 2.0 * pi * rv.zw;
    let s = r.xyxy * vec4f(cos(t), sin(t));
    var curr = (s.xz + s.yw) / sqrt(2.0);

    for (var k = 0; k < 32; k++) {
        shuffle();

        // Choose random corner point
        let point = get_random_corner_point();

        // Step towards that point
        curr = mix(curr, point, custom.step_size);

        // Compute the pixel coordinate of the current point
        let coord = vec2i(0.8 * f32(R.y) * curr.xy + vec2f(R)) / 2;

        // Check if coordinate is on-screen
        if all(coord < vec2i(0)) || all(coord >= R)  {
            continue;
        }

        // Increment atomic for this coordinate
        let idx = coord.x + R.x * coord.y;
        let acc = &accumulator[idx];
        atomicAdd(acc, 1);
    }
}

// The `present` stage reads values from the accumulator and presents the
// results on the screen.
//
// Since no `workgroup_count` is given, this runs just enough to cover the
// screen. We still process 16*16 pixels at once.
@compute @workgroup_size(16, 16)
fn present(@builtin(global_invocation_id) id: vec3u) {
    let R = textureDimensions(screen);

    // Ensure the current pixel is actually on the screen
    //
    // (yes, this is something we need to worry about - unless the width and
    // height of the screen are both multiples of 16, some of the workgroups
    // are going to hang off the edge of the screen, and we need to clip those
    // pixels manually)
    if any(id.xy >= R) { return; }

    // Read and reset the accumulator for the current pixel
    let idx = id.x + R.x * id.y;
    let acc = &accumulator[idx];
    let val = atomicExchange(acc, 0);

    // Compute pixel brightness
    //
    // Here, `pdf` is the expected value of any pixel on the screen. It is a
    // good baseline for adjusting the brightness of the image and helps to
    // make the effect resolution independent.
    let pdf = f32(32 * 32 * 16 * 16 * 32) / f32(R.x * R.y);
    let col = pow(custom.alpha, 2.2) * f32(val) / pdf;

    // Write to screen
    textureStore(screen, id.xy, vec4f(col));
}
