#define COUNT 16 // million

#define ITERATIONS (COUNT * 4 * 16 * 16 * 977)
#workgroup_count deJongAttractor COUNT 4 1

struct parameters{a: f32, b: f32, c: f32, d: f32, t: f32};
#storage param parameters;
#storage buf array<array<atomic<u32>,3>>

fn pcg3d(vin: vec3u) -> vec3u
{
    var v = vin * 1664525u + 1013904223u;
    v.x += v.y*v.z; v.y += v.z*v.x; v.z += v.x*v.y;
    v ^= v >> vec3u(16u);
    v.x += v.y*v.z; v.y += v.z*v.x; v.z += v.x*v.y;
    return v;
}

fn pcg3df(vin: vec3u) -> vec3f
{
    return vec3f(pcg3d(vin)) / f32(0xffffffffu);
}

@compute @workgroup_size(16, 16)
fn clear(@builtin(global_invocation_id) id: vec3u) {
    let scsz = textureDimensions(screen);
    if (id.x >= scsz.x || id.y >= scsz.y) { return; }
    let idx = id.x + id.y * scsz.x;
    atomicStore(&buf[idx][0], 0u);
    atomicStore(&buf[idx][1], 0u);
    atomicStore(&buf[idx][2], 0u);

    if(id.x != 0 || id.y != 0) {return;}

    // attractor parameters
    param.t += pow(10., custom.animationSpeed / 3.33 - 5);
    param.a = 4 * sin((custom.timeOffset + param.t) * 1.03);
    param.b = 4 * sin((custom.timeOffset + param.t) * 1.07);
    param.c = 4 * sin((custom.timeOffset + param.t) * 1.09);
    param.d = 4 * sin((custom.timeOffset + param.t) * 1.13);
}

@compute @workgroup_size(16, 16)
fn deJongAttractor(@builtin(global_invocation_id) id: vec3u)
{
    let scsz = textureDimensions(screen);
    let txsz = min(scsz.x, scsz.y);
    let xoff = (scsz.x - txsz) / 2;
    let yoff = (scsz.y - txsz) / 2;

    // random starting point per thread
    let rnd = pcg3df(vec3u(id.xy, u32(time.frame)));
    var x1 = 2 * sin(6.28 * rnd.x);
    var y1 = 2 * sin(6.28 * rnd.y);

    // prerun to converge onto attractor from random starting point
    for(var i = 0; i < 32; i++)
    {
        let x2 = sin(param.a * y1) - cos(param.b * x1);
        let y2 = sin(param.c * x1) - cos(param.d * y1);
        x1 = x2;
        y1 = y2;
    }

    // 977 iterations per thread (977 * 16 * 16 * 4 = 1,000,448)
    for(var i = 0; i < 977; i++)
    {
        let x2 = sin(param.a * y1) - cos(param.b * x1);
        let y2 = sin(param.c * x1) - cos(param.d * y1);
        let x = xoff + u32(x2 * 0.25 * f32(txsz) * 0.96 + f32(txsz) * 0.5);
        let y = yoff + u32(y2 * 0.25 * f32(txsz) * 0.96 + f32(txsz) * 0.5);
        let idx = y * scsz.x + x;
        let dx = x2 - x1;
        let dy = y2 - y1;
        atomicAdd(&buf[idx][0], u32(256. * abs(dx)));
        atomicAdd(&buf[idx][1], u32(256. * abs(dy)));
        atomicAdd(&buf[idx][2], 256u);
        x1 = x2;
        y1 = y2;
    }
}

@compute @workgroup_size(16, 16)
fn fragment(@builtin(global_invocation_id) id: vec3u) {
    let scsz = textureDimensions(screen);
    let txsz = min(scsz.x, scsz.y);
    if (id.x >= scsz.x || id.y >= scsz.y) { return; }

    let idx = id.x + (scsz.y - id.y - 1) * scsz.x;
    var col = vec3f(f32(atomicLoad(&buf[idx][0])), f32(atomicLoad(&buf[idx][1])), f32(atomicLoad(&buf[idx][2])));
    col = col * f32(txsz) * f32(txsz) / (2048.0 * ITERATIONS);
    col = pow(col, float3(1.5));

    textureStore(screen, id.xy, vec4f(col, 1.));
}
