#define TAU 6.283185307179586

// Dynamic variables (change over time)
struct Storage {
    // x - angle1
    // y - angle2
    // z - velocity1
    // w - velocity2
    ar: array<float4, SCREEN_WIDTH * SCREEN_HEIGHT>,
}

// Constant parameters (do not change after initialization)
struct Constant {
    m1: f32, // mass 1
    l1: f32, // length 1
    m2: f32, // mass 2
    l2: f32, // length 2
    g: f32,  // gravity
    fc: f32, // friction coefficient
    dt: f32, // delta time
}

#storage c Constant
#storage s Storage

fn ftdd1(t1: f32, t2: f32, td1: f32, td2: f32) -> f32 {
    let num = -c.m2 * c.l1 * td1 * td1 * sin(t1 - t2) * cos(t1 - t2)
              + c.m2 * c.g * sin(t2) * cos(t1 - t2)
              - c.m2 * c.l2 * td2 * td2 * sin(t1 - t2)
              - (c.m1 + c.m2) * c.g * sin(t1);
    
    let den = (c.m1 + c.m2) * c.l1 - c.m2 * c.l1 * cos(t1 - t2) * cos(t1 - t2);
    
    return num / den - c.fc * sign(td1);
}

fn ftdd2(t1: f32, t2: f32, td1: f32, td2: f32) -> f32 {
    let num = c.m2 * c.l2 * td2 * td2 * sin(t1 - t2) * cos(t1 - t2)
              + (c.m1 + c.m2) * c.g * sin(t1) * cos(t1 - t2)
              + c.l1 * td1 * td1 * sin(t1 - t2) * (c.m1 + c.m2)
              - c.g * sin(t2) * (c.m1 + c.m2);
    
    let den = c.l2 * (c.m1 + c.m2) - c.m2 * c.l2 * cos(t1 - t2) * cos(t1 - t2);
    
    return num / den - c.fc * sign(td2);
}

fn gpos(a: float2) -> float4 {
    // Position of the first pendulum bob
    let pos1 = float2(
        c.l1 * sin(a.x),
        c.l1 * cos(a.x)
    );

    // Position of the second pendulum bob
    let pos2 = float2(
        pos1.x + c.l2 * sin(a.y),
        pos1.y + c.l2 * cos(a.y)
    );

    // Return both positions as a float4
    return float4(pos1.x, pos1.y, pos2.x, pos2.y);
}

#dispatch_once init
@compute @workgroup_size(16, 16)
fn init(@builtin(global_invocation_id) id: uint3) {
    c.m1 = 1.0;           // mass of pendulum 1 (kg)
    c.l1 = 1.0;           // length of pendulum 1 (m)
    c.m2 = 1.0;           // mass of pendulum 2 (kg)
    c.l2 = 1.0;           // length of pendulum 2 (m)
    c.g = custom.gravity; // acceleration due to gravity (m/s^2)
    c.fc = 0.1;           // friction coefficient
    c.dt = custom.time_step;         // time step (s)
   
    let screen_size = uint2(textureDimensions(screen));
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    var uv = ((2.0 * (vec2<f32>(id.xy)) - vec2<f32>(screen_size)) / f32(screen_size.y));
    uv *= 0.8;
    uv -= 1.0;

    let idx = id.y * screen_size.x + id.x;

    s.ar[idx] = float4(
        (uv.x + 1.0) / 2.0 * TAU,  // angle1
        (uv.y + 1.0) / 2.0 * TAU,  // angle2
        custom.initial_velocity_1, // velocity1
        custom.initial_velocity_2, // velocity2
    );
}

@compute @workgroup_size(16, 16)
fn tick(@builtin(global_invocation_id) id: uint3) {
    let screen_size = uint2(textureDimensions(screen));
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    c.dt = custom.time_step;
    c.g = custom.gravity;

    let idx = id.y * screen_size.x + id.x;
    var state = s.ar[idx];

    let t1 = state.x;
    let t2 = state.y;
    let td1 = state.z;
    let td2 = state.w;

    let tdd1 = ftdd1(t1, t2, td1, td2);
    let tdd2 = ftdd2(t1, t2, td1, td2);

    let new_td1 = td1 + tdd1 * c.dt;
    let new_td2 = td2 + tdd2 * c.dt;
    let new_t1 = t1 + new_td1 * c.dt;
    let new_t2 = t2 + new_td2 * c.dt;

    s.ar[idx] = float4(new_t1, new_t2, new_td1, new_td2);
}

fn sd_circle(p: float2, r: f32) -> f32 {
    return length(p) - r;
}

fn ud_segment(p: float2, a: float2, b: float2) -> f32 {
    let ba = b - a;
    let pa = p - a;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - h * ba);
}

fn map(p: float2, angles: float2) -> f32 {
    let gp = gpos(angles);
    let p1 = gp.xy;
    let p2 = gp.zw;

    let nd1 = sd_circle(p - p1, 0.02);
    let nd2 = sd_circle(p - p2, 0.02);

    let l1 = ud_segment(p, float2(0.0), p1);
    let l2 = ud_segment(p, p1, p2);
    let nd3 = min(l1, l2);

    return min(min(nd1, nd2), nd3);
}

fn render(ic: float3, p: float2, angles: float2, muv: float2) -> float3 {
    let d = map(p, angles);

    var col = ic;

    if d <= 0.0
    {
      col = float3(muv.x, muv.y, 0.0);
    }

    col = mix( col, float3(1.0), 1.0-smoothstep(0.0,0.01,abs(d)) );

    return col;
}

fn gidx() -> vec2<u32> {

    return vec2<u32>(mouse.pos);
}

fn wrap_angle(angle: f32) -> f32 {
    return angle - TAU * floor(angle / TAU);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: uint3) {
    let screen_size = uint2(textureDimensions(screen));

    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    var uv = (2.0 * (vec2<f32>(id.xy)) - vec2<f32>(screen_size)) / f32(screen_size.y);
    uv *= 3.0;

    let idx = id.y * screen_size.x + id.x;
    var a = s.ar[idx];

    let ps = gpos(a.xy);
    let p1 = ps.xy;
    let p2 = ps.zw;

    let r = p1.x;
    let g = mix(p1.y, p2.x, 0.5);
    let b = p2.y;

    var col = vec3<f32>(r,g,b);

    let mpos = gidx();
    if (mpos.x > 0 || mpos.y > 0) {
      let midx = mpos.y * screen_size.x + mpos.x;
      var muv = (2.0 * (vec2<f32>(mpos.xy)) - vec2<f32>(screen_size)) / f32(screen_size.y);
      var angles = s.ar[midx];
      col = render(col, uv, angles.xy, muv);
    }

    col = pow(col, vec3<f32>(2.2));

    textureStore(screen, int2(id.xy), vec4<f32>(col, 1.0));
}
