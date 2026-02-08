struct StateDynamicPart {
    x1: f32, y1: f32, vx1: f32, vy1: f32,
    x2: f32, y2: f32, vx2: f32, vy2: f32,
    x3: f32, y3: f32, vx3: f32, vy3: f32,
};

struct StateConstantPart {
    m1: f32,
    m2: f32,
    m3: f32,
    g: f32,
    soft: f32,
    dt: f32,
};

#storage state_dynamic array<StateDynamicPart>;
#storage state_constant array<StateConstantPart>;


fn init_state(idx: u32, x: f32, y: f32) {
    state_dynamic[idx] = StateDynamicPart( // CHANGE THESE VALUES
        15, y, -1.1, -2.8,
        y, -10, -x, -2.1,
        10, x, -0.1, 0.2
    );

    state_constant[idx] = StateConstantPart( // OR THESE VALUES
        1.0, 2.0, 1.0,
        y,
        1,
        0.8
    );
}

fn grav_force(
    x1: f32, y1: f32,
    x2: f32, y2: f32,
    m1: f32, m2: f32,
    g: f32, soft: f32
) -> vec2f {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let r2 = dx*dx + dy*dy + soft*soft;
    let inv_r = 1.0 / sqrt(r2);
    let f = g * m1 * m2 * inv_r * inv_r;
    return vec2f(dx, dy) * (f * inv_r);
}


fn update_sim(idx: u32) {
    let d = state_dynamic[idx];
    let c = state_constant[idx];

    let f12 = grav_force(d.x1, d.y1, d.x2, d.y2, c.m1, c.m2, c.g, c.soft);
    let f23 = grav_force(d.x2, d.y2, d.x3, d.y3, c.m2, c.m3, c.g, c.soft);
    let f31 = grav_force(d.x3, d.y3, d.x1, d.y1, c.m3, c.m1, c.g, c.soft);

    var vx1 = d.vx1 + ( f12.x - f31.x) / c.m1 * c.dt;
    var vy1 = d.vy1 + ( f12.y - f31.y) / c.m1 * c.dt;

    var vx2 = d.vx2 + (-f12.x + f23.x) / c.m2 * c.dt;
    var vy2 = d.vy2 + (-f12.y + f23.y) / c.m2 * c.dt;

    var vx3 = d.vx3 + (-f23.x + f31.x) / c.m3 * c.dt;
    var vy3 = d.vy3 + (-f23.y + f31.y) / c.m3 * c.dt;

    state_dynamic[idx].x1 += vx1 * c.dt;
    state_dynamic[idx].y1 += vy1 * c.dt;
    state_dynamic[idx].vx1 = vx1;
    state_dynamic[idx].vy1 = vy1;

    state_dynamic[idx].x2 += vx2 * c.dt;
    state_dynamic[idx].y2 += vy2 * c.dt;
    state_dynamic[idx].vx2 = vx2;
    state_dynamic[idx].vy2 = vy2;

    state_dynamic[idx].x3 += vx3 * c.dt;
    state_dynamic[idx].y3 += vy3 * c.dt;
    state_dynamic[idx].vx3 = vx3;
    state_dynamic[idx].vy3 = vy3;
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(screen);

    if (id.x >= size.x || id.y >= size.y) {
        return;
    }

    let idx = id.x + id.y * size.x;

    let x = custom.min_x + f32(id.x) / f32(size.x) * (custom.max_x - custom.min_x);
    let y = custom.min_y + f32(id.y) / f32(size.y) * (custom.max_y - custom.min_y);

    if (state_constant[idx].dt == 0.0) {
        init_state(idx, x, y);
        return;
    }

    let n = 10u;
    
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        update_sim(idx);
    }

    let d = state_dynamic[idx];

    textureStore(
        screen,
        vec2u(id.xy),
        vec4f(
            sigmoid(d.x1, 1),
            sigmoid(d.y3, 1),
            sigmoid(d.vy1, 1),
            1.0
        )
    );
}

fn dist(x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    return sqrt((x2-x1) * (x2-x1) + (y2-y1) * (y2-y1));
}

fn sigmoid(x: f32, stretch: f32) -> f32 {
    let s = max(stretch, 1e-6);
    return 1.0 / (1.0 + exp(-x / s));
}