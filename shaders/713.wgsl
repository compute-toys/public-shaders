//https://www.shadertoy.com/view/ctK3DV by altunenes
const PI: f32 = 3.14159265;

fn implicit(x: f32, y: f32, time: f32) -> f32 {
    let l = 0.5; 
    let t = time / 4.0;

    let n1 = 6.0 + 3.0 * sin(t);
    let m1 = 4.0 + 3.0 * cos(t);
    let n2 = 5.0 + 2.5 * cos(2.0 * t);
    let m2 = 3.0 + 2.5 * sin(2.0 * t);

    let val1 = cos(n1 * PI * x / l) * cos(m1 * PI * y / l) -
               cos(m1 * PI * x / l) * cos(n1 * PI * y / l);
    let val2 = cos(n2 * PI * x / l) * cos(m2 * PI * y / l) -
               cos(m2 * PI * x / l) * cos(n2 * PI * y / l);

return val1 + val2;
}

fn delf_delx(x: f32, y: f32, time: f32) -> f32 {
    const DX: f32 = 0.001;
    return (implicit(x + DX, y, time) - implicit(x - DX, y, time)) / (2.0 * DX);
}

fn delf_dely(x: f32, y: f32, time: f32) -> f32 {
    const DY: f32 = 0.001;
    return (implicit(x, y + DY, time) - implicit(x, y - DY, time)) / (2.0 * DY);
}

fn gradient(x: f32, y: f32, time: f32) -> vec2<f32> {
    return vec2(delf_delx(x, y, time), delf_dely(x, y, time));
}


@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let uv_x = f32(id.x) / f32(screen_size.x) - 0.5;
    let uv_y = f32(id.y) / f32(screen_size.y) - 0.5;
    let uv = vec2<f32>(uv_x, uv_y);

    let g = gradient(uv.x, uv.y, time.elapsed);
    let unit = 12.0 / f32(screen_size.y);
    let implicit_val = implicit(uv.x, uv.y, time.elapsed);
    let magnitude = sqrt(g.x * g.x + g.y * g.y);
    let sharp_val = smoothstep(-unit, unit, abs(implicit_val) / magnitude); 

    let col = vec3<f32>(0.01 + 1.0 * cos(time.elapsed + vec3<f32>(0.6, 0.8, 1.0) + 2.0 * PI * vec3<f32>(sharp_val)));
    textureStore(screen, vec2<i32>(id.xy), vec4<f32>(col, 1.0));
}