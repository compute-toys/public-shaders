const PI: f32 = 3.14159265359;

fn curve(x: f32, a: f32, b: f32) -> f32 {
    let y: f32 = smoothstep(a, b, x) * smoothstep(b, a, x);
    return pow(y, 0.08);
}


fn drawLeaf(uv: vec2<f32>, ls: f32, le: f32, lw: f32, ang: f32, time: f32, isLeft: bool) -> vec3<f32> {
    let cosA: f32 = cos(ang);
    let sinA: f32 = sin(ang);
    let rotuv: vec2<f32> = vec2(cosA * uv.x - sinA * uv.y, sinA * uv.x + cosA * uv.y);
    let angle: f32 = atan2(rotuv.y, rotuv.x);
    let radius: f32 = length(rotuv);
    let angleMod: f32 = angle % (2.0 * PI);
    let normalang: f32 = angleMod / (2.0 * PI);
    let leafRadius: f32 = mix(ls, le, curve(normalang, 0.5 - (lw / 2.0), 0.5 + (lw / 16.0)));
    let withinLeaf: bool = radius >= ls && radius <= leafRadius;
    let firstcol: f32 = smoothstep(ls, le, radius);
    let colorShift: f32 = 0.5 + 0.5 * sin(time * 2.0 * PI); 
    var animcol: f32 = firstcol;
    if (isLeft) {
        animcol = mix(firstcol, 1.0 - firstcol, colorShift);
    } else {
        animcol = mix(1.0 - firstcol, firstcol, colorShift);
    }

    var first_color: vec3<f32>;
    var ended_color: vec3<f32>;
    if (isLeft) {
        first_color = vec3<f32>(1.0, 1.0, 0.0);
        ended_color = vec3<f32>(0.0, 0.0, 0.0);
    } else {
        first_color = vec3<f32>(0.0, 0.0, 0.0);
        ended_color = vec3<f32>(1.0, 1.0, 0.0);
    }

    var leafColor: vec3<f32>;
    if (withinLeaf) {
        leafColor = mix(first_color, ended_color, animcol);
    } else {
        leafColor = vec3<f32>(1.0, 1.0, 1.0);
    }

    return leafColor;
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let fragCoord = vec2<f32>(f32(id.x) + 0.5, f32(screen_size.y - id.y) - 0.5);
    var uv = 1.8 * (fragCoord - 0.5 * vec2<f32>(screen_size)) / f32(screen_size.y);
    uv.y += 0.25;
    var col = vec3<f32>(1.0, 1.0, 1.0);
    let n = 32;

    var uvLeft = uv - vec2<f32>(-0.80, 0.25);
    var uvRight = uv - vec2<f32>(0.80, 0.25);

    for (var i: i32 = 0; i < n; i = i + 1) {
        let oriant = 50.0 * PI * f32(i) / f32(n);
        let lefone = drawLeaf(uvLeft, 0.2, 0.7, 0.05, oriant, time.elapsed  / 5.0, true);
        let rightone = drawLeaf(uvRight, 0.2, 0.7, 0.05, oriant, -time.elapsed  / 5.0, false);
        col = min(col, lefone);
        col = min(col, rightone);
    }

    textureStore(screen, id.xy, vec4f(col, 1.));
}