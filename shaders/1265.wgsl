const PI = 3.1415926535;


fn x2rgb(hexValue: u32) -> vec3f {
    // 提取每个颜色通道的值
    let r = f32((hexValue >> 16) & 0xff) / 255.0; // 右移16位并取低8位
    let g = f32((hexValue >> 8) & 0xff) / 255.0;  // 右移8位并取低8位
    let b = f32(hexValue & 0xff) / 255.0;       // 直接取低8位

    return vec3f(r, g, b);
}

fn plot(uv:vec2f, x:f32 ) -> f32{
    return 1-smoothstep(.0, .03, abs(uv.y - x));
}
fn plot_c(uv:vec2f, x:f32, color:vec3f ) -> vec3f{
    return plot(uv,x) * color;
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Normalised pixel coordinates (from 0 to 1)
    let uv = vec2f(id.xy)/vec2f(screen_size)
            * vec2f(2.) - vec2f(1.);    // 中心点为原点

    // Time varying pixel colour
    // var col = .5 + .5 * cos(time.elapsed + uv.xyx + vec3f(0.,2.,4.));

    // Convert from gamma-encoded to linear colour space
    // col = pow(col, vec3f(2.2));

    var m = vec2f(mouse.pos)/vec2f(screen_size);

    var col=vec3f(0.);

    var exp_x = pow(1.-abs(uv.x),2.);
    var xt=sin((exp_x+sin(time.elapsed)*1.5 ) * (cos(time.elapsed)*2.5 + 5) );
    // xt = (ceil(xt)+floor(xt))*0.5;

    var l0 = plot_c(uv,xt, x2rgb(0x080808));

    var l1 = plot(uv, exp_x);
    var l2 = plot_c(uv, min(cos(PI*uv.x/2.), 1.-abs(uv.x)), x2rgb(0xff0000));
    // 其实这个min(cos, 1-abs)没啥用，直接1-abs就行。
    col+=l0+l1+l2;

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}
