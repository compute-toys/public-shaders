const PI =3.1415926535;
const PI2=6.28318530718;
const E = 2.718281828;

// 💡hsv，v指明度，色轮不会整体过曝；hsl，l指亮度，更符合人对色彩的认知。
// 各有应用场景。

// 画线函数。圆心在左上角，x轴向右，y轴向下。x、y都属于0~1。
fn plot(uv:vec2f, x:f32 ) -> f32{
    return 1-smoothstep(.0, .03, abs(uv.y - x));
}
fn plot_(uv:vec2f, x:f32, color:vec3f ) -> vec3f{
    return plot(uv,x) * color;
}

//  🐛buggy, don't use! 红黄绿青，然后没回到蓝色、红色。
// Function from Iñigo Quiles
//  https://www.shadertoy.com/view/MsS3Wc
fn hsv2rgb_ ( c:vec3f ) -> vec3f {
    var rgb = clamp(
        abs(
        (c.x*6.0+vec3f(0.0,4.0,2.0)%6.0)-3.0
        )-1.0,
        vec3f(0.0), vec3f(1.0) );
    rgb = rgb*rgb*(3.0-2.0*rgb);
    return c.z * mix( vec3(1.0), rgb, c.y);
}

// 💡 https://www.shadertoy.com/view/XsVSWR
fn hsv2rgb (h:vec3f) -> vec3f {
    let k = vec3f(3,2,1)/3.;
    let p = abs(fract(h.rrr + k.rgb) * 6. - vec3f(3));
    return h.b * mix(k.rrr, clamp(p - k.rrr, vec3f(0), vec3f(1)), h.g);
}

// 💡 https://www.shadertoy.com/view/XsVSWR
fn hsl2rgb (h:vec3f) -> vec3f {
    let h6 = h.x*6.;
    let rgb = clamp(vec3f(
        abs(h6-3.)-1.,
        2.-abs(h6-2.),
        2.-abs(h6-4.)
        ),vec3f(0),vec3f(1));
    let c = (1.-abs(2.*h.z-1.))*h.y;
    return (rgb-.5)*c+h.z;
}

// 💡
fn rgb2hsl( c:vec3f ) -> vec3f {
    let K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    let p = mix(vec4(c.bg, K.wz),
                 vec4(c.gb, K.xy),
                 step(c.b, c.g));
    let q = mix(vec4(p.xyw, c.r),
                 vec4(c.r, p.yzx),
                 step(p.x, c.r));
    let d:float = q.x - min(q.w, q.y);
    let e = 1.0e-10;
    return vec3f(abs(q.z + (q.w - q.y) / (6.0 * d + e)),
                d / (q.x + e),
                q.x);
}

fn x2rgb(hexValue: u32) -> vec3f {
    // 提取每个颜色通道的值
    let r = f32((hexValue >> 16) & 0xff) / 255.0; // 右移16位并取低8位
    let g = f32((hexValue >> 8) & 0xff) / 255.0;  // 右移8位并取低8位
    let b = f32(hexValue & 0xff) / 255.0;       // 直接取低8位

    return vec3<f32>(r, g, b);
}

// 测试函数，无需理会
fn step_ani(x:float,min:float,max:float) -> float {
    let A=max-min;
    let X=x*A+min;
    let ani=step(X,(sin(2.5*time.elapsed)*.5+.5)*A+min);
    return X*ani;   // X*ani
}
fn test(x:float,min:float,max:float) -> vec3f {
    return vec3f(step_ani(x,min,max));
}
fn test3(c:vec3f,min:float,max:float) -> vec3f {
    return vec3f(step_ani(c.x,min,max),step_ani(c.y,min,max),step_ani(c.z,min,max));
}


fn hsv_plot(angle:float, radius:float, uv: vec2f) -> vec3f {
    var col = hsv2rgb(vec3f((angle/PI2),radius,.5*sin(2*time.elapsed)+.5));
    let lr=plot_(uv,hsv2rgb(vec3f(uv.x)).r,x2rgb(0xff0000));
    let lg=plot_(uv,hsv2rgb(vec3f(uv.x)).g,x2rgb(0x00ff00));
    let lb=plot_(uv,hsv2rgb(vec3f(uv.x)).b,x2rgb(0x0000ff));
    col += lr*custom.r+lg*custom.g+lb*custom.b;
    return col;
}
fn hsl_plot(angle:float, radius:float, uv: vec2f) -> vec3f {
    var col = hsl2rgb(vec3f((angle/PI2),radius,.5*sin(2*time.elapsed)+.5));
    let lr=plot_(uv,hsl2rgb(vec3f(uv.x)).r,x2rgb(0xff0000));
    let lg=plot_(uv,hsl2rgb(vec3f(uv.x)).g,x2rgb(0x00ff00));
    let lb=plot_(uv,hsl2rgb(vec3f(uv.x)).b,x2rgb(0x0000ff));
    col += lr*custom.r+lg*custom.g+lb*custom.b;
    return col;
}
// ryb练习完成🎉
fn hsl_ryb_plot(angle:float, radius:float, uv: vec2f) -> vec3f {
    let a=(pow((angle/PI2),1.5))*PI2;
    return hsl_plot(a, radius ,uv);
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Normalised pixel coordinates 0~1, *2-1 for -1~1; 
    let uv = vec2f(id.xy) / vec2f(screen_size);
    // var col = vec3f();

    // 💡mix(colA,colB,0~1)
    // var col = mix(vec3f(.15,.15,.9),vec3f(1.,0.8,0.2),sin(3*time.elapsed));

    // 💡使用极坐标
    var toCenter:vec2f = uv-vec2f(.5);
    var radius = length(toCenter)*2.;
    var angle = atan2(toCenter.y,toCenter.x)+PI;    // atan ∈ [-π,π]
    // var col = test(angle/PI2,0,1);
    
    let col = mix(hsv_plot(angle,radius,uv),hsl_plot(angle,radius,uv),custom.hsl);
    // let col = mix(hsl_ryb_plot(angle,radius,uv),hsl_plot(angle,radius,uv),custom.hsl);

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}
