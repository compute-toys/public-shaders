
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
    var col = vec3f(1.0-uv.y);

    var pos = vec2i(i32(id.x),i32(screen_size.y/2));

    var dify = vec2i(0,0);
    var roty = f32(uv.y*f32(custom.splurg));
    dify.y = i32(sin(roty)*uv.y*custom.splurg*uv.y*uv.y);
    dify.x = i32(cos(roty)*uv.y*custom.splurg*uv.y*uv.y);
    var rol = sin(time.elapsed);
    var difa = vec2i(0,0);
    var difb = vec2i(0,0);
    var difc = vec2i(0,0);
    var difd = vec2i(0,0);
    var amp = f32(custom.wang*sin(uv.x*3.1415926535));
    var cent = f32(uv.x-0.5)+sin(time.elapsed/4)/4;
    var rota = f32(sin(cent*custom.wibble*sin(time.elapsed*custom.wibble/custom.floom)));
    var rotb = f32(sin(cent*custom.wobble*sin(time.elapsed*custom.wobble/custom.floom)));
    var rotc = f32(sin(cent*custom.wabble*sin(time.elapsed*custom.wabble/custom.floom)));
    var rotd = f32(sin(cent*custom.webble*sin(time.elapsed*custom.webble/custom.floom)));
    
    //dif.y += i32(20.*sin(f32(pos.x)/20.));
    difa.y = i32(cos(rota)*amp*2);
    difa.x = i32(sin(rota)*amp*2);
    difb.y = i32(sin(rotb)*amp);
    difb.x = i32(cos(rotb)*amp);
    difc.y = i32(sin(rotc)*amp/2);
    difc.x = i32(cos(rotc)*amp/2);
    difd.y = i32(sin(rotd)*amp/3);
    difd.x = i32(cos(rotd)*amp/3);
    col.r += rota/5.;
    col.g += rotb/5.;
    col.b += rotc/5.;
    var dife = difa*difb*difc*difd;
    pos += difa;
    pos += difb;
    pos += difc;
    pos += dife/i32(custom.wurzle);
    pos += dify;
    
    col=col*col*custom.wazz*((amp)/20);
    col = col * (col.r+col.g+col.b) * (col.r+col.g+col.b);

    textureStore(screen, vec2u(pos), vec4f(col, 1.));
}
