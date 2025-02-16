
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);
    let o = (vec2f(id.xy) - vec2f(screen_size.xy)*.5)/ vec2f(screen_size.yy);
    let om = (vec2f(mouse.pos.xy) - vec2f(screen_size.xy)*.5)/ vec2f(screen_size.yy);
    let n_t = time.elapsed;
    let od =(om-o);
    var o2 = o*33.;
    o2.x *= o2.x*.01;
    let n_id = 2.*o2.x;
    o2.y += sin(n_id);
    let n_id2 = o2.x+od.x;
    let n_id3 = o2.y+od.y;
    var n = sin(n_id2) * sin(n_id3);
    n = sin(n*3.+n_t+n_id+n_id2+n_id3);
    let n1 = sin(n*3.);
    let n2 = sin(n*6.);
    let n3 = sin(n*9.);
    let o_col = vec3f(n1,n2,n3);


    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(o_col, 1.));
}
