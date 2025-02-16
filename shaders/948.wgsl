
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {    
    let screen_size = textureDimensions(screen);
    if any(id.xy >= screen_size) {
        return;
    }
    var cr = passLoad(0, int2(id.xy), 0);
    var bg = vec4f(1);   
    let sizeFactor = float(max(screen_size.x, screen_size.y)) * 0.1;
    let size = max(custom.size * sizeFactor, 5);
    let dist = distance(vec2f(id.xy), vec2f(mouse.pos.xy));
    if (dist < size && mouse.click > 0) { // inside
        
        let a = (1-dist/size) * custom.opacity;
        cr = vec4(custom.r, custom.g, custom.b, a);
        bg = passLoad(0, int2(id.xy), 0);
    }
    cr = vec4f(cr.xyz*cr.a + bg.xyz*(1.0 - cr.a), 1.0);
    passStore(0, int2(id.xy), cr);
    textureStore(screen, id.xy, cr);
}
