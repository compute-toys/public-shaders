#storage c array<atomic<u32>>

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) g: vec3u) {
    let s = textureDimensions(screen);
    if (g.x >= s.x || g.y >= s.y) { return; }
    
    let u = (vec2f(g.xy) * 2. - vec2f(s)) / f32(s.y);
    var o = vec3f(0);
    let t = time.elapsed;
    let idx = g.y * s.x + g.x;
    
    atomicStore(&c[idx], 0u);
    
    for (var i = 0; i < 22; i++) {
        let p = floor(f32(i) * .1 * normalize(vec3f(u, .8)).xy * 22.) / 44.;
        let a = atan2(p.y, p.x) + length(p) * 6.;
        let r = length(p) + sin(a * 4.) * .4;
        let d = abs(fract(r * 25. + t) - .5) - .5 + sin(a) * .02;
        let h = r * 8. + a * 2. + t;
        
        o += vec3f(1., .5 + .5 * sin(h), .2 + .3 * sin(h + h)) 
           * exp(-abs(d) * 8.) * max(2. - r * 3., 0.) 
           * step(length(p * f32(s.y) * .5 + vec2f(s) * .5 - vec2f(g.xy)), 3.);
        
        atomicAdd(&c[idx], u32(step(length(p * f32(s.y) * .5 + vec2f(s) * .5 - vec2f(g.xy)), 3.)));
    }
    
    textureStore(screen, g.xy, vec4f(pow(max(tanh(o * .66), vec3f(.01)), vec3f(1.2)), 1.));
}