fn tri2(x: f32, f: f32) -> f32 {
    let vx = x%(1./f*2.) + .25/f;
    let vf = f*2.;
    let e = floor(vx*vf)%2.;

    return e*(1.-fract(vf*vx)*2.) + (1.-e)*(-1.+fract(vf*vx)*2.);
}

fn center(uv: vec2f) -> f32 {
    let c = 31.;
    let y = ((uv.y*0.5+0.5)-0.25/c)*c;
    return 1.-max(smoothstep(0.002,0.006,length(uv.x)),smoothstep(0.45,0.5,abs(y%1.)));
}

fn cube(origin: vec2f, uv: vec2f, size: vec2f) -> f32
{
    return 1.-max(smoothstep(size.x-0.005,size.x+0.005,length(uv.x-origin.x)),smoothstep(size.y-0.005,size.y+0.005,abs(uv.y-origin.y)));
}

fn scan(uv: vec2f) -> f32
{
    return sin((time.elapsed*.4-uv.y)*300.)*(sin(time.elapsed*45.+uv.y)*0.02+0.03);
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    let fragCoord = vec2f(f32(id.x) + .5, f32(id.y) + .5);
    let aspect = vec2f(f32(screen_size.x)/f32(screen_size.y),1.);
    var uv = (2.*fragCoord-vec2f(screen_size.xy)) / f32(screen_size.y);
    
    uv *= pow(length(uv),0.07);
    
    let co = vec2f(tri2(time.elapsed,0.3),tri2(time.elapsed,0.335));
    let cs = vec2f(0.04);
    let cso = (co*aspect)*vec2f(1.-(2.*cs.x)/aspect.x,1.-cs.y);
    
    let ps   = vec2f(0.04,0.2);
    let ppso =(vec2f(-1.,co.y*pow((-1.*clamp(co.x,-1.,0.)),2.))*aspect)*vec2(1.-ps.x/aspect.x,1.-ps.y);
    
    let pcso = (vec2f(1.,co.y)*aspect)*vec2(1.-ps.x/aspect.x,1.-ps.y);

    let c = cube(cso,uv,cs)+center(uv)+cube(ppso,uv,ps)+cube(pcso,uv,ps);
    var col = vec3f(clamp(c*0.90+0.05+scan(uv),0.,1.));
    let b = step(abs(uv.x),aspect.x) * step(abs(uv.y),aspect.y);
    col=mix(vec3f(0),col,b);
    col = pow(col, vec3f(1.8));
    textureStore(screen, id.xy, vec4f(col, 1.));
}
