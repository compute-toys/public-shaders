const pi = acos(-1.);
const pi2 = pi*2.;

fn ftolinear (val: f32) -> f32 {
    if( val <= 0.04045) {
        return val / 12.92;
    }
    else {
        return pow((val+0.055)/1.055f,2.4);
    }
}

fn v3tolinear(val: vec3f) -> vec3f {
    return vec3f(ftolinear(val.x),ftolinear(val.y),ftolinear(val.z));
}

fn to_rgb(chroma: vec2f) -> vec3f {
    let c709 = mat3x3(0,0,0,0,-0.1873242729,1.8556,1.5748,-0.4681242729,0);
    let rgb = c709 * vec3f(0,chroma);
    let a = clamp(rgb,vec3f(-2.),vec3f(0.));
    return rgb - min(min(a.x,a.y),a.z);
}

fn rot(uv: vec2f, r: f32) -> vec2f {
    let s=sin(r);
    let c=cos(r);
    let mat = mat2x2(c,s,-s,c);
    
    return mat*uv;
}

fn vsphere(x: f32) -> vec2f {
    return vec2f(sin(x*pi),cos(x*pi));
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);
    var uv = (2.*fragCoord-vec2f(screen_size)) / vec2f(screen_size);
    
    let t = time.elapsed;
    
    let scale = 0.4;
    
    uv.y = uv.y*2.5+sin(t*3.+uv.y+uv.x*6.+uv.y*3.)*sin(uv.x*4.)*.1;
    
    let fm = array<f32,4>(1.,60.,3600.,43200.);
    let m = floor(clamp(uv.y+2.5,1.,4.)-1.);
    let r = t%fm[int(m)] * (1./fm[int(m)])*pi2;
        
    let cbcr=mix(rot(vsphere(uv.x)*scale,fract(t*.2)*pi2),rot(vec2(0.0,uv.x*.5+.5),-r)*(scale-pow(m/3.,.8)*.1),float(uv.y>-1.5));
    
    let v = step(fract(uv.y+0.5),0.95);

    let col = v*to_rgb(cbcr);

    textureStore(screen, id.xy, vec4f(v3tolinear(col), 1.));
}
