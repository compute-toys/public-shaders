fn refr( // Refraction fix by ahs3n : https://compute.toys/view/1893
    i: vec3f, // Incoming vector
    n: vec3f, // Normal vector, pointing from high to low ior
    mu: f32,  // Index of refraction ratio high/low
    tir: ptr<function, bool> // Total internal reflection
) -> vec3f {

    *tir = false;

    var t: vec3f;
    if (dot(i, n) > 0.){
        
        if (dot(i, n) > sqrt(mu*mu - 1.)/mu){
            t = refract(i, -n, mu);
        } else {
            t = reflect(i, n);
            *tir = true;
        }

    } else {
        t = refract(i, n, 1./mu);
    }

    return t;
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {

    let r = textureDimensions(screen);
    if (id.x >= r.x || id.y >= r.y) { return; }
    let U = vec2f(f32(id.x) + .5, f32(r.y - id.y) - .5);
    let uv = U / vec2f(r);
    let cuv = (2.*U-vec2f(r)) / float(r.y);

    var O = vec4f(select(custom.ior, 1., cuv.y > 0))/5.;
    O = pow(O, vec4f(2.2));

    var tir = false;

    var ray1 = vec3f(abs(cos(time.elapsed*custom.sweepSpeed)), sin(time.elapsed*custom.sweepSpeed), 0);
    var ray2 = refr(ray1, vec3f(0, 1, 0), custom.ior, &tir);

    var ray = select(ray1, ray2, cuv.x > 0);

    let d = abs(dot(cuv, vec2f(ray.y, -ray.x)));

    O = mix(O, 
        select(vec4f(1.), select(vec4f(.2, .4, .9, 1.), vec4f(.8, .5, .2, 1.), tir), cuv.x > 0), 
        1.-smoothstep(0., 5./f32(r.y), d)
    );

    textureStore(screen, id.xy, O);
}
