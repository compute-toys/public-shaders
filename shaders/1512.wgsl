// Very simple algorithm. Explained great here - https://www.youtube.com/watch?v=kbKtFN71Lfs&ab_channel=Numberphile
// You can basically do whatever transformations you want, which makes it really fun.
// The shader is projecting points, then splatting them on the screen.
// Great demo which uses this effect - https://www.youtube.com/watch?v=xLN3mTRlugs&ab_channel=AssemblyTV
// Example/tutorial on doing chaos game in compute.toys - https://compute.toys/view/120

alias iv4 = vec4<i32>;alias iv3 = vec3<i32>;alias iv2 = vec2<i32>;alias uv4 = vec4<u32>;alias uv3 = vec3<u32>;alias uv2 = vec2<u32>;alias v4 = vec4<f32>;alias v3 = vec3<f32>;alias v2 = vec2<f32>;alias m2 = mat2x2<f32>;alias m3 = mat3x3<f32>;alias m4 = mat4x4<f32>;
#define global var<private>
var<workgroup> texCoords: array<array<float2, 16>, 16>;
#define Frame time.frame
#define T (time.elapsed + 60.) 
#define Tb (time.elapsed) 
#define pi acos(-1.)
#define tau (acos(-1.)*2.)
global R: v2;
global U: v2;
global seed: uint;
global muv: v2;
fn rot(a: float)-> m2{ return m2(cos(a), -sin(a),sin(a), cos(a));}
fn rotX(a: float) -> m3{
    let r = rot(a); return m3(1.,0.,0.,0.,r[0][0],r[0][1],0.,r[1][0],r[1][1]);
}
fn rotY(a: float) -> m3{
    let r = rot(a); return m3(r[0][0],0.,r[0][1],0.,1.,0.,r[1][0],0.,r[1][1]);
}
fn rotZ(a: float) -> m3{
    let r = rot(a); return m3(r[0][0],r[0][1],0.,r[1][0],r[1][1],0.,0.,0.,1.);
}
fn hash_u(_a: uint) -> uint{ var a = _a; a ^= a >> 16;a *= 0x7feb352du;a ^= a >> 15;a *= 0x846ca68bu;a ^= a >> 16;return a; }
fn hash_f() -> float{ var s = hash_u(seed); seed = s;return ( float( s ) / float( 0xffffffffu ) ); }
fn hash_v2() -> v2{ return v2(hash_f(), hash_f()); }
fn hash_v3() -> v3{ return v3(hash_f(), hash_f(), hash_f()); }
fn hash_v4() -> v4{ return v4(hash_f(), hash_f(), hash_f(), hash_f()); }

fn sin_add(a: float)->float{return sin(a)*0.5 + 0.5;}

fn sample_disk() -> v2{
    let r = hash_v2();
    return v2(sin(r.x*tau),cos(r.x*tau))*sqrt(r.y);
}

#storage hist_atomic array<atomic<u32>>

#storage h array<u32>

fn projParticle(_p: v3) -> v3{
    var p = _p*.9; 
    p.x /= R.x/R.y;
    return p;
}

#workgroup_count Splat 4096 1 1
@compute @workgroup_size(256, 1, 1)
fn Splat(@builtin(global_invocation_id) id: uint3) {
    let Ru = uint2(textureDimensions(screen));
    R = v2(Ru);
    muv = (v2(mouse.pos) - 0.5*R)/R.y;
    
    seed = id.x;

    let iters = 180 - int(sin_add(T)*100.);
    
    var t = 0.3 * (T + hash_f()*1./30.);
    
    let md = 5.0;
    
    var env = ((floor(t))%md);
    var env_next = ((floor(t + 1))%md);
    var env_fr = fract(t);

    t += sin(t*3.)*0.25;

    env = mix(
        env,
        env_next,
        smoothstep(0.,1.,max(5.*env_fr - 4., 0.))
    );

    var p = vec3(0.);

    // var focusDist = (custom.DOF_Focal_Dist*2. - 1.)*5.;
    var focusDist = muv.y*4.0;
    let dofFac = 1./vec2(R.x/R.y,1.)*custom.DOF_Amount*2.;
    
    let vert_cnt = 5.0 + 0.1*env;

    for(var i = 0; i < iters; i++){
        let r = hash_f();

        let f_idx = floor(r*vert_cnt*(0.99));
        let next_p = vec3f(
            sin(f_idx/vert_cnt*tau*(1. + env*0.1)+ sin(t)*0.4*float(floor(t/pi + 1)%3 > 1)),
            cos(f_idx/vert_cnt*tau + sin(t)*0.4*float(floor(t/pi)%3 > 1.)),
            sin(f_idx/vert_cnt*tau*4.4+ 0.1*env + sin(t*1. + 4.)*1.),
        );


        p = sin(p*1. + next_p*2. - env*0.1);

        if(i%2 == 0 && r < .5){
        // if(i%2 == 0){
        // if(r < 0.2){
            p += p*0.3;
            p *= rotZ(t*0.01);
            p *= rotY(t*0.01);
        }

        // p = mix(p,next_p,0.3 + sin(T)*0.15 );

        var q = p;
        q += vec3(0, 0, f_idx*0.2);

        q *= rotY(-t*0.01);
        q *= rotX(-t*0.01);
        // if(mouse.click > 0){
            q *= rotY(-muv.x*0.3);
            q *= rotX(-muv.y*0.2);
        // }
        q = projParticle(q);
        
        var k = q.xy;

        let d = q.z - focusDist;
        k += sample_disk()*abs(d)*0.05*dofFac;
        
        let uv = k.xy/2. + 0.5;
        let cc = uv2(uv.xy*R.xy);
        let idx = cc.x + Ru.x * cc.y;
        if ( 
            uv.x > 0. && uv.x < 1. 
            && uv.y > 0. && uv.y < 1. 
            && idx < uint(Ru.x*Ru.y)
            ){     
            atomicAdd(&hist_atomic[idx],uint((1.-r)*100.));
            atomicAdd(&hist_atomic[idx + Ru.x*Ru.y],uint(r*100.));
        }
    }
}

fn aces_tonemap(color: v3) -> v3 {
    const m1 = mat3x3f(
        0.59719, 0.07600, 0.02840,
        0.35458, 0.90834, 0.13383,
        0.04823, 0.01566, 0.83777
	);
	const m2 = mat3x3f(
        1.60475, -0.10208, -0.00327,
        -0.53108,  1.10813, -0.07276,
        -0.07367, -0.00605,  1.07602
	);

	var v = m1 * color;    
	var a = v * (v + 0.0245786) - 0.000090537;
	var b = v * (0.983729 * v + 0.4329510) + 0.238081;
	return m2 * (a / b);
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    let res = uint2(textureDimensions(screen));
    if (id.x >= res.x || id.y >= res.y) { return; }
    
    let hist_id = id.x + uint(res.x) * id.y;

    var col = float(atomicLoad(&hist_atomic[hist_id]))*vec3(0.0,0.5,1.0);
    var col_pal = float(atomicLoad(&hist_atomic[hist_id + res.x*res.y]))*vec3(1.0,0.6,0.2);
    
    col = (col + col_pal) * 0.00003;

    col = mix(col, 1.3*exp(-4*col), smoothstep(0.4,0.6,custom.Absorb));
    col *= vec3(0.9, 1.0, 0.95);
    col = aces_tonemap(col);

    col += vec3(0.0, 0.002, 0.006);

    textureStore(screen, int2(id.xy), float4(col, 1.));
    

    // clear flame
    atomicStore(&hist_atomic[hist_id],0);
    atomicStore(&hist_atomic[hist_id + res.x*res.y],0);
}
