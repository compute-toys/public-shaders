// Very simple algorithm. Explained great here - https://www.youtube.com/watch?v=kbKtFN71Lfs&ab_channel=Numberphile
// You can basically do whatever transformations you want, which makes it really fun.
// The shader is projecting points, then splatting them on the screen.
// Great demo which uses this effect - https://www.youtube.com/watch?v=xLN3mTRlugs&ab_channel=AssemblyTV
// Example/tutorial on doing chaos game in compute.toys - https://compute.toys/view/120

alias iv4 = vec4<i32>;alias iv3 = vec3<i32>;alias iv2 = vec2<i32>;alias uv4 = vec4<u32>;alias uv3 = vec3<u32>;alias uv2 = vec2<u32>;alias v4 = vec4<f32>;alias v3 = vec3<f32>;alias v2 = vec2<f32>;alias m2 = mat2x2<f32>;alias m3 = mat3x3<f32>;alias m4 = mat4x4<f32>;
#define global var<private>
var<workgroup> texCoords: array<array<float2, 16>, 16>;
#define Frame time.frame
#define T (time.elapsed*float(custom.Paused > 0.5) + 5.) 
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


#define COL_CNT 4
global kCols = array<v3, COL_CNT>( 
     vec3(1.,1,1), vec3(0.8,0.4,0.7),
     vec3(1,1,1.)*1.5, vec3(1,1,1.)*1.5
);

fn mix_cols(_idx: float)->v3{
    let idx = _idx%1.;
    var cols_idx = int(idx*float(COL_CNT));
    var fract_idx = fract(idx*float(COL_CNT));
    fract_idx = smoothstep(0.,1.,fract_idx);
    //return oklab_mix( kCols[cols_idx], kCols[(cols_idx + 1)%COL_CNT], fract_idx );
    return mix( kCols[cols_idx], kCols[(cols_idx + 1)%COL_CNT], fract_idx );
}


#storage hist_atomic array<atomic<u32>>

#storage h array<u32>

fn projParticle(_p: v3) -> v3{
    var p = _p; // you still can't modify args in wgsl???
    // p *= rotY(muv.x*2.);
    // let dpp = dot(p,p) + 0.5 + sin(T)*0.4;

    let t = T*1.;

    p += v3(  
        sin(t+ 0.1)*0.1,
        sin(t)*0.1,
        sin(t)*0.2,
    )*1.;
    // p.z += 4.7;
    // p *= rotZ(pow(sin_add(T),2.)*pi*2. * float(floor(T/pi/2.)%4 > 2.) );
    // p /= p.z*0.2;
    p *= 1. + sin(t*0.9 + sin(t*0.8))*0.05;
    // p.x /= dpp;
    // p.y /= dpp;
    p.z = _p.z;
    p.x /= R.x/R.y;
    return p;
}

fn t_spherical(p: v3, rad: float, offs: v3) -> v3{
    return p/(dot(p,p)*rad + offs);
}

#workgroup_count Splat 64 64 12
@compute @workgroup_size(256, 1,1)
fn Splat(@builtin(global_invocation_id) id: uint3) {
    let Ru = uint2(textureDimensions(screen));
    if (id.x >= Ru.x || id.y >= Ru.y) { return; }
    R = v2(Ru); U = v2(id.xy); muv = (v2(mouse.pos) - 0.5*R)/R.y;
    
    seed = hash_u(id.x + hash_u(Ru.x*id.y*200u)*20u + hash_u(id.x)*250u + hash_u(id.z)*250u );
    seed = hash_u(seed);

    let particleIdx = id.x;
    
    let iters = 60 - int(sin_add(T)*30.);
    
    var t = T + sin(T*4.)*0.15*0. - hash_f()*1./30.;
    
    let md = 5.0;
    
    var env = ((floor(t))%md);
    var env_next = ((floor(t + 1))%md);
    var env_fr = fract(t);

    env = mix(
        env,
        env_next,
        smoothstep(0.,1.,
            smoothstep(0.,1.,
                smoothstep(0.,1.,
                    smoothstep(0.,1.,
                        env_fr
                    )
                )
            )
        )
    );

    var p = hash_v3()*2. - 1.;
    p *= 0.;

    var focusDist = (custom.DOF_Focal_Dist*2. - 1.)*5.;
    if(mouse.click > 0){
        focusDist = muv.y;

    }
    // let focusDist = (custom.DOF_Focal_Dist*2. - 1.)*5.;
    let dofFac = 1./vec2(R.x/R.y,1.)*custom.DOF_Amount;

    
    let vert_cnt = 4.0 + env;
    let verts = array<vec2f,5>(
        vec2f(sin(1./vert_cnt*tau),cos(1./vert_cnt*tau)),
        vec2f(sin(2./vert_cnt*tau),cos(2./vert_cnt*tau)),
        vec2f(sin(3./vert_cnt*tau),cos(3./vert_cnt*tau)),
        vec2f(sin(4./vert_cnt*tau),cos(4./vert_cnt*tau)),
        vec2f(sin(5./vert_cnt*tau),cos(5./vert_cnt*tau))
    );

    for(var i = 0; i < iters; i++){
        let r = hash_f();

        // let next_p = verts[int(r*float(vert_cnt)*0.99)];
        // let f_idx = floor(r*vert_cnt*(0.99 + custom.A));
        // let f_idx = floor(r*vert_cnt*(1.2 + sin(T)*0.2));
        let f_idx = floor(r*vert_cnt*(0.99));
        let next_p = vec3f(
            sin(f_idx/vert_cnt*tau*(1. + env*1.)+ sin(T)*0.4*float(floor(T/pi + 1)%3 > 1)),
            cos(f_idx/vert_cnt*tau + sin(T)*0.4*float(floor(T/pi)%3 > 1.)),
            sin(f_idx/vert_cnt*tau*4.4+ env + sin(T*1. + 4.)*1.),
        );

        p = sin(p*1. + next_p*2. - env*0.1);
        if(hash_f() < 0. + sin_add(T + 4.)*0.4){
            p += p*1.;

            p *= rotZ(t*0.1);
            p *= rotY(t*0.1);
        }

        if(hash_f() < 0.){
            p += next_p*1.;
            p.z -= 0.1;
            // p += p*2.;
            // p *= 0.;
            // p.z = 2.
            // 4;
            // p += normalize(hash_v3()*2. - 1.)*3.;
        }
            // let next_p = sin()

        // p = mix(p,next_p,0.4 + sin(T)*0.);

        
        var q = projParticle(p);
        var k = q.xy;

        k += sample_disk()*abs(q.z - focusDist + sin(t + sin(t) + f_idx*0.5))*0.05*dofFac;
        
        let uv = k.xy/2. + 0.5;
        let cc = uv2(uv.xy*R.xy);
        let idx = cc.x + Ru.x * cc.y;
        if ( 
            // q.z > -20.
            uv.x > 0. && uv.x < 1. 
            && uv.y > 0. && uv.y < 1. 
            && idx < uint(Ru.x*Ru.y)
            ){     
            atomicAdd(&hist_atomic[idx],1);
            atomicAdd(&hist_atomic[idx + Ru.x*Ru.y],uint(r*2400.));
        }
    }
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    let res = uint2(textureDimensions(screen));
    if (id.x >= res.x || id.y >= res.y) { return; }

    R = v2(res);
    U = v2(id.xy);
    let U = float2(float(id.x) + .5, float(res.y - id.y) - .5);

    
    let hist_id = id.x + uint(R.x) * id.y;

    var col = float(atomicLoad(&hist_atomic[hist_id]))*vec3(3);
    
    var col_pal = float(atomicLoad(&hist_atomic[hist_id + res.x*res.y]))*vec3(3);
    
    kCols[1] *= rotY(1.-0.347*log(col.x*1.)*4. + col_pal.x*0.0001*(1. - sin_add(T)*0.5));
    // tonemap
    let sc = 124452.7;
    col = log( col*(0.3+custom.B))/ log(sc);


    // col = abs(col);
    // col = col*mix_cols(col.x*(1. + sin(T)*0.4));
    col = pow(col,v3(1. - sin_add(T)*0.))*mix_cols(col.x*(1.));
    // col += 0.01;
    // col = mix_cols(log(col_pal.x)*0.5);
    

    //col = smoothstep(v3(0.),v3(1.),col*mix_cols(col.x*1.));

    col = max(col,v3(0.1));
    col = pow((col), float3(1./0.25454545))*4.;
    col = min(col,v3(0.6));
    
    col = clamp(col,v3(0.002),v3(0.8));
    textureStore(screen, int2(id.xy), float4(col, 1.));
    

    // clear flame
    // hist[id.x + uint(R.x) * id.y] = 0;
    atomicStore(&hist_atomic[hist_id],0);
    atomicStore(&hist_atomic[hist_id + res.x*res.y],0);
}
