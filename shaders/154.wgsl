alias f = float;alias iv4 = vec4<i32>;alias iv3 = vec3<i32>;alias iv2 = vec2<i32>;alias uv4 = vec4<u32>;alias uv3 = vec3<u32>;alias uv2 = vec2<u32>;alias v4 = vec4<f32>;alias v3 = vec3<f32>;alias v2 = vec2<f32>;alias m2 = mat2x2<f32>;alias m3 = mat3x3<f32>;alias m4 = mat4x4<f32>;
#define global var<private>
var<workgroup> texCoords: array<array<float2, 16>, 16>;
#define Frame time.frame
#define T (time.elapsed + 29 + 276+12) 
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
fn hash_f_s(s: float) -> float{ return ( float( hash_u(uint(s*float( 0xffffffffu/10000 ))) ) / float( 0xffffffffu ) ); }
fn hash_v2_s(s: float) -> v2{ return v2(hash_f_s(s), hash_f_s(s)); }
fn hash_v3_s(s: float) -> v3{ return v3(hash_f_s(s), hash_f_s(s), hash_f_s(s)); }
fn hash_v4_s(s: float) -> v4{ return v4(hash_f_s(s), hash_f_s(s), hash_f_s(s), hash_f_s(s)); }

fn quant(a: float, b: float)->float{return floor(a*b)/b;}

fn sample_disk() -> v2{
    let r = hash_v2();
    return v2(sin(r.x*tau),cos(r.x*tau))*sqrt(r.y);
}
fn pmod(p:v3, amt: float)->v3{ return (v3(p + amt*0.5)%amt) - amt*0.5 ;}


#define COL_CNT 4
global kCols = array<v3, COL_CNT>( 
     vec3(1.,1,1), vec3(1.,0.,1.),
     vec3(1,1.,0.1)*1., vec3(0.5,1,1.)*1.5
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


fn projParticle(_p: v3) -> v3{
    var p = _p; // you still can't modify args in wgsl???
    if((T%8.) < 2.){
        p = sin(p*5. + T);
        p *= rotY(T*2. + sin(T*2.) );
        if(hash_f_s(floor(T + 1555))>0.5){
            p /= dot(p,p) - 0.2;
            p += 0.4;
            p /= dot(p,p) - 0.2;
            p *= rotY(T );
        }
    } else{
        p *= rotY(quant(hash_f_s(floor(T + 50)),4.)*pi*2.);
    }
    if(mouse.click > 0){
        p *= rotY(muv.x*2.);
        p *= rotX(muv.y*2.);
    }
    p.z += 4.;
    let z = p.z;
    p /= p.z*0.4;
    p.z = z;
    p.x /= R.x/R.y;
    return p;
}

fn t_spherical(p: v3, rad: float, offs: v3) -> v3{
    return p/(dot(p,p)*rad + offs);
}

fn g_env(_t:float, offs: float)->float{
    let t = _t + offs;
    return _t + sin(_t);
}

fn hash_f_p_neg(prob: f)->f{
    return float(hash_f() < prob)*2 - 1;
}

#workgroup_count Splat 64 64 2
@compute @workgroup_size(16, 1)
fn Splat(@builtin(global_invocation_id) id: uint3) {
    let Ru = uint2(textureDimensions(screen));
    if (id.x >= Ru.x || id.y >= Ru.y) { return; }
    R = v2(Ru); U = v2(id.xy); muv = (v2(mouse.pos) - 0.5*R)/R.y;
    
    seed = hash_u(id.x + hash_u(Ru.x*id.y*200u)*20u + hash_u(id.x)*250u);
    seed = hash_u(seed);

    let particleIdx = id.x;
    
    let iters = 460;
    
    var t = T - hash_f()*(1. + (max(sin(T),0.))*0.1)/30.;
    var env = t + sin(t)*0.8;
    var envb = sin(t*0.45);

    var p = sin(v3(id*uv3(1,2,4)))*0.;


    let focusDist = (custom.DOF_Focal_Dist*1.)*25.;
    let dofFac = 1./vec2(R.x/R.y,1.)*custom.DOF_Amount;
    var pal_idx = 0.;
    
    let tr = hash_v3();
    let rb = hash_v3_s(floor(T));

    // let dir = tr.y < 0.2 ? 0.6 : 0.7;
    var dir = v3(0);
    if(tr.y < 0.5){
        dir = v3(0,-1,0);
    } else {
        dir = v3(1,0,0);
    }

    for(var i = 0; i < iters; i++){
        let r = hash_f();
        p += dir*(0.002*(1. + 20.*rb.x) + sin(float(id.x))*0.01);
        if(tr.y < 0.5){
            if(i%10 == 0){
                dir += p;
            }
            if(i%50 == 10){
                dir *= rotX(0.25*pi*hash_f_p_neg(0.5));
            }
            if(i == 20 + int(sin(T)*45)){
                dir *= rotZ(0.25*pi*hash_f_p_neg(0.5) + T);
                if(rb.z > 0.5){
                    p *= rotZ(rb.x);
                    // p += T;
                }

            }
            
            if(rb.x > 0.7){
                p/= (dot(p,p) - 0.1 + sin(T)*0.5 + 0.5);
                p *= 1. + sin(T + sin(T));
            }
            // p.y -= 0.001;
        } else if(tr.y < 1.){
            if(i == 20){
                dir *= rotY(0.25*pi);
            }
            if(i%50 == 50 + int(sin(T*0.6)*45)){
                dir *= rotX(0.25*pi*hash_f_p_neg(0.5));
                p += dir * 1.;
            }
            if(i == 20 + int(sin(T*0.3 + float(id.x))*25)){
                p += dir * (1. + env) ;
                dir *= rotZ(0.5*pi*hash_f_p_neg(0.5));
            }
            // p.x -= 0.001 + T*0.01;

        } else {


        }
        p = pmod(p,4.);
        

        pal_idx += r;
        
        var q = projParticle(p);
        var k = q.xy;

        k += sample_disk()*abs(q.z - focusDist)*0.05*dofFac;
        
        let uv = k.xy/2. + 0.5;
        let cc = uv2(uv.xy*R.xy);
        let idx = cc.x + Ru.x * cc.y;
        if ( 
            q.z > -0.
            && uv.x > 0. && uv.x < 1. 
            && uv.y > 0. && uv.y < 1. 
            && idx < uint(Ru.x*Ru.y)
            ){ 
            // if(abs(uv.x) > 0.1){
            //     atomicAdd(&hist_atomic[idx],0);
            // }else {
            //     atomicAdd(&hist_atomic[idx],0);
            // }
            atomicAdd(&hist_atomic[idx],7);
            atomicAdd(&hist_atomic[(idx*(1 + uint(rb.x*6. + T)%2))%(Ru.x*Ru.y)],uint(111)); // bug lol, but looks cool
            // atomicAdd(&hist_atomic[idx + Ru.x*Ru.y],uint(pal_idx*1.));
        }
    }
}
fn get_hist_id(c: uv2)-> uint{
    return (c.x + uint(R.x) * c.y)%uint(R.x*R.y);
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    let res = uint2(textureDimensions(screen));
    if (id.x >= res.x || id.y >= res.y) { return; }

    seed = hash_u(id.x + hash_u(res.x*id.y*200u)*20u + hash_u(id.y)*250u);
    seed = hash_u(seed);
    R = v2(res);
    U = v2(id.xy);
    let U = float2(float(id.x) + .5, float(res.y - id.y) - .5);

    
    let hist_id =  get_hist_id(id.xy);

    var col = float(atomicLoad(&hist_atomic[hist_id]))*vec3(1);

    // tonemap
    let sc = 124452.7;
    col *= 0.0001;
    var pal = mix_cols(col.x*115.);
    col = col/(1.+col);
    if(hash_f_s(floor(T*0.8))<0.7){
        col = 1.- col;
    } else {
     
        col = pow(step(col,v3(0.5)),v3(0.02));
        col = pow(abs(col),v3(0.02));

    }
    if(abs(col.x - 0.1) < 0.02 
    //&& hash_f_s(floor(T)) < 1
    ){
        col -= pal;
    }
    
    col = pow(abs(col), float3(2./0.45454545));
    
    textureStore(screen, int2(id.xy), float4(col, 1.));
    

    // clear flame
    // atomicStore(&hist_atomic[hist_id+ res.x*res.y],0);
    let addr = &hist_atomic[hist_id];
    if((T%2.) < 1.){
        atomicStore(addr,0);
    } else {
        if(col.x < 0.0 + 0.2*pow(hash_f_s(floor(T)),2)){
            let offs = -int2(0,2) * ((int(hash_f() < col.x*1111.))*2 - 1)*(1 + 5*int(hash_f_s(floor(T))));
            atomicStore(addr,
                (atomicLoad(&hist_atomic[
                    // get_hist_id(uint2(int2(id.xy) + int2(0,1) * ((int(hash_f() < col.x*200.))*2 - 1)))
                    get_hist_id(uint2(int2(id.xy) + offs))
                ]))
            );
        }else {
            
            atomicStore(addr,0);    
        }
    }
}
