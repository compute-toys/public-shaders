
alias iv4 = vec4<i32>;alias iv3 = vec3<i32>;alias iv2 = vec2<i32>;alias uv4 = vec4<u32>;alias uv3 = vec3<u32>;alias uv2 = vec2<u32>;alias v4 = vec4<f32>;alias v3 = vec3<f32>;alias v2 = vec2<f32>;alias m2 = mat2x2<f32>;alias m3 = mat3x3<f32>;alias m4 = mat4x4<f32>;
#define global var<private>
var<workgroup> texCoords: array<array<float2, 16>, 16>;
#define Frame time.frame
#define T (time.elapsed*float(custom.Paused > 0.5) + 5.) 
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

fn sample_disk() -> v2{
    let r = hash_v2();
    return v2(sin(r.x*tau),cos(r.x*tau))*pow(sqrt(r.y),1.);
}


#define COL_CNT 4
global kCols = array<v3, COL_CNT>( 
     vec3(1.,1,1), vec3(1.,0.4,1.),
     vec3(1,1.,0.1)*1., vec3(0.5,1,1.)*.5
);

fn mix_cols(_idx: float)->v3{
    let idx = _idx%1.;
    var cols_idx = int(idx*float(COL_CNT));
    var fract_idx = fract(idx*float(COL_CNT));
    fract_idx = smoothstep(0.,1.,fract_idx);
    //return oklab_mix( kCols[cols_idx], kCols[(cols_idx + 1)%COL_CNT], fract_idx );
    return mix( kCols[cols_idx], kCols[(cols_idx + 1)%COL_CNT], fract_idx );
}

#define ARR_SZ 768*432*2

struct Hist{
    r: array<atomic<u32>, ARR_SZ>,
    g: array<atomic<u32>, ARR_SZ>,
    b: array<atomic<u32>, ARR_SZ>
}
#storage hist Hist



fn t_spherical(p: v3, rad: float, offs: v3) -> v3{
    return p/(dot(p,p)*rad + offs);
}

fn g_env(_t:float, offs: float)->float{
    let t = _t + offs;
    return _t + sin(_t);
}

#define DISP_SZ_X 64
#define DISP_SZ_Y 16
#define WG_SZ_X 8
#define ITS 60
#define WG_SZ_Y 8

fn projParticle(_p: v3) -> v3{
    var p = _p; // you still can't modify args in wgsl???
    
    //p += sin(v3(3,2,1) + T*0.7)*0.6;

    // p = p/mix(1.,dot(p,p),custom.A);
    // p *= rotZ(T*0.6 + sin(T)*0.6);
    // p *= rotX(muv.y*2.);
    // p *= rotY(T*0.3+ sin(T*0.8)*0.35);
    //p *= rotY(muv.x*2.);
    //p *= rotX(muv.y*2.);
    
    p.z += 8.;
    let z = p.z;
    p.x /= (p.z - 8.)*(1.7 + sin(T));
    p.y /= (p.z - 8.)*(.7-sin(T)*0.1);
    p/=1. ;
    p.z = z;
    p.x /= R.x/R.y;
    return p;
}

fn draw_part( p: float3) -> uint{
    let Ru = uint2(textureDimensions(screen));
    let Rf = float2(textureDimensions(screen));

    var q = p;
    q.x /= Rf.x/Rf.y;
    q = projParticle(q);
    var k = q.xy;


    var dof_samp = sample_disk();
    dof_samp.x /= Rf.x/Rf.y;
    k += dof_samp*(p.z - 4. + sin(T)*2.)*0.1*(1.- 0.5 - 0.2*sin(T*2.));
    
    let uv = k.xy/2. + 0.5;
    let cc = uv2(uv.xy*R.xy);
    var idx = cc.x + Ru.x * cc.y;
    if ( 
        q.z > -0.
        && uv.x > 0. && uv.x < 1. 
        && uv.y > 0. && uv.y < 1. 
        && idx < uint(Ru.x*Ru.y)
        ){     

    }else {
        idx -= idx;
    }

    return idx;
}

fn trans(param: float, k: float, r: float) -> float3{
    var j = v2(
        k*2.*sin(T*0.5) + cos(param +T)*4.0,
        sin(param)*1.0,
    )*rot(floor(r*116.0)*tau/2.);
    // j += sin(j*pi/5.*4.)*14.;
    return v3(j,0. + k);
}
#workgroup_count Splat DISP_SZ_X DISP_SZ_Y 1
@compute @workgroup_size(WG_SZ_X, WG_SZ_Y)
fn Splat(@builtin(global_invocation_id) id: uint3) {
    let Ru = uint2(textureDimensions(screen));
    if (id.x >= Ru.x || id.y >= Ru.y) { return; }
    R = v2(Ru); U = v2(id.xy); muv = (v2(mouse.pos) - 0.5*R)/R.y;
    
    seed = hash_u(id.x + hash_u(Ru.x*id.y*200u)*20u + hash_u(id.x)*250u);
    seed = hash_u(seed);

    let particleIdx = id.x;
    
    var iters = ITS;
    
    iters += int(sin(T)*30);

    var emod =  (max(sin(T),0.))*0.;
    var t = T - hash_f()*(1.)/30.;

    var env = t + sin(t*4.)/4. - emod;
    var envb = sin(t*0.45) - emod;

    var p = hash_v3()*v3(1);

    var pr = hash_v4();
    var prb = hash_v4();

    let focusDist = (custom.DOF_Focal_Dist*1.)*25.;
    let dofFac = 1./vec2(R.x/R.y,1.)*custom.DOF_Amount;
    var pal_idx = 0.;

    var k = 0.0 + pr.w*0.;
    for(var i = 0; i < iters; i++){
        let r = hash_f();

        k += 0.1 + sin(T)*0.01;

        var param = k*2. + T + sin(pr.w*sin(T)-k );
        if(prb.y < 0.5){
            // pr.w += 0.5;
        }
        // pr.w = hash_f();


        var fidx = 1.-float(i)/float(iters);

        p = trans(param, k + fidx*pow(pr.x,15. + sin(T)*10.), pr.w);
        var idx = draw_part(p);
        atomicAdd(&hist.r[idx],uint(255.0));

        p = trans(param, k + fidx*pow(pr.y,1.5), pr.w);
        idx = draw_part(p);
        atomicAdd(&hist.g[idx],uint(255.0));

        p = trans(param, k+ fidx*pow(pr.z,.45), pr.w);
        idx = draw_part(p);
        atomicAdd(&hist.b[idx],uint(255.0));
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


    var col = float3(
        float(atomicLoad(&hist.r[hist_id])),
        float(atomicLoad(&hist.g[hist_id])),
        float(atomicLoad(&hist.b[hist_id])),
    )*float3(1);
    
    // tonemap
    let sc = 111111111111.7 
        / (
            float(DISP_SZ_X)*
            float(DISP_SZ_Y)*
            float(WG_SZ_X)*
            float(WG_SZ_Y)*
            float(ITS)
        );

// #define DISP_SZ_X 64
// #define DISP_SZ_Y 16
// #define WG_SZ_X 8
// #define WG_SZ_Y 8
// #define ITS 160
    col = log(col)/ log(sc);
    // col*=0.0001;

    
    col = pow(max(col,v3(0)), float3(1./.45454545));
    col = 1.-col;
    // col = col * mix_cols(float(pal)*0.2);
    
    textureStore(screen, int2(id.xy), float4(col, 1.));
    

    // clear flame
    atomicStore(&hist.r[hist_id],0);
    atomicStore(&hist.g[hist_id],0);
    atomicStore(&hist.b[hist_id],0);
    // atomicStore(&hist_atomic[hist_id],0);
    // atomicStore(&hist_atomic[hist_id*2],0);
}
