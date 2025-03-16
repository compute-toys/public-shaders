//trying to balance double pendulum
//this shows all positions of the pendulum through time at the same time
//gradient descent minimizes distance from one extreme of the pendulum to a target point
//thats the distance of blue points to the white point
//red point is the anchor point which is controlled by gradient descent
//the other points are only controlled by physics
//also minimize distance traveled by all points except red points
//this basic method gets stuck on a dumb local minimum
//similar to the worst performing method in this video https://youtu.be/-uXFYpVumh4?t=64
//using SLANG automatic differentiation to find derivatives of the physics simulation

struct DiffPair_float_0
{
     primal_0 : f32,
     differential_0 : f32,
};

fn _d_sqrt_0( dpx_0 : ptr<function, DiffPair_float_0>,  dOut_0 : f32)
{
    var _S1 : f32 = 0.5f / sqrt(max(1.00000001168609742e-07f, (*dpx_0).primal_0)) * dOut_0;
    (*dpx_0).primal_0 = (*dpx_0).primal_0;
    (*dpx_0).differential_0 = _S1;
    return;
}

struct DiffPair_vectorx3Cfloatx2C2x3E_0
{
     primal_0 : vec2<f32>,
     differential_0 : vec2<f32>,
};

fn _d_dot_0( dpx_1 : ptr<function, DiffPair_vectorx3Cfloatx2C2x3E_0>,  dpy_0 : ptr<function, DiffPair_vectorx3Cfloatx2C2x3E_0>,  dOut_1 : f32)
{
    var x_d_result_0 : vec2<f32>;
    var y_d_result_0 : vec2<f32>;
    x_d_result_0[i32(0)] = (*dpy_0).primal_0[i32(0)] * dOut_1;
    y_d_result_0[i32(0)] = (*dpx_1).primal_0[i32(0)] * dOut_1;
    x_d_result_0[i32(1)] = (*dpy_0).primal_0[i32(1)] * dOut_1;
    y_d_result_0[i32(1)] = (*dpx_1).primal_0[i32(1)] * dOut_1;
    (*dpx_1).primal_0 = (*dpx_1).primal_0;
    (*dpx_1).differential_0 = x_d_result_0;
    (*dpy_0).primal_0 = (*dpy_0).primal_0;
    (*dpy_0).differential_0 = y_d_result_0;
    return;
}

fn s_primal_ctx_dot_0( _S2 : vec2<f32>,  _S3 : vec2<f32>) -> f32
{
    return dot(_S2, _S3);
}

fn s_bwd_prop_dot_0( _S4 : ptr<function, DiffPair_vectorx3Cfloatx2C2x3E_0>,  _S5 : ptr<function, DiffPair_vectorx3Cfloatx2C2x3E_0>,  _S6 : f32)
{
    _d_dot_0(&((*_S4)), &((*_S5)), _S6);
    return;
}

fn s_bwd_prop_sqrt_0( _S7 : ptr<function, DiffPair_float_0>,  _S8 : f32)
{
    _d_sqrt_0(&((*_S7)), _S8);
    return;
}

fn s_bwd_prop_length_impl_0( dpx_2 : ptr<function, DiffPair_vectorx3Cfloatx2C2x3E_0>,  _s_dOut_0 : f32)
{
    var _S9 : f32 = (*dpx_2).primal_0[i32(0)];
    var _S10 : f32 = (*dpx_2).primal_0[i32(1)];
    var _S11 : DiffPair_float_0;
    _S11.primal_0 = _S9 * _S9 + _S10 * _S10;
    _S11.differential_0 = 0.0f;
    s_bwd_prop_sqrt_0(&(_S11), _s_dOut_0);
    var _S12 : f32 = (*dpx_2).primal_0[i32(1)] * _S11.differential_0;
    var _S13 : f32 = _S12 + _S12;
    var _S14 : f32 = (*dpx_2).primal_0[i32(0)] * _S11.differential_0;
    var _S15 : f32 = _S14 + _S14;
    var _S16 : vec2<f32> = vec2<f32>(0.0f);
    _S16[i32(1)] = _S13;
    _S16[i32(0)] = _S15;
    (*dpx_2).primal_0 = (*dpx_2).primal_0;
    (*dpx_2).differential_0 = _S16;
    return;
}

fn s_bwd_length_impl_0( _S17 : ptr<function, DiffPair_vectorx3Cfloatx2C2x3E_0>,  _S18 : f32)
{
    s_bwd_prop_length_impl_0(&((*_S17)), _S18);
    return;
}

fn s_bwd_prop_barFrc_0( dpp_0 : ptr<function, DiffPair_vectorx3Cfloatx2C2x3E_0>,  dpv_0 : ptr<function, DiffPair_vectorx3Cfloatx2C2x3E_0>,  dpp2_0 : ptr<function, DiffPair_vectorx3Cfloatx2C2x3E_0>,  dpv2_0 : ptr<function, DiffPair_vectorx3Cfloatx2C2x3E_0>,  dpbl_0 : ptr<function, DiffPair_float_0>,  s_diff_f2_T_0 : vec2<f32>)
{
    var _S19 : DiffPair_vectorx3Cfloatx2C2x3E_0 = (*dpv_0);
    var _S20 : DiffPair_vectorx3Cfloatx2C2x3E_0 = (*dpv2_0);
    var _S21 : DiffPair_float_0 = (*dpbl_0);
    var _S22 : vec2<f32> = vec2<f32>(0.0f);
    var np_0 : vec2<f32> = (*dpp_0).primal_0 - (*dpp2_0).primal_0;
    var _S23 : f32 = length(np_0);
    var _S24 : vec2<f32> = vec2<f32>(_S23);
    var _S25 : bool = _S23 != 0.0f;
    var np_1 : vec2<f32>;
    var _S26 : vec2<f32>;
    if(_S25)
    {
        var _S27 : vec2<f32> = vec2<f32>((_S23 * _S23));
        np_1 = np_0 / _S24;
        _S26 = _S27;
    }
    else
    {
        np_1 = np_0;
        _S26 = _S22;
    }
    var _S28 : vec2<f32> = vec2<f32>((_S21.primal_0 - _S23));
    var nv_0 : vec2<f32> = _S19.primal_0 - (_S19.primal_0 + _S20.primal_0) * vec2<f32>(0.5f);
    var _S29 : vec2<f32> = vec2<f32>(s_primal_ctx_dot_0(nv_0, np_1));
    var _S30 : vec2<f32> = vec2<f32>(-1.0f) * s_diff_f2_T_0;
    var _S31 : vec2<f32> = _S29 * _S30;
    var _S32 : vec2<f32> = np_1 * _S30;
    var _S33 : f32 = _S32[i32(0)] + _S32[i32(1)];
    var _S34 : DiffPair_vectorx3Cfloatx2C2x3E_0;
    _S34.primal_0 = nv_0;
    _S34.differential_0 = _S22;
    var _S35 : DiffPair_vectorx3Cfloatx2C2x3E_0;
    _S35.primal_0 = np_1;
    _S35.differential_0 = _S22;
    s_bwd_prop_dot_0(&(_S34), &(_S35), _S33);
    var _S36 : vec2<f32> = vec2<f32>(0.5f) * - _S34.differential_0;
    var _S37 : vec2<f32> = np_1 * s_diff_f2_T_0;
    var _S38 : f32 = _S37[i32(0)] + _S37[i32(1)];
    var _S39 : f32 = - _S38;
    var _S40 : vec2<f32> = _S31 + _S35.differential_0 + _S28 * s_diff_f2_T_0;
    var _S41 : vec2<f32> = _S34.differential_0 + _S36;
    if(_S25)
    {
        var _S42 : vec2<f32> = _S40 / _S26;
        var _S43 : vec2<f32> = _S24 * _S42;
        np_1 = np_0 * - _S42;
        _S26 = _S43;
    }
    else
    {
        np_1 = _S22;
        _S26 = _S40;
    }
    var _S44 : f32 = np_1[i32(0)] + np_1[i32(1)] + _S39;
    var _S45 : DiffPair_vectorx3Cfloatx2C2x3E_0;
    _S45.primal_0 = np_0;
    _S45.differential_0 = _S22;
    s_bwd_length_impl_0(&(_S45), _S44);
    var _S46 : vec2<f32> = _S45.differential_0 + _S26;
    var _S47 : vec2<f32> = - _S46;
    (*dpbl_0).primal_0 = (*dpbl_0).primal_0;
    (*dpbl_0).differential_0 = _S38;
    (*dpv2_0).primal_0 = (*dpv2_0).primal_0;
    (*dpv2_0).differential_0 = _S36;
    (*dpp2_0).primal_0 = (*dpp2_0).primal_0;
    (*dpp2_0).differential_0 = _S47;
    (*dpv_0).primal_0 = (*dpv_0).primal_0;
    (*dpv_0).differential_0 = _S41;
    (*dpp_0).primal_0 = (*dpp_0).primal_0;
    (*dpp_0).differential_0 = _S46;
    return;
}

fn s_bwd_barFrc_0( _S48 : ptr<function, DiffPair_vectorx3Cfloatx2C2x3E_0>,  _S49 : ptr<function, DiffPair_vectorx3Cfloatx2C2x3E_0>,  _S50 : ptr<function, DiffPair_vectorx3Cfloatx2C2x3E_0>,  _S51 : ptr<function, DiffPair_vectorx3Cfloatx2C2x3E_0>,  _S52 : ptr<function, DiffPair_float_0>,  _S53 : vec2<f32>)
{
    s_bwd_prop_barFrc_0(&((*_S48)), &((*_S49)), &((*_S50)), &((*_S51)), &((*_S52)), _S53);
    return;
}

fn s_bwd_prop_dis2_0( dpa_0 : ptr<function, DiffPair_vectorx3Cfloatx2C2x3E_0>,  dpb_0 : ptr<function, DiffPair_vectorx3Cfloatx2C2x3E_0>,  _s_dOut_1 : f32)
{
    var c_0 : vec2<f32> = (*dpa_0).primal_0 - (*dpb_0).primal_0;
    var _S54 : vec2<f32> = vec2<f32>(0.0f);
    var _S55 : DiffPair_vectorx3Cfloatx2C2x3E_0;
    _S55.primal_0 = c_0;
    _S55.differential_0 = _S54;
    var _S56 : DiffPair_vectorx3Cfloatx2C2x3E_0;
    _S56.primal_0 = c_0;
    _S56.differential_0 = _S54;
    s_bwd_prop_dot_0(&(_S55), &(_S56), _s_dOut_1);
    var _S57 : vec2<f32> = _S56.differential_0 + _S55.differential_0;
    var _S58 : vec2<f32> = - _S57;
    (*dpb_0).primal_0 = (*dpb_0).primal_0;
    (*dpb_0).differential_0 = _S58;
    (*dpa_0).primal_0 = (*dpa_0).primal_0;
    (*dpa_0).differential_0 = _S57;
    return;
}

fn s_bwd_dis2_0( _S59 : ptr<function, DiffPair_vectorx3Cfloatx2C2x3E_0>,  _S60 : ptr<function, DiffPair_vectorx3Cfloatx2C2x3E_0>,  _S61 : f32)
{
    s_bwd_prop_dis2_0(&((*_S59)), &((*_S60)), _S61);
    return;
}



#define PI 3.14159265358979323846f
#define B 3            //balls
#define K 2            //bars per ball
#define T 512           //total time to simulate
#define P 2            //properties per ball (pos,vel)
#storage D array<vec2f,B*P*T+1>
var<workgroup> D2: array<vec2f,B*P>;        //current ball properties
var<workgroup> D3: array<vec2f,B*P>;        //current ball properties
var<workgroup> D4: array<vec2f,T>;          //current ball properties
const brsB = array(1,0,-1, -1,2,1);         //2 bars per ball, must not have pointer collisions
const brsL = array(1f,1f,1f, 1f,1f,1f);     //bars length
const brsF =  1f;                           //bar force
const brsR = -1f;                           //bar friction
const grv  = vec2f(0,-.1);                  //gravity
const stp  = .03f;                          //time step
const trp  = vec2f(0,2);                    //pendulum target point
#dispatch_once ini
//#dispatch_once forward
//#dispatch_once backward
#workgroup_count ini 1 1 1 
@compute @workgroup_size(B,1,1)
fn ini(@builtin(global_invocation_id) id3: vec3u)
{
    var id1 = i32(id3.x);
    D[id1 + B*0 + B*P*0] = vec2f(f32(id1));   //pos
    D[id1 + B*1 + B*P*0] = vec2f(0);          //vel
}
#workgroup_count forward 1 1 1
@compute @workgroup_size(B,1,1)
fn forward(@builtin(global_invocation_id) id3: vec3u)
{
    var id1 = i32(id3.x);
    var p = D[id1 + B*0 + B*P*0];  
    var v = D[id1 + B*1 + B*P*0];
    D2[id1 + B*0] = p;
    D2[id1 + B*1] = v;
    var ff = dot(p-trp,p-trp);
    for(var t=1; t<T; t++)//time loops
    {
        workgroupBarrier();
        var f = vec2f(0);
        for(var k=0; k<K; k++)//bar
        {
            var bb = brsB[id1+k*B];
            var bl = brsL[id1+k*B];
            var p2 = vec2f(0);  if(bb>=0){p2 = D2[bb + B*0];}
            var v2 = vec2f(0);  if(bb>=0){v2 = D2[bb + B*1];}
            var np = p-p2;
            var l  = length(np);
            if(l!=0f){np = np/l;}
            var f1 = np*(bl-l) * brsF;      //force bar atraction
            var nv = v-(v+v2)*.5f;
            var f2 = dot(nv,np)*np * brsR;  //force bar friction
            f += (f1+f2)*f32(bb>=0);
        }
        v += f + grv;
        p += v*stp;
        workgroupBarrier();
        //if pendulum anchorpoint
        if(id1==0){v = vec2f(0);}
        if(id1==0){p = D[id1 + B*0 + B*P*t];}
        D2[id1 + B*0] = p;
        D2[id1 + B*1] = v;
        D[id1 + B*0 + B*P*t] = p;
        D[id1 + B*1 + B*P*t] = v;
        ff += dot(p-trp,p-trp);
    }
    //if(id1==B-1){D[B*P*T]=vec2f(ff/f32(T));}//save sums distance to target
}
#workgroup_count backward 1 1 1
@compute @workgroup_size(B,1,1)
fn backward(@builtin(global_invocation_id) id3: vec3u)
{
    var id1 = i32(id3.x);
    //var ff = 1f;    if(id1==B-1){ff = D[B*P*T].x;}
    D3[id1 + B*0] = vec2f(0);
    D3[id1 + B*1] = vec2f(0);
    var tz = 0f;
    for(var t=T-1; t>=0; t--)//time loops
    {
        //var j1_0 = DiffPair_vectorx3Cfloatx2C2x3E_0(D[id1 + B*0 + B*P*t],vec2f(0));
        //var j2_0 = DiffPair_vectorx3Cfloatx2C2x3E_0(trp,vec2f(0));
        //s_bwd_dis2_0(&(j1_0), &(j2_0), 1f); //ff);
        //var drv = j1_0.differential_0;
        var                       ps2 = D[id1 + B*0 + B*P*(t-0)];
        var ps1 = ps2; if(t+1< T){ps1 = D[id1 + B*0 + B*P*(t+1)];}
        var ps3 = ps2; if(t-1>=0){ps3 = D[id1 + B*0 + B*P*(t-1)];}
        D3[id1 + B*0] += f32(id1 == B-1)*(ps2-trp);             //minimize distance to target on blue points
        D3[id1 + B*0] += f32(id1 !=   0)*(ps1-ps2  +  ps2-ps3); //minimize distance traveled  on all points exept red
        D3[id1 + B*1] += D3[id1 + B*0]*stp;
        //if pendulum anchorpoint
        if(id1==0){var a = D3[id1 + B*0]*vec2f(-1,0);  D4[t] = a;  tz+=dot(a,a);}
        if(id1==0){D3[id1 + B*0] = vec2f(0);}
        if(id1==0){D3[id1 + B*1] = vec2f(0);}
        if(t==0){continue;}
        D2[id1 + B*0] = D[id1 + B*0 + B*P*(t-1)];
        D2[id1 + B*1] = D[id1 + B*1 + B*P*(t-1)];
        workgroupBarrier();
        var ps = vec2f(0);
        var vs = vec2f(0);
        for(var k=0; k<K; k++)//bar
        {
            var bb = brsB[id1+k*B];
            var bl = brsL[id1+k*B];
            var p2 = vec2f(0);        if(bb>=0){p2 = D2[bb + B*0];}
            var v2 = vec2f(0);        if(bb>=0){v2 = D2[bb + B*1];}
            var p31 = D3[id1 + B*0];
            var v31 = D3[id1 + B*1];
            var p32 = vec2f(0);       if(bb>=0){p32 = D3[bb + B*0];}
            var v32 = vec2f(0);       if(bb>=0){v32 = D3[bb + B*1];}
            var p_0  = DiffPair_vectorx3Cfloatx2C2x3E_0(D2[id1 + B*0],vec2f(0));
            var v_0  = DiffPair_vectorx3Cfloatx2C2x3E_0(D2[id1 + B*1],vec2f(0));
            var p2_0 = DiffPair_vectorx3Cfloatx2C2x3E_0(p2           ,vec2f(0));
            var v2_0 = DiffPair_vectorx3Cfloatx2C2x3E_0(v2           ,vec2f(0));
            var bl_0 = DiffPair_float_0(bl,0f);
            s_bwd_barFrc_0(&(p_0), &(v_0), &(p2_0), &(v2_0), &(bl_0), v31*f32(bb>=0));
            ps += p_0.differential_0;
            vs += v_0.differential_0;
            s_bwd_barFrc_0(&(p2_0), &(v2_0), &(p_0), &(v_0), &(bl_0), v32*f32(bb>=0));
            ps += p_0.differential_0;
            vs += v_0.differential_0;
        }
        workgroupBarrier();
        D3[id1 + B*0] = D3[id1 + B*0]*f32(id1!=0) + ps;
        D3[id1 + B*1] = D3[id1 + B*1]*f32(id1!=0) + vs;
    }
    //normalize change then apply
    if(tz!=0f){tz = 1f/sqrt(tz);}
    tz *= f32(id1==0)*.1f;
    for(var t=0; t<T; t++)
    {
        D[id1 + B*0 + B*P*t] += D4[t]*tz;
    }
}
@compute @workgroup_size(8, 8)
fn clear(@builtin(global_invocation_id) id: vec3u)
{
    if(id.x >= SCREEN_WIDTH ){ return; }
    if(id.y >= SCREEN_HEIGHT){ return; }
    textureStore(screen, id.xy, vec4f(0));
}
#workgroup_count draw T 1 1
@compute @workgroup_size(B,1,1)
fn draw(@builtin(local_invocation_id) id3: vec3u, @builtin(workgroup_id) iw3: vec3u)
{
    var id1 = i32(id3.x);
    var iw1 = i32(iw3.x);
    var p = vec3f(D[id1 + B*0 + B*P*iw1], f32(iw1)/f32(T));
    var v =       D[id1 + B*1 + B*P*iw1];
    var res = vec2f(SCREEN_WIDTH,SCREEN_HEIGHT);
    var m = (2f*vec2f(mouse.pos)-res)/res.y;
    var camPos = cos(time.elapsed*vec3f(-23,-9,27)*.02f+vec3f(11,2,22));
        camPos = cos(0f*vec3f(-23,-9,27)*.02f+vec3f(11,2,22));
    if(mouse.click!=0){camPos = vec3f(cos(m.x),m.y,sin(m.x));}
        //camPos = normalize(camPos)*4f;
    var camDir = -normalize(camPos);
    var sd = normalize(vec3f(camDir.z,0f,-camDir.x));
    var up = -cross(camDir,sd);
    var a = p - (dot(p-camPos,camDir)*camDir + camPos);
    var b = vec2i(i32(dot(a,sd)*res.y*custom.a*.5f+.5f*res.x),
                  i32(dot(a,up)*res.y*custom.a*.5f+.5f*res.y));
    var        c = vec4f(1,0,0,0);
    if(id1==1){c = vec4f(0,1,0,0);}
    if(id1==2){c = vec4f(0,0,1,0);}
    textureStore(screen, b, c);
    //draw target point
    p = vec3f(trp,0);
    a = p - (dot(p-camPos,camDir)*camDir + camPos);
    b = vec2i(i32(dot(a,sd)*res.y*custom.a*.5f+.5f*res.x),
              i32(dot(a,up)*res.y*custom.a*.5f+.5f*res.y));
    if(id1==0 && iw1==0){textureStore(screen, b, vec4f(1));}
}