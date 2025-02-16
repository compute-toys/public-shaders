const n_tau = 6.2831;
const o_trn__mouse_last_frame = vec2(0);

fn frand(
    o: vec2f
) ->vec2f {
    let od1 = vec2f(o.x *23.14077926, o.y *232.61690225);
    let od2 = vec2f(o.x *54.47856553, o.y *345.84153136);
    var o2 = vec2f(o);
    o2.x = fract(sin(od1.x) * 136.8168);
    o2.y = fract(sin(od2.y) * 534.7645);
    return o2;
}

// fn fdottest(o:float2)->float2{
//     // let o2 = dot(o, float2(2., 2.));
//     // return dot(o, float2(2.));
//     return dot(o, float2(.2, .3));
// }
fn f_o_rnd(
    o: vec2f
)->vec2f{
    // return fract(sin(dot(o, vec2f(12.9898, 78.233))) * 43758.5453);

    return fract(sin(vec2f(o.x*12.19898, o.y*78.21233))*321758.49153);
}

fn fhash22(p: float2) -> float2
{
    /* Hash without Sine https://www.shadertoy.com/view/4djSRW

    Copyright (c) 2014 David Hoskins.
    Copyright (c) 2022 David A Roberts <https://davidar.io/> (WGSL port)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    */

    var p3 = fract(float3(p.xyx) * float3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx+33.33);
    return fract((p3.xx+p3.yz)*p3.zy);
}


// fn f_o_rnd(vec2f o)->f32{
//     return 0.;//vec2f(0.);
//     // return fract(sin(dot(o, vec2(12.9898, 78.233))) * 43758.5453);
// }

@compute @workgroup_size(16, 16)
fn main_image(
    @builtin(global_invocation_id) o_trn: vec3u
) {
    let o_scl = textureDimensions(screen);
    let n_scl_min = min(f32(o_scl.x), f32(o_scl.y));
    let naa = 1./n_scl_min;
    let o_trn_nor = (vec2f(o_trn.xy)-vec2f(o_scl.xy)*.5) / n_scl_min; 
    let o_trn_nor_mou = (vec2f(mouse.pos.xy)-vec2f(o_scl.xy)*.5) / n_scl_min;
    let o_trn_nor_mou_last = textureLoad(
        pass_in, 
        vec2<u32>(0), 
        0, 0
    );

    if(time.frame == 0){

        let o_rnd1 = fhash22(vec2f(o_trn.xy));
        var op = vec4f(
            o_rnd1-.5,
            fhash22(o_rnd1)-float2(0., -.5)
        );
        op.w = 1.;//direction
        if(o_trn.x % 2 == 0){
            op = float4(
                float2(0.0, 0.0001),//velocity
                float2(0.0, 0.0001)//accelleartion
            );
            // op.zw = float2(0.0, 0.1);//accelleartion
        }

        // textureStore(pass_out, o_trn.xy, vec4f(o_rnd.xyz, 1.));
        textureStore(screen, o_trn.xy, op);
        textureStore(pass_out, o_trn.xy, 0, op);

        return;
    }

    let o_dir_mouse = o_trn_nor_mou_last.zw;
    let n_speed_mouse = length(o_dir_mouse);
    let n_rad_mouse = 0.1;
    let n_rad_point_max = 0.03;
    let n_rad_point_min = 0.03;
    var o_pos = textureLoad(
        pass_in, 
        o_trn.xy, 
        0, 0
    );
    var o_vel_acc = textureLoad(
            pass_in, 
            o_trn.xy+vec2(1, 0), 
            0, 0
        );
    if(o_trn.x == 0 && o_trn.y == 0){
        // last mouse 
        let op = vec4<f32>(
            o_trn_nor_mou.xy,
            o_trn_nor_mou_last.xy - o_trn_nor_mou.xy // delta
        );
        // textureStore(pass_out, o_trn.xy, vec4f(o_rnd.xyz, 1.));
        textureStore(screen, vec2<u32>(0), op);
        textureStore(pass_out, vec2<u32>(0), 0, op);
        return;
    }
    if(o_trn.y <= 1 && o_trn.x % 2 == 0){
        return;
    }

    if(o_trn.y <= 1){


        let nrad = o_pos.z * n_rad_point_max+n_rad_point_min;
        let o_ball_pos_nor = (o_pos+.5);
        //top bottom boundries
        if(
            o_ball_pos_nor.y-nrad < 0.
            || 
            o_ball_pos_nor.y+nrad > 1.
        ){ 
            o_vel_acc = float4(
                o_vel_acc.x,
                -.8*o_vel_acc.y,//+ o_vel_acc.zw,
                o_vel_acc.zw//*1.2
            );
        }
        // left right boundries
        if(
            o_ball_pos_nor.x-nrad < 0.
            || 
            o_ball_pos_nor.x+nrad > 1.
        ){ 
            o_vel_acc = float4(
                -.8*o_vel_acc.x,
                o_vel_acc.y,//+ o_vel_acc.zw,
                o_vel_acc.zw//*1.2
            );
        }

        o_vel_acc = float4(
            o_vel_acc.xy + o_vel_acc.zw,
            o_vel_acc.zw
        );

        o_pos = float4(
            o_pos.xy + o_vel_acc.xy,
            o_pos.zw
        );

        //if ball makes it outside if rectangle , reset its position and velocity
        let ndist = max(abs(o_ball_pos_nor.x), abs(o_ball_pos_nor.y));
        if(ndist > 1.01){
            o_pos = float4(
                vec2<f32>(0.),
                o_pos.zw
            );
            o_vel_acc = float4(
                float2(0.0, 0.0001),
                o_vel_acc.zw
            );
        }


        // o_pos.xy += o_pos2.xy;
        let o_dir = o_pos.xy-o_trn_nor_mou.xy;
        let nl = length(o_dir);
        let nt = n_rad_mouse+nrad;
        if(nl < (nt)){
            o_pos = float4(
                float2(
                    o_trn_nor_mou.xy + (normalize(o_dir)*nt)
                ), 
                o_pos.zw 
            );
            o_vel_acc = float4(
                o_vel_acc.xy+n_speed_mouse*normalize(o_dir),
                o_vel_acc.zw//*1.2
            );
        }
        textureStore(screen, o_trn.xy, o_pos);
        textureStore(pass_out, o_trn.xy, 0, o_pos);
        textureStore(pass_out, o_trn.xy+vec2(1, 0), 0, o_vel_acc);


        return;

    }
    let n_its = 20.*2.;
    let n_it_nor_one = 1./n_its;
    var n_min = 1.;
    var n2 = 0.;
    for(var n_it_nor = n_it_nor_one; n_it_nor < 1.; n_it_nor += n_it_nor_one){
        let o_col = textureLoad(
            pass_in,
            vec2u(n_it_nor*vec2f(n_its, 0.)),
        0, 0);
        let nrad = o_col.z * n_rad_point_max + n_rad_point_min;
        var no = length(o_trn_nor.xy-o_col.xy);
        let n = 1.-smoothstep(nrad,nrad-naa*4., no);

        if(n < n_min){
            n_min = n;
            n2 += smoothstep(nrad-0.01,nrad-naa*4.-0.01, no);
        }

    }
    var nm = length(o_trn_nor-o_trn_nor_mou.xy);
    let nm2 = smoothstep(n_rad_mouse, n_rad_mouse-naa*4., nm);
    var nm3 = smoothstep(n_rad_mouse-.01, n_rad_mouse-0.01-naa*4., nm);
    let nmc = nm2*nm3;
    let obg = float3(0.2, 0.2, 0.2);
    let o2 = float3(n_min+nm2);
    let nobg = float3(1.-n_min)+nm2;
    let o = float4(
        (1.-nobg)*obg+nm3+n2,
        // float3(n2),
        1.
    );
    textureStore(screen, o_trn.xy, o);
    // textureStore(screen, o_trn.xy, float4(o_trn_nor.y));
}
