const n_tau = 6.2831;
const o_trn__mouse_last_frame = vec2(0);

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
    var o_scl = textureDimensions(screen);
    
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
            o_rnd1,
            fhash22(o_rnd1)
        );
        op = float4(0.);
        textureStore(screen, o_trn.xy, op);
        textureStore(pass_out, o_trn.xy, 0, op);

        return;
    }
    let o_rand = fhash22(o_trn_nor.xy*200.);

    var o = textureLoad(
        pass_in, 
        vec2<i32>(o_trn.xy), 
        0, 0
    );
    let o_scl_krnl = float2(3.);
    let o_scl_krnl_hlf = floor(o_scl_krnl/2.);
    let n_sum_max_krnl = o_scl_krnl.x * o_scl_krnl.y;
    var n_sum = 0.;

    for(var n_x = -o_scl_krnl_hlf.x; n_x <= o_scl_krnl_hlf.x; n_x+=1){
        for(var n_y = -o_scl_krnl_hlf.y; n_y <= o_scl_krnl_hlf.y; n_y+=1){
            if(n_x == 0 && n_y == 0){continue;}
            let o = textureLoad(
                pass_in, 
                vec2<i32>(o_trn.xy)+vec2<i32>(i32(n_x), i32(n_y)), 
                0, 0
            );
            n_sum+= o.x;
        }
    }
    if(n_sum > (n_sum_max_krnl/2.)){
        o += float4(0.101);
    }
    if(n_sum > (n_sum_max_krnl*(7./8.))){
        o -= float4(0.199);
    }
    if(o_trn_nor.y > .0){
        if(n_sum < (n_sum_max_krnl*0.09)){
            o *= float4(o_rand.x+ f32(o_rand.x>.99));
        }
    }

    var n = 1.-length(o_trn_nor_mou-o_trn_nor);
    n = clamp(n , 0., 1.);

    n = pow(n, 100.);
    let nrad = 0.1;
    o+= float4(n*o_rand.x);
    //o+= (1.-length(o_trn_nor_mou-o_trn_nor))/10.;
    //o += n*fhash22(o_trn_nor.xy*200.).x;
    textureStore(pass_out, o_trn.xy, 0, o);
    textureStore(screen, o_trn.xy, o);
}
