fn fhash22(p: vec2<f32>) -> vec2<f32>
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

    var p3 = fract(vec3<f32>(p.xyx) * vec3<f32>(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx+33.33);
    return fract((p3.xx+p3.yz)*p3.zy);
}

@compute @workgroup_size(16, 16)
fn main_image(
    @builtin(global_invocation_id) o_trn: vec3u
) {

    if(time.frame > 1 && time.frame % 5  != 0){ 
        // slow down the animation by returning from  every non nth frame
        return;
    }
    let n_min_zoom = 1.;
    let n_max_zoom = 9.;
    let n_range_zoom = n_max_zoom-n_min_zoom;
    let n_min_krnl1 = 3;
    let n_max_krnl1 = 24;
    let n_range_krnl1 = n_max_krnl1-n_min_krnl1;
    let n_min_krnl2 = 3;
    let n_max_krnl2 = 24;
    let n_range_krnl2 = n_max_krnl2-n_min_krnl2;


    let o_1 = textureLoad(pass_in, vec2<i32>(1, 0),0, 0);
    let o_2 = textureLoad(pass_in, vec2<i32>(2, 0),0, 0);
    let o_3 = textureLoad(pass_in, vec2<i32>(3, 0),0, 0);
    let o_4 = textureLoad(pass_in, vec2<i32>(4, 0),0, 0);
    let o_5 = textureLoad(pass_in, vec2<i32>(5, 0),0, 0);
    let o_6 = textureLoad(pass_in, vec2<i32>(6, 0),0, 0);
    let o_7 = textureLoad(pass_in, vec2<i32>(7, 0),0, 0);
    let o_8 = textureLoad(pass_in, vec2<i32>(8, 0),0, 0);

    var o_scl_channel0 = vec2<f32>(textureDimensions(channel0));
    var o_scl_canvas = vec2<f32>(textureDimensions(screen));
    let o_trn_nor_full = vec2<f32>(o_trn.xy) / o_scl_canvas.xy;
    let o_trn_nor_full_mou = vec2<f32>(mouse.pos.xy) / o_scl_canvas.xy;
    //
    let n_zoom = n_min_zoom+n_max_zoom*(o_1.x);
    let o_trn2 = vec3u(o_trn/u32(n_zoom));
    o_scl_canvas /= n_zoom;
    o_scl_channel0 /= n_zoom;
    let n_min = min(o_scl_canvas.x, o_scl_canvas.y);
    var o_trn_nor1 = (vec2<f32>(o_trn2.xy))/o_scl_canvas;
    var o_trn_nor_mou = (vec2<f32>(mouse.pos.xy))/o_scl_canvas;
    var o_trn_nor = (vec2<f32>(o_trn2.xy)-(o_scl_canvas.xy*.5))/n_min;
    o_trn_nor += 0.5;
    let n_x_to_y = o_scl_canvas.x / o_scl_canvas.y;
    var o_rule = textureLoad(
        pass_in, 
        vec2<i32>(0,0),
        0, 0
    );



    var o_values = vec2<f32>(3., 8.);
    var otrn3 = (o_trn_nor_full*o_values);
    var otrn_m3 = (o_trn_nor_full_mou*o_values);
    let n_idx = 1.+floor(otrn3.y);
    let n_idx_m = 1.+floor(otrn_m3.y);
    var o_last = textureLoad(
        pass_in, 
        vec2<i32>(i32(n_idx), 0), 
        0, 0
    );
    let otrn3_fract = fract(otrn3);
    let otrn_m3_fract = fract(otrn_m3);

    if(o_trn_nor_full.x > (2./3.)){

        if(mouse.click == 1 && n_idx == n_idx_m){
            var n = (otrn_m3_fract.x);
            if(n_idx == 1.){
                n = floor(n*f32(n_max_zoom))/f32(n_max_zoom);
            }
            if(n_idx == 7.){
                n = floor(n*f32(n_max_krnl1))/f32(n_max_krnl1);
            }
            if(n_idx == 8.){
                n = floor(n*f32(n_max_krnl2))/f32(n_max_krnl2);
            }

            textureStore(pass_out, vec2<i32>(i32(n_idx_m), 0), 0, 
                vec4<f32>(
                    n
                )
            );
        }


        let o = vec4<f32>(
            abs((o_last.xxxx)-otrn3_fract.xxxx),
        );
        var n_min = min(o.x, o.y);
        n_min = pow(1.-n_min, 33.);
        textureStore(screen, o_trn.xy, vec4<f32>(n_min));
        return;
    }
    if(o_trn2.y == 0 && f32(o_trn2.x) <= o_values.y){
        var o_last = textureLoad(
            pass_in, 
            vec2<i32>(i32(n_idx_m), 0), 
            0, 0
        );
        if(time.frame < 1){
            if(o_trn2.x == 1){ o_last = vec4<f32>(((1./f32(n_range_zoom)))*f32(n_min_zoom));}
            if(o_trn2.x == 2){ o_last = vec4<f32>(0.5);}
            if(o_trn2.x == 3){ o_last = vec4<f32>(0.26);}
            if(o_trn2.x == 4){ o_last = vec4<f32>(0.56);}
            if(o_trn2.x == 5){ o_last = vec4<f32>(0.27);}
            if(o_trn2.x == 6){ o_last = vec4<f32>(0.46);}
            if(o_trn2.x == 7){ o_last = vec4<f32>(((1./f32(n_range_krnl1)))*f32(n_min_krnl1));}
            if(o_trn2.x == 8){ o_last = vec4<f32>(((1./f32(n_range_krnl2)))*f32(n_min_krnl2));}
            textureStore(pass_out, o_trn2.xy, 0, 
                o_last
            );

        }
        textureStore(screen, o_trn2.xy, o_last);
        return;
    }

    if(o_trn2.x == 0 && o_trn2.y == 0){


        if(time.frame < 1){
            o_rule.w = f32(161.);
        }
        if(mouse.click == 1 && o_rule.x == 0.){
            let n_sign = sign(o_trn_nor_mou.x-.5);
            o_rule.w = (o_rule.w+1.*n_sign)%255;
        }
        o_rule.x = f32(mouse.click);
        textureStore(pass_out, o_trn2.xy, 0, o_rule);
        textureStore(screen, o_trn.xy, o_rule);

        return;
    }



    var o = textureLoad(
        pass_in, 
        o_trn2.xy,
        0, 0
    );


    //ok , this is the next step, we will take the normalized sum of two kernels 
    let n_scl_kernel1 = i32(n_min_krnl1)+i32(o_7.x*f32(n_max_krnl1));
    let n_scl_kernel1_half = n_scl_kernel1/2; 
    let n_scl_kernel2 = i32(n_min_krnl2)+i32(o_8.x*f32(n_max_krnl2));
    let n_scl_kernel2_half = n_scl_kernel2/2; 

    var n_sum_krnl_1 = 0.0;
    var n_count_krnl_1 = 0.0;
    for(var n = -n_scl_kernel1_half; n <= n_scl_kernel1_half; n+=1){

        let ok = textureLoad(
            pass_in, 
            vec2<i32>(i32(o_trn2.x)+n, 0),
            0, 0
        );
        n_count_krnl_1+=1.;
        n_sum_krnl_1+=ok.x;
    }
    let n_sum_nor_krnl_1 = n_sum_krnl_1/n_count_krnl_1;

    var n_sum_krnl_2 = 0.0;
    var n_count_krnl_2 = 0.0;
    for(var n = -n_scl_kernel2_half; n <= n_scl_kernel2_half; n+=1){

        if(n >= (-n_scl_kernel1_half) && n <= n_scl_kernel1_half){
            // skip values from kernel 1
            continue;
        }
        let ok = textureLoad(
            pass_in, 
            vec2<i32>(i32(o_trn2.x)+n, 0),
            0, 0
        );
        n_count_krnl_2+=1.;
        n_sum_krnl_2+=ok.x;
    }
    let n_sum_nor_krnl_2 = n_sum_krnl_2/n_count_krnl_2;

    // now the rule 
    var n2 = 0.;
    if(
        n_sum_nor_krnl_1 >= o_2.x
        &&
        (
            n_sum_nor_krnl_2 >= o_3.x
            &&
            n_sum_nor_krnl_2 <= o_4.x
        )

    ){ n2 = 1.;}
    if(
        n_sum_nor_krnl_1 < o_2.x
        &&
        (
            n_sum_nor_krnl_2 >= o_5.x
            &&
            n_sum_nor_krnl_2 <= o_6.x
        )

    ){ n2 = 1.;}

    o = vec4<f32>(n2);
    if(o_rule.x == 0. && mouse.click == 1){

        let or1= fhash22(vec2<f32>(o_trn2.xy)+time.elapsed);
        o = vec4<f32>(or1.x);

    }

    if(
        time.frame < 1
        // || u32(time.frame)%1000== 0
    ){        
        let or1= fhash22(vec2<f32>(o_trn2.xy)+time.elapsed);
        o = vec4<f32>(or1.x);
    }

    if(o_trn2.y > 0){
        // visualize the timeline of the 1d image
        // each pixel takes the value from the last pixel from the previous row
        o = textureLoad(
            pass_in, 
            vec2<i32>(o_trn2.xy)+vec2<i32>(0, -1), 
            0, 0
        );
    }


    
    textureStore(pass_out, o_trn2.xy, 0, o);
    textureStore(screen, o_trn.xy, o);
    // textureStore(screen, o_trn.xy, vec4<f32>(o_1.x));

}
