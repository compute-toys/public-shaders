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

@compute @workgroup_size(16, 16)
fn main_image(
    @builtin(global_invocation_id) o_trn: vec3u
) {

    let o_scl_channel0 = vec2<f32>(textureDimensions(channel0));
    let o_scl_canvas = vec2<f32>(textureDimensions(screen));
    let n_min = min(o_scl_canvas.x, o_scl_canvas.y);
    var o_trn_nor1 = (vec2<f32>(o_trn.xy))/o_scl_canvas;
    var o_trn_nor_mou = (vec2<f32>(mouse.pos.xy))/o_scl_canvas;
    var o_trn_nor = (vec2<f32>(o_trn.xy)-(o_scl_canvas.xy*.5))/n_min;
    o_trn_nor += 0.5;
    let n_x_to_y = o_scl_canvas.x / o_scl_canvas.y;
    var o_rule = textureLoad(
        pass_in, 
        vec2<i32>(0,0),
        0, 0
    );

    if(o_trn.x == 0 && o_trn.y == 0){


        if(time.frame < 1){
            o_rule.w = float(161.);
        }
        if(mouse.click == 1 && o_rule.x == 0.){
            let n_sign = sign(o_trn_nor_mou.x-.5);
            o_rule.w = (o_rule.w+1.*n_sign)%255;
        }
        o_rule.x = float(mouse.click);
        textureStore(pass_out, o_trn.xy, 0, o_rule);
        textureStore(screen, o_trn.xy, o_rule);

        return;
    }



    var o = textureLoad(
        pass_in, 
        o_trn.xy,
        0, 0
    );
    let om1 = textureLoad(
        pass_in, 
        vec2<i32>(o_trn.xy)+vec2(-1, 0), 
        0, 0
    );
    let op1 = textureLoad(
        pass_in, 
        vec2<i32>(o_trn.xy)+vec2(1, 0), 
        0, 0
    );
    // o = float4(0.);

    // Convert pixel intensity to binary states (0 or 1)
    let n_b_om1 = u32(float(om1.r > 0.5));
    let n_b_o = u32(float(o.r > 0.5));
    let n_b_op1 = u32(float(op1.r > 0.5));

    let n_index_bit = (n_b_om1 << 2) | (n_b_o << 1) | n_b_op1;
    let n_rule = u32(o_rule.w);// rule 30 ->  0b00011110
    let n_b = (n_rule >> n_index_bit) & 1;

    o = vec4<f32>(vec3<f32>(float(n_b)), 1.);


    if(o_rule.x == 0. && mouse.click == 1){

        let or1= fhash22(vec2<f32>(o_trn.xy));
        let or2= fhash22(or1*200.);
        o = vec4<f32>(or1, or2);

    }

    if(o_trn.y > 0){
        // visualize the timeline of the 1d image
        // each pixel takes the value from the last pixel from the previous row
        o = textureLoad(
            pass_in, 
            vec2<i32>(o_trn.xy)+vec2<i32>(0, -1), 
            0, 0
        );
    }
    let n_numbers = 3.;

    o_trn_nor *= vec2f(n_numbers, 1.);
    let nds = float(
        o_trn_nor.x < n_numbers
        &&
        o_trn_nor.x > 0.
    );

    let n_idx_decimal_place = floor(n_numbers-o_trn_nor.x);
//    o_trn_nor = clamp(o_trn_nor, vec2f(0.), vec2f(1.));

    o_trn_nor = fract(o_trn_nor);
    o_trn_nor *= float2(1.,2.);
    o_trn_nor -= float2(0, 0.5+((1./n_numbers)/2.));
    o_trn_nor *= float2(1., 1.+(1./n_numbers));
    o_trn_nor.y = clamp(o_trn_nor.y, 0., 1.);

    var n_num =  o_rule.w;//654321.;//o_rule.w;//4.;
    n_num = floor(n_num/pow(10., n_idx_decimal_place)) % 10.;

    var o_trn_char = o_trn_nor/16.+(vec2<f32>(n_num,3.)/16.);

    let od1 = textureSampleLevel(channel0, bilinear, o_trn_char, 0.);
    let nm = min(od1.x, min(od1.y, od1.z));
    // let n1 = smoothstep(0.01, 0.9, od1.x);

    textureStore(pass_out, o_trn.xy, 0, o);
    textureStore(screen, o_trn.xy, o*(1.-od1.x*nds+nm));
    // textureStore(screen, o_trn.xy, o+float4(nds));
}
