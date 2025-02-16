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

//----------------------------------------------------------------------------------------
//  1 out, 1 in...
fn hash11(p1: float) -> float
{
    var p = fract(p1 * .1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}

//----------------------------------------------------------------------------------------
//  1 out, 2 in...
fn hash12(p: float2) -> float
{
    var p3  = fract(float3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

//----------------------------------------------------------------------------------------
//  1 out, 3 in...
fn hash13(p: float3) -> float
{
    var p3  = fract(p * .1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}

//----------------------------------------------------------------------------------------
//  2 out, 1 in...
fn hash21(p: float) -> float2
{
    var p3 = fract(float3(p) * float3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx+p3.yz)*p3.zy);
}

//----------------------------------------------------------------------------------------
//  2 out, 2 in...
fn hash22(p: float2) -> float2
{
    var p3 = fract(float3(p.xyx) * float3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx+33.33);
    return fract((p3.xx+p3.yz)*p3.zy);
}

//----------------------------------------------------------------------------------------
//  2 out, 3 in...
fn hash23(p: float3) -> float2
{
    var p3 = fract(p * float3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx+33.33);
    return fract((p3.xx+p3.yz)*p3.zy);
}

//----------------------------------------------------------------------------------------
//  3 out, 1 in...
fn hash31(p: float) -> float3
{
    var p3 = fract(float3(p) * float3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx+33.33);
    return fract((p3.xxy+p3.yzz)*p3.zyx); 
}

//----------------------------------------------------------------------------------------
//  3 out, 2 in...
fn hash32(p: float2) -> float3
{
    var p3 = fract(float3(p.xyx) * float3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yxz+33.33);
    return fract((p3.xxy+p3.yzz)*p3.zyx);
}

//----------------------------------------------------------------------------------------
//  3 out, 3 in...
fn hash33(p: float3) -> float3
{
    var p3 = fract(p * float3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yxz+33.33);
    return fract((p3.xxy + p3.yxx)*p3.zyx);
}

//----------------------------------------------------------------------------------------
// 4 out, 1 in...
fn hash41(p: float) -> float4
{
    var p4 = fract(float4(p) * float4(.1031, .1030, .0973, .1099));
    p4 += dot(p4, p4.wzxy+33.33);
    return fract((p4.xxyz+p4.yzzw)*p4.zywx);
}

//----------------------------------------------------------------------------------------
// 4 out, 2 in...
fn hash42(p: float2) -> float4
{
    var p4 = fract(float4(p.xyxy) * float4(.1031, .1030, .0973, .1099));
    p4 += dot(p4, p4.wzxy+33.33);
    return fract((p4.xxyz+p4.yzzw)*p4.zywx);
}

//----------------------------------------------------------------------------------------
// 4 out, 3 in...
fn hash43(p: float3) -> float4
{
    var p4 = fract(float4(p.xyzx)  * float4(.1031, .1030, .0973, .1099));
    p4 += dot(p4, p4.wzxy+33.33);
    return fract((p4.xxyz+p4.yzzw)*p4.zywx);
}

//----------------------------------------------------------------------------------------
// 4 out, 4 in...
fn hash44(p: float4) -> float4
{
    var p4 = fract(p  * float4(.1031, .1030, .0973, .1099));
    p4 += dot(p4, p4.wzxy+33.33);
    return fract((p4.xxyz+p4.yzzw)*p4.zywx);
}

//----------------------------------------------------------------------------------------
fn hashOld12(p: float2) -> float
{
    // Two typical hashes...
    return fract(sin(dot(p, float2(12.9898, 78.233))) * 43758.5453);
    
    // This one is better, but it still stretches out quite quickly...
    // But it's really quite bad on my Mac(!)
    //return fract(sin(dot(p, float2(1.0,113.0)))*43758.5453123);
}

//----------------------------------------------------------------------------------------

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    let screen_size = uint2(textureDimensions(screen));
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    let position = float2(id.xy) + 0.5;
    let uv = position / float2(screen_size);
    let pos = position * .152 + float(time.frame) * 25. + 50.;
    let a = hash12(pos);
    let b = hashOld12(pos);
    var col = float3(mix(b, a, step(uv.x, .5)));
    col = mix(float3(.4, 0.0, 0.0), col, 1 - smoothstep(.495, .5, uv.x) + smoothstep(.5, .505, uv.x));
    textureStore(screen, int2(id.xy), float4(col, 1.));
}
