//MIT License
//Copyright (c) 2019 niall tl
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:
//The above copyright notice and this permission notice shall be included in all
//copies or substantial portions of the Software.
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.

#define paz 64              //paz * paz = total particles
#define cez 128             //cez * cez = total cells
const dt         = .1f;     //time step
const gravity    = .5f;
const PI         = 3.14159265358979f;
const elastic_la = 10f;
const elastic_mu = 20f;
const mouseForce = 9f;
struct Particle {
    pos : vec2f  ,
    vel : vec2f  ,
    mom : mat2x2f, // affine momentum matrix
    mas : f32    ,
    vol : f32    ,
    dfr : mat2x2f, // deformation gradient
}
struct Cell {
    vel : vec2f,
    mas : f32  ,
}
#storage P array<   Particle, paz*paz  >;
#storage C array<atomic<u32>, cez*cez*3>;
fn atomicAddFloat(w: int, v: f32)
{
    var old = atomicLoad(&C[w]);
    loop {
        var n = bitcast<u32>(bitcast<f32>(old) + v);
        var r = atomicCompareExchangeWeak(&C[w], old, n);
        if r.exchanged { break; }
        old = r.old_value;
    }
}
fn cellAdd(w: i32, c: Cell)
{
    atomicAddFloat(w+0*cez*cez, c.vel.x);
    atomicAddFloat(w+1*cez*cez, c.vel.y);
    atomicAddFloat(w+2*cez*cez, c.mas  );
}
fn cellWrite(w: i32, c: Cell)
{
    atomicStore(&C[w+0*cez*cez], bitcast<u32>(c.vel.x));
    atomicStore(&C[w+1*cez*cez], bitcast<u32>(c.vel.y));
    atomicStore(&C[w+2*cez*cez], bitcast<u32>(c.mas  ));
}
fn cellRead(r: i32) -> Cell
{
    var velx = bitcast<f32>(atomicLoad(&C[r+0*cez*cez]));
    var vely = bitcast<f32>(atomicLoad(&C[r+1*cez*cez]));
    var mass = bitcast<f32>(atomicLoad(&C[r+2*cez*cez]));
    return Cell(vec2f(velx,vely), mass);
}
fn mod2(a: vec2f, b: f32) -> vec2f
{
    return fract(a/b)*b;
}
@compute @workgroup_size(8,8,1)
fn clearScreen(@builtin(global_invocation_id) id3: uint3)
{
    var screen_size = textureDimensions(screen);
    if(id3.x >= screen_size.x){return;}
    if(id3.y >= screen_size.y){return;}
    textureStore(screen, id3.xy, vec4f(0));
}
#workgroup_count cellClear 4 cez 1
@compute @workgroup_size(cez/4,1,1)
fn cellClear(@builtin(global_invocation_id) id3: uint3)
{
    var id2 = vec2i(id3.xy);
    var id1 = dot(id2,vec2i(1,cez));
    var c   = Cell(vec2f(0), 0f);
    cellWrite(id1, c);
}
#dispatch_once particleIni
#workgroup_count particleIni 2 paz 1
@compute @workgroup_size(paz/2,1,1)
fn particleIni(@builtin(global_invocation_id) id3: uint3)
{
    var id2 = vec2i(id3.xy);
    var id1 = dot(id2,vec2i(1,paz));
    var p = Particle(
        vec2f(id3.xy)/f32(paz)*f32(cez)*.5f + f32(cez)*.4f,
        vec2f(0),
        mat2x2f(0,0,0,0),
        1f,
        1f,
        mat2x2f(1,0,0,1)
    );
    P[id1] = p;
}
#workgroup_count particleToCell 2 paz 1
@compute @workgroup_size(paz/2,1,1)
fn particleToCell(@builtin(global_invocation_id) id3: uint3)
{
    var id2  = vec2i(id3.xy);
    var id1  = dot(id2,vec2i(1,paz));
    var p    = P[id1];
    var F       = p.dfr;
    var J       = determinant(F);
    var F_T     = transpose(F);
    var Jinv    = 1f;   if(J!=0f){Jinv = 1f/J;}
    var F_inv_T = mat2x2f(
         F_T[1][1], -F_T[0][1],
        -F_T[1][0],  F_T[0][0]) * Jinv;
    var F_minus_F_inv_T = F - F_inv_T;
    var P_term_0 = elastic_mu * F_minus_F_inv_T;
    var P_term_1 = elastic_la * log(J) * F_inv_T;
    var P        = P_term_0 + P_term_1;
    var stress   = Jinv * (P*F_T);
    var volume   = p.vol * J;
    var eq_16_term_0 = -volume * 4f * stress * dt;
    var cell_idx  = floor(p.pos);
    var cell_diff = fract(p.pos) - .5f;
    var weights = array(vec2f(0),vec2f(0),vec2f(0));
    weights[0] = .50f * pow(.5f - cell_diff, vec2f(2));
    weights[1] = .75f - pow(      cell_diff, vec2f(2));
    weights[2] = .50f * pow(.5f + cell_diff, vec2f(2));
    for(var gy = 0; gy < 3; gy++){
    for(var gx = 0; gx < 3; gx++){
        var cell_pos   = floor(p.pos + vec2f(f32(gx),f32(gy))-1f);
        if(cell_pos.x<0f || cell_pos.x>=f32(cez)){continue;}
        if(cell_pos.y<0f || cell_pos.y>=f32(cez)){continue;}
        var cell_dist  = (cell_pos - p.pos) + .5f;
        var cell_index = i32(dot(cell_pos,vec2f(1,cez)));
        var weight     = weights[gx].x * weights[gy].y;
        var Q          = p.mom * cell_dist;
        var weighted_mass = weight * p.mas;
        var cell = Cell(
            weighted_mass * (p.vel + Q) + eq_16_term_0 * weight * cell_dist,
            weighted_mass
        );
        cellAdd(cell_index, cell);
    }}
}
#workgroup_count cellUpdate 4 cez 1
@compute @workgroup_size(cez/4,1,1)
fn cellUpdate(@builtin(global_invocation_id) id3: uint3)
{
    var id2  = vec2i(id3.xy);
    var id1  = dot(id2,vec2i(1,cez));
    var cell = cellRead(id1);
    if(cell.mas != 0f){cell.vel /= cell.mas;}
    else              {cell.vel  = vec2f(0);}
    cell.vel += dt * vec2f(0,gravity);// * f32(cell.mas != 0f);
    if(id2.x < 2 || id2.x > cez - 3){ cell.vel *= vec2f(0,1); }
    if(id2.y < 2 || id2.y > cez - 3){ cell.vel *= vec2f(1,0); }
    cellWrite(id1, cell);
}
#workgroup_count cellToParticle 2 paz 1
@compute @workgroup_size(paz/2,1,1)
fn cellToParticle(@builtin(global_invocation_id) id3: uint3)
{
    var id2  = vec2i(id3.xy);
    var id1  = dot(id2,vec2i(1,paz));
    var p = P[id1];
    p.vel = vec2f(0);
    var cell_idx  = floor(p.pos);
    var cell_diff = fract(p.pos) - .5f;
    var weights   = array(vec2f(0),vec2f(0),vec2f(0));
    weights[0] = .50f * pow(.5f - cell_diff, vec2f(2));
    weights[1] = .75f - pow(      cell_diff, vec2f(2));
    weights[2] = .50f * pow(.5f + cell_diff, vec2f(2));
    var B = mat2x2f(0,0,0,0);
    for(var gy = 0; gy < 3; gy++){
    for(var gx = 0; gx < 3; gx++){
        var cell_pos   = floor(p.pos + vec2f(f32(gx),f32(gy))-1f);
        if(cell_pos.x<0f || cell_pos.x>=f32(cez)){continue;}
        if(cell_pos.y<0f || cell_pos.y>=f32(cez)){continue;}
        var cell_dist  = (cell_pos - p.pos) + .5f;
        var cell_index = i32(dot(cell_pos,vec2f(1,cez)));
        var cell       = cellRead(cell_index);
        var weight     = weights[gx].x * weights[gy].y;
        var weighted_velocity = cell.vel * weight;
        var term = mat2x2f(
            weighted_velocity * cell_dist.x,
            weighted_velocity * cell_dist.y);
        B += term;
        p.vel += weighted_velocity;
    }}
    p.mom = B * 4f;
    p.pos+= p.vel * dt;
    p.pos = clamp(p.pos, vec2f(1), vec2f(cez - 2));
    var res = float2(textureDimensions(screen));
    if(true){
        var m = vec2f(mouse.pos)/res.y-p.pos/f32(cez);
        p.vel -= m*mouseForce/exp(dot(m,m)*111f)*f32(mouse.click!=0);
    }
    p.dfr = (mat2x2f(1,0,0,1) + dt * p.mom) * p.dfr;
    P[id1] = p;
    //render screen
    textureStore(screen, vec2i(p.pos/f32(cez)*res.y), vec4f(1));
}