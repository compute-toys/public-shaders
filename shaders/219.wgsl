// A rasterizer is used which follows this process:
// 1. Screen is subdivided into cells. Each cell has an array of shapes.
// 2. Shapes are inserted into the cells, in accordance to their bounding boxes.
// 3. They are drawn using SDFs.

alias f = float;alias iv4 = vec4<i32>;alias iv3 = vec3<i32>;alias iv2 = vec2<i32>;alias uv4 = vec4<u32>;alias uv3 = vec3<u32>;alias uv2 = vec2<u32>;alias v4 = vec4<f32>;alias v3 = vec3<f32>;alias v2 = vec2<f32>;alias m2 = mat2x2<f32>;alias m3 = mat3x3<f32>;alias m4 = mat4x4<f32>;
#define global var<private>
var<workgroup> texCoords: array<array<float2, 16>, 16>;
#define F time.frame
#define T (time.elapsed) 
#define pi acos(-1.)
#define tau (acos(-1.)*2.)
global R: v2; global Ru: uv2; global U: v2; global seed: uint; global muv: v2;
fn rot(a: float)-> m2{ return m2(cos(a), -sin(a),sin(a), cos(a));}
fn rotX(a: float) -> m3{ let r = rot(a); return m3(1.,0.,0.,0.,r[0][0],r[0][1],0.,r[1][0],r[1][1]); }
fn rotY(a: float) -> m3{ let r = rot(a); return m3(r[0][0],0.,r[0][1],0.,1.,0.,r[1][0],0.,r[1][1]); }
fn rotZ(a: float) -> m3{ let r = rot(a); return m3(r[0][0],r[0][1],0.,r[1][0],r[1][1],0.,0.,0.,1.); }
fn emod (a: f32, b: f32) -> f32 {var m = a % b;if (m < 0.0) {if (b < 0.0) {m -= b;} else {m += b;}}return m;}
fn gmod(a: v3, b: v3)->v3{return v3(emod(a.x,b.x),emod(a.y,b.y),emod(a.z,b.z));}
fn gmod_v2(a: v2, b: v2)->v2{return v2(emod(a.x,b.x),emod(a.y,b.y));}
fn hash_u(_a: uint) -> uint{ var a = _a; a ^= a >> 16;a *= 0x7feb352du;a ^= a >> 15;a *= 0x846ca68bu;a ^= a >> 16;return a; }
fn hash_f() -> float{ var s = hash_u(seed); seed = s;return ( float( s ) / float( 0xffffffffu ) ); }
fn hash_v2() -> v2{ return v2(hash_f(), hash_f()); }
fn hash_v3() -> v3{ return v3(hash_f(), hash_f(), hash_f()); }
fn hash_v4() -> v4{ return v4(hash_f(), hash_f(), hash_f(), hash_f()); }
fn hash_f_s(s: float) -> float{ return ( float( hash_u(uint(s*float( 0xffffffffu/10000 ))) ) / float( 0xffffffffu ) ); }
fn hash_v2_s(s: float) -> v2{ return v2(hash_f_s(s), hash_f_s(s)); }
fn hash_v3_s(s: float) -> v3{ return v3(hash_f_s(s), hash_f_s(s), hash_f_s(s)); }
fn hash_v4_s(s: float) -> v4{ return v4(hash_f_s(s), hash_f_s(s), hash_f_s(s), hash_f_s(s)); }
fn sample_disk() -> v2{ let r = hash_v2(); return v2(sin(r.x*tau),cos(r.x*tau))*sqrt(r.y); }
fn pmod(p:v3, amt: float)->v3{ return (v3(p + amt*0.5)%amt) - amt*0.5 ;}
fn sdLine( p: v2, a: v2, b: v2 ) -> float{ let pa = p-a; let ba = b-a; let h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 ); return length( pa - ba*h ); }
fn sdBox( _p: v2, _sz: v2 ) -> float{ let p = abs(_p) - _sz; return max(p.x,p.y); }
#define INIT Ru = uint2(textureDimensions(screen)); R = v2(Ru); U = float2(float(id.x) + .5, float(Ru.y - id.y) - .5); seed = hash_u(id.x + hash_u(Ru.x*id.y*200u)*20u + hash_u(id.y)*250u + hash_u(id.x*21832)); seed = hash_u(seed); seed = hash_u(seed); seed = hash_u(seed); seed = hash_u(seed);


#define COL_CNT 4
global kCols = array<v3, COL_CNT>( 
     vec3(0.2,.1,0.2), vec3(1.,0.5,1.)*0.3,
     vec3(1.,0.5,0.1)*1., vec3(0.,1,1.)*0.5
);


fn oklab_mix( colA: v3, colB: v3, h: float  ) -> v3{
    let kCONEtoLMS = m3(
        v3(0.4121656120,  0.2118591070,  0.0883097947),
        v3(0.5362752080,  0.6807189584,  0.2818474174),
        v3(0.0514575653,  0.1074065790,  0.6302613616)
    );  
    let kLMStoCONE = m3(
        v3(4.0767245293, -1.2681437731, -0.0041119885),
        v3(-3.3072168827,  2.6093323231, -0.7034763098),
        v3(0.2307590544, -0.3411344290,  1.7068625689)
    );
    let lmsA = pow( kCONEtoLMS*colA, vec3(1.0/3.0) );
    let lmsB = pow( kCONEtoLMS*colB, vec3(1.0/3.0) );
    let lms = mix( lmsA, lmsB, h );
    return kLMStoCONE*(lms*lms*lms);
}

fn mix_cols(_idx: float)->v3{
    let idx = _idx%1.;
    var cols_idx = int(idx*float(COL_CNT));
    var fract_idx = fract(idx*float(COL_CNT));
    fract_idx = smoothstep(0.,1.,fract_idx);
    return oklab_mix( kCols[cols_idx], kCols[(cols_idx + 1)%COL_CNT], fract_idx );
    return mix( kCols[cols_idx], kCols[(cols_idx + 1)%COL_CNT], fract_idx );
}

fn uv_to_pixel_idx(k: v2)->uint{
    var uv = (k.xy*v2(R.y/R.x,1) + 0.5);
    let cc = uv2(uv.xy*R.xy);
    let idx = cc.x - Ru.x * cc.y + Ru.x*Ru.y ;
    return idx;
}



// WG_RES * 4
#define BIN_RES_X 14
#define BIN_RES_Y 10
#define WG_RES 16 
#define WG_PER_BIN 16
#define SQRT_WG_PER_BIN 4

#define MAX_SHAPES_PER_BIN 4000

#define SHAPE_LINE 0
#define SHAPE_POINT 1
#define SHAPE_TRI 2
#define SHAPE_RECT 3

#define BLEND_ADD 0
#define BLEND_HUE 1
#define BLEND_INV 2
#define BLEND_OP  3

struct Shape{
    ty: u32,
    data: array<vec3f,6>,
    bb: array<vec2f, 2>
}
struct Bin{
    shape_cnt: atomic<uint>,
    shapes: array<Shape, MAX_SHAPES_PER_BIN>,
}

struct ShapeData{
    bins: array<array<Bin, BIN_RES_Y>, BIN_RES_X>
}

#storage shape_data ShapeData
// #storage screendata array<atomic<uint>>
#storage screen_data array<vec3f>

#define fovsc 3.
fn proj_P(_p: vec3f)->vec3f{
    var p = _p;
    p.z += 2.;
    p.x /= p.z*fovsc;
    p.y /= p.z*fovsc;
    return p;
}


fn draw_point(_pos: vec3f, col: vec3f, w: float, op: float, blend: float) -> Shape{
    // let pos = proj_P(_pos);
    let pos = _pos;

    var shape = Shape();
    shape.ty = SHAPE_POINT;
    shape.data[0] = pos;
    shape.data[3].x = w;
    shape.data[3].y = op;
    shape.data[3].z = blend;
    shape.data[4] = col;
    // let diag = sqrt(2*w*w);
    let diag = sqrt(2*w*w);
    // shape.bb[0]= (pos.xy - diag)/(_pos.z*fovsc);
    // shape.bb[1]= (pos.xy + diag)/(_pos.z*fovsc);
    shape.bb[0]= (pos.xy - diag);
    shape.bb[1]= (pos.xy + diag);
    return shape;
}

fn draw_tri(_a: vec3f, _b: vec3f, _c: vec3f, col: vec3f, w: float, op: float, blend: float) -> Shape{
    var shape = Shape();
    shape.ty = SHAPE_TRI;
    shape.data[0] = _a;
    shape.data[1] = _b;
    shape.data[2] = _c;
    shape.data[3].x = w;
    shape.data[3].y = op;
    shape.data[3].z = blend;
    shape.data[4] = col;
    // TODO: better bb
    shape.bb[0]= vec2f(min(_a.x,min(_b.x,_c.x)), min(_a.y,min(_b.y,_c.y)));
    shape.bb[1]= vec2f(max(_a.x,max(_b.x,_c.x)), max(_a.y,max(_b.y,_c.y)));
    return shape;
}
fn draw_rect(_p: vec3f, _sz: vec3f, _rot: float, col: vec3f, op: float, blend: float) -> Shape{
    var shape = Shape();
    shape.ty = SHAPE_RECT;
    shape.data[0] = _p;
    shape.data[1] = _sz;
    shape.data[2].x = _rot;

    // shape.data[3].x = w;
    shape.data[3].y = op;
    shape.data[3].z = blend;
    shape.data[4] = col;

    let diag = sqrt(2*max(_sz.x, _sz.y)*max(_sz.x, _sz.y));
    // if(abs(_rot) < 0.01){
    if(abs(_rot) < 0.01){
        shape.bb[0]= (_p.xy - diag);
        shape.bb[1]= (_p.xy + diag);
    } else {
        let verts = array<v2, 4>(
            rot(-_rot)*(_p.xy + diag*v2(1,1)),
            rot(-_rot)*(_p.xy + diag*v2(1,-1)),
            rot(-_rot)*(_p.xy + diag*v2(-1,-1)),
            rot(-_rot)*(_p.xy + diag*v2(-1,1))
        );

        shape.bb[0]= vec2f(
            min(verts[0].x,min(verts[1].x,min(verts[2].x,verts[3].x))), 
            min(verts[0].y,min(verts[1].y,min(verts[2].y,verts[3].y)))
        );
        shape.bb[1]= vec2f(
            max(verts[0].x,max(verts[1].x,max(verts[2].x,verts[3].x))), 
            max(verts[0].y,max(verts[1].y,max(verts[2].y,verts[3].y)))
        );
    }
    // TODO: better bb
    return shape;
}
fn draw_line(origin: vec3f, tar: vec3f, col: vec3f, w: float, op: float, blend: float) -> Shape{
    var shape = Shape();
    shape.ty = SHAPE_LINE;
    shape.data[0] = origin.xyz;
    shape.data[1] = tar.xyz;
    shape.data[3].x = w;
    shape.data[3].y = op;
    shape.data[3].z = blend;
    shape.data[4] = col;
    shape.bb[0]= vec2f(min(origin.x,tar.x), min(origin.y, tar.y));
    shape.bb[1]= vec2f(max(origin.x,tar.x), max(origin.y, tar.y));
    return shape;
}
fn to_bucket(shape: Shape){
    // let idx = atomicAdd(&shape_data.shape_cnt, 1) - 1;
    // shape_data.shapes[idx] = shape;
    // (uv*iResolution.y + 0.5*iResolution.xy)/R.y;

    let sc = R.xy/(WG_RES*SQRT_WG_PER_BIN);

    var bb_min_f = (shape.bb[0]*v2(R.y/R.x,1.) + 0.5)*sc;
    var bb_max_f = (shape.bb[1]*v2(R.y/R.x,1.) + 0.5)*sc;

    // clamp to screen
    bb_max_f.x = max(bb_max_f.x,0);
    bb_max_f.y = max(bb_max_f.y,0);
    bb_max_f.x = min(bb_max_f.x,BIN_RES_X - 1);
    bb_max_f.y = min(bb_max_f.y,BIN_RES_Y - 1);
    bb_min_f.x = max(bb_min_f.x,0);
    bb_min_f.y = max(bb_min_f.y,0);
    bb_min_f.x = min(bb_min_f.x,BIN_RES_X - 1);
    bb_min_f.y = min(bb_min_f.y,BIN_RES_Y - 1);

    var bb_min = vec2i(bb_min_f);
    var bb_max = vec2i(bb_max_f);


    if(shape.ty == SHAPE_LINE){
        if(true){
            for(var x = bb_min.x; x <= bb_max.x; x++){
                for(var y = bb_min.y; y <= bb_max.y; y++){
                    let idx = atomicAdd(&shape_data.bins[x][y].shape_cnt, 1);
                    if(idx < MAX_SHAPES_PER_BIN){
                        shape_data.bins[x][y].shapes[idx] = shape;
                    }
                }
            }
        } else {
            var a = shape.data[0].xy;
            var b = shape.data[1].xy;
            a = (a*v2(R.y/R.x,1.) + 0.5)*sc;
            b = (b*v2(R.y/R.x,1.) + 0.5)*sc;

            let rd = normalize(b - a);
            let idir = vec2i(sign(rd.xy));
            let ff = sign(rd)/(sign(rd)*max(abs(rd),v2(0.001)));
            let len = length(a - b);
            var p = a;
            var id = vec2i(p);
            var st = abs((floor(p) + max(sign(rd), vec2f(0)) - p) /rd);

            for(var i = 0; i < 60; i++){
                let idx = atomicAdd(&shape_data.bins[id.x][id.y].shape_cnt, 1);
                if(idx < MAX_SHAPES_PER_BIN){
                    shape_data.bins[id.x][id.y].shapes[idx] = shape;
                }

                if(min(st.x,st.y) > len){
                    break;
                }
                if(st.x < st.y){
                    id.x += idir.x;
                    st.x += ff.x;
                } else {
                    id.y += idir.y;
                    st.y += ff.y;
                }

            }

        }
    } else {
        for(var x = bb_min.x; x <= bb_max.x; x++){
            for(var y = bb_min.y; y <= bb_max.y; y++){
                let idx = atomicAdd(&shape_data.bins[x][y].shape_cnt, 1);
                if(idx < MAX_SHAPES_PER_BIN){
                    shape_data.bins[x][y].shapes[idx] = shape;
                }
            }
        }
    }
}

#workgroup_count clean_bins 1 1 1
@compute @workgroup_size(1, 1, 1)
fn clean_bins( @builtin(global_invocation_id) id: vec3u){
    for(var x = 0; x < BIN_RES_X; x++){
        for(var y = 0; y < BIN_RES_Y; y++){
            atomicStore(&shape_data.bins[x][y].shape_cnt,0);
        }
    }
}


fn getOrthogonalBasis(_direction: v3)-> m3{
    let direction = normalize(_direction);
    let right = normalize(cross(vec3f(0,1,0),direction));
    let up = normalize(cross(direction, right));
    return m3(right,up,direction);
}
#define seg (floor(T))
#define segh (floor(T/2))

// Explained here:
// https://www.shadertoy.com/view/3tcyD7
fn cyclicNoise(_p: v3)->float{
    var p = _p;
    var noise = 0.;
    
    var amp = 1.;
    let gain = 0.2 + sin(segb)*.0;
    let lacunarity = .5;
    let octaves = 2;
    
    let warp = 1.;    
    var warpTrk = 1.2 ;
    let warpTrkGain = 1.5;
    
    var seed = vec3f(-1,-2.,0.5);
    seed = rotZ(sin(T)*0.00001)*seed;
    let rotMatrix = getOrthogonalBasis(seed);
    
    for(var i = 0; i < octaves; i+=1){
        p += sin(p.zxy*warpTrk - 2.*warpTrk + T*0.0000001)*warp; 
        noise += sin(dot(cos(p), sin(p.zxy )))*amp;
        p *= rotMatrix;
        p *= lacunarity;
        warpTrk *= warpTrkGain;
        amp *= gain;
    }
    
    
    // return 1. - abs(noise)*0.5;
    return (noise*0.25 + 0.5);
}

#workgroup_count background 1 1 1 
@compute @workgroup_size(11) 
fn background(@builtin(global_invocation_id) id: uint3) {
    INIT

    to_bucket( draw_rect( v3(0), v3(1),0.,v3(1,1,1), 1., BLEND_OP ));
}

global segb: f32;

#workgroup_count shapes_splat_trails 20 1 1 
@compute @workgroup_size(256, 1) 
fn shapes_splat_trails(
    @builtin(global_invocation_id) id: uint3,
    @builtin(workgroup_id) wid: uint3
    ) {
    // return;
    INIT

    if(floor(T)%4.0 < 2.){
        if(floor(T)%8.0 < 4.){
            if (wid.x > 10){
                return;
            }
        } else {
            if (wid.x > 5){
                return;
            }
        }
    }

    seed = 1000 + hash_u(id.x + hash_u(Ru.x*id.y*200u)*20u + hash_u(id.y)*250u + hash_u(id.x*21832)); seed = hash_u(seed); seed = hash_u(seed); seed = hash_u(seed); seed = hash_u(seed);

    segb = 1.;

    // var p = v3(hash_f()*2.-1., 0.,0); 
    // p += sample_disk().xyy*0.; 
    // p.y += 0. + sin(T + seg)*0.7;
    // p.x += 0. + sin(T +0.5)*0.2;
    // p.x *= 0.7;
    // p.y -= 0.2;
    // p = rotZ(0.5 + hash_f_s(seg +10)*40.)*p;
    // if(hash_f_s(seg)<1.){
    
    var p = v3(0); 
        p = sample_disk().xyy*0.5; 

        var offs =  (hash_v3()*2. - 1.)*1.;
        // if(hash_f_s(seg)<0.5){
        if(floor(T)%4 < 2){
            offs *= 0.1 + sin(T)*0.05;
            segb = 0.;
        }

        p += offs;
    // }
    var accl = 0.;
    for(var i = 0.; i < 1 + 30.*pow(sin(float(id.x))*0.5 + 0.5,20.); i+=1){
        var st = 0.3 + segb*0.5;

        let sc = 5.0;
        let offs = v3(0,0,19. + T + floor(T)*0.);
        // var n = v3(
        //     cyclicNoise(p*sc + v3(st,0,0) + offs),
        //     cyclicNoise(p*sc + v3(0,st,0) + offs),
        //     cyclicNoise(p*sc + v3(0,0,st) + offs)
        // );
        var n = v3(
            cyclicNoise(p*sc + v3(st,0,0) + offs) - 
            cyclicNoise(p*sc - v3(st,0,0) + offs),
            cyclicNoise(p*sc + v3(0,st,0) + offs) -
            cyclicNoise(p*sc - v3(0,st,0) + offs),
            cyclicNoise(p*sc + v3(0,0,st) + offs) -
            cyclicNoise(p*sc - v3(0,0,st) + offs)
        );

        var dir = (n)*1.;

        p += (dir)*0.1;
        accl += length(dir);
        // p = normalize(p)*pow(length(p),3.)*0.3;

        // to_bucket( draw_point( p, v3(1,1,1), .0002, 0.4, BLEND_OP ));
        // to_bucket( draw_rect( p, v3(0.2,0.02,0.1),0.4,v3(1,1,1), 0.3, BLEND_OP ));

        var col = v3(0.);
        col = mix_cols(i*0.1 + hash_f_s(float(seed)*0.000001));
        var l = pow(hash_f(),5.)*0.3;


        if(hash_f_s(seg+ 25)<0.4){
            var l = pow(hash_f(),0.5)*2.;
        }

        var w = 0.001;
        if(hash_f_s(seg + 125) < 0.5){
            w = 0.001;
        }
        // w *= 2. + i/30*5.;
        w *= 2. + accl/3*2.*0.2;
        to_bucket( draw_line( p,p+dir*l, col, w, 1., BLEND_OP  ) );
    }
}


fn blend(canv_col: ptr<function, v3>, obj_col: v3, obj: float, op: float, blend: float){
    if(blend == BLEND_ADD){
        *canv_col += obj_col * obj * op;
    }else if(blend == BLEND_HUE){
        *canv_col = mix(*canv_col, obj_col, obj*op); // TODO
    }else if(blend == BLEND_INV){
        if(obj < .5){
            return;
        }
        *canv_col = mix(*canv_col, obj_col - *canv_col, obj*op);
    }else if(blend == BLEND_OP){
        *canv_col = mix(*canv_col, obj_col, obj*op);
    }

    if(custom.BIN_DBG > 0.01){
        let amt = 20*(custom.BIN_DBG/MAX_SHAPES_PER_BIN);
        (*canv_col).r -= amt;
        (*canv_col).b -= amt;
    }
}

#workgroup_count shapes_draw BIN_RES_X BIN_RES_Y WG_PER_BIN
@compute @workgroup_size(WG_RES, WG_RES,1)
fn shapes_draw(
    @builtin(workgroup_id) wid: uint3,
    @builtin(global_invocation_id) gid: uint3,
    @builtin(local_invocation_id) lid: uint3,
    ) {
    Ru = uint2(textureDimensions(screen)); R = v2(Ru); 
    // seed = hash_u(id.x + hash_u(Ru.x*id.y*200u)*20u + hash_u(id.y)*250u + hash_u(id.x*21832)); seed = hash_u(seed); seed = hash_u(seed); seed = hash_u(seed); seed = hash_u(seed)

    var bid = wid.xy;

    var iuv = 
        // global offs
        wid.xy * WG_RES * SQRT_WG_PER_BIN
        // local offs
        + lid.xy;
    

    iuv.y += WG_RES*(wid.z/SQRT_WG_PER_BIN);
    iuv.x += WG_RES*(wid.z%SQRT_WG_PER_BIN);

    if(iuv.x >= Ru.x || iuv.y >= Ru.y ){ return; }

    let screen_idx = iuv.x + iuv.y * Ru.x;

    var uv = (v2(iuv) - 0.5*R)/min(R.x,R.y);

    let bin = &shape_data.bins[bid.x][bid.y];
    let shapes = &(*bin).shapes;
    let shape_cnt = atomicLoad(&(*bin).shape_cnt);
    
    var col = screen_data[screen_idx]*v3();
    if( 
        (wid.z%SQRT_WG_PER_BIN == 0 && lid.x == 0) || 
        (wid.z%WG_PER_BIN < SQRT_WG_PER_BIN && lid.y == 0)
        ){
        col += 0.022;
    }

    for(var i = 0u; i < shape_cnt; i++){
        let shape = (*shapes)[i];
        if(shape.ty == SHAPE_LINE){
            let w = shape.data[3].x;
    
            var a = shape.data[0].xy;
            var b = shape.data[1].xy;

            let sd = sdLine(uv,a,b) - w;
            let line = smoothstep(1./R.y,0.,sd);

            blend(&col, shape.data[4], line, shape.data[3].y, shape.data[3].z);
        } else if (shape.ty == SHAPE_POINT){
            let sd = length(uv - shape.data[0].xy) - shape.data[3].x;
            let pt = smoothstep(1./R.y,0.,sd);

            blend(&col, shape.data[4], pt, shape.data[3].y, shape.data[3].z);
        } else if (shape.ty == SHAPE_TRI){
            var a = shape.data[0].xy;
            var b = shape.data[1].xy;
            var c = shape.data[2].xy;
                
            var n = normalize(a.xy-b.xy);
            n = vec2f(n.y,-n.x);
            var sdtri = dot(uv - a.xy, normalize(n.xy));
            
            n = normalize(b.xy-c.xy);
            n = vec2f(n.y,-n.x);
            sdtri = max(sdtri,dot(uv - b.xy, normalize(n.xy)));
            
            n = normalize(c.xy-a.xy);
            n = vec2f(n.y,-n.x);
            sdtri = max(sdtri,dot(uv - a.xy, normalize(n.xy)));

            let tri = smoothstep(1./R.y,0.,sdtri);

            blend(&col, shape.data[4], tri, shape.data[3].y, shape.data[3].z);
        } else if (shape.ty == SHAPE_RECT){
            let sd = sdBox(
                rot(shape.data[2].x)*(uv - shape.data[0].xy),
                shape.data[1].xy
            );
            let box = smoothstep(1./R.y,0.,sd);
            blend(&col, shape.data[4], box, shape.data[3].y, shape.data[3].z);
        }
    }
    screen_data[screen_idx] = col;
}


#include "Dave_Hoskins/hash"

#define noise_simplex_2d_hash hash22
#include "iq/noise_simplex_2d"

@compute @workgroup_size(16, 16)
fn Draw(@builtin(global_invocation_id) id: uint3) {
    INIT
    if (id.x >= Ru.x || id.y >= Ru.y) { return; } 
    var uv = (U - 0.5*R)/R.y;
    
    let idx = uv_to_pixel_idx(uv);
    // let idx = id.x + id.y * Ru.y;

    var col = v3(0);

    
    col += screen_data[idx];

    col -= pow(hash_v3_s(float(idx)*0.01),v3(15.))*0.4;
    col *= 1. - cyclicNoise(v3(uv.xy*0.1,floor(T)))*0.01;
    col -= pow(noise_simplex_2d( 46.0*uv )*0.5 + 1.,2.4)*0.05;
    col = pow(col,v3(0.9,1.0,1.));

    textureStore(screen, id.xy, v4(col, 1.));
    screen_data[idx] = v3(0);
}
