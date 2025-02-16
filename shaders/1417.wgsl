// Copyright 2024 Google LLC.
// SPDX-License-Identifier: Apache-2.0

// Screen space uv corresponding to terrain. 
// Trick is to CAS and previous to do depth testing
#storage texel_camera array<array<atomic<u32>,SCREEN_HEIGHT>,SCREEN_WIDTH>;

// How much pre smmothing to apply: 0 - 20
const SMOOTH_KERNEL_SIZE = 1;

// Must match size of texture in channel0
const FIELD_SIZE = 1024;
#storage field array<array<f32,FIELD_SIZE>, FIELD_SIZE>;

@compute @workgroup_size(16, 16)
fn clear(@builtin(global_invocation_id) id: vec3u) {
    // Clear screen atomic pixels. 
    if (id.x >= SCREEN_WIDTH || id.y >= SCREEN_HEIGHT) { return; }
    atomicStore(&texel_camera[id.x][id.y], 0);
}

fn TexelToUV(id:vec3u) -> vec2f {
    return vec2f(id.xy)/f32(FIELD_SIZE);
}

fn SubTexelToUV(id_sub:vec2f) -> vec2f {
    return id_sub /f32(FIELD_SIZE);
}

//#dispatch_once build_field  -- this doesnt work because texture loading
#workgroup_count build_field 64 64 1
@compute @workgroup_size(16, 16)
fn build_field(@builtin(global_invocation_id) id: vec3u) {
    var sum:f32 = 0.0;
    var factor = 0.0;
    const kernel_size = SMOOTH_KERNEL_SIZE;
    for(var i = -kernel_size ; i <=kernel_size; i++){
        for(var j = -kernel_size ; j <=kernel_size; j++){
            let uv = TexelToUV( vec3u(vec3i(id) + vec3i(i,j,0)));
            let tex = textureSampleLevel(channel0, nearest, uv, 0.0);
            let curr_factor =  f32( (1+kernel_size*2) - (abs(i) + abs(j)));
            sum = sum + tex.r * curr_factor;
            factor = factor + curr_factor;
        }
    }
    if(FIELD_SIZE == 1024){
        field[id.x][id.y]= sum/factor;
    }
    else
    {
        field[id.x][id.y]= 0;
        if(id.x == 2 && id.y ==2){
              field[id.x][id.y]= 1;
        }
    }
}

fn UVToPos(uv:vec2f) -> vec3f{
    var id = vec2u(uv * FIELD_SIZE);
    let f_frac = fract(uv * FIELD_SIZE);
    let r_frac = vec2f(1.0)- f_frac;
    let tex00 = field[id.x][id.y] * r_frac.x * r_frac.y;
    let tex10 = field[id.x+1][id.y]* f_frac.x * r_frac.y;
    let tex01 = field[id.x][id.y+1]* r_frac.x * f_frac.y;
    let tex11 = field[id.x+1][id.y+1]* f_frac.x * f_frac.y;
    let tex = tex00 + tex10 + tex01 + tex11;
    // Fake modelspace position
    let pos = vec3f(uv.x-0.5, uv.y-0.5, (1- tex)*0.5-0.3);
    return pos;
}

fn PosToWorld(posin:vec3f) -> vec3f{
    var pos = posin;
    let theta =  time.elapsed*0.1;
    let rot_mat = mat2x2<f32>(vec2f(cos(theta), sin(theta)),
    vec2f(-sin(theta), cos(theta)) );

    var pos2d = pos.xy;
    pos2d =  rot_mat * pos2d;
    pos.x = pos2d.x;
    pos.y = pos2d.y;
    // Zoom level
    pos = pos * vec3f(0.7);
    return pos;
}

fn PosToCamera(posin:vec3f) -> vec3f{
    var pos = PosToWorld(posin);

    var splat_pos = vec2(pos.y, pos.z);
    let tilt_angle =-0.8;
    let tilt_mat = mat2x2<f32>(vec2f(cos(tilt_angle), sin(tilt_angle)),
    vec2f(-sin(tilt_angle), cos(tilt_angle)) );
    splat_pos = tilt_mat * splat_pos;
    pos.y = splat_pos.x;
    pos.z = splat_pos.y;
    return pos;
}

fn PosToScreenUV(pos:vec3f) -> vec2u {
    return vec2u((pos.xz+0.5) * vec2f(textureDimensions(screen)));
}

fn UVToWord(uv:vec2f) -> uint {
    let x = uint(uv.x * 16384.0);
    let y = uint(uv.y * 16384.0);
    return (y<<16) | x;
}

fn WordToUV(word: uint) -> vec2f {
    let x = f32(word & 0xFFFF) /16384.0;
    let y = f32(word >>16) /16384.0;
    return vec2f(x,y);
}

fn UVToNormal(uv:vec2f) -> vec3f {
    let half_uv = 1.0/ FIELD_SIZE;
    let pos00 = PosToWorld(UVToPos(uv));
    let pos01 = PosToWorld(UVToPos(uv+vec2f(half_uv,0)));
    let pos10 = PosToWorld(UVToPos(uv+vec2f(0.0,half_uv)));
    let dirx = normalize(pos01 - pos00);
    let diry = normalize(pos10 - pos00);
    return cross(dirx,diry);
}


// For testing of 4x4 field
//#workgroup_count splat_camera 1 1 1
//@compute @workgroup_size(4, 4)

// These values related to texture size
#workgroup_count splat_camera 64 64 1
@compute @workgroup_size(16, 16)
fn splat_camera(@builtin(global_invocation_id) id: vec3u) {
    
    let uv = TexelToUV(id);
    var pos = PosToCamera(UVToPos(uv));
    var uv_int = PosToScreenUV(pos);
    var min_screen_int = uv_int;
    var max_screen_int = uv_int;

    // Quad corners should cover screen area even though quad is not actually flat.
    for(var i=0u; i < 2u;i++){
        for(var j=0u; j < 2u;j++){
            let uv_11 = TexelToUV(id + vec3u(i,j,0));
            let pos_11 = PosToCamera(UVToPos(uv_11));
            let uv_int_11 = PosToScreenUV(pos_11);
            let num_iter = vec2u(1,1);
            min_screen_int = min(min_screen_int, uv_int_11);
            max_screen_int = max(max_screen_int, uv_int_11);
        }
    }

    let diff_screen_int = vec2u(max_screen_int-min_screen_int);
    var span_screen_int = max(1, max(diff_screen_int.x,diff_screen_int.y)+1);

    var inv_count = 1.0/f32(span_screen_int);
    for(var i=0u; i <span_screen_int;i++){
        for(var j=0u; j <span_screen_int;j++){
            var uv_shift =  SubTexelToUV(vec2f(id.xy) + inv_count * vec2f(f32(i),f32(j)) );
            var pos = PosToCamera(UVToPos(uv_shift));
            var uv_int = PosToScreenUV(pos);
            if( uv_int.x < 0 || uv_int.y < 0 ||
                uv_int.x >=SCREEN_WIDTH || uv_int.y>=SCREEN_HEIGHT){
                return;
            }
            var current = uint(0);
            var res = atomicCompareExchangeWeak(&texel_camera[uv_int.x][uv_int.y],current, UVToWord(uv));
            while(!res.exchanged){
            current = res.old_value;
            var curr_pos = PosToCamera(UVToPos(WordToUV(current)));
            if(curr_pos.y < pos.y) {
                break; 
            }
            res = atomicCompareExchangeWeak(&texel_camera[uv_int.x][uv_int.y], 
                        current, UVToWord(uv));
            }
        }
    }
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }


    let load_val = atomicLoad(&texel_camera[id.x][id.y]);
    if(load_val == 0) {
        textureStore(screen, id.xy, vec4f(0.0));
        return; 
    }
    let height_uv = WordToUV(load_val);
    let normal_temp = UVToNormal(height_uv);

    var col = vec3f(max(0,dot(-normal_temp,normalize(vec3f(-0.7,-0.2,-1)))));
    col = pow(col, vec3f(2.0));
    textureStore(screen, id.xy, vec4f(col, 1.));
}
