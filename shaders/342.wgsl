const NUM_CHANNELS = 12u;
//const DATA_BUFFER_SIZE = u32(SCREEN_WIDTH) * u32(SCREEN_HEIGHT) * NUM_CHANNELS;
// const DATA_BUFFER_SIZE = 10972800u; // 1280 * 720 * 12

const DATA_BUFFER_SIZE = SCREEN_WIDTH * SCREEN_HEIGHT * NUM_CHANNELS;

#storage data_buffer array<f32,DATA_BUFFER_SIZE>
fn hash43(p: float3) -> float4 {
  var p4: float4 = fract(float4(p.xyzx)  * float4(.1031, .1030, .0973, .1099));
    p4 += dot(p4, p4.wzxy+33.33);
    return fract((p4.xxyz+p4.yzzw)*p4.zywx);
}

fn data_buffer_idx(x: u32, y: u32, c: u32) -> u32 {
    return c + (x % u32(SCREEN_WIDTH) + y % u32(SCREEN_HEIGHT) * u32(SCREEN_WIDTH) ) * NUM_CHANNELS;
}

fn data_buffer_read_c4(x: u32, y: u32, c: u32) -> float4 {
    let i = data_buffer_idx(x, y, c);
    return float4(data_buffer[i], data_buffer[i + 1u], data_buffer[i + 2u], data_buffer[i + 3u]);
}

fn data_buffer_write_c4(v: float4, x: u32, y: u32, c: u32) {
    let i = data_buffer_idx(x, y, c);
    data_buffer[i] = v[0];
    data_buffer[i+1u] = v[1];
    data_buffer[i+2u] = v[2];
    data_buffer[i+3u] = v[3];
}

var<private> current_index: int2;

fn R(x: i32, y: i32, c: i32) -> float4 {
    return data_buffer_read_c4(u32(current_index.x + x), u32(current_index.y + y), u32(c));
}


fn F(yi: float4, a: array<float4, 4>, b: array<float4, 4>) -> float4 {
    return
        yi[0] * a[0] + abs(yi[0]) * b[0] +
        yi[1] * a[1] + abs(yi[1]) * b[1] +
        yi[2] * a[2] + abs(yi[2]) * b[2] +
        yi[3] * a[3] + abs(yi[3]) * b[3]
    ;
}

fn f4(x: i32, y: i32, z: i32, w: i32) -> float4 {
    return float4(f32(x), f32(y), f32(z), f32(w));
}


fn M(m0: i32, m1: i32, m2: i32, m3: i32,
    m4: i32, m5:i32, m6: i32, m7: i32,
    m8: i32, m9: i32, m10: i32, m11: i32,
    m12: i32, m13: i32, m14: i32, m15: i32) -> array<float4,4> {
        return array<float4,4>(
            float4(f32(m0), f32(m1), f32(m2), f32(m3)),
            float4(f32(m4), f32(m5), f32(m6), f32(m7)),
            float4(f32(m8), f32(m9), f32(m10), f32(m11)),
            float4(f32(m12), f32(m13), f32(m14), f32(m15)),
        );
    }

fn update(band: u32, y: array<float4,6>) -> float4 {
  //#define M mat4x4<f32>
//   #define F(i,_a,_b) {M a=_a,b=_b; float4 yi=y[i]; dx+=G(0)+G(1)+G(2)+G(3);}
  //#define G(i) yi[i]*((yi[i]>0.0)?a[i]:b[i])
//   #define G(i) (yi[i]*a[i]+abs(yi[i])*b[i])
  var dx = float4(0.0);
  if (band == 0u) { dx = f4(17,3,-17,15);
    dx = dx + F(y[0], M(-32,13,-9,20,-3,-58,6,27,2,21,-39,-5,-10,-9,4,-41), M(11,14,2,-34,-1,6,51,-26,-9,-41,-15,-19,9,14,9,-3));
    dx = dx + F(y[1], M(6,-1,-5,4,-5,4,-1,5,-12,13,15,26,0,4,0,0), M(-16,-8,-10,4,1,13,18,-3,-6,-2,-2,3,14,-9,-7,-19));
    dx = dx + F(y[2], M(-3,8,7,7,12,-4,-7,-11,2,-2,-5,1,-2,-2,-2,0), M(-7,-12,-8,0,-11,2,4,12,-17,2,21,10,12,3,4,-10));
    dx = dx + F(y[3], M(23,4,-1,6,-24,11,0,4,7,-1,14,0,-4,8,-4,40), M(2,-1,0,-7,-3,2,0,2,2,1,0,10,11,10,10,-10));
    dx = dx + F(y[4], M(-12,-17,-14,-13,6,7,7,-1,19,17,10,14,4,11,8,10), M(-4,-7,-8,3,-4,6,6,-1,3,0,-6,7,-3,-1,0,-5));
    dx = dx + F(y[5], M(10,14,12,13,-9,-16,-13,-12,5,3,5,-9,-9,-15,-16,6), M(-1,-1,1,-3,-12,-8,-7,6,-13,0,3,0,4,4,5,5));
  } else if (band == 1u) { dx = f4(12,0,-2,-7);
    dx = dx + F(y[0], M(-10,15,11,-2,6,21,-1,12,18,9,-10,16,-27,-12,-3,19), M(-13,16,-21,4,-5,-23,-3,5,-13,-38,-6,19,2,23,2,9));
    dx = dx + F(y[1], M(-81,11,10,11,-6,-70,2,11,21,12,-52,-2,24,-7,-10,-68), M(0,1,-6,-7,20,3,24,15,-12,-6,-4,0,15,-10,-16,8));
    dx = dx + F(y[2], M(-3,7,-7,5,12,0,18,0,-7,-12,2,-6,-8,15,2,-6), M(10,-4,29,-2,8,0,13,1,-5,14,12,-5,-15,16,-26,1));
    dx = dx + F(y[3], M(-2,4,4,2,4,4,3,2,7,-1,-3,5,0,2,16,4), M(0,-3,-2,-1,4,-1,2,1,1,3,1,2,-3,-5,-3,3));
    dx = dx + F(y[4], M(-21,-5,-19,-13,-1,-24,8,2,-10,11,33,-1,-9,2,-1,7), M(2,2,1,-2,-3,-3,17,24,-4,-2,2,-2,10,-6,1,-6));
    dx = dx + F(y[5], M(2,10,2,2,-12,2,-18,-8,5,-6,3,7,-8,-7,0,0), M(-1,-1,3,1,-5,0,-9,-1,-7,8,-2,-4,-3,4,-1,-4));
  } else { dx = f4(-18,9,-2,27);
    dx = dx + F(y[0], M(22,-20,17,19,13,-14,-9,7,-4,-4,-15,-2,-4,9,-16,19), M(0,-28,-6,-12,23,-30,-3,5,-3,37,41,-2,16,-6,-21,3));
    dx = dx + F(y[1], M(-4,-1,-2,-3,3,-8,-1,-14,-6,-15,-11,10,1,-11,-15,-2), M(-9,1,-1,-16,3,11,-4,1,9,-2,-21,-11,0,-8,17,11));


    dx = dx + F(y[2], M(-63,26,2,10,-22,-56,7,4,3,-8,-60,6,0,-2,-2,-66), M(-2,-17,-9,-20,-6,-3,-16,-13,-26,-25,-9,7,23,-9,0,-7));
    dx = dx + F(y[3], M(7,1,2,5,3,3,-4,5,-1,0,-9,-2,9,18,-1,0), M(1,0,1,2,0,3,1,2,-4,-1,-5,-1,4,0,-4,2));
    dx = dx + F(y[4], M(-7,3,16,-12,4,6,-8,4,7,1,-11,10,11,2,-10,12), M(-6,1,4,-9,-2,6,-4,1,15,5,6,5,-1,2,0,-3));
    dx = dx + F(y[5], M(24,7,-6,-3,-12,-30,10,0,-4,1,4,14,-6,4,4,-31), M(-7,4,0,10,16,4,1,-8,-7,-3,-10,-6,15,5,4,3));
  }
  return dx/500.0;
}


@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    let screen_size = uint2(textureDimensions(screen));
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    current_index = int2(int(id.x), int(id.y));

    if (time.frame == 0u) {
        for (var i = 0u; i < NUM_CHANNELS / 4u; i = i+1u) {
            let noise = hash43(float3(float2(id.xy), 0.0) - 0.5);

            data_buffer[data_buffer_idx(id.x, id.y, i +0u)] = noise.x;
            data_buffer[data_buffer_idx(id.x, id.y, i + 1u)] = noise.y;
            data_buffer[data_buffer_idx(id.x, id.y, i + 2u)] = noise.z;
            data_buffer[data_buffer_idx(id.x, id.y, i + 3u)] = noise.w;
        }
    }

    // 2
    let vert_sobel = R(-1, 1, 8) + R(-1, 0, 8)*2.0 + R(-1,-1, 8)
                    -R( 1, 1, 8) - R( 1, 0, 8)*2.0 - R( 1,-1, 8);
    // 1
    let hor_sobel = R( 1, 1, 4)+R( 0, 1, 4)*2.0+R(-1, 1, 4)
                   -R( 1,-1, 4)-R( 0,-1, 4)*2.0-R(-1,-1, 4);

    // 0
    var lap = R(1,1, 0)+R(1,-1, 0)+R(-1,1, 0)+R(-1,-1, 0)
              +2.0*(R(0,1, 0)+R(0,-1, 0)+R(1,0, 0)+R(-1,0, 0))- 12.0*R(0, 0, 0);

    let ys = array<float4, 6>(
        data_buffer_read_c4(id.x, id.y, 0u),
        data_buffer_read_c4(id.x, id.y, 4u),
        data_buffer_read_c4(id.x, id.y, 8u),
        lap,
        hor_sobel,
        vert_sobel,
    );

    let clamp_low = float4(-1.5);
    let clamp_high = float4(1.5);

    let dx0 = clamp(ys[0] + update(0u, ys), clamp_low, clamp_high);
    let dx1 = clamp(ys[1] + update(1u, ys), clamp_low, clamp_high);
    let dx2 = clamp(ys[2] + update(2u, ys), clamp_low, clamp_high);

    data_buffer_write_c4(dx0, id.x, id.y, 0u);
    data_buffer_write_c4(dx1, id.x, id.y, 4u);
    data_buffer_write_c4(dx2, id.x, id.y, 8u);

    textureStore(
        screen,
        int2(id.xy),
        dx0 //+ .5
    );
}