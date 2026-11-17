const VIDEO_DURATION: f32 = 217.766667; // 视频总时长(秒)
const VIDEO_FRAMES:i32 = 3267;        // 总帧数
const VIDEO_CELLS:i32 = 817;        // 图集格子数(每格 4 帧)
const VIDEO_GRID_X:i32 = 29;        // 图集列数
const VIDEO_GRID_Y:i32 = 29;        // 图集行数
const VIDEO_ATLAS_W:i32 = 5568;        // 图集像素宽
const VIDEO_ATLAS_H:i32 = 3132;        // 图集像素高

const KEY_SPACE:u32 = 32;
@group(0) @binding(0)var<storage, read_write> curSecond: f32;
@group(0) @binding(1)var<storage, read_write> isPressSpace: f32;

fn KeyIsPressed(key:u32)->bool 
{
    return keyDown(key);
}

// GLSL 的 mod 是 floor 语义，WGSL 的 % 是截断语义；参数可能为负时用这个
fn modFloor(x: f32, y: f32) -> f32 
{
    return x - y * floor(x / y);
}

// 返回 true 表示亮度低于阈值（对应原 GLSL 的 TRUE）
fn lumaBelowThreshold(pixel: vec2u) -> bool
{
    if(KeyIsPressed(KEY_SPACE))
    {
        if(isPressSpace == 1)
        {

        }
        else
        {
            curSecond = time.elapsed;
            isPressSpace = 1;
        }
    }
    else
    {
        isPressSpace = 0;
    }
    // ★ 像素坐标 -> 归一化 uv（原来缺的就是这一步）
    let dims = vec2f(textureDimensions(screen));
    let uv   = vec2f(f32(pixel.x), dims.y - f32(pixel.y)) / dims;

    // 循环播放：当前时间 -> 帧号
    var t = modFloor(time.elapsed, VIDEO_DURATION); //改成var才能重新赋值
    if(KeyIsPressed(KEY_SPACE))
    {
        t = modFloor(curSecond, VIDEO_DURATION);
    }
    let f = i32(floor(t / VIDEO_DURATION * f32(VIDEO_FRAMES)));

    // 帧 -> 格子(每格 4 帧)与通道(R/G/B/A)
    let c  = f / 4;
    let ch = f - c * 4;
    let gx = c % VIDEO_GRID_X;
    let gy = c / VIDEO_GRID_X;

    let cell = vec2f(1.0) / vec2f(f32(VIDEO_GRID_X), f32(VIDEO_GRID_Y));
    let px   = vec2f(1.0) / vec2f(f32(VIDEO_ATLAS_W), f32(VIDEO_ATLAS_H));

    let vRow    = f32(gy);
    let atlasUV = vec2f(f32(gx), vRow) * cell
                + px * 0.5
                + vec2f(uv.x, 1.0 - uv.y) * (cell - px);

    let vid  = textureSampleLevel(channel0, bilinear, atlasUV, 0.0);
    let luma = vid[ch];

    return luma < 0.5;
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) 
{
   
    // Time varying pixel colour
    var col = vec4f(1.,1.,1.,1.);

    if(lumaBelowThreshold(id.xy))
    {
        col = vec4f(0.,0.,0.,1.);
    }

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, col);
}

//LoongLy Software 2026
//LZX-2026-11-16-001

