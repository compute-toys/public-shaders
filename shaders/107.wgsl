// @nagakagachi
//Note that it does not take into account thread contention for reading and writing to the particle buffer.

#define SIM_GROUP_SIZE 64u

#define NUM_PARTICLE 128
// 本来ならThreadGroupSizeとパーティクル数からプリプロセッサで最小限Group数を決めたい.
// しかし計算を伴うマクロ展開にまだ対応してなさそうなのでとりあえず確実なGroupCountを指定.
#define SIM_PARTICLE_GROUP_COUNT NUM_PARTICLE
//#define SIM_PARTICLE_GROUP_COUNT (NUM_PARTICLE + SIM_GROUP_SIZE - 1) / SIM_GROUP_SIZE

struct Particle {
    pos : float3,
    vel : float3,
    life : float,
};

// Particle Data Buffer.
#storage buffer_particle array<Particle,NUM_PARTICLE>

// Particle初期化Pass.
#workgroup_count InitPass SIM_PARTICLE_GROUP_COUNT 1 1
@compute @workgroup_size(SIM_GROUP_SIZE, 1)
fn InitPass(@builtin(global_invocation_id) id : uint3)
{
    let screen_size = uint2(textureDimensions(screen));

    // パッファ範囲外リターン.
    if(id.x >= NUM_PARTICLE) { return;}

    // 簡易な初期化パス.
    // システムによってBuffer要素がすべてゼロクリアされているものとして初回のみ実行.
    if(0.0 >= buffer_particle[id.x].life)
    {
        // 適当な位置ズラシ配置.
        let noise2 = float2(sin(float(id.x)*1.2345), sin(float(id.x)*3.1433254)) * 0.5 + 0.5;
        //let init_pos = float2(float(id.x%screen_size.x), float(id.x/screen_size.x));
        let init_pos = noise2 * 0.5 * float2(screen_size);
        
        buffer_particle[id.x].pos = float3(float(init_pos.x), float(init_pos.y), 0.0);
        buffer_particle[id.x].vel = float3(0.0);
        buffer_particle[id.x].life = 1.0; // 初回実行完了
    }
}

// Particle更新Pass.
#workgroup_count SimPass0 SIM_PARTICLE_GROUP_COUNT 1 1
@compute @workgroup_size(SIM_GROUP_SIZE, 1)
fn SimPass0(@builtin(global_invocation_id) id : uint3)
{
    let screen_size = uint2(textureDimensions(screen));
    // パッファ範囲外リターン.
    if(id.x >= NUM_PARTICLE) { return;}

    // 引力定数.
    let k_force_rate = 200.0;
    let k_vel_atten = 0.99;

    // 注意!!
    //  本来であればThread間でのbuffer_particle書き込みの競合を考慮してダブルバッファやatomic操作をすべき.
    //  今回は検証のため, 気にせず他Threadが同時に書き込んでいる可能性のある要素にアクセスしている.

    // 画面中心への引力.
    let to_center = float3(float2(screen_size), 0.0) * 0.5 - buffer_particle[id.x].pos;
    // 粒子間引力計算(Bruteforce).
    var next_vel = normalize(to_center) * 5000.0;
    for(var i=0u; i < NUM_PARTICLE; i++)
    {
        if(id.x == i) {continue;}
        
        let dist = buffer_particle[id.x].pos - buffer_particle[i].pos;
        let force_point_dist = (-length(dist));

        let intensity = k_force_rate * 1.0/(force_point_dist + 0.001);
        let dir = normalize(dist);
        // 引力合計.
        next_vel += intensity * dir;
    }
    // 更新.
    buffer_particle[id.x].pos += buffer_particle[id.x].vel * time.delta;
    buffer_particle[id.x].vel = buffer_particle[id.x].vel * k_vel_atten + next_vel * time.delta;
}

// 描画Pass.
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    // Viewport resolution (in pixels)
    let screen_size = uint2(textureDimensions(screen));
    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = float2(float(id.x) + .5, float(screen_size.y - id.y) - .5);
    // Normalised pixel coordinates (from 0 to 1)
    let uv = fragCoord / float2(screen_size);


    var col = float3(0.0);
    
    // ピクセルの最近傍パーティクル距離計算.
    let p_pos = float2(id.xy);
    var min_dist = 100000.0;
    // 粒子間引力計算(Bruteforce).
    for(var i=0u; i < NUM_PARTICLE; i++)
    {
        let dist = distance(buffer_particle[i].pos.xy, p_pos);
        min_dist = min(dist, min_dist);
    }

    // 最近傍パーティクル距離から適当に色計算.
    var min_dist_rate = min_dist / 70.0;
    min_dist_rate = pow(min_dist_rate, 1.0/2.0);

    col = float3(1.0 - saturate(min_dist_rate));

    // Output to screen (linear colour space)
    textureStore(screen, int2(id.xy), float4(col, 1.));
}
