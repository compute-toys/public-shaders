#storage camera vec3f
fn imod(a:vec3i,b:vec3i)->vec3i{
    return (a%b+b)%b;
}
fn fmod(a:vec3f,b:vec3f)->vec3f{
    return fract(a / b) * b;
}
@compute @workgroup_size(1, 1)
#dispatch_once initialization
fn initialization() {
    camera = vec3f(7);
}

@compute @workgroup_size(16,16,1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3u){
        // Prevent overdraw for workgroups on the edge of the viewport

    let res = textureDimensions(screen);
    if (global_id.x >= res.x || global_id.y >= res.y) { return; }

    let p = (vec2f(global_id.xy) - vec2f(res.xy/2))/vec2f(res.yy) * 2.0;
    let a = (vec2f(mouse.pos)/vec2f(res)-vec2f(0,.5))*vec2f(10,4);
    let rot_y = mat3x3f(cos(a.x),.0,sin(a.x),
                        .0,1.,.0,
                        -sin(a.x),.0,cos(a.x));
    let rot_x = mat3x3f(1.,.0,.0,
                        .0,cos(a.y),-sin(a.y),
                        .0,sin(a.y),cos(a.y));
    let ray_dir = rot_y * rot_x * vec3f(p,1.0);
    if(all(global_id.xy==vec2u(0,0))){
        camera += rot_y * vec3f(select(0.,1.,keyDown(68))-select(0.,1.,keyDown(81)),
                                select(0.,1.,keyDown(70))-select(0.,1.,keyDown(82)),
                                select(0.,1.,keyDown(90))-select(0.,1.,keyDown(83)))*0.1;
    }
    let ray_pos = camera;

    var scale = 1.0;
    let size = vec3f(f32(20),f32(10),f32(20));
    let size2 = size * vec3f(2) + vec3f(1);
    let abs_ray_dir = abs(ray_dir);
    let delta_dist = 1 / abs_ray_dir;


    let ray_step = sign(ray_dir);
    var side_dist = (ray_step*.5+.5-ray_step*fract(ray_pos))*delta_dist;
    var map_pos = ray_pos;
    var old_chunk_pos = vec3i(floor(map_pos /8.0));
    var mask = vec3f(0);
    var i=0;
    for(i=0;i<300;i++) {
        let floor_pos = floor(map_pos/ scale);
        let chunk_pos = vec3i(floor(map_pos / 8.0));
        let grid_pos = fmod(floor_pos+size,size2)-size;
        let id = dot(vec3i(grid_pos),vec3i(1,i32(size2.x),i32(size2.x * size2.y)));
        var hit=false;
        if(scale==8.0){
            hit = dot(floor_pos,floor_pos)%30 >5*5;
            if(hit){
                scale=1.0;

                let start_side_dist = (ray_step*.5+.5-ray_step*fract(ray_pos));
                side_dist = (floor((dot(side_dist-delta_dist, mask)*abs_ray_dir)*8-start_side_dist) + start_side_dist+1)*delta_dist;
                old_chunk_pos = chunk_pos;
                continue;
            }
        }
        else{
            let f = fract(floor_pos/8)-.5;
            hit = dot(f,f) <0.1;
            if(hit){
                break;
            }
            if(any(old_chunk_pos!=chunk_pos)){
                scale = 8.0;
                let start_side_dist = (ray_step*.5+.5-ray_step*fract(ray_pos/scale));
                side_dist = (floor((dot(side_dist-delta_dist, mask)*abs_ray_dir)/8-start_side_dist+0.5*mask) + start_side_dist+1)*delta_dist;
                old_chunk_pos = chunk_pos;
                continue;
            }
        }
        old_chunk_pos = chunk_pos;
        mask = step(side_dist.xyz, side_dist.yzx) * step(side_dist.xyz, side_dist.zxy);
        map_pos = ray_pos + ray_dir * dot(side_dist*scale, mask) + ray_step * mask * 0.1;
        side_dist += mask * delta_dist;
    }
    textureStore(screen, global_id.xy, vec4f(mask,1) * exp2(- f32(i)/30));
}