fn pcg(n: u32) -> u32 {
    var h = n * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    return (h >> 22u) ^ h;
}

fn getP(coord: vec2f, mx: int, my: int, f: uint) -> float4 {
    var next = int2(coord);
    next.x+=mx;
    next.y+=my;
    return float4(textureLoad(pass_in, next, f, 0));
}

fn setP(coord: vec2f, mx: int, my: int, f: uint, c: float4) {
    var next = int2(coord);
    next.x+=mx;
    next.y+=my;
    textureStore(pass_out, next, f, c);
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    var fragCoord = vec2f(f32(id.x) + .5, f32(id.y) - .5);
    let uv = fragCoord / vec2f(screen_size);

    // Define boundary
    // Get existing pixel colour
    var vel = abs(sin(f32(pcg(u32(time.elapsed*fragCoord.x)))))*0.03;
    var border = uint(8);
    var flip = time.frame%2;
    var flop = 1-flip;
    var flap = int(flip);
    if(flap==0){flap = -1;}
    var flep = int(flop);
    if(flep==0){flep = -1;}
    var old = float4(textureLoad(pass_in, int2(fragCoord), flop, 0));
    var col = vec3f(0);
    if (mouse.click>0 && abs(fragCoord.x - f32(mouse.pos.x)) < 4 && abs(fragCoord.y - f32(mouse.pos.y))<4){
        
        col = vec3f(1.0,vel,0.);
        textureStore(pass_out, id.xy, flip, vec4f(col, 1.));
    }

    
    var top = float2(f32(screen_size.x)/2, f32(screen_size.y)/4);
    var bot = float2(f32(screen_size.x)/2, f32(screen_size.y)*.75);
    if(abs(fragCoord.x-f32(screen_size.x)/2)>2||abs(fragCoord.y-f32(screen_size.y)/2)>20){
        if(distance(fragCoord, top)>(f32(screen_size.y)*.24) && distance(fragCoord, top)<(f32(screen_size.y)*.25)){
            col = vec3f(0.5,0.,0.);
            textureStore(pass_out, id.xy, flip, vec4f(col, 1.));
        }
        if(distance(fragCoord, bot)>(f32(screen_size.y)*.24) && distance(fragCoord, bot)<(f32(screen_size.y)*.25)){
            col = vec3f(0.5,0.,0.);
            textureStore(pass_out, id.xy, flip, vec4f(col, 1.));
        }
    }
    if(time.frame<1 && distance(fragCoord, top)<(f32(screen_size.y)*.24) && fragCoord.y>(f32(screen_size.y)*.1)){
            col = vec3f(1.0,0.,0.);
            textureStore(pass_out, id.xy, flip, vec4f(col, 1.));
        }

    if(old.r == 1.){
        
        if(getP(fragCoord, 0, 1, flop).r < 0.4){
            var fallmax = int(1);
            var fall = int(old.y*100);
            for (var i: i32 = 2; i < int(custom.maxspeed); i += 1) {
                if(getP(fragCoord, 0, i, flop).r < 0.4){
                    fallmax+=1;
                    var red = f32(i)/50.;
                    if(fallmax<fall){setP(fragCoord, 0, i, flip, vec4f(red,0.0,0.,1.));}
                } else {
                    break;
                }
            }
            
            if(fall>fallmax){fall=fallmax;} else {old.y += vel;}
            setP(fragCoord, 0, fall, flip, old);

            
        } else if(getP(fragCoord, flap, -1, flop).r < 0.4 &&getP(fragCoord, flap*2, 1, flop).r < 0.4 &&getP(fragCoord, flap, 1, flop).r < 0.4 && getP(fragCoord, flap, 0, flop).r  < 0.4){
            old.y = vel;
            
            setP(fragCoord, flap, 0, flip, old);
        } else {
            old.y = 0.0;
            
            setP(fragCoord, 0, 0, flip, old);
        }

    }
        textureStore(pass_out, id.xy, flop, vec4f(0.,0.,0., 1.));
        textureStore(screen, id.xy, float4(textureLoad(pass_in, int2(fragCoord), flop, 0)));
}
