//based on https://www.youtube.com/watch?v=tCsl6ZcY9ag
//what happens if its extended into the time dimension ... thinking emoji :/
//here you have more control with sliders
//#dispatch_once main_image
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id3: vec3u)
{
    var screen_size = textureDimensions(screen);
    if(id3.x >= screen_size.x){return;}
    if(id3.y >= screen_size.y){return;}
    var res = vec2f(screen_size);
    var u   = vec2f(id3.xy)+ .5f;
        u   = 2f*(2f*u-res.xy)/res.y;
        u  *= vec2f(custom.a,custom.b);
    var ys = abs(u.y);
    var x  = 0f;
    var y  = 0f;
    var y2 = 0f;
    var a  = vec4f(0);
    for(var j = 0f; j < 11f; j+=1f){
    for(var i = 0f; i < 222f; i+=1f){
        x = abs(u.x-log2(1f+i)+log2(1f+j))*custom.c;
        y = x*pow(2f,-11f*x)/(1f+i+j);
        var c = cos(log2(1f+i)*custom.d+vec4f(1,2,3,4)+custom.e)*.5f+.5f;
        a += c*f32(y+y2>ys && y2<ys)/(1f+j);
        y2 += y;
    }}
    textureStore(screen, id3.xy, a);
}