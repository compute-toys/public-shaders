//const N = 1;
const N = 16;

@compute @workgroup_size(N, N)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    var col = vec3f(vec3u((id.x ^ id.y) & N)) / N;
    //var col = vec3f(vec3u((id.x ^ id.y) % N)) / N;
    
    textureStore(screen, id.xy, vec4f(col, 1.));
}
