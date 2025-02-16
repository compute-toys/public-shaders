
fn f_n_sdf_heart(
    o: vec2f
)->f32{
    let o2 = vec2f(abs(o.x), o.y)+vec2f(0.,-.5);
    let n1 = dot(o2, vec2f(1.,1.));
    let n2 = length(o2-vec2f(.25,-.5))-.25;
    var n = max(n1,n2);
    let nt = o.y*2.;
    let n3 = nt*n1+(1.-nt)*n2;
    //n = max(n, n3);
    // if(o2.y > -.48){
    //     n = n1;
    // }
    return n3;
}
@compute @workgroup_size(16, 16)
fn main_image(
    @builtin(global_invocation_id) 
    o_trn_pixel: vec3u
) {
    // Viewport resolution (in pixels)
    let o_scl_screen = textureDimensions(screen);
    // Prevent overdraw for workgroups on the edge of the viewport
    if (o_trn_pixel.x >= o_scl_screen.x || o_trn_pixel.y >= o_scl_screen.y) { return; }
    // Pixel coordinates (centre of pixel, origin at bottom left)
    let o_trn_nor_pix = (vec2f(o_trn_pixel.xy)-vec2f(o_scl_screen)*.5)
        /vec2f(o_scl_screen).yy;
    let n = f_n_sdf_heart(o_trn_nor_pix-vec2(0.,-0.2));
    var o_col: vec3f;
    let oc1 = vec3f(.1,.8,.1);
    let oc2 = vec3f(.8,.12,.1);
    let nsin = sin(n*99.+time.elapsed)*0.1;
    let nabs = abs(n);
    o_col = nabs*oc1+nsin;
    if(n < .0){
        o_col =nabs*oc2+nsin;
    }
    // Convert from gamma-encoded to linear colour space
    //o_col = pow(o_col, vec3f(2.2));

    // Output to screen (linear o_colour space)
    textureStore(screen, o_trn_pixel.xy, vec4f(o_col, 1.));
}
