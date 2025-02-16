
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let p = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);

    //Initialize hue and clear fragcolor
    var h=vec4f(0);
    var c=vec4f(1);
    
    //Resolution for scaling
    var r = vec2f(screen_size);
    //Alpha, length, angle
    var A=0f;
    var l=0f;
    var a=0f;
    //Loop through layer
    for(var i=0.6; i>0.1; i-=0.1)
    {
        //Smoothly rotate a quarter at a time
        a=(time.elapsed+i)*4;
        a-=sin(a); a-=sin(a);

        //Rotate
        var t = cos(a/4+vec2f(0,11));
        var R = mat2x2(t.x, -t.y, t.y, t.x);

        //Scale and center
        var u =(p*2 - r)/ r.y;
        //Compute round square SDF
        u -= R*clamp(u*R,-vec2f(i),vec2f(i));
        l = max(length(u),0.1);
        //Compute anti-aliased alpha using SDF
        A = min((l-0.1)*r.y/5,1);
        //Pick layer color
        h = sin(i*10+a/3+vec4(1,3,5,0))/5+0.8;
        //Color blending and lighting
        c = mix(h,c,A) * mix(h/h,h+A*u.y/l/2,0.1/l);
    }

    // Output to screen (tanh tonemap)
    textureStore(screen, id.xy, tanh(c*c));
}
