fn f_o_col_grid(
    o_trn: vec2f, 
    n_aa: f32
)->vec3f{

    let nf1 = 10.;
    let o2 = abs(fract(o_trn * vec2f(nf1)));
    let on = vec2f(
        smoothstep(0.02+n_aa,0.0, o2.x),
        smoothstep(0.02+n_aa,0.0, o2.y)
    );
    var o_col = vec3f(on.x+on.y)*.2;

    let o22 = abs(fract(o_trn * vec2f(nf1*2.)));
    let on22 = vec2f(
        smoothstep(0.2+n_aa,0.0, o22.x),
        smoothstep(0.2+n_aa,0.0, o22.y)
    );
    o_col += vec3f(on22.x+on22.y)*.02;

    return o_col;
}

@compute @workgroup_size(16, 16)
fn main_image(
    @builtin(global_invocation_id) 
    o_trn_pixel: vec3u
) {
    // Viewport resolution (in pixels)
    let o_scl_screen = textureDimensions(screen);

    let n_aa = (1./ f32(o_scl_screen.x))*3.;
    // Prevent overdraw for workgroups on the edge of the viewport
    // if (o_trn_pixel.x >= o_scl_screen.x || o_trn_pixel.y >= o_scl_screen.y) { return; }
    // Pixel coordinates (centre of pixel, origin at bottom left)
    let o_ihat = vec2f(1.,0.);
    let o_jhat = vec2f(0.,1.);
    let n_t = sin(time.elapsed)*.5+.5;
    let o_ihat_transformed = o_ihat + vec2f(0., -2*n_t);
    let o_jhat_transformed = o_jhat + vec2f(-1.*n_t, 3*n_t);
    let o_trn_nor_pix = (vec2f(o_trn_pixel.xy)-vec2f(o_scl_screen)*.5)
        /vec2f(o_scl_screen).yy;
    let o_col1 = f_o_col_grid(o_trn_nor_pix, n_aa);

    
    let o_col2 = f_o_col_grid(
        vec2f(
            o_trn_nor_pix.x * o_ihat_transformed.x + o_trn_nor_pix.y * o_jhat_transformed.x, 
            o_trn_nor_pix.x * o_ihat_transformed.y + o_trn_nor_pix.y * o_jhat_transformed.y, 
        )
        , n_aa)*vec3f(0.1, 0.6, 1.);

    let o_col = o_col1 + o_col2;
    // Output to screen (linear o_colour space)
    textureStore(screen, o_trn_pixel.xy, vec4f(o_col, 1.));
}
