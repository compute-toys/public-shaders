// https://www.karlsims.com/rd.html
// Some typical values used, for those interested, are: 
// DA=1.0, 
#define DA 1.0f
// DB=.5, 
#define DB 0.5f
// f=.055, 
#define f_ 0.055f
// k=.062 (f and k vary for different patterns), 
#define k_ 0.062f
// and Δt=1.0. 
#define dt 1.0f

const kernel = mat3x3(
    vec3(.05,  .2,  .05),
    vec3(.2,  -1.,  .2),
    vec3(.05,  .2,  .05)
    );

fn next(id: vec3u) -> vec4f {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    // if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);

    // Normalised pixel coordinates (from 0 to 1)
    let uv: vec2f  = fragCoord / vec2f(screen_size);
    let pxw: vec2f = 1.f / vec2f(screen_size);
    
    // // vec2 cell    = floor(uv * GDW);
    // // vec2 cell_uv = (cell + 0.5) * iGDW;
    
    // vec2  mouse      = iMouse.zw/iResolution.xy;
    // // vec2  mouse_cell = floor(mouse * GDW);
    // // float mouse_cell_dist = length(mouse_cell - cell);
    // //float mouse_dot_color = 1. - step(10., mouse_cell_dist);
    // float mouse_dot_color = 1. - step(.1, length(mouse - uv));
    
    // vec4 prev = texture(iChannel0, uv);
    // if (iFrame == 0) 
    // {
    //     //vec4 noise = texture(iChannel1, cell_uv);
    //     prev = vec4(1.,mouse_dot_color,0.,0.);
    // }
    
    // float A = prev.r;
    // float B = prev.g;
    // float ABB = A * B * B;
    
    // // -----------------
    // // (laplacian part)
    // vec2 LAP = vec2(0);
    // for (int r = -1; r < 2; r++)
    // for (int c = -1; c < 2; c++) 
    // {
    //     vec2 offs = vec2(float(c), float(r));
    //     vec4 color = texture(iChannel0, (uv + offs * pxw));
    //     vec2 attenuated = color.rg * kernel[c + 1][r + 1];
    //     LAP += attenuated;
    // }
    // // -----------------
    
    // float A_ = A + (DA * LAP.r - ABB + f_ * (1.f - A)) * dt;
    // float B_ = B + (DB * LAP.g + ABB - (k_ + f_) * B)  * dt;
    
    // fragColor = vec4(A_, B_, 0., 1.) + vec4(0.,mouse_dot_color,0.,0.);

    return vec4f(0.f);
}

@compute @workgroup_size(16, 16)
fn pass1(@builtin(global_invocation_id) id: vec3u)
{
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);

    // Normalised pixel coordinates (from 0 to 1)
    let uv: vec2f = fragCoord / vec2f(screen_size);

    // Time varying pixel colour
    var col = .5 + .5 * cos(time.elapsed + uv.xyx + vec3f(0.,2.,4.));

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}
