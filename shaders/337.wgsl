// Include the following library to enable string literals
#include <string>

#define DEBUG false

#define TERMINAL_ROWS 10
#define TERMINAL_COLS 80
#define TERMINAL_CHAR_WIDTH 15
#define TERMINAL_CHAR_HEIGHT 25

#define PI  3.14159265358
#define TAU 6.28318530718

var<private> terminal_cursor: uint2;

#storage terminal_grid array<array<uint,TERMINAL_COLS>,TERMINAL_ROWS>

fn terminal_write_char(ascii: uint)
{
    if (ascii == 0) {
    } else if (ascii == 0x0a) { // '\n'
        terminal_cursor.x = 0u;
        terminal_cursor.y += 1u;
    } else {
        let x = terminal_cursor.x%TERMINAL_COLS;
        let y = terminal_cursor.y%TERMINAL_ROWS;
        terminal_grid[y][x] = ascii;
        terminal_cursor.x += 1u;
    }
}

fn terminal_write(s: String) {
    for (var i = 0u; i < s.len; i++) {
        let ascii = s.chars[i];
        terminal_write_char(ascii);
    }
}

fn terminal_writei_auto(x_: int)
{   
    let x = float(x_);
    if(x<0.0)
    {
        terminal_write_char(0x2d); //minus sign
    }

    let d = 1 + int(floor(log(abs(x)+1)/log(10.0)));
    var d0 = pow(10, float(d-1));
    for(var i = 0; i < d; i++)
    {
        let digit = uint(abs(x)/d0); 
        terminal_write_char(0x30 + (digit % 10));
        d0 /= 10.0;
    }
}

fn terminal_writei(x_: int, d: int)
{   
    let x = float(x_);
    var d0 = pow(10, float(d-1));
    for(var i = 0; i < d; i++)
    {
        let digit = uint(abs(x)/d0); 
        terminal_write_char(0x30 + (digit % 10));
        d0 /= 10.0;
    }
}

fn terminal_writef(x: float, fraction_digits: int)
{
    let m = pow(10.0, float(fraction_digits));
    if(x<0.0)
    {
        terminal_write("-"); //minus sign
    }
    else
    {
        terminal_write(" ");
    }
    terminal_writei_auto(int(abs(x)));
    terminal_write(".");
    terminal_writei(int(fract(abs(x)) * m), fraction_digits);
}

fn terminal_write2f(x: float2, fd: int)
{
    terminal_write("(");
    terminal_writef(x.x,fd);
    terminal_write(",");
    terminal_writef(x.y,fd);
    terminal_write(")");
}

fn terminal_write3f(x: float3, fd: int)
{
    terminal_write("(");
    terminal_writef(x.x,fd);
    terminal_write(",");
    terminal_writef(x.y,fd);
    terminal_write(",");
    terminal_writef(x.z,fd);
    terminal_write(")");
}

fn terminal_write4f(x: float4, fd: int)
{
    terminal_write("(");
    terminal_writef(x.x,fd);
    terminal_write(",");
    terminal_writef(x.y,fd);
    terminal_write(",");
    terminal_writef(x.z,fd);
    terminal_write(",");
    terminal_writef(x.w,fd);
    terminal_write(")");
}

fn terminal_write4x4f(m: float4x4, fd: int)
{
    terminal_write4f(m[0],fd);
    terminal_write("\n");
    terminal_write4f(m[1],fd);
    terminal_write("\n");
    terminal_write4f(m[2],fd);
    terminal_write("\n");
    terminal_write4f(m[3],fd);
}

fn terminal_clear() {
    for (var i = 0; i < TERMINAL_ROWS; i += 1) {
        for (var j = 0; j < TERMINAL_COLS; j += 1) {
            terminal_grid[i][j] = 0;
        }
    }
}


fn terminal_render(pos: uint2) -> float4 {
    let char_size = uint2(TERMINAL_CHAR_WIDTH, TERMINAL_CHAR_HEIGHT);
    let texel = 1 / float(char_size.y);
    let terminal_size = uint2(TERMINAL_COLS, TERMINAL_ROWS) * char_size;

    if(pos.x > terminal_size.x || pos.y > terminal_size.y)
    {
        return float4(0);
    }

    var uv = float2(pos) / float2(char_size);
    let x = int(uv.x)%TERMINAL_COLS;
    let y = int(uv.y)%TERMINAL_ROWS;

    let ascii = terminal_grid[y][x];

    if (0x20 < ascii && ascii < 0x80) { // printable character
        uv = fract(uv);
        uv = (uv - 0.5) * float2(0.7, 1.0) + 0.5;
        uv += float2(uint2(ascii % 16u, ascii / 16u)); // character lookup
        let sdf = textureSampleLevel(channel1, trilinear, uv / 16., 0.).a;

        var col = float4(0);
        let t = custom.terminal_border_thickness;
        col = mix(col, float4(0,0,0,1), smoothstep(texel, - texel, sdf - 0.5 - t));
        col = mix(col, float4(1,1,1,1), smoothstep(texel, - texel, sdf - 0.5));
        return col;
    }
    return float4(0);
}

fn minkowski_metric(position: float4) -> float4x4
{
    return float4x4(
        -1.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0, 
         0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 1.0
    );
}

fn minkowski_inv_metric(position: float4) -> float4x4
{
    return float4x4(
        -1.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0, 
         0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 1.0
    );
}

// https://en.wikipedia.org/wiki/Schwarzschild_metric#Alternative_coordinates
// Isotropic coordinates
fn schwarzschild_metric(position: float4) -> float4x4
{
    let rs = 2.0 * custom.mass; // Schwarzschild radius

    let R = length(position.yzw);
    let ratio = rs/(4.0 * R);
    let r1 = 1 - ratio;
    let r2 = 1 + ratio;
    let factor1 = r1*r1/(r2*r2);
    let factor2 = r2*r2*r2*r2;
    return float4x4(
        -factor1,       0,       0, 0      ,
               0, factor2,       0, 0      , 
               0,       0, factor2, 0      ,
               0,       0,       0, factor2
    );
} 

fn schwarzschild_inv_metric(position: float4) -> float4x4
{
    let rs = 2.0 * custom.mass; // Schwarzschild radius

    let R = length(position.yzw);
    let ratio = rs/(4.0 * R);
    let r1 = 1 - ratio;
    let r2 = 1 + ratio;
    let factor1 = r1*r1/(r2*r2);
    let factor2 = r2*r2*r2*r2;
    return float4x4(
        -1.0/factor1,       0,       0, 0      ,
               0, 1.0/factor2,       0, 0      , 
               0,       0, 1.0/factor2, 0      ,
               0,       0,       0, 1.0/factor2
    );
} 

fn kerr_newman_params(position: float4) -> float
{
    let M = custom.mass;
    let J = custom.angular_momentum;
    let Q = custom.charge;

    let a = J/M;
    let a2 = a*a;
    let a4 = a2*a2;
    let R2 = dot(position.yzw,position.yzw);
    let R4 = R2*R2;

    let x = position.y;
    let y = position.z;
    let z = position.w;
    let z2 = z*z;
    
    let r2 = 0.5 * ( -a2 + sqrt(a4 - 2.0 * a2 * (R2 - 2.0 * z2) + R4) + R2);
    let r4 = r2*r2;
    let r = sqrt(r2);
    let k = float4(1.0, (r*x+a*y)/(r2+a2), (r*y-a*x)/(r2+a2), z/r);
    let f = r2 / (r4 + a2 * z2) * (2 * M * r - Q * Q);

    return f;
}

// https://en.wikipedia.org/wiki/Kerr%E2%80%93Newman_metric#Kerr%E2%80%93Schild_coordinates
// Kerr–Newman metric
// Kerr–Schild coordinates
fn kerr_newman_metric(position: float4) -> float4x4
{
    let M = custom.mass;
    let J = custom.angular_momentum;
    let Q = custom.charge;

    let a = J/M;
    let a2 = a*a;
    let a4 = a2*a2;

    let R2 = dot(position.yzw,position.yzw);
    let R4 = R2*R2;

    let x = position.y;
    let y = position.z;
    let z = position.w;
    let y2 = y*y;

    let r2 = 0.5 * ( -a2 + sqrt(a4 - 2.0 * a2 * (R2 - 2.0 * y2) + R4) + R2);
    let r4 = r2*r2;
    let r = sqrt(r2);

    let f = r2 / (r4 + a2 * y2) * (2 * M * r - Q * Q);
    let k = float4(1.0, (r*x+a*z)/(r2+a2), y/r, (r*z-a*x)/(r2+a2));
    let k2 = float4x4(
        k.x*k.x, k.x*k.y, k.x*k.z, k.x*k.w,
        k.y*k.x, k.y*k.y, k.y*k.z, k.y*k.w,
        k.z*k.x, k.z*k.y, k.z*k.z, k.z*k.w,
        k.w*k.x, k.w*k.y, k.w*k.z, k.w*k.w
    );
    return minkowski_metric(position) + f*k2;
}

fn kerr_newman_inv_metric(position: float4) -> float4x4
{
    let M = custom.mass;
    let J = custom.angular_momentum;
    let Q = custom.charge;

    let a = J/M;
    let a2 = a*a;
    let a4 = a2*a2;

    let R2 = dot(position.yzw,position.yzw);
    let R4 = R2*R2;
    
    let x = position.y;
    let y = position.z;
    let z = position.w;
    let y2 = y*y;

    let r2 = 0.5 * ( -a2 + sqrt(a4 - 2.0 * a2 * (R2 - 2.0 * y2) + R4) + R2);
    let r4 = r2*r2;
    let r = sqrt(r2);

    let f = r2 / (r4 + a2 * y2) * (2 * M * r - Q * Q);
    let k = float4(1.0, (r*x+a*z)/(r2+a2), y/r, (r*z-a*x)/(r2+a2));
    let k2 = float4x4(
       -k.x*k.x,  k.x*k.y,  k.x*k.z,  k.x*k.w,
        k.y*k.x, -k.y*k.y, -k.y*k.z, -k.y*k.w,
        k.z*k.x, -k.z*k.y, -k.z*k.z, -k.z*k.w,
        k.w*k.x, -k.w*k.y, -k.w*k.z, -k.w*k.w
    ); // yes this really works out to be the inverse

    return minkowski_metric(position) + f * k2;
}

fn metric(position: float4) -> float4x4
{
    // return minkowksy_metric(position);
    // return schwarzschild_metric(position);
    return kerr_newman_metric(position);
}

fn inv_metric(position: float4) -> float4x4
{
    // return minkowksy_inv_metric(position);
    // return schwarzschild_inv_metric(position);
    return kerr_newman_inv_metric(position);
}

fn hamiltonian(position: float4, momentum: float4) -> float
{
    let g_inv = inv_metric(position);
    return 0.5 * dot(g_inv*momentum, momentum);
}

fn lagrangian(position: float4, velocity: float4) -> float
{
    let g = metric(position);
    return 0.5 * dot(g*velocity, velocity);
}

// derivative of the lagrangian w.r.t. position
// this becomes a covector
fn lagrangian_derivative(position: float4, velocity: float4) -> float4 
{
    let l = lagrangian(position, velocity);
    let delta = float2(0.0, custom.derivative_delta*0.01);
    return (float4(
        lagrangian(position + delta.yxxx, velocity),
        lagrangian(position + delta.xyxx, velocity),
        lagrangian(position + delta.xxyx, velocity),
        lagrangian(position + delta.xxxy, velocity)) - l)/delta.y;
}

struct State 
{
  initialized: uint,

  position: float4, // not a vector, spacetime coordinate


  // the velocity, forward, right, upward form an orthonormal frame
  velocity: float4, // time-like vector normalized to -1
  momentum: float4, // time-like covector normalized to -1
  forward:  float4, // space-like vector normalized to 1
  right:    float4, // space-like vector normalized to 1
  upward:   float4, // space-like vector normalized to 1

  previous_mouse_click: int,
  previous_mouse: float2
}

#storage state State

fn handle_camera()
{
    let step: f32 = custom.movement_speed * 0.1;

    if(keyDown(87u)) { state.velocity += step * state.forward; } // W
    if(keyDown(83u)) { state.velocity -= step * state.forward; } // S
    if(keyDown(65u)) { state.velocity -= step * state.right;   } // A
    if(keyDown(68u)) { state.velocity += step * state.right;   } // D

    if(keyDown(82u)) { state.velocity *= float4(1.0, 0.99, 0.99, 0.99);} // R

    var roll_angle: f32 = 0.0f;

    if(keyDown(81u)) { roll_angle += custom.roll_sensitivity * 0.05; } // Q
    if(keyDown(69u)) { roll_angle -= custom.roll_sensitivity * 0.05; } // E

    var prev_right   = state.right;
    var prev_upward  = state.upward;
    state.right  =  cos(roll_angle) * prev_right + sin(roll_angle) * prev_upward;
    state.upward = -sin(roll_angle) * prev_right + cos(roll_angle) * prev_upward;
    
    let dmouse = float2(mouse.pos) - state.previous_mouse;

    if(state.previous_mouse_click==1 && length(dmouse) > 0.0)
    {
        var prev_forward = state.forward;
        var prev_right   = state.right;
        var prev_upward  = state.upward;

        let anglex = custom.pan_sensitivity * 0.01 * dmouse.x;

        state.forward =  cos(anglex) * prev_forward + sin(anglex) * prev_right;
        state.right   = -sin(anglex) * prev_forward + cos(anglex) * prev_right;

        prev_forward = state.forward;

        let angley = - custom.pan_sensitivity * 0.01  * dmouse.y;

        state.forward =  cos(angley) * prev_forward + sin(angley) * prev_upward;
        state.upward  = -sin(angley) * prev_forward + cos(angley) * prev_upward;
    }

}

fn integrate()
{
    let g = metric(state.position);
    let g_inv = inv_metric(state.position);

    let derivative = lagrangian_derivative(state.position, state.velocity);

    state.momentum = g * state.velocity;

    state.momentum += custom.time_step * derivative;
    let momentum_norm2 = dot(g_inv*state.momentum,state.momentum);
    state.momentum /= sqrt(- momentum_norm2);

    state.velocity = g_inv * state.momentum;

    state.position += custom.time_step * state.velocity;
}

// orthonomalize the velocity, forward, right, upward frame
// using gram-schmidt
fn orthonomalize_frame()
{
    let g = metric(state.position);

    state.velocity /= sqrt(-dot(g*state.velocity, state.velocity));

    state.forward  +=       dot(g*state.forward , state.velocity) * state.velocity; //  + here because velocity is normalized to -1
    state.forward  /= sqrt( dot(g*state.forward , state.forward ));

    state.right    +=       dot(g*state.right   , state.velocity) * state.velocity;
    state.right    -=       dot(g*state.right   , state.forward ) * state.forward;
    state.right    /= sqrt( dot(g*state.right   , state.right   ));

    state.upward   +=       dot(g*state.upward  , state.velocity) * state.velocity;
    state.upward   -=       dot(g*state.upward  , state.forward ) * state.forward;
    state.upward   -=       dot(g*state.upward  , state.right   ) * state.right;
    state.upward   /= sqrt( dot(g*state.upward  , state.upward  ));
}

fn initialize()
{
    state.position = float4(0.0, 0.0, 0.0,  1.0) * 40.0;
    state.velocity = float4(1.0, 0.1, 0.0, 0.0);
    state.forward  = float4(0.0, 0.0, 0.0, -1.0);
    state.right    = float4(0.0, 1.0, 0.0,  0.0);
    state.upward   = float4(0.0, 0.0, 1.0,  0.0);
    state.previous_mouse_click = 0;
    state.previous_mouse = float2(0.0, 0.0);
    state.initialized = 1;
}

@compute @workgroup_size(1)
#workgroup_count update_state 1 1 1
fn update_state() 
{
    if(state.initialized != 1)
    {
      initialize();
    }

    integrate();

    handle_camera();

    orthonomalize_frame();
  
    state.previous_mouse_click = mouse.click;
    state.previous_mouse = float2(mouse.pos);
}

@compute @workgroup_size(1)
#workgroup_count singlethreaded 1 1 1
fn singlethreaded() {
    terminal_clear();

    let f =  kerr_newman_params(state.position);
    let g = metric(state.position);
    let g_inv = inv_metric(state.position);

    let M = custom.mass;
    let J = custom.angular_momentum;
    let Q = custom.charge;

    if(J*J/(M*M) + Q*Q > M*M)
    {
        terminal_write("INVALID PARAMS");
    }

    // terminal_write("position:");
    // terminal_write4f(state.position, 3);

    // terminal_write("\nvelocity:");
    // terminal_write4f(state.velocity, 3);

    // terminal_write("\nmomentum:");
    // terminal_write4f(state.momentum, 3);

    // terminal_write(" norm2:");
    // terminal_writef(dot(g*state.velocity,state.velocity), 3);

    // terminal_write("\ng*g_inv:\n");
    // terminal_write4x4f(g*g_inv, 2);

    // terminal_write("\nf:");
    // terminal_writef(f, 2);

    // terminal_write("\nforward: ");
    // terminal_write4f(state.forward, 3);
    // terminal_write(" norm2:");
    // terminal_writef(dot(g*state.forward,state.forward), 3);

    // terminal_write("\nright:   ");
    // terminal_write4f(state.right, 3);
    // terminal_write(" norm2:");
    // terminal_writef(dot(g*state.right,state.right), 3);

    // terminal_write("\nupward:  ");
    // terminal_write4f(state.upward, 3);
    // terminal_write(" norm2:");
    // terminal_writef(dot(g*state.upward,state.upward), 3);

    // terminal_write("mouse.pos.x:");
    // terminal_writei(int(mouse.pos.x), 3);
    // terminal_write("\n");
    // terminal_write("mouse.pos.y:");
    // terminal_writei(int(mouse.pos.y), 3);
    // terminal_write("\n");
    // terminal_write("mouse.click:");
    // terminal_writei(mouse.click, 3);
    // terminal_write("\n");

}


// https://www.shadertoy.com/view/lsc3z4
// Simple star field 
// @bjarkeck
#define STARDISTANCE 150.
#define STARBRIGHTNESS 0.7
#define STARDENSITY 0.03

fn hash13(p: float3) -> float
{
	var q  = fract(p * vec3(.1031,.11369,.13787));
    q += dot(q, q.yzx + 19.19);
    return fract((q.x + q.y) * q.z);
}

fn stars(ray: float3) -> float
{
    let p = ray * STARDISTANCE;
    let h = hash13(p);
    let flicker = cos(time.elapsed * 1. + hash13(abs(p) * 0.01) * 13.) * 0.5 + 0.5;
    let brigthness = smoothstep(1.0 - STARDENSITY, 1.0, hash13(floor(p)));
    return smoothstep(STARBRIGHTNESS, 0., length(fract(p) - 0.5)) * brigthness * flicker;
}

fn spherical_map(p: float3) -> float2 {
    var uv = vec2(atan2(p.z, p.x), asin(p.y));
    return uv * vec2(1.0 / TAU, 1.0 / PI) + 0.5;
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    // Viewport resolution (in pixels)
    let screen_size = uint2(textureDimensions(screen));

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let uv = (2 * float2(id.xy) - float2(screen_size) ) / float(screen_size.y);

    // create an actual light-like vector, i.e. g(.,.) = 0.
    // I think this is done slightly wrong in Mykhailo Moroz's shaders
    // We need to trace the light that reaches the camera backwards through time
    // That is why we do -state.velocity
    // (the assumption here is that the camera moves forward through time)
    let m = normalize(vec3(1.0, uv.x, -uv.y));
    var position = state.position;
    var velocity = -state.velocity + state.forward * m.x + state.right * m.y + state.upward * m.z;
    
    var g = metric(position);
    var momentum = g * velocity;

    var col = float3(0.0, 0.0, 0.0);
    if(DEBUG)
    {
        col = float3(1.0, 1.0, 1.0);
    }

    var sure: bool = false;
    for(var i=0; i<int(custom.march_steps*200); i++)
    {

        // inside event horizon, no hope of escaping
        let rs = 2.0 * custom.mass; // Schwarzschild radius
        let R = length(position.yzw);
        if(R <= rs * 1.1) // error from somwhere :(?
        {
            col = float3(0);
            sure = true;
            break;
        }

        //escaping trajectory, probably?
        let aligned = dot(position.ywz, velocity.ywz);
        let far = length(position.yzw) > (custom.far * 1000.0);
        let escaping = aligned > 0.0;
        if(far && escaping)
        {
            //col = float3(float(i))/400.0;
            let d = normalize(velocity.yzw);
            // col = float3(stars(d)+0.0001);
            //*(abs(d)*0.9+0.1);
            col = textureSampleLevel(channel0, bilinear, spherical_map(d), 0).rgb;
            sure = true;
            break;
        }

        let g_inv = inv_metric(position);

        velocity = g_inv * momentum;
        
        let derivative = lagrangian_derivative(position, velocity);

        // The black hole is located at x=y=z=0
        // far away we can take larger steps
        // closeby we take smaller steps

        let h = custom.step * length(position.yzw);

        momentum += h * derivative;

        // we are allowed to scale the 4-momentum freely; 
        // the geodesic we trace out is unaffected.
                
        // make sure the 4-momentum doesn't grow or schrink in numerical size
        momentum = normalize(momentum);

        // I should add something that makes sure that momentum stays a light-like covector.

        var mdis = 1e20f;
        if(DEBUG)
        {
            let mp = (modf(abs(position)/50.0 + 0.5).fract - 0.5)*50.0;
            let dis1 = length(mp.yz) - 0.1;
            if(dis1 < 0.01)
            {
                col = float3(1.0, 0.0, 0.0);
                sure = true;
                break;
            }
            let dis2 = length(mp.yw) - 0.1;
            if(dis2 < 0.01)
            {
                col = float3(0.0, 1.0, 0.0);
                sure = true;
                break;
            }
            let dis3 = length(mp.zw) - 0.1;
            if(dis3 < 0.01)
            {
                col = float3(0.0, 0.0, 1.0);
                sure = true;
                break;
            }
            mdis = min(min(dis1, dis2), dis3);
        }
        
        // position += custom.step * 0.1 * velocity;
        position += min(h,mdis) * g_inv * momentum;
    }

    // Render terminal overlay
    let text = terminal_render(id.xy);
    col = mix(col, text.rgb, text.a);

    let old_col = textureLoad(pass_in, int2(id.xy), 0, 0);
    let new_col = float4(col, 1.);
    let mix_col = mix(old_col, new_col, 1.0);

    textureStore(pass_out, int2(id.xy), 0, mix_col);

    textureStore(screen, int2(id.xy), mix_col);
}
