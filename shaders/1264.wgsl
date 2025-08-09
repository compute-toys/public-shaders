/*                                                                
                 ,--.  ,--.                                    
 ,---.  ,--,--.,-'  '-.|  ,---.                                
| .-. |' ,-.  |'-.  .-'|  .-.  |                               
| '-' '\ '-'  |  |  |  |  | |  |                               
|  |-'  `--`--'  `--'  `--' `--'                               
`--'                     ,--.              ,--.                
,--.--. ,---. ,--,--,  ,-|  | ,---. ,--.--.`--',--,--,  ,---.  
|  .--'| .-. :|      \' .-. || .-. :|  .--',--.|      \| .-. | 
|  |   \   --.|  ||  |\ `-' |\   --.|  |   |  ||  ||  |' '-' ' 
`--'    `----'`--''--' `---'  `----'`--'   `--'`--''--'.`-  /  
                                                       `---'   
   
(quadratic bezier curves)

following this article:
https://medium.com/@evanwallace/easy-scalable-text-rendering-on-the-gpu-c3f4d782c5ac

unfortunately, there are some artifacts.
also, no antialiasing

don't miss the debug view and the scaling (see Uniforms)
*/

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Blabla
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);
    let uv = fragCoord / vec2f(screen_size);

    let f = fragCoord / custom.scale - vec2f(0.,0.);

    // The path data
    // quadratic bezier curves
    let path = array(
        vec2f(500.,300.),
        vec2f(600.,200.),
        vec2f(300.,0.),
        vec2f(300.,0.),
        vec2f(0.,200.),
        vec2f(100.,300.),
        vec2f(100.,300.),
        vec2f(200.,400.),
        vec2f(300.,200.),
        vec2f(300.,200.),
        vec2f(400.,400.),
        vec2f(500.,300.),
    );

    /*let path = array(
        vec2f(100.,100.),
        vec2f(0.,200.),
        vec2f(300.,400.),
        vec2f(300.,400.),
        vec2f(600.,200.),
        vec2f(500.,100.),
        vec2f(500.,100.),
        vec2f(400.,0.),
        vec2f(300.,200.),
        vec2f(300.,200.),
        vec2f(200.,0.),
        vec2f(100.,100.),
    );*/

    var col = vec4f(.0,.0,.0, 0.);
    
    var inside = false;
    var acc = 0.;
    //just for computations
    var q: f32;

    // for every pair of adjacent points:
    // cast a triangle to 0,0 (or any other point)
    // inside triangle ?
    //     flip inside variable
    for (var i = 0; i < 12; i++) { 
        // triangle sdf serves as replacement for vertex pipeline
        q = sdTriangle(f, path[i], path[i+1], vec2f(0.0,0.0), );

        if q > 0.1 {
            inside = !inside;
            acc += .05;
        }
    }

    // color
    if inside {
        col = ( vec4f(.5,0.,.6,1.,) );

    } else {
        col = (vec4f(1.,1.,1.,1.,) );
    }

    // add curvy outlines
    //                       +++ note this
    for (var j = 0; j < 12; j+=3) {
        // are we inside the bounding triangle of the curve?
        q = sdTriangle(f, path[j], path[j+1], path[j+2], );
        // if no, we don't care about it
        if q > 0.0 {
            continue;
        }

        // Is the point with the area between the curve and line segment A C?
        let t = bezierTest(f, path[j], path[j+1], path[j+2]);
        
        inside = t;
        if inside {
            acc += .05;
            
        } else {
            col = (vec4f(1.,1.,1.,1.,) );
        }
        
    }

    
    // DEBUG view
    if custom.debug > .5 {
        col = vec4f(acc);
    }
    
    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec4f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, col);
}

fn sdTriangle(p: vec2f, p0: vec2f, p1: vec2f, p2: vec2f) -> f32 {
  let e0 = p1 - p0; let e1 = p2 - p1; let e2 = p0 - p2;
  let v0 = p - p0; let v1 = p - p1; let v2 = p - p2;
  let pq0 = v0 - e0 * clamp(dot(v0, e0) / dot(e0, e0), 0., 1.);
  let pq1 = v1 - e1 * clamp(dot(v1, e1) / dot(e1, e1), 0., 1.);
  let pq2 = v2 - e2 * clamp(dot(v2, e2) / dot(e2, e2), 0., 1.);
  let s = sign(e0.x * e2.y - e0.y * e2.x);
  let d = min(min(vec2f(dot(pq0, pq0), s * (v0.x * e0.y - v0.y * e0.x)),
                  vec2f(dot(pq1, pq1), s * (v1.x * e1.y - v1.y * e1.x))),
                  vec2f(dot(pq2, pq2), s * (v2.x * e2.y - v2.y * e2.x)));
  return -sqrt(d.x) * sign(d.y);
}

/// Is the point with the area between the curve and line segment A C?
fn bezierTest(p: vec2<f32>, A: vec2<f32>, B: vec2<f32>, C: vec2<f32>) -> bool {

    // Compute barycentric coordinates of p.
    // p = s * A + t * B + (1-s-t) * C
    let v0 = B - A; let v1 = C - A; let v2 = p - A;
    let det = v0.x * v1.y - v1.x * v0.y;
    let s = (v2.x * v1.y - v1.x * v2.y) / det;
    let t = (v0.x * v2.y - v2.x * v0.y) / det;

    if(s < 0.0 || t < 0.0 || (1.0-s-t) < 0.0) {
        return false; // outside triangle
    }

    // Transform to canonical coordinte space.
    let u = s * 0.5 + t;
    let v = t;

    return u*u < v;

}
fn sdCircle(p: vec2f, r: f32) -> f32 {
  return length(p) - r;
}



const PI = 3.1415927;
const TAU = 6.2831853;
// solve_quadratic(), solve_cubic(), solve() and sd_bezier() are from
// Quadratic Bezier SDF With L2 - Envy24
// https://www.shadertoy.com/view/7sGyWd
// with modification. Thank you! I tried a lot of different sd_bezier()
// implementations from across Shadertoy (including trying to make it
// myself) and all of them had bugs and incorrect edge case handling
// except this one.

fn solve_quadratic(a: f32, b: f32, c: f32, roots: ptr<function, vec2f>) -> u32 {
    // Return the number of real roots to the equation
    // a*x^2 + b*x + c = 0 where a != 0 and populate roots.
    let discriminant = b * b - 4.0 * a * c;

    if (discriminant < 0.0) {
        return 0;
    }

    if (discriminant == 0.0) {
        (*roots)[0] = -b / (2.0 * a);
        return 1;
    }

    let SQRT = sqrt(discriminant);
    (*roots)[0] = (-b + SQRT) / (2.0 * a);
    (*roots)[1] = (-b - SQRT) / (2.0 * a);
    return 2;
}

fn solve_cubic(a: f32, b: f32, c: f32, d: f32, roots: ptr<function, vec3f>) -> u32 {
    // Return the number of real roots to the equation
    // a*x^3 + b*x^2 + c*x + d = 0 where a != 0 and populate roots.
    let A = b / a;
    let B = c / a;
    let C = d / a;
    let Q = (A * A - 3.0 * B) / 9.0;
    let R = (2.0 * A * A * A - 9.0 * A * B + 27.0 * C) / 54.0;
    let S = Q * Q * Q - R * R;
    let sQ = sqrt(abs(Q));
    *roots = vec3f(-A / 3.0);

    if (S > 0.0) {
        *roots += -2.0 * sQ * cos(acos(R / (sQ * abs(Q))) / 3.0 + vec3(TAU, 0.0, -TAU) / 3.0);
        return 3;
    }
    
    if (Q == 0.0) {
        (*roots)[0] += -pow(C - A * A * A / 27.0, 1.0 / 3.0);
        return 1;
    }
    
    if (S < 0.0) {
        let u = abs(R / (sQ * Q));
        let v = select(sinh(asinh(u) / 3.0), cosh(acosh(u) / 3.0), Q > 0.0);
        (*roots)[0] += -2.0 * sign(R) * sQ * v;
        return 1;
    }
    
    *roots += vec3f(-2.0, 1.0, 0.0) * sign(R) * sQ;
    return 2;
}

fn solve(a: f32, b: f32, c: f32, d: f32, roots: ptr<function, vec3f>) -> u32 {
    // Return the number of real roots to the equation
    // a*x^3 + b*x^2 + c*x + d = 0 and populate roots.
    if (a == 0.0) {
        if (b == 0.0) {
            if (c == 0.0) {
                return 0;
            }
            
            (*roots)[0] = -d/c;
            return 1;
        }
        
        var r: vec2f;
        let num = solve_quadratic(b, c, d, &r);
        *roots = vec3f(r, 0.0);
        return num;
    }
    
    return solve_cubic(a, b, c, d, roots);
}

fn sd_bezier(p: vec2f, a: vec2f, b: vec2f, c: vec2f) -> f32 {
    let A = a - 2.0 * b + c;
    let B = 2.0 * (b - a);
    let C = a - p;
    var T: vec3f;
    let num = solve(
        2.0 * dot(A, A),
        3.0 * dot(A, B),
        2.0 * dot(A, C) + dot(B, B),
        dot(B, C),
        &T
    );
    T = clamp(T, vec3f(0.0), vec3f(1.0));
    var best = 1e30;
    
    for (var i = 0u; i < num; i++) {
        let t = T[i];
        let u = 1.0 - t;
        let d = u * u * a + 2.0 * t * u * b + t * t * c - p;
        best = min(best, dot(d, d));
    }
    
    return sqrt(best);
}

fn sd_segment(p: vec2f, a: vec2f, b: vec2f) -> f32 {
    let ap = p - a;
    let ab = b - a;
    return distance(p, a + ab * clamp(dot(ap, ab) / dot(ab, ab), 0.0, 1.0));
}