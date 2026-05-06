#define NUMPARTICLES 1000u
#define BOXLOWER vec3f(-20, 0, 20)
#define BOXUPPER vec3f(20, 20, 60)
#define SMOOTHING_RADIUS 1.5
#define PARTICLE_RADIUS (SMOOTHING_RADIUS / 2.0)
#define GASCONSTANT 1000.0
#define RESTDENSITY 0.01
#define VISCOSITY 0.5
#define dt 0.002

#define MAX_DIST 100000.0
#define SURFDIST 0.1
#define MAXMARCHES 70u

#define ENABLE_SMOOTHING true

// From @nojima's High-quality hash functions
fn murmurHash11(source: u32) -> u32{
    var src = source;
    const M = 0x5bd1e995u;
    var h = 1190494759u;
    src *= M; src ^= src>>24u; src *= M;
    h *= M; h ^= src;
    h ^= h>>13u; h *= M; h ^= h>>15u;
    return h;
}

fn murmurHash13(source: vec3u) -> u32 {
    var src = source;
    const M = 0x5bd1e995u;
    var h = 1190494759u;
    src *= M;
    src ^= vec3u(src.x>>24u,src.y>>24u,src.z>>24u);
    src *= M;
    h *= M; h ^= src.x; h *= M; h ^= src.y; h *= M; h ^= src.z;
    h ^= h>>13u; h *= M; h ^= h>>15u;
    return h;
}

// PRNG Utility Functions
//--------------------------------------------------------------------

fn seedPRNG(seed: vec3f) -> u32 {
    return murmurHash13(bitcast<vec3u>(seed));
}

fn shuffle(prngState: ptr<function, u32>, ) {
    *prngState = murmurHash11(*prngState);
}

// Random in range [0, Max Uint = 0xFFFFFFFFu]
fn randomUint(prngState: ptr<function, u32>) -> u32 {
    shuffle(prngState);
    return *prngState;
}

fn randomFloat(prngState: ptr<function, u32>) -> f32 {
    return float(randomUint(prngState)) / float(0xFFFFFFFFu);
}

fn randomVec3f(state: ptr<function, u32>) -> vec3f {
    return vec3f(randomFloat(state),randomFloat(state),randomFloat(state));
}

fn randomInBox(state: ptr<function, u32>, lowerBound: vec3f, upperBound: vec3f) -> vec3f {
    let lower = min(lowerBound, upperBound);
    let upper = max(lowerBound, upperBound);
    let rand = randomVec3f(state);
    return (upper - lower) * rand + lower;
}

struct Particle {
    position: vec3f,
    velocity: vec3f,
    mass: float,
    density: float,
    pressure: float,
    force: vec3f,
};


#storage particles array<Particle, NUMPARTICLES>;


// Camera Stuff
struct Camera {
    // Normalized basis vectors:
    w_forward: vec3f,
    u_right: vec3f,
    v_up: vec3f,

    position: vec3f,
    topLeft: vec3f,
    width: float,
    height: float,
    focalDist: float,
}

struct Ray {
    origin: vec3f,
    direction: vec3f,
};

fn rayAt(ray: Ray, t: f32) -> vec3f {
    return ray.origin + ray.direction * t;
}

fn newCam(position: vec3f, direction: vec3f, aspect: float, fov_degrees: float) -> Camera {
    var camera: Camera;// = Camera(vec3f(0f), vec3f(0f), vec3f(0f), vec3f(0f), vec3f(0f), 0f, 0f);
    let scene_up = vec3(0.0, 1.0, 0.0);
    camera.w_forward = normalize(direction);
    camera.u_right = normalize(cross(camera.w_forward, scene_up));
    camera.v_up = normalize(cross(camera.u_right, camera.w_forward));
    camera.focalDist = 1.0;
    let theta = radians(fov_degrees);
    let half_theta = 0.5 * theta;
    let half_width = tan(half_theta) * camera.focalDist;
    camera.width = 2.0 * half_width;
    camera.height = camera.width / aspect;
    let half_height = 0.5 * camera.height;

    camera.position = position;
    camera.topLeft = position + camera.w_forward * camera.focalDist
                            - half_width * camera.u_right
                            + half_height * camera.v_up;
    return camera;
}

fn sample_ray_viewport(cam: Camera, uv: vec2f) -> Ray {
    var ray: Ray;// = Ray(vec3f(0f), vec3f(0f));
    ray.origin = cam.position;
    let destination = cam.topLeft + uv.x * cam.width * cam.u_right
                                  - uv.y * cam.height * cam.v_up;
    ray.direction = normalize(destination - ray.origin);
                                
    return ray;
}

fn remap(_01: f32, newMin: f32, newMax: f32) -> f32 {
    return _01 * (newMax - newMin) + newMin;
}

fn remap_vec2f(_01: vec2f, newMin: vec2f, newMax: vec2f) -> vec2f {
    return vec2f(remap(_01.x, newMin.x, newMax.x),
                 remap(_01.y, newMin.y, newMax.y));
}

fn sample_ray(cam: Camera, pixel: vec2u) -> Ray {
    let screen_res = textureDimensions(screen);
    var ray: Ray;
    let topLeft = vec2f(pixel) / vec2f(screen_res);
    let bottomRight = vec2f(pixel + vec2u(1u)) / vec2f(screen_res);
    let uv = remap_vec2f(vec2f(0.5, 0.5), topLeft, bottomRight);
    return sample_ray_viewport(cam, uv);
}

#dispatch_once init_particles
#workgroup_count init_particles NUMPARTICLES 1 1
@compute @workgroup_size(64)
fn init_particles(@builtin(global_invocation_id) id: vec3u) {
    var state = seedPRNG(vec3f(vec3f(id).xy, time.elapsed));

    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);
    let index = id.x;
    
    let pos = randomInBox(&state, BOXLOWER, BOXUPPER);

    if (index >= NUMPARTICLES) {
        return;
    }

    particles[index] = Particle(pos, vec3f(0), 1.0, 1.0, 1.0, vec3f(1.0));
}

//fn gravityForce() -> vec3f {
//    return vec3f(0, -custom.gravity_strength, 0);
//}

struct Plane {
    position: vec3f,
    normal: vec3f,
};

fn planeSDF(point: vec3f, plane: Plane) -> f32 {
    return dot(point - plane.position, plane.normal);
}

struct Sphere {
    position: vec3f,
    radius: f32,
};

fn sphereSDF(point: vec3f, sphere: Sphere) -> f32 {
    return distance(point, sphere.position) - sphere.radius;
}

fn belowPlane(point: vec3f, plane: Plane) -> bool {
    let dist = planeSDF(point, plane);
    return dist < 0.0;
}

fn smin(a: float, b: float, k: float) -> vec2f {
    let h = 1.0 - min( abs(a-b)/(6.0*k), 1.0 );
    let w = h*h*h;
    let m = w*0.5;
    let s = w*k; 
    return select(vec2(b-s,1.0-m), vec2(a-s,m), a < b);
}

fn sdScene(p: vec3f) -> vec4f {
    let boxCenter = (BOXUPPER + BOXLOWER) / 2.0;
    let boxRadius = distance(boxCenter, BOXUPPER) * 2.0;
    if (distance(p, boxCenter) > boxRadius) {
        return vec4f(distance(p, boxCenter), vec3f(1));
    }

    var dist = 1.0e30;
    var col = vec3f(0);
    for (var i = 0u; i < NUMPARTICLES; i++) {
        let sphere = Sphere(particles[i].position, PARTICLE_RADIUS);
        let d = sphereSDF(p, sphere);
        var newCol = vec3f(0.0);
        if (d < dist) {
            let pressure = particles[i].pressure;
            let t1 = 400.0;
            let t2 = 900.0;
            let t3 = 1500.0;
            if (pressure < t1) {
                newCol = vec3f(0.1,0.1,1);
            } else if (pressure < t2) {
                newCol = mix(vec3f(0.1,0.1,1), vec3f(0,1,0), (pressure - t1) / (t2 - t1));
            } else if (pressure < t3) {
                newCol = mix(vec3f(0,1,0), vec3f(1,0.2,0.1), (pressure - t2) / (t3 - t2));
            } else {
                newCol = vec3f(1,.2,0.1);
            }
        }
        let smoothmin = smin(dist, sphereSDF(p, sphere), 0.5);
        
        if (ENABLE_SMOOTHING) {
            dist = smoothmin.x;
            col = vec3f(0.1, 0.1, 1.0);
        } else {
            col = smoothmin.y * newCol + (1.0 - smoothmin.y) * col;
            dist = min(dist, d);
        }
    }
    let ground = Plane(vec3f(0,0,0), normalize(vec3f(0, 1, 0)));
    //dist = min(dist, planeSDF(p, ground));
    return vec4f(dist, col);
}

fn calcNormal(p: vec3f) -> vec3f {
    const h = 0.0001; // replace by an appropriate value
    const k = vec2f(1,-1);
    return normalize( k.xyy*sdScene( p + k.xyy*h ).x + 
                      k.yyx*sdScene( p + k.yyx*h ).x + 
                      k.yxy*sdScene( p + k.yxy*h ).x + 
                      k.xxx*sdScene( p + k.xxx*h ).x );
}


fn rayMarch(ray: Ray, color: ptr<function, vec3f>) -> float {
    var d0 = 0.;
    //var col = vec3f(0);
    for (var b = 0u; b < MAXMARCHES; b++) {
        if (d0 > MAX_DIST) {
            break;
        }
        let distCol = sdScene(rayAt(ray, d0));
        let dist = distCol.x;
        let col = distCol.yzw;
        if (dist < SURFDIST) {
            *color = col;
            break;
        }
        d0 += dist;
    }
    return d0;
}

fn getLight(p: vec3f) -> vec3f {
    let intensity = 1.0;
    var color = vec3f(1.0) * intensity;

    var lightPos = vec3f(5.0 * sin(time.elapsed), 5., 6. + 5.*cos(time.elapsed));
    //lightPos = normalize(vec3f(0.0,0.0,50.0));
    var l = normalize(lightPos - p);
    l = normalize(vec3(1,1,-1));
    let n = calcNormal(p);
    var diff = dot(n, l);
    diff = clamp(diff, 0., 1.);
    let d = rayMarch(Ray(p + n*SURFDIST * 2.,l), &color);
    if (d < length(lightPos - p)) {
        diff *= .1;
    }
    return color * diff;


    // Shadows

}

// (Gingold & Monaghan 1977)
fn gaussian(h: f32, r: f32) -> f32 {
    const pi = 3.14159;
    const sqrtpi = sqrt(pi);
    let eTerm = exp(-(r * r / (h * h)));
    let invTerm = 1.0 / (h * sqrtpi);
    return eTerm * invTerm;
}

fn W(a: Particle, b: Particle) -> f32 {
    let h = SMOOTHING_RADIUS;
    let r = distance(a.position, b.position);
    return gaussian(h, r);
}

fn density(a: Particle, b: Particle) -> f32 {
    return b.mass * W(a, b);
}

fn spiky_gradient_kernel(r: f32, h: f32) -> f32 {
    const pi = 3.14159;
    let factor = -45.0 / (pi * h * h * h * h * h * h);
    let inner = h - r;
    return factor * inner * inner;
}

fn laplacian_kernel(r: f32, h: f32) -> f32 {
    const pi = 3.14159;
    let factor = 45.0 / (pi * h * h * h * h * h * h);
    let inner = h - r;
    return factor * inner;
}

fn W_boundary(h: f32, r: f32) -> f32 {
    return gaussian(h, r);
}

fn calcDensity(index: u32) {
    let ground = Plane(vec3f(0), normalize(vec3f(0, 1, 0)));
    let wall1 = Plane(vec3f(0, 0, BOXUPPER.z) , vec3f(0, 0, -1));
    let wall2 = Plane(vec3f(0, 0, BOXLOWER.z), vec3f(0, 0, 1));
    let wall3 = Plane(vec3f(BOXUPPER.x, 0, 0), vec3f(-1, 0, 0));
    let wall4 = Plane(vec3f(BOXLOWER.x, 0, 0), vec3f(1, 0, 0));
    let walls = array<Plane, 5>(ground, wall1, wall2, wall3, wall4);
    
    var part = particles[index];

    var densitySum = 0.0;

    for (var i = 0u; i < NUMPARTICLES; i++) {
        let otherPart = particles[i];
        let d = distance(part.position, particles[i].position);
        if (d < SMOOTHING_RADIUS) {
            densitySum += density(part, otherPart);
        }
    }

    for (var w = 0u; w < 5u; w++) {
        let distToWall = planeSDF(part.position, walls[w]);
        if (distToWall < SMOOTHING_RADIUS) {
            densitySum += RESTDENSITY * W_boundary(distToWall, SMOOTHING_RADIUS);
        }
    }

    part.density = max(0.1, densitySum);
    part.pressure = GASCONSTANT * (densitySum - RESTDENSITY);
    part.pressure = max(0.0, part.pressure);
    particles[index] = part;
}

fn calcForce(index: u32) {
    var part = particles[index];

    var pressureForce = vec3f(0.0);
    var viscosityForce = vec3f(0.0);

    for (var i = 0u; i < NUMPARTICLES; i++) {
        if (i == index) { continue; }
        let otherPart = particles[i];
        let dist = distance(part.position, particles[i].position);
        if (dist < SMOOTHING_RADIUS) {
            let dir = normalize(part.position - otherPart.position);
            let sharedPressure = (part.pressure + otherPart.pressure) / 2.0;
            pressureForce += -dir * part.mass * sharedPressure / otherPart.density * spiky_gradient_kernel(dist, SMOOTHING_RADIUS);
        
            let relVel = otherPart.velocity - part.velocity;
            viscosityForce += VISCOSITY * part.mass * (relVel / otherPart.density) * laplacian_kernel(dist, SMOOTHING_RADIUS);
        }
    }

    let ground = Plane(vec3f(0), normalize(vec3f(0, 1, 0)));
    let wall1 = Plane(vec3f(0, 0, BOXUPPER.z) , vec3f(0, 0, -1));
    let wall2 = Plane(vec3f(0, 0, BOXLOWER.z), vec3f(0, 0, 1));
    let wall3 = Plane(vec3f(BOXUPPER.x, 0, 0), vec3f(-1, 0, 0));
    let wall4 = Plane(vec3f(BOXLOWER.x, 0, 0), vec3f(1, 0, 0));
    let walls = array<Plane, 5>(ground, wall1, wall2, wall3, wall4);
    for (var w = 0u; w < 5u; w++) {
        let distToWall = planeSDF(part.position, walls[w]);
        if (distToWall < SMOOTHING_RADIUS && distToWall > 0.001) {
            let dir = walls[w].normal;

            let wallPressure = GASCONSTANT * (RESTDENSITY - RESTDENSITY);
            let sharedPressure = part.pressure;
            let forceMag = 500.0 * part.mass * sharedPressure / part.density * spiky_gradient_kernel(max(distToWall, 1.0), SMOOTHING_RADIUS);
            //pressureForce += -dir * part.mass * sharedPressure / part.density * spiky_gradient_kernel(distToWall, SMOOTHING_RADIUS);
            pressureForce += -dir * forceMag;
        }
    }

    var externalForce = normalize(vec3f(0,-1,0)) * 9.8;
    externalForce += 6.0 * vec3(sin(time.elapsed/1.0),0,cos(time.elapsed/1.0));
    part.force = pressureForce + viscosityForce + externalForce;

    particles[index] = part;
}

fn integrateParticles(index: u32) {
    var part = particles[index];

    let acceleration = part.force / part.density;
    part.velocity += acceleration * dt;
    part.position += part.velocity * dt;
    part.velocity *= 0.999;


    //let ground = Plane(vec3f(0), normalize(vec3f(cos(time.elapsed), 10, sin(time.elapsed))));
    //let wall1 = Plane(vec3f(0, 0, BOXUPPER.z) , vec3f(0, 0, -1));
    //let wall2 = Plane(vec3f(0, 0, BOXLOWER.z), vec3f(0, 0, 1));
    //let wall3 = Plane(vec3f(BOXUPPER.x, 0, 0), vec3f(-1, 0, 0));
    //let wall4 = Plane(vec3f(BOXLOWER.x, 0, 0), vec3f(1, 0, 0));
    //let walls = array<Plane, 5>(ground, wall1, wall2, wall3, wall4);
    //if belowPlane(part.position, ground) {
    //    part.velocity = 0.9 * reflect(part.velocity, ground.normal);
    //}
    //if belowPlane(part.position, wall1) {
    //    part.velocity = 0.9 * reflect(part.velocity, wall1.normal);
    //}
    //if belowPlane(part.position, wall2) {
    //    part.velocity = 0.9 * reflect(part.velocity, wall2.normal);
    //}
    //if belowPlane(part.position, wall3) {
    //    part.velocity = 0.9 * reflect(part.velocity, wall3.normal);
    //}
    //if belowPlane(part.position, wall4) {
    //    part.velocity = 0.9 * reflect(part.velocity, wall4.normal);
    //}

    particles[index] = part;
}

#dispatch_count integrate_particles 20
#workgroup_count integrate_particles NUMPARTICLES 1 1
@compute @workgroup_size(64)
fn integrate_particles(@builtin(global_invocation_id) id: vec3u) {
    let index = id.x;

    if (index >= NUMPARTICLES) {
        return;
    }

    calcDensity(index);
    calcForce(index);
    integrateParticles(index);
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);

    let position = vec3f(0, 20, -15);
    let direction = normalize(vec3f(0, -.5, 1.5));
    let aspect = 16.0 / 9.0;
    let fov_degrees = 90.0;

    let cam = newCam(position, direction, aspect, fov_degrees);
    var ray = sample_ray(cam, id.xy);

    // Normalised pixel coordinates (from 0 to 1)
    let uv = fragCoord / vec2f(screen_size);

    // Time varying pixel colour
    var col = .5 + .5 * cos(time.elapsed + uv.xyx + vec3f(0.,2.,4.));

    let d = rayMarch(ray, &col);
    let p = rayAt(ray, d);
    col = getLight(p) * col;

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));
    

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}
