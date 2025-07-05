import std;

// const float infinity = 1.0/0.0;
const static float infinity = 100000.0;
const static float pi = 3.1415926535897932385;

inline float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0;
}

float length_squared(float3 v)
{
    return dot(v, v);
}

// Random number generation using pcg32i_random_t, using inc = 1. Our random state is a uint.
uint stepRNG(uint rngState)
{
    return rngState * 747796405 + 1;
}

// Steps the RNG and returns a floating-point value between 0 and 1 inclusive.
float stepAndOutputRNGFloat(inout uint rngState)
{
    // Condensed version of pcg_output_rxs_m_xs_32_32, with simple conversion to floating-point [0,1].
    rngState  = stepRNG(rngState);
    uint word = ((rngState >> ((rngState >> 28) + 4)) ^ rngState) * 277803737;
    word      = (word >> 22) ^ word;
    return float(word) / 4294967295.0f;
}

float random_float(inout uint rngState) {
    return stepAndOutputRNGFloat(rngState);
}

float random_float(float min, float max, inout uint rngState) {
    // Returns a random real in [min,max).
    return min + (max-min)*random_float(rngState);
}

float3 random_vec3(inout uint rngState) {
    return float3(random_float(rngState), random_float(rngState), random_float(rngState));
}

float3 random_vec3(float min, float max, inout uint rngState) {
    return float3(random_float(min,max,rngState), random_float(min,max,rngState), random_float(min,max,rngState));
}

float3 random_unit_vector(inout uint rngState) {
    float3 p = random_vec3(-1,1,rngState);
    float lensq = length_squared(p) + 0.00000001; // add small size;
    return normalize(p / sqrt(lensq));
    // while(true) {
    //     float3 p = random_vec3(-1,1,rngState);
    //     float lensq = length_squared(p);
    //     if (1e-20 < lensq && lensq <= 1)
    //         return p / sqrt(lensq);
    // }
    // return float3(1,0,0);
}

float3 random_on_hemisphere(float3 normal, inout uint rngState) {
    float3 on_unit_sphere = random_unit_vector(rngState);
    if (dot(on_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}

struct interval
{
    float min, max;

    __init() {
        min = +infinity;
        max = -infinity;
    }

    __init(float _min, float _max) {
        min = _min;
        max = _max;
    }

    float size() {
        return max - min;
    }

    bool contains(float x) {
        return min <= x && x <= max;
    }

    bool surrounds(float x) {
        return min < x && x < max;
    }

    float clamp(float x) {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }
}

const static interval interval_empty    = interval(+infinity, -infinity);
const static interval interval_universe = interval(-infinity, +infinity);

struct ray
{
    float3 orig;
    float3 dir;
    __init() {}
    __init(float3 origin, float3 direction) {
        orig = origin;
        dir = direction;
    }

    float3 origin() { return orig; }
    float3 direction() { return dir; }

    float3 at(float t) {
        return orig + t*dir;
    }
}

struct hit_record
{
    float3 p;
    float3 normal;
    float t;
    bool front_face;

    [mutating]
    void set_face_normal(ray r, float3 outward_normal) {
        // Sets the hit record normal vector.
        // NOTE: the parameter `outward_normal` is assumed to have unit length.

        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
}

interface hittable 
{
    bool hit(ray r, interval ray_t, out hit_record rec);
}

struct sphere : hittable 
{
    float3 center;
    float radius;
    __init(float3 _center, float _radius) {
        center = _center;
        radius = max(_radius, 0.0);
    }

    bool hit(ray r, interval ray_t, out hit_record rec) {
        float3 oc = center - r.origin();
        float a = length_squared( r.direction() );
        float h = dot(r.direction(), oc);
        float c = length_squared(oc) - radius*radius;

        float discriminant = h*h - a*c;
        if (discriminant < 0)
            return false;

        float sqrtd = sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        float root = (h - sqrtd) / a;
        if (!ray_t.surrounds(root)) {
            root = (h + sqrtd) / a;
            if (!ray_t.surrounds(root))
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        float3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);

        return true;
    }
}

struct hittable_list : hittable
{
    Array<sphere, 64> objects;
    int size;

    __init() { size = 0; }
    __init(sphere object) {
        size = 0; 
        add(object);
    }

    [mutating]
    void clear() { size = 0; }

    [mutating]
    void add(sphere object) {
        if (size < 64) {
            objects[size] = object;
            size += 1;
        }
    }

    bool hit(ray r, interval ray_t, out hit_record rec) {
        hit_record temp_rec;
        bool hit_anything = false;
        float closest_so_far = ray_t.max;

        for (int i = 0; i < size; i++) {
            if (objects[i].hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }
}

float hit_sphere(float3 center, float radius, ray r)
{
    float3 oc = center - r.origin();
    float a = dot(r.direction(), r.direction());
    float h = dot(r.direction(), oc);
    float c = dot(oc, oc) - radius*radius;
    float discriminant = h*h - a*c;
    
    if (discriminant < 0) {
        return -1.0;
    } else {
        return (h - sqrt(discriminant) ) / (2.0*a);
    }
}

float hit_sphere(sphere sph, ray r)
{
    float3 center = sph.center;
    float radius = sph.radius;
    float3 oc = center - r.origin();
    float a = dot(r.direction(), r.direction());
    float h = dot(r.direction(), oc);
    float c = dot(oc, oc) - radius*radius;
    float discriminant = h*h - a*c;
    
    if (discriminant < 0) {
        return -1.0;
    } else {
        return (h - sqrt(discriminant) ) / (2.0*a);
    }
}

float3 ray_color(ray r, hittable world)
{
    hit_record rec;

    if (world.hit(r, interval(0, infinity), rec)) {
        return 0.5 * (rec.normal + float3(1,1,1));
    }

    float3 unit_direction = normalize(r.direction());
    float3 a = 0.5*(unit_direction.y + 1.0);
    return (1.0-a)*float3(1.0, 1.0, 1.0) + a*float3(0.5, 0.7, 1.0);
}

struct camera
{
    float  image_width;          // Rendered image height
    float  image_height;         // Rendered image height
    float  aspect_ratio;         // Ratio of image width over height
    float3 center;               // Camera center
    float3 pixel00_loc;          // Location of pixel 0, 0
    float3 pixel_delta_u;        // Offset to pixel to the right
    float3 pixel_delta_v;        // Offset to pixel below

    int    samples_per_pixel;    // Count of random samples for each pixel
    float  pixel_samples_scale;  // Color scale factor for a sum of pixel samples

    uint   rngState;             // random seed

    __init(float _image_width, float _image_height, int spp) {
        samples_per_pixel = spp;
        pixel_samples_scale = 1.0 / samples_per_pixel;

        rngState = uint(custom.RandomSeed * 1234); // magic number/random seed

        image_width  = _image_width;
        image_height = _image_height;
        aspect_ratio = image_width/image_height;

        center = float3(0, 0, 0);

        // Determine viewport dimensions.
        float focal_length = 1.0;
        float viewport_height = 2.0;
        float viewport_width = viewport_height * (image_width/image_height);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        float3 viewport_u = float3(viewport_width, 0, 0);
        float3 viewport_v = float3(0, -viewport_height, 0);

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // Calculate the location of the upper left pixel.
        float3 viewport_upper_left =
            center - float3(0, 0, focal_length) - viewport_u/2 - viewport_v/2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
    }

    [mutating]
    float3 ray_color(ray r, hittable world) {
        // hit_record rec;

        // // if (world.hit(r, interval(0, infinity), rec)) {
        // //     return 0.5 * (rec.normal + float3(1,1,1));
        // // }

        // if (world.hit(r, interval(0, infinity), rec)) {
        //     float3 direction = random_on_hemisphere(rec.normal, rngState);
        //     return 0.5 * ray_color(ray(rec.p, direction), world);
        // }

        // float3 unit_direction = normalize(r.direction());
        // float3 a = 0.5*(unit_direction.y + 1.0);
        // return (1.0-a)*float3(1.0, 1.0, 1.0) + a*float3(0.5, 0.7, 1.0);

        float3 color = float3(1.0, 1.0, 1.0);
        ray current_ray = r;
        
        const int max_depth = 50;
        
        for (int depth = 0; depth < max_depth; depth++) {
            hit_record rec;
            
            if (world.hit(current_ray, interval(0, infinity), rec)) {
                float3 direction = rec.normal + random_on_hemisphere(rec.normal, rngState);
                color *= 0.5;
                current_ray = ray(rec.p, direction);
            } else {
                float3 unit_direction = normalize(current_ray.direction());
                float a = 0.5*(unit_direction.y + 1.0);
                float3 background_color = (1.0-a)*float3(1.0, 1.0, 1.0) + a*float3(0.5, 0.7, 1.0);
                
                color *= background_color;
                break;
            }
        }
        
        return color;
    }

    [mutating]
    float3 render(hittable world, uint3 id) {
        // Render

        float3 pixel_color = float3(0,0,0);
        for (int sample = 0; sample < samples_per_pixel; sample++) {
            ray r = get_ray(id.x, id.y);
            pixel_color += ray_color(r, world);
        }

        return pixel_samples_scale * pixel_color;
    }

    [mutating]
    ray get_ray(int i, int j) {
        // Construct a camera ray originating from the origin and directed at randomly sampled
        // point around the pixel location i, j.

        float3 offset = sample_square();
        float3 pixel_sample = pixel00_loc
                          + ((i + offset.x) * pixel_delta_u)
                          + ((j + offset.y) * pixel_delta_v);

        float3 ray_origin = center;
        float3 ray_direction = pixel_sample - ray_origin;

        return ray(ray_origin, ray_direction);
    }

    [mutating]
    float3 sample_square() {
        // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
        return float3(random_float(rngState) - 0.5, random_float(rngState) - 0.5, 0);
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main_image(uint3 id : SV_DispatchThreadID)
{
    uint image_width, image_height;
    screen.GetDimensions(image_width, image_height);

    if (id.x >= image_width || id.y >= image_height) {
        return;
    }

    float2 fragCoord = float2(float(id.x) + 0.5, float(image_height - id.y) - 0.5);
    float2 uv = fragCoord / float2(image_width, image_height);

    // World

    hittable_list world;

    world.add(sphere(float3(0,0,-1), 0.5));
    world.add(sphere(float3(0,-100.5,-1), 100));

    // Camera

    camera cam = camera(float(image_width), float(image_height), 10);
    float3 pixel_color = cam.render(world, id);

    // gamma
    // pixel_color = pow( pixel_color, float3(2.2) );

    screen[id.xy] = float4(pixel_color, 1.0);

    // screen[id.xy] = float4(float3(custom.RandomSeed), 1.0);
}
