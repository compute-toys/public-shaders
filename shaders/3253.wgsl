// ============================================================
// Beginner Ray Marching Scene for compute.toys
//
// The scene contains:
// - An infinite checkerboard floor
// - A red sphere
// - A blue box
// - A camera that automatically rotates around the scene
// - A blue gradient sky
// - Sunlight, soft shadows, ambient occlusion, and fog
// ============================================================


// Maximum number of ray-marching steps.
//
// A higher value improves accuracy but reduces performance.
const MAX_STEPS: i32 = 100;

// Maximum distance a ray is allowed to travel.
//
// If the ray travels farther than this, we assume that it
// did not hit anything.
const MAX_DISTANCE: f32 = 60.0;

// If the distance to a surface is smaller than this value,
// we consider the ray to have hit the surface.
const SURFACE_DISTANCE: f32 = 0.001;


// ============================================================
// Box SDF
// ============================================================
//
// An SDF, or Signed Distance Function, tells us the distance
// from a point to the surface of an object.
//
// p         = point relative to the center of the box
// half_size = distance from the center to each side
// ============================================================

fn box_sdf(p: vec3f, half_size: vec3f) -> f32 {
    // Use symmetry to perform the calculation in one octant.
    let q = abs(p) - half_size;

    // The first part calculates distance outside the box.
    let outside_distance = length(max(q, vec3f(0.0)));

    // The second part calculates distance inside the box.
    let inside_distance = min(
        max(q.x, max(q.y, q.z)),
        0.0
    );

    return outside_distance + inside_distance;
}


// ============================================================
// Scene SDF
// ============================================================
//
// This function describes all objects in the scene.
//
// Input:
// p = a point in 3D world space
//
// Output:
// x = distance to the nearest surface
// y = material ID of the nearest object
//
// Material IDs:
// 1 = floor
// 2 = sphere
// 3 = box
// ============================================================

fn scene_sdf(p: vec3f) -> vec2f {
    // --------------------------------------------------------
    // Floor
    // --------------------------------------------------------
    //
    // The floor is a plane located at y = 0.
    //
    // If p.y is 2, the point is 2 units above the floor.
    // If p.y is 0, the point is on the floor.
    var nearest = vec2f(p.y, 1.0);


    // --------------------------------------------------------
    // Sphere
    // --------------------------------------------------------

    // Position of the sphere's center.
    let sphere_position = vec3f(-1.05, 0.8, 0.1);

    // The radius is 0.8.
    //
    // Because the sphere center is also 0.8 units above the
    // floor, the bottom of the sphere touches the floor.
    let sphere_radius = 0.8;

    // Sphere SDF:
    //
    // First calculate the distance from p to the sphere center,
    // then subtract the sphere radius.
    let sphere_distance =
        length(p - sphere_position) - sphere_radius;

    // Keep the sphere if it is closer than the previous object.
    if (sphere_distance < nearest.x) {
        nearest = vec2f(sphere_distance, 2.0);
    }


    // --------------------------------------------------------
    // Box
    // --------------------------------------------------------

    // Position of the center of the box.
    let box_position = vec3f(1.05, 0.65, 0.15);

    // Distance from the center to each side of the box.
    //
    // The complete box size is 1.3 × 1.3 × 1.3.
    let box_half_size = vec3f(0.65, 0.65, 0.65);

    // Move the point into the box's local coordinate system.
    let box_distance = box_sdf(
        p - box_position,
        box_half_size
    );

    // Keep the box if it is closer than the previous object.
    if (box_distance < nearest.x) {
        nearest = vec2f(box_distance, 3.0);
    }

    return nearest;
}


// ============================================================
// Ray Marching
// ============================================================
//
// ro = ray origin, usually the camera position
// rd = normalized ray direction
//
// Output:
// x = total distance traveled
// y = material ID
//
// A material ID of 0 means that the ray missed the scene.
// ============================================================

fn ray_march(ro: vec3f, rd: vec3f) -> vec2f {
    // Total distance traveled by the ray.
    var traveled = 0.0;

    // Material 0 means that no object was hit.
    var material_id = 0.0;

    for (var i: i32 = 0; i < MAX_STEPS; i++) {
        // Calculate the ray's current position.
        let current_position = ro + rd * traveled;

        // Ask the scene SDF how far the nearest object is.
        let scene_result = scene_sdf(current_position);

        let distance_to_scene = scene_result.x;

        // If the distance is very small, we are close enough
        // to consider this a surface hit.
        if (distance_to_scene < SURFACE_DISTANCE) {
            material_id = scene_result.y;
            break;
        }

        // The SDF guarantees that we can safely move forward
        // by this distance without passing through an object.
        traveled += distance_to_scene;

        // Stop if the ray travels too far.
        if (traveled > MAX_DISTANCE) {
            material_id = 0.0;
            break;
        }
    }

    return vec2f(traveled, material_id);
}


// ============================================================
// Surface Normal
// ============================================================
//
// A normal is a direction perpendicular to a surface.
//
// Lighting uses the normal to determine whether a surface is
// facing toward or away from a light source.
//
// This function estimates the normal by sampling the SDF at
// nearby positions.
// ============================================================

fn get_normal(p: vec3f) -> vec3f {
    let e = 0.001;

    // Measure the change in distance along the X axis.
    let normal_x =
        scene_sdf(p + vec3f(e, 0.0, 0.0)).x -
        scene_sdf(p - vec3f(e, 0.0, 0.0)).x;

    // Measure the change in distance along the Y axis.
    let normal_y =
        scene_sdf(p + vec3f(0.0, e, 0.0)).x -
        scene_sdf(p - vec3f(0.0, e, 0.0)).x;

    // Measure the change in distance along the Z axis.
    let normal_z =
        scene_sdf(p + vec3f(0.0, 0.0, e)).x -
        scene_sdf(p - vec3f(0.0, 0.0, e)).x;

    // Convert the result into a unit-length direction.
    return normalize(vec3f(
        normal_x,
        normal_y,
        normal_z
    ));
}


// ============================================================
// Soft Shadows
// ============================================================
//
// This function sends another ray from a surface toward the sun.
//
// If that ray hits an object, the sun is blocked and the point
// is in shadow.
//
// Return value:
// 0 = completely shadowed
// 1 = completely illuminated
// ============================================================

fn soft_shadow(ro: vec3f, rd: vec3f) -> f32 {
    // Start with no shadow.
    var shadow = 1.0;

    // Start slightly away from the surface to avoid immediately
    // hitting the same object.
    var traveled = 0.02;

    for (var i: i32 = 0; i < 48; i++) {
        let current_position = ro + rd * traveled;

        let distance_to_scene =
            scene_sdf(current_position).x;

        // An object is blocking the light.
        if (distance_to_scene < 0.001) {
            return 0.0;
        }

        // This creates a soft transition near shadow edges.
        //
        // Increasing 12.0 generally makes the shadow sharper.
        // Decreasing it generally makes the shadow softer.
        shadow = min(
            shadow,
            12.0 * distance_to_scene / traveled
        );

        // Limit each step to balance quality and performance.
        traveled += clamp(
            distance_to_scene,
            0.01,
            0.25
        );

        // Stop after traveling far enough toward the sun.
        if (traveled > 20.0) {
            break;
        }
    }

    return clamp(shadow, 0.0, 1.0);
}


// ============================================================
// Ambient Occlusion
// ============================================================
//
// Ambient occlusion, or AO, darkens corners, contact areas,
// and places where objects are close together.
//
// The function samples several positions along the surface
// normal to estimate how enclosed the point is.
//
// Return value:
// 0 = strongly occluded
// 1 = not occluded
// ============================================================

fn ambient_occlusion(
    p: vec3f,
    surface_normal: vec3f
) -> f32 {
    var occlusion = 0.0;

    // Nearby samples have a stronger effect.
    var weight = 1.0;

    for (var i: i32 = 1; i <= 5; i++) {
        // Distance of this sample from the surface.
        let sample_distance = f32(i) * 0.12;

        // Move away from the surface along its normal.
        let sample_position =
            p + surface_normal * sample_distance;

        // Measure the actual free distance at that position.
        let scene_distance =
            scene_sdf(sample_position).x;

        // If the actual distance is smaller than expected,
        // another surface may be nearby.
        occlusion +=
            (sample_distance - scene_distance) * weight;

        // Reduce the influence of more distant samples.
        weight *= 0.55;
    }

    return clamp(1.0 - occlusion, 0.0, 1.0);
}


// ============================================================
// Sky Color
// ============================================================
//
// rd            = current viewing direction
// sun_direction = direction from the scene toward the sun
//
// The sky is darker at the top and brighter near the horizon.
// A small sun glow is also added.
// ============================================================

fn get_sky_color(
    rd: vec3f,
    sun_direction: vec3f
) -> vec3f {
    // This value becomes larger near the horizon.
    let horizon_amount = pow(
        1.0 - max(rd.y, 0.0),
        3.0
    );

    // Darker blue at the top of the sky.
    let sky_top = vec3f(0.10, 0.36, 0.78);

    // Brighter blue near the horizon.
    let sky_horizon = vec3f(0.55, 0.82, 1.0);

    // Blend between the two sky colors.
    var sky = mix(
        sky_top,
        sky_horizon,
        horizon_amount
    );

    // Check how closely the view direction points toward
    // the sun.
    let sun_amount = max(
        dot(rd, sun_direction),
        0.0
    );

    // A wide and soft glow around the sun.
    let sun_glow = pow(sun_amount, 16.0);

    // A small and bright center.
    let sun_core = pow(sun_amount, 256.0);

    sky +=
        vec3f(1.0, 0.72, 0.35) *
        sun_glow *
        0.18;

    sky +=
        vec3f(1.0, 0.90, 0.70) *
        sun_core *
        2.0;

    return sky;
}


// ============================================================
// Material Color
// ============================================================
//
// Returns the base color for each material.
//
// Material IDs:
// 1 = checkerboard floor
// 2 = red sphere
// 3 = blue box
// ============================================================

fn get_material_color(
    material_id: f32,
    p: vec3f
) -> vec3f {
    // --------------------------------------------------------
    // Floor material
    // --------------------------------------------------------

    if (material_id < 1.5) {
        // Divide the XZ plane into square cells.
        //
        // Increase 1.5 to make the squares smaller.
        // Decrease 1.5 to make the squares larger.
        let grid = floor(p.xz * 1.5);

        let grid_sum = grid.x + grid.y;

        // Produce an alternating value of 0 or 1.
        let checker = grid_sum -
            2.0 * floor(grid_sum * 0.5);

        let dark_color = vec3f(0.16, 0.18, 0.21);
        let light_color = vec3f(0.62, 0.66, 0.70);

        return mix(
            dark_color,
            light_color,
            checker
        );
    }

    // --------------------------------------------------------
    // Sphere material
    // --------------------------------------------------------

    if (material_id < 2.5) {
        return vec3f(0.92, 0.16, 0.07);
    }

    // --------------------------------------------------------
    // Box material
    // --------------------------------------------------------

    return vec3f(0.08, 0.48, 0.95);
}


// ============================================================
// Main Compute Shader
// ============================================================
//
// This function runs once for every output pixel.
//
// id.xy contains the coordinates of the current pixel.
// ============================================================

@compute @workgroup_size(16, 16)
fn main_image(
    @builtin(global_invocation_id) id: vec3u
) {
    // Get the output texture resolution.
    let screen_size = textureDimensions(screen);

    // Workgroups are launched in blocks of 16 × 16 threads.
    //
    // If the screen dimensions are not exact multiples of 16,
    // some threads may be outside the texture. Stop those
    // threads before they attempt to write a pixel.
    if (
        id.x >= screen_size.x ||
        id.y >= screen_size.y
    ) {
        return;
    }


    // --------------------------------------------------------
    // Pixel coordinates
    // --------------------------------------------------------

    // Adding 0.5 selects the center of the pixel.
    //
    // The Y coordinate is flipped so that the origin behaves
    // like it is at the bottom-left of the screen.
    let pixel_coord = vec2f(
        f32(id.x) + 0.5,
        f32(screen_size.y - id.y) - 0.5
    );

    // Convert the integer resolution into floating-point values.
    let resolution = vec2f(screen_size);

    // Convert pixel coordinates into centered screen coordinates.
    //
    // The center of the screen becomes approximately (0, 0).
    // Dividing by resolution.y corrects the aspect ratio, so
    // the sphere does not become stretched on a wide screen.
    let uv =
        (pixel_coord * 2.0 - resolution) /
        resolution.y;


    // --------------------------------------------------------
    // Automatically rotating camera
    // --------------------------------------------------------

    // time.elapsed contains the elapsed time in seconds.
    //
    // Multiplying by 0.35 controls the rotation speed.
    let camera_angle = time.elapsed * 0.35;

    // The camera always looks at this position.
    //
    // We use camera_look_at instead of target because target
    // cannot be used as a variable name in this environment.
    let camera_look_at = vec3f(0.0, 0.65, 0.1);

    // Horizontal distance between the camera and the scene.
    let camera_radius = 5.0;

    // sin() and cos() move the camera around a circular path.
    let camera_position =
        camera_look_at +
        vec3f(
            sin(camera_angle) * camera_radius,
            2.65,
            cos(camera_angle) * camera_radius
        );


    // --------------------------------------------------------
    // Camera coordinate system
    // --------------------------------------------------------

    // Direction from the camera toward the scene.
    let camera_forward = normalize(
        camera_look_at - camera_position
    );

    // The global upward direction.
    let world_up = vec3f(0.0, 1.0, 0.0);

    // Calculate the camera's right direction using a cross
    // product.
    let camera_right = normalize(
        cross(camera_forward, world_up)
    );

    // Calculate the camera's local upward direction.
    let camera_up = cross(
        camera_right,
        camera_forward
    );


    // --------------------------------------------------------
    // Create the viewing ray
    // --------------------------------------------------------

    // Every primary ray begins at the camera.
    let ray_origin = camera_position;

    // camera_forward * 1.8 acts like the camera's focal length.
    //
    // A larger number produces a narrower field of view.
    // A smaller number produces a wider field of view.
    //
    // uv.x and uv.y move the ray across the virtual screen.
    let ray_direction = normalize(
        camera_forward * 1.8 +
        camera_right * uv.x +
        camera_up * uv.y
    );


    // --------------------------------------------------------
    // Sun direction
    // --------------------------------------------------------

    // This direction points from a surface toward the sun.
    let sun_direction = normalize(
        vec3f(-0.5, 0.8, -0.35)
    );


    // --------------------------------------------------------
    // Trace the primary ray
    // --------------------------------------------------------

    let hit = ray_march(
        ray_origin,
        ray_direction
    );

    // Use the sky as the initial color.
    //
    // If the ray misses every object, this remains the final
    // scene color.
    var color = get_sky_color(
        ray_direction,
        sun_direction
    );


    // --------------------------------------------------------
    // Shade the surface if the ray hit something
    // --------------------------------------------------------

    if (
        hit.y > 0.5 &&
        hit.x < MAX_DISTANCE
    ) {
        // Calculate the exact 3D hit position.
        let hit_position =
            ray_origin +
            ray_direction * hit.x;

        // Calculate the surface normal.
        let surface_normal =
            get_normal(hit_position);

        // Get the object's base color.
        let base_color = get_material_color(
            hit.y,
            hit_position
        );


        // ----------------------------------------------------
        // Diffuse lighting
        // ----------------------------------------------------
        //
        // dot(normal, light direction) tells us how directly
        // the surface faces the sun.
        //
        // 1 means that the surface faces the sun directly.
        // 0 means that the surface receives no direct sunlight.
        let diffuse = max(
            dot(surface_normal, sun_direction),
            0.0
        );


        // ----------------------------------------------------
        // Shadow
        // ----------------------------------------------------

        // Move the shadow-ray origin slightly away from the
        // surface. This prevents the surface from incorrectly
        // shadowing itself.
        let shadow_origin =
            hit_position +
            surface_normal * 0.01;

        let shadow = soft_shadow(
            shadow_origin,
            sun_direction
        );


        // ----------------------------------------------------
        // Ambient occlusion
        // ----------------------------------------------------

        let ao = ambient_occlusion(
            hit_position,
            surface_normal
        );


        // ----------------------------------------------------
        // Ambient sky lighting
        // ----------------------------------------------------
        //
        // Upward-facing surfaces receive more blue sky light.
        let sky_light =
            0.5 + 0.5 * surface_normal.y;

        let ambient_light =
            vec3f(0.12, 0.25, 0.42) *
            sky_light;


        // ----------------------------------------------------
        // Direct sunlight
        // ----------------------------------------------------

        // Use a slightly warm color for the sunlight.
        let direct_light =
            vec3f(1.0, 0.82, 0.62) *
            diffuse *
            shadow *
            1.7;


        // ----------------------------------------------------
        // Specular highlight
        // ----------------------------------------------------

        // The half vector lies between the light direction
        // and the direction toward the camera.
        let half_vector = normalize(
            sun_direction - ray_direction
        );

        // A higher exponent creates a smaller and sharper
        // highlight.
        let specular = pow(
            max(
                dot(surface_normal, half_vector),
                0.0
            ),
            48.0
        ) * shadow;


        // ----------------------------------------------------
        // Combine the lighting
        // ----------------------------------------------------

        color =
            base_color *
            (ambient_light + direct_light) *
            ao +
            vec3f(1.0, 0.85, 0.65) *
            specular *
            0.45;


        // ----------------------------------------------------
        // Distance fog
        // ----------------------------------------------------
        //
        // Distant objects are blended toward the sky color.
        // This adds depth and hides the hard distance limit.
        let fog_amount =
            1.0 -
            exp(-hit.x * hit.x * 0.004);

        color = mix(
            color,
            get_sky_color(
                ray_direction,
                sun_direction
            ),
            fog_amount
        );
    }


    // --------------------------------------------------------
    // Tone mapping
    // --------------------------------------------------------
    //
    // Lighting calculations may produce values above 1.
    // Tone mapping compresses bright colors into a displayable
    // range without simply cutting them off.
    color = color / (color + vec3f(1.0));


    // --------------------------------------------------------
    // Write the final pixel
    // --------------------------------------------------------

    textureStore(
        screen,
        id.xy,
        vec4f(color, 1.0)
    );
}