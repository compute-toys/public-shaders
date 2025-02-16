//https://www.shadertoy.com/view/cs3BWM (altunenes)
//Gabor function (ref: https://en.wikipedia.org/wiki/Gabor_filter)  
const PI: f32 = 3.1415926535897932384626433832795;
@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    let fragCoord: vec2<f32> = vec2<f32>(f32(id.x), f32(id.y));
    let iResolution: vec2<f32> = vec2<f32>(f32(textureDimensions(screen).x), f32(textureDimensions(screen).y));
    let iTime: f32 = time.elapsed;
    let iMouse: vec2<f32> = vec2<f32>(f32(mouse.pos.x), f32(mouse.pos.y));
    let uv: vec2<f32> = 2.0 * (fragCoord / iResolution) - vec2<f32>(1.0, 1.0);

    // Parameters of the Gabor function see 
    let lambda: f32 = 0.1;  // wavelength
    let theta: f32 = iMouse.x / iResolution.x * PI;  // orientation
    let psi: f32 = iTime * 5.5;  // phase offset
    let sigma: f32 = 0.1;  // sd of Gaussian
    let gamma: f32 = 1.0;  // spatial aspect ratio

    // Rotation transformation
    let xp: f32 = 2.0 * uv.x * cos(theta) - uv.y * sin(theta);
    let yp: f32 = 2.0 * uv.x * sin(theta) + uv.y * cos(theta);

    // Gabor function
    let envelope: f32 = exp(-((xp * xp) + (gamma * gamma * yp * yp)) / (2.0 * sigma * sigma));
    let carrier: f32 = cos(2.0 * PI * xp / lambda + psi);
    let gabor: f32 = envelope * carrier;

    let colorModulation: vec3<f32> = vec3<f32>(0.5) + vec3<f32>(0.5) * cos(1.5 * PI * xp / lambda + vec3<f32>(0.0, 2.0, 4.0));
    var col: vec3<f32> = vec3<f32>(0.1) + vec3<f32>(0.5) * gabor * colorModulation;

    // Set cutoff for the patch (uncomment)
    //let radius: f32 = 0.2;
    //if (length(uv) > radius) {
    //    col = vec3<f32>(0.5);
    //}

    
    // To use a plain Gabor patch without color modulation, uncomment the following line:
    // col = vec3<f32>(0.5) + vec3<f32>(0.5) * gabor;

    textureStore(screen, vec2<i32>(id.xy), vec4<f32>(col, 1.0));
}