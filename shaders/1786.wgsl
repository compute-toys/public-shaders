fn rotateYawPitch(t: vec2f) -> mat3x3f {
    var stx = sin(t.x);
    var ctx = cos(t.x);
    var sty = sin(t.y);
    var cty = cos(t.y);
    var xRotation = mat3x3f(
        1, 0, 0,
        0, ctx, -stx,
        0, stx, ctx
    );
    
    var yRotation = mat3x3f(
        cty, 0, -sty,
        0, 1, 0,
        sty, 0, cty
    );
    
    return xRotation * yRotation;
}

fn getColorAtPos(pos: vec3f) -> vec4f {
        let mapPosX = f32(i32(pos.x) & 1023); //f32(1024);\
        let mapPosY = f32(i32(pos.z) & 1023);
        let scaledMapPos = vec2f(mapPosX, mapPosY) / 1024.0;
        return textureSampleLevel(channel0, nearest, scaledMapPos, 0);
}

fn getHeightAtPos(pos: vec3f) -> f32 {
        let mapPosX = f32(i32(pos.x) & 1023); //f32(1024);\
        let mapPosY = f32(i32(pos.z) & 1023);
        let scaledMapPos = vec2f(mapPosX, mapPosY) / 1024.0;

        return textureSampleLevel(channel1, nearest, scaledMapPos, 0).r * 256.;
}


fn raycast(x: i32, yScale: f32, maxY: u32, pitch: f32, origin: vec3f, direction: vec3f) {

    let minDrawY = maxY;
    let skyCol = vec4f(.3,.7,.8,1.0);

    //let pitch = cos(direction.y);
    let dirNoY = vec3f(direction.x, 0., direction.z);
    
    var maxDrawableY = i32(maxY-1);
    let halfY = f32(maxY)/2.;

    for (var dist = 1; dist < 8192; dist++) {
        let pdist = (f32(dist)/8192.)*(f32(dist)/8192.);

        let pos = origin + (f32(dist) * dirNoY);
        let mapPosX = f32(i32(pos.x) & 1023); //f32(1024);\
        let mapPosY = f32(i32(pos.z) & 1023);
        let scaledMapPos = vec2f(mapPosX, mapPosY) / 1024.0;
        let col = getColorAtPos(pos);
        let col2 = mix(col, skyCol, pdist);
        
        // 
        let height = getHeightAtPos(pos);
        let vec = pos - origin;
        var transY = origin.y - height;
        var heightInt = max(0, i32(pitch + halfY + yScale * transY / f32(dist))); //i32(halfY + halfY * projY);
        

        if(heightInt < maxDrawableY) {

            for(var y = heightInt; y < i32(maxDrawableY); y++) {
                textureStore(pass_out, vec2(x, y), 0, col2);
            }
            maxDrawableY = heightInt;
        }
    }
    
    for(var y = 0; y < i32(maxDrawableY); y++) {
        textureStore(pass_out, vec2(x, y), 0, skyCol);
    }

}

const PI = 3.14159;

fn rotX(p: vec3f, a: f32) -> vec3f { 
    let r = p.yz * cos(a) + vec2f(-p.z, p.y) * sin(a); 
    return vec3f(p.x, r); 
}
fn rotY(p: vec3f, a: f32) -> vec3f { 
    let r = p.xz * cos(a) + vec2f(-p.z, p.x) * sin(a); 
    return vec3f(r.x, p.y, r.y); 
}
fn rotM(p: vec3f, m: vec2f) -> vec3f { 
    return rotY(rotX(p, -PI * m.y), 2 * PI * m.x); 
}


// no reason for this to be in a separate pass, really
// but this could be used to store into a spherical/whatever buffer for 6dof

@compute @workgroup_size(64) // 13*64 => 832
#workgroup_count raycast_pass SCREEN_WIDTH/64 1 1
fn raycast_pass(@builtin(global_invocation_id) id: vec3u) {
    
    let res = textureDimensions(screen);
    if (id.x >= res.x) { return; }

    let screen_size = textureDimensions(screen);


    let u = f32(id.x) / f32(screen_size.x);
    
    var origin = vec3f(0, 130, 0);

    let uv = (2.*(vec2f(id.xy) + .5) - vec2f(res)) / f32(res.y);
    var ro = vec3f(0, 100, 2);
    var rd = normalize(vec3f(uv, -2));  
    let yRotAngle = f32(mouse.pos.x) / (f32(res.x/2));
    rd = rotY(rd, yRotAngle);

    // hack for pitch :)
    var pitch = 300. * cos(f32(mouse.pos.y) / (f32(res.y/2)-.5)) - 200;

    let rd2 = vec3f(rd.x, 0, rd.z);
    let ro2 = vec3f(0.0, 60., 3.5);// + vec3f(0, 0., -1.*time.elapsed*100);

    let aspect = f32(screen_size.x) / f32(screen_size.y);
    let fovScale = 1. / tan(.78/2.);
    let yScale = f32(screen_size.y)/2. * fovScale * aspect;

    raycast(i32(id.x), yScale, screen_size.y, pitch, ro2, rd2);


}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = uint2(textureDimensions(screen));

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // load from pass1 buffer
    let col = passLoad(0, vec2(i32(id.x), i32(id.y)), 0);

    textureStore(screen, id.xy, col);

}