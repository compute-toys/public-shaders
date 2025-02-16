#define GRID_SIZE 100
#define STEP 0.01 //1.0/50.0 //GRID_SIZE
#define GOLDEN_RATIO 1.61803398875
#define TWOPI 6.28318531
#define MAX_DIST 50

#storage grid array<atomic<u32>>

fn getCameraMatrix(ro : vec3f, ta : vec3f) -> mat3x3<f32> {
    let a = normalize(ta - ro);
    let b = cross(a, vec3(0., 1., 0.));
    let c = cross(b, a);
    return mat3x3<f32>(b, c, a);
}

fn translate( x : f32, y : f32, z : f32 ) -> mat4x4<f32>
{
    return mat4x4<f32>( 1.0, 0.0, 0.0, 0.0,
                        0.0, 1.0, 0.0, 0.0,
                        0.0, 0.0, 1.0, 0.0,
                        x,   y,   z,   1.0 );
}

fn inverse( m : mat4x4<f32> ) -> mat4x4<f32>
{
    return mat4x4<f32>(
        m[0][0], m[1][0], m[2][0], 0.0,
        m[0][1], m[1][1], m[2][1], 0.0,
        m[0][2], m[1][2], m[2][2], 0.0,
        -dot(m[0].xyz,m[3].xyz),
        -dot(m[1].xyz,m[3].xyz),
        -dot(m[2].xyz,m[3].xyz),
        1.0 );
}


fn scaleAndBias(p : vec3f) -> vec3f {return p*0.5 + vec3f(0.5);}

fn getGridCenter(q : vec3f) -> vec3f {
    var p = scaleAndBias(q);
    p *= GRID_SIZE;
    p = floor(p);
    p += 0.5;
    p /= GRID_SIZE;
    p *= 2.;
    p -= 1.;
    return p;
}

fn pos2id(q : vec3f) -> u32 {
    var p = scaleAndBias(q);
    p *= GRID_SIZE;
    p = vec3f(floor(p.x), floor(p.y), floor(p.z) );
    return u32(p.x + p.y*GRID_SIZE + p.z*GRID_SIZE*GRID_SIZE);
}

fn grid2id(p : vec3u) -> u32 {
    return u32(p.x + p.y*GRID_SIZE + p.z*GRID_SIZE*GRID_SIZE);
}

fn hash( a : vec3u ) -> vec3f {
    var x = ((a>>vec3u(8u))^a.yzx)*1103515245u;
    x = ((x>>vec3u(8u))^x.yzx)*1103515245u;
    x = ((x>>vec3u(8u))^x.yzx)*1103515245u;
    
    return vec3f(x)*(1.0/f32(0xffffffffu));
}

// returns t and normal
fn getCube( ro : vec3f, rd : vec3f, c : vec3f) -> vec4f {

    let txi = translate(c.x, c.y, c.z);
    let txx = inverse( txi );
    //vec3 box = vec3(0.2,0.5,0.6) ;

    // convert from ray to box space
    let rdd = (txx*vec4(rd,0.0)).xyz;
    let roo = (txx*vec4(ro,1.0)).xyz;

    // ray-box intersection in box space
    let m = 1.0/rdd;
    let n = m*roo;
    let k = abs(m)*STEP;
    
    let t1 = -n - k;
    let t2 = -n + k;

    let tN = max( max( t1.x, t1.y ), t1.z );
    let tF = min( min( t2.x, t2.y ), t2.z );
    
    if( tN > tF || tF < 0.0) { return vec4(-1.0); }

    var nor = -sign(rdd)*step(t1.yzx,t1.xyz)*step(t1.zxy,t1.xyz);

    // convert to ray space
    nor = (txi * vec4(nor,0.0)).xyz;
    return vec4( tN, nor );
}


fn boxIntersection( ro : vec3f, rd : vec3f) -> vec2f{

    let m = 1.0/rd; // can precompute if traversing a set of aligned boxes
    let n = m*ro;   // can precompute if traversing a set of aligned boxes
    let k = abs(m);
    let t1 = -n - k;
    let t2 = -n + k;
    let tN = max( max( t1.x, t1.y ), t1.z );
    let tF = min( min( t2.x, t2.y ), t2.z );
    if( tN>tF || tF<0.0) { return vec2f(-1.0); } // no intersection
    return vec2f( tN, tF );
}

fn map(ro : vec3f, rd : vec3f) -> vec4f {

    let enterExit = boxIntersection(ro, rd);
    if(enterExit.x == -1.) { return vec4(-1.); }

    let enter = enterExit.x + 0.01;
    let exit = enterExit.y - 0.01;

    var currDist = enter;
    while(currDist < exit){
        var p = ro + rd*currDist;
        let presence = atomicLoad( &grid[pos2id(p)] );
        //float presence = texture(cubeTex, scaleAndBias(p)).r;
        if(presence > 0){
            var c = getGridCenter(p);
            return getCube(ro, rd, c);  
        }
        currDist += STEP;
        if(currDist > MAX_DIST) {return vec4(-1.); }
    }
    return vec4(-1.);

}

fn traverseShadow(ro : vec3f, rd_in : vec3f, rand : vec3f) -> f32{
    var rd = rd_in + rand*0.1;
    rd = normalize(rd);
    let exitDist = boxIntersection(ro, rd).y;
    var dist = STEP; 
    while(dist < exitDist){
        var p = ro + rd*dist;
        if(atomicLoad( &grid[pos2id(p)] ) > 0) { return 0.; }
        dist += STEP;
    }
    return 1.;
}

fn getAO(ro : vec3f, rd_in : vec3f, rand : vec3f) -> f32{
    let aoStep = STEP*1.;
    var dist = aoStep;
    //vec3 rand = hash(uvec3(abs(ro)*12084.39874))-vec3(0.5);
    let rd = normalize(rd_in + rand);
    let numSteps = 90;
    for(var i = 0; i < numSteps; i++){
        var p = ro + rd*dist;
        if(atomicLoad( &grid[pos2id(p)] ) > 0) {
            return 1. -  (f32(numSteps - i) / f32(numSteps))*0.99;
        }
        dist += aoStep;
    }
    return 1.;
}


fn getLight(p : vec3f, n : vec3f, v : vec3f, rand : vec3f) -> vec3f {

    //float dist = length(p);
    let alb = vec3(0.2); //vec3(sin(dist*30.)*0.5 + 0.5, cos(dist*17.)*0.5 + 0.5, 0.2);

    let r = reflect(v, n);
    let kD = 0.8;
    let kS = 1. - kD;

    var col = vec3f(0.);

    {
        let ligDir = normalize(vec3f(1., 1., 0.3));
        let ligCol = vec3f(3., 2., 1.)*3;
        let dif = max(0., dot(ligDir, n));
        let sha = traverseShadow(p, ligDir, rand);
        col += kD * dif * alb * ligCol * sha;
        var spe = max(0., dot(ligDir, r));
        spe *= spe;
        spe *= spe;
        spe *= spe;
        spe *= spe;
        spe *= spe;
        spe *= spe;
        spe *= spe;
        col += kS * spe * alb * ligCol * sha;
    }

    {
        let ligDir = normalize(vec3f(-1., -0.5, -0.3));
        let ligCol = vec3f(1., 2., 3.)*6;
        let dif = max(0., dot(ligDir, n));
        let sha = traverseShadow(p, ligDir, rand);
        col += kD * dif * alb * ligCol * sha;
        var spe = max(0., dot(ligDir, r));
        spe *= spe;
        spe *= spe;
        spe *= spe;
        spe *= spe;
        spe *= spe;
        spe *= spe;
        spe *= spe;
        col += kS * spe * alb * ligCol * sha;
    }
    {
        let ligDir = normalize(vec3f(0., 1, -2.));
        let ligCol =  vec3f(3);
        let dif = max(0., dot(ligDir, n));
        let sha = traverseShadow(p, ligDir, rand);
        col += kD * dif * alb * ligCol * sha;
        var spe = max(0., dot(ligDir, r));
        spe *= spe;
        spe *= spe;
        spe *= spe;
        spe *= spe;
        spe *= spe;
        spe *= spe;
        spe *= spe;
        col += kS * spe * alb * ligCol * sha;
    }

    var ao = getAO(p, n, rand);
    //ao *= ao;
    col += vec3f(ao*0.14);

    return col;
}



#workgroup_count fill_grid 10 1 1
#dispatch_count fill_grid 1

@compute @workgroup_size(4, 4)
fn fill_grid(  @builtin(global_invocation_id) id: uint3,
                @builtin(workgroup_id) w_id: vec3<u32>
            ) {

    let rand_pos = hash(id)*2 - 1;
    let grid_id = pos2id(rand_pos);
    atomicStore(&grid[grid_id], 1u);
}


/*
#workgroup_count fill_grid 1 1 1
#dispatch_count fill_grid 1

@compute @workgroup_size(1, 1)
fn fill_grid( ) { atomicStore(&grid[pos2id(vec3f(0.0, 0.0, 0.0))], 1u); }
*/
#workgroup_count aggregate 1000 1 1
//#dispatch_count aggregate 1

@compute @workgroup_size(3, 3, 3)
fn aggregate(  @builtin(local_invocation_id) id: uint3,
                @builtin(workgroup_id) w_id: vec3<u32>
            ) {

    if(id.x == 1 && id.y == 1 && id.z == 1) {return; }

    let seed = vec3u(   w_id.x + uint(time.elapsed*21345),
                        uint(time.elapsed*43261),
                        uint(time.elapsed*82738));

    let rand_pos = vec3u(hash(seed) * GRID_SIZE);
    let rand_pos_grid = grid2id(rand_pos);
    let pos_grid = rand_pos + vec3u(id.x - 1, id.y - 1, id.z - 1);
    if( pos_grid.x < 0 ||
        pos_grid.y < 0 ||
        pos_grid.z < 0 ||
        pos_grid.x >= GRID_SIZE ||
        pos_grid.y >= GRID_SIZE ||
        pos_grid.z >= GRID_SIZE) {return; }

    let grid_id = grid2id(pos_grid);

    atomicAdd(&grid[rand_pos_grid], atomicLoad(&grid[grid_id]));
}

//#workgroup_count main_image 8192 1 1

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id_line: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);
    let ratio = f32(screen_size.x) / f32(screen_size.y);

    let id = id_line;
    //let id = vec2u(id_line.x % screen_size.x, id_line.x/screen_size.x);
    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.y >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);

    // Normalised pixel coordinates (from 0 to 1)
    var uv = fragCoord / vec2f(screen_size);
    uv *= 2;
    uv -= 1;
    uv *= vec2f(ratio, 1.0);

    var col = vec3f(0.);

    let dof = floor(custom.samples*16.0) + 1.0;

    for(var i = 1; i < (int(dof)); i++){

        var seed = vec3u(   id.x + uint(time.elapsed*21345),
                            id.y + uint(time.elapsed*43261),
                            uint(time.elapsed*82738));

        var camRand = vec3f(hash(seed));

        camRand = vec3f(    fract(camRand.x + GOLDEN_RATIO * 1. * f32(i)),
                            fract(camRand.y + GOLDEN_RATIO * 2. * f32(i)),
                            fract(camRand.z + GOLDEN_RATIO * 3. * f32(i))); 

        var zoom = sin(time.elapsed*0.5)*2 + 9.0;

        var ro = vec3f(     cos(f32(time.frame)*0.03*custom.rotation_speed)*zoom, 
                            cos(f32(time.frame)*0.006*custom.rotation_speed)*5,
                            sin(f32(time.frame)*0.03*custom.rotation_speed)*zoom);

        let ro_prev = vec3f(    cos(f32(time.frame-1)*0.03*custom.rotation_speed)*zoom, 
                                cos(f32(time.frame-1)*0.006*custom.rotation_speed)*5,
                                sin(f32(time.frame-1)*0.03*custom.rotation_speed)*zoom);
        
        ro = mix(ro, ro_prev, camRand.x);

        var ta = vec3f(0., 0., 0.); //vec3f(cos(time.elapsed*0.35)*0.3, sin(time.elapsed*0.15)*0.4, sin(time.elapsed*0.13)*0.3);
        var planeDist = 6.0;
        var camMat = getCameraMatrix(ro, ta);
        var rd = normalize(camMat * vec3f(uv, planeDist));

        var focalPoint = ro + rd*((custom.focal_dist*2.5 - 1.0) + length(ro));
        var bokhe = vec3f(cos(camRand.x*TWOPI), sin(camRand.x*TWOPI), 0.)*sqrt(camRand.y);
        ro += camMat * (bokhe * custom.aperture);

        rd = normalize(focalPoint - ro);
        ro += camMat * vec3(camRand.yz / vec2f(screen_size), 0.); //antialiasing
        var scene = map(ro, rd);

        if(scene.x > 0.){ 

            let seed2 = vec3u(   id.x + uint(time.elapsed*21345) + 1,
                                id.y + uint(time.elapsed*43261) + 1,
                                uint(time.elapsed*82738) + 1);

            let rand = vec3f(hash(seed2))*2 - 1;
            let p = ro + rd*scene.x;
            let v = normalize(p - ro);
            //rand *= 2.;
            //rand -= 1.;
            col += getLight(p, scene.yzw, v, rand);
        }
    }

    col /= dof;
    col /= col + 1.0;
    col = pow(col, vec3f(0.4545454545));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}
