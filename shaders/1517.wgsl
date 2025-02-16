const PI = 3.14159274;
const TWO_PI = 6.28318548;

// !! lower this if performance is poor
#define D_COUNT 64

var<private> seed: u32;


#define SPECTRUM_LEN 45

const spectrum = array<vec3<f32>, SPECTRUM_LEN>(
    vec3( 0.002362f, 0.000253f, 0.010482f ),    // 390 nm
    vec3( 0.019110f, 0.002004f, 0.086011f ),    // 400 nm
    vec3( 0.084736f, 0.008756f, 0.389366f ),    // 410 nm
    vec3( 0.204492f, 0.021391f, 0.972542f ),    // 420 nm
    vec3( 0.314679f, 0.038676f, 1.553480f ),    // 430 nm
    vec3( 0.383734f, 0.062077f, 1.967280f ),    // 440 nm
    vec3( 0.370702f, 0.089456f, 1.994800f ),    // 450 nm
    vec3( 0.302273f, 0.128201f, 1.745370f ),    // 460 nm
    vec3( 0.195618f, 0.185190f, 1.317560f ),    // 470 nm
    vec3( 0.080507f, 0.253589f, 0.772125f ),    // 480 nm
    vec3( 0.016172f, 0.339133f, 0.415254f ),    // 490 nm
    vec3( 0.003816f, 0.460777f, 0.218502f ),    // 500 nm
    vec3( 0.037465f, 0.606741f, 0.112044f ),    // 510 nm
    vec3( 0.117749f, 0.761757f, 0.060709f ),    // 520 nm
    vec3( 0.236491f, 0.875211f, 0.030451f ),    // 530 nm
    vec3( 0.376772f, 0.961988f, 0.013676f ),    // 540 nm
    vec3( 0.529826f, 0.991761f, 0.003988f ),    // 550 nm
    vec3( 0.705224f, 0.997340f, 0.000000f ),    // 560 nm
    vec3( 0.878655f, 0.955552f, 0.000000f ),    // 570 nm
    vec3( 1.014160f, 0.868934f, 0.000000f ),    // 580 nm
    vec3( 1.118520f, 0.777405f, 0.000000f ),    // 590 nm
    vec3( 1.123990f, 0.658341f, 0.000000f ),    // 600 nm
    vec3( 1.030480f, 0.527963f, 0.000000f ),    // 610 nm
    vec3( 0.856297f, 0.398057f, 0.000000f ),    // 620 nm
    vec3( 0.647467f, 0.283493f, 0.000000f ),    // 630 nm
    vec3( 0.431567f, 0.179828f, 0.000000f ),    // 640 nm
    vec3( 0.268329f, 0.107633f, 0.000000f ),    // 650 nm
    vec3( 0.152568f, 0.060281f, 0.000000f ),    // 660 nm
    vec3( 0.081261f, 0.031800f, 0.000000f ),    // 670 nm
    vec3( 0.040851f, 0.015905f, 0.000000f ),    // 680 nm
    vec3( 0.019941f, 0.007749f, 0.000000f ),    // 690 nm
    vec3( 0.009577f, 0.003718f, 0.000000f ),    // 700 nm
    vec3( 0.004553f, 0.001768f, 0.000000f ),    // 710 nm
    vec3( 0.002175f, 0.000846f, 0.000000f ),    // 720 nm
    vec3( 0.001045f, 0.000407f, 0.000000f ),    // 730 nm
    vec3( 0.000508f, 0.000199f, 0.000000f ),    // 740 nm
    vec3( 0.000251f, 0.000098f, 0.000000f ),    // 750 nm
    vec3( 0.000126f, 0.000050f, 0.000000f ),    // 760 nm
    vec3( 0.000065f, 0.000025f, 0.000000f ),    // 770 nm
    vec3( 0.000033f, 0.000013f, 0.000000f ),    // 780 nm
    vec3( 0.000018f, 0.000007f, 0.000000f ),    // 790 nm
    vec3( 0.000009f, 0.000004f, 0.000000f ),    // 800 nm
    vec3( 0.000005f, 0.000002f, 0.000000f ),    // 810 nm
    vec3( 0.000003f, 0.000001f, 0.000000f ),    // 820 nm
    vec3( 0.000002f, 0.000001f, 0.000000f )     // 830 nm
);

const xyz_to_rgb = mat3x3f(
     3.2404542,-0.9692660, 0.0556434,
    -1.5371385, 1.8760108,-0.2040259,
    -0.4985314, 0.0415560, 1.0572252
);

fn rot(a: float) -> mat2x2f{ return mat2x2f(cos(a), -sin(a),sin(a), cos(a));}

fn rotX(a: float) -> mat3x3f{
    let r = rot(a); return mat3x3f(1.,0.,0.,0.,r[0][0],r[0][1],0.,r[1][0],r[1][1]);
}
fn rotY(a: float) -> mat3x3f{
    let r = rot(a); return mat3x3f(r[0][0],0.,r[0][1],0.,1.,0.,r[1][0],0.,r[1][1]);
}
fn rotZ(a: float) -> mat3x3f{
    let r = rot(a); return mat3x3f(r[0][0],r[0][1],0.,r[1][0],r[1][1],0.,0.,0.,1.);
}

fn hash_u(_a: uint) -> uint {
    var a = _a;
    a ^= a >> 16;
    a *= 0x7feb352du;
    a ^= a >> 15;
    a *= 0x846ca68bu;
    a ^= a >> 16;
    return a;
}

fn rand() -> float {
    var s = hash_u(seed);
    seed = s;
    return  float( s ) / float( 0xffffffffu ); 
}

fn rand_vec2() -> vec2<f32> {
    return vec2(rand(), rand()); 
}

fn rand_vec3() -> vec3<f32> {
    return vec3(rand(), rand(), rand()); 
}

// wavelength
fn wl_to_xyz(wl: f32) -> vec3<f32> {
    let x = (wl - 390.0) * 0.1;
    let index = u32(x);
    if(index < 0 || index >= SPECTRUM_LEN-1) 
    {
        return vec3(0.0);
    }

    return mix(spectrum[index], spectrum[index+1], fract(x));
}


fn lens(_p: vec2<f32>, z: f32, wl: f32) -> vec2<f32> {
    let wl_d = (wl - 600.0) / 200.0;

    var p = _p;

    let d = dot(p, p);

    let r_sq = rand();
    let r = sqrt(r_sq);

    let o = 1.-sqrt(1-r_sq);

    let focus = 2.*z - 0.1*d + 0.25*o - 0.1*wl_d;
    var dof_size = 0.1 * focus;

    let a = TWO_PI * rand();
    var c = dof_size * r * vec2(cos(a), sin(a));

    let coma = 1.5 - 1.5 * wl_d;
    p += coma * (p*dot(c,c) + c*dot(p,c));

    let curv = -0.05 + 0.01 * wl_d;
    p += curv * (c*dot(p,p) + 2*p*dot(p,c));

    p += c;

    p /= sqrt(1. + 0.12 * dot(p,p));

    return p;
}

#storage hist array<array<array<atomic<u32>, 3>, SCREEN_HEIGHT>, SCREEN_WIDTH>

#workgroup_count Splat 4096 1 1
@compute @workgroup_size(D_COUNT, 1, 1)
fn Splat(@builtin(global_invocation_id) id: uint3) {
    let res = uint2(textureDimensions(screen));
    let res_f = vec2<f32>(res);
    let aspect = f32(res.y) / f32(res.x);

    let mpos = (vec2<f32>(mouse.pos) - 0.5*res_f)/res_f.y;

    // seed = id.x + u32(time.elapsed * 15675416.);
    seed = id.x;

    let time = 420.0 + time.elapsed + rand()/30.;


    let wl = 390.0 + 440 * rand();
    let wl_x = (wl - 600.) / 200.0;

    let col = wl_to_xyz(wl);

    var p = vec3(0., 0.2, 0.);
    // var p = rand_vec3();

    let iters = 160;
    for(var i = 0; i < iters; i++){ 

        let r = rand() + 0.22*wl_x;


        if(r<0.33){
            p += vec3(0.1, 0.1*sin(0.2*time), 0.);
            p *= rotZ(1.8);
            let d = dot(p, p);
            p += 0.05*sin(0.168 + 0.012*time + p*2.75 )/(1. + d);
            p *= 0.98;
        } else if(r<0.75){
            p += vec3(0.3,-0.1,0.1 );
            p *= rotX(1.8);
            p *= 0.5 ;

        } else {
            p += vec3(0.1,0.1,0.1);

            var k = rand() + wl_x;
            k *= pow(max(0., sin(0.1*time)), 20.0);
            p *= rotZ(0.6 + k);

            p *= 1. + 0.2*sin(0.012*time) ;

            var d = max(2.0, dot(p,p));
            p /= d;
        }

        var q = p;

        let z_off = custom.P + 0.1*sin(0.1*time);

        q *= rotX(- 0.02*time - 0.5);
        q *= rotY(-mpos.x*1. - 0.01*time - 2.5);

        q.z += z_off;

        let z = q.z;

        var l = min(2., 0.2 / (1.0 + q.z)); 
        q /= (1.0+q.z);

        var f = custom.focus + mpos.y;
        f -= 0.4*pow(sin(0.23*time)*0.5+0.5,20.);
        f += 0.3*z_off;
        var uv = lens(q.xy, f-z*0.5, wl);

        uv.x *= aspect;
        uv = uv*0.5 + 0.5;
        let cc = uint2(uv * float2(res));
        if (z > -1.0 && uv.x > 0. && uv.x < 1. && uv.y > 0. && uv.y < 1.){
            let c = uint3(l * col * 256);   
            atomicAdd(&hist[cc.x][cc.y][0], c.r);
            atomicAdd(&hist[cc.x][cc.y][1], c.g);
            atomicAdd(&hist[cc.x][cc.y][2], c.b);
        }
    }
}

fn aces_tonemap(color: vec3<f32>) -> vec3<f32> {
    const m1 = mat3x3f(
        0.59719, 0.07600, 0.02840,
        0.35458, 0.90834, 0.13383,
        0.04823, 0.01566, 0.83777
	);
	const m2 = mat3x3f(
        1.60475, -0.10208, -0.00327,
        -0.53108,  1.10813, -0.07276,
        -0.07367, -0.00605,  1.07602
	);

	var v = m1 * color;    
	var a = v * (v + 0.0245786) - 0.000090537;
	var b = v * (0.983729 * v + 0.4329510) + 0.238081;
	return m2 * (a / b);
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    let res = uint2(textureDimensions(screen));
    if (id.x >= res.x || id.y >= res.y) { return; }
    
    let s_size = (res.x*res.y);
    var col = vec3(
        float(atomicLoad(&hist[id.x][id.y][0])),
        float(atomicLoad(&hist[id.x][id.y][1])),
        float(atomicLoad(&hist[id.x][id.y][2])),
    );

    // white balance
    col = col * vec3(0.95047, 1.0, 1.08883);

    col = xyz_to_rgb * col;
    col = col * f32(s_size) * 2e-9 / D_COUNT;
    col = col * pow(2, custom.exposure*0.5);
    col = max(vec3(0.0), col);
    col = aces_tonemap(col);

    textureStore(screen, int2(id.xy), float4(col, 1.));

    // clear
    atomicStore(&hist[id.x][id.y][0],0);
    atomicStore(&hist[id.x][id.y][1],0);
    atomicStore(&hist[id.x][id.y][2],0);
}
