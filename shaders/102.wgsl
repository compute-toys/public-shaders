
var<private> iTime:f32;
var<private> trap:vec3f;

// https://www.shadertoy.com/view/4lXyWN
fn hash(_x:vec3u)->vec3f{
    let k=1103515245u;
    var x=_x*k;
    x = ((x>>vec3u(2))^(x.yzx>>vec3u(1))^x.zxy)*k;
    return vec3f(x)*(1/f32(0xffffffffu));
}

fn hue(a:f32)->vec3f{
    return cos(a*6.3+vec3f(0,21,23))*.5+.5;
}

fn euler(a:vec2f)->vec3f{
    return vec3f(cos(a.y)*sin(a.x),sin(a.y),cos(a.y)*cos(a.x));
}

fn rot(p:vec3f, _a:vec3f, t:f32)->vec3f{
  	let a=normalize(_a);
  	return mix(a*dot(p,a),p,cos(t))+sin(t)*cross(p,a);
}

fn polarAbs(p:vec2f, n:f32)->vec2f{
  	let a=asin(sin(atan2(p.y,p.x)*n))/n;
  	return vec2f(cos(a),sin(a))*length(p);
}

fn Q(t:f32, a:f32, b:f32)->vec3f{
    let i=floor(t);
    return mix(
        hash(bitcast<u32>(i)  +vec3u(9,7,5)),
        hash(bitcast<u32>(i+1)+vec3u(9,7,5)),
        smoothstep(a,b,fract(t))
    );
}

fn map(_p:vec3f)->f32{
    var p=_p;
    p=asin(sin(p/3)*.997)*3;
    p=rot(p,vec3f(1),iTime);
    p+=cross(cos(p*.3),sin(p*.4));
    //p.xy=polarAbs(p.xy,3.); // bug?
    p=vec3f(polarAbs(p.xy,10.),p.z);
    //p.x-=2.;
    p.x-= 5*custom.A;
    //p.zx=polarAbs(p.zx,3.); // bug?
    p=vec3f(polarAbs(p.zx,3.),p.y).yzx;
    //p.z-=1.5;
    p.z-=5*custom.B;
     p.z=asin(sin(p.z));
    trap=p;
    var q=p;
    p-=clamp(p,vec3f(-.15),vec3f(.15));
    var de=1.;
    de=min(de,length(p)-.03);
    de=min(de,length(q.xy)-.01);
    return abs(de)+.002;
}

fn lookat()->mat3x3f{
    let n=Q(iTime*.1,.1,.3)*2-1;
    let w=euler(n.xy);
    //let w=euler(vec2f(-0.3,0.2));
    let u=cross(vec3(0,1,0),w);
    return mat3x3f(u,cross(u,w),w);
    /*
    return mat3x3f(
        1,0,0,
        0,1,0,
        0,0,1
    );
    */
}

fn eye()->vec3f{
    let n=Q(iTime*.1,.7,.9)*2-1;
    return n*3+vec3f(0,0,iTime);
    //return vec3f(0,0,-5);
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let iResolution = vec2u(textureDimensions(screen));
    if( any( id.xy >= iResolution ) ){return;}
    iTime = time.elapsed;
    let fragCoord = vec2f(f32(id.x)+.5, f32(iResolution.y-id.y)-.5);
    let uv = (vec2f(id.xy)*2-vec2f(iResolution))/f32(iResolution.y);
    var col = vec3f(0);
    col+=hash(id+time.frame)*.05;
    let n=f32(u32(fragCoord.x)+u32(fragCoord.y)*iResolution.x);
    col+=Q(n*.02+iTime,0,1).xxx*.01*dot(uv,uv);
    var rd=lookat()*normalize(vec3f(uv,.5));
    var g=0.;
    var e=0.;
    for(var i=0.;i<100.;i+=1.){
        var p=rd*g+eye()-i/1e4;
        e=map(p)*.8;
        g+=e;
        col+=mix(vec3f(1),hue(trap.z*.5),.4)*.13/exp(i*i*e);
    }
    col*=col*col*col;
    textureStore(screen, vec2i(id.xy), vec4f(col, 1.));
}
