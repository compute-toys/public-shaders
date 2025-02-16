// created by florian berger (flockaroo) - 2023
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

// experiment to efficiently draw triangles in WebGPU
// + some terrain test

#define NumVert 30000
#define MaxTriPerDiv 1000
// screen in divided into ScrDiv sections in both x and y direction
#define ScrDiv 32

fn UV6(idx: uint) -> vec2f { let i3=idx/3; let i=idx-i3*3; let uv=vec2u((i&1)|(i>>1),i>>1); return vec2f(uv)+f32(i3&1)*(1.-2.*vec2f(uv)); }
fn ROTM(ang:f32) -> float2x2 { return float2x2(cos(ang),sin(ang),-sin(ang),cos(ang)); }

// squared signed distance to triangle in pixel (squared to avoid sqrt)
fn distTriSq(a:vec2f, b:vec2f, c:vec2f, p:vec2f) -> f32
{
    let ab=a-b;
    let bc=b-c;
    let ca=c-a;
    var s=sign(ab.x*ca.y-ab.y*ca.x);
    let d1=dot(p-a,ab.yx*vec2f(1,-1))*s;
    let d2=dot(p-b,bc.yx*vec2f(1,-1))*s;
    let d3=dot(p-c,ca.yx*vec2f(1,-1))*s;
    return max(d1*abs(d1)/dot(ab,ab),max(d2*abs(d2)/dot(bc,bc),d3*abs(d3)/dot(ca,ca)));
}

struct State {
    vert: array<vec4f, NumVert>,
    trilist: array<array<int, MaxTriPerDiv+1>,ScrDiv*ScrDiv>,
}

#storage state State

fn FRUSTUM(l:f32,r:f32,b:f32,t:f32,n:f32,f:f32) ->float4x4 
{ 
    return float4x4( 2.*n/((r)-(l)),0,0,0, 
                     0,2.*n/((t)-(b)),0,0, 
                     ((r)+(l))/((r)-(l)), ((t)+(b))/((t)-(b)), ((n)+(f))/((n)-(f)), -1, 
                     0,0,2.*(f)*(n)/((n)-(f)),0 ); 
}

fn mytex(tex:texture_2d<f32>, uv0:vec2f) -> vec4f {
    let uv=fract(uv0);
    let r=textureDimensions(tex);
    let c0=uv*vec2f(r);
    let c0i=vec2u(c0);
    let cf=smoothstep(vec2f(0.),vec2f(1.),fract(c0));
    return mix(
    mix(textureLoad(tex,(c0i+vec2u(0,0))%r,0),textureLoad(tex,(c0i+vec2u(1,0))%r,0),cf.x),
    mix(textureLoad(tex,(c0i+vec2u(0,1))%r,0),textureLoad(tex,(c0i+vec2u(1,1))%r,0),cf.x),
    cf.y);
}

// create a bunch of triangle coordinates
#workgroup_count setup_verts NumVert 1 1
@compute @workgroup_size(1,1)
fn setup_verts(@builtin(global_invocation_id) id: vec3u) {
    let i=id.x;
    let R = textureDimensions(screen);
    var r=f32(R.x)/30.;
    r*=(f32((i/3)%3)+3.)/5.;

#define XNUM uint(sqrt(NumVert/6))
    let i6=i/6;
    var uv6=UV6(i);
    var p=vec3f((vec2f(f32(i6%XNUM),f32(i6/XNUM))+uv6)/f32(XNUM)*2.-1.,0);
    p+=vec3(vec2(time.elapsed*.3),0);
    p.z+=(mytex(channel0,p.xy*.08).x-.5)*.05;
    p.z+=(mytex(channel0,p.xy*.04).x-.5)*.10;
    p.z+=(mytex(channel0,p.xy*.02).x-.5)*.20;
    p.z+=(mytex(channel0,p.xy*.01).x-.5)*.40;
    p-=vec3(vec2(time.elapsed*.3),0);
    var ph=f32(mouse.pos.x)/f32(R.x)*10.+.5+time.elapsed*.13;
    var th=f32(mouse.pos.y)/f32(R.y)*10.+1.;
    p=vec3f(p.xy*ROTM(ph),p.z);
    p=vec3f(p.yz*ROTM(th),p.x).zxy;
    p.z-=3.;

    let aspect = float(R.y)/float(R.x);
    let tanFOVh=.5;
    var p4 = FRUSTUM(-.1,.1, -.1*aspect,.1*aspect, .1/tanFOVh,100.) * vec4(p,1);

    p4=p4/p4.w;

    state.vert[i]=vec4((p4.xy*vec2f(1,-1))*vec2f(R)*.5+vec2f(R)*.5,p4.zw);

    let c=r+(sin(f32(i/3)*vec2f(1,.27)+time.elapsed*.1)*.5+.5)*(vec2f(R.xy)-r*2.);    
    state.vert[i]=mix(state.vert[i],
    vec4(c+vec2f(sin(f32(i*2)+time.elapsed*4.*(f32(((i/3)*17)%6)/2.5-.7)+vec2f(0,1.57)))*r,0,1),
    smoothstep(0.,1.,-(cos(time.elapsed*.2)-.5)*.7)*.99
    );
}

// just sort out all triangles that potentially intersect with a screen-division section
#workgroup_count trilists ScrDiv ScrDiv 1
@compute @workgroup_size(1,1)
fn trilists(@builtin(global_invocation_id) id: vec3u) {
    let R = textureDimensions(screen);
    
    let S = (R+ScrDiv-1)/ScrDiv;
    let dmax2=dot(vec2f(S),vec2f(S))*.25;
    let c = (vec2f(id.xy)+.5)*vec2f(S);
    
    let listIdx = id.y*ScrDiv+id.x;

    var cnt=0;
    for (var i = 0; i < NumVert/3; i++) {
        let d=distTriSq(state.vert[i*3+0].xy,state.vert[i*3+1].xy,state.vert[i*3+2].xy,c);
        if (d>dmax2) { continue; }
        state.trilist[listIdx][cnt]=i;
        cnt++;
        if (cnt>=MaxTriPerDiv) { break; }
    }
    state.trilist[listIdx][cnt]=-1;
}

// draw all triangles in actual screen-division section
@compute @workgroup_size(16, 16)
fn draw_triangles(@builtin(global_invocation_id) id: vec3u) {
    let R = textureDimensions(screen);
    if any(id.xy >= R) { return; }

    let S = (R+ScrDiv-1)/ScrDiv;

    var sec2=id.xy/S;
    var sect=sec2.x+sec2.y*ScrDiv;
    
    var col=vec3f(0,0,0);
    var z=1000.;
    //for (var i = 0; i < NumVert/3; i++) { let ti=i;
    for (var i = 0; state.trilist[sect][i]>=0; i++) { let ti=state.trilist[sect][i];
        var d=distTriSq(state.vert[ti*3+0].xy,state.vert[ti*3+1].xy,state.vert[ti*3+2].xy,vec2f(id.xy)+.5);
        if(d>0.) { continue; }
        d=sqrt(abs(d))*sign(d);
        let tf=f32(ti);
        if (state.vert[ti*3+0].z<z) {
            col=mix(col,(.2+.8*fract(vec3f(tf*.21,tf*.35,tf*.77)))*clamp(sin(-d)*3.-2.,.0,1.),sin(-d)*.3+.7);
            z=state.vert[ti*3+0].z;
        }
    }
    
    textureStore(screen, id.xy, vec4f(col,1));
}
