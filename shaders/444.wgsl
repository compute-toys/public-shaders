// created by florian berger (flockaroo) - 2023
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

// experiment to efficiently draw triangles in WebGPU

#define NumTri 10000
//#define MaxTriPerDiv NumTri
#define MaxTriPerDiv 1000
// screen in divided into ScrDiv sections in both x and y direction
#define ScrDiv 32

// squared signed distance to triangle in pixel (squared to avoid sqrt)
fn distTriSq(a:vec2f, b:vec2f, c:vec2f, p:vec2f) -> f32
{
    let ab=a-b;
    let bc=b-c;
    let ca=c-a;
    let d1=dot(p-a,ab.yx*vec2f(1,-1));
    let d2=dot(p-b,bc.yx*vec2f(1,-1));
    let d3=dot(p-c,ca.yx*vec2f(1,-1));
    return max(d1*abs(d1)/dot(ab,ab),max(d2*abs(d2)/dot(bc,bc),d3*abs(d3)/dot(ca,ca)));
}

#storage s_vert array<vec2f,NumTri*3>
#storage s_trilist array<array<int,MaxTriPerDiv+1>,ScrDiv*ScrDiv>

// create a bunch of triangle coordinates
//#dispatch_count setup_verts 64
#workgroup_count setup_verts 256 256 1
@compute @workgroup_size(16,16)
fn setup_verts(@builtin(global_invocation_id) id: vec3u) {
    let i=id.x+id.y*1024;
    if (i>=NumTri*3) { return; }
    let R = textureDimensions(screen);
    var r=f32(R.x)/30.;
    r*=(f32((i/3)%3)+3.)/5.;

    let c=r+(sin(f32(i/3)*vec2f(1,.27)+time.elapsed*.1)*.5+.5)*(vec2f(R.xy)-r*2.);
    s_vert[i]=c+vec2f(sin(f32(i*2)+time.elapsed*4.*(f32(((i/3)*17)%6)/2.5-.7)+vec2f(0,1.57)))*r;
}

// just sort out all triangles that potentially intersect with a screen-division section
#workgroup_count trilists 16 16 1
@compute @workgroup_size(16,16)
fn trilists(@builtin(global_invocation_id) id: vec3u) {
    let listIdx = id.y*100+id.x;
    if(listIdx>=ScrDiv*ScrDiv) { return; }
    let R = textureDimensions(screen);
    
    let S = (R+ScrDiv-1)/ScrDiv;
    let dmax2=dot(vec2f(S),vec2f(S))*.25;
    let c = (vec2f(vec2u(listIdx%ScrDiv,listIdx/ScrDiv))+.5)*vec2f(S);
    
    var cnt=0;
    for (var i = 0; i < NumTri; i++) {
        let d=distTriSq(s_vert[i*3+0],s_vert[i*3+1],s_vert[i*3+2],c);
        if (d>dmax2) { continue; }
        s_trilist[listIdx][cnt]=i;
        cnt++;
        if(cnt>=MaxTriPerDiv) { break; }
    }
    s_trilist[listIdx][cnt]=-1;
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
    //for (var i = 0; i < NumTri; i++) { let ti=i;
    for (var i = 0; s_trilist[sect][i]>=0; i++) { let ti=s_trilist[sect][i];
        var d=distTriSq(s_vert[ti*3+0],s_vert[ti*3+1],s_vert[ti*3+2],vec2f(id.xy)+.5);
        if(d>0.) { continue; }
        d=sqrt(abs(d))*sign(d);
        let tf=f32(ti);
        d=d/f32(R.x)*1200.;
        col=mix(col,(.2+.8*fract(vec3f(tf*.21,tf*.35,tf*.77)))*clamp(sin(-d)*3.-2.,.0,1.),sin(-d)*.3+.7);
    }
    
    textureStore(screen, id.xy, vec4f(col,1));
}
