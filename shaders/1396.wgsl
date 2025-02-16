fn sun(uv: vec2f, battery: f32) -> f32 {
 	let val = 1. - smoothstep(0.29, 0.3, length(uv));

 	let bloom = 1. - smoothstep(0.0, 0.7, length(uv));
    var cut = 3.0 * sin((uv.y + time.elapsed * 0.2 * (battery + 0.02)) * 100.0) 
				+ clamp(uv.y * 14.0 + 1.0, -6.0, 6.0);
    cut = clamp(cut, 0.0, 1.0);
    return clamp(val * cut, 0.0, 1.0) + bloom * 0.6;
}

fn grid(u: vec2f, battery: f32) -> f32 {
    var uv =u;
    let size = vec2f(uv.y, uv.y * uv.y * 0.2) * 0.01;
    uv += vec2f(0.0, time.elapsed * 4.0 * (battery + 0.05));
    uv = abs(fract(uv) - 0.5);
 	var lines = smoothstep(size, vec2f(0.0), uv);
 	lines += smoothstep(size * 5.0, vec2f(0.0), uv) * 0.4 * battery;
    return clamp(lines.x + lines.y, 0.0, 3.0);
}

fn dot2(v:vec2f) -> f32{ 
    return dot(v,v); 
}

fn sdTrapezoid(p: vec2f, r1: f32, r2: f32, he: f32) -> f32 {
  let k1 = vec2f(r2, he);
  let k2 = vec2f(r2 - r1, 2. * he);
  let q = vec2f(abs(p.x), p.y);
  let ca = vec2f(q.x - min(q.x, select(r2, r1, q.y < 0.0)), abs(q.y) - he);
  let cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / dot(k2, k2), 0., 1.);
  let s = select(1., -1., cb.x < 0.0 && ca.y < 0.0);
  return s * sqrt(min(dot(ca, ca), dot(cb, cb)));
}

fn sdLine(p: vec2f, a: vec2f, b: vec2f) -> f32 {
  let pa = p - a;
  let ba = b - a;
  let h = clamp(dot(pa, ba) / dot(ba, ba), 0., 1.);
  return length(pa - ba * h);
}

fn sdBox(p: vec2f, b: vec2f) -> f32 {
  let d = abs(p) - b;
  return length(max(d, vec2f(0.))) + min(max(d.x, d.y), 0.);
}

fn opSmoothUnion(d1: f32, d2: f32, k: f32) -> f32 {
	let h = clamp(0.5 + 0.5 * (d2 - d1) /k,0.0,1.0);
    return mix(d2, d1 , h) - k * h * ( 1.0 - h);
}

fn sdCloud(
        p_: vec2f, 
        a1_: vec2f, 
        b1_: vec2f, 
        a2_: vec2f, 
        b2_: vec2f, 
        w: f32
    ) -> f32 {

    //😥  https://github.com/gpuweb/gpuweb/issues/4113
	var p = p_;
    var a1 = a1_;
    var b1 = b1_;
    var a2 = a2_;
    var b2 = b2_;

    //var lineVal1 = smoothstep(w - 0.0001, w, sdLine(p, a1, b1));
    var lineVal1 = sdLine(p, a1, b1);
    var lineVal2 = sdLine(p, a2, b2);
    var ww = vec2f(w*1.5, 0.0);
    var left = max(a1 + ww, a2 + ww);
    var right = min(b1 - ww, b2 - ww);
    var boxCenter = (left + right) * 0.5;
    //var boxW = right.x - left.x;
    var boxH = abs(a2.y - a1.y) * 0.5;
    //var boxVal = sdBox(p - boxCenter, vec2f(boxW, boxH)) + w;
    var boxVal = sdBox(p - boxCenter, vec2f(0.04, boxH)) + w;
    
    var uniVal1 = opSmoothUnion(lineVal1, boxVal, 0.05);
    var uniVal2 = opSmoothUnion(lineVal2, boxVal, 0.05);
    
    return min(uniVal1, uniVal2);
}

// thanks: https://github.com/gpuweb/gpuweb/issues/3987#issuecomment-1528783750
fn modulo_euclidean (a: f32, b: f32) -> f32 {
	var m = a % b;
	if (m < 0.0) {
		if (b < 0.0) {
			m -= b;
		} else {
			m += b;
		}
	}
	return m;
}


@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);

    //var uv = fragCoord / vec2f(screen_size);
    var uv = (2. * fragCoord.xy - vec2f(screen_size).xy) / vec2f(screen_size).y;


    var battery = 1.0;

    var col1 = vec3f(0.);

    //if (abs(uv.x) < (9.0 / 16.0)) 
    {

        // Grid
        var fog = 1. - smoothstep( -0.02, 0.1, abs(uv.y + 0.2));
        var col = vec3f(0.0, 0.1, 0.2);
        if (uv.y < -0.2)
        {
            uv.y = 3.0 / (abs(uv.y + 0.2) + 0.05);
            uv.x *= uv.y * 1.0;
            var gridVal = grid(uv, battery);
            col = mix(col, vec3(1.0, 0.5, 1.0), gridVal);
        }
        else
        {
            var fujiD = min(uv.y * 4.5 - 0.5, 1.0);
            uv.y -= battery * 1.1 - 0.51;
            
            var sunUV = uv;
            var fujiUV = uv;
            
            // Sun
            sunUV += vec2f(0.75, 0.2);
            //uv.y -= 1.1 - 0.51;
            col = vec3(1.0, 0.2, 1.0);
            var sunVal = sun(sunUV, battery);
            
            col = mix(col, vec3(1.0, 0.4, 0.1), sunUV.y * 2.0 + 0.2);
            col = mix(vec3(0.0, 0.0, 0.0), col, sunVal);
            
            // fuji
            var fujiVal = sdTrapezoid( uv  + vec2f(-0.75+sunUV.y * 0.0, 0.5), 1.75 + pow(uv.y * uv.y, 2.1), 0.2, 0.5);
            var waveVal = uv.y + sin(uv.x * 20.0 + time.elapsed * 2.0) * 0.05 + 0.2;
            var wave_width =  smoothstep(0.0,0.01,(waveVal));
            
            // fuji color
            col = mix( col, mix(vec3(0.0, 0.0, 0.25), vec3(1.0, 0.0, 0.5), fujiD), step(fujiVal, 0.0));
            // fuji top snow
            col = mix( col, vec3(1.0, 0.5, 1.0), wave_width * step(fujiVal, 0.0));
            // fuji outline
            col = mix( col, vec3(1.0, 0.5, 1.0), 1.0-smoothstep(0.0,0.01,abs(fujiVal)) );
            //col = mix( col, vec3(1.0, 1.0, 1.0), 1.0-smoothstep(0.03,0.04,abs(fujiVal)) );
            //col = vec3(1.0, 1.0, 1.0) *(1.0-smoothstep(0.03,0.04,abs(fujiVal)));
            
            // horizon color
            col += mix( col, mix(vec3(1.0, 0.12, 0.8), vec3(0.0, 0.0, 0.2), clamp(uv.y * 3.5 + 3.0, 0.0, 1.0)), step(0.0, fujiVal) );
            
            // cloud
            var cloudUV = uv;
            cloudUV.x = modulo_euclidean (
                cloudUV.x + time.elapsed * 0.1, 4.0
                ) - 2.0;
            var cloudTime = time.elapsed * 0.5;
            var cloudY = -0.5;
            var cloudVal1 = sdCloud(
                cloudUV, 
                vec2f(0.1 + sin(cloudTime + 140.5)*0.1,cloudY), 
                vec2f(1.05 + cos(cloudTime * 0.9 - 36.56) * 0.1, cloudY), 
                vec2f(0.2 + cos(cloudTime * 0.867 + 387.165) * 0.1,0.25+cloudY), 
                vec2f(0.5 + cos(cloudTime * 0.9675 - 15.162) * 0.09, 0.25+cloudY), 
                0.075
            );
            cloudY = -0.6;
            var cloudVal2 = sdCloud(
                cloudUV, 
                vec2f(-0.9 + cos(cloudTime * 1.02 + 541.75) * 0.1,cloudY), 
                vec2f(-0.5 + sin(cloudTime * 0.9 - 316.56) * 0.1, cloudY), 
                vec2f(-1.5 + cos(cloudTime * 0.867 + 37.165) * 0.1,0.25+cloudY), 
                vec2f(-0.6 + sin(cloudTime * 0.9675 + 665.162) * 0.09, 0.25+cloudY), 
                0.075
            );
            
            var cloudVal = min(cloudVal1, cloudVal2);
            
            //col = mix(col, vec3(1.0,1.0,0.0), smoothstep(0.0751, 0.075, cloudVal));
            col = mix(col, vec3(0.0, 0.0, 0.2), 1.0 - smoothstep(0.075 - 0.0001, 0.075, cloudVal));
            col += vec3(1.0, 1.0, 1.0)*(1.0 - smoothstep(0.0,0.01,abs(cloudVal - 0.075)));
        }

        col += fog * fog * fog;
        col = mix(vec3(col.r, col.r, col.r) * 0.5, col, battery * 0.7);

        col1 = vec3(col);
    }

    

    // Convert from gamma-encoded to linear colour space
    col1 = pow(col1, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col1, 1.));
}