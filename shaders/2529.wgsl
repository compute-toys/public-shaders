#define WG_X 16
#define WG_Y 16
#define MAX_BALLS 256
#define HEADER_WORDS 1
#define WORDS_PER_BALL 2

#storage balls array<float4, HEADER_WORDS + MAX_BALLS*WORDS_PER_BALL>

fn idx_header() -> int { return 0; }
fn idx_ball(i: uint) -> int { return int(HEADER_WORDS + i*WORDS_PER_BALL); }

fn read_count() -> uint { return uint(balls[idx_header()].x); }
fn write_count(c: uint) { balls[idx_header()].x = f32(c); }
fn read_last_click() -> f32 { return balls[idx_header()].y; }
fn write_last_click(v: f32) { balls[idx_header()].y = v; }

fn read_pv(i: uint) -> float4 { return balls[idx_ball(i)+0]; }
fn write_pv(i: uint, v: float4) { balls[idx_ball(i)+0] = v; }
fn read_pr(i: uint) -> float4 { return balls[idx_ball(i)+1]; }
fn write_pr(i: uint, v: float4) { balls[idx_ball(i)+1] = v; }

fn hash12(p: float2) -> f32 {
  let q = fract(float2(dot(p, float2(127.1,311.7)), dot(p, float2(269.5,183.3))) * 0.1031);
  let r = q + dot(q, q.yx + float2(33.33,33.33));
  return fract((r.x + r.y) * r.x);
}

fn spawn_ball(sp: float2, sz: float2) {
  var n = read_count();
  if (n >= MAX_BALLS) { return; }
  let r = mix(70.0, 14.0, hash12(sp*0.017));
  let ph = hash12(sp*0.029) * 6.2831853;
  let pos = clamp(sp, float2(r+1.0,r+1.0), sz - float2(r+1.0,r+1.0));
  write_pv(n, float4(pos, 0.0, 0.0));
  write_pr(n, float4(r, 0.0, ph, 0.0));
  write_count(n+1u);
}

fn sd_jelly(p: float2, c: float2, r: f32, wob: f32, ph: f32) -> f32 {
  let d = p - c;
  let a = atan2(d.y, d.x);
  let rr = r * (1.0 + 0.18*wob*sin(6.0*a + ph) + 0.10*sin(3.0*a - 0.7*ph));
  return length(d) - rr;
}

#workgroup_count step 1 1 1
@compute @workgroup_size(1,1,1)
fn step(@builtin(global_invocation_id) gid: vec3u) {
  let sz = vec2f(SCREEN_WIDTH, SCREEN_HEIGHT);
  let dt = max(time.delta, 1.0/120.0);
  var n = read_count();
  if (time.frame == 0u) { write_count(0u); write_last_click(0.0); n = 0u; }
  let clicked = (mouse.click != 0);
  let was = read_last_click();
  if (clicked && was <= 0.0) { spawn_ball(vec2f(mouse.pos), sz); }
  write_last_click(select(0.0, 1.0, clicked));

  for (var i: uint = 0u; i < n; i++) {
    var pv = read_pv(i);
    var pr = read_pr(i);
    var pos = pv.xy;
    var vel = pv.zw;
    var rad = pr.x;
    var wob = pr.y;
    var ph  = pr.z;
    var wv  = pr.w;

    let force = -28.0*wob - 6.0*wv;
    wv += force*dt;
    wob += wv*dt;
    wob *= 0.985;
    wv  *= 0.98;

    vel += vec2f(0.0, -0.45)*dt;
    pos += vel*dt;

    if (pos.x < rad) { pos.x = rad; vel.x = -vel.x*0.72; wob += 0.18; }
    if (pos.x > sz.x - rad) { pos.x = sz.x - rad; vel.x = -vel.x*0.72; wob += 0.18; }
    if (pos.y < rad) { pos.y = rad; vel.y = -vel.y*0.72; wob += 0.18; }
    if (pos.y > sz.y - rad) { pos.y = sz.y - rad; vel.y = -vel.y*0.72; wob += 0.18; }

    ph += dt*3.2;

    write_pv(i, float4(pos, vel));
    write_pr(i, float4(rad, wob, ph, wv));
  }

  for (var i: uint = 0u; i < n; i++) {
    var pvi = read_pv(i);
    var pri = read_pr(i);
    var pi = pvi.xy;
    var vi = pvi.zw;
    var ri = pri.x;
    for (var j: uint = i+1u; j < n; j++) {
      var pvj = read_pv(j);
      var prj = read_pr(j);
      var pj = pvj.xy;
      var vj = pvj.zw;
      var rj = prj.x;

      let d = pj - pi;
      let L = length(d);
      let minL = ri + rj;
      if (L > 0.0 && L < minL) {
        let nrm = d / L;
        let dep = minL - L;
        pi -= nrm*(0.5*dep);
        pj += nrm*(0.5*dep);
        let rel = dot(vj - vi, nrm);
        let imp = -1.15*rel;
        vi -= nrm*(0.5*imp);
        vj += nrm*(0.5*imp);
        pri.y += dep*0.28;
        prj.y += dep*0.28;
        write_pv(i, float4(pi, vi));
        write_pv(j, float4(pj, vj));
        write_pr(i, pri);
        write_pr(j, prj);
      }
    }
  }
}

#workgroup_count image 128 128 1
@compute @workgroup_size(WG_X, WG_Y, 1)
fn image(@builtin(global_invocation_id) id: vec3u) {
  let coord = vec2i(id.xy);
  if (coord.x >= SCREEN_WIDTH || coord.y >= SCREEN_HEIGHT) { return; }
  let px = vec2f(coord);
  let n = read_count();
  var mind = 1e9;
  var g = vec2f(0.0, 0.0);
  var col = vec3f(0.0, 0.0, 0.0);
  for (var i: uint = 0u; i < n; i++) {
    let pv = read_pv(i);
    let pr = read_pr(i);
    let d = sd_jelly(px, pv.xy, pr.x, clamp(pr.y, -1.0, 1.0), pr.z);
    if (d < mind) {
      mind = d;
      let e = 0.8;
      let dx = sd_jelly(px + vec2f(e,0.0), pv.xy, pr.x, clamp(pr.y,-1.0,1.0), pr.z) - d;
      let dy = sd_jelly(px + vec2f(0.0,e), pv.xy, pr.x, clamp(pr.y,-1.0,1.0), pr.z) - d;
      g = normalize(vec2f(dx, dy));
      let t = 0.5 + 0.5*sin(pr.z*0.5);
      col = mix(vec3f(0.15,0.35,0.9), vec3f(0.95,0.55,0.2), t);
    }
  }
  let ao = clamp(1.0 - max(-mind*0.09, 0.0), 0.0, 1.0);
  let L = normalize(vec2f(0.6, 0.8));
  let diff = clamp(dot(-g, L), 0.0, 1.0);
  let V = normalize(vec2f(0.4, 0.9));
  let R = reflect(L, vec2f(g.x, -g.y));
  let spec = pow(clamp(dot(R, V), 0.0, 1.0), 24.0);
  let edge = smoothstep(1.2, 0.0, mind);
  let bg = vec3f(0.06,0.07,0.10) + 0.05*vec3f(f32(coord.y)/f32(SCREEN_HEIGHT));
  let s = mix(vec3f(0.02,0.02,0.03), col*diff*ao + spec*0.35, edge);
  textureStore(screen, coord, vec4f(bg + s, 1.0));
}
