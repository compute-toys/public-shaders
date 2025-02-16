// forked from https://compute.toys/view/213

fn hue2rgb(f1: f32, f2: f32, hue0: f32) -> f32 {
  var hue = hue0;
  if hue < 0.0 {
    hue += 1.0;
  } else if hue > 1.0 {
    hue -= 1.0;
  }
  var res: f32;
  if (6.0 * hue) < 1.0 {
    res = f1 + (f2 - f1) * 6.0 * hue;
  } else if (2.0 * hue) < 1.0 {
    res = f2;
  } else if (3.0 * hue) < 2.0 {
    res = f1 + (f2 - f1) * ((2.0 / 3.0) - hue) * 6.0;
  } else {
    res = f1;
  }
  return res;
}

fn hsl2rgb(hsl: vec3f) -> vec3f {
  var rgb = vec3f(0.0, 0.0, 0.0);
  if hsl.y == 0.0 {
    rgb = vec3f(hsl.z); // Luminance
  } else {
    var f2: f32;
    if hsl.z < 0.5 {
      f2 = hsl.z * (1.0 + hsl.y);
    } else {
      f2 = hsl.z + hsl.y - hsl.y * hsl.z;
    }
    let f1 = 2.0 * hsl.z - f2;
    rgb.r = hue2rgb(f1, f2, hsl.x + (1.0 / 3.0));
    rgb.g = hue2rgb(f1, f2, hsl.x);
    rgb.b = hue2rgb(f1, f2, hsl.x - (1.0 / 3.0));
  }
  return rgb;
}

// h from 0 to 1
fn hsl(h: f32, s: f32, l: f32) -> vec3f {
  return hsl2rgb(vec3f(h, s, l));
}

// complex product
fn product(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(
    a.x * b.x - a.y * b.y,
    a.x * b.y + a.y * b.x,
  );
}

// return the number of iterations before the point escapes
fn julia_escape(p: vec2<f32>, c: float2, max_iter: u32) -> u32 {
  var z = p;
  for (var i = 0u; i < max_iter; i = i + 1u) {
    z = product(z, z) + c;
    if length(z) > 2.0 {
      return i;
    }
  }

  return max_iter;
}

struct State {
  offset: vec2f,
  scale: f32,
  control: vec2<f32>,
  delta_time: f32,
  cached_elapsed: f32,
}

#storage global_state State

@compute @workgroup_size(1, 1, 1)
#dispatch_once init
fn init() {
  global_state.scale = 1.0;
  global_state.offset = vec2f(0., 0.);
  global_state.control = vec2f(0.272, 0.0);
  global_state.cached_elapsed = time.elapsed;
}


const key_i = 73u;
const key_k = 75u;
const key_j = 74u;
const key_l = 76u;
const key_w = 87u;
const key_s = 83u;
const key_a = 65u;
const key_d = 68u;

#workgroup_count update_state 1 1 1
@compute @workgroup_size(1, 1)
fn update_state() {
  let ratio = 1.004;
  let dt = time.elapsed - global_state.cached_elapsed;
  global_state.cached_elapsed = time.elapsed;
  var faster = 40.0 * dt;
  if keyDown(16) {
    faster = 10.0;
  }
  /* - */
  if keyDown(189)  {
    global_state.scale /= pow(ratio, faster);
  }
  /* = */
  if (keyDown(187)) {
    global_state.scale *= pow(ratio, faster);
  }

  let step = 0.008 * faster / global_state.scale;
  if keyDown(key_w) {
    global_state.offset += vec2f(0., step);
  }
  if keyDown(key_s) {
    global_state.offset += vec2f(0., -step);
  }
  if keyDown(key_a) {
    global_state.offset += vec2f(-step, 0.);
  }
  if keyDown(key_d) {
    global_state.offset += vec2f(step, 0.);
  }

  let refine = 0.0002 * faster / global_state.scale;

  if keyDown(key_i) {
    global_state.control += vec2f(0., refine);
  }
  if keyDown(key_k) {
    global_state.control += vec2f(0., -refine);
  }
  if keyDown(key_j) {
    global_state.control += vec2f(-refine, 0.);
  }
  if keyDown(key_l) {
    global_state.control += vec2f(refine, 0.);
  }

}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    // Viewport resolution (in pixels)
  let screen_size = textureDimensions(screen);
  let ratio = f32(screen_size.y) / f32(screen_size.x);

    // Prevent overdraw for workgroups on the edge of the viewport
  if id.x >= screen_size.x || id.y >= screen_size.y { return; }
  let narrow = f32(min(screen_size.x, screen_size.y));
  let x = (f32(id.x) - f32(screen_size.x) * 0.5) / narrow * 20.;
  let y = -(f32(id.y) - f32(screen_size.y) * 0.5) / narrow * 20.;

    // if abs(x) < 0.02 || abs(y) < 0.02 {
    //   textureStore(screen, id.xy, float4(1.0, 1., 1., 1.));
    //   return;
    // }
    // let r = length(vec2f(x,y));
    // if r < 10. && r > 9.88 {
    //   textureStore(screen, id.xy, float4(1.0, 1., 1., 1.));
    //   return;
    // }

    // if distance(vec2f(x,y), vec2f(5., 5.)) < 0.2 {
    //   textureStore(screen, id.xy, float4(1.0, 1., 1., 1.));
    //   return;
    // }

  let escape_steps = julia_escape(vec2f(x, y) * 0.1 / global_state.scale + global_state.offset, global_state.control, 500);
  let v = f32(escape_steps) * 0.001;
  let col = hsl(
    fract(0.6 + v * 70.), 0.5 + v * 0.5, fract(0.5 + v * 4.0)
  );
    // var col = float3(v, v, v);

    // Output to screen (linear colour space)
  textureStore(screen, id.xy, float4(col, 1.));
}