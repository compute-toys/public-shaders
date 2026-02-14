
#storage atomic_storage atomic < i32>;

#workgroup_count bababooey 1 1 1
@compute @workgroup_size(1, 1)
fn bababooey(@builtin(global_invocation_id) id : vec3u)
{
    atomicStore(&atomic_storage, 0);
}

@compute @workgroup_size(8, 8)
fn main_image(@builtin(global_invocation_id) id : vec3u)
{
  let screen_size = textureDimensions(screen);

  if (id.x >= screen_size.x || id.y >= screen_size.y)
  {
    return;
  }

  let prev = atomicAdd(&atomic_storage, 1);

  var col = vec3f(f32(prev) / (f32(screen_size.x) * f32(screen_size.y)));

  textureStore(screen, id.xy, vec4f(col, 1.));
}
