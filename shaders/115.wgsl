
#define SIM_GROUP_SIZE 128

#define NUM_PARTICLE 3300

#define SIM_PARTICLE_GROUP_COUNT NUM_PARTICLE


struct FluidParticle {
    position : float2,
    velocity : float2,
    force : float2,
    density:float,
    pressure:float
};


//particle array buffer
#storage fluid_particle array<FluidParticle,NUM_PARTICLE>
//atomic storage buffer to draw
#storage atomic_storage array<atomic<i32>>



#define PI 3.1415927410125732421
#define PARTICLE_RADIUS 10./522.
#define PARTICLE_RESTING_DENSITY 200
#define PARTICLE_MASS 0.2
#define SMOOTHING_LENGTH (4 * PARTICLE_RADIUS)
#define PARTICLE_STIFFNESS 3000
#define PARTICLE_VISCOSITY 10000.f
#define GRAVITY_FORCE vec2(0, -9806.65)






#workgroup_count ComputePressure SIM_PARTICLE_GROUP_COUNT 1 1
@compute @workgroup_size(SIM_GROUP_SIZE, 1)
fn ComputePressure(@builtin(global_invocation_id) id : uint3)
{
     let i = id.x;
      if(id.x >= NUM_PARTICLE) { return;}
     var density_sum = 0.f;
     
    for (var j = 0; j < NUM_PARTICLE; j++)
    {
      //  if(i == j) {continue;}
        var delta = fluid_particle[i].position - fluid_particle[j].position;
        var r = length(delta);
          if (r < SMOOTHING_LENGTH){
              density_sum += PARTICLE_MASS * /* poly6 kernel */ 315.f * pow(SMOOTHING_LENGTH * SMOOTHING_LENGTH - r * r, 3) / (64.f * PI * pow(SMOOTHING_LENGTH, 9));

          }
    }
     fluid_particle[i].density = density_sum;
    // compute pressure
    fluid_particle[i].pressure = max(PARTICLE_STIFFNESS * (density_sum - PARTICLE_RESTING_DENSITY), 0.f);
}

#workgroup_count ComputeForce SIM_PARTICLE_GROUP_COUNT 1 1
@compute @workgroup_size(SIM_GROUP_SIZE, 1)
fn ComputeForce(@builtin(global_invocation_id) id : uint3)
{
    let i = id.x;
    if(id.x >= NUM_PARTICLE) { return;}
    var pressure_force = vec2(0., 0.);
    var viscosity_force = vec2(0., 0.);
    for (var j = 0; j < NUM_PARTICLE; j++)
    {
         if (int(i) == j)
        {
            continue;
        }
         var delta = fluid_particle[i].position - fluid_particle[j].position;
        var r = length(delta);
          if (r < SMOOTHING_LENGTH){

              pressure_force -= PARTICLE_MASS * (fluid_particle[i].pressure + fluid_particle[j].pressure) / (2.f * fluid_particle[j].density) *
            // gradient of spiky kernel
                -45.f / (PI * pow(SMOOTHING_LENGTH, 6)) * pow(SMOOTHING_LENGTH - r, 2) * normalize(delta);
          
              viscosity_force += PARTICLE_MASS * (fluid_particle[j].velocity - fluid_particle[i].velocity) / fluid_particle[j].density *
            // Laplacian of viscosity kernel
                45.f / (PI * pow(SMOOTHING_LENGTH, 6)) * (SMOOTHING_LENGTH - r);
          }

    }  
    viscosity_force *= PARTICLE_VISCOSITY;
    var external_force = fluid_particle[i].density * GRAVITY_FORCE;

    fluid_particle[i].force = pressure_force + viscosity_force + external_force;  

}
@compute @workgroup_size(16, 16)
fn Clear(@builtin(global_invocation_id) id: uint3) {
    let screen_size = int2(textureDimensions(screen));
    let idx0 = int(id.x) + int(screen_size.x * int(id.y));

    atomicStore(&atomic_storage[idx0*4+0], 0);
    atomicStore(&atomic_storage[idx0*4+1], 0);
    atomicStore(&atomic_storage[idx0*4+2], 0);
    atomicStore(&atomic_storage[idx0*4+3], 0);
}
fn AdditiveBlend(color: float4, index: int)
{
    let scaledColor = 256.0 * color;

    atomicMax(&atomic_storage[index*4+0], int(scaledColor.x));
    atomicMax(&atomic_storage[index*4+1], int(scaledColor.y));
    atomicMax(&atomic_storage[index*4+2], int(scaledColor.z));
    atomicMax(&atomic_storage[index*4+3], int(scaledColor.a));
}


#workgroup_count Advect SIM_PARTICLE_GROUP_COUNT 1 1
@compute @workgroup_size(SIM_GROUP_SIZE, 1)
fn Advect(@builtin(global_invocation_id) id : uint3)
{

    let i = id.x;
    if(id.x >= NUM_PARTICLE) { return;}
    var acceleration = fluid_particle[i].force / fluid_particle[i].density;
    var new_velocity = fluid_particle[i].velocity + 0.0001 * acceleration;
    var new_position = fluid_particle[i].position + 0.0001 * new_velocity;

    var screen_size = int2(textureDimensions(screen));

   if (new_position.x < -1)
    {
        new_position.x = -1;
        new_velocity.x *= -1;
    }
    else if (new_position.x > 1)
    {
        new_position.x = 1;
        new_velocity.x *= -1 ;
    }
    else if (new_position.y < -1)
    {
        new_position.y = -1;
        new_velocity.y *= -1 ;
    }
    else if (new_position.y > 1)
    {
        new_position.y = 1;
        new_velocity.y *= -1 ;
    }
    if(time.frame <2){
        let space = 3.0;
        let grid = 20.0;
        let off = 0.0;
        new_position = vec2(off+float(int(i)%int(grid))*(space),off+floor(float(i)/grid)*space);
        new_position = new_position/float2(screen_size);
        new_position =(new_position*2.)-1.;
        new_velocity = vec2(sin(new_position.x),0.);
    }
    if(mouse.click == 1){
      var mp =   float2(mouse.pos.xy)/float2(screen_size);
      mp.y = 1.-mp.y;
    mp = (mp*2)-1.;
        if(length(new_position-mp)<0.3){
            var del = mp-new_position;
          //  float angl = del.x
            if(abs(del.x)>abs(del.y)){
                new_velocity.y+=sign(del.x)*abs(del.x)*4.;
            }else{
                 new_velocity.x-=sign(del.y)*abs(del.x)*20.;
            }
        
        }
    }
    fluid_particle[i].velocity = new_velocity;
    fluid_particle[i].position = new_position;
    var pressure = fluid_particle[i].pressure;
    var density = fluid_particle[i].density;
    var vel = new_velocity;
    var vmag = dot(new_velocity,new_velocity);
    var pcol = max(1.-(pressure*0.00000015),0.);
    var finalCol = vec3(0.01,0.01,0.07)+vmag*0.00001*vec3(12.0,0.1,0.1)+pcol;
  //  buffer_particle[id.x].position += buffer_particle[id.x].velocity * time.delta;
    var conv = int2((fluid_particle[id.x].position.xy*0.5+0.5)*float2(screen_size));
    conv.y = screen_size.y-conv.y;
    let k = 10;
    for(var i = -k;i<=k;i++){
        for(var j = -k;j<=k;j++){
                var c = vec2(float(i),float(j));
            
                var gauss = (30./(dot(c,c)));
                var col =  vec4(0.4)*gauss;
              // let newCol = oldCol*2.+vec4(0.5)*gauss;
              var pos = conv+int2(i,j);
              let idx = pos.x + screen_size.x * pos.y;
              //let cc = vec3(sin(float(id.x)*0.0002)*0.5+0.5,cos(float(id.x)*0.0002)*0.5+0.5,sin(float(id.x+5)*0.0002)*0.5+0.5)*0.02;
              col = float4(col.r,finalCol);
             //  textureStore(pass_out, conv+int2(i,j), 2,col );
               AdditiveBlend(col,idx);
        }
    }

}
fn Sample(pos: int2) -> float4
{
    let screen_size = int2(textureDimensions(screen));
    let idx = pos.x + screen_size.x * pos.y;

    var color: float4;
        let x = float(atomicLoad(&atomic_storage[idx*4+0]))/(256.0);
        let y = float(atomicLoad(&atomic_storage[idx*4+1]))/(256.0);
        let z = float(atomicLoad(&atomic_storage[idx*4+2]))/(256.0);
        let a = float(atomicLoad(&atomic_storage[idx*4+3]))/(256.0);
        color = float4(x,y,z,a);

    return abs(color);
}



@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    // Viewport resolution (in pixels)
     let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = float2(float(id.x) + .5, float(screen_size.y - id.y) - .5);

    // Normalised pixel coordinates (from 0 to 1)
    let uv = fragCoord / float2(screen_size);



    
  
   
    var col = (Sample(int2(id.xy)));
 var sm =smoothstep(0.0,12.2,col.x);
//sm = smoothstep(0.5,-0.01,sm);
 col = float4(col.gba,1.)*sm;

    textureStore(screen, int2(id.xy), col);
}
