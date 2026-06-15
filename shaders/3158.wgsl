
#define PI (acos(-1.))
#define prec 1000.

#define width float(textureDimensions(screen).x)
#define height float(textureDimensions(screen).y)

#define numAgent 50000

#define speed 1.

#define decayRate 0.007
#define diffuseRate .5

#define sensorOffset 40.
#define angleSpacing (40. * (PI/ 180.))
#define tSpeed .5

struct Agent 
{
	pos: float2,
	angle: float
}

fn hash(state: uint) -> uint
{
    var s = state;
    s ^= 2747636419u;
    s *= 2654435769u;
    s ^= s >> 16;
    s *= 2654435769u;
    s ^= s >> 16;
    s *= 2654435769u;
    return s;
}

#storage agents array<Agent>
#storage trail array<atomic<u32>>


fn trailAdd (pix: uint, f: float)
{
    atomicAdd(&trail[pix], uint(f * prec));
}

fn trailWrite (pix: uint, f: float)
{
    atomicStore(&trail[pix], uint(f * prec));
}

fn trailRead (pix: uint) -> float
{
    return min(float(atomicLoad(&trail[pix]))/prec, 1.0);
}

fn coordToId(coord: int2) -> uint
{
    return uint(coord.y * int(textureDimensions(screen).x) + coord.x);
}

fn sense (A: Agent, angleOffset: float) -> float
{
	let newAngle: float = A.angle + angleOffset;
	let dir = float2(cos(newAngle), sin(newAngle));
	let center: float2 = A.pos + dir * sensorOffset;
	
    var sum = 0.;
	
	for (var i = -1; i <= 1; i++)
	{
		for (var j = -1; j <= 1; j++)
		{
			//if (i == 0 && j == 0) {continue;}
			let sampleX: float = min(width - 1, max(0, center.x + float(i))); 
			let sampleY: float = min(height - 1, max(0, center.y + float(j)));
			let pos = int2(int(floor(sampleX)), int(floor(sampleY)));
			let newID = coordToId(pos);

			sum += trailRead(newID);
		}
	}
	return sum;
}

@compute @workgroup_size(16, 16)
#dispatch_once initialization
fn initialization(@builtin(global_invocation_id) id: vec3u) 
{
    let screen_size = textureDimensions(screen);
    let pixID: uint = id.y * screen_size.x + id.x;	
    if (pixID > numAgent) {return;}

    let r = sqrt((float(hash(pixID+5)) / pow(2., 32))) * 50.;
    let theta = (float(hash(pixID)) / pow(2., 32)) * 2. * PI;
    let center = float2(width/2., height/2.);
    let startPos = float2(center.x + r * cos(theta), center.y + r * sin(theta));
    agents[pixID].pos = startPos; 
    agents[pixID].angle = atan2(normalize(center - startPos).y, normalize(center - startPos).x);

    trailWrite(pixID, 0.);
}

@compute @workgroup_size(16, 16)
fn simulation(@builtin(global_invocation_id) id: vec3u) 
{
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);
    let pixID: uint = id.y * screen_size.x + id.x;	
    if (pixID > numAgent) {return;}

    let A: Agent = agents[pixID];
    let agent_id: uint = uint(floor(A.pos.y * width + A.pos.x));
    let rand: uint = hash(agent_id + hash(pixID));

    let turnSpeed = tSpeed * PI * 2.;
    let sensorF = sense(A, 0);
    let sensorL = sense(A, angleSpacing);
    let sensorR = sense(A, -angleSpacing);

    if (sensorF > sensorL && sensorF > sensorR)
    {
        agents[pixID].angle += 0.;
    }
    else if (sensorF < sensorL && sensorF < sensorR)
    {
        agents[pixID].angle += ((float(rand) / pow(2., 32))-0.5)*2.* turnSpeed;
    }
    else if (sensorR > sensorL)
    {
        agents[pixID].angle -= (float(rand) / pow(2., 32)) * turnSpeed;
    }
    else if (sensorL > sensorR)
    {
        agents[pixID].angle += (float(rand) / pow(2., 32)) * turnSpeed;
    }

    // update position
    let dir: float2 = float2(cos(A.angle), sin(A.angle));
    var newPos: float2 = A.pos + dir * speed;

    // bounce off of corners
    if (newPos.x < 0. || newPos.x >= width || newPos.y < 0. || newPos.y >= height)
    {
        newPos.x = min(width - 1., max(0., newPos.x));
        newPos.y = min(height - 1., max(0., newPos.y));
        agents[pixID].angle += float(rand)/pow(2., 32) * PI;
    }
    else
    {
        let coord: uint = uint(floor(newPos.y) * width + floor(newPos.x));
        trailAdd(coord, 1.);
    }
    agents[pixID].pos = newPos;
}


@compute @workgroup_size(16, 16)
fn diffusion(@builtin(global_invocation_id) id: vec3u) 
{
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);
    // Prevent overdraw for workgroups on the edge of the viewport
    if (any(id.xy >= screen_size)) { return; }

    let pixID: uint = id.y * screen_size.x + id.x;
    let intID: int2 = int2(int(id.x), int(id.y));

	var mean: float = 0.;
	let originalTrail: float = trailRead(pixID);
	for (var i = -1; i <= 1; i++)
	{
		for (var j = -1; j <= 1; j++)
		{
			let off: int2 = int2(i, j);
            let newID: uint = coordToId(max(int2(0), intID + off));

			mean += trailRead(newID);
		}
	}
	mean /= 9.;
	let diffuseColor: float = mix(originalTrail, mean, diffuseRate);
 
	trailWrite(pixID, max(0., diffuseColor - decayRate));
}


@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) 
{
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);
    // Prevent overdraw for workgroups on the edge of the viewport
    if (any(id.xy >= screen_size)) { return; }

    let pixID: uint = id.y * screen_size.x + id.x;

    let col: float = trailRead(pixID);
    textureStore(screen, id.xy, vec4f(col));
}
