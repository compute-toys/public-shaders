#storage state array<vec3i>

const DIM: vec2u = vec2(16, 9);
const SQRT2: float = 1.41421356237;
const STEP_DURATION = .1666;

// Buffer layout
const BDIR    = 0u; // Direction
const BNDIR   = 1u; // Next direction (input)
const BBUFDIR = 2u; // Buffered input
const BLASTIN = 3u; // Last input (for edge detection)
const BLEN    = 4u; // Length (current / max)
const BAPPLE  = 5u; // Apple position
const BTIME   = 6u; // last update time
const BFLASH  = 7u; // Death flash
const BRES    = 8u; // Detect resolution changes
const BSNAKE  = 9u; // Segment locations start here

const SNAKE_COL = vec3f(0.18, 0.45, 0.99);

fn random (st: vec2f) -> float {
    return fract(sin(dot(st.xy, vec2(12.9898,78.233)))*43758.5453123);
}

fn sdBox(p: vec2f, b: vec2f) -> float
{
    let d = abs(p)-b;
    return length(max(d,vec2f(0))) + min(max(d.x,d.y),0.0);
}

fn sdSegment(p: vec2f, a: vec2f, b: vec2f) -> float
{
    let pa = p-a;
    let ba = b-a;
    let h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    return length( pa - ba*h );
}

fn sdCircle(p: vec2f, r: float) -> float
{
    return length(p) - r;
}

fn sdEquilateralTriangle(pa: vec2f, sz: float) -> float
{
    const k = sqrt(3.0f);
    var p = pa / sz;
    p.x = abs(p.x) - 1.0;
    p.y = p.y + 1.0/k;
    if( p.x+k*p.y>0.0 ) { p = vec2(p.x-k*p.y,-k*p.x-p.y)/2.0; }
    p.x -= clamp( p.x, -2.0, 0.0 );
    return -length(p)*sign(p.y)*sz;
}

fn step_prog() -> float {
    return (time.elapsed - bitcast<float>(state[BTIME].x)) / STEP_DURATION;
}


fn blend(src: vec4f, dest: vec3f) -> vec3f
{
    return mix(dest, src.rgb, src.a);
}

fn dist2col(dist: float, fg: vec3f) -> vec4f {
    return vec4f(fg, saturate(0.5 - dist / SQRT2));
}

fn place_apple() 
{
    let useed2 = vec2u(state[BDIR].xy ^ state[BSNAKE].xy ^ state[BSNAKE + 2].xy ^ state[BLEN].xy ^ state[BAPPLE].xy);
    let useed = ((useed2.x >> 16) ^ useed2.x ^ (useed2.y >> 16) ^ useed2.y) & 0xffff;
    let seed = vec2f(time.elapsed, float(useed));
    let rands = vec2f(random(seed), random(seed + vec2f(1.3)));
    var pos = vec2i(rands * vec2f(DIM));
    pos = clamp(pos, vec2i(0), vec2i(DIM - 1));
    var valid = true;
    let len = vec2u(state[BLEN].xy);
    for (var i = 0u; i < len.x; i += 1)
    {
        valid = valid && !all(state[BSNAKE + i].xy == pos);
    }
    if (valid)
    {
        state[BAPPLE] = vec3i(pos, 0);
    }
    else
    {
        state[BAPPLE] = vec3i(-1);
    }
}


fn reset()
{
    let dir = vec2i(1, 0);
    state[BDIR]  = vec3i(dir, 0);
    state[BNDIR] = vec3i(dir, 0);
    state[BBUFDIR] = vec3i(0);
    state[BLEN]  = vec3i(2, 3, 0);
    let head = vec2i(1, 5);
    state[BSNAKE] = vec3i(head, 0);
    state[BSNAKE + 1] = vec3i(head - dir, 0);
    state[BSNAKE + 2] = vec3i(head - dir - dir, 0);
    state[BAPPLE] = vec3(6, 5, 0);
    state[BFLASH] = vec2i(bitcast<int>(-9999999.0), 0).xyy;
    state[BTIME] = vec2i(bitcast<int>(time.elapsed), 0).xyy;
}

fn input(early: bool) {
    var ddir = vec2i(0);
    if (keyDown(39) || keyDown(68))
    {
        ddir = vec2i(1, 0);
    }
    if (keyDown(38) || keyDown(87))
    {
        ddir = vec2i(0, 1);
    }
    if (keyDown(37) || keyDown(65))
    {
        ddir = vec2i(-1, 0);
    }
    if (keyDown(40) || keyDown(83))
    {
        ddir = vec2i(0, -1);
    }

    if (all(ddir == state[BLASTIN].xy)) { return; } // Edge detection

    state[BLASTIN] = vec3i(ddir, 0);

    if (all(ddir == vec2i(0)))
    {
        return;
    }

    let dir = state[BDIR].xy;
    if (early) // Early: snap retroactively
    {
        let prev = state[BSNAKE + 1].xy;
        if (any(ddir != state[BSNAKE + 2].xy - prev))
        {
            state[BDIR] = vec3i(ddir, 0);
            state[BNDIR] = vec3i(ddir, 0);
            let nwhead = prev + ddir;
            state[BSNAKE] = vec3i(nwhead, 0);
        }
        else // Would turn in place, buffer instead
        {
            state[BNDIR] = vec3i(ddir, 0);
        }
    }
    else // Late: Store input for next update
    {
        if (all(ddir == dir) || all(ddir == -dir))
        {
            state[BBUFDIR] = vec3i(ddir, 0);
        }
        else
        {
            state[BNDIR] = vec3i(ddir, 0);    
        }
    }
}

fn death_check()
{
    let head = state[BSNAKE].xy;
    let len = vec2u(state[BLEN].xy);
    var dead = any(head < vec2(0)) || any(head >= vec2i(DIM));
    for (var i = 1u; i < len.x; i += 1)
    {
        dead = dead || all(state[BSNAKE + i].xy == head);
    }
    if (dead)
    {
        reset();
        state[BFLASH] = vec2i(bitcast<int>(time.elapsed), 0).xyy;
    }
}

#workgroup_count step 1 1 1
@compute @workgroup_size(1)
fn step(@builtin(global_invocation_id) id: uint3) {
    if (any(id != vec3(0u))) { return; }
    
    // Initialization / resolution change check
    let screen_size = textureDimensions(screen);
    if (any(screen_size != vec2u(state[BRES].xy))) // Not initialized
    {
        state[BRES] = vec3i(vec2i(screen_size), 0);
        reset();
        return;
    }

    // Input
    let prog = step_prog();
    input(prog < 0.5);

    // Death check
    if (prog > 0.4)
    {
        death_check();
    }

    // Time check & update
    let last_time = bitcast<float>(state[BTIME].x);
    if (time.elapsed < last_time + STEP_DURATION) { return; }
    state[BTIME] = vec2i(bitcast<int>(last_time + STEP_DURATION), 0).xyy;

    // Grow
    var len = vec2u(state[BLEN].xy);
    if (len.x < len.y)
    {
       len.x += 1;
       state[BLEN] = vec3i(vec2i(len), 0);
    }

    // Eat
    let eaten = all(state[BSNAKE] == state[BAPPLE]);
    if (eaten)
    {
        state[BAPPLE] = vec3i(-1);
        len.y += 3;
        state[BLEN] = vec3i(vec2i(len), 0);
        state[BSNAKE] = vec3i(state[BSNAKE].xy, 1);
    }

    // Process input
    var dir = state[BDIR].xy;
    let ndir = state[BNDIR].xy;
    if (all(dir != ndir) && all(dir != -ndir))
    {
        dir = ndir;
        state[BDIR] = vec3i(ndir, 0);
    }
    var bufdir = state[BBUFDIR].xy;
    if (any(bufdir != vec2i(0)))
    {
        state[BNDIR] = vec3i(bufdir, 0);
        state[BBUFDIR] = vec3i(vec2i(0), 0);
    }
    else
    {
        state[BNDIR] = vec3i(dir, 0);
    }

    // Move snake
    for (var i = 0u; i <= len.x; i += 1) // One extra for animation
    {
        let cur = state[BSNAKE + len.x - i];
        state[BSNAKE + len.x - i + 1] = cur;
    }
    let head = state[BSNAKE].xy + dir;
    state[BSNAKE] = vec3i(head, 0);
    
    // Place apple (one attempt per frame, may be unsuccesful)
    if (all(state[BAPPLE].xy < vec2i(0)))
    {
        place_apple();
    }
}

fn draw_eye(frag: vec2f, hpos: vec2f, fw: vec2f, sw: vec2f, grid2px: float, cola: vec3f) -> vec3f
{
    var col = cola;
    let rad = grid2px * 0.23;
    let eye = sdCircle(frag - (hpos + (.1 * fw + 0.23 * sw) * grid2px), rad);
    col = blend(dist2col(eye, SNAKE_COL), col);
    col = blend(dist2col(eye + 0.3 * rad, vec3f(1.0)), col);
    let pupil = sdCircle(frag - (hpos + (.17 * fw + 0.23 * sw) * grid2px), 0.3 * rad);
    col = blend(dist2col(pupil, vec3f()), col);
    return col;
}

@compute @workgroup_size(16, 16)
fn render(@builtin(global_invocation_id) id: uint3) {
    let screen_size = textureDimensions(screen);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let pxpergrid = vec2f(screen_size) / vec2f(DIM);
    let grid2px = min(pxpergrid.x, pxpergrid.y);
    let rfrag = float2(float(id.x) + .5, float(screen_size.y - id.y) - .5) - 0.5 * grid2px;
    let frag = rfrag - 0.5 * (vec2f(screen_size) - vec2f(DIM) * grid2px);
    var prog: float = step_prog();

    var col = vec3f(.51, .95, .52);
    let gridpos = vec2i(round(frag / grid2px));
    if ((gridpos.x % 2 == 0) == (gridpos.y % 2 == 0))
    {
        col *= .9;
    }
    if (any(gridpos < vec2i(0)) || any(gridpos >= vec2i(DIM)))
    {
        col = vec3f();
    }

    // Render Apple
    let apple = state[BAPPLE].xy;
    if (all(apple >= vec2(0i)))
    {
        let rad = .34 * grid2px;
        let adist = sdCircle(frag - vec2f(apple) * grid2px, rad);
        col = blend(dist2col(adist, vec3f(.6, 0, 0)), col);
        col = blend(dist2col(adist + .1 * rad, vec3f(.87, 0.05, 0.0)), col);
    }

    // Render Body
    let head = state[BSNAKE].xy;
    let len  = vec2u(state[BLEN].xy);
    var prev = head;
    var hpos = vec2f(0);
    for (var i = 0u; i < len.x; i += 1)
    {
        let nxt = state[BSNAKE + i + 1];
        var fprev = vec2f(prev) * grid2px;
        var fnxt = vec2f(nxt.xy) * grid2px;

        // Eaten apple
        if (nxt.z != 0 && i != len.x - 1)
        {
            let eadist = sdCircle(frag - fnxt, grid2px * 0.44);
            col = blend(dist2col(eadist, SNAKE_COL), col);
        }

        // Segment
        if (i == 0)
        {
            fprev = mix(fnxt, fprev, prog);
            hpos = fprev;
        }
        else if (i == len.x - 1 && len.x == len.y)
        {
            fnxt = mix(fnxt, fprev, prog);
        }
        let dist = sdSegment(frag, fprev, fnxt) - 0.33 * grid2px;
        col = blend(dist2col(dist, SNAKE_COL), col);
        prev = nxt.xy;
    }

    // Render head
    let hdist = sdCircle(frag - hpos, grid2px * 0.47);
    col = blend(dist2col(hdist, SNAKE_COL), col);
    let forward = vec2f(state[BDIR].xy);
    let right = vec2f(forward.y, -forward.x);
    col = draw_eye(frag, hpos, forward, right, grid2px, col);
    col = draw_eye(frag, hpos, forward, -right, grid2px, col);

    // Death flash
    col = blend(vec4f(1, 0, 0, 1.0 - saturate(5 * (time.elapsed - bitcast<float>(state[BFLASH].x)))), col);

    col = pow(col, vec3f(2.2));
    textureStore(screen, id.xy, float4(col.rgb, 1.));
}
