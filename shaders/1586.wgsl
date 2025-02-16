#define CELLS_X 48
#define CELLS_Y 27
const startEmpty = true;
const numPlayers = 6u;
const cells = vec2i(CELLS_X, CELLS_Y);

struct player_t
{
    pos: vec2f,
    vel: vec2f,
    color: vec3f
}

#storage board array<array<u32, CELLS_X>, CELLS_Y>
#storage player array<player_t, numPlayers>

fn fmod(a: vec3f, b: vec3f) -> vec3f
{
    return modf(a / b).fract * b;
}

fn fmodf(a: vec3f, b: f32) -> vec3f
{
    return fmod(a, vec3f(b));
}

fn hsv2rgbSmooth(c: vec3f) -> vec3f
{
    var rgb = abs(fmodf(c.x * 6. + vec3f(0., 4., 2.), 6.) - 3.) - 1.;
	rgb = smoothstep(vec3f(0.), vec3f(1.), rgb);
	return c.z * mix(vec3f(1.), rgb, c.y);
}

fn pcg3d(vin: vec3u) -> vec3u
{
    var v = vin * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> vec3u(16u);
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return v;
}

fn pcg3df(vin: vec3u) -> vec3f
{
    return vec3f(pcg3d(vin)) / f32(0xffffffffu);
}

#workgroup_count init 1 CELLS_Y 1
@compute @workgroup_size(CELLS_X)
fn init(@builtin(global_invocation_id) id: vec3u)
{
    if (time.frame == 0)
    {
        var owner = id.x * numPlayers / u32(cells.x);
        if (startEmpty) {owner = 0xffffffffu;}
        board[id.y][id.x] = owner;
        if (id.y == 0 && id.x < numPlayers)
        {
            player[id.x].pos = pcg3df(vec3u(id.xy, u32(custom.randomSeed))).xy;
            let xcpp = f32(cells.x) / f32(numPlayers);
            player[id.x].pos.x = (f32(id.x) + .5) * xcpp;
            player[id.x].pos.y = player[id.x].pos.y * f32(cells.y - 1) + .5;
            player[id.x].vel = normalize(pcg3df(vec3u(id.xy, u32(custom.randomSeed + 1))).xy * 2. - 1.) * .01;
            //player[id.x].vel = (pcg3df(vec3u(id.xy, u32(custom.randomSeed + 1))).xy * 2. - 1.) * .01;
        }
    }
}

fn edgeBounce(p: u32)
{
    let r = .5;
    let fcells = vec2f(cells);
    if (player[p].pos.x + r > fcells.x)
    {
        player[p].vel.x *= -1.;
        player[p].pos.x -= 2. * (player[p].pos.x + r - fcells.x);
    }
    if (player[p].pos.x - r < 0.)
    {
        player[p].vel.x *= -1.;
        player[p].pos.x -= 2. * (player[p].pos.x - r);
    }
    if (player[p].pos.y + r > fcells.y)
    {
        player[p].vel.y *= -1.;
        player[p].pos.y -= 2. * (player[p].pos.y + r - fcells.y);
    }
    if (player[p].pos.y - r < 0.)
    {
        player[p].vel.y *= -1.;
        player[p].pos.y -= 2. * (player[p].pos.y - r);
    }
}

fn collide(p: u32)
{
    let r = .5;
    let r1 = 1. -r;
    var delta = vec2i(1, 1);
    var ipos = vec2i(player[p].pos);
    board[ipos.y][ipos.x] = p;
    let flipx = (player[p].vel.x < 0);
    if (flipx) {
        delta.x *= -1;
        player[p].pos.x *= -1.;
        player[p].vel.x *= -1.;
    }
    let flipy = (player[p].vel.y < 0);
    if (flipy) {
        delta.y *= -1;
        player[p].pos.y *= -1.;
        player[p].vel.y *= -1.;
    }
    if (fract(player[p].pos.y) > r1 && board[ipos.y + delta.y][ipos.x] != p)
    {
        board[ipos.y + delta.y][ipos.x] = p;
        player[p].pos.y -= 2. * fract(player[p].pos.y + r);
        player[p].vel.y *= -1.;
    }
    if (fract (player[p].pos.x) > r1 && board[ipos.y][ipos.x + delta.x] != p)
    {
        board[ipos.y][ipos.x + delta.x] = p;
        player[p].pos.x -= 2. * fract(player[p].pos.x + r);
        player[p].vel.x *= -1.;
    }
    if (fract (player[p].pos.x) > r1 && fract(player[p].pos.y) > r1 && board[ipos.y + delta.y][ipos.x + delta.x] != p)
    {
        let d = length(vec2f(1.) - fract(player[p].pos));
        if (d < r)
        {
            board[ipos.y + delta.y][ipos.x + delta.x] = p;
            let normal = normalize(fract(player[p].pos) - vec2f(1.));
            player[p].vel = player[p].vel - 2. * dot(player[p].vel, normal) * normal;
        }
    }
    if (fract (player[p].pos.x) < r && fract(player[p].pos.y) > r1 && board[ipos.y + delta.y][ipos.x - delta.x] != p)
    {
        let d = length(vec2f(0., 1.) - fract(player[p].pos));
        if (d < r)
        {
            board[ipos.y + delta.y][ipos.x - delta.x] = p;
            let normal = normalize(fract(player[p].pos) - vec2f(0., 1.));
            player[p].vel = player[p].vel - 2. * dot(player[p].vel, normal) * normal;
        }
    }
    if (fract (player[p].pos.x) > r1 && fract(player[p].pos.y) < r && board[ipos.y - delta.y][ipos.x + delta.x] != p)
    {
        let d = length(vec2f(1., 0.) - fract(player[p].pos));
        if (d < r)
        {
            board[ipos.y - delta.y][ipos.x + delta.x] = p;
            let normal = normalize(fract(player[p].pos) - vec2f(1., 0.));
            player[p].vel = player[p].vel - 2. * dot(player[p].vel, normal) * normal;
        }
    }
    if (flipx) {
        player[p].pos.x *= -1.;
        player[p].vel.x *= -1.;
    }
    if (flipy) {
        player[p].pos.y *= -1.;
        player[p].vel.y *= -1.;
    }
}

#workgroup_count simulate 1 1 1
@compute @workgroup_size(numPlayers)
fn simulate(@builtin(global_invocation_id) id: vec3u)
{
    let fcells = vec2f(cells);
    let p = id.x;
    for(var i = 0u; i < u32(custom.speed); i++)
    {
        player[p].pos += player[p].vel;
        edgeBounce(p);
        collide(p);
    }
}

fn playerColor(player: u32) -> vec3f{
    if (player >= numPlayers) {return vec3f(.5);}
    var waves = numPlayers / 8u + 1;
    var hue = f32(player % waves) * (1. / f32(waves)) + f32(player / waves) / f32(numPlayers);
    return hsv2rgbSmooth(vec3f(custom.hueOffset + hue, .6, .8));
}

@compute @workgroup_size(16, 16)
fn draw(@builtin(global_invocation_id) id: vec3u)
{
    let screenSize = vec2i(textureDimensions(screen));
    let screenSizeAR = vec2i((screenSize.y * cells.x / cells.y), (screenSize.x * cells.y / cells.x));
    let screenSizeM = min(screenSize, screenSizeAR);
    let off = (screenSize - screenSizeM) / 2;
    let mid = vec2i(id.xy) - off;
    if (all(mid < screenSizeM) && all(mid >= vec2i(0)))
    {
        let c = mid * vec2i(cells) / screenSizeM;
        let cb = (mid - vec2i(1)) * vec2i(cells) / screenSizeM;
        let cn = (mid + vec2i(1)) * vec2i(cells) / screenSizeM;
        var col = playerColor(board[c.y][c.x]);

        if (any(cb != c) || any(cn != c)) { col *= .8;}
        
        for (var i = 0u; i < numPlayers; i++)
        {
            let pcol = playerColor(i);
            let d = length(vec2f(mid) + vec2f(.5) - player[i].pos * vec2f(screenSizeM) / vec2f(cells));
            let r = f32(screenSizeM.x) / f32(cells.x) / 2 - .5;
            let m = smoothstep(-.1, .1, d - r);
            col = mix(pcol * .8, col, m);
            let m2 = smoothstep(-.5, .5, abs(d - r)-1.0);
            col = mix(pcol * .4, col, m2);
        }

        col = pow(col, vec3f(2.2));
        textureStore(screen, id.xy, vec4f(col, 1.));
    }
}
