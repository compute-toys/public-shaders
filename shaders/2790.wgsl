#storage statePing array<u32>
#storage statePong array<u32>

#define PING (time.frame % 2 == 0)


fn getCellState(position: vec2u) -> bool {
    let index = position.y * textureDimensions(screen).x + position.x;
    if (PING) {
        return statePing[index] != 0;
    }

    return statePong[index] != 0;
}

fn setCellState(position: vec2u, value: bool) {
    let index = position.y * textureDimensions(screen).x + position.x;
    if (PING) {
        statePong[index] = u32(value);
        return;
    }

    statePing[index] = u32(value);
}

fn isCellAlive(position: vec2u) -> bool {
    let simulationSize = textureDimensions(screen).xy;

    let hasBottom = u32(position.y > 0);
    let hasTop = u32(position.y < simulationSize.y - 1);
    let hasLeft = u32(position.x > 0);
    let hasRight = u32(position.x < simulationSize.x - 1);

    let hasTopLeft = hasTop & hasLeft;
    let hasTopRight = hasTop & hasRight;
    let hasBottomLeft = hasBottom & hasLeft;
    let hasBottomRight = hasBottom & hasRight;

    let bottomIndex = position - vec2u(0, hasBottom);
    let topIndex = position + vec2u(0, hasTop);
    let leftIndex = position - vec2u(hasLeft, 0);
    let rightIndex = position + vec2u(hasRight, 0);

    let bottomLeftIndex = position - vec2u(hasLeft, hasBottom);
    let bottomRightIndex = position - vec2u(0, hasBottom) + vec2u(hasRight, 0);
    let topLeftIndex = position + vec2u(0, hasTop) - vec2u(hasLeft, 0);
    let topRightIndex = position + vec2u(hasRight, hasTop);

    let numAliveNeighbors = u32(getCellState(bottomLeftIndex)) * hasBottomLeft +
                            u32(getCellState(bottomRightIndex)) * hasBottomRight +
                            u32(getCellState(topLeftIndex)) * hasTopLeft +
                            u32(getCellState(topRightIndex)) * hasTopRight +
                            u32(getCellState(bottomIndex)) * hasBottom +
                            u32(getCellState(topIndex)) * hasTop +
                            u32(getCellState(leftIndex)) * hasLeft +
                            u32(getCellState(rightIndex)) * hasRight;


    return numAliveNeighbors == 3 || (numAliveNeighbors == 2 && getCellState(position));
}

#dispatch_once initState
@compute @workgroup_size(16, 16)
fn initState(@builtin(global_invocation_id) id: vec3u) {
    let screenSize = textureDimensions(screen);

    if (any(id.xy >= screenSize)) {
        return;
    }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = vec2f(f32(id.x) + .5, f32(screenSize.y - id.y) - .5);

    // Normalised pixel coordinates (from 0 to 1)
    let uv = fragCoord / vec2f(screenSize);

    let index = id.y * screenSize.x + id.x;

    statePing[index] = u32(textureSampleLevel(channel0, bilinear, uv, 0).r > 0.5);
    
}


@compute @workgroup_size(16, 16)
fn updateState(@builtin(global_invocation_id) id: vec3u) {
    let screenSize = textureDimensions(screen);

    if (any(id.xy >= screenSize)) {
        return;
    }

    setCellState(id.xy, isCellAlive(id.xy));
}


@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    // Viewport resolution (in pixels)
    let screen_size = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = vec2f(f32(id.x) + .5, f32(screen_size.y - id.y) - .5);

    // Normalised pixel coordinates (from 0 to 1)
    let uv = fragCoord / vec2f(screen_size);

    // Time varying pixel colour
    var col = select(vec3(0.0), vec3(1.0), getCellState(id.xy));

    // Convert from gamma-encoded to linear colour space
    col = pow(col, vec3f(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, id.xy, vec4f(col, 1.));
}
