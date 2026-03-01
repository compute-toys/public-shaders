@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    // Viewport resolution (in pixels)
    let screen_size: vec2<u32> = textureDimensions(screen);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) {
        return;
    }

    let pos = vec3<f32>(
        f32(id.x),
        f32(id.y),
        time.elapsed * 10.
    );

    let voronoi = voronoi_3d_four_nearest(pos);

    var final_color = vec3<f32>(0.0, 0.0, 0.0);

    for (var i = 0; i < 4; i+=1) {
        let color = hash_vec3_to_color(voronoi.cells[i]);
        final_color.x += color.x * voronoi.weights[i];
        final_color.y += color.y * voronoi.weights[i];
        final_color.z += color.z * voronoi.weights[i];
    }

    // Output to screen (linear colour space)
    textureStore(
        screen,
        vec2<i32>(i32(id.x), i32(id.y)),
        vec4<f32>(final_color, 0.2)
    );
}

const cell_size = 40.;
const NEAREST_COUNT: u32 = 4u;
const SHARPNESS: f32 = 13.0;

struct Voronoi4 {
    distances: array<f32, 4>,
    cells: array<vec3<i32>, 4>,
    weights: array<f32, 4>
};

fn voronoi_3d_four_nearest(location: vec3<f32>) -> Voronoi4 {
    let baseCell = get_current_cell(location);

    var distances: array<f32, 4> = array<f32, 4>(3.402823466e+38, 3.402823466e+38, 3.402823466e+38, 3.402823466e+38);
    var cells = array<vec3<i32>, 4>(vec3<i32>(0), vec3<i32>(0), vec3<i32>(0), vec3<i32>(0));

    for (var offsetX = -1; offsetX <= 1; offsetX++) {
        for (var offsetY = -1; offsetY <= 1; offsetY++) {
            for (var offsetZ = -1; offsetZ <= 1; offsetZ++) {

                let candidateCell = baseCell + vec3<i32>(offsetX, offsetY, offsetZ);
                let candidatePoint = get_cell_point(candidateCell);
                let candidateDistance = distance(location, candidatePoint);

                // ---------- First nearest ----------
                let closerThanFirst = candidateDistance < distances[0];
                let previousFirstDistance = distances[0];
                let previousFirstCell = cells[0];

                distances[0] = select(distances[0], candidateDistance, closerThanFirst);
                cells[0] = select(cells[0], candidateCell, closerThanFirst);

                // ---------- Second nearest ----------
                let distanceForSecond =
                    select(candidateDistance, previousFirstDistance, closerThanFirst);
                let cellForSecond =
                    select(candidateCell, previousFirstCell, closerThanFirst);

                let closerThanSecond = distanceForSecond < distances[1];
                let previousSecondDistance = distances[1];
                let previousSecondCell = cells[1];

                distances[1] = select(distances[1], distanceForSecond, closerThanSecond);
                cells[1] = select(cells[1], cellForSecond, closerThanSecond);

                // ---------- Third nearest ----------
                let distanceForThird =
                    select(distanceForSecond, previousSecondDistance, closerThanSecond);
                let cellForThird =
                    select(cellForSecond, previousSecondCell, closerThanSecond);

                let closerThanThird = distanceForThird < distances[2];
                let previousThirdDistance = distances[2];
                let previousThirdCell = cells[2];

                distances[2] = select(distances[2], distanceForThird, closerThanThird);
                cells[2] = select(cells[2], cellForThird, closerThanThird);

                // ---------- Fourth nearest ----------
                let distanceForFourth =
                    select(distanceForThird, previousThirdDistance, closerThanThird);
                let cellForFourth =
                    select(cellForThird, previousThirdCell, closerThanThird);

                let closerThanFourth = distanceForFourth < distances[3];

                distances[3] = select(distances[3], distanceForFourth, closerThanFourth);
                cells[3] = select(cells[3], cellForFourth, closerThanFourth);
            }
        }
    }

    var sum_weights: f32 = 0.0;

    var weights: array<f32, 4> = array<f32, 4>(0.0, 0.0, 0.0, 0.0);

    if distances[0] < 1e-9 {
        weights[0] += 100.0;
    } else {
        weights[0] += 1.0 / pow(distances[0], SHARPNESS);
    }
    sum_weights += weights[0];

    if distances[1] < 1e-9 {
        weights[1] += 100.0;
    } else {
        weights[1] += 1.0 / pow(distances[1], SHARPNESS);
    }
    sum_weights += weights[1];

    if distances[2] < 1e-9 {
        weights[2] += 100.0;
    } else {
        weights[2] += 1.0 / pow(distances[2], SHARPNESS);
    }
    sum_weights += weights[2];

    if distances[3] < 1e-9 {
        weights[3] += 100.0;
    } else {
        weights[3] += 1.0 / pow(distances[3], SHARPNESS);
    }
    sum_weights += weights[3];

    for (var i = 0; i < 4; i+=1) {
        weights[i] /= sum_weights;
    }

    for (var i = 0; i < 4; i+=1) {
        distances[i] /= cell_size;
    }
    
    return Voronoi4(
        distances,

        cells,

        weights
    );
}

struct VoronoiReturn {
    distances: array<f32, NEAREST_COUNT>,
    cells: array<vec3<i32>, NEAREST_COUNT>,
    weights: array<f32, NEAREST_COUNT>
}

fn voronoi_3d(loc: vec3<f32>) -> VoronoiReturn {
    let base_cell = get_current_cell(loc);

    // --- inicjalizacja ---
    var distances: array<f32, NEAREST_COUNT>;
    var cells: array<vec3<i32>, NEAREST_COUNT>;

    for (var i = 0u; i < NEAREST_COUNT; i++) {
        distances[i] = 3.402823466e+38;
        cells[i] = vec3<i32>(0);
    }

    // --- przeszukiwanie sąsiednich komórek ---
    for (var x = -1; x <= 1; x += 1) {
        for (var y = -1; y <= 1; y += 1) {
            for (var z = -1; z <= 1; z += 1) {

                let cell = base_cell + vec3<i32>(x, y, z);
                let point = get_cell_point(cell);

                var dist = distance(loc, point);
                var cell_id = cell;

                // --- insertion sort (GPU-friendly) ---
                for (var i = 0u; i < NEAREST_COUNT; i++) {
                    let better = dist < distances[i];

                    let tmp_dist = distances[i];
                    let tmp_cell = cells[i];

                    distances[i] = select(distances[i], dist, better);
                    cells[i] = select(cells[i], cell_id, better);

                    dist = select(dist, tmp_dist, better);
                    cell_id = select(cell_id, tmp_cell, better);
                }
            }
        }
    }

    var sum_weights: f32 = 0.0;

    var weights: array<f32, NEAREST_COUNT>;

    // --- normalizacja odległości ---
    for (var i = 0u; i < NEAREST_COUNT; i++) {
        distances[i] /= cell_size;
        
         if distances[i] < 1e-9 {
            weights[i] += 100.0;
        } else {
            weights[i] += 1.0 / pow(distances[i], SHARPNESS);
        }
        sum_weights += weights[i];
    }

    for (var i = 0; i < 4; i+=1) {
        weights[i] /= sum_weights;
    }

    return VoronoiReturn(distances, cells, weights);
}

fn randomVec3(x: i32, y: i32, z: i32) -> vec3<f32> {
    let h = hash3D(x, y, z);

    return vec3<f32>(
        f32( h        & 0xFFu) / 255.0,
        f32((h >> 8u) & 0xFFu) / 255.0,
        f32((h >>16u) & 0xFFu) / 255.0
    );
}

fn hash3D(x: i32, y: i32, z: i32) -> u32 {
    var h: u32 =
          u32(x) * 374761393u
        + u32(y) * 668265263u
        + u32(z) * 2147483647u;

    h = (h ^ (h >> 13u)) * 1274126177u;
    h = h ^ (h >> 16u);

    return h;
}

fn get_current_cell(loc: vec3<f32>) -> vec3<i32> {
    let cell = floor(loc / cell_size);

    return vec3<i32>(cell);
}

fn get_cell_point(cell: vec3<i32>) -> vec3<f32> {
    var middle_point = vec3<f32>(cell) * cell_size + vec3<f32>(cell_size * 0.5);
    middle_point += randomVec3(cell.x, cell.y, cell.z) * 19.;

    return middle_point;
}

fn hash_vec3_to_color(v: vec3<i32>) -> vec3<f32> {
    var h: u32 =
          u32(v.x) * 374761393u
        + u32(v.y) * 668265263u
        + u32(v.z) * 2147483647u;

    h = (h ^ (h >> 13u)) * 1274126177u;
    h = h ^ (h >> 16u);

    return vec3<f32>(
        f32( h        & 0xFFu),
        f32((h >> 8u) & 0xFFu),
        f32((h >>16u) & 0xFFu)
    ) / 255.0;
}