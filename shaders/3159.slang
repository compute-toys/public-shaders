// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║              🌟 CelestialSim Web Demo — compute.toys port 🌟          ║
// ╚══════════════════════════════════════════════════════════════════════════════╝
//
// This is a demo for a procedural generation project being built in Godot.
// More details at: https://celestialsim.github.io/
//
// HOW TO USE:
//   1. Press 'I' / 'II' / 'III' style buttons on screen to switch showcases.
//   2. Mode I:  LOD system. Use the +/- buttons (top-right) to zoom.
//   3. Mode II: Click planet to add water; toggle the red button to remove water.
//   4. Mode III: Click to place an example tree; similar terrain spawns trees.
//
// CelestialSim Web Demo - Interactive Planet Renderer with:
// LOD System, simulations and example based procedural generation
// Copyright © 2025 Alessandro La Conca (CelestialSim Project)
// Licensed under MIT License - see LICENSE file for details
//
// ACKNOWLEDGMENTS:
// Ray intersection algorithms from https://iquilezles.org/articles/intersectors/
//   (MIT License © 2014 Inigo Quilez)
// Simplex noise from https://github.com/ashima/webgl-noise
//   (MIT License © 2011 Ashima Arts)

import std;


// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║ 🪐 Aliases                                                                   ║
// ╚══════════════════════════════════════════════════════════════════════════════╝
typealias vec2 = float2;
typealias vec3 = float3;
typealias vec4 = float4;
typealias mat2 = matrix<float,2,2>;

//WebGPU does not support some types like uint_8.
//Aliases are defined here to distinguish values of a more specific type in another backend.
typealias bool_w = uint;
typealias uint8_w = uint;

// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║ 🪐 Constants                                                                 ║
// ╚══════════════════════════════════════════════════════════════════════════════╝
static const uint initialNTris = 20;
static const uint initialNVerts = 12;
static const float minCamDist = -1.3;
static const float maxCamDist = 5;
static const float initialRotationSpeed = 1;
static const uint initalExampleTreeIdx = 2899;

// Trianlges and vertices for each level
//Level 0: nTris=20, nVerts=12
//Level 1: nTris=100, nVerts=72
//Level 2: nTris=420, nVerts=312
//Level 3: nTris=1700, nVerts=1272
//Level 4: nTris=6820, nVerts=5112
static const uint maxTris = 6820;
static const uint maxVerts = 5112;

static const uint maxDivLevel = 4;
// Constants
static const float3 initialCameraPos = vec3(0,0,3);
static const bool activateNoise = true; //Default: true
static const float max_tri_size = 0.26;  //Default: 0.3
static const float radius = 1;

// compute.toys runtime inputs (replace the [playground::TIME] / [playground::MOUSE_POSITION] uniforms)
static float timeFromStart = time.elapsed;

// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║ 🪐 Buffers                                                                   ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

// May be warped
[StorageBuffer(1)]
RWStructuredBuffer<float> rotationMs;
[StorageBuffer(1)]
RWStructuredBuffer<float> deltaTMs;

[StorageBuffer(maxTris)]
RWStructuredBuffer<Tris> t;

[StorageBuffer(maxTris)]
RWStructuredBuffer<Tree> trees;

[StorageBuffer(maxVerts)]
RWStructuredBuffer<float3> vPos;

[StorageBuffer(1)]
RWStructuredBuffer<State> state;

// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║ 🪐 Operators                                                                 ║
// ╚══════════════════════════════════════════════════════════════════════════════╝
bool operator==(TrisPtr v0, TrisPtr v1)
{
    return v0.idx == v1.idx;
}

bool operator!=(TrisPtr v0, TrisPtr v1)
{
    return v0.idx != v1.idx;
}

bool operator==(bool_w a, bool operand) {
    return a == (operand ? 1 : 0);
}

bool operator==(bool operand, bool_w a) {
    return a == (operand ? 1 : 0);
}

// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║ 🪐 Structs                                                                   ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

enum Edge
{
    ab,
    bc,
    ca
}

enum Subtris
{
    a,
    b,
    c
}

enum UI {
    LODShowcase,
    SimShowcase,
    GenShowcase,
    IncreaseZoom,
    DecreaseZoom,
    DeleteWater,
    AddWater
}

// App state deciding active showcase
struct State {
    uint nTris;
    uint nVerts;
    uint nTrisAdded;
    float camDistance;
    float rotationSpeed;
    //float rotationMs;
    UI activeShowcase;
    bool_w showLodAnimation;
    TrisPtr _pressedTris;
    bool_w deleteWaterToggle;
    float lastZoomTime; // Timestamp of last zoom operation
    uint exampleTreeIdx; // Index of the example tree in GenShowcase mode (0 means no example tree)
    uint2 screenSize;

    property bool mouseDown {
        get {
            // compute.toys: mouse.click is >0 while a button is held (left = 1, right = 3)
            return mouse.click > 0;
        }
    }
    property uint2 mousePos {
        get {
            // The original demo uses a bottom-left origin (y up), while compute.toys'
            // mouse.pos is top-left origin, so flip y to match the fragCoord below.
            return uint2(uint(mouse.pos.x), state[0].screenSize.y - uint(mouse.pos.y));
        }
    }

    // Needed Optionals can't be stored in a Buffer in webGPU
    property Optional<TrisPtr> pressedTris {
        get {
            if (_pressedTris.idx == -1)
                return none;
            return _pressedTris;
        }
        set {
            if (newValue.hasValue) {
                _pressedTris = newValue.value;
            } else {
                _pressedTris = TrisPtr(-1);
            }
        }
    }

    static float3 cameraPos() {
        // Animate Cam distance
        float t = (sin(timeFromStart/2) + 1) * 0.5;
        float animatedDist = lerp(minCamDist, maxCamDist, t);
        if (state[0].showLodAnimation == 1){
            state[0].camDistance = animatedDist;
        }
        return rotate(initialCameraPos+float3(0,0,state[0].camDistance),false);
    }
}

// This helps with performance since we are not passing the whole tris stuct but only a pointer
struct TrisPtr {
    int idx;

    property Tris tris {
        get {
            return t[idx];
        }
    }

    property TrisPtr neigh_ab {
        get {
            return t[idx].neigh_ab;
        }
    }

    property TrisPtr neigh_bc {
        get {
            return t[idx].neigh_bc;
        }
    }

    property TrisPtr neigh_ca {
        get {
            return t[idx].neigh_ca;
        }
    }
}


struct TreePtr {
    int idx = -1;

    property Tris tris {
        get {
            return t[idx];
        }
    }

    property Tree tree {
        get {
            return trees[idx];
        }
    }
}



struct Tris {
    TrisPtr ptr;
    uint3 abc;
    bool_w divided;
    bool_w to_div;
    uint8_w ico_idx;
    TrisPtr neigh_ab;
    TrisPtr neigh_bc;
    TrisPtr neigh_ca;
    TrisPtr a_t;
    TrisPtr b_t;
    TrisPtr c_t;
    TrisPtr center_t;
    uint8_w lv;
    SimValue sim;

    // This represents what side of each triangle faces the other
    // ab[1] = 0 means the edge ab of triangle 1 is facing the ab edge of another triangle
    static const uint[] ab = { 2, 0, 2, 1, 2, 2, 1, 2, 1, 0, 2, 2, 2, 1, 2, 1, 0, 2, 2, 0 };
    static const uint[] bc = { 1, 1, 2, 1, 1, 0, 0, 2, 1, 1, 0, 1, 1, 1, 2, 0, 2, 0, 2, 0 };
    static const uint[] ca = { 1, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1 };

    property bool isPressed {
        get {
            return state[0].pressedTris.hasValue && state[0].pressedTris.value == ptr;
        }
    }

    property vec3 normal {
        get {

            // Get the vertex positions using indices
            vec3 v0 = vPos[abc.x];
            vec3 v1 = vPos[abc.y];
            vec3 v2 = vPos[abc.z];

            // Calculate the normal using the cross product
            return - normalize(cross(v1 - v0, v2 - v0));
        }
    }

    property bool abNeightDivided {
        get {
            return neigh_ab.tris.divided == 1 || neigh_ab.tris.to_div == 1;
        }
    }
    property bool bcNeightDivided {
        get {
            return neigh_bc.tris.divided == 1 || neigh_bc.tris.to_div == 1;
        }
    }
    property bool caNeightDivided {
        get {
            return neigh_ca.tris.divided == 1 || neigh_ca.tris.to_div == 1;
        }
    }

    TrisPtr counterClockWiseNeigh(TrisPtr posOrigin, TrisPtr angleOrigin){
        if (posOrigin.neigh_ab == angleOrigin)
            return posOrigin.neigh_ca;
        else if (posOrigin.neigh_bc == angleOrigin)
            return posOrigin.neigh_ab;
        else return posOrigin.neigh_bc;
    }


    void getNeigths(out TrisPtr[13] neighs, TrisPtr centerNeigh) {
        neighs[0] = centerNeigh;
        // Add the starting neighbors at distance 1
        neighs[1] = neigh_ab;
        neighs[5] = neigh_bc;
        neighs[9] = neigh_ca;

        // Neighs at dist 2 and 3
        neighs[2] = counterClockWiseNeigh(neighs[1],centerNeigh);
        neighs[3] = counterClockWiseNeigh(neighs[2],neighs[1]);
        neighs[4] = counterClockWiseNeigh(neighs[3],neighs[2]);

        neighs[6] = counterClockWiseNeigh(neighs[5],centerNeigh);
        neighs[7] = counterClockWiseNeigh(neighs[6],neighs[5]);
        neighs[8] = counterClockWiseNeigh(neighs[7],neighs[6]);

        neighs[10] = counterClockWiseNeigh(neighs[9],centerNeigh);
        neighs[11] = counterClockWiseNeigh(neighs[10],neighs[9]);
        neighs[12] = counterClockWiseNeigh(neighs[11],neighs[10]);

    }

    property float3 center{
        get {
        var center = (vPos[abc.x] + vPos[abc.y] + vPos[abc.z])/3;
        return center;
        }
    }

    // From https://iquilezles.org/articles/intersectors/
    Optional<TrisAndDist> intersect(vec3 ro, vec3 rd)
    {
        vec3 v0 = vPos[abc.x];
        vec3 v1 = vPos[abc.y];
        vec3 v2 = vPos[abc.z];

        vec3 v1v0 = v1 - v0;
        vec3 v2v0 = v2 - v0;
        vec3 rov0 = ro - v0;
        vec3  n = cross( v1v0, v2v0 );
        vec3  q = cross( rov0, rd );
        float d = 1.0/dot( rd, n );
        float u = d*dot( -q, v2v0 );
        float v = d*dot(  q, v1v0 );
        float t = d*dot( -n, rov0 );
        if( u<0.0 || v<0.0 || (u+v)>1.0 ) return none;
        return TrisAndDist(ptr,t);
    }

    void updateSimValues() {
        // Set colors
        t[ptr.idx].sim.color = getColor();

        if (sim.isInitialized == true)
            return;

        bool hasDetailedLod = lv >= maxDivLevel;
        bool isValley = distance(center,float3(0)) < radius*1;

        // Set triangle as water if it's in a valley (closer to center) and meets noise criteria
        if (hasDetailedLod && isValley) {
            t[ptr.idx].sim.hasWater = true; // Mark as water
        }

        trees[ptr.idx] = Tree(ptr,0);
        t[ptr.idx].sim.treePtr = TreePtr(ptr.idx);
        t[ptr.idx].sim.isInitialized = true;
    }

    float3 getColor() {

        if (lv < maxDivLevel) {
            var color = (a_t.tris.sim.color + b_t.tris.sim.color + c_t.tris.sim.color + center_t.tris.sim.color) / 4;
            if (!float3(0).equals(color))
                return color;
        }

        if (sim.hasWater == true){
            return isPressed ?
                float3(0.3, 0.5, 1.0):  // Brighter blue when pressed
                float3(0.2, 0.4, 0.8);  // Normal water blue

        }

        if (isPressed)
            return float3(1, 0.2, 0.4);

        // Calculate height as distance from center (0,0,0) normalized by radius
        float height = distance(center, float3(0)) / radius;

        // Colors based on height instead of LOD level
        if (height < 1.01) {
            // Lower areas (valleys) - darker green
            return vec3(0.4, 0.61, 0.42)*0.8;
        } else if (height < 1.04) {
            // Middle areas - regular green
            return vec3(0.4, 0.61, 0.42);
        } else if (height < 1.07) {
            // Higher areas - lighter green
            return vec3(0.5, 0.7, 0.5);
        } else {
            // Peaks - brown/tan for mountain tops
            return vec3(0.6, 0.52, 0.36);
        }
    }

    // Similarity to another tris based on height, humidity, and terrain type
    float similarity(TrisPtr other) {
        // Calculate similarity based on height
        float h1 = distance(center, float3(0));
        float h2 = distance(other.tris.center, float3(0));
        float heightDiff = 150*abs(h1 - h2);
        float heightSim = 1.0 / (heightDiff + 1e-5); // avoid div by zero
        heightSim = clamp(heightSim, 0.0, 1.0);

        // Calculate similarity based on humidity
        float hum1 = sim.humidity;
        float hum2 = other.tris.sim.humidity;
        float humidityDiff = 8 * abs(hum1 - hum2);
        float humiditySim = 1.0 - humidityDiff; // 1.0 is perfect match, 0.0 is maximum difference
        humiditySim = clamp(humiditySim, 0.0, 1.0);

        // Calculate similarity based on the unified terrainFactor
        float factor1 = sim.terrainFactor;
        float factor2 = other.tris.sim.terrainFactor;

        // abs(factor1 - factor2) ranges from 0 (identical) to 2 (opposite extremes).
        // Scale this difference to a 0-1 range for terrainTypeDiff.
        float terrainTypeDiff = abs(factor1 - factor2) / 2.0f;
        float terrainTypeSim = 1.0f - terrainTypeDiff;

        // Combine all similarities with weights
        const float heightWeight = 0.4;      // Weight for height similarity
        const float humidityWeight = 0.3;    // Weight for humidity similarity
        const float terrainTypeWeight = 0.3; // Weight for terrain type similarity

        float combinedSim = (heightWeight * heightSim) +
                        (humidityWeight * humiditySim) +
                        (terrainTypeWeight * terrainTypeSim);

        return combinedSim;
    }
}

struct Tree {
    TrisPtr ptr;
    float3 pa;
    float3 pb;
    float ra;
    float rb;
    float scale;

    property bool isExampleTree {
        get {
            return ptr.idx == state[0].exampleTreeIdx;
        }
    }

    __init(TrisPtr tr, float _scale){
        ptr = tr;
        scale = _scale;
        var trisAbc = ptr.tris.abc;
        float3 vertA = vPos[trisAbc.x];
        float3 vertB = vPos[trisAbc.y];
        float3 vertC = vPos[trisAbc.z];

        // Generate two random values using the hash function
        float r1 = hash(vertA);
        float r2 = hash(vertB);

        // Convert random values to barycentric coordinates for uniform sampling
        float u = 1.0 - sqrt(r1);
        float v = sqrt(r1) * (1.0 - r2);
        float w = sqrt(r1) * r2;

        // Calculate the position of the tree based on barycentric coordinates
        pa = u * vertA + v * vertB + w * vertC;
        //pa = centerOfTris(tr);
        pb = pa + normalize(pa) * 0.1 * scale;
        ra = 0.03 * scale;
        rb = 0.01 * scale;
    }

    __init(TrisPtr tr){
        var trisAbc = tr.tris.abc;
        float3 vertC = vPos[trisAbc.z];

        float r3 = hash(vertC);
        float scale = (r3/4)+0.3;

        return Tree(tr, scale);
    }

    property float3 treeColor {
        get {
            float3 baseColor = isExampleTree ? float3(1.0, 0.2, 0.2) : float3(0.4, 0.61, 0.42);
            baseColor = baseColor * 0.8; // Darken the base color for trees
            // Add slight variation based on position hash
            float variation = 0.3 * (hash(pa) - 0.5); // range ~[-0.035, 0.035]
            return baseColor + float3(variation, 0, 0);
        }
    }

    // From https://iquilezles.org/articles/intersectors/
    Optional<TrisAndDist> intersect(in vec3 ro, in vec3 rd)
    {
        vec3  ba = pb - pa;
        vec3  oa = ro - pa;
        vec3  ob = ro - pb;
        float rr = ra - rb;
        float m0 = dot(ba,ba);
        float m1 = dot(ba,oa);
        float m2 = dot(ba,rd);
        float m3 = dot(rd,oa);
        float m5 = dot(oa,oa);
        float m6 = dot(ob,rd);
        float m7 = dot(ob,ob);

        // body
        float d2 = m0-rr*rr;
        float k2 = d2    - m2*m2;
        float k1 = d2*m3 - m1*m2 + m2*rr*ra;
        float k0 = d2*m5 - m1*m1 + m1*rr*ra*2.0 - m0*ra*ra;
        float h = k1*k1 - k0*k2;
        if( h<0.0) return none;
        float t = (-sqrt(h)-k1)/k2;
    //if( t<0.0 ) return vec4(-1.0);
        float y = m1 - ra*rr + t*m2;
        if( y>0.0 && y<d2 ) return TrisAndDist(ptr,t,normalize(d2*(oa+t*rd)-ba*y),Object.Tree);
        //if( y>0.0 && y<d2 ) return vec4(t, normalize(d2*(oa+t*rd)-ba*y));


        // caps
        float h1 = m3*m3 - m5 + ra*ra;
        float h2 = m6*m6 - m7 + rb*rb;
        if( max(h1,h2)<0.0 ) return none;
        vec4 r = vec4(1e20);
        if( h1>0.0 )
        {
            t = -m3 - sqrt( h1 );
            r = vec4( t, (oa+t*rd)/ra );
        }
        if( h2>0.0 )
        {
            t = -m6 - sqrt( h2 );
            if( t<r.x )
            r = vec4( t, (ob+t*rd)/rb );
        }
        return TrisAndDist(ptr,r.x,r.yzw,Object.Tree);
    }
}

struct Atmosphere{
    float3 origin;
    float radius;

    // From https://iquilezles.org/articles/intersectors/
    Optional<TrisAndDist> intersectSphere(vec3 ro,vec3 rd)
    {
        vec3 oc = ro - origin;
        float b = dot( oc, rd );
        float c = dot( oc, oc ) - radius*radius;
        float h = b*b - c;
        if( h<0.0 ) return none;
        var dist = -b - sqrt( h );
        return TrisAndDist(TrisPtr(0),dist,sphNormal(ro + dist*rd),Object.Atmosphere);
    }

    // Modified from https://iquilezles.org/articles/intersectors/
    Optional<TrisAndDist> intersect(vec3 ro, vec3 rd) {
        vec3 oc = ro - origin; // Vector from the ray origin to the sphere center
        oc = oc*0.95; //Show the circle a bit closer to the camera

        float b = dot(oc, rd); // Dot product of oc and ray direction
        if (b == 0.0) {
            // The ray origin lies on the plane perpendicular to rd through the sphere center
            return none;
        }

        float dist = -b / dot(rd, rd); // Solve for distance to intersection

        if (dist < 0.0) {
            // Intersection is behind the ray origin
            return none;
        }

        return TrisAndDist(
            TrisPtr(0),
            dist,
            sphNormal(ro + dist * rd),
            Object.Atmosphere
        );
    }


    vec3 sphNormal(vec3 hitPos)
    {
        return normalize(hitPos-origin);
    }

}

struct SimValue {
    float3 color;
    TreePtr treePtr;
    bool_w isInitialized;
    bool_w hasWater;

    property bool hasTree {
        get {
            return treePtr.tree.scale > 0.0f;
        }
    }

    property float humidity {
        get {
            // Get neighbors and calculate humidity based on water presence
            float humidityValue = 0.0f;
            TrisPtr ptr = TrisPtr(treePtr.idx);
            TrisPtr[13] neighs;
            ptr.tris.getNeigths(neighs, ptr);

            // Sum up contributions from water tiles in neighborhood
            for (var i = 0; i < neighs.getCount(); i++) {
                if (neighs[i].tris.sim.hasWater)
                    humidityValue += 1.0f/13.0f;
            }

            return humidityValue;
        }
    }

    // Terrain is a peak 1 or a valley -1
    property float terrainFactor {
        get {
            TrisPtr ptr = TrisPtr(treePtr.idx);

            // Only calculate for detailed triangles
            if (ptr.tris.lv < maxDivLevel) {
                return 0.0f; // Neutral for non-detailed or if treePtr is not set for this SimValue context
            }

            float currentHeight = distance(ptr.tris.center, float3(0));

            // Calculate the center of the three direct neighbors
            float3 center_of_neighbors = (ptr.neigh_ab.tris.center + ptr.neigh_bc.tris.center + ptr.neigh_ca.tris.center) / 3.0f;
            float dist_neighbor_center = distance(center_of_neighbors, float3(0));

            // Calculate scaled difference: positive for peak-like, negative for valley-like
            float height_difference = currentHeight - dist_neighbor_center;
            float scaled_difference = height_difference * 400; // Using the 0.05 scaling factor

            // Clamp to the range [-1.0, 1.0]
            return clamp(scaled_difference, -1.0f, 1.0f);
        }
    }
}

struct Button {
    uint2 bottomLeft;
    uint size;
    UI id;

    bool intersect(uint2 pos) {
        if (hidden)
            return false;
        int radius = 7;
        if (pos.x >= bottomLeft.x + radius && pos.x <= bottomLeft.x + size - radius &&
            pos.y >= bottomLeft.y && pos.y <= bottomLeft.y + size) {
            return true;
        }

        if (pos.y >= bottomLeft.y + radius && pos.y <= bottomLeft.y + size - radius &&
            pos.x >= bottomLeft.x && pos.x <= bottomLeft.x + size) {
            return true;
        }

        int2 cornerCenters[4] = {
            {bottomLeft.x + radius, bottomLeft.y + radius},
            {bottomLeft.x + size - radius, bottomLeft.y + radius},
            {bottomLeft.x + radius, bottomLeft.y + size - radius},
            {bottomLeft.x + size - radius, bottomLeft.y + size - radius}
        };

        for (int i = 0; i < 4; i++) {
            int dx = pos.x - cornerCenters[i].x;
            int dy = pos.y - cornerCenters[i].y;
            if (dx * dx + dy * dy <= radius * radius) {
                return true;
            }
        }
        return false;
    }

    property bool hidden {
        get {
            switch (state[0].activeShowcase) {
            case UI.SimShowcase:
                return id == UI.DecreaseZoom || id == UI.IncreaseZoom;
            default:
                return id == UI.DeleteWater || id == UI.AddWater;
            }
        }
    }

    bool drawRectangle(uint2 fragCoord, uint2 offset, uint2 rectSize) {
        return fragCoord.x >= bottomLeft.x + offset.x
            && fragCoord.y >= bottomLeft.y + offset.y
            && fragCoord.x <= bottomLeft.x + offset.x + rectSize.x
            && fragCoord.y <= bottomLeft.y + offset.y + rectSize.y;
    }

    bool drawPlus(uint2 fragCoord) {
        const uint padding = 8;
        const uint thickness = 7;
        // Horizontal bar
        if (drawRectangle(fragCoord, uint2(padding, size/2 - thickness/2), uint2(size-2*padding, thickness))) return true;
        // Vertical bar
        if (drawRectangle(fragCoord, uint2(size/2 - thickness/2, padding), uint2(thickness, size-2*padding))) return true;
        return false;
    }

    bool drawX(uint2 fragCoord) {
        const uint padding = 6;
        const uint thickness = 9;

        // Convert to local coordinates
        float2 local = float2(fragCoord - bottomLeft);
        local -= float2(size/2); // Center at origin

        // Rotate coordinates by +45 and -45 degrees
        float angle = 0.785398; // 45 degrees in radians
        float2x2 rotPos = float2x2(
            cos(angle), -sin(angle),
            sin(angle), cos(angle)
        );
        float2x2 rotNeg = float2x2(
            cos(-angle), -sin(-angle),
            sin(-angle), cos(-angle)
        );

        float2 rot1 = mul(rotPos, local);
        float2 rot2 = mul(rotNeg, local);

        // Check if point is within either rotated rectangle
        float halfLength = (size - 2*padding) / 2;
        bool inRect1 = abs(rot1.x) < halfLength && abs(rot1.y) < thickness/2;
        bool inRect2 = abs(rot2.x) < halfLength && abs(rot2.y) < thickness/2;

        return inRect1 || inRect2;
    }

    bool drawMinus(uint2 fragCoord) {
        const uint padding = 8;
        const uint thickness = 7;
        // Horizontal bar only
        return drawRectangle(fragCoord, uint2(padding, size/2 - thickness/2), uint2(size-2*padding, thickness));
    }

    bool drawRomanI(uint2 fragCoord) {
        const uint padding = 12;
        const uint thickness = 7;
        return drawRectangle(fragCoord, uint2(size/2 - thickness/2, padding), uint2(thickness, size-2*padding));
    }

    bool drawRomanII(uint2 fragCoord) {
        const uint padding = 12;
        const uint thickness = 7;
        const uint spacing = 10;
        if (drawRectangle(fragCoord, uint2(size/2 - spacing - thickness/2, padding), uint2(thickness, size-2*padding))) return true;
        if (drawRectangle(fragCoord, uint2(size/2 + spacing - thickness/2, padding), uint2(thickness, size-2*padding))) return true;
        return false;
    }

    bool drawRomanIII(uint2 fragCoord) {
        const uint padding = 12;
        const uint thickness = 7;
        const uint spacing = 10;
        if (drawRectangle(fragCoord, uint2(size/2 - spacing*1.5 - thickness/2, padding), uint2(thickness, size-2*padding))) return true;
        if (drawRectangle(fragCoord, uint2(size/2 - thickness/2, padding), uint2(thickness, size-2*padding))) return true;
        if (drawRectangle(fragCoord, uint2(size/2 + spacing*1.5 - thickness/2, padding), uint2(thickness, size-2*padding))) return true;
        return false;
    }

    void onPress() {
        // Current time in milliseconds
        float currentTime = timeFromStart * 1000.0;

        // Minimum time between zoom operations (in milliseconds)
        const float zoomThrottleTime = 16.0; // ~60fps rate

        switch (id) {
            case UI.IncreaseZoom:
                state[0].showLodAnimation = false;

                // Only perform zoom if enough time has passed since last zoom
                if (currentTime - state[0].lastZoomTime >= zoomThrottleTime) {
                    // Use a fixed zoom amount that's independent of triangle count
                    // Base speed is 0.1 units per second
                    float zoomAmount = 1.5 * (zoomThrottleTime / 1000.0);
                    state[0].camDistance = max(state[0].camDistance - zoomAmount, minCamDist);

                    // Update last zoom time
                    state[0].lastZoomTime = currentTime;
                }
                break;
            case UI.DecreaseZoom:
                state[0].showLodAnimation = false;

                // Only perform zoom if enough time has passed since last zoom
                if (currentTime - state[0].lastZoomTime >= zoomThrottleTime) {
                    // Use a fixed zoom amount that's independent of triangle count
                    // Base speed is 0.1 units per second
                    float zoomAmount = 1.5 * (zoomThrottleTime / 1000.0);
                    state[0].camDistance = min(state[0].camDistance + zoomAmount, maxCamDist);

                    // Update last zoom time
                    state[0].lastZoomTime = currentTime;
                }
                break;
            case UI.LODShowcase:
                state[0].activeShowcase = UI.LODShowcase;
                state[0].rotationSpeed = initialRotationSpeed;
                state[0].showLodAnimation = true;
                break;
            case UI.SimShowcase:
                state[0].activeShowcase = UI.SimShowcase;
                state[0].rotationSpeed = initialRotationSpeed * 0.1;
                state[0].showLodAnimation = false;
                state[0].camDistance = -1.3;
                break;
            case UI.GenShowcase:
                state[0].activeShowcase = UI.GenShowcase;
                state[0].camDistance = -1;
                state[0].rotationSpeed = initialRotationSpeed * 0.1;
                state[0].showLodAnimation = false;
                break;
            case UI.AddWater:
                state[0].deleteWaterToggle = false;
                break;
            case UI.DeleteWater:
                state[0].deleteWaterToggle = true;
                break;
        }
    }

    float3 color(uint2 fragCoord, uint2 pos, bool mouseDown) {
        bool active = (id == UI.IncreaseZoom | id == UI.DecreaseZoom | id == state[0].activeShowcase);
        active = id == UI.AddWater ? state[0].deleteWaterToggle == false : active;
        active = id == UI.DeleteWater ? state[0].deleteWaterToggle == true : active;

        var baseColor = float3(0.9,0.9,1);
        var inactiveColor = float3(0.55,0.55,0.65);

        if (id == UI.AddWater){
            baseColor = float3(0.6,0.7,1);
            inactiveColor = float3(0.35,0.4,0.6);
        }
        if (id == UI.DeleteWater){
            baseColor = float3(1,0.6,0.7);
            inactiveColor = float3(0.5,0.3,0.4);
        }

        float3 color = active  ? baseColor : inactiveColor;

        bool isBeingPressed = intersect(pos) && mouseDown;
        float3 finalColor = isBeingPressed ? baseColor * float3(1.2,1.2,1.6) : color;

        bool isBlack = false;

        switch(id) {
            case UI.AddWater:
            case UI.IncreaseZoom:
                isBlack = drawPlus(fragCoord);
                break;
            case UI.DecreaseZoom:
                isBlack = drawMinus(fragCoord);
                break;
            case UI.LODShowcase:
                isBlack = drawRomanI(fragCoord);
                break;
            case UI.SimShowcase:
                isBlack = drawRomanII(fragCoord);
                break;
            case UI.GenShowcase:
                isBlack = drawRomanIII(fragCoord);
                break;
            case UI.DeleteWater:
                isBlack = drawX(fragCoord);
                break;
        }

        return isBlack ? float3(0,0,0) : finalColor;
    }
};

// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║ 🪐 Functions                                                                 ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

float getDeltaT(){
    return deltaTMs[0] / 1000;
}

float hash(vec3 p3)
{
    p3  = fract(p3 * .1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}

// Adds 3 vertices in the middle of the edges of the triangle
// defined by the vertices abc.
// divTris is the index of the triangle (0 to nTrisAdded)
// returns the index of the added vertices
int3 add_middle_vertices(TrisPtr parent, int3 abc, TrisPtr divTris) {
    // add vertices
    float3 ab_mid = (vPos[abc.x] + vPos[abc.y]) / 2;
    float3 bc_mid = (vPos[abc.y] + vPos[abc.z]) / 2;
    float3 ca_mid = (vPos[abc.z] + vPos[abc.x]) / 2;
    uint v_idx_start = state[0].nVerts + 3 * divTris.idx;
    int3 v_idx = int3(v_idx_start, v_idx_start + 1, v_idx_start + 2);

    vPos[v_idx.x] = ab_mid;
    vPos[v_idx.y] = bc_mid;
    vPos[v_idx.z] = ca_mid;

    // update vertex position
    vPos[v_idx.x] = applyNoise(vPos[v_idx.x]);
    vPos[v_idx.y] = applyNoise(vPos[v_idx.y]);
    vPos[v_idx.z] = applyNoise(vPos[v_idx.z]);

    return v_idx;
}

TrisPtr chooseNeight(Edge edge, TrisPtr parent, Subtris subtris) {
    var divided = false;
    TrisPtr neight_idx;

    if(edge == Edge.ab) {
        divided = parent.tris.abNeightDivided;
        neight_idx = parent.neigh_ab;
    } else if(edge == Edge.bc) {
        divided = parent.tris.bcNeightDivided;
        neight_idx = parent.neigh_bc;
    } else {
        divided = parent.tris.caNeightDivided;
        neight_idx = parent.neigh_ca;
    }

    if (divided) {
        var neighOrientationOffset = 0;
        if (parent.tris.ico_idx != neight_idx.tris.ico_idx) {
            if(edge == Edge.ab) {
                neighOrientationOffset = Tris.ab[parent.tris.ico_idx];
            } else if(edge == Edge.bc) {
                neighOrientationOffset = Tris.bc[parent.tris.ico_idx];
            } else {
                neighOrientationOffset = Tris.ca[parent.tris.ico_idx];
            }
        }

        if (subtris == Subtris.a) {
            neighOrientationOffset += 0;
        } else if (subtris == Subtris.b) {
            neighOrientationOffset += 1;
        } else {
            neighOrientationOffset += 2;
        }

        neighOrientationOffset = neighOrientationOffset % 3;
        switch (neighOrientationOffset) {
            case 1:
                return neight_idx.tris.b_t;
            case 2:
                return neight_idx.tris.c_t;
            default:
                return neight_idx.tris.a_t;
        }
    }
    return neight_idx;
}

Optional<TrisPtr> getClickedTris(vec3 ro, vec3 rd){
     if (let hit = icoIntersect(ro, rd)) {
        switch (hit.object ) {
            case Object.Tris:
            case Object.Tree:
                return hit.ptr;
            default:
                return none;
        }
    }
    return none;
}

void handlePlanetClick(vec3 ro, vec3 rd, bool mouseDown) {
    // Handle GenShowcase mode clicks
    if (state[0].activeShowcase == UI.GenShowcase) {
        if (mouseDown) {
            if (let trisPtr = getClickedTris(ro, rd)) {
                TrisPtr prt = trisPtr;
                if (prt.tris.lv < maxDivLevel) {
                    return; // Only place trees on detailed triangles
                }
                // If there was a previous example tree, reset it
                uint oldIdx = state[0].exampleTreeIdx;
                if (oldIdx != prt.idx)
                    trees[oldIdx] = Tree(TrisPtr(oldIdx), 0); // Scale 0 effectively removes it

                state[0].exampleTreeIdx = prt.idx;
                createExampleTree();
            }
        }
        return;
    }

    // Standard SimShowcase handling
    if (state[0].activeShowcase != UI.SimShowcase)
        return;

    if (let trisPtr = getClickedTris(ro, rd)) {
        TrisPtr prt = trisPtr;
        if (prt.tris.lv < maxDivLevel) {
            return;
        }
        // If the same triangle as before is being pressed
        if (state[0].pressedTris.hasValue && state[0].pressedTris.value != trisPtr){
            // Move away
            state[0].pressedTris = none;
        }
        if (mouseDown) {
            if (state[0].pressedTris.hasValue)
                return; // Only pressed once
            // On click down
            t[trisPtr.idx].sim.hasWater = !state[0].deleteWaterToggle;
            state[0].pressedTris = trisPtr;
        }
        else { // On click up
            state[0].pressedTris = none;
        }
    }
}

uint4 add_triangles(TrisPtr parent, int3 abc, int3 v_idx, TrisPtr divTris) {
    uint t_idx_start = state[0].nTris + 4 * divTris.idx;
    uint4 t_idx = uint4(t_idx_start, t_idx_start + 1, t_idx_start + 2, t_idx_start + 3);

    // Add triangles
    t[t_idx.x].abc = int3(abc.x, v_idx.x, v_idx.z);
    t[t_idx.y].abc = int3(v_idx.x, abc.y, v_idx.y);
    t[t_idx.z].abc = int3(v_idx.z, v_idx.y, abc.z);
    t[t_idx.w].abc = int3(v_idx.y, v_idx.z, v_idx.x);

    // Add indexes
    t[t_idx.x].ptr = TrisPtr(t_idx.x);
    t[t_idx.y].ptr = TrisPtr(t_idx.y);
    t[t_idx.z].ptr = TrisPtr(t_idx.z);
    t[t_idx.w].ptr = TrisPtr(t_idx.w);

    // Set ico_idx
    t[t_idx.x].ico_idx = parent.tris.ico_idx;
    t[t_idx.y].ico_idx = parent.tris.ico_idx;
    t[t_idx.z].ico_idx = parent.tris.ico_idx;
    t[t_idx.w].ico_idx = parent.tris.ico_idx;

    // Set children
    t[parent.idx].a_t = TrisPtr(t_idx.x);
    t[parent.idx].b_t = TrisPtr(t_idx.y);
    t[parent.idx].c_t = TrisPtr(t_idx.z);
    t[parent.idx].center_t = TrisPtr(t_idx.w);

    // Update neighbors for center triangle
    t[t_idx.w].neigh_ab = parent.tris.c_t;
    t[t_idx.w].neigh_bc = parent.tris.a_t;
    t[t_idx.w].neigh_ca = parent.tris.b_t;

    // Update neighbors for a triangle
    t[t_idx.x].neigh_ab = chooseNeight(Edge.ab, TrisPtr(parent), Subtris.b);
    t[t_idx.x].neigh_bc = parent.tris.center_t;
    t[t_idx.x].neigh_ca = chooseNeight(Edge.ca, TrisPtr(parent), Subtris.c);

    // Update neighbors for b triangle
    t[t_idx.y].neigh_ab = chooseNeight(Edge.ab, TrisPtr(parent), Subtris.a);
    t[t_idx.y].neigh_bc = chooseNeight(Edge.bc, TrisPtr(parent), Subtris.c);
    t[t_idx.y].neigh_ca = parent.tris.center_t;

    // Update neighbors for c triangle
    t[t_idx.z].neigh_ab = parent.tris.center_t;
    t[t_idx.z].neigh_bc = chooseNeight(Edge.bc, TrisPtr(parent), Subtris.b);
    t[t_idx.z].neigh_ca = chooseNeight(Edge.ca, TrisPtr(parent), Subtris.a);

    return t_idx;
}


float3 rotate(float3 initialRotation, bool inverted)
{
    // rotate tris
    float an = 0.2*(timeFromStart * state[0].rotationSpeed);
    an = inverted ? - an : an;
    float3x3 rotationMatrix = float3x3(
        cos(an),  0, sin(an),
        0,        1, 0,
        -sin(an), 0, cos(an)
    );

    //float3 cpos = float3(cameraPos[0],cameraPos[1],cameraPos[2]);
    // Apply rotationMs
    float3 cpos = mul(rotationMatrix, initialRotation);
    return cpos;
}



float3 applyNoise(float3 pos) {
    pos = normalize(pos);
    if (!activateNoise)
        return pos;
    var noiseVal = snoise(pos) + 0.2;
    float amplitude = 0.13; // Adjust amplitude for noise effect
    float height = noiseVal * amplitude;
    if (height < 0.0) {
        height = 0.0; // Ensure no negative heights
    }
    return pos + pos * height; // Scale sin for more variation
}


// Noise functions from Ashima webgl-noise library
// https://github.com/ashima/webgl-noise
vec3 mod289(vec3 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 mod289(vec4 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 permute(vec4 x) {
    return mod289(((x*34.0)+10.0)*x);
}

vec4 taylorInvSqrt(vec4 r)
{
    return 1.79284291400159 - 0.85373472095314 * r;
}

// From https://github.com/ashima/webgl-noise/blob/master/src/noise3D.glsl
float snoise(vec3 v) {
    const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
    const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

    // First corner
    vec3 i  = floor(v + dot(v, C.yyy) );
    vec3 x0 =   v - i + dot(i, C.xxx) ;

    // Other corners
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min( g.xyz, l.zxy );
    vec3 i2 = max( g.xyz, l.zxy );

    //   x0 = x0 - 0.0 + 0.0 * C.xxx;
    //   x1 = x0 - i1  + 1.0 * C.xxx;
    //   x2 = x0 - i2  + 2.0 * C.xxx;
    //   x3 = x0 - 1.0 + 3.0 * C.xxx;
    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
    vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

    // Permutations
    i = mod289(i);
    vec4 p = permute( permute( permute(
                i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
            + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
            + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

    // Gradients: 7x7 points over a square, mapped onto an octahedron.
    // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
    float n_ = 0.142857142857; // 1.0/7.0
    vec3  ns = n_ * D.wyz - D.xzx;

    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);

    vec4 b0 = vec4( x.xy, y.xy );
    vec4 b1 = vec4( x.zw, y.zw );

    //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
    //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));

    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

    vec3 p0 = vec3(a0.xy,h.x);
    vec3 p1 = vec3(a0.zw,h.y);
    vec3 p2 = vec3(a1.xy,h.z);
    vec3 p3 = vec3(a1.zw,h.w);

    //Normalise gradients
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    // Mix final noise value
    vec4 m = max(0.5 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 105.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1),
                                dot(p2,x2), dot(p3,x3) ) );
}

enum Object{
    Tris,
    Tree,
    Atmosphere
}

struct TrisAndDist {
    TrisPtr ptr;
    float dist;
    Object object;
    float3 _hitNormal;

    __init(TrisPtr tptr, float tdist){
        ptr = tptr;
        dist = tdist;
        object = Object.Tris;
    }

    __init(TrisPtr tptr, float tdist, float3 hitNormal, Object hitObject) {
        ptr = tptr;
        dist = tdist;
        _hitNormal = hitNormal;
        object = hitObject;
    }

    float3 hitPos(float3 ro, float3 rd){
        return ro + dist * rd;
    }

    property Tris tris {
        get {
            return ptr.tris;
        }
    }

    property float3 hitNormal {
        get {
            if (object == Object.Tris) {
                return tris.normal;
            } else return _hitNormal;
        }
    }
}

bool trisShouldBeDivided(TrisAndDist trisAndDist){
    // If the tris is small, don't divide it

    if (trisAndDist.tris.lv > maxDivLevel)
        return false;

    // Approximate triangle sceen size
    var screen_size = radius / (
            pow(1.2, trisAndDist.tris.lv) * trisAndDist.dist
        );
    //screen_size = dot(calcNormal(trisAndDist.tris),float3(0,0,1));
    //screen_size = smoothstep(-1,1,screen_size);
    // todo: remove invisible tris by checking its direction
    return screen_size > max_tri_size;
}

TrisAndDist nearestSubtris(TrisAndDist trisAndDist, vec3 ro, vec3 rd) {
    if (trisAndDist.tris.divided == 0)
        return trisAndDist;

    // Initialize with first subtriangle
    var tris = trisAndDist.tris;
    //projecting on triangle surface, to be more precise we can use sphere or continious noise
    var pos = trisAndDist.hitPos(ro,rd);
    var trisA = tris.a_t;
    var minTrisAndDist = TrisAndDist(trisA,distance(trisA.tris.center,pos));

    // Check triangle B
    var trisB = tris.b_t;
    var bTrisAndDist = TrisAndDist(trisB,distance(trisB.tris.center,pos));

    // Check triangle C
    var trisC = tris.c_t;
    var cTrisAndDist = TrisAndDist(trisC,distance(trisC.tris.center,pos));

    // Check triangle Center
    var trisCenter = tris.center_t;
    var centerTrisAndDist = TrisAndDist(trisCenter,distance(trisCenter.tris.center,pos));

    if (bTrisAndDist.dist < minTrisAndDist.dist){
        minTrisAndDist = bTrisAndDist;
    }
    if (cTrisAndDist.dist < minTrisAndDist.dist){
        minTrisAndDist = cTrisAndDist;
    }
    if (centerTrisAndDist.dist < minTrisAndDist.dist){
        minTrisAndDist = centerTrisAndDist;
    }

    return minTrisAndDist;
}

Optional<TrisAndDist> refineNearestFromNeigh(TrisAndDist minTris, vec3 ro, vec3 rd) {

    Optional<TrisAndDist> minTrisAndDist = minTris.ptr.tris.intersect(ro,rd);
    // Calculate distance from neight to fix holes

    TrisPtr[13] neighs;
    minTris.ptr.tris.getNeigths(neighs, minTris.ptr);

    for (var i = 0; i < neighs.getCount(); i++) {
        Optional<TrisAndDist> trisAndDist = neighs[i].tris.intersect(ro,rd);
        if (trisAndDist.hasValue){
            if (!minTrisAndDist.hasValue || minTrisAndDist.hasValue && trisAndDist.value.dist < minTrisAndDist.value.dist)
                minTrisAndDist = trisAndDist.value;
        }

        //
        if (neighs[i].tris.sim.hasTree) {
            trisAndDist = neighs[i].tris.sim.treePtr.tree.intersect(ro,rd);
            if (trisAndDist.hasValue){
                if (!minTrisAndDist.hasValue || minTrisAndDist.hasValue && trisAndDist.value.dist < minTrisAndDist.value.dist)
                    minTrisAndDist = trisAndDist.value;
            }
        }
    }

    return minTrisAndDist;
}

Optional<TrisAndDist> icoIntersect(float3 ro, float3 rd)
{

    Atmosphere atm = Atmosphere(float3(0,0,0),radius*1.25);
    var atmIntersect = atm.intersect(ro,rd);
    var atmIntersectSphere = atm.intersectSphere(ro,rd);
    if (!atmIntersectSphere.hasValue)
        return none;

    var atmHitPos = atmIntersect.value.hitPos(ro,rd);

    Optional<TrisAndDist> minTris = none;
    TrisAndDist minAtmoTris = TrisAndDist(TrisPtr(0),distance(TrisPtr(0).tris.center,atmHitPos));

    // Find best starting tris using distance
    //[ForceUnroll]
    const float3 cameraPos = normalize(State.cameraPos());
    for (int i = 20; i < 100; i++) {
        bool forwardFacing = dot(t[i].normal,cameraPos) > 0;
        if (forwardFacing){
            var tris = TrisPtr(i).tris.intersect(ro,rd);
            if (tris.hasValue){
                if (!minTris.hasValue || minTris.hasValue && tris.value.dist < minTris.value.dist)
                    minTris = tris;
            }
            var trisPtr = TrisPtr(i);
            var tris1 = TrisAndDist(trisPtr,distance(trisPtr.tris.center,atmHitPos));
            if (tris1.dist < minAtmoTris.dist)
                minAtmoTris = tris1;
        }
    }

    if (!minTris.hasValue)
        minTris = minAtmoTris;

    if (minTris.hasValue){
        var closestSubtris = nearestSubtris(minTris.value,ro,rd);
        if (let val = refineNearestFromNeigh(closestSubtris,ro,rd))
            minTris = val;
    }

    if (minTris.hasValue){
        var closestSubtris = nearestSubtris(minTris.value,ro,rd);
        if (let val = refineNearestFromNeigh(closestSubtris,ro,rd))
            minTris = val;
    }

    if (minTris.hasValue){
        var closestSubtris = nearestSubtris(minTris.value,ro,rd);
        if (let val = refineNearestFromNeigh(closestSubtris,ro,rd))
            minTris = val;
    }


    if (minTris.hasValue){
        var closestSubtris = nearestSubtris(minTris.value,ro,rd);
        minTris = refineNearestFromNeigh(closestSubtris,ro,rd);
    }

    var minTris1 = refineNearestFromNeigh(minTris.value,ro,rd);
    minTris = minTris1;
    if (minTris.hasValue && dot(minTris.value.tris.normal,cameraPos) < 0){
            minTris = none;
    }

    if (!minTris.hasValue)
        minTris = atmIntersectSphere;

    return minTris;

}

TrisAndDist trisAndDistFromCenter(TrisPtr tr){
    var dist = distance(tr.tris.center,State.cameraPos());
    return TrisAndDist(tr,dist);
}

void makeTris(uint2 dispatchThreadId : SV_DispatchThreadID)
{
    uint idx = dispatchThreadId.x;

    //todo: remove the idx >= nTris check after confirming issue origin
    if (t[idx].to_div == 0 || idx >= state[0].nTris) return;

    //index starting from zero and increasing for each divided tris
    // 0 to nTrisAdded
    TrisPtr divTris = TrisPtr(idx-(state[0].nTris-state[0].nTrisAdded));
    //uint divTris = idx;
    int3 abc = t[idx].abc;

    var v_idx = add_middle_vertices(TrisPtr(idx),abc, divTris);
    // add triangles
    var t_idx = add_triangles(TrisPtr(idx), abc, v_idx, divTris);
    // set divided flag
    t[idx].divided = true;

    // set t_lv
    t[t_idx.x].lv = t[idx].lv + 1;
    t[t_idx.y].lv = t[idx].lv + 1;
    t[t_idx.z].lv = t[idx].lv + 1;
    t[t_idx.w].lv = t[idx].lv + 1;


    t[t_idx.x].to_div = trisShouldBeDivided(trisAndDistFromCenter(TrisPtr(t_idx.x)));
    t[t_idx.y].to_div = trisShouldBeDivided(trisAndDistFromCenter(TrisPtr(t_idx.y)));
    t[t_idx.z].to_div = trisShouldBeDivided(trisAndDistFromCenter(TrisPtr(t_idx.z)));
    t[t_idx.w].to_div = trisShouldBeDivided(trisAndDistFromCenter(TrisPtr(t_idx.w)));

    // Fix holes between different lod levels
    if (!t[idx].abNeightDivided) {
        vPos[v_idx.x] = (vPos[abc.x]+vPos[abc.y])/2;
    }
    if (!t[idx].bcNeightDivided) {
        vPos[v_idx.y] = (vPos[abc.y]+vPos[abc.z])/2;
    }
    if (!t[idx].caNeightDivided) {
        vPos[v_idx.z] = (vPos[abc.z]+vPos[abc.x])/2;
    }

    // # Set simulation values
    //calculate colors
    TrisPtr(t_idx.x).tris.updateSimValues();
    TrisPtr(t_idx.y).tris.updateSimValues();
    TrisPtr(t_idx.z).tris.updateSimValues();
    TrisPtr(t_idx.w).tris.updateSimValues();

    // set to_div flag
    // Doing this at the end solves flickering
    // there is some concurrency issue
    t[idx].to_div = false;

}

void increaseTris(uint2 dispatchThreadId : SV_DispatchThreadID){
    if (dispatchThreadId.x > 0)
        return;
    if (state[0].nTris == initialNTris)
    {
        state[0].nTrisAdded = initialNTris;
    }
    state[0].nTris += state[0].nTrisAdded*4;
    state[0].nVerts += state[0].nTrisAdded*3;
    state[0].nTrisAdded = state[0].nTrisAdded*4;
}

void createExampleTree() {
    // Create a large example tree
    TreePtr exampleTreePtr =  TreePtr(state[0].exampleTreeIdx);
    trees[exampleTreePtr.idx] = Tree(TrisPtr(exampleTreePtr.idx), 1);
    t[exampleTreePtr.idx].sim.treePtr = exampleTreePtr;

    // Mark it as the example tree
    state[0].exampleTreeIdx = exampleTreePtr.idx;
}


// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║ 🪐 Compute Shaders                                                           ║
// ╚══════════════════════════════════════════════════════════════════════════════╝
//
// Entry points run in source order every frame. The icosphere LOD is rebuilt
// from scratch each frame, so initVerts / initTris / makeTrisN run every frame;
// only `init` is dispatched once.


// Fills buffer core vertices
[WorkgroupCount(1, 1, 1)]
[shader("compute")]
[numthreads(12, 1, 1)]
void initVerts(uint2 dispatchThreadId : SV_DispatchThreadID)
{
    uint idx = dispatchThreadId.x;
    if (idx >= initialNVerts)
        return;
    float3 coreVPos[12] = {
        float3(-0.5257f, 0.0000f, 0.8507f),
        float3( 0.5257f, 0.0000f, 0.8507f),
        float3(-0.5257f, 0.0000f, -0.8507f),
        float3( 0.5257f, 0.0000f, -0.8507f),
        float3( 0.0000f, 0.8507f, 0.5257f),
        float3( 0.0000f, 0.8507f, -0.5257f),
        float3( 0.0000f, -0.8507f, 0.5257f),
        float3( 0.0000f, -0.8507f, -0.5257f),
        float3( 0.8507f, 0.5257f, 0.0000f),
        float3(-0.8507f, 0.5257f, 0.0000f),
        float3( 0.8507f, -0.5257f, 0.0000f),
        float3(-0.8507f, -0.5257f, 0.0000f)
    };
    vPos[idx] = coreVPos[idx];

    vPos[idx] = applyNoise(vPos[idx]);

    state[0].nVerts = initialNVerts;
    //cameraPos[0] = initalCameraPos;
}


[Cover("t")]
[shader("compute")]
[numthreads(20, 1, 1)]
void initTris(uint2 dispatchThreadId : SV_DispatchThreadID)
{
    uint idx = dispatchThreadId.x;
    if (idx >= initialNTris) {
        // Reset triangle
        t[idx].divided = false;
        return;
    }
    int3 coreTAbc[20] = {
        int3( 0,  4,  1),
        int3( 0,  9,  4),
        int3( 9,  5,  4),
        int3( 4,  5,  8),
        int3( 4,  8,  1),
        int3( 8, 10,  1),
        int3( 8,  3, 10),
        int3( 5,  3,  8),
        int3( 5,  2,  3),
        int3( 2,  7,  3),
        int3( 7, 10,  3),
        int3( 7,  6, 10),
        int3( 7, 11,  6),
        int3(11,  0,  6),
        int3( 0,  1,  6),
        int3( 6,  1, 10),
        int3( 9,  0, 11),
        int3( 9, 11,  2),
        int3( 9,  2,  5),
        int3( 7,  2, 11)
    };

    TrisPtr[] tNeightAb = { 1, 16, 18, 2, 3, 6, 7, 8, 18, 19, 11, 12, 19, 16, 0, 14, 1, 16, 17, 9 };
    TrisPtr[] tNeightBc = { 4, 2, 3, 7, 5, 15, 10, 6, 9, 10, 6, 15, 13, 14, 15, 5, 13, 19, 8, 17 };
    TrisPtr[] tNeightCa = { 14, 0, 1, 4, 0, 4, 5, 3, 7, 8, 9, 10, 11, 12, 13, 11, 17, 18, 2, 12 };

    t[idx].ptr = TrisPtr(idx);
    t[idx].abc = coreTAbc[idx];
    // Add an index for icosphere face
    t[idx].ico_idx = idx;
    // Mark tris to divide
    t[idx].to_div = trisShouldBeDivided(trisAndDistFromCenter(TrisPtr(idx)));;

    // Init neighbours
    t[idx].neigh_ab = tNeightAb[idx];
    t[idx].neigh_bc = tNeightBc[idx];
    t[idx].neigh_ca = tNeightCa[idx];

    // Init sim values
    // TrisPtr(idx).tris.updateSimValues();

    state[0].nTris = initialNTris;
    state[0].nTrisAdded = 20;
}

[WorkgroupCount(1, 1, 1)]
[shader("compute")]
[numthreads(1, 1, 1)]
void init(uint2 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x > 0) return;
    // compute.toys' [DispatchOnce] did not reliably apply these writes at startup,
    // so we self-initialize exactly once. Storage starts zeroed and rotationSpeed
    // is never 0 again once set (mode II sets it to 0.1), so it works as a sentinel.
    if (state[0].rotationSpeed != 0.0) return;
    state[0].showLodAnimation = true;
    state[0].rotationSpeed = initialRotationSpeed;
    rotationMs[0] = timeFromStart * 1000;
    state[0].lastZoomTime = 0.0; // Initialize lastZoomTime
    state[0].exampleTreeIdx = initalExampleTreeIdx; // Initialize example tree index
}

//Lv 1
[Cover("t")]
[shader("compute")]
[numthreads(10, 1, 1)]
void makeTris1(uint2 dispatchThreadId : SV_DispatchThreadID)
{
    makeTris(dispatchThreadId);
}

//Lv 2
[WorkgroupCount(1, 1, 1)]
[shader("compute")]
[numthreads(1, 1, 1)]
void increaseTris2(uint2 dispatchThreadId : SV_DispatchThreadID)
{
    increaseTris(dispatchThreadId);
}
[Cover("t")]
[shader("compute")]
[numthreads(10, 1, 1)]
void makeTris2(uint2 dispatchThreadId : SV_DispatchThreadID)
{
    makeTris(dispatchThreadId);
}

//Lv 3
[WorkgroupCount(1, 1, 1)]
[shader("compute")]
[numthreads(1, 1, 1)]
void increaseTris3(uint2 dispatchThreadId : SV_DispatchThreadID)
{
    increaseTris(dispatchThreadId);
}
[Cover("t")]
[shader("compute")]
[numthreads(10, 1, 1)]
void makeTris3(uint2 dispatchThreadId : SV_DispatchThreadID)
{
    makeTris(dispatchThreadId);
}

//Lv 4
[WorkgroupCount(1, 1, 1)]
[shader("compute")]
[numthreads(1, 1, 1)]
void increaseTris4(uint2 dispatchThreadId : SV_DispatchThreadID)
{
    increaseTris(dispatchThreadId);
}
[Cover("t")]
[shader("compute")]
[numthreads(10, 1, 1)]
void makeTris4(uint2 dispatchThreadId : SV_DispatchThreadID)
{
    makeTris(dispatchThreadId);
}

[Cover("t")]
[shader("compute")]
[numthreads(32, 1, 1)]
void treeSimStep(uint2 dispatchThreadId : SV_DispatchThreadID)
{
    uint idx = dispatchThreadId.x;
    TrisPtr ptr = TrisPtr(idx);
    if (idx == state[0].exampleTreeIdx)
        createExampleTree();

    if (idx == state[0].exampleTreeIdx || ptr.tris.lv < maxDivLevel) return;
    TrisPtr examplePtr = TrisPtr(state[0].exampleTreeIdx);
    float low_cutoff_threshold = 0.1;
    float sim = ptr.tris.similarity(examplePtr);
    // Scale trees smoothly based on similarity (0 = no tree, 1 = full size)
    float targetScale = low_cutoff_threshold > sim ? 0.0 : sim;
    if (ptr.tris.sim.hasWater)
        targetScale = 0.0;
    // float targetScale = (sim > threshold) ? 1.0 : 0.0;
    float prevScale = trees[idx].scale;
    float newScale = lerp(prevScale, targetScale, getDeltaT());
    trees[idx] = Tree(ptr, newScale);
}

[WorkgroupCount(1, 1, 1)]
[shader("compute")]
[numthreads(1, 1, 1)]
void updateTime(uint2 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x > 0) return;
    deltaTMs[0] = (timeFromStart * 1000) -  rotationMs[0];
    rotationMs[0] = timeFromStart * 1000;
}

[WorkgroupCount(1, 1, 1)]
[shader("compute")]
[numthreads(1, 1, 1)]
void mouseClick(uint2 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x > 0) return;

    if (state[0].mouseDown) {
        let mouseUI = UIintersect(state[0].mousePos,state[0].screenSize);
        if (mouseUI.hasValue) {
            mouseUI.value.onPress();
            return;
        }
    }
    vec3 ro = State.cameraPos();
    vec2 iResolution = state[0].screenSize;
    vec2 p1 = (2.0*state[0].mousePos.xy-iResolution.xy) / iResolution.y;
    vec3 rd1 = normalize(vec3(p1,-2.0));
    rd1 = rotate(rd1,false);
    handlePlanetClick(ro, rd1, state[0].mouseDown);
}


Optional<Button> UIintersect(uint2 pos, int2 screenSize)
{
    const uint size = 55;
    const uint padding = 8;
    var addWater = Button(uint2(screenSize.x-2*size-2*padding,padding),size,UI.AddWater);
    var deleteWater = Button(uint2(screenSize.x-size-padding,padding),size,UI.DeleteWater);
    var increaseZoom = Button(uint2(screenSize.x-2*size-2*padding,padding),size,UI.DecreaseZoom);
    var decreaseZoom = Button(uint2(screenSize.x-size-padding,padding),size,UI.IncreaseZoom);
    // Models buttons
    var LODShowcase = Button(uint2(padding,screenSize.y-size-50),size,UI.LODShowcase);
    var SimShowcase = Button(uint2(padding,screenSize.y-2*size-58),size,UI.SimShowcase);
    var GenShowcase = Button(uint2(padding,screenSize.y-3*size-66),size,UI.GenShowcase);

    Button[] allButtons = {LODShowcase, SimShowcase, GenShowcase, addWater, deleteWater , increaseZoom, decreaseZoom};

    for (int i; i< allButtons.getCount();i++){
        var button = allButtons[i];
        if (button.intersect(pos))
            return button;
    }

    return none;
}

// ╔══════════════════════════════════════════════════════════════════════════════╗
// ║ 🪐 Main                                                                      ║
// ╚══════════════════════════════════════════════════════════════════════════════╝

//https://www.shadertoy.com/view/4d2XWV
// Default full-screen dispatch (covers `screen`), like the multi.slang example.
[shader("compute")]
[numthreads(16, 16, 1)]
void imageMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint width = 0;
    uint height = 0;
    screen.GetDimensions(width, height);

    if (dispatchThreadID.x >= width || dispatchThreadID.y >= height)
        return;

    int2 screenSize = int2(int(width), int(height));
    state[0].screenSize = uint2(width, height);
    // Bottom-left origin (y up) to match the original demo's coordinate convention.
    uint2 fragCoord = uint2(dispatchThreadID.x, height - 1u - dispatchThreadID.y);

    // UI overlay
    if (let uiElem = UIintersect(fragCoord, screenSize)) {
        vec3 uiCol = uiElem.color(fragCoord, state[0].mousePos, state[0].mouseDown);
        screen[dispatchThreadID.xy] = vec4(pow(uiCol, vec3(2.2)), 1);
        return;
    }

    vec2 iResolution = screenSize;
    vec2 p = (2.0*vec2(fragCoord)-iResolution.xy) / iResolution.y;

    vec3 ro = State.cameraPos();
    vec3 rd = normalize( vec3(p,-2.0) );
    rd = rotate(rd,false);

    // sphere animation
    vec3 lig = rotate(normalize( vec3(0.6,0.3,0.4) ),false);
    vec3 col = vec3(0.0);

    Optional<TrisAndDist> triAndDO = icoIntersect( ro, rd );
    if(let triAndD = triAndDO)
    {
        float t1 = triAndD.dist;
        vec3 pos = ro + t1*rd;

        float lightIntensity = clamp( (dot(triAndD.hitNormal,lig)+1)/2, 0.3, 1.0 );
        switch (triAndD.object){
            case Object.Tris:
                col = triAndD.tris.sim.color;

                // Add a subtle shimmer effect to water
                if (triAndD.tris.sim.hasWater) {
                    // Add time-based shimmer using sine waves
                    float shimmer = 0.1 * sin(timeFromStart * 3.0 + dot(pos, float3(1.0, 2.0, 3.0)) * 5.0);
                    col += float3(0.05, 0.1, 0.15) * shimmer;

                    // Add a subtle specular highlight
                    float3 halfVec = normalize(lig + normalize(ro - pos));
                    float spec = pow(max(0.0, dot(triAndD.hitNormal, halfVec)), 32.0) * 0.5;
                    col += float3(0.3, 0.4, 0.5) * spec;
                }
                break;
            case Object.Tree:
                // Get the tree color
                col = triAndD.ptr.tris.sim.treePtr.tree.treeColor;
                // Tree are in shadow if the light is behind them
                // Smoothly mix between shadow and sunny intensity
                float shadowLightIntensity = clamp( (dot(triAndD.tris.normal,lig)+1)/2, 0.3, 1.0 );
                float sunnyLightIntensity = lightIntensity;

                float dotProduct = dot(triAndD.tris.normal, lig);
                float mixFactor = smoothstep(-0.2, 0.1, dotProduct);
                lightIntensity = lerp(shadowLightIntensity, sunnyLightIntensity, mixFactor);
                break;
            case Object.Atmosphere:
                col = vec3(0.1,0.1,0.2)*3;
            default:
        }
        col = col * lightIntensity;
    }

    // compute.toys treats `screen` as linear and applies sRGB at present time, so
    // encode the demo's gamma/sRGB-authored colour to linear before writing.
    screen[dispatchThreadID.xy] = vec4( pow(col, vec3(2.2)), 1.0 );
}
