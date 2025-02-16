/* https://shader-slang.org/slang-playground/

/* 
 * 2D Gaussian Splatting Example in Slang
 *
 * This example demonstrates the use of Slang's differentiable programming capabilities to implement 
 * a 2D Gaussian splatting algorithm that can be trained within the browser using the Slang Playground.
 * 
 * This algorithm represents a simplified version of the 3D Gaussian Splatting algorithm detailed in 
 * this paper (https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). 
 * This 2D demonstration does not have the 3D->2D projection step & assumes that the Gaussian blobs 
 * are presented in order of depth (higher index = farther away). Further, this implementation does 
 * not perform adaptive density control to add or remove blobs.
 * 
 * See the `computeDerivativesMain()` kernel and the `splatBlobs()` function for the bulk of the key
 * pieces of the code. 
 *
 * Key Slang features used in this example include the autodiff operator `bwd_diff(fn)`, the 
 * `[Differentiable]` attribute, and custom derivatives for a few specific components via 
 * the `[BackwardDerivative(fn)]` attribute.
 * 
 * For a full 3D Gaussian Splatting implementation written in Slang, see this repository: 
 * https://github.com/google/slang-gaussian-rasterization
 *
 */

import playground;

// ----- Constants and definitions --------

static const int BLOB_BUFFER_SIZE = 184320;
static const int NUM_FIELDS = 9;

static const int GAUSSIANS_PER_BLOCK = 256;
static const int WG_X = 16;
static const int WG_Y = 16;

static const float ADAM_ETA = 0.002;
static const float ADAM_BETA_1 = 0.9;
static const float ADAM_BETA_2 = 0.999;
static const float ADAM_EPSILON = 1e-8;

// ------ Global buffers and textures --------

// The "//!" directives instruct slang-playground to allocate and initialize the buffers 
// with the appropriate data. 
//
// When using this sample code locally, your own engine is responsible for allocating, 
// initializing & binding these buffers.
//

[playground::RAND(BLOB_BUFFER_SIZE)]
RWStructuredBuffer<float> blobsBuffer;

[playground::ZEROS(BLOB_BUFFER_SIZE)]
RWStructuredBuffer<Atomic<uint>> derivBuffer;

[playground::ZEROS(BLOB_BUFFER_SIZE)]
RWStructuredBuffer<float> adamFirstMoment;

[playground::ZEROS(BLOB_BUFFER_SIZE)]
RWStructuredBuffer<float> adamSecondMoment;

[playground::URL("static/jeep.jpg")]
Texture2D<float4> targetTexture;

// ----- Shared memory declarations --------

// Note: In Slang, the 'groupshared' identifier is used to define
// workgroup-level shared memory. This is equivalent to '__shared__' in CUDA

// blobCountAT is used when storing blob IDs into the blobs buffer. It needs to be atomic 
// since multiple threads will be in contention to increment it.
//
// Atomic<T> is the most portable way to express atomic operations. Slang supports basic 
// operations like +, -, ++, etc.. on Atomic<T> types.
// 
groupshared Atomic<uint> blobCountAT;

// This is used after the coarse rasterization step as a non-atomic 
// location to store the blob count, since atomics are not necessary after the coarse
// rasterization step.
//
groupshared uint blobCount;

// The blobs buffer is used to store the indices of the blobs that intersect 
// with the current tile.
//
groupshared uint blobs[GAUSSIANS_PER_BLOCK];

// The maxCount and finalVal buffers are used to store the final PixelState objects
// after the forward pass. This data is read-back for the backwards pass.
// 
groupshared uint maxCount[WG_X * WG_Y];
groupshared float4 finalVal[WG_X * WG_Y];

// The reductionBuffer is used for the binary reduction in the loadFloat_bwd() function.
groupshared float reductionBuffer[WG_X * WG_Y];

// -----------------------------------------

// Some types to hold state info on the 'blobs' buffer.
// This makes it easy to make sure we're not accidentally using the buffer
// in the wrong state.
//
// The actual data is in the 'blobs' object.
//
struct InitializedShortList { int _dummy = 0; };
struct FilledShortList { int _dummy = 0; };
struct PaddedShortList { int _dummy = 0; };
struct SortedShortList { int _dummy = 0; };

/* 
* Oriented bounding box (OBB) data-structure
*
* Can be used to represent the bounds of an anisotropic Gaussian blob.
* The bounding box can be extracted by taking a canonical box 
* formed by (-1,-1), (1,-1), (1,1), (-1,1), then translating, rotating, and scaling it.
*/
struct OBB
{
    float2 center;
    float2x2 rotation;
    float2 scale;

    /* 
    * intersects() returns true if the OBB intersects with another OBB.
    * 
    * The implementation is based on the separating axis theorem (see 
    * https://dyn4j.org/2010/01/sat/#sat-algo for a detailed explanation). 
    * At a high level, the SAT algorithm checks if the projections of the 
    * points of the two OBBs are disjoint along the normals of all of the 
    * faces of each OBB.
    */
    bool intersects(OBB other)
    {
        float2 canonicalPts[4] = float2[4](float2(-1, -1), float2(1, -1), float2(1, 1), float2(-1, 1));

        float2x2 invRotation = inverse(rotation);
        float2x2 otherInvRotation = inverse(other.rotation);
        float2 pts[4];
        for (int i = 0; i < 4; i++)
            pts[i] = center + float2(
                                dot(invRotation[0], (canonicalPts[i] * scale)),
                                dot(invRotation[1], (canonicalPts[i] * scale)));
    
        float2 otherPts[4];
        for (int i = 0; i < 4; i++)
            otherPts[i] = other.center + float2(
                dot(otherInvRotation[0], (canonicalPts[i] * other.scale)),
                dot(otherInvRotation[1], (canonicalPts[i] * other.scale)));

        return !(arePtsSeparatedAlongAxes(pts, otherPts, rotation) ||
                arePtsSeparatedAlongAxes(pts, otherPts, other.rotation));
    }

    static bool arePtsSeparatedAlongAxes(float2[4] pts, float2[4] otherPts, float2x2 axes)
    {
        // If any set of points are entirely on one side of the other, they are separated.
        //
        for (int i = 0; i < 2; i++)
        {
            float2 axis = axes[i];
            float2 proj = float2(dot(pts[0], axis), dot(pts[0], axis));
            float2 otherProj = float2(dot(otherPts[0], axis), dot(otherPts[0], axis));

            for (int j = 1; j < 4; j++)
            {
                proj.x = min(proj.x, dot(pts[j], axis));
                proj.y = max(proj.y, dot(pts[j], axis));

                otherProj.x = min(otherProj.x, dot(otherPts[j], axis));
                otherProj.y = max(otherProj.y, dot(otherPts[j], axis));
            }

            if (proj.y < otherProj.x || otherProj.y < proj.x)
                return true;
        }

        return false;
    }

    // In Slang, constructors are defined through special methods named `__init`.
    // Several constructors can be defined, and overload resolution will pick the right one.
    //
    __init(float2 center, float2x2 rotation, float2 scale)
    {
        this.center = center;
        this.rotation = rotation;
        this.scale = scale;
    }
};

/*
* smoothStep maps a value from the range [minval, maxval] to the range [0, 1] using a smooth function.

* The Hermite interpolation function makes sure the derivative is 0 at the ends of the range. 
* This is helpful for representing optimizable parameters since it prevents the parameters from exceeding 
* the valid range and losing gradients.
* 
* Note that this function is marked `[Differentiable]`, which allows it to be used in other differentiable functions
* and will be differentiated automatically by the compiler whenever necessary.
*/
[Differentiable]
vector<float, N> smoothStep<let N : int>(vector<float, N> x, vector<float, N> minval, vector<float, N> maxval)
{
    vector<float, N> y = clamp((x - minval) / (maxval - minval), 0.f, 1.f);
    return y * y * (3.f - 2.f * y);
}

// Scalar variant of the above function.
[Differentiable]
float smoothStep(float x, float minval, float maxval)
{
    float y = clamp((x - minval) / (maxval - minval), 0.f, 1.f);
    return y * y * (3.f - 2.f * y);
}

/*
* A utility function to premultiply the color by the alpha value. 
* This is a key part of the alpha blending routine used in the 
* Gaussian splatting algorithm.
*/
[Differentiable]
float4 preMult(float4 pixel)
{
    return float4(pixel.rgb * pixel.a, pixel.a);
}

/*
* alphaBlend() implements the standard alpha blending algorithm.
* 
* Takes the current pixel value 'pixel' & blends it with a 
* contribution 'gval' from a new Gaussian.
*/
[Differentiable]
float4 alphaBlend(float4 pixel, float4 gval)
{
    gval = preMult(gval);

    return float4(
        pixel.rgb + gval.rgb * pixel.a,
        pixel.a * (1 - gval.a));
}

/*
* undoAlphaBlend() implements the reverse of the alpha blending algorithm.
* 
* Takes a pixel value 'pixel' and the same 'gval' contribution & 
* computes the previous pixel value.
* 
* This is a critical piece of the backwards pass.
*/
float4 undoAlphaBlend(float4 pixel, float4 gval)
{
    gval = preMult(gval);

    var oldPixelAlpha = pixel.a / (1 - gval.a);
    return float4(
        pixel.rgb - gval.rgb * oldPixelAlpha,
        oldPixelAlpha);
}

/*
* PixelState encapsulates all the info for a pixel as it is being rasterized
* through the sorted list of blobs.
*/
struct PixelState : IDifferentiable
{
    float4 value;
    uint finalCount;
};

/* 
* transformPixelState() applies the alpha blending operation to the pixel state & 
* updates the counter accordingly. 
* 
* This state transition also stops further blending once the pixel is effectively opaque. 
* This is important to avoid the alpha becoming too low (or even 0), at which point
* the blending is not reversible.
*
*/
[Differentiable]
PixelState transformPixelState(PixelState pixel, float4 gval)
{
    var newState = alphaBlend(pixel.value, gval);

    if (pixel.value.a < 1.f / 255.f)
        return { pixel.value, pixel.finalCount };

    return { newState, pixel.finalCount + 1 };
}

/* 
* undoPixelState() reverses the alpha blending operation and restores the previous pixel 
* state.
*/
PixelState undoPixelState(PixelState nextState, uint index, float4 gval)
{
    if (index > nextState.finalCount)
        return { nextState.value, nextState.finalCount };
    
    return { undoAlphaBlend(nextState.value, gval), nextState.finalCount - 1 };
}

/*
* loadFloat() is a helper method that loads a float from the buffer in a *differentiable* manner.
*
* The function itself is fairly straightforward, but the key part is the `[BAckwardDerivative]` attribute.
*
* loadFloat_bwd() is the corresponding user-defined backwards function that is responsible for writing
* back the gradient associated with the loaded float.
*
* Using the [BackwardDerivative] attributes instructs the auto-diff pass to call the provided function to
* backpropagate the gradient (rather than trying to differentiate the function body automatically).
*
* This system is the primary approach to dealing with memory loads & stores since there are many approaches to
* accumulating gradients of memory accesses.
*/
[BackwardDerivative(loadFloat_bwd)]
float loadFloat(uint idx, uint localDispatchIdx)
{
    return blobsBuffer[idx];
}

/* 
* loadFloat_bwd() is the user-defined derivative for loadFloat()
*
* Since loadFloat() is always used to load values that are uniform across the workgroup,
* the differentials must be accumulated before writing back, since each thread will 
* have a different derivative value.
* 
* The function uses a workgroup-level binary reduction to add up the gradients across the workgroup.
* Then the first thread in the workgroup atomically accumulates the gradient to the global derivative buffer.
*/
void loadFloat_bwd(uint idx, uint localDispatchIdx, float dOut)
{
    // Clamp the gradients to avoid any weird problems with the optimization.
    if (abs(dOut) < 10.f)
        reductionBuffer[localDispatchIdx] = dOut;
    else
        reductionBuffer[localDispatchIdx] = 10.f * sign(dOut);
    
    GroupMemoryBarrierWithGroupSync();
    
    // Binary reduction
    for (uint stride = (WG_X * WG_Y) / 2; stride > 0; stride /= 2)
    {
        if (localDispatchIdx < stride)
            reductionBuffer[localDispatchIdx] += reductionBuffer[localDispatchIdx + stride];

        GroupMemoryBarrierWithGroupSync();
    }

    if (localDispatchIdx == 0)
        atomicAccumulate(reductionBuffer[0], idx);
}

/*
* atomicAccumulate() is a helper method that atomically accumulates a float value to the global derivative buffer.
*
* Unfortunately, WGSL does not have floating-point atomics, so this method uses a compare-and-swap (i.e. compareExchange()) loop to perform
* this operation. This is a common pattern for implementing floating-point atomics on platforms that do not support them.
*
* This function makes use of 'bitcasting' which is a way of reinterpreting the bits of one type as another type. Note that this is
* different from type-casting, which changes the value of the data. 
* In Slang, this can be done via type-specific methods such as `asfloat()` or `asuint()` or more generally via `bit_cast<T, U>()`
*
*/
void atomicAccumulate(float val, uint idx)
{
    // No need to accumulate zeros.
    if (val == 0.f)
        return; 

    // Loop for as long as the compareExchange() fails, which means another thread 
    // is trying to write to the same location.
    //
    for (;;)
    {
        uint oldInt = derivBuffer[idx].load();
        float oldFloat = asfloat(oldInt);

        float newFloat = oldFloat + val;

        uint newInt = asuint(newFloat);

        // compareExchange() returns the value at the location before the operation.
        // If it's changed, we have contention between threads & need to try again.
        //
        if (derivBuffer[idx].compareExchange(oldInt, newInt) == oldInt)
            break;
    }
}

[Differentiable]
float2x2 inverse(float2x2 mat)
{
    float2x2 output;

    float det = determinant(mat);
    output[0][0] = mat[1][1] / det;
    output[0][1] = -mat[0][1] / det;
    output[1][0] = -mat[1][0] / det;
    output[1][1] = mat[0][0] / det;

    return output;
}

struct Gaussian2D : IDifferentiable
{
    float2 center;
    float2x2 sigma;
    float3 color;
    float opacity;

    [Differentiable]
    static Gaussian2D load(uint idx, uint localIdx)
    {
        uint total = Gaussian2D.count();
        Gaussian2D gaussian;

        gaussian.center = smoothStep(
            float2(
                loadFloat(total * 0 + idx, localIdx),
                loadFloat(total * 1 + idx, localIdx)),
            float2(0, 0),
            float2(1, 1));
        
        // Add a small padding value to avoid singularities or unstable Gaussians.
        gaussian.sigma[0][0] = smoothStep(
            loadFloat(total * 2 + idx, localIdx) * 0.8f, 0.f, 1.f) + 0.005f; 
        gaussian.sigma[1][1] = smoothStep(
            loadFloat(total * 3 + idx, localIdx) * 0.8f, 0.f, 1.f) + 0.005f; 

        float aniso = (smoothStep(
            loadFloat(total * 4 + idx, localIdx) * 0.6f, 0.f, 1.f) - 0.5f) * 1.65f;
        
        gaussian.sigma[0][1] = sqrt(gaussian.sigma[0][0] * gaussian.sigma[1][1]) * aniso;
        gaussian.sigma[1][0] = sqrt(gaussian.sigma[0][0] * gaussian.sigma[1][1]) * aniso;

        
        gaussian.color = smoothStep(
            float3(
                loadFloat(total * 5 + idx, localIdx) * 0.8f,
                loadFloat(total * 6 + idx, localIdx) * 0.8f,
                loadFloat(total * 7 + idx, localIdx) * 0.8f),
            float3(0, 0, 0),
            float3(1, 1, 1));

        gaussian.opacity = smoothStep(
            loadFloat(total * 8 + idx, localIdx) * 0.9f + 0.1f, 0, 1);

        // Scale the sigma so the blobs aren't too large
        gaussian.sigma *= 0.0001;

        return gaussian;
    }

    // Simple helper method to get the number of elements in the buffer
    static uint count()
    {
        uint elementCount = (uint)BLOB_BUFFER_SIZE;
        return elementCount / NUM_FIELDS;
    }

    /*
    * eval() calculates the color and weight of the Gaussian at a given UV coordinate.
    * 
    * This method calculates an alpha by applying the standard multi-variate Gaussian formula 
    * to calculate the power which is then scaled by an opacity value. The color components 
    * are represented by additional fields.
    */
    [Differentiable]
    float4 eval(float2 uv)
    {
        float2x2 invCov = inverse(sigma);
        float2 diff = uv - center;
        float power = -0.5f * ((diff.x * diff.x * invCov[0][0]) +
                            (diff.y * diff.y * invCov[1][1]) +
                            (diff.x * diff.y * invCov[0][1]) +
                            (diff.y * diff.x * invCov[1][0]));
        
        float weight = min(.99f, opacity * exp(power));
        return float4(color, weight);
    }

    OBB bounds()
    {
        // Calculate eigenvectors for the 2x2 matrix.
        float2x2 cov = sigma;

        float a = cov[0][0];
        float b = cov[0][1];
        float c = cov[1][0];
        float d = cov[1][1];

        float n_stddev = 4.f;

        if (abs(b) < 1e-6 || abs(c) < 1e-6)
        {
            // The covariance matrix is diagonal (or close enough..), so the eigenvectors are the x and y axes.
            float2x2 eigenvectors = float2x2(float2(1, 0), float2(0, 1));
            float2 scale = float2(sqrt(a), sqrt(d));

            return OBB(center, eigenvectors, scale * n_stddev);
        }
        else
        {
            float trace = a + d;
            float det = a * d - b * c;

            float lambda1 = 0.5 * (trace + sqrt(trace * trace - 4 * det));
            float lambda2 = 0.5 * (trace - sqrt(trace * trace - 4 * det));

            float2x2 eigenvectors;
            eigenvectors[0] = float2(lambda1 - d, c) / length(float2(lambda1 - d, c));
            eigenvectors[1] = float2(b, lambda2 - a) / length(float2(b, lambda2 - a));

            // Calculate the scale of the OBB
            float2 scale = float2(sqrt(lambda1), sqrt(lambda2));

            return OBB(center, eigenvectors, scale * n_stddev);
        }
    }
};

/*
* padBuffer() is a helper method that fills the unused space in the buffer with a sentinel value (uint::maxValue).
* This is just because bitonicSort requires all elements to have a valid value. padBuffer filles these in with
* maxValue, which are effectively pushed to the end of the list.
*/
PaddedShortList padBuffer(FilledShortList, uint localIdx)
{
    GroupMemoryBarrierWithGroupSync();

    var maxN = blobCount;
    for (uint i = localIdx; i < GAUSSIANS_PER_BLOCK; i += (WG_X * WG_Y))
    {
        if (i >= maxN)
            blobs[i] = uint::maxValue;
    }

    return { 0 };
}

/*
* bitonicSort() implements a workgroup-level parallel sorting algorithm to sort indices in the short-list.
* Requires all elements in the buffer to be valid (invalid elements should be set to infinity, or its equivalent).
*
* Bitonic sorting is an efficient, deterministic, parallel sorting algorithm particularly well-suited for GPUs. 
* At a high-level, it operates by comparing & swapping elements in parallel in (logN)^2 stages.
*
* More info on the bitonic sort algorithm: https://en.wikipedia.org/wiki/Bitonic_sorter
* The code was adapted from the Wikipedia sample pseudocode here: https://en.wikipedia.org/wiki/Bitonic_sorter#Example_code
* 
*/
SortedShortList bitonicSort(PaddedShortList, uint localIdx)
{
    GroupMemoryBarrierWithGroupSync();

    uint maxN = blobCount;
    for (uint k = 2; k <= GAUSSIANS_PER_BLOCK; k *= 2)
    {
        for (uint j = k / 2; j > 0; j /= 2)
        {
            for (uint i = localIdx; i < GAUSSIANS_PER_BLOCK; i += WG_X * WG_Y)
            {
                uint l = i ^ j;
                if (l > i)
                {
                    if ((((i & k) == 0) && (blobs[i] > blobs[l])) ||
                        (((i & k) != 0) && (blobs[i] < blobs[l])))
                    {
                        // Swap
                        var temp = blobs[i];
                        blobs[i] = blobs[l];
                        blobs[l] = temp;
                    }
                }
            }

            GroupMemoryBarrierWithGroupSync();
        }
    }

    return { 0 };
}

/*
* coarseRasterize() calculates a subset of blobs that intersect with the current tile. Expects the blob counters to be reset before calling.
*
* The coarse rasterization step determines a subset of blobs that intersect with the tile.
* Each thread in the workgroup takes a subset of blobs and uses bounding-box intersection tests to determine
* if the tile associated with this workgroup overlaps with the blob's bounds.
* 
* Note: This is a simplistic implementation, so there is a limit to the number of blobs in the short-list (NUM_GAUSSIANS_PER_BLOCK).
* In practice, if the number of blobs per tile exceeds this, NUM_GAUSSIANS_PER_BLOCK must be increased manually. 
* A more sophisticated implementation would perform multiple passes to handle this case.
*
*/
FilledShortList coarseRasterize(InitializedShortList sList, OBB tileBounds, uint localIdx)
{
    GroupMemoryBarrierWithGroupSync();

    Gaussian2D gaussian;
    uint numGaussians = Gaussian2D.count();
    for (uint i = localIdx; i < numGaussians; i += (WG_X * WG_Y))
    {
        gaussian = Gaussian2D.load(i, localIdx);
        OBB bounds = gaussian.bounds();
        if (bounds.intersects(tileBounds))
        {
            blobs[blobCountAT++] = i;
        }
    }

    GroupMemoryBarrierWithGroupSync();

    blobCount = blobCountAT.load();

    return { 0 };
}

[Differentiable]
float4 eval(uint blob_id, no_diff float2 uv, uint localIdx)
{
    Gaussian2D gaussian = Gaussian2D.load(blob_id, localIdx);
    return gaussian.eval(uv);
}

/* 
* fineRasterize() produces the per-pixel final color from a sorted list of blobs that overlap the current tile.
*
* The fine rasterizeration is where the calculation of the per-pixel color happens. 
* This uses the multiplicative alpha blending algorithm laid out in the original GS paper (https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
* This is represented as a 'state transition' (transformPixelState) as we go through the blobs in order, so that we can 
* concisely represent the 'state undo' operation in the backwards pass.
* 
* In Slang, custom derivative functions can be defiened using the `[BackwardDerivative(custom_fn)]` attribute.
*/
[BackwardDerivative(fineRasterize_bwd)]
float4 fineRasterize(SortedShortList, uint localIdx, no_diff float2 uv)
{
    GroupMemoryBarrierWithGroupSync();

    PixelState pixelState = PixelState(float4(0, 0, 0, 1), 0);
    uint count = blobCount;
    // The forward rasterization 
    for (uint i = 0; i < count; i++)
        pixelState = transformPixelState(pixelState, eval(blobs[i], uv, localIdx));

    maxCount[localIdx] = pixelState.finalCount;
    finalVal[localIdx] = pixelState.value;
    return pixelState.value;
}

/*
* fineRasterize_bwd() is the user-provided backwards pass for the fine rasterization step.
* 
* This is implemented as a custom derivative function because, while applying auto-diff directly to a function
* with a loop can result in excessive state caching (a necessary part of standard automatic differentiation methods)
*
* For Gaussian splatting, there is a 'state undo' (undoPixelState) operation available. fineRasterize_bwd takes advantage of this 
* to recreate the states at each step of the forward pass instead of letting auto-diff store them.
* 
* While it is important to represent the backwards loop explicitly in this way, the contents of the loop body (loading, evaluation, 
* blending, etc..) can still be differentiated automatically (and it would be tedioush to do so manually). 
*
* The loop body therefore invokes `bwd_diff` to backprop the derivatives via auto-diff.
*/
void fineRasterize_bwd(SortedShortList, uint localIdx, float2 uv, float4 dOut)
{
    GroupMemoryBarrierWithGroupSync();

    PixelState pixelState = { finalVal[localIdx], maxCount[localIdx] };

    PixelState.Differential dColor = { dOut };
    
    // `workgroupUniformLoad` is a WGSL-specific intrinsic that marks a load as uniform across the workgroup.
    // This is necessary to prevent errors from uniformity analysis.
    //
    uint count = workgroupUniformLoad(blobCount);

    // The backwards pass manually performs an 'undo' to reproduce the state at each step.
    // The inner loop body still uses auto-diff, so the bulk of the computation is still
    // handled by the auto-diff engine.
    //
    for (uint _i = count; _i > 0; _i--)
    {
        uint i = _i - 1;
        var blobID = blobs[i];
        var gval = eval(blobID, uv, localIdx);
        var prevState = undoPixelState(pixelState, i+1, gval);

        var dpState = diffPair(prevState);
        var dpGVal = diffPair(gval);
        
        // Once we have the previous state, we can continue with the backpropagation via auto-diff within
        // the loop body. Note that the `bwd_diff` calls writeback the differentials to dpState and dpGVal,
        // and can be obtained via `getDifferential()` (or simply '.d')
        // 
        bwd_diff(transformPixelState)(dpState, dpGVal, dColor);
        bwd_diff(eval)(blobID, uv, localIdx, dpGVal.getDifferential());

        pixelState = prevState;
        dColor = dpState.getDifferential();
    }
}

InitializedShortList initShortList(uint2 dispatchThreadID)
{
    GroupMemoryBarrierWithGroupSync();

    if (dispatchThreadID.x % WG_X == 0 && dispatchThreadID.y % WG_Y == 0)
    {
        blobCount = 0; blobCountAT = 0;
    }

    return { 0 };
}

/* 
* calcUV() computes a 'stretch-free' mapping from the requested render-target dimensions (renderSize) to the
* image in the texture (imageSize)
*/
float2 calcUV(uint2 dispatchThreadID, int2 renderSize, int2 imageSize)
{
    // Easy case.
    if (all(renderSize == imageSize))
        return ((float2)dispatchThreadID) / renderSize;
    
    float aspectRatioRT = ((float) renderSize.x) / renderSize.y;
    float aspectRatioTEX = ((float) imageSize.x) / imageSize.y;

    if (aspectRatioRT > aspectRatioTEX)
    {
        // Render target is wider than the texture. 
        // Match the widths.
        //
        float xCoord = ((float) dispatchThreadID.x) / renderSize.x;
        float yCoord = ((float) dispatchThreadID.y * aspectRatioTEX) / renderSize.x;

        // We'll re-center the y-coord around 0.5.
        float yCoordMax = aspectRatioTEX / aspectRatioRT;
        yCoord = yCoord + (1.0 - yCoordMax) / 2.0f;
        return float2(xCoord, yCoord);
    }
    else
    {
        // Render target is taller than the texture. 
        // Match the heights.
        //
        float yCoord = ((float) dispatchThreadID.y) / renderSize.y;
        float xCoord = ((float) dispatchThreadID.x) / (renderSize.y * aspectRatioTEX);

        // We'll recenter the x-coord around 0.5.
        float xCoordMax = aspectRatioRT / aspectRatioTEX;
        xCoord = xCoord + (1.0 - xCoordMax) / 2.0f;
        return float2(xCoord, yCoord);
    }
}

/* 
* splatBlobs() is the main rendering routine that computes a final color for the pixel.
* 
* It proceeds in 4 stages: 
*  1. Coarse rasterization: Short-list blobs that intersect with the current tile through 
*                           bounding-box intersection tests.
*  2. Padding: Fill the unused space in the buffer with a sentinel value.
*  3. Sorting: Sort the short list of blobs.
*  4. Fine rasterization: Calculate the final color for the pixel.
* 
* Note that only the final stage is differentiable since it is the only stage that produces 
* the final color. 
* The other stages are just optimizations to reduce the blobs under consideration.
*
* The produced derivative function will re-use the same optimizations as-is.
* 
*/
[Differentiable]
float4 splatBlobs(uint2 dispatchThreadID, int2 dispatchSize)
{
    uint globalID = dispatchThreadID.x + dispatchThreadID.y * dispatchSize.x;
    
    int texWidth;
    int texHeight;
    targetTexture.GetDimensions(texWidth, texHeight);
    int2 texSize = int2(texWidth, texHeight);

    // Calculate effective uv coordinate for the current pixel. This is used for 
    // evaluating the 2D Daussians.
    float2 uv = calcUV(dispatchThreadID, dispatchSize, texSize);
    
    //
    // Calculate a bounding box in uv coordinates for the current workgroup.
    //

    uint2 tileCoords = uint2(dispatchThreadID.x / WG_X, dispatchThreadID.y / WG_Y);

    float2 tileLow = calcUV(tileCoords * uint2(WG_X, WG_Y), dispatchSize, texSize);
    float2 tileHigh = calcUV((tileCoords + 1) * uint2(WG_X, WG_Y), dispatchSize, texSize);

    float2 tileCenter = (tileLow + tileHigh) / 2;
    float2x2 tileRotation = float2x2(1, 0, 0, 1);
    float2 tileScale = (tileHigh - tileLow) / 2;

    OBB tileBounds = OBB(tileCenter, tileRotation, tileScale);
    
    // -------------------------------------------------------------------

    // Main rendering steps..

    // Initialize the short list (by resetting counters)
    InitializedShortList sList = initShortList(dispatchThreadID);

    uint2 localID = dispatchThreadID % uint2(WG_X, WG_Y);
    uint localIdx = localID.x + localID.y * WG_X;

    // Short-list blobs that overlap with the local tile.
    FilledShortList filledSList = coarseRasterize(sList, tileBounds, localIdx);

    // Pad the unused space in the buffer
    PaddedShortList paddedSList = padBuffer(filledSList, localIdx);

    // Sort the short list
    SortedShortList sortedList = bitonicSort(paddedSList, localIdx);

    // Perform per-pixel fine rasterization
    float4 color = fineRasterize(sortedList, localIdx, uv);

    // Blend with background
    return float4(color.rgb * (1.0 - color.a) + color.a, 1.0);
}

/*
* loss() implements the standard L2 loss function to quantify the difference between 
* the rendered image and the target texture.
*/
[Differentiable]
float loss(uint2 dispatchThreadID, int2 imageSize)
{
    // Splat the blobs and calculate the color for this pixel.
    float4 color = splatBlobs(dispatchThreadID, imageSize);

    float4 targetColor;
    float weight;
    if (dispatchThreadID.x >= imageSize.x || dispatchThreadID.y >= imageSize.y)
    {
        return 0.f;
    }
    else
    {
        uint2 flippedCoords = uint2(dispatchThreadID.x, imageSize.y - dispatchThreadID.y);
        targetColor = no_diff targetTexture[flippedCoords];
        return dot(color.rgb - targetColor.rgb, color.rgb - targetColor.rgb);
    }

    return 0.f;
}

/*
* clearDerivativesMain() is a kernel that resets the derivative buffer to all 0s
*/
[shader("compute")]
[numthreads(64, 1, 1)]
void clearDerivativesMain(uint2 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x >= BLOB_BUFFER_SIZE)
        return;
    
    derivBuffer[dispatchThreadID.x].store(asuint(0.f));
}

/*
* computeDerivativesMain() is a kernel that computes the derivatives of the loss function with respect to the blobs.
* 
* It uses Slang's auto-diff capabilities by simply calling `bwd_diff()` on the loss function to generate a new function
* that is the derivative of the loss function.
*/
[shader("compute")]
[numthreads(16, 16, 1)]
void computeDerivativesMain(uint2 dispatchThreadID : SV_DispatchThreadID)
{
    uint dimX;
    uint dimY;
    targetTexture.GetDimensions(dimX, dimY);

    int2 targetImageSize = int2(dimX, dimY);

    // Distribute the 1.f total weight across all pixels
    float perPixelWeight = 1.f / (targetImageSize.x * targetImageSize.y);

    // Backprop (will write derivatives to the derivBuffer)
    bwd_diff(loss)(dispatchThreadID, targetImageSize, perPixelWeight);
}

/* 
* updateBlobsMain() is a kernel that updates the blob parameters using the Adam optimizer.
* 
* Since all the parameters are laid out in a single float buffer, there is no need to re-interpret 
* the buffer into a struct.
* 
* The Adam optimization method (https://arxiv.org/abs/1412.6980) is used to process the gradients before
* applying the update. It acts as a temporal filter on the gradients, and stores per-parameter state that
* persists across iterations to help stabilize the optimization process.
*
*/
[shader("compute")]
[numthreads(256, 1, 1)]
void updateBlobsMain(uint2 dispatchThreadID: SV_DispatchThreadID)
{
    var globalID = dispatchThreadID.x;
    if (globalID >= BLOB_BUFFER_SIZE)
        return;

    // Read & reset the derivative
    float g_t = asfloat(derivBuffer[globalID].load());
    derivBuffer[globalID] = asuint(0.f);

    float g_t_2 = g_t * g_t;

    // 
    // Perform a gradient update using Adam optimizer rules for
    // a smoother optimization.
    // 

    float m_t_prev = adamFirstMoment[globalID];
    float v_t_prev = adamSecondMoment[globalID];
    float m_t = ADAM_BETA_1 * m_t_prev + (1 - ADAM_BETA_1) * g_t;
    float v_t = ADAM_BETA_2 * v_t_prev + (1 - ADAM_BETA_2) * g_t_2;

    adamFirstMoment[globalID] = m_t;
    adamSecondMoment[globalID] = v_t;

    float m_t_hat = m_t / (1 - ADAM_BETA_1);
    float v_t_hat = v_t / (1 - ADAM_BETA_2);

    float update = (ADAM_ETA / (sqrt(v_t_hat) + ADAM_EPSILON)) * m_t_hat;

    blobsBuffer[globalID] -= update;
}

// Sequence of additional kernel calls to be performed before imageMain
// By default, imageMain is always the last kernel to be dispatched
// 
// "//! CALL" is a slang-playground directive that helps us queue up additional kernels.
// Note that if this sample code is outside the playground, your engine is responsible for
// dispatching these kernels in this order.
//
//! CALL(clearDerivativesMain, SIZE_OF(blobsBuffer))
//! CALL(computeDerivativesMain, SIZE_OF(targetTexture))
//! CALL(updateBlobsMain, SIZE_OF(blobsBuffer))
//

float4 imageMain(uint2 dispatchThreadID, uint2 screenSize)
{
    return splatBlobs(dispatchThreadID, screenSize);
}

*/



/// compute.toys glue code ///

const BLOB_BUFFER_SIZE = 184320u;

struct SimulationBuffers {
    blobs_buffer: array<f32, BLOB_BUFFER_SIZE>,
    deriv_buffer: array<atomic<u32>, BLOB_BUFFER_SIZE>,
    adam_first_moment: array<f32, BLOB_BUFFER_SIZE>,
    adam_second_moment: array<f32, BLOB_BUFFER_SIZE>,
}

#storage fields SimulationBuffers

#define blobsBuffer_0 fields.blobs_buffer
#define derivBuffer_0 fields.deriv_buffer
#define adamFirstMoment_0 fields.adam_first_moment
#define adamSecondMoment_0 fields.adam_second_moment

#define targetTexture_0 channel0

#define outputTexture_0 screen

// BLOB_BUFFER_SIZE / 64
#workgroup_count computeMain 2880 1 1
#workgroup_count clearDerivativesMain 2880 1 1

// (channel0 dims) 512 / 16
#workgroup_count computeDerivativesMain 32 32 1

// BLOB_BUFFER_SIZE / 256
#workgroup_count updateBlobsMain 720 1 1

#dispatch_once computeMain
#define outputBuffer_0 blobsBuffer_0
#define seed_0 42



/// Initialise the RAND buffer ///

/* https://github.com/shader-slang/slang-playground/blob/main/demos/rand_float.slang

// Hybrid Tausworthe PRNG
//
// Code adapted from: https://indico.cern.ch/event/93877/papers/2118070/files/4416-acat3.pdf (See document for license)
//

uniform float seed;
RWStructuredBuffer<float> outputBuffer;

uint seedPerThread(uint idx)
{
    return ((uint)idx + (uint)(seed * 1000000)) * 1099087573UL;
}

uint tauStep(uint z, uint s1, uint s2, uint s3, uint M)
{
    uint b = (((z << s1) ^ z) >> s2);
    return (((z & M) << s3) ^ b);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint2 dispatchThreadId : SV_DispatchThreadID)
{
    uint idx = dispatchThreadId.x;
    uint val = ((uint)idx) * 1099087573UL + ((uint)seed) * 12003927;

    uint z = tauStep(val, 13, 19, 12, 4294967294);
    z = tauStep(z, 2, 25, 4, 4294967288);
    z = tauStep(z, 3, 11, 17, 4294967280);

    uint z1, z2, z3, z4;
    uint r0, r1, r2, r3;

    // STEP 1
    uint _seed = seedPerThread(idx);
    z1 = tauStep(_seed, 13, 19, 12, 429496729UL);
    z2 = tauStep(_seed, 2, 25, 4, 4294967288UL);
    z3 = tauStep(_seed, 3, 11, 17, 429496280UL);
    z4 = (1664525 * _seed + 1013904223UL);
    r0 = (z1 ^ z2 ^ z3 ^ z4);
    // STEP 2
    z1 = tauStep(r0, 13, 19, 12, 429496729UL);
    z2 = tauStep(r0, 2, 25, 4, 4294967288UL);
    z3 = tauStep(r0, 3, 11, 17, 429496280UL);
    z4 = (1664525 * r0 + 1013904223UL);
    r1 = (z1 ^ z2 ^ z3 ^ z4);
    // STEP 3
    z1 = tauStep(r1, 13, 19, 12, 429496729UL);
    z2 = tauStep(r1, 2, 25, 4, 4294967288UL);
    z3 = tauStep(r1, 3, 11, 17, 429496280UL);
    z4 = (1664525 * r1 + 1013904223UL);
    r2 = (z1 ^ z2 ^ z3 ^ z4);
    // STEP 4
    z1 = tauStep(r2, 13, 19, 12, 429496729UL);
    z2 = tauStep(r2, 2, 25, 4, 4294967288UL);
    z3 = tauStep(r2, 3, 11, 17, 429496280UL);
    z4 = (1664525 * r2 + 1013904223UL);
    r3 = (z1 ^ z2 ^ z3 ^ z4);

    float u4 = r3 * 2.3283064365387e-10;

    outputBuffer[idx] = u4;
}

*/

// @binding(1) @group(0) var<storage, read_write> outputBuffer_0 : array<f32>;

// struct GlobalParams_std140_0
// {
//     @align(16) seed_0 : f32,
// };

// @binding(0) @group(0) var<uniform> globalParams_0 : GlobalParams_std140_0;
fn tauStep_0( z_0 : u32,  s1_0 : u32,  s2_0 : u32,  s3_0 : u32,  M_0 : u32) -> u32
{
    return (((((z_0 & (M_0))) << (s3_0))) ^ (((((((z_0 << (s1_0))) ^ (z_0))) >> (s2_0)))));
}

fn seedPerThread_0( idx_0 : u32) -> u32
{
    return (idx_0 + u32(seed_0 * 1.0e+06f)) * u32(1099087573);
}

@compute
@workgroup_size(64, 1, 1)
fn computeMain(@builtin(global_invocation_id) dispatchThreadId_0 : vec3<u32>)
{
    var idx_1 : u32 = dispatchThreadId_0.xy.x;
    var _seed_0 : u32 = seedPerThread_0(idx_1);
    var r0_0 : u32 = ((((((tauStep_0(_seed_0, u32(13), u32(19), u32(12), u32(429496729))) ^ ((tauStep_0(_seed_0, u32(2), u32(25), u32(4), u32(4294967288)))))) ^ ((tauStep_0(_seed_0, u32(3), u32(11), u32(17), u32(429496280)))))) ^ ((u32(1664525) * _seed_0 + u32(1013904223))));
    var r1_0 : u32 = ((((((tauStep_0(r0_0, u32(13), u32(19), u32(12), u32(429496729))) ^ ((tauStep_0(r0_0, u32(2), u32(25), u32(4), u32(4294967288)))))) ^ ((tauStep_0(r0_0, u32(3), u32(11), u32(17), u32(429496280)))))) ^ ((u32(1664525) * r0_0 + u32(1013904223))));
    var r2_0 : u32 = ((((((tauStep_0(r1_0, u32(13), u32(19), u32(12), u32(429496729))) ^ ((tauStep_0(r1_0, u32(2), u32(25), u32(4), u32(4294967288)))))) ^ ((tauStep_0(r1_0, u32(3), u32(11), u32(17), u32(429496280)))))) ^ ((u32(1664525) * r1_0 + u32(1013904223))));
    outputBuffer_0[idx_1] = f32(((((((tauStep_0(r2_0, u32(13), u32(19), u32(12), u32(429496729))) ^ ((tauStep_0(r2_0, u32(2), u32(25), u32(4), u32(4294967288)))))) ^ ((tauStep_0(r2_0, u32(3), u32(11), u32(17), u32(429496280)))))) ^ ((u32(1664525) * r2_0 + u32(1013904223))))) * 2.32830643653869629e-10f;
    return;
}



/// clearDerivativesMain entrypoint ///

// @binding(3) @group(0) var<storage, read_write> derivBuffer_0 : array<atomic<u32>>;

@compute
@workgroup_size(64, 1, 1)
fn clearDerivativesMain(@builtin(global_invocation_id) dispatchThreadID_0 : vec3<u32>)
{
    var _S1 : u32 = dispatchThreadID_0.xy.x;
    if(_S1 >= u32(184320))
    {
        return;
    }
    atomicStore(&(derivBuffer_0[_S1]), (bitcast<u32>((0.0f))));
    return;
}



/// computeDerivativesMain entrypoint ///

// @binding(6) @group(0) var targetTexture_0 : texture_2d<f32>;

// @binding(3) @group(0) var<storage, read_write> derivBuffer_0 : array<atomic<u32>>;

// @binding(2) @group(0) var<storage, read_write> blobsBuffer_0 : array<f32>;

const _S1 : mat2x2<f32> = mat2x2<f32>(0.00009999999747379f, 0.00009999999747379f, 0.00009999999747379f, 0.00009999999747379f);
fn calcUV_0( dispatchThreadID_0 : vec2<u32>,  renderSize_0 : vec2<i32>,  imageSize_0 : vec2<i32>) -> vec2<f32>
{
    if(all(renderSize_0 == imageSize_0))
    {
        return vec2<f32>(dispatchThreadID_0) / vec2<f32>(renderSize_0);
    }
    var _S2 : f32 = f32(renderSize_0.x);
    var _S3 : f32 = f32(renderSize_0.y);
    var aspectRatioRT_0 : f32 = _S2 / _S3;
    var aspectRatioTEX_0 : f32 = f32(imageSize_0.x) / f32(imageSize_0.y);
    if(aspectRatioRT_0 > aspectRatioTEX_0)
    {
        return vec2<f32>(f32(dispatchThreadID_0.x) / _S2, f32(dispatchThreadID_0.y) * aspectRatioTEX_0 / _S2 + (1.0f - aspectRatioTEX_0 / aspectRatioRT_0) / 2.0f);
    }
    else
    {
        return vec2<f32>(f32(dispatchThreadID_0.x) / (_S3 * aspectRatioTEX_0) + (1.0f - aspectRatioRT_0 / aspectRatioTEX_0) / 2.0f, f32(dispatchThreadID_0.y) / _S3);
    }
}

struct OBB_0
{
     center_0 : vec2<f32>,
     rotation_0 : mat2x2<f32>,
     scale_0 : vec2<f32>,
};

fn OBB_x24init_0( center_1 : vec2<f32>,  rotation_1 : mat2x2<f32>,  scale_1 : vec2<f32>) -> OBB_0
{
    var _S4 : OBB_0;
    _S4.center_0 = center_1;
    _S4.rotation_0 = rotation_1;
    _S4.scale_0 = scale_1;
    return _S4;
}

var<workgroup> blobCount_0 : u32;

var<workgroup> blobCountAT_0 : atomic<u32>;

struct InitializedShortList_0
{
     _dummy_0 : i32,
};

fn initShortList_0( dispatchThreadID_1 : vec2<u32>) -> InitializedShortList_0
{
    workgroupBarrier();
    var _S5 : bool;
    if((dispatchThreadID_1.x % u32(16)) == u32(0))
    {
        _S5 = (dispatchThreadID_1.y % u32(16)) == u32(0);
    }
    else
    {
        _S5 = false;
    }
    if(_S5)
    {
        blobCount_0 = u32(0);
        atomicStore(&(blobCountAT_0), u32(0));
    }
    var _S6 : InitializedShortList_0 = InitializedShortList_0( i32(0) );
    return _S6;
}

fn Gaussian2D_count_0() -> u32
{
    return u32(20480);
}

var<workgroup> reductionBuffer_0 : array<f32, i32(256)>;

fn atomicAccumulate_0( val_0 : f32,  idx_0 : u32)
{
    if(val_0 == 0.0f)
    {
        return;
    }
    for(;;)
    {
        var oldInt_0 : u32 = atomicLoad(&(derivBuffer_0[idx_0]));
        var _S7 : u32 = atomicCompareExchangeWeak(&(derivBuffer_0[idx_0]), oldInt_0, (bitcast<u32>(((bitcast<f32>((oldInt_0))) + val_0)))).old_value;
        if(_S7 == oldInt_0)
        {
            break;
        }
    }
    return;
}

fn loadFloat_bwd_0( idx_1 : u32,  localDispatchIdx_0 : u32,  dOut_0 : f32)
{
    if((abs(dOut_0)) < 10.0f)
    {
        reductionBuffer_0[localDispatchIdx_0] = dOut_0;
    }
    else
    {
        reductionBuffer_0[localDispatchIdx_0] = 10.0f * f32(sign(dOut_0));
    }
    workgroupBarrier();
    var _S8 : bool = localDispatchIdx_0 == u32(0);
    var stride_0 : u32 = u32(128);
    for(;;)
    {
        if(stride_0 > u32(0))
        {
        }
        else
        {
            break;
        }
        if(localDispatchIdx_0 < stride_0)
        {
            reductionBuffer_0[localDispatchIdx_0] = reductionBuffer_0[localDispatchIdx_0] + reductionBuffer_0[localDispatchIdx_0 + stride_0];
        }
        workgroupBarrier();
        stride_0 = stride_0 / u32(2);
    }
    if(_S8)
    {
        atomicAccumulate_0(reductionBuffer_0[i32(0)], idx_1);
    }
    return;
}

fn loadFloat_0( idx_2 : u32,  localDispatchIdx_1 : u32) -> f32
{
    return blobsBuffer_0[idx_2];
}

struct DiffPair_float_0
{
     primal_0 : f32,
     differential_0 : f32,
};

fn _d_min_0( dpx_0 : ptr<function, DiffPair_float_0>,  dpy_0 : ptr<function, DiffPair_float_0>,  dOut_1 : f32)
{
    var _S9 : DiffPair_float_0 = (*dpx_0);
    var _S10 : f32;
    if(((*dpx_0).primal_0) < ((*dpy_0).primal_0))
    {
        _S10 = dOut_1;
    }
    else
    {
        _S10 = 0.0f;
    }
    (*dpx_0).primal_0 = _S9.primal_0;
    (*dpx_0).differential_0 = _S10;
    var _S11 : DiffPair_float_0 = (*dpy_0);
    if(((*dpy_0).primal_0) < (_S9.primal_0))
    {
        _S10 = dOut_1;
    }
    else
    {
        _S10 = 0.0f;
    }
    (*dpy_0).primal_0 = _S11.primal_0;
    (*dpy_0).differential_0 = _S10;
    return;
}

fn _d_clamp_0( dpx_1 : ptr<function, DiffPair_float_0>,  dpMin_0 : ptr<function, DiffPair_float_0>,  dpMax_0 : ptr<function, DiffPair_float_0>,  dOut_2 : f32)
{
    var _S12 : DiffPair_float_0 = (*dpx_1);
    var _S13 : bool;
    if(((*dpx_1).primal_0) > ((*dpMin_0).primal_0))
    {
        _S13 = ((*dpx_1).primal_0) < ((*dpMax_0).primal_0);
    }
    else
    {
        _S13 = false;
    }
    var _S14 : f32;
    if(_S13)
    {
        _S14 = dOut_2;
    }
    else
    {
        _S14 = 0.0f;
    }
    (*dpx_1).primal_0 = _S12.primal_0;
    (*dpx_1).differential_0 = _S14;
    var _S15 : DiffPair_float_0 = (*dpMin_0);
    if((_S12.primal_0) <= ((*dpMin_0).primal_0))
    {
        _S14 = dOut_2;
    }
    else
    {
        _S14 = 0.0f;
    }
    (*dpMin_0).primal_0 = _S15.primal_0;
    (*dpMin_0).differential_0 = _S14;
    var _S16 : DiffPair_float_0 = (*dpMax_0);
    if(((*dpx_1).primal_0) >= ((*dpMax_0).primal_0))
    {
        _S14 = dOut_2;
    }
    else
    {
        _S14 = 0.0f;
    }
    (*dpMax_0).primal_0 = _S16.primal_0;
    (*dpMax_0).differential_0 = _S14;
    return;
}

struct DiffPair_vectorx3Cfloatx2C2x3E_0
{
     primal_0 : vec2<f32>,
     differential_0 : vec2<f32>,
};

fn _d_clamp_vector_0( dpx_2 : ptr<function, DiffPair_vectorx3Cfloatx2C2x3E_0>,  dpy_1 : ptr<function, DiffPair_vectorx3Cfloatx2C2x3E_0>,  dpz_0 : ptr<function, DiffPair_vectorx3Cfloatx2C2x3E_0>,  dOut_3 : vec2<f32>)
{
    var left_d_result_0 : vec2<f32>;
    var middle_d_result_0 : vec2<f32>;
    var right_d_result_0 : vec2<f32>;
    var left_dp_0 : DiffPair_float_0;
    left_dp_0.primal_0 = (*dpx_2).primal_0[i32(0)];
    left_dp_0.differential_0 = 0.0f;
    var middle_dp_0 : DiffPair_float_0;
    middle_dp_0.primal_0 = (*dpy_1).primal_0[i32(0)];
    middle_dp_0.differential_0 = 0.0f;
    var right_dp_0 : DiffPair_float_0;
    right_dp_0.primal_0 = (*dpz_0).primal_0[i32(0)];
    right_dp_0.differential_0 = 0.0f;
    _d_clamp_0(&(left_dp_0), &(middle_dp_0), &(right_dp_0), dOut_3[i32(0)]);
    left_d_result_0[i32(0)] = left_dp_0.differential_0;
    middle_d_result_0[i32(0)] = middle_dp_0.differential_0;
    right_d_result_0[i32(0)] = right_dp_0.differential_0;
    var left_dp_1 : DiffPair_float_0;
    left_dp_1.primal_0 = (*dpx_2).primal_0[i32(1)];
    left_dp_1.differential_0 = 0.0f;
    var middle_dp_1 : DiffPair_float_0;
    middle_dp_1.primal_0 = (*dpy_1).primal_0[i32(1)];
    middle_dp_1.differential_0 = 0.0f;
    var right_dp_1 : DiffPair_float_0;
    right_dp_1.primal_0 = (*dpz_0).primal_0[i32(1)];
    right_dp_1.differential_0 = 0.0f;
    _d_clamp_0(&(left_dp_1), &(middle_dp_1), &(right_dp_1), dOut_3[i32(1)]);
    left_d_result_0[i32(1)] = left_dp_1.differential_0;
    middle_d_result_0[i32(1)] = middle_dp_1.differential_0;
    right_d_result_0[i32(1)] = right_dp_1.differential_0;
    (*dpx_2).primal_0 = (*dpx_2).primal_0;
    (*dpx_2).differential_0 = left_d_result_0;
    (*dpy_1).primal_0 = (*dpy_1).primal_0;
    (*dpy_1).differential_0 = middle_d_result_0;
    (*dpz_0).primal_0 = (*dpz_0).primal_0;
    (*dpz_0).differential_0 = right_d_result_0;
    return;
}

struct DiffPair_vectorx3Cfloatx2C3x3E_0
{
     primal_0 : vec3<f32>,
     differential_0 : vec3<f32>,
};

fn _d_clamp_vector_1( dpx_3 : ptr<function, DiffPair_vectorx3Cfloatx2C3x3E_0>,  dpy_2 : ptr<function, DiffPair_vectorx3Cfloatx2C3x3E_0>,  dpz_1 : ptr<function, DiffPair_vectorx3Cfloatx2C3x3E_0>,  dOut_4 : vec3<f32>)
{
    var left_d_result_1 : vec3<f32>;
    var middle_d_result_1 : vec3<f32>;
    var right_d_result_1 : vec3<f32>;
    var left_dp_2 : DiffPair_float_0;
    left_dp_2.primal_0 = (*dpx_3).primal_0[i32(0)];
    left_dp_2.differential_0 = 0.0f;
    var middle_dp_2 : DiffPair_float_0;
    middle_dp_2.primal_0 = (*dpy_2).primal_0[i32(0)];
    middle_dp_2.differential_0 = 0.0f;
    var right_dp_2 : DiffPair_float_0;
    right_dp_2.primal_0 = (*dpz_1).primal_0[i32(0)];
    right_dp_2.differential_0 = 0.0f;
    _d_clamp_0(&(left_dp_2), &(middle_dp_2), &(right_dp_2), dOut_4[i32(0)]);
    left_d_result_1[i32(0)] = left_dp_2.differential_0;
    middle_d_result_1[i32(0)] = middle_dp_2.differential_0;
    right_d_result_1[i32(0)] = right_dp_2.differential_0;
    var left_dp_3 : DiffPair_float_0;
    left_dp_3.primal_0 = (*dpx_3).primal_0[i32(1)];
    left_dp_3.differential_0 = 0.0f;
    var middle_dp_3 : DiffPair_float_0;
    middle_dp_3.primal_0 = (*dpy_2).primal_0[i32(1)];
    middle_dp_3.differential_0 = 0.0f;
    var right_dp_3 : DiffPair_float_0;
    right_dp_3.primal_0 = (*dpz_1).primal_0[i32(1)];
    right_dp_3.differential_0 = 0.0f;
    _d_clamp_0(&(left_dp_3), &(middle_dp_3), &(right_dp_3), dOut_4[i32(1)]);
    left_d_result_1[i32(1)] = left_dp_3.differential_0;
    middle_d_result_1[i32(1)] = middle_dp_3.differential_0;
    right_d_result_1[i32(1)] = right_dp_3.differential_0;
    var left_dp_4 : DiffPair_float_0;
    left_dp_4.primal_0 = (*dpx_3).primal_0[i32(2)];
    left_dp_4.differential_0 = 0.0f;
    var middle_dp_4 : DiffPair_float_0;
    middle_dp_4.primal_0 = (*dpy_2).primal_0[i32(2)];
    middle_dp_4.differential_0 = 0.0f;
    var right_dp_4 : DiffPair_float_0;
    right_dp_4.primal_0 = (*dpz_1).primal_0[i32(2)];
    right_dp_4.differential_0 = 0.0f;
    _d_clamp_0(&(left_dp_4), &(middle_dp_4), &(right_dp_4), dOut_4[i32(2)]);
    left_d_result_1[i32(2)] = left_dp_4.differential_0;
    middle_d_result_1[i32(2)] = middle_dp_4.differential_0;
    right_d_result_1[i32(2)] = right_dp_4.differential_0;
    (*dpx_3).primal_0 = (*dpx_3).primal_0;
    (*dpx_3).differential_0 = left_d_result_1;
    (*dpy_2).primal_0 = (*dpy_2).primal_0;
    (*dpy_2).differential_0 = middle_d_result_1;
    (*dpz_1).primal_0 = (*dpz_1).primal_0;
    (*dpz_1).differential_0 = right_d_result_1;
    return;
}

fn smoothStep_0( x_0 : vec2<f32>,  minval_0 : vec2<f32>,  maxval_0 : vec2<f32>) -> vec2<f32>
{
    var y_0 : vec2<f32> = clamp((x_0 - minval_0) / (maxval_0 - minval_0), vec2<f32>(0.0f), vec2<f32>(1.0f));
    return y_0 * y_0 * (vec2<f32>(3.0f) - vec2<f32>(2.0f) * y_0);
}

fn smoothStep_1( x_1 : vec3<f32>,  minval_1 : vec3<f32>,  maxval_1 : vec3<f32>) -> vec3<f32>
{
    var y_1 : vec3<f32> = clamp((x_1 - minval_1) / (maxval_1 - minval_1), vec3<f32>(0.0f), vec3<f32>(1.0f));
    return y_1 * y_1 * (vec3<f32>(3.0f) - vec3<f32>(2.0f) * y_1);
}

fn smoothStep_2( x_2 : f32,  minval_2 : f32,  maxval_2 : f32) -> f32
{
    var y_2 : f32 = clamp((x_2 - minval_2) / (maxval_2 - minval_2), 0.0f, 1.0f);
    return y_2 * y_2 * (3.0f - 2.0f * y_2);
}

fn _d_sqrt_0( dpx_4 : ptr<function, DiffPair_float_0>,  dOut_5 : f32)
{
    var _S17 : f32 = 0.5f / sqrt(max(1.00000001168609742e-07f, (*dpx_4).primal_0)) * dOut_5;
    (*dpx_4).primal_0 = (*dpx_4).primal_0;
    (*dpx_4).differential_0 = _S17;
    return;
}

struct Gaussian2D_0
{
     center_2 : vec2<f32>,
     sigma_0 : mat2x2<f32>,
     color_0 : vec3<f32>,
     opacity_0 : f32,
};

fn Gaussian2D_load_0( idx_3 : u32,  localIdx_0 : u32) -> Gaussian2D_0
{
    var total_0 : u32 = Gaussian2D_count_0();
    var gaussian_0 : Gaussian2D_0;
    gaussian_0.center_2 = smoothStep_0(vec2<f32>(loadFloat_0(idx_3, localIdx_0), loadFloat_0(total_0 + idx_3, localIdx_0)), vec2<f32>(0.0f, 0.0f), vec2<f32>(1.0f, 1.0f));
    gaussian_0.sigma_0[i32(0)][i32(0)] = smoothStep_2(loadFloat_0(total_0 * u32(2) + idx_3, localIdx_0) * 0.80000001192092896f, 0.0f, 1.0f) + 0.00499999988824129f;
    var _S18 : f32 = smoothStep_2(loadFloat_0(total_0 * u32(3) + idx_3, localIdx_0) * 0.80000001192092896f, 0.0f, 1.0f) + 0.00499999988824129f;
    gaussian_0.sigma_0[i32(1)][i32(1)] = _S18;
    var aniso_0 : f32 = (smoothStep_2(loadFloat_0(total_0 * u32(4) + idx_3, localIdx_0) * 0.60000002384185791f, 0.0f, 1.0f) - 0.5f) * 1.64999997615814209f;
    gaussian_0.sigma_0[i32(0)][i32(1)] = sqrt(gaussian_0.sigma_0[i32(0)][i32(0)] * _S18) * aniso_0;
    gaussian_0.sigma_0[i32(1)][i32(0)] = sqrt(gaussian_0.sigma_0[i32(0)][i32(0)] * gaussian_0.sigma_0[i32(1)][i32(1)]) * aniso_0;
    gaussian_0.color_0 = smoothStep_1(vec3<f32>(loadFloat_0(total_0 * u32(5) + idx_3, localIdx_0) * 0.80000001192092896f, loadFloat_0(total_0 * u32(6) + idx_3, localIdx_0) * 0.80000001192092896f, loadFloat_0(total_0 * u32(7) + idx_3, localIdx_0) * 0.80000001192092896f), vec3<f32>(0.0f, 0.0f, 0.0f), vec3<f32>(1.0f, 1.0f, 1.0f));
    gaussian_0.opacity_0 = smoothStep_2(loadFloat_0(total_0 * u32(8) + idx_3, localIdx_0) * 0.89999997615814209f + 0.10000000149011612f, 0.0f, 1.0f);
    var _S19 : mat2x2<f32> = mat2x2<f32>(0.00009999999747379f, 0.00009999999747379f, 0.00009999999747379f, 0.00009999999747379f);
    var _S20 : mat2x2<f32> = gaussian_0.sigma_0;
    gaussian_0.sigma_0 = mat2x2<f32>(_S20[0] * _S19[0], _S20[1] * _S19[1]);
    return gaussian_0;
}

fn _d_dot_0( dpx_5 : ptr<function, DiffPair_vectorx3Cfloatx2C3x3E_0>,  dpy_3 : ptr<function, DiffPair_vectorx3Cfloatx2C3x3E_0>,  dOut_6 : f32)
{
    var x_d_result_0 : vec3<f32>;
    var y_d_result_0 : vec3<f32>;
    x_d_result_0[i32(0)] = (*dpy_3).primal_0[i32(0)] * dOut_6;
    y_d_result_0[i32(0)] = (*dpx_5).primal_0[i32(0)] * dOut_6;
    x_d_result_0[i32(1)] = (*dpy_3).primal_0[i32(1)] * dOut_6;
    y_d_result_0[i32(1)] = (*dpx_5).primal_0[i32(1)] * dOut_6;
    x_d_result_0[i32(2)] = (*dpy_3).primal_0[i32(2)] * dOut_6;
    y_d_result_0[i32(2)] = (*dpx_5).primal_0[i32(2)] * dOut_6;
    (*dpx_5).primal_0 = (*dpx_5).primal_0;
    (*dpx_5).differential_0 = x_d_result_0;
    (*dpy_3).primal_0 = (*dpy_3).primal_0;
    (*dpy_3).differential_0 = y_d_result_0;
    return;
}

fn Gaussian2D_bounds_0( this_0 : Gaussian2D_0) -> OBB_0
{
    var _S21 : bool;
    if((abs(this_0.sigma_0[i32(0)][i32(1)])) < 9.99999997475242708e-07f)
    {
        _S21 = true;
    }
    else
    {
        _S21 = (abs(this_0.sigma_0[i32(1)][i32(0)])) < 9.99999997475242708e-07f;
    }
    if(_S21)
    {
        return OBB_x24init_0(this_0.center_2, mat2x2<f32>(vec2<f32>(1.0f, 0.0f), vec2<f32>(0.0f, 1.0f)), vec2<f32>(sqrt(this_0.sigma_0[i32(0)][i32(0)]), sqrt(this_0.sigma_0[i32(1)][i32(1)])) * vec2<f32>(4.0f));
    }
    else
    {
        var trace_0 : f32 = this_0.sigma_0[i32(0)][i32(0)] + this_0.sigma_0[i32(1)][i32(1)];
        var _S22 : f32 = sqrt(trace_0 * trace_0 - 4.0f * (this_0.sigma_0[i32(0)][i32(0)] * this_0.sigma_0[i32(1)][i32(1)] - this_0.sigma_0[i32(0)][i32(1)] * this_0.sigma_0[i32(1)][i32(0)]));
        var lambda1_0 : f32 = 0.5f * (trace_0 + _S22);
        var lambda2_0 : f32 = 0.5f * (trace_0 - _S22);
        var eigenvectors_0 : mat2x2<f32>;
        var _S23 : vec2<f32> = vec2<f32>(lambda1_0 - this_0.sigma_0[i32(1)][i32(1)], this_0.sigma_0[i32(1)][i32(0)]);
        eigenvectors_0[i32(0)] = _S23 / vec2<f32>(length(_S23));
        var _S24 : vec2<f32> = vec2<f32>(this_0.sigma_0[i32(0)][i32(1)], lambda2_0 - this_0.sigma_0[i32(0)][i32(0)]);
        eigenvectors_0[i32(1)] = _S24 / vec2<f32>(length(_S24));
        return OBB_x24init_0(this_0.center_2, eigenvectors_0, vec2<f32>(sqrt(lambda1_0), sqrt(lambda2_0)) * vec2<f32>(4.0f));
    }
}

fn inverse_0( mat_0 : mat2x2<f32>) -> mat2x2<f32>
{
    var det_0 : f32 = determinant(mat_0);
    var output_0 : mat2x2<f32>;
    output_0[i32(0)][i32(0)] = mat_0[i32(1)][i32(1)] / det_0;
    output_0[i32(0)][i32(1)] = - mat_0[i32(0)][i32(1)] / det_0;
    output_0[i32(1)][i32(0)] = - mat_0[i32(1)][i32(0)] / det_0;
    output_0[i32(1)][i32(1)] = mat_0[i32(0)][i32(0)] / det_0;
    return output_0;
}

fn OBB_arePtsSeparatedAlongAxes_0( pts_0 : array<vec2<f32>, i32(4)>,  otherPts_0 : array<vec2<f32>, i32(4)>,  axes_0 : mat2x2<f32>) -> bool
{
    var i_0 : i32 = i32(0);
    for(;;)
    {
        if(i_0 < i32(2))
        {
        }
        else
        {
            break;
        }
        var _S25 : i32 = i_0;
        var _S26 : f32 = dot(pts_0[i32(0)], axes_0[i_0]);
        var proj_0 : vec2<f32> = vec2<f32>(_S26, _S26);
        var _S27 : f32 = dot(otherPts_0[i32(0)], axes_0[i_0]);
        var otherProj_0 : vec2<f32> = vec2<f32>(_S27, _S27);
        var j_0 : i32 = i32(1);
        for(;;)
        {
            if(j_0 < i32(4))
            {
            }
            else
            {
                break;
            }
            var _S28 : f32 = dot(pts_0[j_0], axes_0[_S25]);
            proj_0[i32(0)] = min(proj_0.x, _S28);
            proj_0[i32(1)] = max(proj_0.y, _S28);
            var _S29 : f32 = dot(otherPts_0[j_0], axes_0[_S25]);
            otherProj_0[i32(0)] = min(otherProj_0.x, _S29);
            otherProj_0[i32(1)] = max(otherProj_0.y, _S29);
            j_0 = j_0 + i32(1);
        }
        var _S30 : bool;
        if((proj_0.y) < (otherProj_0.x))
        {
            _S30 = true;
        }
        else
        {
            _S30 = (otherProj_0.y) < (proj_0.x);
        }
        if(_S30)
        {
            return true;
        }
        i_0 = i_0 + i32(1);
    }
    return false;
}

fn OBB_intersects_0( this_1 : OBB_0,  other_0 : OBB_0) -> bool
{
    var _S31 : array<vec2<f32>, i32(4)> = array<vec2<f32>, i32(4)>( vec2<f32>(-1.0f, -1.0f), vec2<f32>(1.0f, -1.0f), vec2<f32>(1.0f, 1.0f), vec2<f32>(-1.0f, 1.0f) );
    var _S32 : mat2x2<f32> = inverse_0(this_1.rotation_0);
    var _S33 : mat2x2<f32> = inverse_0(other_0.rotation_0);
    var pts_1 : array<vec2<f32>, i32(4)>;
    var i_1 : i32 = i32(0);
    for(;;)
    {
        if(i_1 < i32(4))
        {
        }
        else
        {
            break;
        }
        var _S34 : vec2<f32> = _S31[i_1] * this_1.scale_0;
        pts_1[i_1] = this_1.center_0 + vec2<f32>(dot(_S32[i32(0)], _S34), dot(_S32[i32(1)], _S34));
        i_1 = i_1 + i32(1);
    }
    var otherPts_1 : array<vec2<f32>, i32(4)>;
    i_1 = i32(0);
    for(;;)
    {
        if(i_1 < i32(4))
        {
        }
        else
        {
            break;
        }
        var _S35 : vec2<f32> = _S31[i_1] * other_0.scale_0;
        otherPts_1[i_1] = other_0.center_0 + vec2<f32>(dot(_S33[i32(0)], _S35), dot(_S33[i32(1)], _S35));
        i_1 = i_1 + i32(1);
    }
    var _S36 : bool;
    if(OBB_arePtsSeparatedAlongAxes_0(pts_1, otherPts_1, this_1.rotation_0))
    {
        _S36 = true;
    }
    else
    {
        _S36 = OBB_arePtsSeparatedAlongAxes_0(pts_1, otherPts_1, other_0.rotation_0);
    }
    return !_S36;
}

var<workgroup> blobs_0 : array<u32, i32(256)>;

struct FilledShortList_0
{
     _dummy_1 : i32,
};

fn coarseRasterize_0( sList_0 : InitializedShortList_0,  tileBounds_0 : OBB_0,  localIdx_1 : u32) -> FilledShortList_0
{
    workgroupBarrier();
    var _S37 : u32 = Gaussian2D_count_0();
    var _S38 : FilledShortList_0 = FilledShortList_0( i32(0) );
    var i_2 : u32 = localIdx_1;
    for(;;)
    {
        if(i_2 < _S37)
        {
        }
        else
        {
            break;
        }
        if(OBB_intersects_0(Gaussian2D_bounds_0(Gaussian2D_load_0(i_2, localIdx_1)), tileBounds_0))
        {
            var _S39 : u32 = atomicAdd(&(blobCountAT_0), u32(1));
            blobs_0[_S39] = i_2;
        }
        i_2 = i_2 + u32(256);
    }
    workgroupBarrier();
    var _S40 : u32 = atomicLoad(&(blobCountAT_0));
    blobCount_0 = _S40;
    return _S38;
}

struct PaddedShortList_0
{
     _dummy_2 : i32,
};

fn padBuffer_0( SLANG_anonymous_0_0 : FilledShortList_0,  localIdx_2 : u32) -> PaddedShortList_0
{
    workgroupBarrier();
    var _S41 : u32 = blobCount_0;
    var _S42 : PaddedShortList_0 = PaddedShortList_0( i32(0) );
    var i_3 : u32 = localIdx_2;
    for(;;)
    {
        if(i_3 < u32(256))
        {
        }
        else
        {
            break;
        }
        if(i_3 >= _S41)
        {
            blobs_0[i_3] = u32(4294967295);
        }
        i_3 = i_3 + u32(256);
    }
    return _S42;
}

struct SortedShortList_0
{
     _dummy_3 : i32,
};

fn bitonicSort_0( SLANG_anonymous_1_0 : PaddedShortList_0,  localIdx_3 : u32) -> SortedShortList_0
{
    workgroupBarrier();
    var _S43 : SortedShortList_0 = SortedShortList_0( i32(0) );
    var k_0 : u32 = u32(2);
    for(;;)
    {
        if(k_0 <= u32(256))
        {
        }
        else
        {
            break;
        }
        var j_1 : u32 = k_0 / u32(2);
        for(;;)
        {
            if(j_1 > u32(0))
            {
            }
            else
            {
                break;
            }
            var i_4 : u32 = localIdx_3;
            for(;;)
            {
                if(i_4 < u32(256))
                {
                }
                else
                {
                    break;
                }
                var l_0 : u32 = (i_4 ^ (j_1));
                if(l_0 > i_4)
                {
                    var _S44 : u32 = (i_4 & (k_0));
                    var _S45 : bool;
                    if(_S44 == u32(0))
                    {
                        _S45 = (blobs_0[i_4]) > (blobs_0[l_0]);
                    }
                    else
                    {
                        _S45 = false;
                    }
                    var _S46 : bool;
                    if(_S45)
                    {
                        _S46 = true;
                    }
                    else
                    {
                        if(_S44 != u32(0))
                        {
                            _S46 = (blobs_0[i_4]) < (blobs_0[l_0]);
                        }
                        else
                        {
                            _S46 = false;
                        }
                    }
                    if(_S46)
                    {
                        var temp_0 : u32 = blobs_0[i_4];
                        blobs_0[i_4] = blobs_0[l_0];
                        blobs_0[l_0] = temp_0;
                    }
                }
                i_4 = i_4 + u32(256);
            }
            workgroupBarrier();
            j_1 = j_1 / u32(2);
        }
        k_0 = k_0 * u32(2);
    }
    return _S43;
}

var<workgroup> finalVal_0 : array<vec4<f32>, i32(256)>;

var<workgroup> maxCount_0 : array<u32, i32(256)>;

fn _d_exp_0( dpx_6 : ptr<function, DiffPair_float_0>,  dOut_7 : f32)
{
    var _S47 : f32 = exp((*dpx_6).primal_0) * dOut_7;
    (*dpx_6).primal_0 = (*dpx_6).primal_0;
    (*dpx_6).differential_0 = _S47;
    return;
}

fn Gaussian2D_eval_0( this_2 : Gaussian2D_0,  uv_0 : vec2<f32>) -> vec4<f32>
{
    var invCov_0 : mat2x2<f32> = inverse_0(this_2.sigma_0);
    var diff_0 : vec2<f32> = uv_0 - this_2.center_2;
    var _S48 : f32 = diff_0.x;
    var _S49 : f32 = diff_0.y;
    return vec4<f32>(this_2.color_0, min(0.99000000953674316f, this_2.opacity_0 * exp(-0.5f * (_S48 * _S48 * invCov_0[i32(0)][i32(0)] + _S49 * _S49 * invCov_0[i32(1)][i32(1)] + _S48 * _S49 * invCov_0[i32(0)][i32(1)] + _S49 * _S48 * invCov_0[i32(1)][i32(0)]))));
}

fn eval_0( blob_id_0 : u32,  uv_1 : vec2<f32>,  localIdx_4 : u32) -> vec4<f32>
{
    return Gaussian2D_eval_0(Gaussian2D_load_0(blob_id_0, localIdx_4), uv_1);
}

fn preMult_0( pixel_0 : vec4<f32>) -> vec4<f32>
{
    var _S50 : f32 = pixel_0.w;
    return vec4<f32>(pixel_0.xyz * vec3<f32>(_S50), _S50);
}

fn undoAlphaBlend_0( pixel_1 : vec4<f32>,  gval_0 : vec4<f32>) -> vec4<f32>
{
    var _S51 : vec4<f32> = preMult_0(gval_0);
    var oldPixelAlpha_0 : f32 = pixel_1.w / (1.0f - _S51.w);
    return vec4<f32>(pixel_1.xyz - _S51.xyz * vec3<f32>(oldPixelAlpha_0), oldPixelAlpha_0);
}

struct PixelState_0
{
     value_0 : vec4<f32>,
     finalCount_0 : u32,
};

fn undoPixelState_0( nextState_0 : PixelState_0,  index_0 : u32,  gval_1 : vec4<f32>) -> PixelState_0
{
    if(index_0 > (nextState_0.finalCount_0))
    {
        var _S52 : PixelState_0 = PixelState_0( nextState_0.value_0, nextState_0.finalCount_0 );
        return _S52;
    }
    var _S53 : PixelState_0 = PixelState_0( undoAlphaBlend_0(nextState_0.value_0, gval_1), nextState_0.finalCount_0 - u32(1) );
    return _S53;
}

struct PixelState_Differential_0
{
     value_1 : vec4<f32>,
};

fn PixelState_x24_syn_dzero_0() -> PixelState_Differential_0
{
    var result_0 : PixelState_Differential_0;
    result_0.value_1 = vec4<f32>(0.0f);
    return result_0;
}

fn PixelState_x24_syn_dadd_0( SLANG_anonymous_0_1 : PixelState_Differential_0,  SLANG_anonymous_1_1 : PixelState_Differential_0) -> PixelState_Differential_0
{
    var result_1 : PixelState_Differential_0;
    result_1.value_1 = SLANG_anonymous_0_1.value_1 + SLANG_anonymous_1_1.value_1;
    return result_1;
}

fn alphaBlend_0( pixel_2 : vec4<f32>,  gval_2 : vec4<f32>) -> vec4<f32>
{
    var _S54 : vec4<f32> = preMult_0(gval_2);
    var _S55 : f32 = pixel_2.w;
    return vec4<f32>(pixel_2.xyz + _S54.xyz * vec3<f32>(_S55), _S55 * (1.0f - _S54.w));
}

fn transformPixelState_0( pixel_3 : PixelState_0,  gval_3 : vec4<f32>) -> PixelState_0
{
    var newState_0 : vec4<f32> = alphaBlend_0(pixel_3.value_0, gval_3);
    if((pixel_3.value_0.w) < 0.00392156885936856f)
    {
        var _S56 : PixelState_0 = PixelState_0( pixel_3.value_0, pixel_3.finalCount_0 );
        return _S56;
    }
    var _S57 : PixelState_0 = PixelState_0( newState_0, pixel_3.finalCount_0 + u32(1) );
    return _S57;
}

struct DiffPair_PixelState_0
{
     primal_0 : PixelState_0,
     differential_0 : PixelState_Differential_0,
};

struct DiffPair_vectorx3Cfloatx2C4x3E_0
{
     primal_0 : vec4<f32>,
     differential_0 : vec4<f32>,
};

fn s_primal_ctx_preMult_0( dppixel_0 : vec4<f32>) -> vec4<f32>
{
    var _S58 : f32 = dppixel_0.w;
    return vec4<f32>(dppixel_0.xyz * vec3<f32>(_S58), _S58);
}

fn s_bwd_prop_preMult_0( dppixel_1 : ptr<function, DiffPair_vectorx3Cfloatx2C4x3E_0>,  _s_dOut_0 : vec4<f32>)
{
    var _S59 : vec3<f32> = _s_dOut_0.xyz;
    var _S60 : vec3<f32> = (*dppixel_1).primal_0.xyz * _S59;
    var _S61 : vec3<f32> = vec3<f32>((*dppixel_1).primal_0.w) * _S59;
    var _S62 : vec4<f32> = vec4<f32>(_S61[i32(0)], _S61[i32(1)], _S61[i32(2)], _s_dOut_0[i32(3)] + _S60[i32(0)] + _S60[i32(1)] + _S60[i32(2)]);
    (*dppixel_1).primal_0 = (*dppixel_1).primal_0;
    (*dppixel_1).differential_0 = _S62;
    return;
}

fn s_bwd_prop_alphaBlend_0( dppixel_2 : ptr<function, DiffPair_vectorx3Cfloatx2C4x3E_0>,  dpgval_0 : ptr<function, DiffPair_vectorx3Cfloatx2C4x3E_0>,  _s_dOut_1 : vec4<f32>)
{
    var _S63 : vec4<f32> = s_primal_ctx_preMult_0((*dpgval_0).primal_0);
    var _S64 : f32 = (*dppixel_2).primal_0.w;
    var _S65 : vec3<f32> = _s_dOut_1.xyz;
    var _S66 : vec3<f32> = _S63.xyz * _S65;
    var _S67 : vec3<f32> = vec3<f32>(_S64) * _S65;
    var _S68 : f32 = (1.0f - _S63.w) * _s_dOut_1[i32(3)] + _S66[i32(0)] + _S66[i32(1)] + _S66[i32(2)];
    var _S69 : vec4<f32> = vec4<f32>(_S67[i32(0)], _S67[i32(1)], _S67[i32(2)], - (_S64 * _s_dOut_1[i32(3)]));
    var _S70 : vec4<f32> = vec4<f32>(0.0f);
    var _S71 : DiffPair_vectorx3Cfloatx2C4x3E_0;
    _S71.primal_0 = (*dpgval_0).primal_0;
    _S71.differential_0 = _S70;
    s_bwd_prop_preMult_0(&(_S71), _S69);
    (*dpgval_0).primal_0 = (*dpgval_0).primal_0;
    (*dpgval_0).differential_0 = _S71.differential_0;
    var _S72 : vec4<f32> = vec4<f32>(_S65[i32(0)], _S65[i32(1)], _S65[i32(2)], _S68);
    (*dppixel_2).primal_0 = (*dppixel_2).primal_0;
    (*dppixel_2).differential_0 = _S72;
    return;
}

fn s_bwd_prop_transformPixelState_0( dppixel_3 : ptr<function, DiffPair_PixelState_0>,  dpgval_1 : ptr<function, DiffPair_vectorx3Cfloatx2C4x3E_0>,  _s_dOut_2 : PixelState_Differential_0)
{
    var _S73 : DiffPair_PixelState_0 = (*dppixel_3);
    var _S74 : DiffPair_vectorx3Cfloatx2C4x3E_0 = (*dpgval_1);
    var _S75 : bool = ((*dppixel_3).primal_0.value_0.w) < 0.00392156885936856f;
    var _S76 : vec4<f32> = vec4<f32>(0.0f);
    var _S77 : PixelState_Differential_0 = PixelState_x24_syn_dzero_0();
    var _S78 : PixelState_Differential_0 = PixelState_x24_syn_dadd_0(_s_dOut_2, _S77);
    var _S79 : PixelState_Differential_0;
    var _S80 : vec4<f32>;
    if(!_S75)
    {
        _S79 = _S77;
        _S80 = _S78.value_1;
    }
    else
    {
        _S79 = PixelState_x24_syn_dadd_0(_S78, _S77);
        _S80 = _S76;
    }
    var _S81 : vec4<f32>;
    if(_S75)
    {
        _S81 = _S79.value_1;
    }
    else
    {
        _S81 = _S76;
    }
    var _S82 : DiffPair_vectorx3Cfloatx2C4x3E_0;
    _S82.primal_0 = _S73.primal_0.value_0;
    _S82.differential_0 = _S76;
    var _S83 : DiffPair_vectorx3Cfloatx2C4x3E_0;
    _S83.primal_0 = _S74.primal_0;
    _S83.differential_0 = _S76;
    s_bwd_prop_alphaBlend_0(&(_S82), &(_S83), _S80);
    var _S84 : vec4<f32> = _S82.differential_0 + _S81;
    (*dpgval_1).primal_0 = (*dpgval_1).primal_0;
    (*dpgval_1).differential_0 = _S83.differential_0;
    var _S85 : PixelState_Differential_0 = _S77;
    _S85.value_1 = _S84;
    (*dppixel_3).primal_0 = (*dppixel_3).primal_0;
    (*dppixel_3).differential_0 = _S85;
    return;
}

fn s_bwd_transformPixelState_0( _S86 : ptr<function, DiffPair_PixelState_0>,  _S87 : ptr<function, DiffPair_vectorx3Cfloatx2C4x3E_0>,  _S88 : PixelState_Differential_0)
{
    s_bwd_prop_transformPixelState_0(&((*_S86)), &((*_S87)), _S88);
    return;
}

struct s_bwd_prop_Gaussian2D_load_Intermediates_0
{
     _S89 : f32,
     _S90 : f32,
     _S91 : f32,
     _S92 : f32,
     _S93 : f32,
     _S94 : f32,
     _S95 : f32,
     _S96 : f32,
     _S97 : f32,
};

struct s_bwd_prop_eval_Intermediates_0
{
     _S98 : s_bwd_prop_Gaussian2D_load_Intermediates_0,
     _S99 : Gaussian2D_0,
};

fn s_primal_ctx_loadFloat_0( _S100 : u32,  _S101 : u32) -> f32
{
    return loadFloat_0(_S100, _S101);
}

fn s_primal_ctx_clamp_0( _S102 : vec2<f32>,  _S103 : vec2<f32>,  _S104 : vec2<f32>) -> vec2<f32>
{
    return clamp(_S102, _S103, _S104);
}

fn s_primal_ctx_smoothStep_0( dpx_7 : vec2<f32>,  dpminval_0 : vec2<f32>,  dpmaxval_0 : vec2<f32>) -> vec2<f32>
{
    var _S105 : vec2<f32> = s_primal_ctx_clamp_0((dpx_7 - dpminval_0) / (dpmaxval_0 - dpminval_0), vec2<f32>(0.0f), vec2<f32>(1.0f));
    return _S105 * _S105 * (vec2<f32>(3.0f) - vec2<f32>(2.0f) * _S105);
}

fn s_primal_ctx_clamp_1( _S106 : f32,  _S107 : f32,  _S108 : f32) -> f32
{
    return clamp(_S106, _S107, _S108);
}

fn s_primal_ctx_smoothStep_1( dpx_8 : f32,  dpminval_1 : f32,  dpmaxval_1 : f32) -> f32
{
    var _S109 : f32 = s_primal_ctx_clamp_1((dpx_8 - dpminval_1) / (dpmaxval_1 - dpminval_1), 0.0f, 1.0f);
    return _S109 * _S109 * (3.0f - 2.0f * _S109);
}

fn s_primal_ctx_sqrt_0( _S110 : f32) -> f32
{
    return sqrt(_S110);
}

fn s_primal_ctx_clamp_2( _S111 : vec3<f32>,  _S112 : vec3<f32>,  _S113 : vec3<f32>) -> vec3<f32>
{
    return clamp(_S111, _S112, _S113);
}

fn s_primal_ctx_smoothStep_2( dpx_9 : vec3<f32>,  dpminval_2 : vec3<f32>,  dpmaxval_2 : vec3<f32>) -> vec3<f32>
{
    var _S114 : vec3<f32> = s_primal_ctx_clamp_2((dpx_9 - dpminval_2) / (dpmaxval_2 - dpminval_2), vec3<f32>(0.0f), vec3<f32>(1.0f));
    return _S114 * _S114 * (vec3<f32>(3.0f) - vec3<f32>(2.0f) * _S114);
}

fn s_primal_ctx_Gaussian2D_load_0( idx_4 : u32,  localIdx_5 : u32,  _s_diff_ctx_0 : ptr<function, s_bwd_prop_Gaussian2D_load_Intermediates_0>) -> Gaussian2D_0
{
    (*_s_diff_ctx_0)._S89 = 0.0f;
    (*_s_diff_ctx_0)._S90 = 0.0f;
    (*_s_diff_ctx_0)._S91 = 0.0f;
    (*_s_diff_ctx_0)._S92 = 0.0f;
    (*_s_diff_ctx_0)._S93 = 0.0f;
    (*_s_diff_ctx_0)._S94 = 0.0f;
    (*_s_diff_ctx_0)._S95 = 0.0f;
    (*_s_diff_ctx_0)._S96 = 0.0f;
    (*_s_diff_ctx_0)._S97 = 0.0f;
    (*_s_diff_ctx_0)._S89 = 0.0f;
    (*_s_diff_ctx_0)._S90 = 0.0f;
    (*_s_diff_ctx_0)._S91 = 0.0f;
    (*_s_diff_ctx_0)._S92 = 0.0f;
    (*_s_diff_ctx_0)._S93 = 0.0f;
    (*_s_diff_ctx_0)._S94 = 0.0f;
    (*_s_diff_ctx_0)._S95 = 0.0f;
    (*_s_diff_ctx_0)._S96 = 0.0f;
    (*_s_diff_ctx_0)._S97 = 0.0f;
    var total_1 : u32 = Gaussian2D_count_0();
    var _S115 : mat2x2<f32> = mat2x2<f32>(0.0f, 0.0f, 0.0f, 0.0f);
    var _S116 : vec3<f32> = vec3<f32>(0.0f);
    var _S117 : f32 = s_primal_ctx_loadFloat_0(idx_4, localIdx_5);
    (*_s_diff_ctx_0)._S89 = _S117;
    var _S118 : f32 = s_primal_ctx_loadFloat_0(total_1 + idx_4, localIdx_5);
    (*_s_diff_ctx_0)._S90 = _S118;
    var _S119 : vec2<f32> = s_primal_ctx_smoothStep_0(vec2<f32>(_S117, _S118), vec2<f32>(0.0f, 0.0f), vec2<f32>(1.0f, 1.0f));
    var _S120 : f32 = s_primal_ctx_loadFloat_0(total_1 * u32(2) + idx_4, localIdx_5);
    (*_s_diff_ctx_0)._S91 = _S120;
    var _S121 : f32 = s_primal_ctx_smoothStep_1(_S120 * 0.80000001192092896f, 0.0f, 1.0f) + 0.00499999988824129f;
    var _S122 : Gaussian2D_0;
    _S122.center_2 = _S119;
    _S122.sigma_0 = _S115;
    _S122.color_0 = _S116;
    _S122.opacity_0 = 0.0f;
    _S122.sigma_0[i32(0)][i32(0)] = _S121;
    var _S123 : f32 = s_primal_ctx_loadFloat_0(total_1 * u32(3) + idx_4, localIdx_5);
    (*_s_diff_ctx_0)._S92 = _S123;
    var _S124 : f32 = s_primal_ctx_smoothStep_1(_S123 * 0.80000001192092896f, 0.0f, 1.0f) + 0.00499999988824129f;
    _S122.sigma_0[i32(1)][i32(1)] = _S124;
    var _S125 : f32 = s_primal_ctx_loadFloat_0(total_1 * u32(4) + idx_4, localIdx_5);
    (*_s_diff_ctx_0)._S93 = _S125;
    var _S126 : f32 = s_primal_ctx_sqrt_0(_S121 * _S124) * ((s_primal_ctx_smoothStep_1(_S125 * 0.60000002384185791f, 0.0f, 1.0f) - 0.5f) * 1.64999997615814209f);
    _S122.sigma_0[i32(0)][i32(1)] = _S126;
    _S122.sigma_0[i32(1)][i32(0)] = _S126;
    var _S127 : f32 = s_primal_ctx_loadFloat_0(total_1 * u32(5) + idx_4, localIdx_5);
    (*_s_diff_ctx_0)._S94 = _S127;
    var _S128 : f32 = _S127 * 0.80000001192092896f;
    var _S129 : f32 = s_primal_ctx_loadFloat_0(total_1 * u32(6) + idx_4, localIdx_5);
    (*_s_diff_ctx_0)._S95 = _S129;
    var _S130 : f32 = _S129 * 0.80000001192092896f;
    var _S131 : f32 = s_primal_ctx_loadFloat_0(total_1 * u32(7) + idx_4, localIdx_5);
    (*_s_diff_ctx_0)._S96 = _S131;
    var _S132 : vec3<f32> = s_primal_ctx_smoothStep_2(vec3<f32>(_S128, _S130, _S131 * 0.80000001192092896f), vec3<f32>(0.0f, 0.0f, 0.0f), vec3<f32>(1.0f, 1.0f, 1.0f));
    var _S133 : Gaussian2D_0 = _S122;
    _S133.color_0 = _S132;
    var _S134 : f32 = s_primal_ctx_loadFloat_0(total_1 * u32(8) + idx_4, localIdx_5);
    (*_s_diff_ctx_0)._S97 = _S134;
    _S133.opacity_0 = s_primal_ctx_smoothStep_1(_S134 * 0.89999997615814209f + 0.10000000149011612f, 0.0f, 1.0f);
    var _S135 : mat2x2<f32> = mat2x2<f32>(0.00009999999747379f, 0.00009999999747379f, 0.00009999999747379f, 0.00009999999747379f);
    var _S136 : mat2x2<f32> = _S122.sigma_0;
    _S133.sigma_0 = mat2x2<f32>(_S136[0] * _S135[0], _S136[1] * _S135[1]);
    return _S133;
}

struct s_bwd_prop_inverse_Intermediates_0
{
     _S137 : f32,
};

fn s_primal_ctx_inverse_0( dpmat_0 : mat2x2<f32>,  _s_diff_ctx_1 : ptr<function, s_bwd_prop_inverse_Intermediates_0>) -> mat2x2<f32>
{
    (*_s_diff_ctx_1)._S137 = 0.0f;
    var _S138 : mat2x2<f32> = mat2x2<f32>(0.0f, 0.0f, 0.0f, 0.0f);
    var _S139 : f32 = determinant(dpmat_0);
    (*_s_diff_ctx_1)._S137 = _S139;
    var _S140 : f32 = dpmat_0[i32(1)][i32(1)] / _S139;
    var _S141 : mat2x2<f32> = _S138;
    _S141[i32(0)][i32(0)] = _S140;
    _S141[i32(0)][i32(1)] = - dpmat_0[i32(0)][i32(1)] / _S139;
    _S141[i32(1)][i32(0)] = - dpmat_0[i32(1)][i32(0)] / _S139;
    _S141[i32(1)][i32(1)] = dpmat_0[i32(0)][i32(0)] / _S139;
    return _S141;
}

fn s_primal_ctx_exp_0( _S142 : f32) -> f32
{
    return exp(_S142);
}

fn s_primal_ctx_min_0( _S143 : f32,  _S144 : f32) -> f32
{
    return min(_S143, _S144);
}

fn s_primal_ctx_Gaussian2D_eval_0( dpthis_0 : Gaussian2D_0,  dpuv_0 : vec2<f32>) -> vec4<f32>
{
    var _S145 : s_bwd_prop_inverse_Intermediates_0;
    _S145._S137 = 0.0f;
    var _S146 : mat2x2<f32> = s_primal_ctx_inverse_0(dpthis_0.sigma_0, &(_S145));
    var diff_1 : vec2<f32> = dpuv_0 - dpthis_0.center_2;
    var _S147 : f32 = diff_1.x;
    var _S148 : f32 = diff_1.y;
    return vec4<f32>(dpthis_0.color_0, s_primal_ctx_min_0(0.99000000953674316f, dpthis_0.opacity_0 * s_primal_ctx_exp_0(-0.5f * (_S147 * _S147 * _S146[i32(0)][i32(0)] + _S148 * _S148 * _S146[i32(1)][i32(1)] + _S147 * _S148 * _S146[i32(0)][i32(1)] + _S148 * _S147 * _S146[i32(1)][i32(0)]))));
}

fn s_primal_ctx_eval_0( blob_id_1 : u32,  uv_2 : vec2<f32>,  localIdx_6 : u32,  _s_diff_ctx_2 : ptr<function, s_bwd_prop_eval_Intermediates_0>) -> vec4<f32>
{
    var _S149 : s_bwd_prop_Gaussian2D_load_Intermediates_0 = s_bwd_prop_Gaussian2D_load_Intermediates_0( 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f );
    var _S150 : vec2<f32> = vec2<f32>(0.0f);
    var _S151 : mat2x2<f32> = mat2x2<f32>(0.0f, 0.0f, 0.0f, 0.0f);
    var _S152 : vec3<f32> = vec3<f32>(0.0f);
    var _S153 : Gaussian2D_0 = Gaussian2D_0( _S150, _S151, _S152, 0.0f );
    (*_s_diff_ctx_2)._S98 = _S149;
    (*_s_diff_ctx_2)._S99 = _S153;
    (*_s_diff_ctx_2)._S99.center_2 = _S150;
    (*_s_diff_ctx_2)._S99.sigma_0 = _S151;
    (*_s_diff_ctx_2)._S99.color_0 = _S152;
    (*_s_diff_ctx_2)._S99.opacity_0 = 0.0f;
    var _S154 : s_bwd_prop_Gaussian2D_load_Intermediates_0 = (*_s_diff_ctx_2)._S98;
    var _S155 : Gaussian2D_0 = s_primal_ctx_Gaussian2D_load_0(blob_id_1, localIdx_6, &(_S154));
    (*_s_diff_ctx_2)._S98 = _S154;
    (*_s_diff_ctx_2)._S99 = _S155;
    return s_primal_ctx_Gaussian2D_eval_0(_S155, uv_2);
}

fn Gaussian2D_x24_syn_dzero_0() -> Gaussian2D_0
{
    var result_2 : Gaussian2D_0;
    result_2.center_2 = vec2<f32>(0.0f);
    result_2.sigma_0 = mat2x2<f32>(0.0f, 0.0f, 0.0f, 0.0f);
    result_2.color_0 = vec3<f32>(0.0f);
    result_2.opacity_0 = 0.0f;
    return result_2;
}

struct DiffPair_Gaussian2D_0
{
     primal_0 : Gaussian2D_0,
     differential_0 : Gaussian2D_0,
};

fn s_bwd_prop_min_0( _S156 : ptr<function, DiffPair_float_0>,  _S157 : ptr<function, DiffPair_float_0>,  _S158 : f32)
{
    _d_min_0(&((*_S156)), &((*_S157)), _S158);
    return;
}

fn s_bwd_prop_exp_0( _S159 : ptr<function, DiffPair_float_0>,  _S160 : f32)
{
    _d_exp_0(&((*_S159)), _S160);
    return;
}

struct DiffPair_matrixx3Cfloatx2C2x2C2x3E_0
{
     primal_0 : mat2x2<f32>,
     differential_0 : mat2x2<f32>,
};

fn s_bwd_prop_determinant_impl_0( dpm_0 : ptr<function, DiffPair_matrixx3Cfloatx2C2x2C2x3E_0>,  _s_dOut_3 : f32)
{
    var _S161 : f32 = - _s_dOut_3;
    var _S162 : f32 = (*dpm_0).primal_0[i32(0)][i32(1)] * _S161;
    var _S163 : f32 = (*dpm_0).primal_0[i32(1)][i32(0)] * _S161;
    var _S164 : f32 = (*dpm_0).primal_0[i32(0)][i32(0)] * _s_dOut_3;
    var _S165 : f32 = (*dpm_0).primal_0[i32(1)][i32(1)] * _s_dOut_3;
    var _S166 : vec2<f32> = vec2<f32>(0.0f);
    var _S167 : vec2<f32> = _S166;
    _S167[i32(0)] = _S162;
    _S167[i32(1)] = _S164;
    var _S168 : vec2<f32> = _S166;
    _S168[i32(1)] = _S163;
    _S168[i32(0)] = _S165;
    var _S169 : mat2x2<f32> = mat2x2<f32>(0.0f, 0.0f, 0.0f, 0.0f);
    _S169[i32(1)] = _S167;
    _S169[i32(0)] = _S168;
    (*dpm_0).primal_0 = (*dpm_0).primal_0;
    (*dpm_0).differential_0 = _S169;
    return;
}

fn s_bwd_determinant_impl_0( _S170 : ptr<function, DiffPair_matrixx3Cfloatx2C2x2C2x3E_0>,  _S171 : f32)
{
    s_bwd_prop_determinant_impl_0(&((*_S170)), _S171);
    return;
}

fn s_bwd_prop_inverse_0( dpmat_1 : ptr<function, DiffPair_matrixx3Cfloatx2C2x2C2x3E_0>,  _s_dOut_4 : mat2x2<f32>,  _s_diff_ctx_3 : s_bwd_prop_inverse_Intermediates_0)
{
    var _S172 : f32 = _s_diff_ctx_3._S137;
    var _S173 : f32 = _S172 * _S172;
    var _S174 : f32 = _s_dOut_4[i32(1)][i32(1)] / _S173;
    var _S175 : f32 = (*dpmat_1).primal_0[i32(0)][i32(0)] * - _S174;
    var _S176 : f32 = _s_diff_ctx_3._S137 * _S174;
    var _S177 : f32 = _s_dOut_4[i32(1)][i32(0)] / _S173;
    var _S178 : f32 = - (*dpmat_1).primal_0[i32(1)][i32(0)] * - _S177;
    var _S179 : f32 = - (_s_diff_ctx_3._S137 * _S177);
    var _S180 : f32 = _s_dOut_4[i32(0)][i32(1)] / _S173;
    var _S181 : f32 = - (*dpmat_1).primal_0[i32(0)][i32(1)] * - _S180;
    var _S182 : f32 = - (_s_diff_ctx_3._S137 * _S180);
    var _S183 : vec2<f32> = vec2<f32>(0.0f);
    var _S184 : vec2<f32> = _S183;
    _S184[i32(0)] = _S176;
    _S184[i32(1)] = _S182;
    var _S185 : f32 = _s_dOut_4[i32(0)][i32(0)] / _S173;
    var _S186 : f32 = (*dpmat_1).primal_0[i32(1)][i32(1)] * - _S185;
    var _S187 : f32 = _s_diff_ctx_3._S137 * _S185;
    var _S188 : vec2<f32> = _S183;
    _S188[i32(0)] = _S179;
    _S188[i32(1)] = _S187;
    var _S189 : f32 = _S175 + _S178 + _S181 + _S186;
    var _S190 : mat2x2<f32> = mat2x2<f32>(0.0f, 0.0f, 0.0f, 0.0f);
    var _S191 : DiffPair_matrixx3Cfloatx2C2x2C2x3E_0;
    _S191.primal_0 = (*dpmat_1).primal_0;
    _S191.differential_0 = _S190;
    s_bwd_determinant_impl_0(&(_S191), _S189);
    var _S192 : mat2x2<f32> = _S190;
    _S192[i32(0)] = _S184;
    _S192[i32(1)] = _S188;
    var _S193 : mat2x2<f32> = _S191.differential_0 + _S192;
    (*dpmat_1).primal_0 = (*dpmat_1).primal_0;
    (*dpmat_1).differential_0 = _S193;
    return;
}

fn s_bwd_prop_Gaussian2D_eval_0( dpthis_1 : ptr<function, DiffPair_Gaussian2D_0>,  dpuv_1 : ptr<function, DiffPair_vectorx3Cfloatx2C2x3E_0>,  _s_dOut_5 : vec4<f32>)
{
    var _S194 : s_bwd_prop_inverse_Intermediates_0;
    _S194._S137 = 0.0f;
    var _S195 : mat2x2<f32> = s_primal_ctx_inverse_0((*dpthis_1).primal_0.sigma_0, &(_S194));
    var diff_2 : vec2<f32> = (*dpuv_1).primal_0 - (*dpthis_1).primal_0.center_2;
    var _S196 : f32 = diff_2.x;
    var _S197 : f32 = _S196 * _S196;
    var _S198 : f32 = diff_2.y;
    var _S199 : f32 = _S198 * _S198;
    var _S200 : f32 = _S196 * _S198;
    var _S201 : f32 = _S198 * _S196;
    var power_0 : f32 = -0.5f * (_S197 * _S195[i32(0)][i32(0)] + _S199 * _S195[i32(1)][i32(1)] + _S200 * _S195[i32(0)][i32(1)] + _S201 * _S195[i32(1)][i32(0)]);
    var _S202 : f32 = s_primal_ctx_exp_0(power_0);
    var _S203 : f32 = (*dpthis_1).primal_0.opacity_0 * _S202;
    var _S204 : vec3<f32> = _s_dOut_5.xyz;
    var _S205 : DiffPair_float_0;
    _S205.primal_0 = 0.99000000953674316f;
    _S205.differential_0 = 0.0f;
    var _S206 : DiffPair_float_0;
    _S206.primal_0 = _S203;
    _S206.differential_0 = 0.0f;
    s_bwd_prop_min_0(&(_S205), &(_S206), _s_dOut_5[i32(3)]);
    var _S207 : f32 = (*dpthis_1).primal_0.opacity_0 * _S206.differential_0;
    var _S208 : f32 = _S202 * _S206.differential_0;
    var _S209 : DiffPair_float_0;
    _S209.primal_0 = power_0;
    _S209.differential_0 = 0.0f;
    s_bwd_prop_exp_0(&(_S209), _S207);
    var _S210 : f32 = -0.5f * _S209.differential_0;
    var _S211 : f32 = _S201 * _S210;
    var _S212 : f32 = _S195[i32(1)][i32(0)] * _S210;
    var _S213 : f32 = _S198 * _S212;
    var _S214 : f32 = _S196 * _S212;
    var _S215 : f32 = _S200 * _S210;
    var _S216 : f32 = _S195[i32(0)][i32(1)] * _S210;
    var _S217 : f32 = _S196 * _S216;
    var _S218 : f32 = _S198 * _S216;
    var _S219 : f32 = _S199 * _S210;
    var _S220 : f32 = _S195[i32(1)][i32(1)] * _S210;
    var _S221 : vec2<f32> = vec2<f32>(0.0f);
    var _S222 : vec2<f32> = _S221;
    _S222[i32(0)] = _S211;
    _S222[i32(1)] = _S219;
    var _S223 : f32 = _S198 * _S220;
    var _S224 : f32 = _S214 + _S217 + _S223 + _S223;
    var _S225 : f32 = _S197 * _S210;
    var _S226 : f32 = _S195[i32(0)][i32(0)] * _S210;
    var _S227 : vec2<f32> = _S221;
    _S227[i32(1)] = _S215;
    _S227[i32(0)] = _S225;
    var _S228 : f32 = _S196 * _S226;
    var s_diff_diff_T_0 : vec2<f32> = vec2<f32>(_S213 + _S218 + _S228 + _S228, _S224);
    var _S229 : vec2<f32> = - s_diff_diff_T_0;
    var _S230 : mat2x2<f32> = mat2x2<f32>(0.0f, 0.0f, 0.0f, 0.0f);
    var _S231 : mat2x2<f32> = _S230;
    _S231[i32(1)] = _S222;
    _S231[i32(0)] = _S227;
    var _S232 : DiffPair_matrixx3Cfloatx2C2x2C2x3E_0;
    _S232.primal_0 = (*dpthis_1).primal_0.sigma_0;
    _S232.differential_0 = _S230;
    s_bwd_prop_inverse_0(&(_S232), _S231, _S194);
    (*dpuv_1).primal_0 = (*dpuv_1).primal_0;
    (*dpuv_1).differential_0 = s_diff_diff_T_0;
    var _S233 : Gaussian2D_0 = Gaussian2D_x24_syn_dzero_0();
    _S233.color_0 = _S204;
    _S233.opacity_0 = _S208;
    _S233.center_2 = _S229;
    _S233.sigma_0 = _S232.differential_0;
    (*dpthis_1).primal_0 = (*dpthis_1).primal_0;
    (*dpthis_1).differential_0 = _S233;
    return;
}

fn s_bwd_prop_clamp_0( _S234 : ptr<function, DiffPair_float_0>,  _S235 : ptr<function, DiffPair_float_0>,  _S236 : ptr<function, DiffPair_float_0>,  _S237 : f32)
{
    _d_clamp_0(&((*_S234)), &((*_S235)), &((*_S236)), _S237);
    return;
}

fn s_bwd_prop_smoothStep_0( dpx_10 : ptr<function, DiffPair_float_0>,  dpminval_3 : ptr<function, DiffPair_float_0>,  dpmaxval_3 : ptr<function, DiffPair_float_0>,  _s_dOut_6 : f32)
{
    var _S238 : f32 = (*dpx_10).primal_0 - (*dpminval_3).primal_0;
    var _S239 : f32 = (*dpmaxval_3).primal_0 - (*dpminval_3).primal_0;
    var _S240 : f32 = _S238 / _S239;
    var _S241 : f32 = _S239 * _S239;
    var _S242 : f32 = s_primal_ctx_clamp_1(_S240, 0.0f, 1.0f);
    var _S243 : f32 = _S242 * ((3.0f - 2.0f * _S242) * _s_dOut_6);
    var _S244 : f32 = 2.0f * - (_S242 * _S242 * _s_dOut_6) + _S243 + _S243;
    var _S245 : DiffPair_float_0;
    _S245.primal_0 = _S240;
    _S245.differential_0 = 0.0f;
    var _S246 : DiffPair_float_0;
    _S246.primal_0 = 0.0f;
    _S246.differential_0 = 0.0f;
    var _S247 : DiffPair_float_0;
    _S247.primal_0 = 1.0f;
    _S247.differential_0 = 0.0f;
    s_bwd_prop_clamp_0(&(_S245), &(_S246), &(_S247), _S244);
    var _S248 : f32 = _S245.differential_0 / _S241;
    var _S249 : f32 = _S238 * - _S248;
    var _S250 : f32 = _S239 * _S248;
    var _S251 : f32 = - _S249;
    var _S252 : f32 = - _S250;
    (*dpmaxval_3).primal_0 = (*dpmaxval_3).primal_0;
    (*dpmaxval_3).differential_0 = _S249;
    var _S253 : f32 = _S251 + _S252;
    (*dpminval_3).primal_0 = (*dpminval_3).primal_0;
    (*dpminval_3).differential_0 = _S253;
    (*dpx_10).primal_0 = (*dpx_10).primal_0;
    (*dpx_10).differential_0 = _S250;
    return;
}

fn s_bwd_prop_loadFloat_0( _S254 : u32,  _S255 : u32,  _S256 : f32)
{
    loadFloat_bwd_0(_S254, _S255, _S256);
    return;
}

fn s_bwd_prop_clamp_1( _S257 : ptr<function, DiffPair_vectorx3Cfloatx2C3x3E_0>,  _S258 : ptr<function, DiffPair_vectorx3Cfloatx2C3x3E_0>,  _S259 : ptr<function, DiffPair_vectorx3Cfloatx2C3x3E_0>,  _S260 : vec3<f32>)
{
    _d_clamp_vector_1(&((*_S257)), &((*_S258)), &((*_S259)), _S260);
    return;
}

fn s_bwd_prop_smoothStep_1( dpx_11 : ptr<function, DiffPair_vectorx3Cfloatx2C3x3E_0>,  dpminval_4 : ptr<function, DiffPair_vectorx3Cfloatx2C3x3E_0>,  dpmaxval_4 : ptr<function, DiffPair_vectorx3Cfloatx2C3x3E_0>,  _s_dOut_7 : vec3<f32>)
{
    var _S261 : vec3<f32> = (*dpx_11).primal_0 - (*dpminval_4).primal_0;
    var _S262 : vec3<f32> = (*dpmaxval_4).primal_0 - (*dpminval_4).primal_0;
    var _S263 : vec3<f32> = _S261 / _S262;
    var _S264 : vec3<f32> = _S262 * _S262;
    var _S265 : vec3<f32> = vec3<f32>(0.0f);
    var _S266 : vec3<f32> = vec3<f32>(1.0f);
    var _S267 : vec3<f32> = s_primal_ctx_clamp_2(_S263, _S265, _S266);
    var _S268 : vec3<f32> = _S267 * ((vec3<f32>(3.0f) - vec3<f32>(2.0f) * _S267) * _s_dOut_7);
    var _S269 : vec3<f32> = vec3<f32>(2.0f) * - (_S267 * _S267 * _s_dOut_7) + _S268 + _S268;
    var _S270 : vec3<f32> = vec3<f32>(0.0f);
    var _S271 : DiffPair_vectorx3Cfloatx2C3x3E_0;
    _S271.primal_0 = _S263;
    _S271.differential_0 = _S270;
    var _S272 : DiffPair_vectorx3Cfloatx2C3x3E_0;
    _S272.primal_0 = _S265;
    _S272.differential_0 = _S270;
    var _S273 : DiffPair_vectorx3Cfloatx2C3x3E_0;
    _S273.primal_0 = _S266;
    _S273.differential_0 = _S270;
    s_bwd_prop_clamp_1(&(_S271), &(_S272), &(_S273), _S269);
    var _S274 : vec3<f32> = _S271.differential_0 / _S264;
    var _S275 : vec3<f32> = _S261 * - _S274;
    var _S276 : vec3<f32> = _S262 * _S274;
    var _S277 : vec3<f32> = - _S275;
    var _S278 : vec3<f32> = - _S276;
    (*dpmaxval_4).primal_0 = (*dpmaxval_4).primal_0;
    (*dpmaxval_4).differential_0 = _S275;
    var _S279 : vec3<f32> = _S277 + _S278;
    (*dpminval_4).primal_0 = (*dpminval_4).primal_0;
    (*dpminval_4).differential_0 = _S279;
    (*dpx_11).primal_0 = (*dpx_11).primal_0;
    (*dpx_11).differential_0 = _S276;
    return;
}

fn Gaussian2D_x24_syn_dadd_0( SLANG_anonymous_0_2 : Gaussian2D_0,  SLANG_anonymous_1_2 : Gaussian2D_0) -> Gaussian2D_0
{
    var result_3 : Gaussian2D_0;
    result_3.center_2 = SLANG_anonymous_0_2.center_2 + SLANG_anonymous_1_2.center_2;
    result_3.sigma_0 = SLANG_anonymous_0_2.sigma_0 + SLANG_anonymous_1_2.sigma_0;
    result_3.color_0 = SLANG_anonymous_0_2.color_0 + SLANG_anonymous_1_2.color_0;
    result_3.opacity_0 = SLANG_anonymous_0_2.opacity_0 + SLANG_anonymous_1_2.opacity_0;
    return result_3;
}

fn s_bwd_prop_sqrt_0( _S280 : ptr<function, DiffPair_float_0>,  _S281 : f32)
{
    _d_sqrt_0(&((*_S280)), _S281);
    return;
}

fn s_bwd_prop_clamp_2( _S282 : ptr<function, DiffPair_vectorx3Cfloatx2C2x3E_0>,  _S283 : ptr<function, DiffPair_vectorx3Cfloatx2C2x3E_0>,  _S284 : ptr<function, DiffPair_vectorx3Cfloatx2C2x3E_0>,  _S285 : vec2<f32>)
{
    _d_clamp_vector_0(&((*_S282)), &((*_S283)), &((*_S284)), _S285);
    return;
}

fn s_bwd_prop_smoothStep_2( dpx_12 : ptr<function, DiffPair_vectorx3Cfloatx2C2x3E_0>,  dpminval_5 : ptr<function, DiffPair_vectorx3Cfloatx2C2x3E_0>,  dpmaxval_5 : ptr<function, DiffPair_vectorx3Cfloatx2C2x3E_0>,  _s_dOut_8 : vec2<f32>)
{
    var _S286 : vec2<f32> = (*dpx_12).primal_0 - (*dpminval_5).primal_0;
    var _S287 : vec2<f32> = (*dpmaxval_5).primal_0 - (*dpminval_5).primal_0;
    var _S288 : vec2<f32> = _S286 / _S287;
    var _S289 : vec2<f32> = _S287 * _S287;
    var _S290 : vec2<f32> = vec2<f32>(0.0f);
    var _S291 : vec2<f32> = vec2<f32>(1.0f);
    var _S292 : vec2<f32> = s_primal_ctx_clamp_0(_S288, _S290, _S291);
    var _S293 : vec2<f32> = _S292 * ((vec2<f32>(3.0f) - vec2<f32>(2.0f) * _S292) * _s_dOut_8);
    var _S294 : vec2<f32> = vec2<f32>(2.0f) * - (_S292 * _S292 * _s_dOut_8) + _S293 + _S293;
    var _S295 : vec2<f32> = vec2<f32>(0.0f);
    var _S296 : DiffPair_vectorx3Cfloatx2C2x3E_0;
    _S296.primal_0 = _S288;
    _S296.differential_0 = _S295;
    var _S297 : DiffPair_vectorx3Cfloatx2C2x3E_0;
    _S297.primal_0 = _S290;
    _S297.differential_0 = _S295;
    var _S298 : DiffPair_vectorx3Cfloatx2C2x3E_0;
    _S298.primal_0 = _S291;
    _S298.differential_0 = _S295;
    s_bwd_prop_clamp_2(&(_S296), &(_S297), &(_S298), _S294);
    var _S299 : vec2<f32> = _S296.differential_0 / _S289;
    var _S300 : vec2<f32> = _S286 * - _S299;
    var _S301 : vec2<f32> = _S287 * _S299;
    var _S302 : vec2<f32> = - _S300;
    var _S303 : vec2<f32> = - _S301;
    (*dpmaxval_5).primal_0 = (*dpmaxval_5).primal_0;
    (*dpmaxval_5).differential_0 = _S300;
    var _S304 : vec2<f32> = _S302 + _S303;
    (*dpminval_5).primal_0 = (*dpminval_5).primal_0;
    (*dpminval_5).differential_0 = _S304;
    (*dpx_12).primal_0 = (*dpx_12).primal_0;
    (*dpx_12).differential_0 = _S301;
    return;
}

fn s_bwd_prop_Gaussian2D_load_0( idx_5 : u32,  localIdx_7 : u32,  _s_dOut_9 : Gaussian2D_0,  _s_diff_ctx_4 : s_bwd_prop_Gaussian2D_load_Intermediates_0)
{
    var total_2 : u32 = Gaussian2D_count_0();
    var _S305 : u32 = total_2 + idx_5;
    var _S306 : vec2<f32> = vec2<f32>(_s_diff_ctx_4._S89, _s_diff_ctx_4._S90);
    const _S307 : vec2<f32> = vec2<f32>(0.0f, 0.0f);
    const _S308 : vec2<f32> = vec2<f32>(1.0f, 1.0f);
    var _S309 : u32 = total_2 * u32(2) + idx_5;
    var _S310 : f32 = _s_diff_ctx_4._S91 * 0.80000001192092896f;
    var _S311 : f32 = s_primal_ctx_smoothStep_1(_S310, 0.0f, 1.0f) + 0.00499999988824129f;
    var _S312 : u32 = total_2 * u32(3) + idx_5;
    var _S313 : f32 = _s_diff_ctx_4._S92 * 0.80000001192092896f;
    var _S314 : f32 = s_primal_ctx_smoothStep_1(_S313, 0.0f, 1.0f) + 0.00499999988824129f;
    var _S315 : u32 = total_2 * u32(4) + idx_5;
    var _S316 : f32 = _s_diff_ctx_4._S93 * 0.60000002384185791f;
    var aniso_1 : f32 = (s_primal_ctx_smoothStep_1(_S316, 0.0f, 1.0f) - 0.5f) * 1.64999997615814209f;
    var _S317 : f32 = _S311 * _S314;
    var _S318 : f32 = s_primal_ctx_sqrt_0(_S317);
    var _S319 : u32 = total_2 * u32(5) + idx_5;
    var _S320 : u32 = total_2 * u32(6) + idx_5;
    var _S321 : u32 = total_2 * u32(7) + idx_5;
    var _S322 : vec3<f32> = vec3<f32>(_s_diff_ctx_4._S94 * 0.80000001192092896f, _s_diff_ctx_4._S95 * 0.80000001192092896f, _s_diff_ctx_4._S96 * 0.80000001192092896f);
    const _S323 : vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    const _S324 : vec3<f32> = vec3<f32>(1.0f, 1.0f, 1.0f);
    var _S325 : u32 = total_2 * u32(8) + idx_5;
    var _S326 : f32 = _s_diff_ctx_4._S97 * 0.89999997615814209f + 0.10000000149011612f;
    var _S327 : mat2x2<f32> = mat2x2<f32>(0.0f, 0.0f, 0.0f, 0.0f);
    var _S328 : Gaussian2D_0 = _s_dOut_9;
    _S328.sigma_0 = _S327;
    var _S329 : mat2x2<f32> = _s_dOut_9.sigma_0;
    var _S330 : mat2x2<f32> = mat2x2<f32>(_S1[0] * _S329[0], _S1[1] * _S329[1]);
    _S328.opacity_0 = 0.0f;
    var _S331 : DiffPair_float_0;
    _S331.primal_0 = _S326;
    _S331.differential_0 = 0.0f;
    var _S332 : DiffPair_float_0;
    _S332.primal_0 = 0.0f;
    _S332.differential_0 = 0.0f;
    var _S333 : DiffPair_float_0;
    _S333.primal_0 = 1.0f;
    _S333.differential_0 = 0.0f;
    s_bwd_prop_smoothStep_0(&(_S331), &(_S332), &(_S333), _s_dOut_9.opacity_0);
    s_bwd_prop_loadFloat_0(_S325, localIdx_7, 0.89999997615814209f * _S331.differential_0);
    var _S334 : vec3<f32> = vec3<f32>(0.0f);
    _S328.color_0 = _S334;
    var _S335 : DiffPair_vectorx3Cfloatx2C3x3E_0;
    _S335.primal_0 = _S322;
    _S335.differential_0 = _S334;
    var _S336 : DiffPair_vectorx3Cfloatx2C3x3E_0;
    _S336.primal_0 = _S323;
    _S336.differential_0 = _S334;
    var _S337 : DiffPair_vectorx3Cfloatx2C3x3E_0;
    _S337.primal_0 = _S324;
    _S337.differential_0 = _S334;
    s_bwd_prop_smoothStep_1(&(_S335), &(_S336), &(_S337), _s_dOut_9.color_0);
    s_bwd_prop_loadFloat_0(_S321, localIdx_7, 0.80000001192092896f * _S335.differential_0[i32(2)]);
    s_bwd_prop_loadFloat_0(_S320, localIdx_7, 0.80000001192092896f * _S335.differential_0[i32(1)]);
    s_bwd_prop_loadFloat_0(_S319, localIdx_7, 0.80000001192092896f * _S335.differential_0[i32(0)]);
    var _S338 : Gaussian2D_0 = Gaussian2D_x24_syn_dzero_0();
    _S338.sigma_0 = _S330;
    var _S339 : Gaussian2D_0 = Gaussian2D_x24_syn_dadd_0(_S328, _S338);
    _S328 = _S339;
    _S328.sigma_0[i32(1)][i32(0)] = 0.0f;
    _S328.sigma_0[i32(0)][i32(1)] = 0.0f;
    var _S340 : f32 = _S339.sigma_0[i32(1)][i32(0)] + _S339.sigma_0[i32(0)][i32(1)];
    var s_diff_aniso_T_0 : f32 = _S318 * _S340;
    var _S341 : f32 = aniso_1 * _S340;
    var _S342 : DiffPair_float_0;
    _S342.primal_0 = _S317;
    _S342.differential_0 = 0.0f;
    s_bwd_prop_sqrt_0(&(_S342), _S341);
    var _S343 : f32 = _S311 * _S342.differential_0;
    var _S344 : f32 = _S314 * _S342.differential_0;
    var _S345 : f32 = 1.64999997615814209f * s_diff_aniso_T_0;
    var _S346 : DiffPair_float_0;
    _S346.primal_0 = _S316;
    _S346.differential_0 = 0.0f;
    var _S347 : DiffPair_float_0;
    _S347.primal_0 = 0.0f;
    _S347.differential_0 = 0.0f;
    var _S348 : DiffPair_float_0;
    _S348.primal_0 = 1.0f;
    _S348.differential_0 = 0.0f;
    s_bwd_prop_smoothStep_0(&(_S346), &(_S347), &(_S348), _S345);
    s_bwd_prop_loadFloat_0(_S315, localIdx_7, 0.60000002384185791f * _S346.differential_0);
    _S328.sigma_0[i32(1)][i32(1)] = 0.0f;
    var _S349 : f32 = _S343 + _S339.sigma_0[i32(1)][i32(1)];
    var _S350 : DiffPair_float_0;
    _S350.primal_0 = _S313;
    _S350.differential_0 = 0.0f;
    var _S351 : DiffPair_float_0;
    _S351.primal_0 = 0.0f;
    _S351.differential_0 = 0.0f;
    var _S352 : DiffPair_float_0;
    _S352.primal_0 = 1.0f;
    _S352.differential_0 = 0.0f;
    s_bwd_prop_smoothStep_0(&(_S350), &(_S351), &(_S352), _S349);
    s_bwd_prop_loadFloat_0(_S312, localIdx_7, 0.80000001192092896f * _S350.differential_0);
    _S328.sigma_0[i32(0)][i32(0)] = 0.0f;
    var _S353 : f32 = _S344 + _S339.sigma_0[i32(0)][i32(0)];
    var _S354 : DiffPair_float_0;
    _S354.primal_0 = _S310;
    _S354.differential_0 = 0.0f;
    var _S355 : DiffPair_float_0;
    _S355.primal_0 = 0.0f;
    _S355.differential_0 = 0.0f;
    var _S356 : DiffPair_float_0;
    _S356.primal_0 = 1.0f;
    _S356.differential_0 = 0.0f;
    s_bwd_prop_smoothStep_0(&(_S354), &(_S355), &(_S356), _S353);
    s_bwd_prop_loadFloat_0(_S309, localIdx_7, 0.80000001192092896f * _S354.differential_0);
    var _S357 : vec2<f32> = vec2<f32>(0.0f);
    var _S358 : DiffPair_vectorx3Cfloatx2C2x3E_0;
    _S358.primal_0 = _S306;
    _S358.differential_0 = _S357;
    var _S359 : DiffPair_vectorx3Cfloatx2C2x3E_0;
    _S359.primal_0 = _S307;
    _S359.differential_0 = _S357;
    var _S360 : DiffPair_vectorx3Cfloatx2C2x3E_0;
    _S360.primal_0 = _S308;
    _S360.differential_0 = _S357;
    s_bwd_prop_smoothStep_2(&(_S358), &(_S359), &(_S360), _S328.center_2);
    s_bwd_prop_loadFloat_0(_S305, localIdx_7, _S358.differential_0[i32(1)]);
    s_bwd_prop_loadFloat_0(idx_5, localIdx_7, _S358.differential_0[i32(0)]);
    return;
}

fn s_bwd_prop_eval_0( blob_id_2 : u32,  uv_3 : vec2<f32>,  localIdx_8 : u32,  _s_dOut_10 : vec4<f32>,  _s_diff_ctx_5 : s_bwd_prop_eval_Intermediates_0)
{
    var _S361 : Gaussian2D_0 = Gaussian2D_x24_syn_dzero_0();
    var _S362 : DiffPair_Gaussian2D_0;
    _S362.primal_0 = _s_diff_ctx_5._S99;
    _S362.differential_0 = _S361;
    var _S363 : vec2<f32> = vec2<f32>(0.0f);
    var _S364 : DiffPair_vectorx3Cfloatx2C2x3E_0;
    _S364.primal_0 = uv_3;
    _S364.differential_0 = _S363;
    s_bwd_prop_Gaussian2D_eval_0(&(_S362), &(_S364), _s_dOut_10);
    s_bwd_prop_Gaussian2D_load_0(blob_id_2, localIdx_8, _S362.differential_0, _s_diff_ctx_5._S98);
    return;
}

fn s_bwd_eval_0( _S365 : u32,  _S366 : vec2<f32>,  _S367 : u32,  _S368 : vec4<f32>)
{
    var _S369 : s_bwd_prop_eval_Intermediates_0;
    var _S370 : vec4<f32> = s_primal_ctx_eval_0(_S365, _S366, _S367, &(_S369));
    s_bwd_prop_eval_0(_S365, _S366, _S367, _S368, _S369);
    return;
}

fn fineRasterize_bwd_0( SLANG_anonymous_3_0 : SortedShortList_0,  localIdx_9 : u32,  uv_4 : vec2<f32>,  dOut_8 : vec4<f32>)
{
    workgroupBarrier();
    var _S371 : vec4<f32> = finalVal_0[localIdx_9];
    var _S372 : u32 = maxCount_0[localIdx_9];
    var count_0 : u32 = (workgroupUniformLoad(&((blobCount_0))));
    var _S373 : PixelState_Differential_0 = PixelState_x24_syn_dzero_0();
    var _S374 : vec4<f32> = vec4<f32>(0.0f);
    var pixelState_0 : PixelState_0;
    pixelState_0.value_0 = _S371;
    pixelState_0.finalCount_0 = _S372;
    var dColor_0 : PixelState_Differential_0;
    dColor_0.value_1 = dOut_8;
    var _i_0 : u32 = count_0;
    for(;;)
    {
        if(_i_0 > u32(0))
        {
        }
        else
        {
            break;
        }
        var i_5 : u32 = _i_0 - u32(1);
        var blobID_0 : u32 = blobs_0[i_5];
        var gval_4 : vec4<f32> = eval_0(blobs_0[i_5], uv_4, localIdx_9);
        var prevState_0 : PixelState_0 = undoPixelState_0(pixelState_0, i_5 + u32(1), gval_4);
        var dpState_0 : DiffPair_PixelState_0;
        dpState_0.primal_0 = prevState_0;
        dpState_0.differential_0 = _S373;
        var dpGVal_0 : DiffPair_vectorx3Cfloatx2C4x3E_0;
        dpGVal_0.primal_0 = gval_4;
        dpGVal_0.differential_0 = _S374;
        s_bwd_transformPixelState_0(&(dpState_0), &(dpGVal_0), dColor_0);
        s_bwd_eval_0(blobID_0, uv_4, localIdx_9, dpGVal_0.differential_0);
        var _S375 : DiffPair_PixelState_0 = dpState_0;
        pixelState_0 = prevState_0;
        dColor_0 = _S375.differential_0;
        _i_0 = i_5;
    }
    return;
}

fn fineRasterize_0( SLANG_anonymous_2_0 : SortedShortList_0,  localIdx_10 : u32,  uv_5 : vec2<f32>) -> vec4<f32>
{
    workgroupBarrier();
    var _S376 : u32 = blobCount_0;
    var pixelState_1 : PixelState_0;
    pixelState_1.value_0 = vec4<f32>(0.0f, 0.0f, 0.0f, 1.0f);
    pixelState_1.finalCount_0 = u32(0);
    var i_6 : u32 = u32(0);
    for(;;)
    {
        if(i_6 < _S376)
        {
        }
        else
        {
            break;
        }
        var _S377 : PixelState_0 = transformPixelState_0(pixelState_1, eval_0(blobs_0[i_6], uv_5, localIdx_10));
        var _S378 : u32 = i_6 + u32(1);
        pixelState_1 = _S377;
        i_6 = _S378;
    }
    maxCount_0[localIdx_10] = pixelState_1.finalCount_0;
    var _S379 : PixelState_0 = pixelState_1;
    finalVal_0[localIdx_10] = pixelState_1.value_0;
    return _S379.value_0;
}

struct s_bwd_prop_splatBlobs_Intermediates_0
{
     _S380 : SortedShortList_0,
     _S381 : vec4<f32>,
};

struct s_bwd_prop_loss_Intermediates_0
{
     _S382 : s_bwd_prop_splatBlobs_Intermediates_0,
     _S383 : vec4<f32>,
};

fn s_primal_ctx_fineRasterize_0( _S384 : SortedShortList_0,  _S385 : u32,  _S386 : vec2<f32>) -> vec4<f32>
{
    var _S387 : vec4<f32> = fineRasterize_0(_S384, _S385, _S386);
    return _S387;
}

fn s_primal_ctx_splatBlobs_0( dispatchThreadID_2 : vec2<u32>,  dispatchSize_0 : vec2<i32>,  _s_diff_ctx_6 : ptr<function, s_bwd_prop_splatBlobs_Intermediates_0>) -> vec4<f32>
{
    var _S388 : SortedShortList_0 = SortedShortList_0( i32(0) );
    var _S389 : vec4<f32> = vec4<f32>(0.0f);
    (*_s_diff_ctx_6)._S380 = _S388;
    (*_s_diff_ctx_6)._S381 = _S389;
    (*_s_diff_ctx_6)._S380._dummy_3 = i32(0);
    (*_s_diff_ctx_6)._S381 = _S389;
    var _S390 : i32 = i32(0);
    var _S391 : i32 = i32(0);
    {var dim = textureDimensions((targetTexture_0));((_S390)) = bitcast<i32>(dim.x);((_S391)) = bitcast<i32>(dim.y);};
    var texSize_0 : vec2<i32> = vec2<i32>(_S390, _S391);
    var uv_6 : vec2<f32> = calcUV_0(dispatchThreadID_2, dispatchSize_0, texSize_0);
    var tileCoords_0 : vec2<u32> = vec2<u32>(dispatchThreadID_2.x / u32(16), dispatchThreadID_2.y / u32(16));
    const _S392 : vec2<u32> = vec2<u32>(u32(16), u32(16));
    var tileLow_0 : vec2<f32> = calcUV_0(tileCoords_0 * _S392, dispatchSize_0, texSize_0);
    var tileHigh_0 : vec2<f32> = calcUV_0((tileCoords_0 + vec2<u32>(u32(1))) * _S392, dispatchSize_0, texSize_0);
    var _S393 : vec2<f32> = vec2<f32>(2.0f);
    var tileBounds_1 : OBB_0 = OBB_x24init_0((tileLow_0 + tileHigh_0) / _S393, mat2x2<f32>(1.0f, 0.0f, 0.0f, 1.0f), (tileHigh_0 - tileLow_0) / _S393);
    var sList_1 : InitializedShortList_0 = initShortList_0(dispatchThreadID_2);
    var localID_0 : vec2<u32> = dispatchThreadID_2 % _S392;
    var localIdx_11 : u32 = localID_0.x + localID_0.y * u32(16);
    var filledSList_0 : FilledShortList_0 = coarseRasterize_0(sList_1, tileBounds_1, localIdx_11);
    var paddedSList_0 : PaddedShortList_0 = padBuffer_0(filledSList_0, localIdx_11);
    var sortedList_0 : SortedShortList_0 = bitonicSort_0(paddedSList_0, localIdx_11);
    (*_s_diff_ctx_6)._S380 = sortedList_0;
    var _S394 : vec4<f32> = s_primal_ctx_fineRasterize_0(sortedList_0, localIdx_11, uv_6);
    (*_s_diff_ctx_6)._S381 = _S394;
    var _S395 : f32 = _S394.w;
    return vec4<f32>(_S394.xyz * vec3<f32>((1.0f - _S395)) + vec3<f32>(_S395), 1.0f);
}

fn s_primal_ctx_dot_0( _S396 : vec3<f32>,  _S397 : vec3<f32>) -> f32
{
    return dot(_S396, _S397);
}

fn s_primal_ctx_loss_0( dispatchThreadID_3 : vec2<u32>,  imageSize_1 : vec2<i32>,  _s_diff_ctx_7 : ptr<function, s_bwd_prop_loss_Intermediates_0>) -> f32
{
    var _S398 : SortedShortList_0 = SortedShortList_0( i32(0) );
    var _S399 : vec4<f32> = vec4<f32>(0.0f);
    var _S400 : s_bwd_prop_splatBlobs_Intermediates_0 = s_bwd_prop_splatBlobs_Intermediates_0( _S398, _S399 );
    (*_s_diff_ctx_7)._S382 = _S400;
    (*_s_diff_ctx_7)._S383 = _S399;
    (*_s_diff_ctx_7)._S383 = _S399;
    var _S401 : s_bwd_prop_splatBlobs_Intermediates_0 = (*_s_diff_ctx_7)._S382;
    var _S402 : vec4<f32> = s_primal_ctx_splatBlobs_0(dispatchThreadID_3, imageSize_1, &(_S401));
    (*_s_diff_ctx_7)._S382 = _S401;
    (*_s_diff_ctx_7)._S383 = _S402;
    var _S403 : u32 = dispatchThreadID_3.x;
    var _S404 : bool;
    if(_S403 >= u32(imageSize_1.x))
    {
        _S404 = true;
    }
    else
    {
        _S404 = (dispatchThreadID_3.y) >= u32(imageSize_1.y);
    }
    var _S405 : f32;
    if(_S404)
    {
        _S405 = 0.0f;
    }
    else
    {
        var _S406 : vec3<i32> = vec3<i32>(vec3<u32>(vec2<u32>(_S403, u32(imageSize_1.y) - dispatchThreadID_3.y), u32(0)));
        var _S407 : vec3<f32> = _S402.xyz - (textureLoad((targetTexture_0), ((_S406)).xy, ((_S406)).z)).xyz;
        _S405 = s_primal_ctx_dot_0(_S407, _S407);
    }
    return _S405;
}

fn s_bwd_prop_dot_0( _S408 : ptr<function, DiffPair_vectorx3Cfloatx2C3x3E_0>,  _S409 : ptr<function, DiffPair_vectorx3Cfloatx2C3x3E_0>,  _S410 : f32)
{
    _d_dot_0(&((*_S408)), &((*_S409)), _S410);
    return;
}

fn s_bwd_prop_fineRasterize_0( _S411 : SortedShortList_0,  _S412 : u32,  _S413 : vec2<f32>,  _S414 : vec4<f32>)
{
    fineRasterize_bwd_0(_S411, _S412, _S413, _S414);
    return;
}

fn s_bwd_prop_splatBlobs_0( dispatchThreadID_4 : vec2<u32>,  dispatchSize_1 : vec2<i32>,  _s_dOut_11 : vec4<f32>,  _s_diff_ctx_8 : s_bwd_prop_splatBlobs_Intermediates_0)
{
    var _S415 : i32 = i32(0);
    var _S416 : i32 = i32(0);
    {var dim = textureDimensions((targetTexture_0));((_S415)) = bitcast<i32>(dim.x);((_S416)) = bitcast<i32>(dim.y);};
    var uv_7 : vec2<f32> = calcUV_0(dispatchThreadID_4, dispatchSize_1, vec2<i32>(_S415, _S416));
    var localID_1 : vec2<u32> = dispatchThreadID_4 % vec2<u32>(u32(16), u32(16));
    var _S417 : vec3<f32> = _s_dOut_11.xyz;
    var _S418 : vec3<f32> = _s_diff_ctx_8._S381.xyz * _S417;
    var _S419 : vec3<f32> = vec3<f32>((1.0f - _s_diff_ctx_8._S381.w)) * _S417;
    s_bwd_prop_fineRasterize_0(_s_diff_ctx_8._S380, localID_1.x + localID_1.y * u32(16), uv_7, vec4<f32>(_S419[i32(0)], _S419[i32(1)], _S419[i32(2)], - (_S418[i32(0)] + _S418[i32(1)] + _S418[i32(2)]) + _S417[i32(0)] + _S417[i32(1)] + _S417[i32(2)]));
    return;
}

fn s_bwd_prop_loss_0( dispatchThreadID_5 : vec2<u32>,  imageSize_2 : vec2<i32>,  _s_dOut_12 : f32,  _s_diff_ctx_9 : s_bwd_prop_loss_Intermediates_0)
{
    var _S420 : vec3<f32> = vec3<f32>(0.0f);
    var _S421 : u32 = dispatchThreadID_5.x;
    var _S422 : bool;
    if(_S421 >= u32(imageSize_2.x))
    {
        _S422 = true;
    }
    else
    {
        _S422 = (dispatchThreadID_5.y) >= u32(imageSize_2.y);
    }
    var _S423 : vec3<f32>;
    if(_S422)
    {
        _S423 = _S420;
    }
    else
    {
        var _S424 : vec3<i32> = vec3<i32>(vec3<u32>(vec2<u32>(_S421, u32(imageSize_2.y) - dispatchThreadID_5.y), u32(0)));
        _S423 = _s_diff_ctx_9._S383.xyz - (textureLoad((targetTexture_0), ((_S424)).xy, ((_S424)).z)).xyz;
    }
    var _S425 : vec4<f32> = vec4<f32>(0.0f);
    var _S426 : vec4<f32>;
    if(_S422)
    {
        _S426 = _S425;
    }
    else
    {
        var _S427 : DiffPair_vectorx3Cfloatx2C3x3E_0;
        _S427.primal_0 = _S423;
        _S427.differential_0 = _S420;
        var _S428 : DiffPair_vectorx3Cfloatx2C3x3E_0;
        _S428.primal_0 = _S423;
        _S428.differential_0 = _S420;
        s_bwd_prop_dot_0(&(_S427), &(_S428), _s_dOut_12);
        var _S429 : vec3<f32> = _S428.differential_0 + _S427.differential_0;
        _S426 = vec4<f32>(_S429[i32(0)], _S429[i32(1)], _S429[i32(2)], 0.0f);
    }
    s_bwd_prop_splatBlobs_0(dispatchThreadID_5, imageSize_2, _S426, _s_diff_ctx_9._S382);
    return;
}

fn s_bwd_loss_0( _S430 : vec2<u32>,  _S431 : vec2<i32>,  _S432 : f32)
{
    var _S433 : s_bwd_prop_loss_Intermediates_0;
    var _S434 : f32 = s_primal_ctx_loss_0(_S430, _S431, &(_S433));
    s_bwd_prop_loss_0(_S430, _S431, _S432, _S433);
    return;
}

@compute
@workgroup_size(16, 16, 1)
fn computeDerivativesMain(@builtin(global_invocation_id) dispatchThreadID_6 : vec3<u32>)
{
    var dispatchThreadID_7 : vec2<u32> = dispatchThreadID_6.xy;
    var dimX_0 : u32;
    var dimY_0 : u32;
    {var dim = textureDimensions((targetTexture_0));((dimX_0)) = dim.x;((dimY_0)) = dim.y;};
    var _S435 : i32 = i32(dimX_0);
    var _S436 : i32 = i32(dimY_0);
    s_bwd_loss_0(dispatchThreadID_7, vec2<i32>(_S435, _S436), 1.0f / f32(_S435 * _S436));
    return;
}



/// updateBlobsMain entrypoint ///

// @binding(3) @group(0) var<storage, read_write> derivBuffer_0 : array<atomic<u32>>;

// @binding(4) @group(0) var<storage, read_write> adamFirstMoment_0 : array<f32>;

// @binding(5) @group(0) var<storage, read_write> adamSecondMoment_0 : array<f32>;

// @binding(2) @group(0) var<storage, read_write> blobsBuffer_0 : array<f32>;

@compute
@workgroup_size(256, 1, 1)
fn updateBlobsMain(@builtin(global_invocation_id) dispatchThreadID_0 : vec3<u32>)
{
    var globalID_0 : u32 = dispatchThreadID_0.xy.x;
    if(globalID_0 >= u32(184320))
    {
        return;
    }
    var _S1 : u32 = atomicLoad(&(derivBuffer_0[globalID_0]));
    var g_t_0 : f32 = (bitcast<f32>((_S1)));
    atomicStore(&(derivBuffer_0[globalID_0]), (bitcast<u32>((0.0f))));
    var m_t_0 : f32 = 0.89999997615814209f * adamFirstMoment_0[globalID_0] + 0.10000002384185791f * g_t_0;
    var v_t_0 : f32 = 0.99900001287460327f * adamSecondMoment_0[globalID_0] + 0.00099998712539673f * (g_t_0 * g_t_0);
    adamFirstMoment_0[globalID_0] = m_t_0;
    adamSecondMoment_0[globalID_0] = v_t_0;
    blobsBuffer_0[globalID_0] = blobsBuffer_0[globalID_0] - 0.0020000000949949f / (sqrt(v_t_0 / 0.00099998712539673f) + 9.99999993922529029e-09f) * (m_t_0 / 0.10000002384185791f);
    return;
}



/// imageMain entrypoint ///

fn splatBlobs_0( dispatchThreadID_2 : vec2<u32>,  dispatchSize_0 : vec2<i32>) -> vec4<f32>
{
    var texWidth_0 : i32;
    var texHeight_0 : i32;
    {var dim = textureDimensions((targetTexture_0));((texWidth_0)) = bitcast<i32>(dim.x);((texHeight_0)) = bitcast<i32>(dim.y);};
    var texSize_0 : vec2<i32> = vec2<i32>(texWidth_0, texHeight_0);
    var uv_3 : vec2<f32> = calcUV_0(dispatchThreadID_2, dispatchSize_0, texSize_0);
    var tileCoords_0 : vec2<u32> = vec2<u32>(dispatchThreadID_2.x / u32(16), dispatchThreadID_2.y / u32(16));
    const _S46 : vec2<u32> = vec2<u32>(u32(16), u32(16));
    var tileLow_0 : vec2<f32> = calcUV_0(tileCoords_0 * _S46, dispatchSize_0, texSize_0);
    var tileHigh_0 : vec2<f32> = calcUV_0((tileCoords_0 + vec2<u32>(u32(1))) * _S46, dispatchSize_0, texSize_0);
    var _S47 : vec2<f32> = vec2<f32>(2.0f);
    var tileBounds_1 : OBB_0 = OBB_x24init_0((tileLow_0 + tileHigh_0) / _S47, mat2x2<f32>(1.0f, 0.0f, 0.0f, 1.0f), (tileHigh_0 - tileLow_0) / _S47);
    var sList_1 : InitializedShortList_0 = initShortList_0(dispatchThreadID_2);
    var localID_0 : vec2<u32> = dispatchThreadID_2 % _S46;
    var localIdx_6 : u32 = localID_0.x + localID_0.y * u32(16);
    var filledSList_0 : FilledShortList_0 = coarseRasterize_0(sList_1, tileBounds_1, localIdx_6);
    var paddedSList_0 : PaddedShortList_0 = padBuffer_0(filledSList_0, localIdx_6);
    var sortedList_0 : SortedShortList_0 = bitonicSort_0(paddedSList_0, localIdx_6);
    var color_1 : vec4<f32> = fineRasterize_0(sortedList_0, localIdx_6, uv_3);
    var _S48 : f32 = color_1.w;
    return vec4<f32>(color_1.xyz * vec3<f32>((1.0f - _S48)) + vec3<f32>(_S48), 1.0f);
}

fn imageMain_0( dispatchThreadID_3 : vec2<u32>,  screenSize_0 : vec2<u32>) -> vec4<f32>
{
    var _S49 : vec4<f32> = splatBlobs_0(dispatchThreadID_3, vec2<i32>(screenSize_0));
    return _S49;
}

@compute
@workgroup_size(16, 16, 1)
fn imageMain(@builtin(global_invocation_id) dispatchThreadID_4 : vec3<u32>)
{
    var width_0 : u32 = u32(0);
    var height_0 : u32 = u32(0);
    {var dim = textureDimensions((outputTexture_0));((width_0)) = dim.x;((height_0)) = dim.y;};
    var _S50 : vec2<u32> = dispatchThreadID_4.xy;
    var color_2 : vec4<f32> = imageMain_0(_S50, vec2<u32>(vec2<i32>(i32(width_0), i32(height_0))));
    var _S51 : bool;
    if((dispatchThreadID_4.x) >= width_0)
    {
        _S51 = true;
    }
    else
    {
        _S51 = (dispatchThreadID_4.y) >= height_0;
    }
    if(_S51)
    {
        return;
    }

    /// Manually edited to flip the image the right way up ///
    _S50.y = height_0 - _S50.y;

    textureStore((outputTexture_0), (_S50), (color_2));
    return;
}
