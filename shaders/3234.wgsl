//algorithm that trains a neural network but using
//new way to normalize the computed change to the weights and biases
//by backpropagating the value of 1.0 instead of backpropagating the error

//one day when my pull request get accepted #pipeline_once #repeat #pipeline will work
//#pipeline_once  iniNN iniLinks iniTrainDataCenter iniTrainDataWidth iniTrainDataWidth2
//#repeat         train 4 trainDataToNN forwBackwProp batchsNNch reduceNNch applyNNch
//#pipeline       train render render2
const PI = 3.1415926535897932f;
#define nnI  2                  //neural network input  neurons
#define nnO  3                  //neural network output neurons
#define nnM  99                 //neural network middle neurons
#define nnN  (nnI+nnO+nnM)      //neural network total  neurons
#define nc1  (nnI + nnM - 1)
#define nc2  (nnI       - 1)
#define nnW  (nc1*(nc1+1)/2 - nc2*(nc2+1)/2 + (nnI+nnM)*nnO) //neural network total weights
#define nnWB (nnW + nnM+nnO)    //neural network total weights + biases
#define trnU 4                  //training neuron has value and derivative and normaliz and nothing
#define trnB (1<<12)            //training batches
const   trnRz = 4f;             //resize training data
const   rczZ  = 1+1+nnI+nnO;    //ReCenSiz size
const   imgz  = 512;            //size of image to see learning progress
const   imgz2 = imgz*imgz;      //2D image total size
const   errhz = 1<<10;          //error history size
const   erraz = 1<<5;           //average of erraz error samples
const   dbg = true;             //show screen error,weight,biases
#define numTreds0  (1<<6)
#define numTreds1  (1<<7)
#define numTreds2  (1<<6)
#define numTreds3  (1<<8)
#define numTreds4  (1<<6)
#define numTreds5  (1<<6)
#define numTreds6  (1<<6)
#define numTreds6B (1<<3)       //numTreds6B != imgz2/numTreds6 because that uses too much memory
const imgB = numTreds6*numTreds6B; //batches to render image
const nfsiz = max(
    nnN*trnB*trnU+trnB,     //size for training, +trnB for the error
    nnN*imgB                //size for rendering, no trnU because no derivative
);
struct Train
{
    @align(256) nf  : array<f32,nfsiz>,         //forward propagation neuron values from batches
    @align(256) nn  : array<f32,nnWB>,          //          weigts and bias interlaced
    @align(256) nn2 : array<f32,nnWB+1>,        //change to weigts and bias interlaced, +1 for the error
    @align(256) nn3 : array<f32,nnWB+1>,        //normalizator                        , +1 not used
    @align(256) lk  : array<i32,(nnWB+1)*2>,    //links between neurons, union of read 16bit and write 16bit
    @align(256) rcz : array<f32,rczZ>,          //recenter resize training data, interpreted as ReCenSiz
    @align(256) err : array<f32,errhz*2>,       //training error history
    @align(256) cnt : i32,                      //counter
    @align(256) img : array<vec3f,imgz2>,       //image
}
struct ReCenSiz
{
    rzi : f32,              //resize   training data input
    rzo : f32,              //resize   training data output
    rci : array<f32,nnI>,   //recenter training data input
    rco : array<f32,nnO>,   //recenter training data output
}
#storage D Train
//fn reduce<T>(data: ptr<workgroup,array<T>,read_write>, id1: i32, sumThrds: i32)//result in data[0] and only thread id1==0 has it
//{
//    for(var stride = sumThrds >> 1; stride > 0; stride >>= 1){
//        workgroupBarrier();
//        if(id1 < stride){ data[id1] += data[id1 + stride];}
//    }
//}
fn hash(a: u32) -> u32
{
    var x = a;
    x ^= x >> 17;  x *= 0xed5ad4bbu;
    x ^= x >> 11;  x *= 0xac4c1b51u;
    x ^= x >> 15;  x *= 0x31848babu;
    x ^= x >> 14;  return x;
}
fn rnd(a: u32) -> f32
{
    var h   = hash(a);
    var msk = (1u << 23u) - 1u;
    return f32(h & msk) / f32(1u << 23u);
}
fn gaus(a: u32) -> f32//remember this uses 2 seeds from "a"
{
    var r1 = 1.f-rnd(a+0u);
    var r2 =     rnd(a+1u);
    return sqrt(-2.f*log(r1))*cos(2.f*PI*r2);
}
fn hash2(x2:u32, y2:u32) -> u32
{
    var x = x2;
    x ^= x >> 17;  x *= 0xed5ad4bbu;
    x ^= x >> 11;  x *= 0xac4c1b51u;
    x ^= x >> 15;  x *= 0x31848babu;
    x ^= x >> 14;
    //uint n = 5u; for(uint i=0u;i<n;i++){y = ((y>>8)^y)*0x6Bu+n;}
    var y = y2;
    y ^= y >> 15;
    y ^= (y * y) | 1u;
    y ^= y >> 17;
    y *= 0x9E3779B9u;
    y ^= y >> 13;
    return x+y;
}
struct ActResult {
    r: f32,
    d: f32,
};
fn act(v:f32) -> ActResult
{
    var r = 0f;
    var d = 0f;
    var mode = 0;
    if(mode==0)
    {
        var a = abs(v);
        var b = 1.f - 1.f/exp(a);
        var c = 1.f; if(v<0.f){c = -1.f;}
        d = c*b + 1.f;
        r = a - b + v;
    }
    if(mode==1)
    {
        r = max(v,0.f);
        d = f32(v>=0.f);
    }
    return ActResult(r,d);
}
#dispatch_once iniNN
#workgroup_count iniNN 1 1 1
@compute @workgroup_size(1,1,1)
fn iniNN(@builtin(global_invocation_id) id3: vec3u)
{
    var seed = 1926347346u;
    var w = 0;
    for(var i = nnI; i < nnN; i++)
    {
        var s  = 0.f;
        var i2 = min(i,nnN-nnO);
        for(var j = 0; j < i2; j++)
        {
            var g = gaus(seed);  seed+=2u;  if(i>=nnN-nnO){g=f32(max(0,j-nnI)+1);}
            s += g*g;
            D.nn[w+j] = g;      //weight
        }
        s = 1.f/sqrt(s);        //if(i>=nnN-nnO){s = 0.f;}
        for(var j = 0; j < i2; j++)
        {
            var s2 = s;  if(trnRz==4.f && j>=nnI){s2*=trnRz/4.583989812529729f;}
            D.nn[w+j] *= s2;
        }
        w += i2;
        D.nn[w] = 0.f;  w++;    //bias
    }
}
#dispatch_once iniLinks
#workgroup_count iniLinks 1 1 1
@compute @workgroup_size(1, 1, 1)
fn iniLinks(@builtin(global_invocation_id) id3: vec3u)
{
    var w = 0;
    for(var i=nnI; i<nnN; i++){ var i2 = min(i,nnN-nnO);
    for(var j=  0; j< i2; j++){ D.lk[w] = j   ;  w++;
                                D.lk[w] = i   ;  w++;}    //load weight
                                D.lk[w] = nnWB;  w++;     //load nothing
                                D.lk[w] = i   ;  w++;     //load bias
    }                           D.lk[w] = nnN ;  w++;     //load error
                                D.lk[w] = nnWB;  w++;     //load nothing
}
var<workgroup> sum2numTreds0: array<vec2f,numTreds0>;
var<workgroup> sum3numTreds0: array<vec3f,numTreds0>;
#dispatch_once iniTrainDataCenter
#workgroup_count iniTrainDataCenter 1 1 1
@compute @workgroup_size(numTreds0,1,1)
fn iniTrainDataCenter(@builtin(global_invocation_id) id3: vec3u)
{
    var texdim = vec2i(textureDimensions(channel0));
    var texdiv = 1f/vec2f(texdim);
    var texpxt = texdim.x * texdim.y;
    var divpxt = 1f/f32(texpxt);
    var id1    = i32(id3.x);
    //calculate center of input data
    var cid = vec2f(0);
    for(var i = id1; i < texpxt; i += numTreds0)
    {
        cid += vec2f(
            f32(i % texdim.x),
            f32(i / texdim.x))*texdiv.x;
    }
    //reduce(sum2numTreds0, id1, numTreds0);
    {
        sum2numTreds0[id1] = cid;
        var idth = id1 % numTreds0;
        for(var stride = numTreds0 >> 1; stride > 0; stride >>= 1){
            workgroupBarrier();
            if(idth < stride){ sum2numTreds0[idth] += sum2numTreds0[idth + stride];}
        }
    }
    if(id1==0){
        var r = sum2numTreds0[0]*divpxt;
        D.rcz[1+1+0] = r.x;
        D.rcz[1+1+1] = r.y;
    }
    //calculate center of output data
    var cod = vec3f(0);
    for(var i = id1; i < texpxt; i += numTreds0)
    {
        var r = vec2i(
            i % texdim.x,
            i / texdim.x
        );
        cod += textureLoad(channel0,r,0).xyz;
    }
    //reduce(sum3numTreds0, id1, numTreds0);
    {
        sum3numTreds0[id1] = cod;
        var idth = id1 % numTreds0;
        for(var stride = numTreds0 >> 1; stride > 0; stride >>= 1){
            workgroupBarrier();
            if(idth < stride){ sum3numTreds0[idth] += sum3numTreds0[idth + stride];}
        }
    }
    if(id1==0){
        var r = sum3numTreds0[0]*divpxt;
        D.rcz[1+1+nnI+0] = r.x;
        D.rcz[1+1+nnI+1] = r.y;
        D.rcz[1+1+nnI+2] = r.z;
    }
    if(id1==0){D.cnt = 0;}
}
var<workgroup> sum1numTreds0: array<f32,numTreds0>;
#dispatch_once iniTrainDataWidth
#workgroup_count iniTrainDataWidth 1 1 1
@compute @workgroup_size(numTreds0,1,1)
fn iniTrainDataWidth(@builtin(global_invocation_id) id3: vec3u)
{
    var texdim = vec2i(textureDimensions(channel0));
    var texdiv = 1f/vec2f(texdim);
    var texpxt = texdim.x * texdim.y;
    var divpxt = 1f/f32(texpxt);
    var id1    = i32(id3.x);
    //calculate gaussian width of input data
    var centerI = vec2f(
        D.rcz[1+1+0],
        D.rcz[1+1+1]
    );
    var wid = 0f;
    for(var i = id1; i < texpxt; i += numTreds0)
    {
        var a = vec2f(
            f32(i % texdim.x),
            f32(i / texdim.x))*texdiv.x-centerI;
        wid += dot(a,a);
    }
    //reduce(sum1numTreds0, id1, numTreds0);
    {
        sum1numTreds0[id1] = wid;
        var idth = id1 % numTreds0;
        for(var stride = numTreds0 >> 1; stride > 0; stride >>= 1){
            workgroupBarrier();
            if(idth < stride){ sum1numTreds0[idth] += sum1numTreds0[idth + stride];}
        }
    }
    if(id1==0){D.rcz[0] = trnRz/sqrt(sum1numTreds0[0]*divpxt/f32(nnI));}
    //calculate gaussian width of output data
    var centerO = vec3f(
        D.rcz[1+1+nnI+0],
        D.rcz[1+1+nnI+1],
        D.rcz[1+1+nnI+2]
    );
    var wod = 0f;
    for(var i = id1; i < texpxt; i += numTreds0)
    {
        var r = vec2i(
            i % texdim.x,
            i / texdim.x
        );
        var a = textureLoad(channel0,r,0).xyz-centerO;
        wod += dot(a,a);
    }
    //reduce(sum1numTreds0, id1, numTreds0);
    {
        sum1numTreds0[id1] = wod;
        var idth = id1 % numTreds0;
        for(var stride = numTreds0 >> 1; stride > 0; stride >>= 1){
            workgroupBarrier();
            if(idth < stride){ sum1numTreds0[idth] += sum1numTreds0[idth + stride];}
        }
    }
    if(id1==0){D.rcz[1] = trnRz/sqrt(sum1numTreds0[0]*divpxt/f32(nnO));}
}
#dispatch_once iniTrainDataWidth2
#workgroup_count iniTrainDataWidth2 1 1 1
@compute @workgroup_size(numTreds0,1,1)
fn iniTrainDataWidth2(@builtin(global_invocation_id) id3: vec3u)
{
    var id1     = i32(id3.x);
    var reszinp = D.rcz[0];
    var reszout = D.rcz[1];
    for(var i = id1; i < nnI; i += numTreds0){D.rcz[1+1+i    ] *= -reszinp;}
    for(var i = id1; i < nnO; i += numTreds0){D.rcz[1+1+nnI+i] *= -reszout;}
}
var<workgroup> rczWGM: array<f32,rczZ>;
#calcdefine trnBdnumTreds1 (trnB/numTreds1)
#workgroup_count trainDataToNN trnBdnumTreds1 1 1
@compute @workgroup_size(numTreds1,1,1)
fn trainDataToNN(@builtin(global_invocation_id) id3: vec3u)
{
    var id1 = i32(id3.x);
    //load rczWGM
    for(var i = id1 % numTreds1; i < rczZ; i += numTreds1)
    {
        rczWGM[i] = D.rcz[i];
    }
    workgroupBarrier();

    var texdim = vec2i(textureDimensions(channel0));
    var texpxt = texdim.x * texdim.y;
    var divpxt = 1f/f32(texpxt);

    //var tottrns = bitcast<i32>(rczWGM[1+1+nnI+nnO  ]);
    var counter = D.cnt;
    var reszinp = rczWGM[0];
    var reszout = rczWGM[1];
    var ha      = i32(hash2(u32(id1),u32(counter)) % u32(texpxt));
    //input
    var rtx = vec2i(
        ha % texdim.x,
        ha / texdim.x
    );
    var rtxf = vec2f(rtx)/f32(texdim.x);
    var txl = textureLoad(channel0,rtx,0);
    var w   = id1;
    for(var i = 0; i < nnI; i++)
    {
        D.nf[w] = reszinp*rtxf[i] + rczWGM[1+1+i];
        w += trnU*trnB;
    }
    //output
    w = id1 + (nnN-nnO)*trnU*trnB;
    for(var i = 0; i < nnO; i++)
    {
        D.nf[w] = reszout*txl[i] + rczWGM[1+1+nnI+i];
        w += trnU*trnB;
    }
}
#calcdefine trnBdnumTreds2 (trnB/numTreds2)
#workgroup_count forwBackwProp trnBdnumTreds2 1 1
@compute @workgroup_size(numTreds2,1,1)
fn forwBackwProp(@builtin(global_invocation_id) id3: vec3u)
{
    var id1 = i32(id3.x);
    //forw propagation
    {
        var er2 = 0f;
        var w1  = nnI*trnU*trnB + id1;
        var lr  = 0;
        for(var i = nnI; i < nnN-nnO; i++)
        {
            var r  = 0f;
            var j0 = id1;
            for(var j = 0; j < i; j++)
            {
                var w = D.nn[lr];       lr++;
                var n = D.nf[j0];       j0+=trnU*trnB;
                r += n*w;
            }   r += D.nn[lr];          lr++;
            var resdrv = act(r);
            var o = resdrv.r;
            var d = resdrv.d;
            D.nf[w1] = o;               w1+=trnB;
            D.nf[w1] = d;               w1+=trnB;w1+=trnB;w1+=trnB;
        }
        for(var i = nnN-nnO; i < nnN; i++)
        {
            var r  = 0f;
            var j0 = id1;
            for(var j = 0; j < nnN-nnO; j++)
            {
                var w = D.nn[lr];       lr++;
                var n = D.nf[j0];       j0+=trnU*trnB;
                r += n*w;
            }   r += D.nn[lr];          lr++;
            var d = D.nf[w1] - r;
            er2 += d*d;
            D.nf[w1] = r;               w1+=trnB;
            D.nf[w1] = d;               w1+=trnB;
            D.nf[w1] = 1.f;             w1+=trnB;w1+=trnB;
        }
        D.nf[w1] = er2;
    }
    //backw propagation
    {
        var lr = nnWB -(nnN-nnO + 1)*nnO + (nnN-nnO) - 1;
        var w1 = id1 + ((nnN-nnO-1)*trnU+1)*trnB;
        for(var i = nnN-nnO; i > nnI; i--)
        {
            var sum = 0f;
            var som = 0f;
            var r   = w1 + trnU*trnB;
            var lr2 = lr;
            for(var j = i; j < nnN; j++)
            {
                sum += D.nn[lr2] * D.nf[r     ];
                som += D.nn[lr2] * D.nf[r+trnB];
                r   += trnU*trnB;
                lr2 += min(j, nnN-nnO)+1;
            }
            var v = D.nf[w1];
            D.nf[w1     ] = v*sum;     //apply derivative*backpro, sum=backpro
            D.nf[w1+trnB] = v*som;     //normalizator
            w1 -= trnU*trnB;
            lr -= i+1;
        }
    }
}
var<workgroup> sum1numTreds3: array<f32,numTreds3>;
#calcdefine nnWBs1 (nnWB+1)
#workgroup_count batchsNNch nnWBs1 1 1  //+1 is the error array
@compute @workgroup_size(numTreds3,1,1)
fn batchsNNch(@builtin(global_invocation_id) id3: vec3u)//compute each batch NN change and reduce batches
{
    var id1 = i32(id3.x);
    var idT = id1 % numTreds3;
    var idB = id1 / numTreds3;
    var np1 = D.lk[idB*2+0];  var mrk1 = np1 != nnWB; //do not load mark
    var np2 = D.lk[idB*2+1];  var mrk2 = np2 != nnWB; //do not load mark
    var n1  = idT + (np1*trnU +0)*trnB;  //source neuron
    var n2  = idT + (np2*trnU +1)*trnB;  //destin neuron
    var n3  = idT + (np2*trnU +2)*trnB;  //destin neuron normalizator
    var sum = 0.f;
    var som = 0.f;
    for(var j = 0; j < trnB/numTreds3; j++)
    {
        var      v1  = 1.f;
        var      v2  = 1.f;
        if(mrk1){v1 *= D.nf[n1];}
        if(mrk2){v1 *= D.nf[n2];}
        if(mrk1){v2 *= D.nf[n1];}
        if(mrk2){v2 *= D.nf[n3];}
        sum += v1;
        som += v2;
        n1 += numTreds3;
        n2 += numTreds3;
        n3 += numTreds3;
    }
    //reduce(sum1numTreds3, id1, numTreds3);
    {
        sum1numTreds3[idT] = sum;
        var idth = id1 % numTreds3;
        for(var stride = numTreds3 >> 1; stride > 0; stride >>= 1){
            workgroupBarrier();
            if(idth < stride){ sum1numTreds3[idth] += sum1numTreds3[idth + stride];}
        }
    }
    if(idT == 0){D.nn2[idB] = sum1numTreds3[0];}
    workgroupBarrier();
    //reduce(sum1numTreds3, id1, numTreds3);
    {
        sum1numTreds3[idT] = som;
        var idth = id1 % numTreds3;
        for(var stride = numTreds3 >> 1; stride > 0; stride >>= 1){
            workgroupBarrier();
            if(idth < stride){ sum1numTreds3[idth] += sum1numTreds3[idth + stride];}
        }
    }
    if(idT == 0){D.nn3[idB] = sum1numTreds3[0];}
}
var<workgroup> sum1numTreds4: array<f32,numTreds4>;
#workgroup_count reduceNNch 1 1 1
@compute @workgroup_size(numTreds4,1,1)
fn reduceNNch(@builtin(global_invocation_id) id3: vec3u)//reduce NN change
{
    var id1 = i32(id3.x);
    var sum = 0.f;
    for(var j = id1; j < nnWB; j+=numTreds4)
    {
        sum += abs(D.nn3[j]);
    }
    //reduce(sum1numTreds4, id1, numTreds4);
    {
        sum1numTreds4[id1] = sum;
        var idth = id1 % numTreds4;
        for(var stride = numTreds4 >> 1; stride > 0; stride >>= 1){
            workgroupBarrier();
            if(idth < stride){ sum1numTreds4[idth] += sum1numTreds4[idth + stride];}
        }
    }
    if(id1 != 0){return;}
    sum = sum1numTreds4[0];
    var count = D.cnt;  D.cnt += 1;
    var err2 = D.nn2[nnWB];
    var ergs = sqrt(err2/f32(trnB*nnO))*(1.f/trnRz);
    var nrml = 1f/sum;  if(sum==0f){nrml = 0f;}
    D.nf[trnB*nnN*trnU] = nrml *8f; //*X because training can handle little more chaos
    
    if(dbg)//show error
    {
        var wm  = count % erraz;
        var w   = (count/erraz) % errhz + errhz*0;
        var vew = ergs;
            vew*= 1.f/f32(erraz);
        if(wm==0){D.err[w] = vew;}
        else     {D.err[w]+= vew;}
    }
}
#calcdefine nnWBdnumTreds5s1 (nnWB/numTreds5+1)
#workgroup_count applyNNch nnWBdnumTreds5s1 1 1
@compute @workgroup_size(numTreds5,1,1)
fn applyNNch(@builtin(global_invocation_id) id3: vec3u)//apply NN change
{
    var id1 = i32(id3.x);
    if(id1 >= nnWB){return;}
    D.nn[id1] += D.nn2[id1] * D.nf[trnB*nnN*trnU];
}
#calcdefine numTreds6Bc numTreds6B
#workgroup_count render numTreds6Bc 1 1
@compute @workgroup_size(numTreds6,1,1)
fn render(@builtin(global_invocation_id) id3: vec3u)
{
    if((time.frame & 63u)!=0u){return;}
    var id1 = i32(id3.x);
    //load rczWGM
    for(var i = id1 % numTreds6; i < rczZ; i += numTreds6)
    {
        rczWGM[i] = D.rcz[i];
    }
    workgroupBarrier();
    var zoom = custom.zoom*1.f;
        //zoom = 2.5f;
    for(var k = id1; k < imgz2; k += imgB)
    {
        //fill input
        D.nf[id1+0*imgB] = ((f32(k % imgz)+.5f)*(1.f/f32(imgz))*2.f - 1.f)*trnRz*zoom;
        D.nf[id1+1*imgB] = ((f32(k / imgz)+.5f)*(1.f/f32(imgz))*2.f - 1.f)*trnRz*zoom;
        //forward propagation
        {
            var lr = 0;
            for(var i = nnI; i < nnN; i++)
            {
                var r  = 0.f;
                var i2 = min(i,nnN-nnO);
                for(var j = 0; j < i2; j++)
                {
                    var w = D.nn[lr];      lr++;
                    var n = D.nf[id1+j*imgB];
                    r += n*w;
                }   r += D.nn[lr];         lr++;
                var resdrv = act(r);
                var o = resdrv.r;
                if(i >= nnN-nnO){o = r;}
                D.nf[id1+i*imgB] = o;
            }
        }
        //pass output
        var col = vec3f(
            D.nf[id1 + (nnN-nnO+0)*imgB],
            D.nf[id1 + (nnN-nnO+1)*imgB],
            D.nf[id1 + (nnN-nnO+2)*imgB]
        );
        var rco = vec3f(
            rczWGM[1+1+nnI+0],
            rczWGM[1+1+nnI+1],
            rczWGM[1+1+nnI+2]
        );
        var rzo = rczWGM[1];
        D.img[k] = (col-rco)/rzo;
        //var r = id1 + (nnN-nnO)*imgB;
        //for(var j = 0; j < nnO; ++j)
        //{
        //    var v = D.nf[r+j*imgB];
        //        v = (v - rczWGM[1+1+nnI+j])/rczWGM[1];
        //}
    }
}
@compute @workgroup_size(8,8,1)
fn render2(@builtin(global_invocation_id) id3: vec3u)
{
    let screenZ = textureDimensions(screen);
    if(any(id3.xy >= screenZ) || (!dbg && any(id3.xy >= vec2u(imgz)))){ return; }
    var col = vec4f(0);
    if(all(id3.xy < vec2u(imgz)))
    {
        col = vec4f(D.img[dot(id3,vec3u(1,imgz,1))],0);
    }
    var y   = 1.f-f32(id3.y)/f32(screenZ.y);
    if(dbg){//total error
        var a = 0f;
        if(id3.x < errhz){a = D.err[id3.x];}
        var yer = a-y;
        yer = max(0.f,1.f-yer*yer*55555.f);
        col = mix(col,vec4f(1,1,1,0),yer);
    }
    if(dbg){//weights and biases interlaced
        var a = 0f;
        if(id3.x < nnWB){a = D.nn[id3.x];}
        var yer = a*.125f-(y*2f-1f);
        yer = max(0.f,1.f-yer*yer*55555.f);
        col = mix(col,vec4f(0,0,1,0),yer);
    }
    if(dbg){//weights and biases interlaced change
        var a = 0f;
        if(id3.x < nnWB){a = D.nn2[id3.x];}
        var yer = a*(1f/f32(trnB)/trnRz)-(y*2f-1f);
        yer = max(0.f,1.f-yer*yer*55555.f);
        col = mix(col,vec4f(0,1,0,0),yer);
    }
    textureStore(screen, id3.xy, col);
}