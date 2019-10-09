using Flux
using Glob
using Images
using Base.Iterators: partition
using Flux: onehotbatch,onecold, crossentropy, throttle
using Flux:@treelike
using Statistics:mean

# Using only modern tables
@show("Load image paths")
img_tr = glob("*_t1*.jpg","icdar2019/ICDAR2019_cTDaR/training/TRACKA/ground_truth/")
mask_tr = glob("*_t1*_mask.png","icdar2019/ICDAR2019_cTDaR/training/TRACKA/ground_truth/")

img_ts = glob("*_t1*.jpg","icdar2019/ICDAR2019_cTDaR/test/TRACKA/")
mask_ts = glob("*_t1*_mask.png","icdar2019/ICDAR2019_cTDaR/test/TRACKA/")

function preprocess(imlist)
    imageArray = []
    for image in imlist
        im = load(image)
        im = Gray.(im)  #Convert to grayscale
        im = imresize(im,(256,256))  # Resize to 2^n
        im = convert(Array{Float32},im) # Convert to Float32
        im = reshape(im,256,256,1,:)  # Reshape array
        append!(imageArray,im)
    end
    return Float32.(reshape(imageArray,256,256,1,:))  # Reshape whole array
end

@show("Converting images to arrays")
trainX = preprocess(img_tr)
trainY = preprocess(mask_tr)
testX = preprocess(img_ts)
testY = preprocess(mask_ts)

@show("Create minibatches")
function make_partition_index(X,batch_size)
    idx = partition(1:Int32(length(X)/(256*256*1)),batch_size)
    indices = [(minimum(i),maximum(i)) for i in idx]
    return indices
end

function make_minibatch(X,Y,batch_size)
    typeof(X)
    indices = [i for i in make_partition_index(X,batch_size)]
    minibatch_X = [X[:,:,:,indices[i][1]:indices[i][2]] for i in 1:length(indices)]
    print(typeof(minibatch_X))
    minibatch_Y = [Y[:,:,:,indices[i][1]:indices[i][2]] for i in 1:length(indices)]
    dataset = [(minibatch_X[i],minibatch_Y[i]) for i in 1:length(indices)]
    return dataset
end

train_set = make_minibatch(trainX,trainY,8)
test_set = make_minibatch(testX,testY,8)


@show("Build U-net")

# Neural Network blocks
# Down Block
struct DownBlock
    filter_in
    filter_out
    kernel_size
    padding
    stride
end

function (m::DownBlock)(x)
    c = Conv(m.kernel_size,m.filter_in=>m.filter_out, relu, pad=m.padding, stride=m.stride)(x)
    c = Conv(m.kernel_size,m.filter_out=>m.filter_out, relu, pad=m.padding, stride=m.stride)(c)
    p = MaxPool((2,2))(c)
    return c,p
end
@treelike DownBlock

# Up Block
struct UpBlock
    skip
    filter_in
    filter_out
    kernel_size
    padding
    stride
end

function (m::UpBlock)(x)
    up = ConvTranspose((2,2),m.filter_in=>m.filter_out, relu, stride=(2,2))(x)
    concat = up .+ m.skip
    c = Conv(m.kernel_size,m.filter_out=>m.filter_out,pad=m.padding, stride=m.stride)(concat)
    c = Conv(m.kernel_size,m.filter_out=>m.filter_out,pad=m.padding, stride=m.stride)(c)
    return c
end
@treelike UpBlock

# Bottleneck layer
struct Bottleneck
    filter_in
    filter_out
    kernel_size
    padding
    stride
end

function (m::Bottleneck)(x)
    c = Conv(m.kernel_size,m.filter_in=>m.filter_out, relu, pad=m.padding, stride=m.stride)(x)
    c = Conv(m.kernel_size,m.filter_out=>m.filter_out, relu, pad=m.padding, stride=m.stride)(c)
    return c
end
@treelike Bottleneck

# Unet
struct UNet
    filters
end

function (m::UNet)(x)
    c1,p1 = DownBlock(1,m.filters[1],(3,3),(1,1),(1,1))(x)
    c2,p2 = DownBlock(m.filters[1],m.filters[2],(3,3),(1,1),(1,1))(p1)
    c3,p3 = DownBlock(m.filters[2],m.filters[3],(3,3),(1,1),(1,1))(p2)
    c4,p4 = DownBlock(m.filters[3],m.filters[4],(3,3),(1,1),(1,1))(p3)
    bn = Bottleneck(m.filters[4],m.filters[5],(3,3),(1,1),(1,1))(p4)
    u1 = UpBlock(c4,m.filters[5],m.filters[4],(3,3),(1,1),(1,1))(bn)
    u2 = UpBlock(c3,m.filters[4],m.filters[3],(3,3),(1,1),(1,1))(u1)
    u3 = UpBlock(c2,m.filters[3],m.filters[2],(3,3),(1,1),(1,1))(u2)
    u4 = UpBlock(c1,m.filters[2],m.filters[1],(3,3),(1,1),(1,1))(u3)
    output = Conv((3,3),m.filters[1]=>1,sigmoid)(u4)
    return output
end
@treelike UNet
# Build model
model() = Chain(UNet([64,128,256,512,1024]))

m = model()
