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
# Collect parameters:
@treelike DownBlock

struct UpBlock

end




# function down_block(x, filter_in, filter_out, kernel_size=(3,3),padding=(1,1), stride=(1,1))
#     c = Conv(kernel_size,filter_in=>filter_out, relu, pad=padding, stride=stride)(x)
#     c = Conv(kernel_size,filter_out=>filter_out, relu, pad=padding, stride=stride)(c)
#     p = MaxPool((2,2))(c)
#     return c,p
# end



function up_block(x,skip, filter_in,filter_out,kernel_size=(3,3), padding=(1,1), stride=(1,1))
    up = ConvTranspose((2,2),filter_in=>filter_out, relu, stride=(2,2))(x)
    # concat=cat(up,skip;dims=4)
    # print(size(concat))
    concat = up .+ skip
    println(size(concat))
    c = Conv(kernel_size,filter_out=>filter_out,pad=padding, stride=stride)(concat)
    println(size(c))
    c = Conv(kernel_size,filter_out=>filter_out,pad=padding, stride=stride)(c)
    println(size(c))
    return c
end

function bottleneck(x, filter_in, filter_out, kernel_size=(3,3),padding=(1,1),stride=(1,1))
    c = Conv(kernel_size,filter_in=>filter_out, relu, pad=padding, stride=stride)(x)
    c = Conv(kernel_size,filter_out=>filter_out, relu, pad=padding, stride=stride)(c)
    print(typeof(c))
    return c
end

function unet(x)
    filters = [64,128,256,512,1024]
    # Down path
    c1,p1 = down_block(x,1,filters[1])  # 256x256x1 => 128x128x64
    c2,p2 = down_block(p1,filters[1],filters[2]) #128x128x64 => 64x64x128
    c3,p3 = down_block(p2,filters[2],filters[3]) #64x64x128 => 32x32x256
    c4,p4 = down_block(p3,filters[3],filters[4]) #32x32x256 => 16x16x512
    # Bottleneck
    bn = bottleneck(p4,filters[4],filters[5])  #16x16x512 => 16x16x1024
    # Up Path
    u1 = up_block(bn,c4,filters[5],filters[4]) #16x16x1024 => 32x32x512
    u2 = up_block(u1,c3,filters[4],filters[3]) #32x32x512 => 64x64x256
    u3 = up_block(u2,c2,filters[3],filters[2]) #64x64x256 => 128x128x128
    u4 = up_block(u3,c1,filters[2],filters[1]) #128x128x128 => 256x256x64

    output = Conv((3,3),64=>1,sigmoid)(u4)  #256x256x64 => 256x256x1
    return output
end


model() = Chain(unet)

am = model()
params(am)

model() = Chain(Dense(3,3))

mx = model()

params(mx)
