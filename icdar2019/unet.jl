using Flux
using Glob
using Images
using Base.Iterators: partition
using Flux: onehotbatch,onecold, crossentropy, throttle
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

function down_block(x, filter_in, filter_out, kernel_size=(3,3),padding=(1,1), stride=(1,1))
    c = Conv(kernel_size,filter_in=>filter_out, relu, pad=padding, stride=stride)(x)
    c = Conv(kernel_size,filter_out=>filter_out, relu, pad=padding, stride=stride)(c)
    p = MaxPool((2,2))(c)
    return c,p
end

function up_block(x,skip, filter_in,filter_out,kernel_size=(3,3), padding=(1,1), stride=(1,1))
    up = ConvTranspose((2,2))(x)
    concat=hcat(up,skip)
    c = Conv(kernel_size,filter_in=>filter_out,pad=padding, stride=stride)(concat)
    c = Conv(kernel_size,filter_out=>filter_out,pad=padding, stride=stride)(c)
    return c
end

function bottleneck(x, filter_in, filter_out, kernel_size=(3,3),padding=(1,1),stride=(1,1))
    c = Conv(kernel_size,filter_in=>filter_out, relu, pad=padding, stride=stride)(x)
    c = Conv(kernel_size,filter_out=>filter_out, relu, pad=padding, stride=stride)(c)
    return c
end
