using Flux
using Images
using FileIO
using Glob
using Mmap
using ImageShow
using Base.Iterators: partition
using Flux: onehotbatch,onecold, crossentropy, throttle
using Statistics:mean

# Read CIFAR10 data.(https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz)
# Data is given in binary format where the first byte is the label of
# the first image which is a number between 0 and 9. The next 3072 bytes
# are the values of the pixels. The first 1024 are R, the next 1024 are G and
# the last 1024 are B.

function loadBatches(path)
    batch = Int.(open(read,path))
    labels = [batch[1+(n-1)*3073] for n in 1:1000]
    pixels = [reshape(batch[2+(n-1)*3073:3073+(n-1)*3073],(32,32,3)) for n in 1:1000]/255
    return (pixels,labels)
end

# Get file names
path = "./cifar-10-batches-bin/"
trainbatch = readdir(glob"data_batch_*.bin",path)
testbatch = readdir(glob"test_batch.bin",path)

@show("[INFO] Reading Data...")
# read files and prepare train and test datasets
for file in trainbatch
    if  file==trainbatch[1]
        global X_t, Y_train = loadBatches(file)
    else
        data = loadBatches(file)
        append!(X_t,data[1])
        append!(Y_train,data[2])
    end
end

# Reshape Xt
X_train =[]
for i in 1:size(X_t)[1]
    append!(X_train,X_t[i])
end
X_train = Float32.(reshape(X_train,32,32,3,:))

X_tt,Y_test = loadBatches(testbatch[1])

X_test =[]
for i in 1:size(X_tt)[1]
    append!(X_test,X_tt[i])
end
X_test = Float32.(reshape(X_test,32,32,3,:))


function make_partition_index(X,batch_size)
    idx = partition(1:Int32(length(X)/(32*32*3)),batch_size)
    indices = [(minimum(i),maximum(i)) for i in idx]
    return indices
end

function make_minibatch(X,Y,batch_size)
    indices = [i for i in make_partition_index(X,batch_size)]
    minibatch_X = [X[:,:,:,indices[i][1]:indices[i][2]] for i in 1:length(indices)]
    minibatch_Y = [Y[indices[i][1]:indices[i][2]] for i in 1:length(indices)]
    dataset = [(minibatch_X[i],onehotbatch(minibatch_Y[i],0:9)) for i in 1:length(indices)]
    return dataset
end

@show("[INFO] Creating minibatches...")
# Create minibatches
train_set = gpu.(make_minibatch(X_train,Y_train,128))
test_set = gpu.(make_minibatch(X_test,Y_test,1))
# allowscalar(false)

# VGG16
@show("[INFO] Building CNN...")
model() = Chain(
    # Size 32x32
    Conv((3,3), 3=>64,relu, pad=(1,1), stride=(1,1)),
    BatchNorm(64),
    Dropout(0.3),
    # Size 32x32
    Conv((3,3), 64=>64,relu, pad=(1,1), stride=(1,1)),
    BatchNorm(64),
    # Size 32x32
    x -> MaxPool((2,2))(x),
    # Size 16x16
    Conv((3,3), 64=>128,relu, pad=(1,1), stride=(1,1)),
    BatchNorm(128),
    Dropout(0.3),
    # Size 16x16
    Conv((3,3), 128=>128,relu, pad=(1,1), stride=(1,1)),
    BatchNorm(128),
    x -> MaxPool((2,2))(x),
    # Size 8x8
    Conv((3,3), 128=>256,relu, pad=(1,1), stride=(1,1)),
    BatchNorm(256),
    Dropout(0.4),
    # Size 8x8
    Conv((3,3),256=>256, relu, pad=(1,1), stride=(1,1)),
    BatchNorm(256),
    Dropout(0.4),
    # Size 8x8
    Conv((3,3),256=>256, relu, pad=(1,1), stride=(1,1)),
    BatchNorm(256),
    x -> MaxPool((2,2))(x),
    # Size 4x4
    Conv((3,3), 256=>512,relu, pad=(1,1), stride=(1,1)),
    BatchNorm(512),
    Dropout(0.4),
    # Size 4x4
    Conv((3,3),512=>512, relu, pad=(1,1), stride=(1,1)),
    BatchNorm(512),
    Dropout(0.4),
    # Size 2x2
    x -> reshape(x,:,size(x,4)),
    Dense(8192,4096,relu),
    Dropout(0.5),
    Dense(4096,10),
    softmax
)|>gpu

# Make model

m = model()

loss(x,y) = crossentropy(m(x),y)

# Calculate accuracy
function accuracy(x)
    yhat = [onecold(m(x[i][1]),1:10) for i in 1:size(x)[1]]
    yact = [onecold(x[i][2],1:10) for i in 1:size(x)[1]]
    return mean(yhat .== yact)
end

evalcb = throttle(()-> @show(accuracy(test_set)),10)

opt = ADAM()

@show("[INFO] Training ...")
n_epochs = 100
for i in 1:n_epochs
    Flux.train!(loss, params(m),train_set,opt, cb=evalcb)
end
