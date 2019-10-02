using Flux
using Images
using FileIO
using Glob
using Mmap
using ImageShow
using Base.Iterators: partition
using Flux: onehotbatch,onecold, crossentropy, throttle


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
X_train = Float64.(reshape(X_train,32,32,3,:))

X_tt,Y_test = loadBatches(testbatch[1])

X_test =[]
for i in 1:size(X_tt)[1]
    append!(X_test,X_tt[i])
end
X_test = Float64.(reshape(X_test,32,32,3,:))

# One-hot encoding
Y_train = onehotbatch(Y_train,0:9)
Y_test = onehotbatch(Y_test,0:9)


function make_partition_index(X,batch_size)
    idx = partition(1:Int64(length(X)/(32*32*3)),batch_size)
    indices = [(minimum(i),maximum(i)) for i in idx]
    return indices
end

function make_minibatch(X,Y,batch_size)
    indices = [i for i in make_partition_index(X,batch_size)]
    minibatch_X = [X[:,:,:,indices[i][1]:indices[i][2]] for i in 1:length(indices)]
    minibatch_Y = [Y[indices[i][1]:indices[i][2]] for i in 1:length(indices)]
    return (minibatch_X,minibatch_Y)
end

# Create minibatches
train_set = make_minibatch(X_train,Y_train,128);
test_set = make_minibatch(X_test,Y_test,1);
