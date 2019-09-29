using Flux
using Images
using FileIO
using Glob
using Mmap
using ImageShow

# Read CIFAR10 data.
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
        global X_train, Y_train = loadBatches(file)
    else
        data = loadBatches(file)
        append!(X_train,data[1])
        append!(Y_train,data[2])
    end
end

X_test,Y_test = loadBatches(testbatch[1])

# Build NNet
