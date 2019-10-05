using Images, Colors
using FileIO
using ImageView
using LinearAlgebra
using LightXML
using Glob

function getCoords(fileName)
    xml = parse_file(fileName)
    xroot = root(xml)
    ces = xroot["table"]
    out = []
    for i in 1:length(ces)
        t=find_element(ces[i], "Coords")
        p = attribute(t,"points")
        a = [split(i," ") for i in split(p,",")]
        c = collect(Iterators.flatten(a))
        toInt = [parse(Int,num) for num in c]
        xs=[toInt[2+(n-1)*2] for n in 1:4]
        ys=[toInt[1+(n-1)*2] for n in 1:4]
        xmin = minimum(xs)
        xmax = maximum(xs)
        ymin = minimum(ys)
        ymax = maximum(ys)
        coords = [xmin,xmax,ymin,ymax]
        append!(out,coords)
    end
    return reshape(out,(4,length(ces)))
end

function createMask(filename)
    coords = getCoords(filename)
    img = load(replace(filename,".xml"=>".jpg"))
    mask = zeros(size(img))
    for i in 1:size(coords)[2]
        mask[coords[1,i]:coords[2,i],coords[3,i]:coords[4,i]] .= 1;
    end
    return img .* mask
end


createMask("icdar2019/ICDAR2019_cTDaR/training/TRACKA/ground_truth/cTDaR_t10569.xml")
