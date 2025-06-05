using CSV

using Colors, ColorSchemes

using DataFrames


using CairoMakie

using GLMakie

using Clustering

using CPUTime

function mnist()
    data = DataFrame(CSV.File(file, header=false))
    # skip label
    mat = data[:, 2:end] |> Matrix
    nbdata = size(data)[1]
    # reshape to plot
    images = reshape(mat, (nbdata, 28, 28))
    #
    tmat = transpose(mat)
    @CPUtime @time kres = kmeans(mat, 25, maxiter=100, display=:iter)
end

function higgs(filename)
    data = DataFrame(CSV.File(filename))
    # skip label
    mat = data[:, 2:end] |> Matrix
    #
    @CPUtime @time kres = kmeans(transpose(mat), 25, maxiter=100, display=:iter)
end
