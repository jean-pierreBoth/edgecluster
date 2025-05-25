using CSV

using Colors, ColorSchemes

using DataFrames


using CairoMakie

using GLMakie

data = DataFrame(CSV.File(digits, header=false));
mat = data[:, 2:end] |> Matrix;
nbclust = 30
images = reshape(mat, (nbclust, 28, 28));