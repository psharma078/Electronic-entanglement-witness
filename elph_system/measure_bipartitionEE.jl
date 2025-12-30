using ITensors, ITensorMPS, Plots, HDF5, Random, TOML

#Bipartition EE of quantum system
function vN_entropy(psi::MPS,b::Int)
    psi = orthogonalize(psi, b)
    U,S,V = svd(psi[b], (linkinds(psi, b-1)..., siteinds(psi, b)...), cutoff=1e-6)
    SvN = 0.0
    for n=1:dim(S, 1)
        p = S[n,n]^2
        SvN -= p * log(p)
    end
    return SvN
end


#Swap el-ph local MPS sites
function swap_sites!(psi::MPS, i::Int, sites, maxDim::Int)
    # tensors
    A = psi[i]
    B = psi[i+1]

    # extract indices
    si = sites[i]
    sj = sites[i+1]

    li = linkind(psi, i-1)
    lj = linkind(psi, i+1)

    # contract A and B
    T = A * B

    # new order: sj, si
    inds_new = (li, sj, si, lj)
    T = permute(T, inds_new...)

    # split back
    U, S, V = svd(T, (li, sj); cutoff=1e-6, maxdim=maxDim)

    psi[i]   = U
    psi[i+1] = S * V

    #println(i, " maxlinkdim = ", maxlinkdim(psi))

    # update sites vector
    sites[i], sites[i+1] = sites[i+1], sites[i]
end

#We need all the electron sites to the left and phonons to the right
function electrons_left!(psi, sites, maxDim)
    N = length(sites)
    for i in 1:(N-1)
            if (hastags(sites[i], "Boson") || hastags(sites[i], "Trunc")) && hastags(sites[i+1], "Electron")
            swap_sites!(psi, i, sites, maxDim)
        end
    end
end

#Full sort [e,e,e,e...,e][b,b,b,b,...,b]
function sort_electron_phonon!(psi, sites)
    maxDim = maxlinkdim(psi)+500
    for _ in 1:length(sites)
        electrons_left!(psi, sites, maxDim)
    end
    return psi
end

if length(ARGS) < 1
    error("Usage: julia run.jl input_file.toml")
end
params = TOML.parsefile(ARGS[1])

N = params["N"]
Ncut = params["Ncut"]

Nup = params["Nup"]
Ndn = params["Ndn"]

U= params["U"]
V= params["V"]
w= params["w"]
g= params["g"]
g1= params["g1"]
LBO_dims = params["LBO_dims"]

    @show N, Nup, Ndn
    @show U, V
    @show w, g, g1
    println(" ")
    println("...Julia warmup 2CRDM... ")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fname = "psi_N$(N)_Nup$(Nup)_Ndn$(Ndn)_U$(U)_w$(w)_g$(g)_LBOdim$(LBO_dims[end]).h5"
f = h5open(fname,"r")
    psi = read(f,"psi",MPS)
close(f)

N = length(psi)
SvN_halfchain = vN_entropy(psi,div(N,2))
@show SvN_halfchain

sites = siteinds(psi)

@show expect(psi,"N",sites=2:2:N)

psi = sort_electron_phonon!(psi, sites)
println()

@show expect(psi,"N",sites=1+div(N,2):N)

SvN_elphPartition = vN_entropy(psi,div(N,2)) #electron-phonon partition
SvN_elelphPartition = vN_entropy(psi,div(N,4)) #el-(el-ph) partition (traces out half of the electrons and all phonons)

@show SvN_elphPartition
@show SvN_elelphPartition
