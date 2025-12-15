using ITensors, ITensorMPS, Plots, HDF5, Random, TOML

include("GroundState.jl")

params = TOML.parsefile("input.toml")

N = params["N"]
Nup = params["Nup"]
Ndn = params["Ndn"]

U= params["U"]
V= params["V"]

pbc = params["pbc"]

@show N, Nup, Ndn
@show U, V
@show pbc

psi = gs_calcs(N,U,V,Nup,Ndn,pbc)

fname = "psi_N$(N)_Nup$(Nup)_Ndn$(Ndn)_U$(U)_V$(abs(V)).h5"
f = h5open(fname,"w")
    write(f,"psi",psi)
close(f)
    
println()
