using ITensors, ITensorMPS, Plots, HDF5, Random, TOML

include("/work2/10514/sharmaprakash078/frontera/entanglement_witness/newProject/dmrg_lbo.jl")
include("GroundState.jl")

#Get input file
if length(ARGS)<1
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

bare_n = params["bare_n"]
init_n = params["init_n"]
LBO_dims = params["LBO_dims"]
pbc = params["pbc"]

@show N, Nup, Ndn
@show U, V
@show w, g, g1
@show bare_n, init_n
@show pbc

psi = gs_calcs(N,U,V,Nup,Ndn,w,g,g1,bare_n,LBO_dims,init_n,pbc)

fname = "psi_N$(N)_Nup$(Nup)_Ndn$(Ndn)_U$(U)_w$(w)_g$(g)_LBOdim$(LBO_dims[end]).h5"
f = h5open(fname,"w")
    write(f,"psi",psi)
close(f)    
    
println()
