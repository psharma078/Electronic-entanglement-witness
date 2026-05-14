using MPI
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)
#println("Hello from rank $rank out of $nprocs")

using ITensors, ITensorMPS, HDF5, TOML

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

if rank==0
    @show N, Nup, Ndn
    @show U, V
    @show w, g, g1
    println(" ")
end

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
function four_pt_correlator(psi::MPS, ops::NTuple{4, String}, op_sites::Vector{NTuple{4, Int64}})
    s = siteinds(psi)
    n_ops = length(op_sites)

    partial_corr = zeros(ComplexF64, n_ops)

    for idx in rank+1:nprocs:n_ops
        s1, s2, s3, s4 = op_sites[idx]
        os = OpSum()
        os += ops[1], s1, ops[2], s2, ops[3], s3, ops[4], s4
        corr = MPO(os, s)
        partial_corr[idx] = inner(psi', corr, psi)
    end

    return  MPI.Reduce(partial_corr, +, 0, comm)
end


fname = "psi_N$(N)_Nup$(Nup)_Ndn$(Ndn)_U$(U)_w$(w)_g$(g)_gp$(g1)_LBOdim$(LBO_dims[end]).h5"
f = h5open(fname,"r")
    psi = read(f,"psi",MPS)
close(f)

L = div(N,2)
file_name = "data_L$(L)_Nup_Ndn$(Nup)_g$(g)_gp$(g1)_lboDim$(LBO_dims[end]).h5"

if rank==0
    #boson occupation
    Nboson = expect(psi,"N",sites=[n for n in 2:2:N])
    h5write(file_name, "Nboson", Nboson)

    #electron occupation
    ntot = expect(psi,"Ntot",sites=[n for n in 1:2:N])
    h5write(file_name, "ntot", ntot)

    #single particle
    CdagC = @time correlation_matrix(psi, "Cdagup", "Cup", sites=1:2:N)
    h5write(file_name, "CdagC", CdagC)
    println("single particle cals done")
    println()

    #charge-charge correlation
    ninj = @time correlation_matrix(psi, "Ntot", "Ntot", sites=1:2:N)
    h5write(file_name, "ninj", ninj)
    println("charge correlation cals done")
    println()
 
    SziSzj = @time correlation_matrix(psi, "Sz", "Sz", sites=1:2:N)
    h5write(file_name, "SziSzj", SziSzj)

    #s-wave pairing
    Ps_wave = @time correlation_matrix(psi, "Cdagup * Cdagdn", "Cdn * Cup", sites=1:2:N)
    h5write(file_name, "Pair_swave", Ps_wave)
    println("s-wave cals done")
    println()
end

#p-wave pairing
op_sites = vec([(i, i+2, j+2, j) for i in 1:2:N-2, j in 1:2:N-2])
op1 = ("Cdagup","Cdagdn","Cdn","Cup")
op2 = ("Cdagup","Cdagdn","Cup","Cdn")
op3 = ("Cdagdn","Cdagup","Cdn","Cup")
op4 = ("Cdagdn","Cdagup","Cup","Cdn")

uddu = @time four_pt_correlator(psi, op1, op_sites)
udud = @time four_pt_correlator(psi, op2, op_sites)
dudu = @time four_pt_correlator(psi, op3, op_sites)
duud = @time four_pt_correlator(psi, op4, op_sites)

if rank == 0
    h5write(file_name, "op_sites", op_sites)
    h5write(file_name, "uddu", uddu)
    h5write(file_name, "udud", udud)
    h5write(file_name, "dudu", dudu)
    h5write(file_name, "duud", duud)
    println("pairing cals done")
end
