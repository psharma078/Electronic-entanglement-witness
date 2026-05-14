using MPI

using ITensors, ITensorMPS, Plots, HDF5, Random, TOML

# Initialize MPI
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

include("measureEngine.jl")

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
    println("...Julia warmup 2CRDM... ")
end

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

s_warm = siteinds("Electron", 10)
phi_warm = randomMPS(s_warm, 10)
m_warm,idx_warm = measurement_sites(Ns=div(10,2), Ncut=0)
m_warm = [Tuple(2x - 1 for x in m) for m in m_warm] 
d_warm = idx_warm[end][end]
Cuu_warm = four_pt_corr_spinless_mpi(phi_warm, ("Cdagup", "Cdagup", "Cup", "Cup"), m_warm)
Cud_warm = four_pt_corr_spinfull_mpi(phi_warm, ("Cdagup","Cdagdn", "Cdn", "Cup"), m_warm)

if rank==0
    Cuu_warm,Cud_warm = @time compute_2CRDM(phi_warm,m_warm,idx_warm,Cuu_warm,Cud_warm,Cuu_warm,spinful=true,TRS=true)
    amin,bmax = @time C2RDM_eigenSolver(Cuu_warm, Cud_warm, d_warm)
    println("warm up done...")
    println()
end

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

fname = "psi_N$(N)_Nup$(Nup)_Ndn$(Ndn)_U$(U)_w$(w)_g$(g)_gp$(g1)_LBOdim$(LBO_dims[end]).h5"
f = h5open(fname,"r")
    psi = read(f,"psi",MPS)
close(f)
MPI.Barrier(comm)
   

measureSites, indices = measurement_sites(Ns=div(N,2), Ncut=Ncut)
elSites = [Tuple(2x - 1 for x in m) for m in measureSites] #for el-ph system
dimensn = indices[end][end]
if rank == 0
    println("measurement between the sites $(measureSites[1][1]) and $(measureSites[end][end]).")
    println("computing four point correlator up-up channel")
end

ops = ("Cdagup","Cdagup", "Cup", "Cup")

upup_2RDM = @time four_pt_corr_spinless_mpi(psi, ops, elSites)
MPI.Barrier(comm)

ops = ("Cdagup","Cdagdn", "Cdn", "Cup")
if rank==0 println("computing four point correlator up-dn channel") end
updn_2RDM = @time four_pt_corr_spinfull_mpi(psi, ops, elSites)

if rank==0
    dndn_2RDM = nothing
    println()
    println("computing CRDM and eigen values...")
    C2RDM_upup,C2RDM_updn = @time compute_2CRDM(psi,elSites,indices,upup_2RDM,updn_2RDM,dndn_2RDM,spinful=true,TRS=true)
    lmin, lmax = @time C2RDM_eigenSolver(C2RDM_upup, C2RDM_updn, dimensn)
    println()
    println("Ground state entanglement witness")
    @show lmin
    @show lmax
end


#MPI.Barrier(comm)
MPI.Finalize()
 
GC.gc(true)

