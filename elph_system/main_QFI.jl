using ITensors, ITensorMPS, Plots, HDF5, Random, TOML

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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Generic function for QFI density in 1D chain
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
function qfi_density(zizjcorr::Matrix{Float64}, zi::Vector{Float64}, exp_iqr::Matrix{ComplexF64})
    N = length(zi)
    zz_connected = vec([zizjcorr[i ,j] - zi[i]*zi[j] for i in 1:N, j in 1:N])
    QFI_q = exp_iqr * zz_connected  ## matrix mult (N,N^2) x (N^2,1) => (N,1). (i,j) being summed over
    return 4.0/N * QFI_q
end

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

s_warm = siteinds("Electron", 6)
phi_warm = randomMPS(s_warm, 10)
zzcorr = @time correlation_matrix(phi_warm,"Sz","Sz")
println("warm up done...")
println()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

fname = "psi_N$(N)_Nup$(Nup)_Ndn$(Ndn)_U$(U)_w$(w)_g$(g)_gp$(g1)_LBOdim$(LBO_dims[end]).h5"
f = h5open(fname,"r")
    psi = read(f,"psi",MPS)
close(f)  

Ncut = 2*Ncut
elSites = [i for i in (Ncut+1):(N-Ncut) if isodd(i)]

Sz = expect(psi, "Sz", sites = elSites)

Ns = length(elSites)

rlist = vec([i-j for i in 1:Ns, j in 1:Ns])
qlist = vec([n * 2*pi/Ns for n in 0:Ns])
qr = [q * r for q in qlist, r in rlist]
exp_iqr = exp.(im * qr)

zzcorr = @time correlation_matrix(psi,"Sz","Sz",sites=elSites)
qfi_gs = @time real.(qfi_density(zzcorr, Sz, exp_iqr))

@show qfi_gs
@show round.(qlist, digits=4)

println()

qid = argmax(qfi_gs)
qmax = qlist[qid]
QFImax = qfi_gs[qid]
@show qid
println("qmax : ", qmax)
println("QFImax : ", QFImax)
