#include("dmrg_lbo.jl")

#conserve electron qns without conserving phonons
function ITensors.space(
  ::SiteType"Qudit";
  dim=2,
  conserve_qns=false,
  conserve_number=conserve_qns,
  qnname_number="Number",
)
  if conserve_number
    return [QN(qnname_number, n - 1) => 1 for n in 1:dim]
  else
    return [QN() => dim]
  end
  return dim
end

function mixed_sites(N::Int,max_boson_dim::Int)
  sites = Vector{Index}(undef,N)
  for n=1:N
    if isodd(n)
      sites[n] = siteind("Electron"; addtags="n=$n", conserve_qns=true)
    else
      sites[n] = siteind("Boson"; addtags ="n=$n", dim = max_boson_dim, conserve_qns=false)
    end
  end
  return sites
end

function Hubbard_exHolstein_1D(N::Int64,U::Float64,V::Float64,ω::Float64,g::Float64, g1::Float64,pbc::Bool=false)
    t = 1.0

    os = OpSum()

    for j in 1:2:(N-2)
    	os += -t, "Cdagup", j, "Cup", j+2
    	os += -t, "Cdagup", j+2, "Cup", j
    	os += -t, "Cdagdn", j, "Cdn", j+2
    	os += -t, "Cdagdn", j+2, "Cdn", j
    end
    #periodic boundary condition
    if pbc
        os += -t, "Cdagup", N-1, "Cup", 1
        os += -t, "Cdagup", 1, "Cup", N-1
        os += -t, "Cdagdn", N-1, "Cdn", 1
        os += -t, "Cdagdn", 1, "Cdn", N-1
    end

    #Hubbard U
    for j in 1:2:N-1
        os += U, "Nupdn", j
    end

    #Extended interaction "V"
    for j in 1:2:N-2
        os += V, "Ntot", j, "Ntot", j+2
    end

    if pbc
        os += V, "Ntot", N-1, "Ntot", 1
    end
 
    #on-site el-ph coupling  
    for j in 1:2:N-1
    	os += ω, "N", j+1  #Einstein phonons
    	os += g, "Ntot", j, "A", j+1
    	os += g, "Ntot", j, "Adag", j+1
    end

    #Nearest neighbor coupling
    for j in 1:2:N-1
        if j%(N-1)!=0
            os += g1, "Ntot", j, "A", j+3
            os += g1, "Ntot", j, "Adag", j+3
        end
        if j!=1
            os += g1, "Ntot", j, "A", j-1
            os += g1, "Ntot", j, "Adag", j-1
        end
    end

    if pbc
        os += g1, "Ntot", N-1, "A", 2
        os += g1, "Ntot", N-1, "Adag", 2
        os += g1, "Ntot", 1, "A", N
        os += g1, "Ntot", 1, "Adag", N
    end
 
    return os
end

function gs_calcs(N,U,V,Nup,Ndn,ω,g,g1,barePhDim,LBO_dims,init_n,pbc)
    os = Hubbard_exHolstein_1D(N, U, V, ω, g, g1, pbc)

    #fname = "psi_N$(N)_Nup$(Nup)_Ndn$(Ndn)_U$(U)_w$(w)_g$(g)_LBOdim$(LBO_dims[end]).h5"
    #f = h5open(fname,"r")
    #psi0 = read(f,"psi",MPS)
    #close(f)

    #sites = siteinds(psi0)
    
    sites = mixed_sites(N,barePhDim)
    H = MPO(os,sites)

    ntot = Nup + Ndn  #init state for dopped system
    select_el_sites = sort(shuffle(collect(1:2:N))[1:ntot])
    state = [isodd(n) ? "Emp" : "$(init_n)" for n in 1:N]
    for (id,s) in enumerate(select_el_sites)
        state[s] = isodd(id) ? "Up" : "Dn"
    end

    println("initial state")
    @show state
    psi0 = MPS(sites,state)
    @show expect(psi0,"N",sites=[n for n in 2:2:N])
    
    nsweeps = 14
    maxdim = [8,16,32,50,100,100,100,200,200,200,400,400,400,800,800,800,1600]
    cutoff = [2e-7]
    noise = [1E-2,1E-2,1E-2,1E-3,1E-3,1E-3,1E-4,1E-4,1e-4,1E-4,1E-5,1E-5,1e-6, 1e-6,1E-6,1E-6,1e-8,0.0,0.0,0.0]
    #significant noise in the starting sweep is important. This is checked and verified.

    obs = DMRGObserver(; energy_tol=1e-10)
    #energy_bare, psi = dmrg(H,psi0;nsweeps=nsweeps,maxdim=maxdim,cutoff=cutoff,noise=noise,observer=obs,eigsolve_krylovdim= 8);
    #@show expect(psi, "N", sites = [n for n in 2:2:N])

    #LBO dmrg starts
    energy_lbo, psi  = dmrg_lbo(H, psi0; nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff, noise=noise, observer=obs,
    		                eigsolve_maxiter=20,eigsolve_krylovdim= 8, LBO=true, max_LBO_dim=LBO_dims,min_LBO_dim=4);
    println()
    Nboson = expect(psi,"N",sites=[n for n in 2:2:N])
    @show Nboson
    println()

    ntot = expect(psi,"Ntot", sites=1:2:N)
    @show ntot    

    Sz = expect(psi,"Sz", sites=1:2:N)
    @show Sz

    file_name = "data_L$(N)_Nup$(Nup)_Ndn$(Ndn)_U$(U)_w$(ω)_g$(g).h5"

    h5write(file_name, "ntot", ntot)

    ninj = correlation_matrix(psi, "Ntot", "Ntot", sites=1:2:N)
    h5write(file_name, "ninj", ninj)

    SzjSzj = correlation_matrix(psi, "Sz", "Sz", sites=1:2:N)
    h5write(file_name, "SziSzj", ninj)

    #single particle
    CdagC = @time correlation_matrix(psi, "Cdagup", "Cup")
    h5write(file_name, "CdagC", CdagC)
    println("single particle cals done")

    println()


    return psi
end
