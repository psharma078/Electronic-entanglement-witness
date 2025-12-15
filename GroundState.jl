function exHubbard_1D(N::Int64,U::Float64,V::Float64,pbc::Bool=false)
    t = 1.0

    os = OpSum()
    for j in 1:N-1
    	os += -t, "Cdagup", j, "Cup", j+1
    	os += -t, "Cdagup", j+1, "Cup", j
    	os += -t, "Cdagdn", j, "Cdn", j+1
    	os += -t, "Cdagdn", j+1, "Cdn", j
    end

    #periodic boundary condition
    if pbc
        os += -t, "Cdagup", N, "Cup", 1
        os += -t, "Cdagup", 1, "Cup", N
        os += -t, "Cdagdn", N, "Cdn", 1
        os += -t, "Cdagdn", 1, "Cdn", N
    end

    #Hubbard U
    for j in 1:N
        os += U, "Nupdn", j
    end

    #Extended interaction "V"
    for j in 1:N-1
        os += V, "Ntot", j, "Ntot", j+1
    end

    if pbc
        os += V, "Ntot", N, "Ntot", 1
    end
 
    return os
end

function gs_calcs(N,U,V,Nup,Ndn,pbc)
    os = exHubbard_1D(N, U, V, pbc)
    
    sites = siteinds("Electron", N, conserve_qns=true)
    H = MPO(os,sites)

    ntot = Nup + Ndn  #init state for dopped system
    select_el_sites = sort(shuffle(collect(1:N))[1:ntot])
    state = ["Emp" for n in 1:N]
    for (id,s) in enumerate(select_el_sites)
        state[s] = isodd(id) ? "Up" : "Dn"
    end

    println("initial state")
    @show state
    psi0 = MPS(sites,state)
    
    nsweeps = 20
    maxdim = [8,16,32,50,100,100,100,200,200,200,400,400,400,800,800,800,1600]
    cutoff = [1e-7]
    noise = [1E-4,1E-5,1E-5,1e-6, 1e-6,1e-8,0.0,0.0,0.0]

    obs = DMRGObserver(; energy_tol=1e-10)
    energy, psi = dmrg(H,psi0;nsweeps=nsweeps,maxdim=maxdim,cutoff=cutoff,noise=noise,observer=obs,eigsolve_krylovdim= 8);
    @show expect(psi, "Sz")

    file_name = "data_L$(N)_Nup$(Nup)_Ndn$(Ndn)_U$(U)_V$(V).h5"


    #electron occupation
    ntot = expect(psi,"Ntot")
    h5write(file_name, "ntot", ntot)

    ntot2 = expect(psi,"Ntot * Ntot")
    h5write(file_name, "ntot2", ntot2)

    ninj = correlation_matrix(psi, "Ntot", "Ntot")
    h5write(file_name, "ninj", ninj)

    SzjSzj = correlation_matrix(psi, "Sz", "Sz")
    h5write(file_name, "SziSzj", ninj)
    
    #single particle
    CdagC = @time correlation_matrix(psi, "Cdagup", "Cup")
    h5write(file_name, "CdagC", CdagC)
    println("single particle cals done")
    println()

    return psi
end

