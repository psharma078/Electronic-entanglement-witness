using Arpack, Base.Threads #, ITensors, ITensorMPS
using Observers: observer

#@show nprocs

## Enabling multithreading option 
function two_pt_correlator(psi::MPS, ops::NTuple{2, String}, op_sites, idx_pairs)
    ss = siteinds(psi)
    len = Int(round(√length(op_sites),digits=2))
    partial_Corr = [zeros(Complex{Float64}, len, len) for _ in 1:nthreads()]
    @show nthreads()
    @threads for idx in 1:length(op_sites)
        thread_id = threadid()
        s1, s2 = op_sites[idx]

        op = OpSum()
        op += ops[1], s1, ops[2], s2
        corr_mpo = MPO(op, ss)
        corr_2p = inner(psi', corr_mpo, psi)

        # Accumulate into the thread-local result
        ii, jj = idx_pairs[idx]
        partial_Corr[thread_id][ii, jj] = corr_2p
    end
    #sum the threads
    Corr = sum(partial_Corr)
    return Corr
end

# Generic four point correlator
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


# MPI parallelized four-point correlator function
function four_pt_corr_spinless_mpi(psi::MPS, ops::NTuple{4, String}, op_sites::Vector{NTuple{4, Int64}})
    s = siteinds(psi)
    n_ops = length(op_sites)

    index_map = Dict(site_tuple => index for (index, site_tuple) in enumerate(op_sites))
    local_corr = zeros(ComplexF64, n_ops)

    # Distribute work
    for idx in rank+1:nprocs:n_ops
        s1, s2, s3, s4 = op_sites[idx]
        if (s1 > s2) && (s3 > s4)
            os = OpSum()
            os += ops[1], s1, ops[2], s2, ops[3], s3, ops[4], s4
            corr = inner(psi', MPO(os, s), psi)
            local_corr[idx] = corr

            # Exploit symmetry to reduce computation
            new_idx = get(index_map, (s2, s1, s4, s3), nothing)
            local_corr[new_idx] = corr

            new_idx = get(index_map, (s2, s1, s3, s4), nothing)
            local_corr[new_idx] = -corr

            new_idx = get(index_map, (s1, s2, s4, s3), nothing)
            local_corr[new_idx] = -corr
        end
    end

    # Reduce all local results to global result at rank 0
    global_corr = MPI.Reduce(local_corr, +, 0, comm)
    return global_corr
end

function four_pt_corr_spinfull_mpi(psi::MPS, ops::NTuple{4, String}, op_sites::Vector{NTuple{4, Int64}})
    s = siteinds(psi)
    n_ops = length(op_sites)

    index_map = Dict(site_tuple => index for (index, site_tuple) in enumerate(op_sites))
   
    partial_corr =  zeros(ComplexF64, n_ops)

    # Distribute work
    for idx in rank+1:nprocs:n_ops
        s1, s2, s3, s4 = op_sites[idx]
        if (s1 >= s2) && (s3 >= s4)
            os = OpSum()
            os += ops[1], s1, ops[2], s2, ops[3], s3, ops[4], s4
            corr = inner(psi', MPO(os, s), psi)

            partial_corr[idx] = corr

            if !((s1 == s2) && (s3 == s4))
                new_idx = get(index_map, (s2, s1, s4, s3), nothing)
                partial_corr[new_idx] = corr
            end

        elseif (s1 > s2) && (s3 < s4)
            os = OpSum()
            os += ops[1], s1, ops[2], s2, ops[3], s3, ops[4], s4
            corr = inner(psi', MPO(os, s), psi)
            partial_corr[idx] = corr

            new_idx = get(index_map, (s2, s1, s4, s3), nothing)
            partial_corr[new_idx] = corr
        end
    end

    return MPI.Reduce(partial_corr, +, 0, comm)
end

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# pass TRS=false if Nup!=Ndn or if TRS is broken. 
# State with TRS follows (i,j,k,l)<->(j,i,l,k) in the up-dn cahnnel of 4 pt correlator
# This code works only for spin conserved system.
# To leverage spin non-conserved system add missing single particle terms in updn and dnup channels.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
function compute_2CRDM(psi::MPS, op_sites::Vector{NTuple{4, Int64}}, indices::Vector{NTuple{4, Int64}}, 
				upup_2RDM, updn_2RDM, dndn_2RDM; spinful::Bool=true, TRS::Bool=true)
    ##1. First, compute 2 point correlation
    site_start = op_sites[1][1]
    site_end = op_sites[end][end]
    el_sites = collect(site_start:site_end)    #for electronic system only
    #el_sites = collect(site_start:2:site_end)  #for el-ph (only odd sites)
    idx = collect(1:length(el_sites))
    idx_pairs = vec([(j, i) for i in idx, j in idx])
    site_pairs=[(el_sites[m],el_sites[n]) for (m,n) in idx_pairs]
    
    corr_2pt_up = two_pt_correlator(psi, ("Cdagup", "Cup"), site_pairs, idx_pairs)
    corr_2pt_dn = nothing
    if TRS==false
        corr_2pt_dn = two_pt_correlator(psi, ("Cdagdn", "Cdn"), site_pairs,idx_pairs)
    end
    

    ##2. compute 2CRDM 
    Cupup_single = vec([-corr_2pt_up[u,x] * corr_2pt_up[v,w] + corr_2pt_up[u,w] * corr_2pt_up[v,x]
                    for (u, v, w, x) in indices])
    upup_2CRDM =  upup_2RDM + Cupup_single

    updn_2CRDM = nothing
    dndn_2CRDM = nothing
    if spinful
        if TRS
            Cupdn_single = vec([-corr_2pt_up[u,x] * corr_2pt_up[v,w] for (u, v, w, x) in indices])
            updn_2CRDM =  updn_2RDM + Cupdn_single
            println("returned 2CRDM for up-up and up-dn channels.")
	    
            return upup_2CRDM, updn_2CRDM
        
        else
            Cupdn_single = vec([-corr_2pt_up[u,x] * corr_2pt_dn[v,w] for (u, v, w, x) in indices])
            updn_2CRDM =  updn_2RDM + Cupdn_single

            Cdndn_single = vec([-corr_2pt_dn[u,x] * corr_2pt_dn[v,w] + corr_2pt_dn[u,w] * corr_2pt_dn[v,x]
                        for (u, v, w, x) in indices])
            dndn_2CRDM = dndn_2RDM + Cdndn_single
            println("returned 2CRDM for up-up, up-dn, and dndn channels.")
            return upup_2CRDM, updn_2CRDM, dndn_2CRDM
        end

    else
        println("returned single 2CRDM as spin = ", spinful)
        return upup_2CRDM
    end
end

#generic function for 2CRDM without symmetry
function compute_2CRDM_noSymm(psi::MPS, op_sites::Vector{NTuple{4, Int64}}, indices::Vector{NTuple{4, Int64}})
    ##1. first compute 2 points correlation
    site_start = op_sites[1][1]
    site_end = op_sites[end][end]
    #el_sites = collect(site_start:site_end)    #for electronic system only
    el_sites = collect(site_start:2:site_end)  #for el-ph (only odd sites for electrons)
    idx = collect(1:length(el_sites))
    idx_pairs = vec([(j, i) for i in idx, j in idx])
    site_pairs=[(el_sites[m],el_sites[n]) for (m,n) in idx_pairs]
    @show idx_pairs
    println()
    @show site_pairs
    
    corr_2pt_up = two_pt_correlator(psi, ("Cdagup", "Cup"), site_pairs, idx_pairs)
    corr_2pt_dn = two_pt_correlator(psi, ("Cdagdn", "Cdn"), site_pairs, idx_pairs)
    
    ##2. compute 4 points correlation for up-up channel
    s = siteinds(psi)
    ops = ("Cdagup","Cdagup", "Cup", "Cup")
    upup_2RDM = four_pt_correlator(psi, ops, op_sites)
    
    ##3. compute 4 points correlation for up-dn channel
    ops = ("Cdagup","Cdagdn", "Cdn", "Cup")
    updn_2RDM = four_pt_correlator(psi, ops, op_sites)
    #@show round.(real(updn_2RDM),digits=3)
    
    ##4. compute 2CRDM for up-up channel
    Cupup_single = vec([-corr_2pt_up[u,x] * corr_2pt_up[v,w] + corr_2pt_up[u,w] * corr_2pt_up[v,x]
            for (u, v, w, x) in indices])
    upup_2CRDM =  upup_2RDM + Cupup_single

    ##5. compute 2CRDM for up-dn channel
    Cupdn_single = vec([-corr_2pt_up[u,x] * corr_2pt_dn[v,w] for (u, v, w, x) in indices])
    updn_2CRDM =  updn_2RDM + Cupdn_single

    ###### ADDED ###############
    ##added. compute 4 points correlation for dn-up channel
    ops = ("Cdagdn","Cdagup", "Cup", "Cdn")
    dnup_2RDM = four_pt_correlator(psi, ops, op_sites)
    ##For dn-dn channel
    ops = ("Cdagdn","Cdagdn", "Cdn", "Cdn")
    dndn_2RDM = four_pt_correlator(psi, ops, op_sites)
    
    ##added. compute 2CRDM for dn-up channel
    Cdnup_single = vec([-corr_2pt_dn[u,x] * corr_2pt_up[v,w] for (u, v, w, x) in indices])
    dnup_2CRDM =  dnup_2RDM + Cdnup_single

    Cdndn_single = vec([-corr_2pt_dn[u,x] * corr_2pt_dn[v,w] + corr_2pt_dn[u,w] * corr_2pt_dn[v,x]
            for (u, v, w, x) in indices])
    dndn_2CRDM =  dndn_2RDM + Cdndn_single
        
    return upup_2CRDM, updn_2CRDM, dnup_2CRDM, dndn_2CRDM
end


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Generic function to flatten the tensor and solve eigen values
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#For spinless fermions 
function C2RDM_eigenSolver(corr::Array{ComplexF64, N},dim::Int) where{N}
    #reshape with a row-major way to feed into itensor directly
    corr = permutedims(reshape(corr, dim, dim, dim, dim), (2, 1, 3,4))
    
    #ITensor indices
    i = Index(dim, "i")
    j = Index(dim, "j")
    k = Index(dim, "k")
    l = Index(dim, "l")
    
    #make sure the original indices match with itensor indices exactly
    Trdm = ITensor(corr,i,j,k,l)
    
    #combinig itensor indices
    C_il = combiner(i,l; tags="il")
    C_jk = combiner(j,k,tags="jk")
    
    Trdm = (C_il * Trdm) * C_jk
    
    il = inds(C_il)[1]  
    jk = inds(C_jk)[1]

    #evals,_ = eigen(Trdm, (il,), (jk,))
    #evals = real(collect(diag(evals)))
    
    eval_max,_ = Arpack.eigs(Matrix(Trdm, (il,), (jk,)), nev=1, which=:LR);
    eval_min,_ = Arpack.eigs(Matrix(Trdm, (il,), (jk,)), nev=1, which=:SR);
    eval_max, eval_min = real(eval_max)[1], real(eval_min)[1]
    
    return eval_min, eval_max
end

##If TRS true 
function C2RDM_eigenSolver(corr_upup::Array{ComplexF64, N}, corr_updn::Array{ComplexF64, N}, dimn::Int) where{N}
    #reshape with a row-major way to feed into itensor directly
    corr_upup = permutedims(reshape(corr_upup, dimn, dimn, dimn, dimn), (2, 1, 3,4))
    corr_updn = permutedims(reshape(corr_updn, dimn, dimn, dimn, dimn), (2, 1, 3,4))
    
    #ITensor indices
    i = Index(dimn, "i")
    j = Index(dimn, "j")
    k = Index(dimn, "k")
    l = Index(dimn, "l")
    
    #make sure the original indices match with itensor indices exactly
    Trdm_upup = ITensor(corr_upup,i,j,k,l)
    Trdm_updn = ITensor(corr_updn,i,j,k,l)
    Trdm_dnup = ITensors.permute(Trdm_updn,(j,i,l,k)) #works 
    
    #combinig itensor indices ((il), (jk))
    C_il = combiner(i,l; tags="il")
    C_jk = combiner(j,k,tags="jk")
    
    Trdm_upup = (C_il * Trdm_upup) * C_jk
    Trdm_updn = (C_il * Trdm_updn) * C_jk

    #For dn-up channel, we have to combine j,k and i,l
    Trdm_dnup = (C_jk * Trdm_dnup) * C_il
    
    il = inds(C_il)[1]  
    jk = inds(C_jk)[1]

    Trdm_upup = Matrix(Trdm_upup, (il,), (jk,))
    Trdm_updn = Matrix(Trdm_updn, (il,), (jk,))
    Trdm_dnup = Matrix(Trdm_dnup, (jk,), (il,))   

    halfdim = dimn * dimn
    Trdm = zeros(ComplexF64, 2 * halfdim, 2 * halfdim)
    Trdm[1:halfdim, 1:halfdim] .= Trdm_upup
    Trdm[1:halfdim, halfdim+1:end] .= Trdm_updn
    Trdm[halfdim+1:end, 1:halfdim] .= Trdm_dnup
    Trdm[halfdim+1:end, halfdim+1:end] .= Trdm_upup

    #evals,_ = eigen(Trdm, (il,), (jk,));
    #evals = real(collect(diag(evals)))
    eval_max,_ = Arpack.eigs(Trdm, nev=1, which=:LR);
    eval_min,_ = Arpack.eigs(Trdm, nev=1, which=:SR);
    eval_max, eval_min = real(eval_max)[1], real(eval_min)[1]
    
    return eval_min, eval_max
end

#If Nup!=Ndn or if state breaks TRS
function C2RDM_eigenSolver(corr_upup::Array{ComplexF64, N}, corr_updn::Array{ComplexF64, N},
                            corr_dndn::Array{ComplexF64, N}, dim::Int) where{N}
    #reshape with a row-major way to feed into itensor directly
    corr_upup = permutedims(reshape(corr_upup, dim, dim, dim, dim), (2, 1, 3,4))
    corr_updn = permutedims(reshape(corr_updn, dim, dim, dim, dim), (2, 1, 3,4))
    corr_dndn = permutedims(reshape(corr_dndn, dim, dim, dim, dim), (2, 1, 3,4))
    
    #ITensor indices
    i = Index(dim, "i")
    j = Index(dim, "j")
    k = Index(dim, "k")
    l = Index(dim, "l")
    
    #make sure the original indices match with itensor indices exactly
    Trdm_upup = ITensor(corr_upup,i,j,k,l)
    Trdm_dndn = ITensor(corr_dndn,i,j,k,l)
    Trdm_updn = ITensor(corr_updn,i,j,k,l)
    Trdm_dnup = ITensors.permute(Trdm_updn,(j,i,l,k)) 
    
    #combinig itensor indices ((il), (jk))
    C_il = combiner(i,l; tags="il")
    C_jk = combiner(j,k,tags="jk")
    
    Trdm_upup = (C_il * Trdm_upup) * C_jk
    Trdm_updn = (C_il * Trdm_updn) * C_jk
    Trdm_dndn = (C_il * Trdm_dndn) * C_jk
    
    #For dn-up channel, we have to combine j,k and i,l
    Trdm_dnup = (C_jk * Trdm_dnup) * C_il
    
    il = inds(C_il)[1]  
    jk = inds(C_jk)[1]

    Trdm_upup = Matrix(Trdm_upup, (il,), (jk,))
    Trdm_dndn = Matrix(Trdm_dndn, (il,), (jk,))
    Trdm_updn = Matrix(Trdm_updn, (il,), (jk,))
    Trdm_dnup = Matrix(Trdm_dnup, (jk,), (il,))  

    halfdim = dim * dim
    Trdm = zeros(ComplexF64, 2 * halfdim, 2 * halfdim)
    Trdm[1:halfdim, 1:halfdim] .= Trdm_upup
    Trdm[1:halfdim, halfdim+1:end] .= Trdm_updn
    Trdm[halfdim+1:end, 1:halfdim] .= Trdm_dnup
    Trdm[halfdim+1:end, halfdim+1:end] .= Trdm_dndn

    #evals,_ = eigen(Trdm, (il,), (jk,));
    #evals = real(collect(diag(evals)))
    eval_max,_ = Arpack.eigs(Trdm, nev=1, which=:LR);
    eval_min,_ = Arpack.eigs(Trdm, nev=1, which=:SR);
    eval_max, eval_min = real(eval_max)[1], real(eval_min)[1]
    
    return eval_min, eval_max
end

function C2RDM_eigenSolver(corr_upup::Array{ComplexF64, N}, corr_updn::Array{ComplexF64, N}, 
        corr_dnup::Array{ComplexF64, N}, corr_dndn::Array{ComplexF64, N}, dim::Int) where{N}
    #reshape with a row-major way to feed into itensor directly
    corr_upup = permutedims(reshape(corr_upup, dim, dim, dim, dim), (2, 1, 3, 4))
    corr_updn = permutedims(reshape(corr_updn, dim, dim, dim, dim), (2, 1, 3, 4))
    corr_dnup = permutedims(reshape(corr_dnup, dim, dim, dim, dim), (2, 1, 3, 4))
    corr_dndn = permutedims(reshape(corr_dndn, dim, dim, dim, dim), (2, 1, 3, 4))
    #ITensor indices
    i = Index(dim, "i")
    j = Index(dim, "j")
    k = Index(dim, "k")
    l = Index(dim, "l")
    
    #make sure the original indices match with itensor indices exactly
    Trdm_upup = ITensor(corr_upup,i,j,k,l)
    Trdm_updn = ITensor(corr_updn,i,j,k,l)
    Trdm_dnup = ITensor(corr_dnup,i,j,k,l)
    Trdm_dndn = ITensor(corr_dndn,i,j,k,l)
    
    #combinig itensor indices ((il), (jk))
    C_il = combiner(i,l; tags="il")
    C_jk = combiner(j,k,tags="jk")
    
    Trdm_upup = (C_il * Trdm_upup) * C_jk
    Trdm_updn = (C_il * Trdm_updn) * C_jk
    Trdm_dnup = (C_il * Trdm_dnup) * C_jk
    Trdm_dndn = (C_il * Trdm_dndn) * C_jk
    
    il = inds(C_il)[1]  
    jk = inds(C_jk)[1]

    Trdm_upup = Matrix(Trdm_upup, (il,), (jk,))
    Trdm_updn = Matrix(Trdm_updn, (il,), (jk,))
    Trdm_dnup = Matrix(Trdm_dnup, (il,), (jk,))
    Trdm_dndn = Matrix(Trdm_dndn, (il,), (jk,))
    
    halfdim = dim * dim
    Trdm = zeros(ComplexF64, 2 * halfdim, 2 * halfdim)
    Trdm[1:halfdim, 1:halfdim] .= Trdm_upup
    Trdm[1:halfdim, halfdim+1:end] .= Trdm_updn
    Trdm[halfdim+1:end, 1:halfdim] .= Trdm_dnup
    Trdm[halfdim+1:end, halfdim+1:end] .= Trdm_dndn

    #evals,_ = eigen(Trdm, (il,), (jk,));
    #evals = real(collect(diag(evals)))
    eval_max,_ = Arpack.eigs(Trdm, nev=1, which=:LR);
    eval_min,_ = Arpack.eigs(Trdm, nev=1, which=:SR);
    eval_max, eval_min = real(eval_max)[1], real(eval_min)[1]
    
    return eval_min, eval_max
end
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# generate sites tuple for measurement
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
function measurement_sites(;Ns,Ncut)
    ni = Ncut
    nf = Ns-ni
    Sites = vec([(j, i, k, l) for i in ni+1:nf, j in ni+1:nf, k in ni+1:nf, l in ni+1:nf]);
    indices =  vec([(x-ni, y-ni, z-ni, w-ni) for (x, y, z, w) in Sites])
    return Sites, indices
end

function vN_entropy(psi::MPS, b::Int)
    s = siteinds(psi)
    orthogonalize!(psi, b)
    _, S = svd(psi[b], (linkind(psi, b-1), s[b]))
    SvN = 0.0
    for n in 1:dim(S, 1)
        p = S[n,n]^2
        SvN -= p * log(p)
    end
    return SvN
end

#=
sites = siteinds("Electron", 8)
psi = random_mps(sites, linkdims=250)
op_sites, _ = measurement_sites(Ns=8, Ncut=2)
ops = ("Cup", "Cdagup", "Cdn", "Cdagdn")

#corrs = @time four_pt_correlator(psi, ops, op_sites);
#corrs2 = @time four_pt_correlator2(psi, ops, op_sites)
#corrs2=real.(corrs2)
#corrs==corrs2
=#
