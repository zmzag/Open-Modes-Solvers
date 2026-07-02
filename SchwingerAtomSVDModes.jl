using LinearAlgebra
using Plots
using SparseArrays
using Arpack

g = 1.5;
κ = 0.2;
L = 40;
dx = .1;
N = Integer(L/dx + 1);

function solve_soliton(g, κ, L, N; Θ::Function = x -> (x ≥ 0 ? 1.0 : 0.0), tol::Real = 1e-9, maxiter::Integer = 30, verbose::Bool = true)
    xrange = range(-L/2, L/2; length=N)
    dx = step(xrange)
    k = κ*sqrt(π) #coeff of sine term
    g0 = g^2 #coeff of mass term
    φ_left, φ_right = 0.0, -sqrt(π) 

    # initial guess is the SG soliton
    φ = @. -(2/sqrt(π))*atan(exp(sqrt(2*π*κ)*xrange)) 
    src = Θ.(xrange)   #applies Heaviside function to defined grid
    F  = zeros(N)   #function we want equal to zero (difference between guess and solution)
    
    function residual!(F, φ)
        F[1] = φ[1] - φ_left
        F[N] = φ[N] - φ_right
        @inbounds for i in 2:N-1 #skips check that array element is a valid one to speed up loop
            laplacian = (φ[i+1] - 2φ[i] + φ[i-1])/dx^2
            F[i] = laplacian - k*sin(2*sqrt(π)*φ[i]) - g0*φ[i] - sqrt(π)*g0*src[i] 
        end
        return F
    end

    dl = zeros(N - 1)   # sub diagonal of Jacobian
    d  = zeros(N)   # diagonal
    du = zeros(N - 1)   # super diagonal
    
    function jacobian!(dl, d, du, φ)
        d[1] = 1.0
        d[N] = 1.0
        @inbounds for i in 2:N-1
            d[i] = -2.0/dx^2 - 2*sqrt(π)*k*cos(2*sqrt(π)*φ[i]) - g0 #derivative of residual w/r/t φ[i] of discretized equation
            dl[i-1] = 1.0/dx^2
            du[i-1] = 1.0/dx^2
        end
        return Tridiagonal(dl, d, du)
    end
    
    for iter in 1:maxiter
        residual!(F, φ)
        resnorm = norm(F)
        verbose && println("iter $iter:  ‖F‖ = $resnorm") #&& can be used for short circuit evaluation - in a && b, the expression b is only evaluated if a evaluates to true
        if resnorm < tol
            verbose && println("Converged in $iter iterations.")
            return xrange, φ
        end
        J  = jacobian!(dl, d, du, φ) #tangent line from F[ϕ] to zero has this slope
        Δφ = J \ (-F) #left division, computes -F = Δϕ J , gives Δϕ which adjusts ϕ to the value where the tangent line hits zero
        φ .+= Δφ
    end
    
    @warn "Newton iteration did not reach tol = $tol within $maxiter steps " *
          "(‖F‖ = $(norm(residual!(F, φ))))."
    return xrange, φ
end

xrange, ϕ_ss = solve_soliton(g, κ, L, N)

function FDmatrix(N, k, dx)
    #building laplacian
    laplacian = ComplexF64.(spdiagm(0 => fill(-2.0, N), -1 => fill(1.0, N-1), 1 => fill(1.0, N-1)))
    laplacian[1,1] = -(1 - im*k*dx)
    laplacian[end,end] = -(1 - im*k*dx)
    laplacianMat = (1/dx^2).*laplacian
    #building cos term
    costerm = zeros(N)
    @inbounds for i in 1:N
            costerm[i] = 2*π*κ*(1 - cos(2*sqrt(π)*ϕ_ss[i]))
        end
    costermMat = Diagonal(costerm)
    #building k term
    ktermMat = Diagonal(fill(k^2,N))
    #building full finite difference matrix
    fdMat = - laplacianMat - costermMat - ktermMat
    return fdMat
end

function singval(kre, kim, N, dx)
    k = kre + im*kim
    testmat = FDmatrix(N, k, dx)
    u, s, v = svd(Matrix(testmat))
    σmin = s[end]
    evec = v[:,end]
    return σmin, evec
end

kimvals = range(-.2, .2; length=401)
krevals = range(-.2, .2; length=401)
singvalslist = zeros(length(krevals),length(kimvals))
@inbounds for i in 1:length(krevals)
    @inbounds for j in 1:length(kimvals)
        σ, vec = singval(krevals[i], kimvals[j], N, dx)
        singvalslist[i,j] = σ
    end
end

function minima_finder(matrix, radius; tol=1e-3)
    rownum, colnum = size(matrix)
    minimalist = []
    for i in (radius+1):(rownum-radius)
        for j in (radius+1):(colnum-radius)
            if matrix[i,j] <= tol
                window = @view matrix[i-radius:i+radius, j-radius:j+radius]
                if matrix[i,j] == minimum(window)
                    push!(minimalist, [i,j])
                end
            end
        end
    end
    return minimalist
end

minpos = minima_finder(singvalslist, 3)
krelist = []
kimlist = []
eveclist = []
for i in 1:length(minpos)
    row, col = minpos[i]
    push!(krelist, krevals[row])
    push!(kimlist, kimvals[col])
    σ, ϕ = singval(krevals[row],kimvals[col],N,dx)
    push!(eveclist, ϕ)
end


#=
for kplot in 1:length(krelist)
    p = plot(xrange, real.(eveclist[kplot]), 
        title = "ϕ for k = $(krelist[kplot]) + $(kimlist[kplot])*i", 
        label = "Re(ϕ)")
    plot!(p, xrange, imag.(eveclist[kplot]), label = "Im(ϕ)")

    savefig(p, "/scratch/gpfs/TURECI/zz3250/zz3250/Julia/eigenmode_k$(kplot).pdf")
end
=#