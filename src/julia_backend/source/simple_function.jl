using PyCall, BenchmarkTools
py"""
import numpy as np
def _fastNormalizedBMLogLik(thetaVec: np.ndarray, habSqrdVec: np.ndarray, sampleSize: int) -> float:
    # thetaVec = bound_away_from_one_and_zero_arrays([thetaVec.astype(float)])[0]
    thetaVec = np.clip(thetaVec,1e-12,1-1e-12)
    negEntVec = thetaVec * np.log(thetaVec) + (1 - thetaVec) * np.log(1 - thetaVec)
    normLogLik = np.sum(habSqrdVec * negEntVec) / sampleSize
    return normLogLik
"""

function loglik(θ, hab_sqrd_vec, n::Int)
    @assert length(θ) == length(hab_sqrd_vec)
    count = 0.0
    @inbounds @simd for i in eachindex(θ)
        θ_t = θ[i]
        θ_c = θ_t < 0 ? 1e-12 : (θ_t > 1 ? 1 - 1e-12 : θ_t)
        count += (θ_c * log(θ_c) + (1 - θ_c) * log(1 - θ_c)) * hab_sqrd_vec[i]
    end
    count / n
end

function test_old_ll()
    θ = rand(100, 100)
    hab_sqrd_vec = rand(100, 100)
    b = @benchmark loglik($θ, $hab_sqrd_vec, 100)
    b2 = @benchmark py"_fastNormalizedBMLogLik"($θ, $hab_sqrd_vec, 100)
    # a ≈ b
    return b, b2
end


if abspath(PROGRAM_FILE) == @__FILE__
    b, b2 = test_old_ll()
    b
    b2
end
