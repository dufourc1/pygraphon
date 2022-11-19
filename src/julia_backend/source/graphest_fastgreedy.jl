using Graphs


"""
Assignment of the nodes.

This will hold the corresponding estimated connection probabily for each group `estimated_theta` and 
    the corresponding counts of potential realization `counts` and realized edges `realized`.
"""
mutable struct Assignment
    node_labels:Vector{Int}
    estimated_theta::Matrix{Float64}
    counts::Matrix{Int}
    realized::Matrix{Float64}
    number_nodes::Int
    number_groups::Int

    function Assignment(node_labels::Vector{Int}, number_groups::Int64, adjacency_matrix::Matrix{Int})

        # create recipient 
        estimated_theta = zeros(number_nodes, number_groups)
        counts = zeros(Int64, number_groups, number_groups)
        realized = zeros(Int64, number_nodes, number_groups)

        for i in 1:number_nodes, j in i+1:number_nodes
            realized[node_labels[i], node_labels[j]] += adjacency_matrix[i, j]
            counts[node_labels[i], node_labels[j]] += 1
        end

        for k in 1:number_groups
            estimated_theta[:, k] = realized[:, k] ./ counts[k, :]
        end

        return new(node_labels, estimated_theta, counts, realized, number_nodes, number_groups)
    end

end




function permute_labels_node(assignment::Assignment, perm::Tuple{Int,Int}, adjacency_matrix::Matrix{Int})
    inter = assignment.node_labels[perm[1]]
    assignment.node_labels[perm[1]] = assignment.node_labels[perm[2]]
    assignment.node_labels[perm[2]] = inter

    assignment.estimated_theta = assignment.estimated_theta[perm, :]

    assignment.realized = assignment.realized[perm, :]
    return assignment
end




function GreedyLLopt(
    A::Matrix{Int},
    cluster_size::Int,
    absolute_tolerance::Float64,
    max_num_restarts::Int,
)
    n = size(A, 1)
    @assert n == size(A, 2)

    throw(ArgumentError("Not implemented yet"))

end

"""
Calculate the log-likelihood based on a specific cluster assignment in the 
network histogram.
"""
function fastLL(assignment::Assignment)
    loglik = 0.0
    @inbounds @simd for i in 1:assignment.number_groups, j in i+1:assignment.number_groups
        θ = assignment.estimated_theta[i, j]
        θ_c = θ < 0 ? 1e-12 : (θ > 1 ? 1 - 1e-12 : θ)
        loglik += (θ_c * log(θ_c) + (1 - θ_c) * log(1 - θ_c)) * assignment.counts[i, j]
    end
    return loglik / assignment.number_nodes
end



if abspath(PROGRAM_FILE) == @__FILE__
    A = Matrix(Graphs.LinAlg.adjacency_matrix(erdos_renyi(6, 0.6)))
    println(A)
    cluster_size = 10
    absolute_tolerance = 1e-6
    max_num_restarts = 10
    println("Running GreedyLLopt")
    GreedyLLopt(A, cluster_size, absolute_tolerance, max_num_restarts)
end