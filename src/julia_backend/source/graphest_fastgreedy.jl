using Graphs
using BenchmarkTools

mutable struct Assignment


    number_nodes::Int
    number_groups::Int
    node_labels::Vector{Int}


    counts::Matrix{Int}
    realized::Matrix{Float64}
    estimated_theta::Matrix{Float64}

end

function create_assignment(node_labels::Vector{Int}, number_groups::Int64, adjacency_matrix::Matrix{Int})

    # create recipient 
    number_nodes = length(node_labels)
    estimated_theta = zeros(Float64, number_nodes, number_groups)
    counts = zeros(Int64, number_groups, number_groups)
    realized = zeros(Int64, number_groups, number_groups)
    size_groups = [count(==(element), node_labels) for element in unique(node_labels)]


    @inbounds @simd for k in 1:number_groups
        for l in k:number_groups
            realized[k, l] = sum(adjacency_matrix[node_labels.==k, node_labels.==l])
            realized[l, k] = realized[k, l]
            counts[k, l] = size_groups[k] * size_groups[l]
            counts[l, k] = counts[k, l]
        end
    end

    @inbounds @simd for k in 1:number_groups
        counts[k, k] = size_groups[k] * (size_groups[k] - 1) / 2
        realized[k, k] = sum(adjacency_matrix[node_labels.==k, node_labels.==k]) / 2
    end

    estimated_theta = realized ./ counts

    return Assignment(number_nodes, number_groups, node_labels, counts, realized, estimated_theta)
end



function permute_labels_node(assignment::Assignment, perm::Tuple{Int,Int}, adjacency_matrix::Matrix{Int})
    inter = assignment.node_labels[perm[1]]
    assignment.node_labels[perm[1]] = assignment.node_labels[perm[2]]
    assignment.node_labels[perm[2]] = inter

    # inneficient but simple
    return create_assignment(assignment.node_labels, assignment.number_groups, adjacency_matrix)
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


function fastLL(assignment::Assignment)
    loglik = 0.0
    @inbounds @simd for i in 1:assignment.number_groups
        for j in i+1:assignment.number_groups
            θ = assignment.estimated_theta[i, j]
            θ_c = θ < 0 ? 1e-12 : (θ > 1 ? 1 - 1e-12 : θ)
            loglik += (θ_c * log(θ_c) + (1 - θ_c) * log(1 - θ_c)) * assignment.counts[i, j]
        end
    end
    return loglik / assignment.number_nodes
end


function test()
    n::Int64 = 1000
    K = 2
    A = Matrix(Graphs.LinAlg.adjacency_matrix(erdos_renyi(n, 0.8)))
    node_labels = [fill(1, round(typeof(n), n / 2)); fill(2, round(typeof(n), n / 3))]
    assignment = create_assignment(node_labels, K, A)
    b = @benchmark fastLL($assignment)
    b2 = @benchmark permute_labels_node($assignment, (1, 2), $A)
    return b, b2
end


if abspath(PROGRAM_FILE) == @__FILE__
    n::Int64 = 60
    K = 3
    A = Matrix(Graphs.LinAlg.adjacency_matrix(erdos_renyi(n, 0.8)))
    node_labels = [fill(1, round(typeof(n), n / 3)); fill(2, round(typeof(n), n / 3)); fill(3, round(typeof(n), n / 3))]
    assignment = create_assignment(node_labels, K, A)
    new_assignment = permute_labels_node(assignment, (1, n - 2), A)
    println(assignment.estimated_theta)
    println(fastLL(assignment))
    println(fastLL(new_assignment))
end