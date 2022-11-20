using Graphs
using BenchmarkTools
using StatsBase
using Plots

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



function permute_labels_node(assignment::Assignment, perm::Tuple{Int,Int}, adjacency_matrix::Matrix{Int}, new_assignment::Assignment)

    # update the labels
    new_assignment.node_labels[perm[1]] = assignment.node_labels[perm[2]]
    new_assignment.node_labels[perm[2]] = assignment.node_labels[perm[1]]

    # update the realized counts
    new_assignment.realized

    throw("unimplemented")
    # inneficient but simple
    # return create_assignment(assignment.node_labels, assignment.number_groups, adjacency_matrix)
end


function create_initial_assignment(adjacency_matrix::Matrix{Int}, number_groups::Int64)
    # create initial assignment by chunking the nodes
    number_nodes = size(adjacency_matrix, 1)
    node_labels = [Int(ceil(i / (number_nodes / number_groups))) for i in 1:number_nodes]
    #node_labels = zeros(Int64, size(adjacency_matrix, 1))
    #for i in 1:number_groups
    #    node_labels[i] = i
    #end
    #for i in number_groups+1:size(adjacency_matrix, 1)
    #    node_labels[i] = rand(1:number_groups)
    #end
    return create_assignment(node_labels, number_groups, adjacency_matrix)
end






function GreedyLLopt(
    A::Matrix{Int},
    cluster_size::Int,
    steps::Int=100,
)
    n = size(A, 1)
    indices = collect(1:n)
    candidate_assignment = create_initial_assignment(A, cluster_size)
    best_ll = fastLL(candidate_assignment)
    best_assignment = deepcopy(candidate_assignment)
    ll_trace = [best_ll]
    best_ll_trace = [best_ll]

    # not sure why both loops are needed in the original script ???
    # What am I missing ?? 
    for _ in 1:steps
        if check_stoppping_condition()
            break
        end
        perm = sample(indices, 2, replace=false)

        ll_candidate = updateLL(best_ll, best_assignment, perm, A, candidate_assignment)
        push!(ll_trace, ll_candidate)
        if ll_candidate > best_ll
            best_ll = ll_candidate
            best_assignment = deepcopy(candidate_assignment)
        end
        push!(best_ll_trace, best_ll)
    end

    return best_assignment.node_labels, best_assignment.estimated_theta, ll_trace, best_ll_trace

end


function check_stoppping_condition()
    @warn "does not do anything yet"
    return false
end


function updateLL(ll::Float64, assignment::Assignment, perm::Vector{Int}, adjacency_matrix::Matrix{Int}, new_assignment::Assignment)
    @warn "Hihgly inefficient implementation"
    new_assignment.node_labels = deepcopy(assignment.node_labels)
    new_assignment.node_labels[perm[1]] = assignment.node_labels[perm[2]]
    new_assignment.node_labels[perm[2]] = assignment.node_labels[perm[1]]
    # inneficient but simple
    new_assignment = create_assignment(new_assignment.node_labels, assignment.number_groups, adjacency_matrix)
    return fastLL(new_assignment)
end


function log_likelihood(node_labels::Vector{Int}, adjacency_matrix::Matrix{Int}, number_groups::Int64)
    return fastLL(create_assignment(node_labels, number_groups, adjacency_matrix))
end

function fastLL(assignment::Assignment)
    loglik = 0.0
    @inbounds @simd for i in 1:assignment.number_groups
        for j in i:assignment.number_groups
            θ = assignment.estimated_theta[i, j]
            θ_c = θ < 0 ? 2.220446049250313e-16 : (θ > 1 ? 1 - 2.220446049250313e-16 : θ)
            loglik += (θ_c * log(θ_c) + (1 - θ_c) * log(1 - θ_c)) * assignment.counts[i, j]
        end
    end
    return loglik / assignment.number_nodes
end


function benchmark_test(n::Int64=10)
    K = 2
    A = Matrix(Graphs.LinAlg.adjacency_matrix(erdos_renyi(n, 0.8)))

    b = @benchmark GreedyLLopt($A, $K, 2)
    return b
end


function test_verbose(n::Int64=10)
    K = 2
    A = Matrix(Graphs.LinAlg.adjacency_matrix(erdos_renyi(n, 0.8)))
    return GreedyLLopt(A, K, 2)

end



if abspath(PROGRAM_FILE) == @__FILE__
    n::Int64 = 30
    K = 3
    A = Matrix(Graphs.LinAlg.adjacency_matrix(erdos_renyi(n, 0.8)))
    node_labels = [fill(1, round(typeof(n), n / 3)); fill(2, round(typeof(n), n / 3)); fill(3, round(typeof(n), n / 3))]
    assignment = create_assignment(node_labels, K, A)
    println(assignment.node_labels)
    println(fastLL(assignment))
    a, ll = GreedyLLopt(A, K, 1e-6, 3)
    println(a.node_labels)
    println(ll)
end
