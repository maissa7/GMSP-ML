######################################################################
#             PIECEWISE LINEAR APPROXIMATION PARAMETERS              #
######################################################################

# Number of breakpoints on the water discharge axis (cf. $\mathcal{N}^{u}$)
N_u = 8;

# Number of breakpoints on the water storage axis (cf. $\mathcal{N}^{u}$)
N_s = 3;

Beta_u_i_k_nu = Dict{String, Dict{Int64, Array{Float64, 1}}}(i => Dict{Int64, Array{Float64, 1}}() for i in I);

Beta_s_i_k_ns = Dict{String, Dict{Int64, Array{Float64, 1}}}(i => Dict{Int64, Array{Float64, 1}}() for i in I);

Beta_p_i_k_ns_nu = Dict{String, Dict{Int64, Array{Array{Float64, 1}, 1}}}(i => Dict{Int64, Array{Array{Float64, 1}, 1}}() for i in I);

K_i_k_ns_nu = Dict{String, Dict{Int64, Array{Array{Float64, 1}, 1}}}(i => Dict{Int64, Array{Array{Float64, 1}, 1}}() for i in I);

# Fetching the values of those coefficients from the data files
for i in I
    for t in T
        for k in K_i_t[i][t]
            open(string(@__DIR__, "/", i, "_", k, ".json"), "r") do piecewise_linear_approximation
                piecewise_linear_variables = JSON.Parser.parse(piecewise_linear_approximation)
                Beta_u_i_k_nu[i][k] = piecewise_linear_variables["Beta_u_i_k_nu"]
                Beta_s_i_k_ns[i][k] = piecewise_linear_variables["Beta_s_i_k_ns"]
                Beta_p_i_k_ns_nu[i][k] = piecewise_linear_variables["Beta_p_i_k_ns_nu"]
                K_i_k_ns_nu[i][k] = piecewise_linear_variables["K_i_k_ns_nu"]
            end
        end
    end
end;

println("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
println("~  Two variables piecewise linear approximation method   ~")
println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
