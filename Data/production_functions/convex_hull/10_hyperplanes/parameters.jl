######################################################################
#                CONVEX HULL APPROXIMATION PARAMETERS                #
######################################################################

# Definition of the set of hyperplanes used to approximate the hydropower function (cf. $\mathcal{H}$)
H = Set{Int64}([h for h in 1:10]);

# Definition of the coefficients of $u$ in the different hyperplanes [MWh*s/(m^3/day)] (cf. $\beta_{h}^{u}$)
Beta_u_i_k_h = Dict{String, Dict{Int64, Array{Float64, 1}}}(i => Dict{Int64, Array{Float64, 1}}() for i in I);

# Definition of the coefficients of $s$ in the different hyperplanes [MWh*s/(m^3/day)] (cf. $\beta_{h}^{s}$)
Beta_s_i_k_h = Dict{String, Dict{Int64, Array{Float64, 1}}}(i => Dict{Int64, Array{Float64, 1}}() for i in I);

# Definition of the independent term in the different hyperplanes [MWh/day] (cf. $\beta_{h}^{0}$)
Beta_0_i_k_h = Dict{String, Dict{Int64, Array{Float64, 1}}}(i => Dict{Int64, Array{Float64, 1}}() for i in I);

# Fetching the values of those coefficients from the data files
for i in I
    for t in T
        for k in K_i_t[i][t]
            open(string(@__DIR__, "/", i, "_", k, ".json"), "r") do convex_hull_approximation
                hyperplanes_equations = JSON.Parser.parse(convex_hull_approximation, dicttype = Dict{String, Array{Float64, 1}})
                Beta_u_i_k_h[i][k] = hyperplanes_equations["Beta_u"]
                Beta_s_i_k_h[i][k] = hyperplanes_equations["Beta_s"]
                Beta_0_i_k_h[i][k] = hyperplanes_equations["Beta_0"]
            end
        end
    end
end;

println("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
println("~  Convex hull approximation method with 10 hyperplanes  ~")
println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
