######################################################################
#                CONVEX HULL APPROXIMATION PARAMETERS                #
######################################################################

# Definition of the set of hyperplanes used to approximate the hydropower function (cf. $\mathcal{H}$)
L_i_k = Dict{String, Dict{Int64, Int64}}(i => Dict{Int64, Int64}() for i in I);

# Definition of the set of hyperplanes used to approximate the hydropower function (cf. $\mathcal{H}$)
U_i_k = Dict{String, Dict{Int64, Int64}}(i => Dict{Int64, Int64}() for i in I);

# Definition of the coefficients of $u$ in the different hyperplanes [MWh*s/(m^3/day)] (cf. $\beta_{h}^{u}$)
P_max_i_k_h = Dict{String, Dict{Int64, Array{Float64, 1}}}(i => Dict{Int64, Array{Float64, 1}}() for i in I);

# Definition of the coefficients of $u$ in the different hyperplanes [MWh*s/(m^3/day)] (cf. $\beta_{h}^{u}$)
Upper_Beta_u_i_k_h = Dict{String, Dict{Int64, Array{Float64, 1}}}(i => Dict{Int64, Array{Float64, 1}}() for i in I);

# Definition of the coefficients of $s$ in the different hyperplanes [MWh*s/(m^3/day)] (cf. $\beta_{h}^{s}$)
Upper_Beta_s_i_k_h = Dict{String, Dict{Int64, Array{Float64, 1}}}(i => Dict{Int64, Array{Float64, 1}}() for i in I);

# Definition of the independent term in the different hyperplanes [MWh/day] (cf. $\beta_{h}^{0}$)
Upper_Beta_0_i_k_h = Dict{String, Dict{Int64, Array{Float64, 1}}}(i => Dict{Int64, Array{Float64, 1}}() for i in I);

# Definition of the coefficients of $u$ in the different hyperplanes [MWh*s/(m^3/day)] (cf. $\beta_{h}^{u}$)
Lower_Beta_u_i_k_h = Dict{String, Dict{Int64, Array{Float64, 1}}}(i => Dict{Int64, Array{Float64, 1}}() for i in I);

# Definition of the coefficients of $s$ in the different hyperplanes [MWh*s/(m^3/day)] (cf. $\beta_{h}^{s}$)
Lower_Beta_s_i_k_h = Dict{String, Dict{Int64, Array{Float64, 1}}}(i => Dict{Int64, Array{Float64, 1}}() for i in I);

# Definition of the independent term in the different hyperplanes [MWh/day] (cf. $\beta_{h}^{0}$)
Lower_Beta_0_i_k_h = Dict{String, Dict{Int64, Array{Float64, 1}}}(i => Dict{Int64, Array{Float64, 1}}() for i in I);

# Fetching the values of those coefficients from the data files
for i in I
    for t in T
        for k in K_i_t[i][t]
            open(string(@__DIR__, "/", i, "_", k, ".json"), "r") do convex_hull_approximation
                data = JSON.Parser.parse(convex_hull_approximation)
                Lower_Beta_0_i_k_h[i][k] = convert(Array{Float64, 1}, data["lower"]["Beta_0"])
                Lower_Beta_u_i_k_h[i][k] = convert(Array{Float64, 1}, data["lower"]["Beta_u"])
                Lower_Beta_s_i_k_h[i][k] = convert(Array{Float64, 1}, data["lower"]["Beta_s"])
                Upper_Beta_0_i_k_h[i][k] = convert(Array{Float64, 1}, data["upper"]["Beta_0"])
                Upper_Beta_u_i_k_h[i][k] = convert(Array{Float64, 1}, data["upper"]["Beta_u"])
                Upper_Beta_s_i_k_h[i][k] = convert(Array{Float64, 1}, data["upper"]["Beta_s"])
                L_i_k[i][k] = convert(Int64, data["lower_number"])
                U_i_k[i][k] = convert(Int64, data["upper_number"])
                P_max_i_k_h[i][k] = convert(Array{Float64, 1}, data["lower_maximas"])
            end
        end
    end
end;

println("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
println("~       Piecewise convex hull approximation method       ~")
println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
