######################################################################
#                POLYNOMIAL APPROXIMATION PARAMETERS                 #
######################################################################

# Definition of the coefficients of the different polynomial approximations of the hydropower function
Poly = Dict{String, Dict{Int64, Array{Float64, 1}}}(i => Dict{Int64, Array{Float64, 1}}() for i in I);

# Fetching the values of those coefficients from the data files
for i in I
    for t in T
        for k in K_i_t[i][t]
            open(string(@__DIR__, "/", i, "_", k, ".json"), "r") do polynomial_approximation
                Poly[i][k] = JSON.Parser.parse(polynomial_approximation)
            end
        end
    end
end;

println("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
println("~            Polynomial approximation method             ~")
println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
