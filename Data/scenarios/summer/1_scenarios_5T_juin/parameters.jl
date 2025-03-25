######################################################################
#                 DEFINITION OF THE SITUATION FOLDER                 #
######################################################################

FOLDER = string(@__DIR__, "/");

######################################################################
#                   DEFINITION OF THE PRIMARY SET                    #
######################################################################

# Definition of the set of scenarios (cf. $\Omega$)
Omega = Set{String}(convert(Array{String, 1}, JSON.Parser.parse(open(string(FOLDER, "general_parameters.json"), "r"))["Omega"]));

######################################################################
#                    WEATHER SCENARIOS PARAMETERS                    #
######################################################################

# Definition of the probabilities of realization of the different scenarios (cf. $\varphi_{\omega}$)
Phi_omega = convert(Dict{String, Float64}, JSON.Parser.parse(open(string(FOLDER, "general_parameters.json"), "r"))["Phi"]);

######################################################################
#                       ELECTRICITY PARAMETERS                       #
######################################################################

# Definition of the prices of electricity sale at each time period of the different scenarios [€/MWh] (cf. $B^{+}_{t,\omega}$)
B_plus_omega_t = convert(Dict{String, Array{Float64, 1}}, JSON.Parser.parse(open(string(FOLDER, "/price_scenarios.json"), "r"))["Beta_plus"]);

# Definition of the prices of electricity purchase at each time period of the different scenarios [€/MWh] (cf. $B^{-}_{t,\omega}$)
B_minus_omega_t = convert(Dict{String, Array{Float64, 1}}, JSON.Parser.parse(open(string(FOLDER, "/price_scenarios.json"), "r"))["Beta_minus"]);

# Definition of the demand of electricity at each time period of the different scenarios [€/MWh] (cf. $J_{t,\omega}$)
J_omega_t =	Dict{String, Array{Float64, 1}}(omega => Array{Float64, 1}([0.0 for t in T]) for omega in Omega);

######################################################################
#                        HYDRAULIC PARAMETERS                        #
######################################################################

# Definition of the lateral inflows in each powerhouse at each time period of the different scenarios [m^3/s] (cf. $F_{i,t,\omega}$)
F_omega_i_t = convert(Dict{String, Dict{String, Array{Float64, 1}}}, JSON.Parser.parse(open(string(FOLDER, "/inflow_scenarios.json"), "r")));

######################################################################
#                   CURRENT SITUATION OF THE MODEL                   #
######################################################################

println("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
println("~                Description of the model                ~")
println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
println("~                                                        ~")
println("~      Season: summer       |   Number of scenarios: 2   ~")
println("~                                                        ~")
println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
