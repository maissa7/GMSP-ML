######################################################################
#                   DEFINITION OF THE PRIMARY SETS                   #
######################################################################

# Definition of the time horizon (cf. $\mathcal{T}$)
T = Set{Int64}([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]);

# Definition of the set of powerhouses (cf. $\mathcal{I}$)
I = Set{String}(["TIGNE", "MTRIG", "POMBL", "COCHE", "RANDE"]);

# Definition of the set of maintenance tasks (cf. $\mathcal{M}$)
M = Set{String}(convert(Array{String, 1}, JSON.Parser.parse(open(string("data/maintenances/maintenance_tasks.json"), "r"))["M"]));

println("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
println("~              Description of the test case              ~");
println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
println("~                                                        ~");
println("~   Maintenance tasks: ", length(M),"   |    Hydropower plants: ", length(I), "    ~");
println("~                                                        ~");
println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");

######################################################################
#                   DEFINITION OF THE DERIVED SETS                   #
######################################################################

# Definition of the sets of the powerhouses upstream of the different powerhouses (cf. $\mathcal{U}_{i}$)
U_i = Dict{String, Set{String}}(
        "TIGNE" => Set{String}([]),
        "MTRIG" => Set{String}(["TIGNE"]),
        "POMBL" => Set{String}(["MTRIG"]),
        "COCHE" => Set{String}([]),
        "RANDE" => Set{String}(["COCHE"])
);

# Definition of the sets of the time periods when a maintenance task can be initiated (cf. $\mathcal{T}_{m}$)
T_m = Dict{String, Set{Int64}}(m => Set{Int64}(convert(Array{Int64, 1}, JSON.Parser.parse(open(string("data/maintenances/maintenance_tasks.json"), "r"))["T_m"][m])) for m in M);

# Definition of the sets of the maintenance tasks that have to be executed in the different powerhouses (cf. $\mathcal{M}_{i}$)
M_i = Dict{String, Set{String}}(i => Set{String}(convert(Array{String, 1}, JSON.Parser.parse(open(string("data/maintenances/maintenance_tasks.json"), "r"))["M_i"][i])) for i in I);

# Definition of the sets of number of turbines that can be active in each powerhouse at each step of time (cf. $\mathcal{K}_{i,t}$)
K_i_t = Dict{String, Array{Set{Int64}, 1}}(
        "TIGNE" => Array{Set{Int64},1}([Set{Int64}([3,2]), Set{Int64}([3,2]), Set{Int64}([3,2]), Set{Int64}([3,2]), Set{Int64}([3,2]), Set{Int64}([3,2]), Set{Int64}([3,2]), Set{Int64}([3,2]), Set{Int64}([3,2]), Set{Int64}([3,2]), Set{Int64}([3,2]), Set{Int64}([3,2]), Set{Int64}([3,2]), Set{Int64}([3,2]), Set{Int64}([3,2]), Set{Int64}([3,2]), Set{Int64}([3,2]), Set{Int64}([3,2]), Set{Int64}([3,2]), Set{Int64}([3,2]), Set{Int64}([3,2]), Set{Int64}([3,2]), Set{Int64}([3,2]), Set{Int64}([3,2]), Set{Int64}([3,2]), Set{Int64}([3,2]), Set{Int64}([3,2]), Set{Int64}([3,2]), Set{Int64}([3,2]), Set{Int64}([3,2])]),
        "MTRIG" => Array{Set{Int64},1}([Set{Int64}([8,7]), Set{Int64}([8,7]), Set{Int64}([8,7]), Set{Int64}([8,7]), Set{Int64}([8,7]), Set{Int64}([8,7]), Set{Int64}([8,7]), Set{Int64}([8,7]), Set{Int64}([8,7]), Set{Int64}([8,7]), Set{Int64}([8,7]), Set{Int64}([8,7]), Set{Int64}([8,7]), Set{Int64}([8,7]), Set{Int64}([8,7]), Set{Int64}([8,7]), Set{Int64}([8,7]), Set{Int64}([8,7]), Set{Int64}([8,7]), Set{Int64}([8,7]), Set{Int64}([8,7]), Set{Int64}([8,7]), Set{Int64}([8,7]), Set{Int64}([8,7]), Set{Int64}([8,7]), Set{Int64}([8,7]), Set{Int64}([8,7]), Set{Int64}([8,7]), Set{Int64}([8,7]), Set{Int64}([8,7])]),
        "POMBL" => Array{Set{Int64},1}([Set{Int64}([1,0]), Set{Int64}([1,0]), Set{Int64}([1,0]), Set{Int64}([1,0]), Set{Int64}([1,0]), Set{Int64}([1,0]), Set{Int64}([1,0]), Set{Int64}([1,0]), Set{Int64}([1,0]), Set{Int64}([1,0]), Set{Int64}([1,0]), Set{Int64}([1,0]), Set{Int64}([1,0]), Set{Int64}([1,0]), Set{Int64}([1,0]), Set{Int64}([1,0]), Set{Int64}([1,0]), Set{Int64}([1,0]), Set{Int64}([1,0]), Set{Int64}([1,0]), Set{Int64}([1,0]), Set{Int64}([1,0]), Set{Int64}([1,0]), Set{Int64}([1,0]), Set{Int64}([1,0]), Set{Int64}([1,0]), Set{Int64}([1,0]), Set{Int64}([1,0]), Set{Int64}([1,0]), Set{Int64}([1,0])]),
        #"MOUTI" => Array{Set{Int64},1}([Set{Int64}([2,1]), Set{Int64}([2,1]), Set{Int64}([2,1]), Set{Int64}([2,1]), Set{Int64}([2,1]), Set{Int64}([2,1]), Set{Int64}([2,1]), Set{Int64}([2,1]), Set{Int64}([2,1]), Set{Int64}([2,1]), Set{Int64}([2,1]), Set{Int64}([2,1]), Set{Int64}([2,1]), Set{Int64}([2,1]), Set{Int64}([2,1]), Set{Int64}([2,1]), Set{Int64}([2,1]), Set{Int64}([2,1]), Set{Int64}([2,1]), Set{Int64}([2,1]), Set{Int64}([2,1]), Set{Int64}([2,1]), Set{Int64}([2,1]), Set{Int64}([2,1]), Set{Int64}([2,1]), Set{Int64}([2,1]), Set{Int64}([2,1]), Set{Int64}([2,1]), Set{Int64}([2,1]), Set{Int64}([2,1])]),
        "COCHE" => Array{Set{Int64},1}([Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3])]),
        "RANDE" => Array{Set{Int64},1}([Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3]), Set{Int64}([4,3])])
);

######################################################################
#                 ELECTRICITY PRODUCTION PARAMETERS                  #
######################################################################

# Definition of the upper limit of electricity sale for the problem [MWh] (cf. $\overline{W}^{+}$)
W_plus = 100000;

# Definition of the upper limit of electricity purchase for the problem [MWh] (cf. $\overline{W}^{+}$)
W_minus = 10000;

# Definition of the generation capacity of each powerhouse when k turbines are active [MWh/day] (cf. $\overline{P}_{i,k}$)
P_max_i_k = Dict{String, Dict{Int64, Float64}}(
        "TIGNE" => Dict{Int64, Float64}(
                2 => 278.08,
                3 => 397.26
        ),
        "MTRIG" => Dict{Int64, Float64}(
                7 => 1711.65,
                8 => 1863.02
        ),
        "POMBL" => Dict{Int64, Float64}(
                0 => 0.0,
                1 => 135.46

	),
        "COCHE" => Dict{Int64, Float64}(
                3 => 1163.46,
                4 => 1380.82
        ),
        "RANDE" => Dict{Int64, Float64}(
                3 => 1070.15,
                4 => 1329.59
        )
);

######################################################################
#                        HYDRAULIC PARAMETERS                        #
######################################################################

# Definition of the factor for conversion from m^3/s to hm^3/day [0.0864 s.hm3/(day.m3)] (cf. $Q$)
Q = 0.0864;

# Definition of the initial stored water in each reservoir [hm^3] (cf. $S_{0,i}$)
S_init_i = Dict{String, Float64}(
        "TIGNE" => 1.59842000e+08 * 1e-06,
        "MTRIG" => 3.67200000e+05 * 1e-06,
        "POMBL" => 0.00000000e+00 * 1e-06,
        "COCHE" => 0.10082,
        "RANDE" => 1.43000000e+05 * 1e-06

);

# Definition of the lower limit on the stored water in each reservoir [hm^3] (cf. $\underbar{S}_{i}$)
S_min_i = Dict{String, Float64}(
        "TIGNE" => 0.00000000e+00 * 1e-06,
        "MTRIG" => 2.23000000e+05 * 1e-06,
        "POMBL" => 0.00000000e+00 * 1e-06,
        "COCHE" => 0.10082,
        "RANDE" => 1.43000000e+05 * 1e-06
);

# Definition of the upper limit on the stored water in each reservoir [hm^3] (cf. $\overline{S}_{i}$)
S_max_i = Dict{String, Float64}(
        "TIGNE" => 2.24273000e+08 * 1e-06,
        "MTRIG" => 4.26000000e+05 * 1e-06,
        "POMBL" => 1.00000000e+00 * 1e-06,
        "COCHE" => 0.10082,
        "RANDE" => 1.43000000e+05 * 1e-06
);

# Definition of the upper limit on the turbine discharge through each of the powerhouses [m^3/s]
U_max_i_k = Dict{String, Dict{Int64, Float64}}(
        "TIGNE" => Dict{Int64, Float64}(
                2 => 1.4000000E+01,
                3 => 2.0000000E+01
        ),
        "MTRIG" => Dict{Int64, Float64}(
                7 => 4.4000000E+01,
                8 => 5.0000000E+01
        ),
        "POMBL" => Dict{Int64, Float64}(
                0 => 0.0000000E+02,
                1 => 1.5000000E+02
               ),
        "COCHE" => Dict{Int64, Float64}(
                3 => 3.0000000E+01,
                4 => 4.0000000E+01
        ),
        "RANDE" => Dict{Int64, Float64}(
                3 => 1.5000000E+02,
                4 => 2.0000000E+02
        )
);

# Definition of the upper limit on the water spill of the reservoir in each powerhouse [m^3/s]
V_max_i = Dict{String, Float64}(
        "TIGNE" => 2.00000000E+02,
        "MTRIG" => 1.00000000E+02,
        "POMBL" => 2.00000000E+02,
        "COCHE" => 3.00000000E+02,
        "RANDE" => 2.00000000E+02
);

######################################################################
#                    MAINTENANCE TASKS PARAMETERS                    #
######################################################################

# Definition of the lower bound on the number of available turbines in each powerhouse [turbines] (cf. $\underbar{G}_{i}$)
G_min =	Dict{String, Int64}(
        "TIGNE" => 2,
        "MTRIG" => 7,
        "POMBL" => 0,
        "COCHE" => 3,
        "RANDE" => 3
);

# Definition of the upper bound on the number of available turbines in each powerhouse at each time period [turbines] (cf. $\overline{G}_{i,t})
G_max = Dict{String, Array{Float64, 1}}(
        "TIGNE"	=> Array{Float64, 1}([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]),
        "MTRIG" => Array{Float64, 1}([8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]),
        "POMBL" => Array{Float64, 1}([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        "COCHE" => Array{Float64, 1}([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]),
        "RANDE" => Array{Float64, 1}([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
);

# Definition of the maximum number of turbine outages in each powerhouse at each time period [turbines] (cf. $O_{i,t}$)
O = Dict{String, Array{Float64, 1}}(
        "TIGNE"	=> Array{Float64, 1}([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        "MTRIG" => Array{Float64, 1}([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        "POMBL" => Array{Float64, 1}([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        
        "COCHE" => Array{Float64, 1}([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        "RANDE" => Array{Float64, 1}([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
);

# Definition of the total cost of each maintenance task depending on the time period of beginning [€] (cf. $C_{m,t}$)
C_m_t = convert(Dict{String, Array{Int64, 1}}, JSON.Parser.parse(open(string("data/maintenances/maintenance_tasks.json"), "r"))["C_m_t"]);

# Nominal duration of each maintenance task [days] (cf. $D_{m}$)
D_m = convert(Dict{String, Int64}, JSON.Parser.parse(open(string("data/maintenances/maintenance_tasks.json"), "r"))["D_m"]);

# Earliest start time period of each maintenance task [days] (cf. $E_{m}$)
E_m = convert(Dict{String, Int64}, JSON.Parser.parse(open(string("data/maintenances/maintenance_tasks.json"), "r"))["E_m"]);

# Latest start time period of each maintenance task [days] (cf. $L_{m}$)
L_m = convert(Dict{String, Int64}, JSON.Parser.parse(open(string("data/maintenances/maintenance_tasks.json"), "r"))["L_m"]);
