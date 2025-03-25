<<<<<<< HEAD
######################################################################
#                     PACKAGES OF JULIA (0.6.4)                      #
######################################################################

            ##############################################
            #                  PACKAGES                  #
            ##############################################
import Pkg
ENV["CPLEX_STUDIO_BINARIES"] =
"/opt/ibm/ILOG/CPLEX_Studio2211/cplex/bin/x86-64_linux/"
Pkg.add("CPLEX")
Pkg.build("CPLEX")
Pkg.add("AmplNLWriter")
Pkg.add("JSON")
#Pkg.add("BARON")
Pkg.add("SCIP")


using SCIP
using JuMP;
using CPLEX;
using AmplNLWriter;
using JSON;
using Dates


######################################################################
#                 GENERAL DATA ON THE GIVEN PROBLEM                  #
######################################################################

include("complete_test_case_Z.jl");

######################################################################
#                         BASIC MODEL SOLVER                         #
######################################################################

function solve_model(APPROXIMATION_METHOD::String, SEASON::String, NB_SCENARIOS::Int64, STOCHASTICITY::String, SCHEDULE_PATH::String="none")

######################################################################
#                       DATA ABOUT THE PROBLEM                       #
######################################################################

            ##############################################
            #     DATA ON THE SITUATION OF THE MODEL     #
            ##############################################
    start_time = now()
    include(string("data/scenarios/", SEASON, "/", NB_SCENARIOS, "_scenarios_5T/parameters.jl"));

            ##############################################
            # DATA ON THE STOCHASTICITY OF MAINTENANCES  #
            ##############################################

    include(string("data/maintenances/", STOCHASTICITY, "_stochasticity.jl"));

######################################################################
#                  SETUP OF AN APPROXIMATION METHOD                  #
######################################################################

            ##############################################
            #            CHOICE OF THE METHOD            #
            ##############################################

    CONVEX_HULL_10_HYPERPLANES = (APPROXIMATION_METHOD == "convex_hull_10_hyperplanes");
    CONVEX_HULL_20_HYPERPLANES = (APPROXIMATION_METHOD == "convex_hull_20_hyperplanes");
    CONVEX_HULL_30_HYPERPLANES = (APPROXIMATION_METHOD == "convex_hull_30_hyperplanes");
    PIECEWISE_CONVEX_HULL = (APPROXIMATION_METHOD == "piecewise_convex_hull");
    PIECEWISE_CONVEX_HULL_BIS = (APPROXIMATION_METHOD == "piecewise_convex_hull_bis");
    PIECEWISE_LINEAR = (APPROXIMATION_METHOD == "piecewise_linear");
    POLYNOME = (APPROXIMATION_METHOD == "polynome");

            ##############################################
            #             DATA OF THE METHOD             #
            ##############################################

    if CONVEX_HULL_10_HYPERPLANES
        include("data/production_functions/convex_hull/10_hyperplanes/parameters.jl")
    end;

    if CONVEX_HULL_20_HYPERPLANES
        include("data/production_functions/convex_hull/20_hyperplanes/parameters.jl")
    end;

    if CONVEX_HULL_30_HYPERPLANES
        include("data/production_functions/convex_hull/30_hyperplanes/parameters.jl")
    end;

    if PIECEWISE_CONVEX_HULL
        include("data/production_functions/piecewise_convex_hull/parameters.jl")
    end;

    if PIECEWISE_CONVEX_HULL_BIS
        include("data/production_functions/piecewise_convex_hull_bis/parameters.jl")
    end;

    if PIECEWISE_LINEAR
        include("data/production_functions/piecewise_linear/parameters.jl")
    end;

    if POLYNOME
        include("data/production_functions/polynomial/parameters.jl")
    end;

            ##############################################
            #            SOLVER OF THE METHOD            #
            ##############################################  
    if POLYNOME
        master_problem =  Model(SCIP.Optimizer)
        set_attribute(master_problem, "limits/time", 3600*4)  
             
        
    else
        master_problem = Model(CPLEX.Optimizer)
        set_optimizer_attributes(master_problem, "CPX_PARAM_SCRIND" => 1,
                                       #"CPX_PARAM_THREADS" => 16,
                                       #"CPX_PARAM_PARALLELMODE" => -1,
                                       "CPX_PARAM_TILIM" => 21600,
                                       "CPX_PARAM_WORKMEM" => 20000,
                                       #"CPX_PARAM_EPINT" => 0,
				       #"CPX_PARAM_EPGAP" => 0.00000001,
                                       "CPX_PARAM_TRELIM" => 40000)                              
    end;

######################################################################
#                      DEFINITION OF THE MODEL                       #
######################################################################

            ##############################################
            #                 VARIABLES                  #
            ##############################################

    @variable(master_problem, c_m[m in M] >= 0);

    @variable(master_problem, r_i_t[i in I, t in T], Bin);

    @variable(master_problem, 0 <= y_m_t[m in M, t in T_m[m]] <= 1);

    @variable(master_problem, z_i_t_k[i in I, t in T, k in K_i_t[i][t]], Bin);

    if (POLYNOME || PIECEWISE_CONVEX_HULL)
        @variable(master_problem, p_omega_i_t_k[omega in Omega, i in I, t in T, k in K_i_t[i][t]])
        @variable(master_problem, p_omega_i_t[omega in Omega, i in I, t in T]>= 0)
    else
        @variable(master_problem, p_omega_i_t_k[omega in Omega, i in I, t in T, k in K_i_t[i][t]] )
        @variable(master_problem, p_omega_i_t[omega in Omega, i in I, t in T] >= 0)
    end;

    @variable(master_problem, S_min_i[i] <= s_omega_i_t[omega in Omega, i in I, t in T] <= S_max_i[i]);

    @variable(master_problem, 0 <= u_omega_i_t[omega in Omega, i in I, t in T] <= maximum(U_max_i_k[i].vals));

    @variable(master_problem, 0 <= v_omega_i_t[omega in Omega, i in I, t in T] <= V_max_i[i]); # 

    @variable(master_problem, 0 <= w_plus_omega_t[omega in Omega, t in T] <= W_plus);

    @variable(master_problem, 0 <= w_minus_omega_t[omega in Omega, t in T] <= W_minus);

            ##############################################
            #                 OBJECTIVE                  #
            ##############################################

    @objective(master_problem, Max, sum(Phi_omega[omega]*(sum(B_plus_omega_t[omega][t] * w_plus_omega_t[omega, t] - B_minus_omega_t[omega][t] * w_minus_omega_t[omega, t] for t in T)) for omega in Omega) - sum(c_m[m] for m in M));

            ##############################################
            #                CONSTRAINTS                 #
            ##############################################

    @constraint(master_problem, TASKS_COMPLETION[m in M], sum(y_m_t[m,t] for t in T_m[m]) == 1);

    @constraint(master_problem, TASKS_UNDER_PROCESS[i in I, t in T], sum(sum(y_m_t[m,tt] for tt in T_m[m] if (tt >= t - D_m[m] + 1 && tt <= t)) + (t-D_m[m] in T_m[m] ? Alpha_1 * y_m_t[m,t-D_m[m]] : 0) + (t-D_m[m]-1 in T_m[m] ? Alpha_2 * y_m_t[m,t-D_m[m]-1] : 0) for m in M_i[i]) == r_i_t[i,t]);
    #@constraint(master_problem, TASKS_UNDER_PROCESS_corec[i in I, t in T], sum(sum(y_m_t[m,tt] for tt in T_m[m] if (tt >= t - D_m[m] + 1 && tt <= t))  for m in M_i[i]), Bin);

    @constraint(master_problem, AVAILABLE_TURBINES_MAP[i in I, t in T], (r_i_t[i,t] - Nu) + sum(k * z_i_t_k[i,t,k] for k in K_i_t[i][t]) <= G_max[i][t]);

    @constraint(master_problem, AVAILABLE_TURBINES_CHOICE[i in I, t in T], sum(z_i_t_k[i,t,k] for k in K_i_t[i][t]) == 1);

    @constraint(master_problem, MAINTENANCE_COST[m in M], c_m[m] == sum(C_m_t[m][t] * y_m_t[m,t] for t in T_m[m]) + Gamma_1 * Alpha_1 * C_m_t[m][L_m[m]] * y_m_t[m,L_m[m]] + Gamma_2 * Alpha_2 * C_m_t[m][L_m[m]-1] * y_m_t[m,L_m[m]-1]);

    @constraint(master_problem, TURBINE_DISCHARGE[omega in Omega, t in T, i in I], u_omega_i_t[omega, i, t] <= sum(z_i_t_k[i, t, k] * U_max_i_k[i][k] for k in K_i_t[i][t]));

    #@constraint(master_problem, INITIAL_VOLUME[omega in Omega, i in I], s_omega_i_t[omega, i, 1] == S_init_i[i]);

    @constraint(master_problem, HYDRAULIC_BALANCE[omega in Omega, t in T, i in I], s_omega_i_t[omega,i,t] - (t > 1 ? s_omega_i_t[omega,i,t-1] : S_init_i[i]) == Q * (F_omega_i_t[omega][i][t] + sum(u_omega_i_t[omega,g,t] + v_omega_i_t[omega,g,t] for g in U_i[i]) - u_omega_i_t[omega,i,t] - v_omega_i_t[omega,i,t]));

    @constraint(master_problem, ENERGY_BALANCE[omega in Omega, t in T], sum(p_omega_i_t[omega,i,t] for i in I) + w_minus_omega_t[omega,t] == J_omega_t[omega][t] + w_plus_omega_t[omega,t]);

    @constraint(master_problem, HYDROPOWER_FUNCTION_CHOICE[omega in Omega, t in T, i in I, k in K_i_t[i][t]], p_omega_i_t_k[omega,i,t,k] <= z_i_t_k[i,t,k] * P_max_i_k[i][k]);

    @constraint(master_problem, HYDROPOWER_FUNCTION_COMPUTATION[omega in Omega, t in T, i in I], p_omega_i_t[omega,i,t] == sum(p_omega_i_t_k[omega,i,t,k] for k in K_i_t[i][t]));

    if (CONVEX_HULL_10_HYPERPLANES || CONVEX_HULL_20_HYPERPLANES || CONVEX_HULL_30_HYPERPLANES)
        @constraint(master_problem, CONVEX_HULL_APPROXIMATION[omega in Omega, t in T, i in I, k in K_i_t[i][t], h in H], p_omega_i_t_k[omega,i,t,k] <= Beta_0_i_k_h[i][k][h] + Beta_u_i_k_h[i][k][h] * u_omega_i_t[omega,i,t] + Beta_s_i_k_h[i][k][h] * s_omega_i_t[omega,i,t])
    end;

    if PIECEWISE_CONVEX_HULL
        @variable(master_problem, z_omega_i_t_k_h[omega in Omega, i in I, t in T, k in K_i_t[i][t], h in 1:L_i_k[i][k]], Bin);
        @variable(master_problem, p_omega_i_t_k_h[omega in Omega, i in I, t in T, k in K_i_t[i][t], h in 1:L_i_k[i][k]] <= (h == L_i_k[i][k] ? P_max_i_k[i][k] : P_max_i_k_h[i][k][h]));
        @variable(master_problem, u_omega_i_t_k_h[omega in Omega, i in I, t in T, k in K_i_t[i][t], h in 1:L_i_k[i][k]] >= 0);
        @variable(master_problem, s_omega_i_t_k_h[omega in Omega, i in I, t in T, k in K_i_t[i][t], h in 1:L_i_k[i][k]] >= 0);

        @constraint(master_problem, UPPER_CONVEX_HULL_APPROXIMATION[omega in Omega, i in I, t in T, k in K_i_t[i][t], h in 1:U_i_k[i][k]], p_omega_i_t_k_h[omega,i,t,k,L_i_k[i][k]] <= Upper_Beta_0_i_k_h[i][k][h] + Upper_Beta_u_i_k_h[i][k][h] * u_omega_i_t_k_h[omega,i,t,k,L_i_k[i][k]] + Upper_Beta_s_i_k_h[i][k][h] * s_omega_i_t_k_h[omega,i,t,k,L_i_k[i][k]])
        @constraint(master_problem, LOWER_CONVEX_HULL_APPROXIMATION[omega in Omega, i in I, t in T, k in K_i_t[i][t], h in 1:L_i_k[i][k]], p_omega_i_t_k_h[omega,i,t,k,h] <= Lower_Beta_0_i_k_h[i][k][h] + Lower_Beta_u_i_k_h[i][k][h] * u_omega_i_t_k_h[omega,i,t,k,h] + Lower_Beta_s_i_k_h[i][k][h] * s_omega_i_t_k_h[omega,i,t,k,h])

        @constraint(master_problem, CHOICE_OF_LOWER_HYPERPLANE[omega in Omega, i in I, t in T, k in K_i_t[i][t], h in 1:L_i_k[i][k]], p_omega_i_t_k_h[omega,i,t,k,h] <= z_omega_i_t_k_h[omega,i,t,k,h] * (h == L_i_k[i][k] ? P_max_i_k[i][k] : P_max_i_k_h[i][k][h]))
        @constraint(master_problem, CHOICE_CONSEQUENCE_ON_U[omega in Omega, i in I, t in T, k in K_i_t[i][t], h in 1:L_i_k[i][k]], u_omega_i_t_k_h[omega,i,t,k,h] <= z_omega_i_t_k_h[omega,i,t,k,h] * U_max_i_k[i][k])
        @constraint(master_problem, CHOICE_CONSEQUENCE_ON_S[omega in Omega, i in I, t in T, k in K_i_t[i][t], h in 1:L_i_k[i][k]], s_omega_i_t_k_h[omega,i,t,k,h] <= z_omega_i_t_k_h[omega,i,t,k,h] * S_max_i[i])
        @constraint(master_problem, CHOICE_OF_FUNCTION[omega in Omega, i in I, t in T, k in K_i_t[i][t]], sum(z_omega_i_t_k_h[omega,i,t,k,h] for h in 1:L_i_k[i][k]) == z_i_t_k[i,t,k])

        @constraint(master_problem, ELECTRICITY_GENERATION_COMPUTATION[omega in Omega, i in I, t in T, k in K_i_t[i][t]], p_omega_i_t_k[omega,i,t,k] == sum(p_omega_i_t_k_h[omega,i,t,k,h] - (1 - z_omega_i_t_k_h[omega,i,t,k,h]) * (h == L_i_k[i][k] ? min(Lower_Beta_0_i_k_h[i][k][h], minimum(Upper_Beta_0_i_k_h[i][k])) : Lower_Beta_0_i_k_h[i][k][h]) for h in 1:L_i_k[i][k]))
        @constraint(master_problem, TURBINE_DISCHARGE_COMPUTATION[omega in Omega, i in I, t in T], u_omega_i_t[omega,i,t] == sum(u_omega_i_t_k_h[omega,i,t,k,h] for k in K_i_t[i][t] for h in 1:L_i_k[i][k]))
        @constraint(master_problem, WATER_STORAGE_COMPUTATION[omega in Omega, i in I, t in T], s_omega_i_t[omega,i,t] == sum(s_omega_i_t_k_h[omega,i,t,k,h] for k in K_i_t[i][t] for h in 1:L_i_k[i][k]))
    end;

    if PIECEWISE_CONVEX_HULL_BIS
        @variable(master_problem, z_omega_i_t_k_h[omega in Omega, i in I, t in T, k in K_i_t[i][t], h in 1:L_i_k[i][k]], Bin);

        @constraint(master_problem, UPPER_CONVEX_HULL_APPROXIMATION[omega in Omega, i in I, t in T, k in K_i_t[i][t], h in 1:U_i_k[i][k]], p_omega_i_t_k[omega,i,t,k] <= Upper_Beta_0_i_k_h[i][k][h] + Upper_Beta_u_i_k_h[i][k][h] * u_omega_i_t[omega,i,t] + Upper_Beta_s_i_k_h[i][k][h] * s_omega_i_t[omega,i,t] + (1-z_omega_i_t_k_h[omega,i,t,k,L_i_k[i][k]]) * Upper_Compensation_i_k_h[i][k][h])
        @constraint(master_problem, LOWER_CONVEX_HULL_APPROXIMATION[omega in Omega, i in I, t in T, k in K_i_t[i][t], h in 1:L_i_k[i][k]], p_omega_i_t_k[omega,i,t,k] <= Lower_Beta_0_i_k_h[i][k][h] + Lower_Beta_u_i_k_h[i][k][h] * u_omega_i_t[omega,i,t] + Lower_Beta_s_i_k_h[i][k][h] * s_omega_i_t[omega,i,t] + (1-z_omega_i_t_k_h[omega,i,t,k,h]) * Lower_Compensation_i_k_h[i][k][h])

        @constraint(master_problem, LOWER_BOUND_ON_POWER_GENERATION[omega in Omega, i in I, t in T, k in K_i_t[i][t]], sum(z_omega_i_t_k_h[omega,i,t,k,h] * P_min_i_k_h[i][k][h] for h in 1:L_i_k[i][k]) <= p_omega_i_t_k[omega,i,t,k])
        @constraint(master_problem, UPPER_BOUND_ON_POWER_GENERATION[omega in Omega, i in I, t in T, k in K_i_t[i][t]], p_omega_i_t_k[omega,i,t,k] <= sum(z_omega_i_t_k_h[omega,i,t,k,h] * P_max_i_k_h[i][k][h] for h in 1:L_i_k[i][k]))
        @constraint(master_problem, LOWER_BOUND_ON_TURBINE_DISCHARGE[omega in Omega, i in I, t in T], sum(sum(z_omega_i_t_k_h[omega,i,t,k,h] * U_min_i_k_h[i][k][h] for h in 1:L_i_k[i][k]) for k in K_i_t[i][t]) <= u_omega_i_t[omega,i,t])
        @constraint(master_problem, UPPER_BOUND_ON_TURBINE_DISCHARGE[omega in Omega, i in I, t in T], u_omega_i_t[omega,i,t] <= sum(sum(z_omega_i_t_k_h[omega,i,t,k,h] * U_max_i_k_h[i][k][h] for h in 1:L_i_k[i][k]) for k in K_i_t[i][t]))
        @constraint(master_problem, CHOICE_OF_FUNCTION[omega in Omega, i in I, t in T, k in K_i_t[i][t]], sum(z_omega_i_t_k_h[omega,i,t,k,h] for h in 1:L_i_k[i][k]) == z_i_t_k[i,t,k])
    end;

    if PIECEWISE_LINEAR
        @variable(master_problem, h_omega_i_t_k_nu[omega in Omega, i in I, t in T, k in K_i_t[i][t], nu in 1:(N_u-1)], Bin);
        @variable(master_problem, 0 <= alpha_omega_i_t_k_nu[omega in Omega, i in I, t in T, k in K_i_t[i][t], nu in 1:N_u] <= 1);
        @variable(master_problem, beta_omega_i_t_k_ns[omega in Omega, i in I, t in T, k in K_i_t[i][t], ns in 1:(N_s-1)], Bin);
        @variable(master_problem, 0 <= gamma_omega_i_t_k_ns[omega in Omega, i in I, t in T, k in K_i_t[i][t], ns in 1:(N_s-1)] <= 1);

        @constraint(master_problem, ALPHA_CHOICE[omega in Omega, t in T, i in I, k in K_i_t[i][t], nu in 1:N_u], alpha_omega_i_t_k_nu[omega, i, t, k, nu] <= (nu < N_u ? h_omega_i_t_k_nu[omega, i, t, k, nu] : 0) + (nu > 1 ? h_omega_i_t_k_nu[omega, i, t, k, nu-1] : 0))
        @constraint(master_problem, ALPHA[omega in Omega, i in I, t in T, k in K_i_t[i][t]], sum(alpha_omega_i_t_k_nu[omega, i, t, k, nu] for nu in 1:N_u) == 1);
        @constraint(master_problem, PW_24[omega in Omega, t in T, i in I, k in K_i_t[i][t]], u_omega_i_t[omega, i, t] == sum(alpha_omega_i_t_k_nu[omega, i, t, k, nu] * Beta_u_i_k_nu[i][k][nu] for nu in 1:N_u))

        @constraint(master_problem, BETA[omega in Omega, i in I, t in T, k in K_i_t[i][t]], sum(beta_omega_i_t_k_ns[omega, i, t, k, ns] for ns in 1:(N_s-1)) == 1);
        @constraint(master_problem, PW_26[omega in Omega, t in T, i in I, k in K_i_t[i][t], ns in 1:(N_s-1)], gamma_omega_i_t_k_ns[omega, i, t, k, ns] <= beta_omega_i_t_k_ns[omega, i, t, k, ns])
        @constraint(master_problem, PW_27[omega in Omega, t in T, i in I, k in K_i_t[i][t]], s_omega_i_t[omega, i, t] == sum(beta_omega_i_t_k_ns[omega, i, t, k, ns] * Beta_s_i_k_ns[i][k][ns] + gamma_omega_i_t_k_ns[omega, i, t, k, ns] * (Beta_s_i_k_ns[i][k][ns+1] - Beta_s_i_k_ns[i][k][ns]) for ns in 1:(N_s-1)))

        @constraint(master_problem, PW_28_29[omega in Omega, t in T, i in I, k in K_i_t[i][t], ns in 1:(N_s-1), nu in 1:(N_u-1)], p_omega_i_t_k[omega,i,t,k] <= sum(alpha_omega_i_t_k_nu[omega, i, t, k, nuu] * Beta_p_i_k_ns_nu[i][k][ns][nuu] for nuu in 1:N_u) + gamma_omega_i_t_k_ns[omega, i, t, k, ns] * K_i_k_ns_nu[i][k][ns][nu] + P_max_i_k[i][k] * (2 - beta_omega_i_t_k_ns[omega, i, t, k, ns] - h_omega_i_t_k_nu[omega,i,t,k,nu]))
    end;
   
    if POLYNOME
        @NLconstraint(master_problem, Polynomial[omega in Omega, t in T, i in I, k in K_i_t[i][t]], p_omega_i_t_k[omega,i,t,k] <= Poly[i][k][1] + s_omega_i_t[omega,i,t]*Poly[i][k][2] + u_omega_i_t[omega,i,t]*Poly[i][k][3] + s_omega_i_t[omega,i,t]^2*Poly[i][k][4] + u_omega_i_t[omega,i,t]*s_omega_i_t[omega,i,t]*Poly[i][k][5] + u_omega_i_t[omega,i,t]^2*Poly[i][k][6] + s_omega_i_t[omega,i,t]^2*u_omega_i_t[omega,i,t]*Poly[i][k][7] + s_omega_i_t[omega,i,t]*u_omega_i_t[omega,i,t]^2*Poly[i][k][8] + u_omega_i_t[omega,i,t]^3*Poly[i][k][9] + s_omega_i_t[omega,i,t]^2*u_omega_i_t[omega,i,t]^2*Poly[i][k][10] + s_omega_i_t[omega,i,t]*u_omega_i_t[omega,i,t]^3*Poly[i][k][11] + u_omega_i_t[omega,i,t]^4*Poly[i][k][12] + s_omega_i_t[omega,i,t]^2*u_omega_i_t[omega,i,t]^3*Poly[i][k][13] + u_omega_i_t[omega,i,t]^4*s_omega_i_t[omega,i,t]*Poly[i][k][14] + u_omega_i_t[omega,i,t]^5*Poly[i][k][15]);
    end;
   
    ##############################################
    #       FIXED SCHEDULE BY OTHER METHOD       #
    ##############################################

    if SCHEDULE_PATH != "none"

        original_results = JSON.Parser.parse(open(SCHEDULE_PATH, "r"));

        # Use 'JuMP.fix' to get the local optimum associated to the method's schedule and 'setvalue' if you just want to help the new method with starting values

        for m in M
            for t in T_m[m]
                JuMP.fix(y_m_t[m,t], original_results["y_m_t"][m][t])
            end
        end;

        for i in I
            for t in T
                JuMP.fix(r_i_t[i,t], original_results["r_i_t"][i][t])
            end
        end;

        for i in I
            for t in T
                for k in K_i_t[i][t]
                    JuMP.fix(z_i_t_k[i,t,k], original_results["z_i_t"][i][t] == k ? 1 : 0)
                end
            end
        end;
    end;

######################################################################
#                         SOLVING THE MODEL                          #
######################################################################

            ##############################################
            #             CALLING THE SOLVER             #
            ##############################################
            
    optimize!(master_problem);
   
    end_time = now()
    elapsed_time = (end_time - start_time)
    status = termination_status(master_problem)

    if status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED
        obj_value = objective_value(master_problem)
        println("Optimal objective value: ", obj_value)
    else
        println("No solution found or the problem is not optimally solved.")
    end

            ##############################################
            #          STATISTICS ON THE SOLVER          #
            ##############################################

    println("\t- Final status: ", status);
    println("\t- Time to solve the model: ", Dates.value(elapsed_time) , " seconds");
    println("\t- Objective value: ", JuMP.objective_value(master_problem));

######################################################################
#                         SAVING THE RESULTS                         #
######################################################################

            ##############################################
            #           FORMATTING THE VALUES            #
            ##############################################

    treated_y_m_t = Dict{String, Array{Float64, 1}}(m => Array{Float64, 1}([sum(((tt >= t - D_m[m] + 1 && tt <= t) ? value(y_m_t[m,tt]) : 0) for tt in T_m[m]) + (t-D_m[m] in T_m[m] ? Alpha_1 * value(y_m_t[m,t-D_m[m]]) : 0) + (t-D_m[m]-1 in T_m[m] ? Alpha_2 * value(y_m_t[m,t-D_m[m]-1]) : 0) for t in sort([t for t in T])]) for m in M);
    raw_y_m_t = Dict{String, Array{Float64, 1}}(m => Array{Float64, 1}([(t in T_m[m] ? value(y_m_t[m, t]) : 0) for t in sort([t for t in T])]) for m in M);
    raw_r_i_t = Dict{String, Array{Float64, 1}}(i => Array{Float64, 1}([value(r_i_t[i, t]) for t in sort([t for t in T])]) for i in I);
    raw_z_i_t = Dict{String, Array{Int64, 1}}(i => Array{Int64, 1}([sum(round(value(z_i_t_k[i, t, k])) * k for k in K_i_t[i][t]) for t in sort([t for t in T])]) for i in I);
    raw_s_omega_i_t = Dict{String, Dict{String, Array{Float64, 1}}}(omega => Dict{String, Array{Float64, 1}}(i => Array{Float64, 1}([value(s_omega_i_t[omega, i, t]) for t in sort([t for t in T])]) for i in I) for omega in Omega);
    raw_v_omega_i_t = Dict{String, Dict{String, Array{Float64, 1}}}(omega => Dict{String, Array{Float64, 1}}(i => Array{Float64, 1}([value(v_omega_i_t[omega, i, t]) for t in sort([t for t in T])]) for i in I) for omega in Omega);
    raw_u_omega_i_t = Dict{String, Dict{String, Array{Float64, 1}}}(omega => Dict{String, Array{Float64, 1}}(i => Array{Float64, 1}([value(u_omega_i_t[omega, i, t]) for t in sort([t for t in T])]) for i in I) for omega in Omega);
    raw_p_omega_i_t = Dict{String, Dict{String, Array{Float64, 1}}}(omega => Dict{String, Array{Float64, 1}}(i => Array{Float64, 1}([value(p_omega_i_t[omega, i, t]) for t in sort([t for t in T])]) for i in I) for omega in Omega);
    raw_c_m = Dict{String, Float64}(m => value(c_m[m]) for m in M);

            ##############################################
            #           FOLDER OF THE RESULTS            #
            ##############################################

    mkpath(string("resultatScip/", SEASON, "/", NB_SCENARIOS, "_scenarios/", STOCHASTICITY, "_stochasticity/"));

            ##############################################
            #             SAVING THE RESULTS             #
            ##############################################

    open(string("resultat/", SEASON, "/", NB_SCENARIOS, "_scenarios/", STOCHASTICITY, "_stochasticity/", APPROXIMATION_METHOD,"_5T_C2.json"), "w") do all_results_file
        JSON.write(all_results_file, JSON.json(Dict{String, Any}("scenarios" => NB_SCENARIOS,
                                                                 "status" => status,
                                                                 "time" => Dates.value(elapsed_time),
                                                                 "objective_value" => JuMP.objective_value(master_problem),
                                                                 #"objective_bound" => getobjectivebound(master_problem),
                                                                 "cumulated_y_m_t" => treated_y_m_t,
                                                                 "y_m_t" => raw_y_m_t,
                                                                 "c_m" => raw_c_m,
                                                                 "r_i_t" => raw_r_i_t,
                                                                 "z_i_t" => raw_z_i_t,
                                                                 "s_omega_i_t" => raw_s_omega_i_t,
                                                                 "v_omega_i_t" => raw_v_omega_i_t,
                                                                 "u_omega_i_t" => raw_u_omega_i_t,
                                                                 "p_omega_i_t" => raw_p_omega_i_t),
                                               4))
    end;
end;

solve_model("piecewise_convex_hull", "winter",2, "no")
println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
println("~               HOW TO USE               ~")
println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
println("~         Function: solve_model          ~")
println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
println("~ - Arg1: APPROXIMATION_METHOD           ~")
println("~ - Arg2: SEASON                         ~")
println("~ - Arg3: NB_SCENARIOS                   ~")
println("~ - Arg4: STOCHASTICITY                  ~")
println("~ - Arg5: SCHEDULE_PATH (OPTIONAL)       ~")
println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
=======
######################################################################
#                     PACKAGES OF JULIA (0.6.4)                      #
######################################################################

            ##############################################
            #                  PACKAGES                  #
            ##############################################
import Pkg
ENV["CPLEX_STUDIO_BINARIES"] =
"/opt/ibm/ILOG/CPLEX_Studio2211/cplex/bin/x86-64_linux/"
Pkg.add("CPLEX")
Pkg.build("CPLEX")
Pkg.add("AmplNLWriter")
Pkg.add("JSON")
#Pkg.add("BARON")
Pkg.add("SCIP")


using SCIP
using JuMP;
using CPLEX;
using AmplNLWriter;
using JSON;
using Dates


######################################################################
#                 GENERAL DATA ON THE GIVEN PROBLEM                  #
######################################################################

include("complete_test_case_Z.jl");

######################################################################
#                         BASIC MODEL SOLVER                         #
######################################################################

function solve_model(APPROXIMATION_METHOD::String, SEASON::String, NB_SCENARIOS::Int64, STOCHASTICITY::String, SCHEDULE_PATH::String="none")

######################################################################
#                       DATA ABOUT THE PROBLEM                       #
######################################################################

            ##############################################
            #     DATA ON THE SITUATION OF THE MODEL     #
            ##############################################
    start_time = now()
    include(string("data/scenarios/", SEASON, "/", NB_SCENARIOS, "_scenarios_5T/parameters.jl"));

            ##############################################
            # DATA ON THE STOCHASTICITY OF MAINTENANCES  #
            ##############################################

    include(string("data/maintenances/", STOCHASTICITY, "_stochasticity.jl"));

######################################################################
#                  SETUP OF AN APPROXIMATION METHOD                  #
######################################################################

            ##############################################
            #            CHOICE OF THE METHOD            #
            ##############################################

    CONVEX_HULL_10_HYPERPLANES = (APPROXIMATION_METHOD == "convex_hull_10_hyperplanes");
    CONVEX_HULL_20_HYPERPLANES = (APPROXIMATION_METHOD == "convex_hull_20_hyperplanes");
    CONVEX_HULL_30_HYPERPLANES = (APPROXIMATION_METHOD == "convex_hull_30_hyperplanes");
    PIECEWISE_CONVEX_HULL = (APPROXIMATION_METHOD == "piecewise_convex_hull");
    PIECEWISE_CONVEX_HULL_BIS = (APPROXIMATION_METHOD == "piecewise_convex_hull_bis");
    PIECEWISE_LINEAR = (APPROXIMATION_METHOD == "piecewise_linear");
    POLYNOME = (APPROXIMATION_METHOD == "polynome");

            ##############################################
            #             DATA OF THE METHOD             #
            ##############################################

    if CONVEX_HULL_10_HYPERPLANES
        include("data/production_functions/convex_hull/10_hyperplanes/parameters.jl")
    end;

    if CONVEX_HULL_20_HYPERPLANES
        include("data/production_functions/convex_hull/20_hyperplanes/parameters.jl")
    end;

    if CONVEX_HULL_30_HYPERPLANES
        include("data/production_functions/convex_hull/30_hyperplanes/parameters.jl")
    end;

    if PIECEWISE_CONVEX_HULL
        include("data/production_functions/piecewise_convex_hull/parameters.jl")
    end;

    if PIECEWISE_CONVEX_HULL_BIS
        include("data/production_functions/piecewise_convex_hull_bis/parameters.jl")
    end;

    if PIECEWISE_LINEAR
        include("data/production_functions/piecewise_linear/parameters.jl")
    end;

    if POLYNOME
        include("data/production_functions/polynomial/parameters.jl")
    end;

            ##############################################
            #            SOLVER OF THE METHOD            #
            ##############################################  
    if POLYNOME
        master_problem =  Model(SCIP.Optimizer)
        set_attribute(master_problem, "limits/time", 3600*4)  
             
        
    else
        master_problem = Model(CPLEX.Optimizer)
        set_optimizer_attributes(master_problem, "CPX_PARAM_SCRIND" => 1,
                                       #"CPX_PARAM_THREADS" => 16,
                                       #"CPX_PARAM_PARALLELMODE" => -1,
                                       "CPX_PARAM_TILIM" => 21600,
                                       "CPX_PARAM_WORKMEM" => 20000,
                                       #"CPX_PARAM_EPINT" => 0,
				       #"CPX_PARAM_EPGAP" => 0.00000001,
                                       "CPX_PARAM_TRELIM" => 40000)                              
    end;

######################################################################
#                      DEFINITION OF THE MODEL                       #
######################################################################

            ##############################################
            #                 VARIABLES                  #
            ##############################################

    @variable(master_problem, c_m[m in M] >= 0);

    @variable(master_problem, r_i_t[i in I, t in T], Bin);

    @variable(master_problem, 0 <= y_m_t[m in M, t in T_m[m]] <= 1);

    @variable(master_problem, z_i_t_k[i in I, t in T, k in K_i_t[i][t]], Bin);

    if (POLYNOME || PIECEWISE_CONVEX_HULL)
        @variable(master_problem, p_omega_i_t_k[omega in Omega, i in I, t in T, k in K_i_t[i][t]])
        @variable(master_problem, p_omega_i_t[omega in Omega, i in I, t in T]>= 0)
    else
        @variable(master_problem, p_omega_i_t_k[omega in Omega, i in I, t in T, k in K_i_t[i][t]] )
        @variable(master_problem, p_omega_i_t[omega in Omega, i in I, t in T] >= 0)
    end;

    @variable(master_problem, S_min_i[i] <= s_omega_i_t[omega in Omega, i in I, t in T] <= S_max_i[i]);

    @variable(master_problem, 0 <= u_omega_i_t[omega in Omega, i in I, t in T] <= maximum(U_max_i_k[i].vals));

    @variable(master_problem, 0 <= v_omega_i_t[omega in Omega, i in I, t in T] <= V_max_i[i]); # 

    @variable(master_problem, 0 <= w_plus_omega_t[omega in Omega, t in T] <= W_plus);

    @variable(master_problem, 0 <= w_minus_omega_t[omega in Omega, t in T] <= W_minus);

            ##############################################
            #                 OBJECTIVE                  #
            ##############################################

    @objective(master_problem, Max, sum(Phi_omega[omega]*(sum(B_plus_omega_t[omega][t] * w_plus_omega_t[omega, t] - B_minus_omega_t[omega][t] * w_minus_omega_t[omega, t] for t in T)) for omega in Omega) - sum(c_m[m] for m in M));

            ##############################################
            #                CONSTRAINTS                 #
            ##############################################

    @constraint(master_problem, TASKS_COMPLETION[m in M], sum(y_m_t[m,t] for t in T_m[m]) == 1);

    @constraint(master_problem, TASKS_UNDER_PROCESS[i in I, t in T], sum(sum(y_m_t[m,tt] for tt in T_m[m] if (tt >= t - D_m[m] + 1 && tt <= t)) + (t-D_m[m] in T_m[m] ? Alpha_1 * y_m_t[m,t-D_m[m]] : 0) + (t-D_m[m]-1 in T_m[m] ? Alpha_2 * y_m_t[m,t-D_m[m]-1] : 0) for m in M_i[i]) == r_i_t[i,t]);
    #@constraint(master_problem, TASKS_UNDER_PROCESS_corec[i in I, t in T], sum(sum(y_m_t[m,tt] for tt in T_m[m] if (tt >= t - D_m[m] + 1 && tt <= t))  for m in M_i[i]), Bin);

    @constraint(master_problem, AVAILABLE_TURBINES_MAP[i in I, t in T], (r_i_t[i,t] - Nu) + sum(k * z_i_t_k[i,t,k] for k in K_i_t[i][t]) <= G_max[i][t]);

    @constraint(master_problem, AVAILABLE_TURBINES_CHOICE[i in I, t in T], sum(z_i_t_k[i,t,k] for k in K_i_t[i][t]) == 1);

    @constraint(master_problem, MAINTENANCE_COST[m in M], c_m[m] == sum(C_m_t[m][t] * y_m_t[m,t] for t in T_m[m]) + Gamma_1 * Alpha_1 * C_m_t[m][L_m[m]] * y_m_t[m,L_m[m]] + Gamma_2 * Alpha_2 * C_m_t[m][L_m[m]-1] * y_m_t[m,L_m[m]-1]);

    @constraint(master_problem, TURBINE_DISCHARGE[omega in Omega, t in T, i in I], u_omega_i_t[omega, i, t] <= sum(z_i_t_k[i, t, k] * U_max_i_k[i][k] for k in K_i_t[i][t]));

    #@constraint(master_problem, INITIAL_VOLUME[omega in Omega, i in I], s_omega_i_t[omega, i, 1] == S_init_i[i]);

    @constraint(master_problem, HYDRAULIC_BALANCE[omega in Omega, t in T, i in I], s_omega_i_t[omega,i,t] - (t > 1 ? s_omega_i_t[omega,i,t-1] : S_init_i[i]) == Q * (F_omega_i_t[omega][i][t] + sum(u_omega_i_t[omega,g,t] + v_omega_i_t[omega,g,t] for g in U_i[i]) - u_omega_i_t[omega,i,t] - v_omega_i_t[omega,i,t]));

    @constraint(master_problem, ENERGY_BALANCE[omega in Omega, t in T], sum(p_omega_i_t[omega,i,t] for i in I) + w_minus_omega_t[omega,t] == J_omega_t[omega][t] + w_plus_omega_t[omega,t]);

    @constraint(master_problem, HYDROPOWER_FUNCTION_CHOICE[omega in Omega, t in T, i in I, k in K_i_t[i][t]], p_omega_i_t_k[omega,i,t,k] <= z_i_t_k[i,t,k] * P_max_i_k[i][k]);

    @constraint(master_problem, HYDROPOWER_FUNCTION_COMPUTATION[omega in Omega, t in T, i in I], p_omega_i_t[omega,i,t] == sum(p_omega_i_t_k[omega,i,t,k] for k in K_i_t[i][t]));

    if (CONVEX_HULL_10_HYPERPLANES || CONVEX_HULL_20_HYPERPLANES || CONVEX_HULL_30_HYPERPLANES)
        @constraint(master_problem, CONVEX_HULL_APPROXIMATION[omega in Omega, t in T, i in I, k in K_i_t[i][t], h in H], p_omega_i_t_k[omega,i,t,k] <= Beta_0_i_k_h[i][k][h] + Beta_u_i_k_h[i][k][h] * u_omega_i_t[omega,i,t] + Beta_s_i_k_h[i][k][h] * s_omega_i_t[omega,i,t])
    end;

    if PIECEWISE_CONVEX_HULL
        @variable(master_problem, z_omega_i_t_k_h[omega in Omega, i in I, t in T, k in K_i_t[i][t], h in 1:L_i_k[i][k]], Bin);
        @variable(master_problem, p_omega_i_t_k_h[omega in Omega, i in I, t in T, k in K_i_t[i][t], h in 1:L_i_k[i][k]] <= (h == L_i_k[i][k] ? P_max_i_k[i][k] : P_max_i_k_h[i][k][h]));
        @variable(master_problem, u_omega_i_t_k_h[omega in Omega, i in I, t in T, k in K_i_t[i][t], h in 1:L_i_k[i][k]] >= 0);
        @variable(master_problem, s_omega_i_t_k_h[omega in Omega, i in I, t in T, k in K_i_t[i][t], h in 1:L_i_k[i][k]] >= 0);

        @constraint(master_problem, UPPER_CONVEX_HULL_APPROXIMATION[omega in Omega, i in I, t in T, k in K_i_t[i][t], h in 1:U_i_k[i][k]], p_omega_i_t_k_h[omega,i,t,k,L_i_k[i][k]] <= Upper_Beta_0_i_k_h[i][k][h] + Upper_Beta_u_i_k_h[i][k][h] * u_omega_i_t_k_h[omega,i,t,k,L_i_k[i][k]] + Upper_Beta_s_i_k_h[i][k][h] * s_omega_i_t_k_h[omega,i,t,k,L_i_k[i][k]])
        @constraint(master_problem, LOWER_CONVEX_HULL_APPROXIMATION[omega in Omega, i in I, t in T, k in K_i_t[i][t], h in 1:L_i_k[i][k]], p_omega_i_t_k_h[omega,i,t,k,h] <= Lower_Beta_0_i_k_h[i][k][h] + Lower_Beta_u_i_k_h[i][k][h] * u_omega_i_t_k_h[omega,i,t,k,h] + Lower_Beta_s_i_k_h[i][k][h] * s_omega_i_t_k_h[omega,i,t,k,h])

        @constraint(master_problem, CHOICE_OF_LOWER_HYPERPLANE[omega in Omega, i in I, t in T, k in K_i_t[i][t], h in 1:L_i_k[i][k]], p_omega_i_t_k_h[omega,i,t,k,h] <= z_omega_i_t_k_h[omega,i,t,k,h] * (h == L_i_k[i][k] ? P_max_i_k[i][k] : P_max_i_k_h[i][k][h]))
        @constraint(master_problem, CHOICE_CONSEQUENCE_ON_U[omega in Omega, i in I, t in T, k in K_i_t[i][t], h in 1:L_i_k[i][k]], u_omega_i_t_k_h[omega,i,t,k,h] <= z_omega_i_t_k_h[omega,i,t,k,h] * U_max_i_k[i][k])
        @constraint(master_problem, CHOICE_CONSEQUENCE_ON_S[omega in Omega, i in I, t in T, k in K_i_t[i][t], h in 1:L_i_k[i][k]], s_omega_i_t_k_h[omega,i,t,k,h] <= z_omega_i_t_k_h[omega,i,t,k,h] * S_max_i[i])
        @constraint(master_problem, CHOICE_OF_FUNCTION[omega in Omega, i in I, t in T, k in K_i_t[i][t]], sum(z_omega_i_t_k_h[omega,i,t,k,h] for h in 1:L_i_k[i][k]) == z_i_t_k[i,t,k])

        @constraint(master_problem, ELECTRICITY_GENERATION_COMPUTATION[omega in Omega, i in I, t in T, k in K_i_t[i][t]], p_omega_i_t_k[omega,i,t,k] == sum(p_omega_i_t_k_h[omega,i,t,k,h] - (1 - z_omega_i_t_k_h[omega,i,t,k,h]) * (h == L_i_k[i][k] ? min(Lower_Beta_0_i_k_h[i][k][h], minimum(Upper_Beta_0_i_k_h[i][k])) : Lower_Beta_0_i_k_h[i][k][h]) for h in 1:L_i_k[i][k]))
        @constraint(master_problem, TURBINE_DISCHARGE_COMPUTATION[omega in Omega, i in I, t in T], u_omega_i_t[omega,i,t] == sum(u_omega_i_t_k_h[omega,i,t,k,h] for k in K_i_t[i][t] for h in 1:L_i_k[i][k]))
        @constraint(master_problem, WATER_STORAGE_COMPUTATION[omega in Omega, i in I, t in T], s_omega_i_t[omega,i,t] == sum(s_omega_i_t_k_h[omega,i,t,k,h] for k in K_i_t[i][t] for h in 1:L_i_k[i][k]))
    end;

    if PIECEWISE_CONVEX_HULL_BIS
        @variable(master_problem, z_omega_i_t_k_h[omega in Omega, i in I, t in T, k in K_i_t[i][t], h in 1:L_i_k[i][k]], Bin);

        @constraint(master_problem, UPPER_CONVEX_HULL_APPROXIMATION[omega in Omega, i in I, t in T, k in K_i_t[i][t], h in 1:U_i_k[i][k]], p_omega_i_t_k[omega,i,t,k] <= Upper_Beta_0_i_k_h[i][k][h] + Upper_Beta_u_i_k_h[i][k][h] * u_omega_i_t[omega,i,t] + Upper_Beta_s_i_k_h[i][k][h] * s_omega_i_t[omega,i,t] + (1-z_omega_i_t_k_h[omega,i,t,k,L_i_k[i][k]]) * Upper_Compensation_i_k_h[i][k][h])
        @constraint(master_problem, LOWER_CONVEX_HULL_APPROXIMATION[omega in Omega, i in I, t in T, k in K_i_t[i][t], h in 1:L_i_k[i][k]], p_omega_i_t_k[omega,i,t,k] <= Lower_Beta_0_i_k_h[i][k][h] + Lower_Beta_u_i_k_h[i][k][h] * u_omega_i_t[omega,i,t] + Lower_Beta_s_i_k_h[i][k][h] * s_omega_i_t[omega,i,t] + (1-z_omega_i_t_k_h[omega,i,t,k,h]) * Lower_Compensation_i_k_h[i][k][h])

        @constraint(master_problem, LOWER_BOUND_ON_POWER_GENERATION[omega in Omega, i in I, t in T, k in K_i_t[i][t]], sum(z_omega_i_t_k_h[omega,i,t,k,h] * P_min_i_k_h[i][k][h] for h in 1:L_i_k[i][k]) <= p_omega_i_t_k[omega,i,t,k])
        @constraint(master_problem, UPPER_BOUND_ON_POWER_GENERATION[omega in Omega, i in I, t in T, k in K_i_t[i][t]], p_omega_i_t_k[omega,i,t,k] <= sum(z_omega_i_t_k_h[omega,i,t,k,h] * P_max_i_k_h[i][k][h] for h in 1:L_i_k[i][k]))
        @constraint(master_problem, LOWER_BOUND_ON_TURBINE_DISCHARGE[omega in Omega, i in I, t in T], sum(sum(z_omega_i_t_k_h[omega,i,t,k,h] * U_min_i_k_h[i][k][h] for h in 1:L_i_k[i][k]) for k in K_i_t[i][t]) <= u_omega_i_t[omega,i,t])
        @constraint(master_problem, UPPER_BOUND_ON_TURBINE_DISCHARGE[omega in Omega, i in I, t in T], u_omega_i_t[omega,i,t] <= sum(sum(z_omega_i_t_k_h[omega,i,t,k,h] * U_max_i_k_h[i][k][h] for h in 1:L_i_k[i][k]) for k in K_i_t[i][t]))
        @constraint(master_problem, CHOICE_OF_FUNCTION[omega in Omega, i in I, t in T, k in K_i_t[i][t]], sum(z_omega_i_t_k_h[omega,i,t,k,h] for h in 1:L_i_k[i][k]) == z_i_t_k[i,t,k])
    end;

    if PIECEWISE_LINEAR
        @variable(master_problem, h_omega_i_t_k_nu[omega in Omega, i in I, t in T, k in K_i_t[i][t], nu in 1:(N_u-1)], Bin);
        @variable(master_problem, 0 <= alpha_omega_i_t_k_nu[omega in Omega, i in I, t in T, k in K_i_t[i][t], nu in 1:N_u] <= 1);
        @variable(master_problem, beta_omega_i_t_k_ns[omega in Omega, i in I, t in T, k in K_i_t[i][t], ns in 1:(N_s-1)], Bin);
        @variable(master_problem, 0 <= gamma_omega_i_t_k_ns[omega in Omega, i in I, t in T, k in K_i_t[i][t], ns in 1:(N_s-1)] <= 1);

        @constraint(master_problem, ALPHA_CHOICE[omega in Omega, t in T, i in I, k in K_i_t[i][t], nu in 1:N_u], alpha_omega_i_t_k_nu[omega, i, t, k, nu] <= (nu < N_u ? h_omega_i_t_k_nu[omega, i, t, k, nu] : 0) + (nu > 1 ? h_omega_i_t_k_nu[omega, i, t, k, nu-1] : 0))
        @constraint(master_problem, ALPHA[omega in Omega, i in I, t in T, k in K_i_t[i][t]], sum(alpha_omega_i_t_k_nu[omega, i, t, k, nu] for nu in 1:N_u) == 1);
        @constraint(master_problem, PW_24[omega in Omega, t in T, i in I, k in K_i_t[i][t]], u_omega_i_t[omega, i, t] == sum(alpha_omega_i_t_k_nu[omega, i, t, k, nu] * Beta_u_i_k_nu[i][k][nu] for nu in 1:N_u))

        @constraint(master_problem, BETA[omega in Omega, i in I, t in T, k in K_i_t[i][t]], sum(beta_omega_i_t_k_ns[omega, i, t, k, ns] for ns in 1:(N_s-1)) == 1);
        @constraint(master_problem, PW_26[omega in Omega, t in T, i in I, k in K_i_t[i][t], ns in 1:(N_s-1)], gamma_omega_i_t_k_ns[omega, i, t, k, ns] <= beta_omega_i_t_k_ns[omega, i, t, k, ns])
        @constraint(master_problem, PW_27[omega in Omega, t in T, i in I, k in K_i_t[i][t]], s_omega_i_t[omega, i, t] == sum(beta_omega_i_t_k_ns[omega, i, t, k, ns] * Beta_s_i_k_ns[i][k][ns] + gamma_omega_i_t_k_ns[omega, i, t, k, ns] * (Beta_s_i_k_ns[i][k][ns+1] - Beta_s_i_k_ns[i][k][ns]) for ns in 1:(N_s-1)))

        @constraint(master_problem, PW_28_29[omega in Omega, t in T, i in I, k in K_i_t[i][t], ns in 1:(N_s-1), nu in 1:(N_u-1)], p_omega_i_t_k[omega,i,t,k] <= sum(alpha_omega_i_t_k_nu[omega, i, t, k, nuu] * Beta_p_i_k_ns_nu[i][k][ns][nuu] for nuu in 1:N_u) + gamma_omega_i_t_k_ns[omega, i, t, k, ns] * K_i_k_ns_nu[i][k][ns][nu] + P_max_i_k[i][k] * (2 - beta_omega_i_t_k_ns[omega, i, t, k, ns] - h_omega_i_t_k_nu[omega,i,t,k,nu]))
    end;
   
    if POLYNOME
        @NLconstraint(master_problem, Polynomial[omega in Omega, t in T, i in I, k in K_i_t[i][t]], p_omega_i_t_k[omega,i,t,k] <= Poly[i][k][1] + s_omega_i_t[omega,i,t]*Poly[i][k][2] + u_omega_i_t[omega,i,t]*Poly[i][k][3] + s_omega_i_t[omega,i,t]^2*Poly[i][k][4] + u_omega_i_t[omega,i,t]*s_omega_i_t[omega,i,t]*Poly[i][k][5] + u_omega_i_t[omega,i,t]^2*Poly[i][k][6] + s_omega_i_t[omega,i,t]^2*u_omega_i_t[omega,i,t]*Poly[i][k][7] + s_omega_i_t[omega,i,t]*u_omega_i_t[omega,i,t]^2*Poly[i][k][8] + u_omega_i_t[omega,i,t]^3*Poly[i][k][9] + s_omega_i_t[omega,i,t]^2*u_omega_i_t[omega,i,t]^2*Poly[i][k][10] + s_omega_i_t[omega,i,t]*u_omega_i_t[omega,i,t]^3*Poly[i][k][11] + u_omega_i_t[omega,i,t]^4*Poly[i][k][12] + s_omega_i_t[omega,i,t]^2*u_omega_i_t[omega,i,t]^3*Poly[i][k][13] + u_omega_i_t[omega,i,t]^4*s_omega_i_t[omega,i,t]*Poly[i][k][14] + u_omega_i_t[omega,i,t]^5*Poly[i][k][15]);
    end;
   
    ##############################################
    #       FIXED SCHEDULE BY OTHER METHOD       #
    ##############################################

    if SCHEDULE_PATH != "none"

        original_results = JSON.Parser.parse(open(SCHEDULE_PATH, "r"));

        # Use 'JuMP.fix' to get the local optimum associated to the method's schedule and 'setvalue' if you just want to help the new method with starting values

        for m in M
            for t in T_m[m]
                JuMP.fix(y_m_t[m,t], original_results["y_m_t"][m][t])
            end
        end;

        for i in I
            for t in T
                JuMP.fix(r_i_t[i,t], original_results["r_i_t"][i][t])
            end
        end;

        for i in I
            for t in T
                for k in K_i_t[i][t]
                    JuMP.fix(z_i_t_k[i,t,k], original_results["z_i_t"][i][t] == k ? 1 : 0)
                end
            end
        end;
    end;

######################################################################
#                         SOLVING THE MODEL                          #
######################################################################

            ##############################################
            #             CALLING THE SOLVER             #
            ##############################################
            
    optimize!(master_problem);
   
    end_time = now()
    elapsed_time = (end_time - start_time)
    status = termination_status(master_problem)

    if status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED
        obj_value = objective_value(master_problem)
        println("Optimal objective value: ", obj_value)
    else
        println("No solution found or the problem is not optimally solved.")
    end

            ##############################################
            #          STATISTICS ON THE SOLVER          #
            ##############################################

    println("\t- Final status: ", status);
    println("\t- Time to solve the model: ", Dates.value(elapsed_time) , " seconds");
    println("\t- Objective value: ", JuMP.objective_value(master_problem));

######################################################################
#                         SAVING THE RESULTS                         #
######################################################################

            ##############################################
            #           FORMATTING THE VALUES            #
            ##############################################

    treated_y_m_t = Dict{String, Array{Float64, 1}}(m => Array{Float64, 1}([sum(((tt >= t - D_m[m] + 1 && tt <= t) ? value(y_m_t[m,tt]) : 0) for tt in T_m[m]) + (t-D_m[m] in T_m[m] ? Alpha_1 * value(y_m_t[m,t-D_m[m]]) : 0) + (t-D_m[m]-1 in T_m[m] ? Alpha_2 * value(y_m_t[m,t-D_m[m]-1]) : 0) for t in sort([t for t in T])]) for m in M);
    raw_y_m_t = Dict{String, Array{Float64, 1}}(m => Array{Float64, 1}([(t in T_m[m] ? value(y_m_t[m, t]) : 0) for t in sort([t for t in T])]) for m in M);
    raw_r_i_t = Dict{String, Array{Float64, 1}}(i => Array{Float64, 1}([value(r_i_t[i, t]) for t in sort([t for t in T])]) for i in I);
    raw_z_i_t = Dict{String, Array{Int64, 1}}(i => Array{Int64, 1}([sum(round(value(z_i_t_k[i, t, k])) * k for k in K_i_t[i][t]) for t in sort([t for t in T])]) for i in I);
    raw_s_omega_i_t = Dict{String, Dict{String, Array{Float64, 1}}}(omega => Dict{String, Array{Float64, 1}}(i => Array{Float64, 1}([value(s_omega_i_t[omega, i, t]) for t in sort([t for t in T])]) for i in I) for omega in Omega);
    raw_v_omega_i_t = Dict{String, Dict{String, Array{Float64, 1}}}(omega => Dict{String, Array{Float64, 1}}(i => Array{Float64, 1}([value(v_omega_i_t[omega, i, t]) for t in sort([t for t in T])]) for i in I) for omega in Omega);
    raw_u_omega_i_t = Dict{String, Dict{String, Array{Float64, 1}}}(omega => Dict{String, Array{Float64, 1}}(i => Array{Float64, 1}([value(u_omega_i_t[omega, i, t]) for t in sort([t for t in T])]) for i in I) for omega in Omega);
    raw_p_omega_i_t = Dict{String, Dict{String, Array{Float64, 1}}}(omega => Dict{String, Array{Float64, 1}}(i => Array{Float64, 1}([value(p_omega_i_t[omega, i, t]) for t in sort([t for t in T])]) for i in I) for omega in Omega);
    raw_c_m = Dict{String, Float64}(m => value(c_m[m]) for m in M);

            ##############################################
            #           FOLDER OF THE RESULTS            #
            ##############################################

    mkpath(string("resultatScip/", SEASON, "/", NB_SCENARIOS, "_scenarios/", STOCHASTICITY, "_stochasticity/"));

            ##############################################
            #             SAVING THE RESULTS             #
            ##############################################

    open(string("resultat/", SEASON, "/", NB_SCENARIOS, "_scenarios/", STOCHASTICITY, "_stochasticity/", APPROXIMATION_METHOD,"_5T.json"), "w") do all_results_file
        JSON.write(all_results_file, JSON.json(Dict{String, Any}("scenarios" => NB_SCENARIOS,
                                                                 "status" => status,
                                                                 "time" => Dates.value(elapsed_time),
                                                                 "objective_value" => JuMP.objective_value(master_problem),
                                                                 #"objective_bound" => getobjectivebound(master_problem),
                                                                 "cumulated_y_m_t" => treated_y_m_t,
                                                                 "y_m_t" => raw_y_m_t,
                                                                 "c_m" => raw_c_m,
                                                                 "r_i_t" => raw_r_i_t,
                                                                 "z_i_t" => raw_z_i_t,
                                                                 "s_omega_i_t" => raw_s_omega_i_t,
                                                                 "v_omega_i_t" => raw_v_omega_i_t,
                                                                 "u_omega_i_t" => raw_u_omega_i_t,
                                                                 "p_omega_i_t" => raw_p_omega_i_t),
                                               4))
    end;
end;

solve_model("piecewise_convex_hull", "winter",2, "no")
println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
println("~               HOW TO USE               ~")
println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
println("~         Function: solve_model          ~")
println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
println("~ - Arg1: APPROXIMATION_METHOD           ~")
println("~ - Arg2: SEASON                         ~")
println("~ - Arg3: NB_SCENARIOS                   ~")
println("~ - Arg4: STOCHASTICITY                  ~")
println("~ - Arg5: SCHEDULE_PATH (OPTIONAL)       ~")
println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
>>>>>>> 0be90e0 (Initial commit)
