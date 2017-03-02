import pylab as pl
import pandas as pd

import casadi as ca
import casiopeia as cp


# Constants

cp_water = 4182.0
layer = 4
Tamb = 20.0

# States

x = ca.MX.sym("x", 4)

TSH0 = x[0]
TSH2  = x[1]
TSH3  = x[2]
TSH1  = x[3]

# Parameters

p = ca.MX.sym("p", 4)
alpha_0 = p[0]
alpha_2 = p[1]
alpha_3 = p[2]
alpha_1 = p[3]

alpha_0_init = 1.0
alpha_2_init = 1.0
alpha_3_init = 1.0
alpha_1_init = 1.0
pinit = ca.vertcat([alpha_0_init, alpha_2_init, alpha_3_init, alpha_1_init]) 


# Controls

u = ca.MX.sym("u", 14)

V_PSOS = u[0]
msto = u[1] 
m0minus = u[2]
m0plus = u[3]
m2minus = u[4]
m2plus = u[5]
m3minus = u[6]
m3plus = u[7]
TSOS = u[8]
TCO_1 = u[9]
VSHP_OP = u[10]
VSHP_CL = u[11]
VSHS_OP = u[12]
VSHS_CL = u[13]




m = 2000.0 / layer

##Massflows storage
#first Layer
dotT0 = 1.0/m * (V_PSOS * TSOS - msto * TSH0 - (m0plus + V_PSOS - msto) * TSH0 + m0plus * TSH2 - (alpha_0 * (TSH0 - Tamb)) / cp_water) 
#m0minus = m0plus + V_PSOS - msto 

#second Layer
dotT2 = 1.0/m * ( -V_PSOS * VSHP_OP * TSH2 + msto * VSHS_OP * TCO_1 + (m0plus + V_PSOS - msto) * TSH0 - m0plus * TSH2  \
    - (-V_PSOS * VSHP_OP + V_PSOS - msto + msto * VSHS_OP + m2plus) * TSH2 + m2plus * TSH3 - (alpha_2 * (TSH2 - Tamb)) / cp_water)
#m2minus = -V_PSOS*VSHP_OP +V_PSOS -msto +msto*VSHS_OP +m2plus

#third Layer
dotT3 = 1.0/m * ((-V_PSOS * VSHP_OP + V_PSOS - msto + msto*VSHS_OP + m2plus) * TSH2 - m2plus * TSH3 \
    - (-V_PSOS * VSHP_OP + V_PSOS - msto + msto * VSHS_OP  + m3plus) * TSH3 + m3plus * TSH1 - (alpha_3 * (TSH3 - Tamb)) / cp_water)
#m3minus = -V_PSOS*VSHP_OP +V_PSOS -msto +msto*VSHS_OP  +m3plus

#fourth Layer
dotT1 = 1.0/m * (-V_PSOS * VSHP_CL * TSH1 + (-V_PSOS * VSHP_OP + V_PSOS - msto + msto * VSHS_OP  + m3plus) * TSH3 \
    - m3plus * TSH1 + msto * VSHS_CL * TCO_1 - (alpha_1 * (TSH1 - Tamb)) / cp_water)
#=================================================================================================================================================


#ODE

f = ca.vertcat([ \
    dotT0, \
    dotT2, \
    dotT3, \
    dotT1])

phi = x

system = cp.system.System(x = x, u = u, f = f, phi = phi, p = p)

pe_setups = []

# Start heating

datatable = "data2017-02-23"
int_start = [0, 5000, 10000, 15000, 20000, 25000,30000, 60000, 65000, 70000, 75000, 80000]
int_end = [4999, 9999, 14999, 19999, 24999, 29999, 35000, 64999, 69999, 74999, 79999, 84999]
int_step = 5

data = pd.read_table("data_storage/"+ datatable + ".csv", \
    delimiter=",", index_col=0)

for k,e in enumerate(int_start):

    time_points = data["time"].values[e:int_end[k]:int_step]

    udata_0 = data["V_PSOS"][:-1].values[e:int_end[k]:int_step]

    udata_1 = data["msto"][:-1].values[e:int_end[k]:int_step]

    udata_2 = data["m0minus"][:-1].values[e:int_end[k]:int_step]
    
    udata_3 = data["m0plus"][:-1].values[e:int_end[k]:int_step]
    
    udata_4 = data["m2minus"][:-1].values[e:int_end[k]:int_step]

    udata_5 = data["m2plus"][:-1].values[e:int_end[k]:int_step]

    udata_6 = data["m3minus"][:-1].values[e:int_end[k]:int_step]

    udata_7 = data["m3plus"][:-1].values[e:int_end[k]:int_step]

    udata_8 = data["TSOS"][:-1].values[e:int_end[k]:int_step]

    udata_9 = data["TCO_1"][:-1].values[e:int_end[k]:int_step]

    udata_10 = data["VSHP_OP"][:-1].values[e:int_end[k]:int_step]

    udata_11 = data["VSHP_CL"][:-1].values[e:int_end[k]:int_step]

    udata_12 = data["VSHS_OP"][:-1].values[e:int_end[k]:int_step]

    udata_13 = data["VSHS_CL"][:-1].values[e:int_end[k]:int_step]

    udata = ca.horzcat([udata_0, udata_1, udata_2, udata_3, udata_4, udata_5, udata_6, udata_7, udata_8, udata_9, \
    udata_10, udata_11, udata_12, udata_13])[:-1,:]



    x0_init = data["TSH0"].values[e:int_end[k]:int_step]
    x1_init = data["TSH2"].values[e:int_end[k]:int_step]
    x2_init = data["TSH3"].values[e:int_end[k]:int_step]
    x3_init = data["TSH1"].values[e:int_end[k]:int_step] 

    xinit = ca.horzcat([pl.atleast_2d(x0_init).T, pl.atleast_2d(x1_init).T, pl.atleast_2d(x2_init).T, pl.atleast_2d(x3_init).T,]) 

    ydata_0 = data["TSH0"].values[e:int_end[k]:int_step]
    ydata_1 = data["TSH2"].values[e:int_end[k]:int_step]
    ydata_2 = data["TSH3"].values[e:int_end[k]:int_step]
    ydata_3 = data["TSH1"].values[e:int_end[k]:int_step]

    ydata = ca.horzcat([pl.atleast_2d(ydata_0).T, pl.atleast_2d(ydata_1).T, pl.atleast_2d(ydata_2).T, pl.atleast_2d(ydata_3).T,]) #ca.repmat(y1_5_init, (1, ydata.shape[0])).T])

    # wv = pl.ones(ydata.shape[0])
    # wv[:int(ydata.shape[0]*0.1)] = 5

    pe_setups.append(cp.pe.LSq(system = system, time_points = time_points, \
        udata = udata, \
        pinit = pinit, \
        ydata = ydata, \
        xinit = xinit)) #, \
        # wv = wv))

##fuer einen Zeitraum
# pe_setups[0].run_parameter_estimation()#{"linear_solver": "ma57"})

##fuer multiparameter
mpe = cp.pe.MultiLSq(pe_setups)
# # mpe.run_parameter_estimation({"linear_solver": "ma57"})
mpe.run_parameter_estimation()

# sim_est = cp.sim.Simulation(system = system, pdata = est_parameter)
sim_est = cp.sim.Simulation(system = system, pdata = mpe.estimated_parameters)
# sim_est.run_system_simulation(time_points = time_points, \
#     x0 = xinit[0,:], udata = udata)

pl.close("all")



# print("alpha_0 = "+ str(pe_setups[0].estimated_parameters[0]))
# print("alpha_2 = "+ str(pe_setups[0].estimated_parameters[1]))
# print("alpha_3 = "+ str(pe_setups[0].estimated_parameters[2]))
# print("alpha_1 = "+ str(pe_setups[0].estimated_parameters[3]))

print("alpha_0 = "+ str(mpe.estimated_parameters[0]))
print("alpha_2 = "+ str(mpe.estimated_parameters[1]))
print("alpha_3 = "+ str(mpe.estimated_parameters[2]))
print("alpha_1 = "+ str(mpe.estimated_parameters[3]))

