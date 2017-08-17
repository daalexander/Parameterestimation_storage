import pylab as pl
import pandas as pd

import casadi as ca
import casiopeia as cp


# Constants

cp_water = 4182.0 
layer = 5.0
Tamb = 20.0

# States

x = ca.MX.sym("x", 5)

TSC1 = x[0]
TSC0  = x[1]
TSC1_1  = x[2]
TSC1_2  = x[3]
TSC1_3  = x[4]


# Parameters

p = ca.MX.sym("p", 5)
alpha_1 = p[0]
alpha_0 = p[1]
alpha_1_1 = p[2]
alpha_1_2 = p[3]
alpha_1_3 = p[4]


alpha_1_init = 1.0
alpha_0_init = 1.0
alpha_1_1_init = 1.0
alpha_1_2_init = 1.0
alpha_1_3_init = 1.0

pinit = ca.vertcat([alpha_1_init, alpha_0_init, alpha_1_1_init, alpha_1_2_init, alpha_1_3_init]) 


# Controls

u = ca.MX.sym("u", 6)

A_IN_2 = u[0]
msto = u[1] 
m1minus = u[2]
m1plus = u[3]
TCHEO_1 = u[4]
TCO_1 = u[5]



m = 1000.0 / layer

# Massflows storage

#=================================================================================================================================================
#first Layer
dotT1 = 1.0/m * ( -A_IN_2 * TSC1 + msto * TCO_1 - m1minus * TSC1 + m1plus * TSC1_1 - (alpha_1 * (TSC1 - Tamb)) / cp_water)
#m0minus = (m0plus + V_PSOS - msto) 

#second layer
dotT0 = 1.0/m * ( A_IN_2 * TCHEO_1 - msto * TSC0 - m1plus * TSC0 + m1minus * TSC1_1 - (alpha_0 * (TSC0 - Tamb)) / cp_water) 

#
dotT1_1 = 1.0/m * ( m1minus * (TSC1 -  TSC1_1) + m1plus* (TSC1_2 - TSC1_1) - (alpha_1_1 * (TSC1_1 - Tamb)) / cp_water)

#
dotT1_2 = 1.0/m * ( m1minus * (TSC1_1 -  TSC1_2) + m1plus* (TSC1_3 - TSC1_2) - (alpha_1_2 * (TSC1_2 - Tamb)) / cp_water)

#
dotT1_3 = 1.0/m * ( m1minus * (TSC1_2 -  TSC1_3) + m1plus* (TSC0 - TSC1_3) - (alpha_1_3 * (TSC1_3 - Tamb)) / cp_water)
#=================================================================================================================================================


#ODE

f = ca.vertcat([ \
    dotT1, \
    dotT0, \
    dotT1_1, \
    dotT1_2, \
    dotT1_3])

phi = x

system = cp.system.System(x = x, u = u, f = f, phi = phi, p = p)

pe_setups = []

# Start heating

datatable = "data2017-05-23"
int_start = [0, 5000, 10000, 15000, 20000, 25000]
int_end = [4999, 9999, 14999, 19999, 24999, 29999]
int_step = 5

data = pd.read_table("data/"+ datatable + ".csv", \
    delimiter=",", index_col=0)

for k,e in enumerate(int_start):

    time_points = data["time"].values[e:int_end[k]:int_step]

    udata_0 = data["A_IN_2"][:-1].values[e:int_end[k]:int_step]

    udata_1 = data["msto"][:-1].values[e:int_end[k]:int_step]

    udata_2 = data["m1minus"][:-1].values[e:int_end[k]:int_step]
    
    udata_3 = data["m1plus"][:-1].values[e:int_end[k]:int_step]
    
    udata_4 = data["TCHEO_1"][:-1].values[e:int_end[k]:int_step]

    udata_5 = data["TCO_1"][:-1].values[e:int_end[k]:int_step]


    udata = ca.horzcat([udata_0, udata_1, udata_2, udata_3, udata_4, udata_5])[:-1,:]



    x0_init = data["TSC1"].values[e:int_end[k]:int_step]
    x1_init = data["TSC0"].values[e:int_end[k]:int_step]
    x2_init = data["TSC1_1"].values[e:int_end[k]:int_step]
    x3_init = data["TSC1_2"].values[e:int_end[k]:int_step] 
    x4_init = data["TSC1_3"].values[e:int_end[k]:int_step]
    

    xinit = ca.horzcat([pl.atleast_2d(x0_init).T, pl.atleast_2d(x1_init).T, pl.atleast_2d(x2_init).T, pl.atleast_2d(x3_init).T, \
        pl.atleast_2d(x4_init).T,]) 

    ydata_0 = data["TSC1"].values[e:int_end[k]:int_step]
    ydata_1 = data["TSC0"].values[e:int_end[k]:int_step]
    ydata_2 = data["TSC1_1"].values[e:int_end[k]:int_step]
    ydata_3 = data["TSC1_2"].values[e:int_end[k]:int_step]
    ydata_4 = data["TSC1_3"].values[e:int_end[k]:int_step]


    ydata = ca.horzcat([pl.atleast_2d(ydata_0).T, pl.atleast_2d(ydata_1).T, pl.atleast_2d(ydata_2).T, pl.atleast_2d(ydata_3).T,\
        pl.atleast_2d(ydata_4).T,]) 

   

    pe_setups.append(cp.pe.LSq(system = system, time_points = time_points, \
        udata = udata, \
        pinit = pinit, \
        ydata = ydata, \
        xinit = xinit)) #, \
        # wv = wv))

##einen Zeitraum
# pe_setups[0].run_parameter_estimation()#{"linear_solver": "ma57"})

##fuer multiparameter
mpe = cp.pe.MultiLSq(pe_setups)
mpe.run_parameter_estimation({"linear_solver": "ma57"})
# mpe.run_parameter_estimation()

# sim_est = cp.sim.Simulation(system = system, pdata = est_parameter)
sim_est = cp.sim.Simulation(system = system, pdata = mpe.estimated_parameters)
# sim_est.run_system_simulation(time_points = time_points, \
#     x0 = xinit[0,:], udata = udata)

pl.close("all")




print("alpha_1 = "+ str(mpe.estimated_parameters[0]))
print("alpha_0 = "+ str(mpe.estimated_parameters[1]))
print("alpha_1_1 = "+ str(mpe.estimated_parameters[2]))
print("alpha_1_2 = "+ str(mpe.estimated_parameters[3]))
print("alpha_1_3 = "+ str(mpe.estimated_parameters[4]))
