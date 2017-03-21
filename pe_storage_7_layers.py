import pylab as pl
import pandas as pd

import casadi as ca
import casiopeia as cp


# Constants

cp_water = 4182.0 ##   evtl. 4200
layer = 7
Tamb = 20.0

# States

x = ca.MX.sym("x", 7)

TSH0 = x[0]
TSH2  = x[1]
TSH3  = x[2]
TSH1  = x[3]
TSH0_5  = x[4]
TSH2_5  = x[5]
TSH3_5  = x[6]

# Parameters

p = ca.MX.sym("p", 7)
alpha_0 = p[0]
alpha_2 = p[1]
alpha_3 = p[2]
alpha_1 = p[3]
alpha_0_5 = p[4]
alpha_2_5 = p[5]
alpha_3_5 = p[6]

alpha_0_init = 1.0
alpha_2_init = 1.0
alpha_3_init = 1.0
alpha_1_init = 1.0
alpha_0_5_init = 1.0
alpha_2_5_init = 1.0
alpha_3_5_init = 1.0
pinit = ca.vertcat([alpha_0_init, alpha_2_init, alpha_3_init, alpha_1_init, alpha_0_5_init, alpha_2_5_init, alpha_3_5_init]) 


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

# Massflows storage

#=================================================================================================================================================
#first Layer
dotT0 = 1.0/m * (V_PSOS * TSOS - msto * TSH0 - (m0plus + V_PSOS - msto) * TSH0 + m0plus * TSH0_5 - (alpha_0 * (TSH0 - Tamb)) / cp_water) + alpha_iso
#m0minus = (m0plus + V_PSOS - msto) 

#layer1.5
dotT0_5 = 1.0/m * ((m0plus + V_PSOS - msto) * TSH0 - m0plus * TSH0_5 - (m0plus + V_PSOS - msto) * TSH0_5 + m0plus * TSH2 \
    - (alpha_0_5 * (TSH0_5 - Tamb)) / cp_water)

#second Layer
dotT2 = 1.0/m * ( -V_PSOS * VSHP_OP * TSH2 + msto * VSHS_OP * TCO_1 + (m0plus + V_PSOS - msto) * TSH0_5 - m0plus * TSH2  \
    - (-V_PSOS * VSHP_OP + V_PSOS - msto + msto * VSHS_OP + m2plus) * TSH2 + m2plus * TSH2_5 - (alpha_2 * (TSH2 - Tamb)) / cp_water)
#m2minus = (-V_PSOS*VSHP_OP +V_PSOS -msto +msto*VSHS_OP +m2plus)

#layer2.5
dotT2_5 = 1.0/m * ((-V_PSOS*VSHP_OP +V_PSOS -msto +msto*VSHS_OP +m2plus) * TSH2 - m2plus * TSH2_5 - \
    (-V_PSOS*VSHP_OP +V_PSOS -msto +msto*VSHS_OP +m2plus) * TSH2_5 + m2plus * TSH3 - alpha_2_5 * (TSH2_5 - Tamb) / cp_water)

#third Layer
dotT3 = 1.0/m * ((-V_PSOS * VSHP_OP + V_PSOS - msto + msto*VSHS_OP + m2plus) * TSH2_5 - m2plus * TSH3 \
    - (-V_PSOS * VSHP_OP + V_PSOS - msto + msto * VSHS_OP  + m2plus) * TSH3 + m2plus * TSH3_5 - (alpha_3 * (TSH3 - Tamb)) / cp_water)
#m3minus = (-V_PSOS*VSHP_OP +V_PSOS -msto +msto*VSHS_OP  +m3plus)## m3minus ist m2minus und m3plus ist m2plus

#leyer3.5
dotT3_5 = 1.0/m * ((-V_PSOS*VSHP_OP +V_PSOS -msto +msto*VSHS_OP  +m2plus) * TSH3 - m2plus * TSH3_5 \
    - (-V_PSOS*VSHP_OP +V_PSOS -msto +msto*VSHS_OP  +m2plus) * TSH3_5 + m2plus * TSH1 - (alpha_3_5 * (TSH3_5 - Tamb)) / cp_water)

#fourth Layer
dotT1 = 1.0/m * (-V_PSOS * VSHP_CL * TSH1 + (-V_PSOS * VSHP_OP + V_PSOS - msto + msto * VSHS_OP  + m2plus) * TSH3_5 \
    - m2plus * TSH1 + msto * VSHS_CL * TCO_1 - (alpha_1 * (TSH1 - Tamb)) / cp_water)
#=================================================================================================================================================


#ODE

f = ca.vertcat([ \
    dotT0, \
    dotT2, \
    dotT3, \
    dotT1, \
    dotT0_5,\
    dotT2_5,\
    dotT3_5])

phi = x

system = cp.system.System(x = x, u = u, f = f, phi = phi, p = p)

pe_setups = []

# Start heating

datatable = "data2017-01-19"
int_start = [0, 5000, 10000, 15000, 20000, 25000]#, 30000, 35000]# #[45000,50000,55000,60000, 65000, 70000, 75000, 80000]##[0, 400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000, 4400, 4800, 5200, 5600, 6000]
int_end = [4999, 9999, 14999, 19999, 24999, 29999]#, 34999, 39999] # #[49999, 54999, 59999, 64999, 69999, 74999, 79999, 86000] ##[399, 799, 1199, 1599, 1999, 2399, 2700, 3199, 3599, 3999, 4399, 4799, 5199, 5599, 5999, 6399]
int_step = 5

data = pd.read_table("data_storage/7_layer/"+ datatable + ".csv", \
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
    x4_init = data["TSH0_5"].values[e:int_end[k]:int_step]
    x5_init = data["TSH2_5"].values[e:int_end[k]:int_step]
    x6_init = data["TSH3_5"].values[e:int_end[k]:int_step] 

    xinit = ca.horzcat([pl.atleast_2d(x0_init).T, pl.atleast_2d(x1_init).T, pl.atleast_2d(x2_init).T, pl.atleast_2d(x3_init).T, \
        pl.atleast_2d(x4_init).T, pl.atleast_2d(x5_init).T, pl.atleast_2d(x6_init).T,]) 

    ydata_0 = data["TSH0"].values[e:int_end[k]:int_step]
    ydata_1 = data["TSH2"].values[e:int_end[k]:int_step]
    ydata_2 = data["TSH3"].values[e:int_end[k]:int_step]
    ydata_3 = data["TSH1"].values[e:int_end[k]:int_step]
    ydata_4 = data["TSH0_5"].values[e:int_end[k]:int_step]
    ydata_5 = data["TSH2_5"].values[e:int_end[k]:int_step]
    ydata_6 = data["TSH3_5"].values[e:int_end[k]:int_step] 


    ydata = ca.horzcat([pl.atleast_2d(ydata_0).T, pl.atleast_2d(ydata_1).T, pl.atleast_2d(ydata_2).T, pl.atleast_2d(ydata_3).T,\
        pl.atleast_2d(ydata_4).T, pl.atleast_2d(ydata_5).T, pl.atleast_2d(ydata_6).T,]) 

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
mpe.run_parameter_estimation({"linear_solver": "ma57"})
# mpe.run_parameter_estimation()

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
print("alpha_0.5 = "+ str(mpe.estimated_parameters[4]))
print("alpha_2.5 = "+ str(mpe.estimated_parameters[5]))
print("alpha_3.5 = "+ str(mpe.estimated_parameters[6]))
