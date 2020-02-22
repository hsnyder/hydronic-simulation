#!/usr/bin/python3
"""
	Copyright (C) 2019, 2020 Harris M. Snyder

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.
	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

"""
This file presents a control model for the hydronic home heating simulation
provided in hydronic.py.

The boiler's outlet water temperature is on PID control (burner power is throttled to aim for a specific outlet water temperature).
The room temperature is controlled with a simple on/off thermostat.

Run this file and tweak parameters to find sensible PID parameters.
"""


import numpy as np
import matplotlib.pyplot as plt
import hydronic as hy
import PID
from functools import reduce

# Construct our simulation
sim = hy.HydronicSimulation(tmax=2000.0)
sim.chMassFlow(0.0)
sim.chBurnerPower(0.0)

#initialized from the steady state sim at 1200 s from arbitrary initial conditions.
#sim.T0 = np.array([ 343.14856376,  293.62723117,  341.65277202,  340.20906104,  338.81495114, 338.92052896,  337.54630079,  336.21938953])

# Construct our PID controller for boiler outlet water temperature, and assign it's setpoint
PI_Tb = PID.PIDController(Kp=0.8, Ti=10.0, Td=0.0, bias=0.3, outMax=1.0, outMin=0.0 )
PI_Tb.SP = hy.CtoK(70.0)

# Intended setpoint for our room thermostat
statSP = hy.CtoK(21.0)


# Control callback for hydronic.py
def Control(t,T):
    Tb, Tz, Tw1, Tw2, Tw3, Tt1, Tt2, Tt3 = T
    
    # Basic thermostat model is implemented right here
    # The stat turns off the hydronic pump if the temperature is more than 1 degree above the setpoint,
    # and turns it back on if the temperature is more than 1 degree below the setpoint.
    if statSP - Tz  > 1.0:
        sim.chMassFlow(1.0)
    elif statSP - Tz < -1.0:
        sim.chMassFlow(0)
    
    # scale boiler burner power according to the PID controller for boiler outlet water temperature.
    Tb_output = PI_Tb.next(Tb, sim.dt) 
    sim.chBurnerPower(Tb_output)
    
    if t == sim.tmax -1.0:
        print("Final T:", T)



# Ploting related variables
basevec = np.zeros(13)
grid = np.arange(0.0, sim.tmax, sim.dt)    
data = np.tile(basevec, (grid.shape[0],1))
i = 0 

# Data storage callback for hydronic.py
def MyStore(t,T):
    global i, data, grid
    print("st t =", t, "s")
    myT = hy.KtoC(T)
    data[i] = np.append(myT, [sim.U1, sim.U2, hy.KtoC(sim.Ta), hy.KtoC(PI_Tb.SP), hy.KtoC(statSP)])
    grid[i] = t
    i += 1


sim.RunSimulation(store=MyStore, control=Control)


# Plot results
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2)#, sharex=True, sharey=True)
fig.tight_layout()

#ax1.set_xlim((tPlotGrid[0]-updateInterval*dt, tPlotGrid[-1]+updateInterval*dt))
#ax1.set_ylim(50, 100)
#ax = plt.axes(xlim=(tPlotGrid[0]-updateInterval*dt, tPlotGrid[-1]+updateInterval*dt))

ax1.set_xlabel('t (s)')
ax2.set_xlabel('t (s)')
ax3.set_xlabel('t (s)')
ax4.set_xlabel('t (s)')
ax5.set_xlabel('t (s)')
ax6.set_xlabel('t (s)')

ax1.set_ylabel('Boiler Temp (C)')
ax1.plot(grid, data[:,0], '-', label='Value')
ax1.plot(grid, data[:,11], ':', label='SP')
ax1.legend(loc='lower center')

ax3.set_ylabel('Zone Temp (C)')
ax3.plot(grid, data[:,1], '-', label='Value')
ax3.plot(grid, data[:,12], ':', label='SP')
ax3.legend(loc='lower center')

ax2.set_ylabel('Water Temp (C)')
ax2.plot(grid, data[:,2], '-', label='Tw1')
ax2.plot(grid, data[:,3], '--', label='Tw2')
ax2.plot(grid, data[:,4], ':', label='Tw3')
ax2.legend()

ax4.set_ylabel('Tube Temp (C)')
ax4.plot(grid, data[:,5], '-', label='Tt1')
ax4.plot(grid, data[:,6], '--', label='Tt2')
ax4.plot(grid, data[:,7], ':', label='Tt3')
ax4.legend()

ax5.set_ylabel('Fraction of capacity')
ax5.plot(grid, data[:,8], label='Pump')
ax5.plot(grid, data[:,9], label='Burner')
ax5.legend()

ax6.set_ylabel('Outdoor Temp (C)')
ax6.plot(grid, data[:,10])

plt.show()
