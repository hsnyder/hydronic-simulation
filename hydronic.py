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
This file presents a simulation of a hydronic home heating loop consisting of a boiler and one heat zone (room)
with one baseboard radiator, discretized into 3 points along the radiator's length.

This file on it's own is not very useful, since it has no thermostat model.

Based on model presented in Zaheer-Uddin & Monastriakos (1998)
"""

import numpy as np
import matplotlib.pyplot as plt
import Sim1D
from functools import reduce


# To/from Kelvins from degrees Celcius
def KtoC(T):
    return T - 273.15

def CtoK(T):
    return T + 273.15



"""
This class uses the tools in Sim1D.py to set up an actual simulation of a (hydronic) home heating loop.
See Zaheer-Uddin & Monastriakos (1998) for model details.
""" 

class HydronicSimulation:
    def __init__(self, tmax=4000.0, dt=1.0):
        
        self.broyden = Sim1D.Broyden()
        
        # Simulation control settings
        self.tmax = tmax         # run simulation until (seconds)
        self.dt   = dt           # time step size (seconds)
        
        
        # Constants from paper
        self.ab  =   33.33    # heat loss coefficient of boiler exterior surface (W / K)
        self.d   =    0.0222  # baseboard outside tube diameter (m)
        self.di  =    0.01994 # baseboard inside tube diameter (m)
        self.cpw = 4180.0     # specific heat capacity of water (J / kg.K)
        self.ct  =  384.0     # specific heat capacity of tube material (J / kg.K)
        self.cf  =  896.0     # specific heat capacity of fin material (J / kg.K)
        self.lr  =    6.0     # length of baseboard radiator (m)
        self.kf  =  236.0     # conductivity of fin (W / m.K)
        self.kw  =    0.658   # conductivity of water (W / m.K)
        self.yf  =    0.00156 # thickness of fin (m)
        self.Te  = CtoK(20.0) # boiler room temp (C)
        
        self.alpha   =     0.12 # flue loss coefficient (dimensionless)
        self.Tbmax   =    90.0  # boiler maximum temperature (C)
        self.Tinf    =    18.0  # city water temperature (C)
        self.rho_fin =  2787.0  # density of fin material [?] (kg / m^3)
        self.U1max   =     0.3  # maximum water mass flow rate (kg/s)
        self.U2max   = 25000.0  # maximum  boiler power (W)
        
        self.U1  =  0.5       # dimensionless mass flow rate
        self.U2  =  0.25      # dimensionless input power
        self.Ta  = CtoK(-5.0) # outside temp
        
        # Constants not from paper (my numbers)
        self.spf =    0.005   # radiator fin spacing (m)
        self.lf  =    0.08    # radiator fin height (m)
        self.rho_w = 980.0 # density of water (kg/m^3), number for ~65 C picked, in reality not constant with temperature.
        self.rho_tube = 8960.0 # density of tube material (kg / m^3)
        self.zone_area = 60.0 # floor area, square meters
        self.zone_airCuM_per_floorSqM = 2.5
        self.zone_heatloss_psqm = 2.5 # heat loss (W/K) per square meter of floor space
        
        # Derived constants
        self.Afin = (4.0*self.yf*self.lf + 2.0*self.lf**2.0 - 2*np.pi* (self.d/2.0)**2.0) / self.spf # area of fin per unit length (m^2 /  m)
        self.At   = np.pi*self.d*(self.spf - self.yf) / self.spf    # area of bare tube per unit length (m^2 / m)
        self.Ao   = self.Afin + self.At                   # total rad heat exchange area per unit length (m^2 / m)
        self.Ait  = np.pi*self.di                    # tube internal heat exchange area per unit length (m^2 / m)
        self.Aci  = np.pi*(self.di/2.0)**2.0         # cross sectional area of tube (m)
        self.az = self.zone_area * self.zone_heatloss_psqm # heated zone heat loss coefficient - from heated zone to outside (W / K)
        
        self.Cb  =  1128748.0   # boiler heat capacity (J / K)
        self.Cz  =  self.zone_area * self.zone_airCuM_per_floorSqM * 1220.0     # heated zone heat capacity (J / K)


        # Our grid for radiator spatial discretization
        g = Sim1D.Grid(0, self.lr, num=3)
        
        
        self.Tb = Sim1D.Scalar(None, 342.7) # boiler outlet temperature
        self.Tz = Sim1D.Scalar(None, 293.9) # zone (i.e. room) temperature
        self.Tw = Sim1D.Scalar(g, 336.0, boundaryStart=self.Tb) # water temperature (in radiator at this grid point)
        self.Tt = Sim1D.Scalar(g, 333.0) # radiator tube temperature (at this grid point)
        
        # Initial conditions
        self.T0 = Sim1D.InitState( self.Tb, self.Tz, self.Tw, self.Tt )


    # Returns a time-based grid for plotting data values over time.
    def GetGrid(self):
        tPlotGrid = np.arange(0.0, self.tmax, self.dt)    
        return tPlotGrid

    # Calculates approximate viscosity of water, given temperature.
    def WaterViscosity (self, T):
        #Vogel's equation, from http://ddbonline.ddbst.de/VogelCalculation/VogelCalculationCGI.exe?component=Water
        A = -3.7188
        B = 578.919
        C = -137.546
        return 1e-3 * np.exp(A + B / (C + T)) # in Pa.s
    
    # Calculates renyolds number for water, at given flow sped and temperature
    def WaterReynolds(self, u, T):
        mu = self.WaterViscosity(T)
        L = self.di # characteristic length is inside diameter of radiator tube
        return self.rho_w * u * L / mu

    # Calculates prandtl number for water, at given temperature.
    def WaterPrandtl(self, T):
        mu = self.WaterViscosity(T)
        return self.cpw * mu / self.kw
    
    # Computes the heat transfer coefficient from water to the tube (h_it, but h_wt might be better notation). 
    # Eqn 4 from Zaheer-Uddin & Monastriakos
    def calc_hit(self, u, T):
        Re = self.WaterReynolds(u,T)
        if Re > 2900:
            return self.kw/self.di * 0.023 * Re**0.8 * self.WaterPrandtl(T)**0.3
        else:
            return 48.0/11.0 * self.kw/self.di
    
    # Computes heat transfer coefficient from radiator tube to air (h_ta, but h_tz might be better notation). 
    # Eqn 5 from Zaheer-Uddin & Monastriakos
    def calc_hta(self, Tt,Tz):
        return 1.32 * ((Tt-Tz)/self.d)**0.25
        

    # Functions to calculate fin-tube surface effectiveness (eta_s,ov):

    # Eqn 6, 7 from Zaheer-Uddin & Monastriakos
    def calc_etas(self, hta):
        gamma = ((2.0 * hta) / (self.kf * self.yf)) ** 0.5
        return np.tanh(gamma * self.lf) / (gamma * self.lf)
    # Eqn 8 from Zaheer-Uddin & Monastriakos
    def calc_etasov(self, etas):
        return 1.0 - self.Afin/self.Ao * (1.0 - etas)

    
    """
    
    OVERVIEW
    
    For each differential equation of the form dy/dt = f(y),
    the user needs to supply a python function for f(y). 
    Time discretization is done automatically. 
    
    """
    
    # RHS function for d/dt Tb = f(T)
    def f_Tb(self, T):
        Tb = self.Tb(T)
        Trtn = self.Tw.last(T)
        
        return ( 
                    self.U2 * self.U2max * (1.0 - self.alpha * Tb / self.Tbmax)  
                    - self.U1 * self.U1max * self.cpw *(Tb-Trtn)  
                    - self.ab * ( Tb - self.Te) 
                ) / self.Cb
    
    # RHS function for d/dt Tz = f(T)
    def f_Tz(self, T):
        Tbh = self.Tt.avg(T)
        Tz = self.Tz(T)        
        
        hta = self.calc_hta(Tbh, Tz)
        etas = self.calc_etas(hta)
        etasov = self.calc_etasov(etas)
        
        return ( 
                    hta * self.Ao * self.lr * etasov * (Tbh-Tz)  
                    - self.az * ( Tz - self.Ta) 
                ) / self.Cz
    
    # RHS function for d/dt Tt = f(T)
    def f_Tt(self, T, idx):
        Twthis = self.Tw(T,idx)
        Ttthis = self.Tt(T,idx)
        Tz = self.Tz(T)
        
        u = self.U1 * self.U1max / ( self.Aci * self.rho_w)
        hit = self.calc_hit(u, Twthis)
        
        hta = self.calc_hta(Ttthis, Tz)
        etas = self.calc_etas(hta)
        etasov = self.calc_etasov(etas)
            
        Mt = self.rho_tube * np.pi * ((self.d/2.0)**2.0 - (self.di/2.0)**2.0)
        Mf = self.rho_fin * self.lf**2.0 * self.yf / self.spf    
        factor = 1.0 / (Mt * self.ct + etas * Mf * self.cf)
        
        return ( 
                    factor * etasov * hta * self.Ao * (Tz-Ttthis)  
                    + factor * hit * self.Ait * (Twthis - Ttthis) 
                )
        
    
    # RHS function for d/dt Tw = f(T)
    def f_Tw(self, T, idx):
        Twthis = self.Tw(T,idx)
        Ttthis = self.Tt(T,idx)
        gradTw = self.Tw.grad(T,idx)
        
        u = self.U1 * self.U1max / ( self.Aci * self.rho_w)
        hit = self.calc_hit(u, Twthis)
        Mw = self.rho_w * self.Aci
        
        return ( 
                    hit * self.Ait * (Ttthis - Twthis)/(Mw * self.cpw)
                    - self.U1 * self.U1max/( self.rho_w * self.Aci) * gradTw
                )
    
    
    # functions for changing water flow and boiler power, for testing external control schemes.

    def chMassFlow(self, newU1):
        self.U1 = newU1
    
    def chBurnerPower (self, newU2):
        self.U2 = newU2
        

    # execute the simulation
    # see the if __name__ == '__main__' clause for example usage.

    def RunSimulation(self, **kwargs):
        
        # Control: callback that gets called every cycle to execute control commands (for testing control strategies)
        # Store: callback for storing data to disk or by some other method .
        ctrl = lambda t, T : None
        store = lambda t, T : None
        if('control' in kwargs):
            ctrl = kwargs['control']
        if('store' in kwargs):
            store = kwargs['store']
        
        print("Simulation starting, zone heat loss coefficient =", self.az, "(W/K)")
        
        # loop
        tGrid = np.arange(0.0, self.tmax, self.dt)
        
        t = 0.0
        T = self.T0
        
        for t in tGrid:        
            # compute next step
            
            # create list of RHS functions for the differential equation solver
            F = [
                lambda S : -S[0] + T[0] + self.dt * self.f_Tb(S),
                lambda S : -S[1] + T[1] + self.dt * self.f_Tz(S),
                lambda S : -S[2] + T[2] + self.dt * self.f_Tw(S,0),
                lambda S : -S[3] + T[3] + self.dt * self.f_Tw(S,1),
                lambda S : -S[4] + T[4] + self.dt * self.f_Tw(S,2),
                lambda S : -S[5] + T[5] + self.dt * self.f_Tt(S,0),
                lambda S : -S[6] + T[6] + self.dt * self.f_Tt(S,1),
                lambda S : -S[7] + T[7] + self.dt * self.f_Tt(S,2)
            ]
        
            T = self.broyden.Solve(T, F)
            
            # execute control commands
            ctrl(t,T)
            
            # store data
            store(t,T)






if __name__ == "__main__":
    
    sim = HydronicSimulation(tmax=1000.0)
    tPlotGrid = sim.GetGrid()
    T_hist    = np.tile(sim.T0, (tPlotGrid.shape[0],1))
    
    
    def PlotOutput(ts, T_hist):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)#, sharex=True, sharey=True)
        fig.tight_layout()
        
        #ax1.set_xlim((tPlotGrid[0]-updateInterval*dt, tPlotGrid[-1]+updateInterval*dt))
        #ax1.set_ylim(50, 100)
        #ax = plt.axes(xlim=(tPlotGrid[0]-updateInterval*dt, tPlotGrid[-1]+updateInterval*dt))
        
        ax1.plot(ts, T_hist[:,0], '-', label='Tb')
        ax3.plot(ts, T_hist[:,1], '-', label='Tz')
        ax2.plot(ts, T_hist[:,2], '-', label='Tw1')
        ax2.plot(ts, T_hist[:,3], '--', label='Tw2')
        ax2.plot(ts, T_hist[:,4], ':', label='Tw3')
        ax4.plot(ts, T_hist[:,5], '-', label='Tt1')
        ax4.plot(ts, T_hist[:,6], '--', label='Tt2')
        ax4.plot(ts, T_hist[:,7], ':', label='Tt3')
        ax1.set_xlabel('t (s)')
        ax2.set_xlabel('t (s)')
        ax3.set_xlabel('t (s)')
        ax4.set_xlabel('t (s)')
        
        ax1.set_ylabel('Boiler Temperature (C)')
        ax2.set_ylabel('Water Temperature (C)')
        ax3.set_ylabel('Zone Temperature (C)')
        ax4.set_ylabel('Tube Temperature (C)')
        
        plt.legend()
        plt.show()
    
    i = 0 
    def DefStorClbk(t,T):
        global i, T_hist, tPlotGrid
        print("t =", t, "s")
        T_hist[i] = KtoC(T)
        tPlotGrid[i] = t
        i += 1
    
    sim.RunSimulation(store=DefStorClbk)
    PlotOutput(tPlotGrid, T_hist)
