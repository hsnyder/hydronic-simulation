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
import numpy as np

# A simple class that implements PID (proportional, integral, derivative) control

class PIDController:
   


    def __init__(self, Kp=0.0, Ti=0.0, Td=0.0, bias=0.0, outMax=None, outMin=None):
        # change these to weight the PID terms 
        self.Kp = Kp
        self.Ti = Ti
        self.Td = Td
        self.bias = bias
        
        #set point
        self.SP = 0.0
        
        #output saturation
        self.outMax = outMax # set this to a float to activate
        self.outMin = outMin # set this to a float to activate
        
        #internal, don't modify from outside
        self.prev_PV = 0.0
        self.integral = 0.0
   
        # You can change the variables Kp, Ti, Td, bias, SP, outMax, and outMin after object construction.
    
    def next(self, PV, dt): # computes the next output point for the controller
        error = self.SP - PV
    
        derivative = (PV-self.prev_PV)/dt
        newintegral = self.integral + error * dt            
        output = self.bias + self.Kp * error + self.Kp/self.Ti * newintegral + self.Kp*self.Td * derivative
        
        
        if self.outMax != None and output > self.outMax:
            newintegral = (self.outMax - self.bias - self.Kp * error - self.Kp*self.Td * derivative) / (self.Kp/self.Ti)
            output = self.outMax
            
        elif self.outMin != None and output < self.outMin:
            newintegral = (self.outMin - self.bias - self.Kp * error - self.Kp*self.Td * derivative) / (self.Kp/self.Ti)
            output = self.outMin
        
        self.integral = newintegral
        self.prev_PV = PV
        return output
