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

from functools import reduce

"""
This class finds the roots of systems of equations using Broyden's method
(which is basically the secant method for systems of equations). 
Code and comments use "secant" and "broyden" interchangeably. 
"""
class Broyden:
    
    def __init__(self, secant_delta=1e-3, secant_tol=1e-6):
        self.secant_delta = secant_delta # tiny number for secant method jacobian calculation
        self.secant_tol   = secant_tol   # tolerance for secant method termination
    
    """
    Pass in:
        -a guessed state that is close to the root - you can just use the previous state (i.e. state vector at last time step)
        -list of euqations to be solved (specified as a list of functions)
    This function finds the correct input state vector to make the output of all functions very close to zero.
    """
    def Solve(self, guess, F):
        delta = self.secant_delta
        T = guess
        f = np.array([fn(T) for fn in F])
        J = np.zeros((8,8))
        for i in range(8):
            for j in range(8):
                Tplus = T.copy()
                Tminus = T.copy()
                Tplus[j] += delta
                Tminus[j] -= delta
                J[i,j] = (F[i](Tplus) - F[i](Tminus))/(2.0*delta)
                
        Jinv = np.linalg.inv(J)
        
        i = 0
        while np.linalg.norm(f) > self.secant_tol:
            i += 1
            
            Tnext = T - np.dot(Jinv, f)
            fnext = np.array([fn(Tnext) for fn in F])
            dT = Tnext-T
            df = fnext-f
            dTT = dT.transpose()
            
            intermed = (dT - np.dot(Jinv,df)) / np.dot(np.dot(dTT,Jinv),df)
            Jinv += np.dot(np.dot(intermed,dTT),Jinv)
            
            f = fnext
            T = Tnext
    
        return T
        
"""
This class is a convenience wrapper around numpy.linspace and numpy.arange.
It creates discrete grids in 1 dimension, given a start point, a stop point,
and either a number of points in the grid, or the spacing between points. 
"""
class Grid:
    def dx(self):
        return self.grid[1] - self.grid[0]
    
    # You must specify either num or step but not both
    def __init__ (self, start, stop, num=None, step=None): 
        if num == None and step != None:
            self.grid = np.arange(start,stop, step)
        elif num != None and step == None:
            self.grid = np.linspace(start,stop, num)
        else:
            raise ValueError("Grid.__init__: you must supply num or step but not both")


"""
Since the solver class (Broyden) takes all variables as a flat vector, but sometimes it's more convenient
to write code that refers to variables by name, the Scalar class acts as a handle to a value stored in a
state vector. It is to be used in conjunction with the function at the bottom of this file, InitState.
"""      
class Scalar: 
    
    # given a solver state vector, retrieve current value of this datum
    #  (at specified grid index, if applicable)
    def __call__(self, state , idx=0):
        if self.grid == None and idx != 0:
            raise ValueError('Scalar.__call__: index must be zero for variables not on a grid')
        elif self.grid != None and (idx < 0 or idx > self.grid.grid.size-1):
            raise ValueError('Scalar.__call__: index out of range')
        return state[self.stateOffset+idx]
    
    # retrieve average value across grid
    def avg(self, state):
        if self.grid == None:
            raise ValueError('Cannot call avg on a non-discretized variable')
        N = self.grid.grid.size
        return sum(state[self.stateOffset:self.stateOffset+N])/N
    
    # retrieve first value on grid
    def first(self, state):
        if self.grid == None:
            raise ValueError('Cannot call fist on a non-discretized variable')
        return state[self.stateOffset]
    
    # retrieve last value on grid
    def last(self, state):
        if self.grid == None:
            raise ValueError('Cannot call last on a non-discretized variable')
        N = self.grid.grid.size
        return state[self.stateOffset+N-1]
    
    # calculate "gradient" (spatial derivative)
    def grad(self, state, idx):
        # currently, linear upwind scheme supported.
        prv = 0.0
        if idx > 0:
            prv = self(state, idx-1)
        else:
            #use boundary condition
            if self.bStart == None:
                raise ValueError('grad: no boundary condition')
            elif isinstance(self.bStart, Scalar):
                prv = self.bStart(state)
            else:
                raise TypeError('grad: unsupported boundary condition')
            
        return (self(state,idx) - prv) / self.grid.dx()
        
        
    def __init__(self, grid, value, boundaryStart=None, boundaryEnd=None):
        self.grid = grid
        self.initVal = value
        self.bStart = boundaryStart
        self.bEnd = boundaryEnd
        
        # for use by the simulation object
        self.stateOffset = -1  # stores the index offset into the state vector
 
"""
Responsible for assigning the acutal index into the state vector to a bunch of scalars.
Pass in a list of scalar objects, and this function will give you an initial state with
everything set appropriately.
"""
def InitState(*variables):
    def expandVar(var):
        if var.grid == None: return var.initVal
        else:  return np.repeat(var.initVal, var.grid.grid.size)
    
    def flatCat(var1,var2): return np.concatenate((var1,var2),axis=None)
        
    state = reduce(flatCat,map(expandVar, variables))
    
    i = 0
    for v in variables:
        v.stateOffset = i
        if v.grid == None:
            i += 1
        else:
            i += v.grid.grid.size
            
    return state
