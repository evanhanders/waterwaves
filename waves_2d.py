"""
Dedalus script for simulating waves at a water-air interface in a 2D horizontally-periodic domain.
"""

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)


# Parameters
rho_water = 1e3 #kg / m^3
rho_air = 1.225 #kg / m^3
rho_water_nd = rho_water / rho_water
rho_air_nd = rho_air / rho_water
gravity = 9.81 #m / s^2
L = 0.2 #m
time = 1 #s
m_nd = rho_water * L**3 #nondimensional mass in kg
gravity_nd = gravity * time**2 / L
water_viscosity_dynamic = 1 #mPa.s = newton-second / m^2 = kg / m / s -- at ~20 degrees celsius. This is dynamic viscosity
water_viscosity_kinematic = water_viscosity_dynamic / rho_water #m^2 / s
water_viscosity_dynamic_nd = water_viscosity_dynamic * L * time / m_nd
water_viscosity_kinematic_nd = water_viscosity_kinematic * time / L**2

sigma_st_water = 0.072 #N/m = kg / s^2
sigma_st_nd = sigma_st_water * time**2 / m_nd

#for simplicity we will assume air viscosity is equal to water viscosity
from scipy.special import erf

def one_to_zero(x, x0, width=0.1):
    """ One minus Smooth Heaviside function (1 - H) """
    return (1 - erf( (x - x0)/width))/2

def one_to_zero2(z, z0, width=0.1):
    """ One minus Smooth Heaviside function (1 - H) """
    return (1 - erf( (z - z0)/width))/2

def zero_to_one(*args, **kwargs):
    """ Smooth Heaviside Function """
    return -(one_to_zero(*args, **kwargs) - 1)

def zero_to_one2(*args, **kwargs):
    """ Smooth Heaviside Function """
    return -(one_to_zero(*args, **kwargs) - 1)

#nondimensionalization above for L and time, rho_nd = rho_water.
Lx, Lz = 2, 1
Nx, Nz = 128*4, 64*4
dealias = 3/2

stop_sim_time = 1
timestepper = d3.SBDF2
max_timestep = 1e-3
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

# Fields
p     = dist.Field(name='p',     bases=(xbasis,zbasis))
alpha = dist.Field(name='alpha', bases=(xbasis,zbasis)) #volume fraction
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
zfield = dist.Field(name='zfield', bases=(xbasis,zbasis))
tau_p = dist.Field(name='tau_p')
tau_a1 = dist.Field(name='tau_a1', bases=xbasis)
tau_a2 = dist.Field(name='tau_a2', bases=xbasis)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)
z2 = dist.local_grid(zbasis, scale=3/2)

# Substitutions
x, z = dist.local_grids(xbasis, zbasis)
#z2 = dist.local_grids(zbasis)
ex, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
grad_alpha = d3.grad(alpha) + ez*lift(tau_a1) # First-order reduction
zfield.change_scales(3/2)
zfield['g'] = z2
zfield.change_scales(1)
print(zfield['g'].shape)
print(alpha['g'].shape)


#Two-fluid terms
rho = rho_water_nd* alpha + rho_air_nd*(1-alpha)
alpha_diffusion = water_viscosity_kinematic_nd/20
grad_alp = d3.grad(alpha)
kap = d3.div(grad_alp / np.sqrt(grad_alp@grad_alp))

print('water_viscosity_kinematic_nd = ', water_viscosity_kinematic_nd)

#F_grav = (gravity_nd*zfield)*d3.grad(rho)/rho + d3.grad(gravity_nd*zfield)
F_grav = -(gravity_nd*zfield)*d3.grad(rho)/rho - gravity_nd*ez
F_grav_eval = F_grav.evaluate()

print(F_grav_eval['g']) 

#F_grav = gravity_nd
#F_grav = -(d3.grad(rho)/rho)@(gravity_nd*ez)
#F_grav = -(d3.grad(rho))@(gravity_nd*ez)

#F_grav = -d3.grad(rho*gravity_nd)/rho
#F_grav = gravity_nd*ez

F_st = (sigma_st_nd/rho)*kap*grad_alp

# Problem
# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"
problem = d3.IVP([p, alpha, u, tau_p, tau_a1, tau_a2, tau_u1, tau_u2], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(alpha) - alpha_diffusion*div(grad_alpha) + lift(tau_a2) = - u@grad(alpha)")
problem.add_equation("dt(u) - water_viscosity_kinematic_nd*div(grad_u) + grad(p)  + lift(tau_u2) = - F_grav - (u@grad(u)) + F_st")
problem.add_equation("ez@grad_alp(z=0) = 0")
problem.add_equation("u(z=0) = 0")
problem.add_equation("ez@grad_alp(z=Lz) = 0")
problem.add_equation("u(z=Lz) = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge

# consider hyper-diffusion?

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions

#alpha['g'] = 1- 1/(1+np.exp(-(z-(0.5+0.1*np.sin(2*np.pi*x/Lx)))/0.001)) #sigmoid with water at bottom and air on top.
alpha.change_scales(1)

#alpha['g'] = 1 - 1/(1+np.exp(-(z-(0.5+0.2*np.sin(2*np.pi*x/Lx)))/0.025))

## IC: Step water dam 
# alpha = 1 at 0 < x < Lx/2
# alpha = 0 at Lx/2 < x < Lx

alpha['g'] = zero_to_one(x, Lx/3, width = 0.1)*one_to_zero(x, 2*Lx/3, width =0.1)* one_to_zero2(z, 3*Lz/4, width =0.1) 
#- zero_to_one(x, Lx/4, width = 0.1)*one_to_zero(x, 3*Lx/4, width =0.1)* one_to_zero2(z, 1*Lz/4, width =0.1)

## 
#alpha['g'] = 1 - (Lz/Lx)*x 

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.1, iter=20, max_writes=50)
snapshots.add_task(alpha, name='alpha')
snapshots.add_task(u, name='u')
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=1, safety=0.2, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u@u)/water_viscosity_kinematic_nd, name='Re')

# Main loop
startup_iter = 10
timestep = CFL.compute_timestep()
try:
    logger.info('Starting main loop')
    while solver.proceed:
        cfl_dt = CFL.compute_timestep()
        if cfl_dt < timestep:
            timestep = cfl_dt
        else:
            CFL.stored_dt = timestep
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_Re = flow.max('Re')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %(solver.iteration, solver.sim_time, timestep, max_Re))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

