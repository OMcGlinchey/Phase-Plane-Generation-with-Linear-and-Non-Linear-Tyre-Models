import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Vehicle parameters
m = 300.0 #kg
Iz = 150.0 #yaw inertia in kgm^2
lf = 0.8 # CoG to front axle in m
lr = 0.8 # CoG to rear axle in m
Caf = 29430  #front axial cornering stiffness N/rad
Car = 29430 # rear axial cornering stiffness N/rad   (2*tyre cornering stiffness)


# Speed and steer input
V = 12 # speed in m/s
delta = 1 #enter the steering angle in deg
delta = np.deg2rad(delta)

#increase this number if you want to increase size of vector field
grid_size = 2   #increase this number of the vector field is cut off
# -------------------------------------------------
# State-space dynamics
# -------------------------------------------------
def f(t, x):
    beta, r = x
    beta_dot = -(Caf + Car)/(m*V) * beta \
               + ((Car*lr - Caf*lf)/(m*V**2) - 1.0) * r \
               + (Caf/(m*V)) * delta

    r_dot    = (lr*Car - lf*Caf)/Iz * beta \
               - (lf**2*Caf + lr**2*Car)/(Iz*V) * r \
               + (lf*Caf/Iz) * delta
    return [beta_dot, r_dot]

# System matrices
A = np.array([
    [-(Caf + Car)/(m*V), (Car*lr - Caf*lf)/(m*V**2) - 1],
    [(lr*Car - lf*Caf)/Iz, -(lf**2*Caf + lr**2*Car)/(Iz*V)]
])
B = np.array([[Caf/(m*V)], [lf*Caf/Iz]])

# Equilibrium point
x_eq = -np.linalg.inv(A) @ B * delta

# -------------------------------------------------
# Phase plane construction
# -------------------------------------------------

beta_vals = np.linspace(-grid_size, grid_size, 80)
r_vals    = np.linspace(-grid_size, grid_size, 80)
BETA, R = np.meshgrid(beta_vals, r_vals)

beta_dot, r_dot = np.zeros_like(BETA), np.zeros_like(R)
for i in range(BETA.shape[0]):
    for j in range(BETA.shape[1]):
        dx = f(0, [BETA[i,j], R[i,j]])
        beta_dot[i,j], r_dot[i,j] = dx

# -------------------------------------------------
# Plot
# -------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 8))
ax.streamplot(BETA, R, beta_dot, r_dot, color="blue", density=3.0, linewidth=1)

# Add trajectories
initial_conditions = [[0.1, 0.2], [-0.1, 0.2], [0.1, -0.2], [-0.1, -0.2]]
for x0 in initial_conditions:
    sol = solve_ivp(f, [0, 5], x0, t_eval=np.linspace(0, 5, 200))
    #ax.plot(sol.y[0], sol.y[1], 'r-', lw=2)


# -------------------------------------------------
# Stability analysis
# -------------------------------------------------
eigenvalues = np.linalg.eigvals(A)
print(f"Equilibrium point: β = {x_eq[0,0]:.4f} rad, r = {x_eq[1,0]:.4f} rad/s")
print(f"Eigenvalues: {eigenvalues[0]:.4f}, {eigenvalues[1]:.4f}")

if np.all(np.real(eigenvalues) < 0):
    print("System is stable")
else:
    print("System is unstable")

# Equilibrium point
ax.plot(x_eq[0], x_eq[1], 'ko', markersize=10, label="Equilibrium")
ax.set_xlabel("β [rad]")
ax.set_ylabel("r [rad/s]")
ax.set_title("Phase Plane Portrait (Linear Tyre Model)")
ax.legend()
ax.grid(True)
plt.show()
