import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

# ---------------------------------------------------------------------------------------
#
# This is the phase portrait analysis tool which is used, it is recommended that the document
# provided is read and used to understand the equations and theory
#
# ----------------------------------------------------------------------------------------

# -------------------------------------------------
# Vehicle Parameters
# -------------------------------------------------
m = 300  # mass in kg
Iz = 150.0  # yaw inertia in kgm^2
lr = 0.8  # CoG to front axle in m
lf = 0.8  # CoG to rear axle in m

# ---------------------------------------------
# Speed and Steering
# ---------------------------------------------
V = 12  # speed in m/s
delta_deg = 1 # steering input angle (degrees)

delta = np.deg2rad(delta_deg)
# -------------------------------------------------
# Tyre Parameters
# ---------------------------------------------------------
F_zf = m * 9.81 * lr / (lf + lr)
F_zr = m * 9.81 * lf / (lf + lr)
F_z0 = 735.75  # nominal tyre load in N

Gamma = np.deg2rad(0)  # camber angle in degrees

# -------------------------------
# Front tyre parameters
p_Cy1_f, p_Dy1_f, p_Dy2_f, p_Dy3_f = 1.4, 1.2, -0.2, 10
p_Ey1_f, p_Ey2_f = -0.1, -0.05
p_Ky1_f, p_Ky2_f, p_Ky3_f = 20, 2, -0.50

# Rear tyre parameters
p_Cy1_r, p_Dy1_r, p_Dy2_r, p_Dy3_r = 1.4, 1.2, -0.2, 10
p_Ey1_r, p_Ey2_r = -0.1, -0.05
p_Ky1_r, p_Ky2_r, p_Ky3_r = 20, 2, -0.50

#==========================Plot control=======================================
#control the analysis region
Horizontal_axis = 1  # range of side slip values / graph axis (radians)
Vertical_axis = 2  # range of yaw rate values /  graph axis(radians)
Plot_Trajectories = False    #red lines, change to False if you want to disable



# -------------------------------------------------
# Simplified Pacejka "pure slip" lateral model, I have used identical tyres front and rear but this can be modified
# -------------------------------------------------
def lateral_pure_slip(alpha, Fz, Fz0, gamma,
                      p_Cy1, p_Dy1, p_Dy2, p_Dy3,
                      p_Ey1, p_Ey2,
                      p_Ky1, p_Ky2, p_Ky3):

    dfz = (Fz - Fz0) / Fz0

    SHy = 0.0001 + 0.0002 * dfz + 0.003 * gamma
    SVy = Fz * ((0.001 + 0.002 * dfz) + (0.003 + 0.004 * dfz) * gamma)

    Cy = p_Cy1
    mu_y = (p_Dy1 + p_Dy2 * dfz) * (1 - p_Dy3 * gamma ** 2)
    Dy = mu_y * Fz

    alpha = np.clip(alpha, -np.pi/2, np.pi/2)
    alpha_y = alpha + SHy

    Ey = p_Ey1 + p_Ey2 * dfz
    Ky = p_Ky1 * Fz0 * np.sin(2 * np.arctan(Fz / (p_Ky2 * Fz0))) * (1 - p_Ky3 * abs(gamma))
    By = Ky / max(Cy * Dy, 1e-6)  # safeguard

    return Dy * np.sin(Cy * np.arctan(By * alpha_y - Ey * (By * alpha_y - np.arctan(By * alpha_y)))) + SVy


# This calculates the slope (dFy/dα) at α = 0 for a single tyre.
eps = 1e-6  # A very small slip angle for numerical differentiation

# Calculate for FRONT tyre
Fy_f_plus = lateral_pure_slip(eps, F_zf, F_z0, Gamma, p_Cy1_f, p_Dy1_f, p_Dy2_f, p_Dy3_f, p_Ey1_f, p_Ey2_f, p_Ky1_f, p_Ky2_f, p_Ky3_f)
Fy_f_minus = lateral_pure_slip(-eps, F_zf, F_z0, Gamma, p_Cy1_f, p_Dy1_f, p_Dy2_f, p_Dy3_f, p_Ey1_f, p_Ey2_f, p_Ky1_f, p_Ky2_f, p_Ky3_f)
C_alpha_tire_front = (Fy_f_plus - Fy_f_minus) / (2 * eps) # Stiffness per SINGLE front tyre (N/rad)

# Calculate for REAR tyre
Fy_r_plus = lateral_pure_slip(eps, F_zr, F_z0, Gamma, p_Cy1_r, p_Dy1_r, p_Dy2_r, p_Dy3_r, p_Ey1_r, p_Ey2_r, p_Ky1_r, p_Ky2_r, p_Ky3_r)
Fy_r_minus = lateral_pure_slip(-eps, F_zr, F_z0, Gamma, p_Cy1_r, p_Dy1_r, p_Dy2_r, p_Dy3_r, p_Ey1_r, p_Ey2_r, p_Ky1_r, p_Ky2_r, p_Ky3_r)
C_alpha_tire_rear = (Fy_r_plus - Fy_r_minus) / (2 * eps) # Stiffness per SINGLE rear tyre (N/rad)

# The bicycle model uses stiffness for the entire AXLE (2 tyres)
C_alpha_axle_front = 2 * C_alpha_tire_front
C_alpha_axle_rear = 2 * C_alpha_tire_rear

print(f"\nCalculated from Pacejka Model:")
print(f"Front Axle Cornering Stiffness: {C_alpha_axle_front:.0f} N/rad")
print(f"Rear Axle Cornering Stiffness: {C_alpha_axle_rear:.0f} N/rad")

# -------------------------------------------------
# System dynamics (β-dot, r-dot)
# -------------------------------------------------
def f_nonlinear(t, x):
    beta, r = x
    alpha_f = delta - beta - (lf / V) * r
    alpha_r = -beta + (lr / V) * r

    # Front tyre force
    Fy_f = 2 * lateral_pure_slip(alpha_f, F_zf, F_z0, Gamma,
                                 p_Cy1_f, p_Dy1_f, p_Dy2_f, p_Dy3_f,
                                 p_Ey1_f, p_Ey2_f,
                                 p_Ky1_f, p_Ky2_f, p_Ky3_f)

    # Rear tyre force
    Fy_r = 2 * lateral_pure_slip(alpha_r, F_zr, F_z0, Gamma,
                                 p_Cy1_r, p_Dy1_r, p_Dy2_r, p_Dy3_r,
                                 p_Ey1_r, p_Ey2_r,
                                 p_Ky1_r, p_Ky2_r, p_Ky3_r)

    beta_dot = (Fy_f + Fy_r) / (m * V) - r
    r_dot = (lf * Fy_f - lr * Fy_r) / Iz
    return [beta_dot, r_dot]


# -------------------------------------------------
# Find equilibrium positions
# -------------------------------------------------
def find_equilibria():
    def equations(x):
        return f_nonlinear(0, x)

    equilibria = []
    beta_guesses = np.linspace(-Horizontal_axis, Horizontal_axis, 20)
    r_guesses = np.linspace(-Vertical_axis, Vertical_axis, 20)

    for b in beta_guesses:
        for r in r_guesses:
            sol = fsolve(equations, [b, r], full_output=True)
            if sol[2] == 1:  # Converged
                eq_point = np.round(sol[0], decimals=4)


                if (abs(eq_point[0]) <= Horizontal_axis and
                    abs(eq_point[1]) <= Vertical_axis):

                    if not any(np.allclose(eq_point, existing, atol=1e-3) for existing in equilibria):
                        equilibria.append(eq_point)

    return equilibria
# -------------------------------------------------
# Phase portrait with smooth streamlines
# -------------------------------------------------
beta_vals = np.linspace(-Horizontal_axis, Horizontal_axis, 60)
r_vals = np.linspace(-Vertical_axis, Vertical_axis, 60)
B, R = np.meshgrid(beta_vals, r_vals)

# Vector field
dB, dR = np.zeros_like(B), np.zeros_like(R)
for i in range(B.shape[0]):
    for j in range(B.shape[1]):
        dB[i, j], dR[i, j] = f_nonlinear(0, [B[i, j], R[i, j]])

# -------------------------------------------------
# Compute divergence and curl using numpy.gradient
# -------------------------------------------------
def calculate_divergence_curl(B, R, dB, dR, beta_vals, r_vals, smooth_sigma=None):
    dbeta = beta_vals[1] - beta_vals[0]  # step size in β
    dr = r_vals[1] - r_vals[0]           # step size in r

    # Gradient returns [∂/∂r , ∂/∂β] because arrays are (r, β)
    dB_dr, dB_dbeta = np.gradient(dB, dr, dbeta, edge_order=2)
    dR_dr, dR_dbeta = np.gradient(dR, dr, dbeta, edge_order=2)

    # Divergence: ∂β̇/∂β + ∂ṙ/∂r
    divergence = dB_dbeta + dR_dr

    # Curl (scalar in 2D): ∂ṙ/∂β - ∂β̇/∂r
    curl = dR_dbeta - dB_dr

    return divergence, curl

# -------------------------------------------------
# Calculate divergence and curl (replace your loop version with this)
# -------------------------------------------------
divergence, curl = calculate_divergence_curl(B, R, dB, dR, beta_vals, r_vals)

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], width_ratios=[1, 1], hspace=0.3, wspace=0.3)

# -------------------------------
# Top: Phase portrait (spans both columns)
# -------------------------------
ax1 = fig.add_subplot(gs[0, :])
ax1.streamplot(B, R, dB, dR, color="blue", density=3.0, linewidth=0.5)

if Plot_Trajectories == True:
    initial_conditions = [[0.1, 0.2], [-0.1, 0.2], [0.1, -0.2], [-0.1, -0.2]]
    for x0 in initial_conditions:
        sol = solve_ivp(f_nonlinear, [0, 15], x0, t_eval=np.linspace(0, 15, 50))
        ax1.plot(sol.y[0], sol.y[1], 'r-', lw=1.2, alpha=0.8)

# Equilibria
for i, eq in enumerate(find_equilibria()):
    print(f"Equilibrium {i + 1}: β = {eq[0]:.4f}, r = {eq[1]:.4f}")
    ax1.plot(eq[0], eq[1], 'ko', markersize=6)

ax1.set_xlabel("β [rad]")
ax1.set_ylabel("r [rad/s]")
ax1.set_title(f"Phase Portrait (δ={delta_deg}°, V={V} m/s)")
ax1.text(0.02, 0.98, '● Equilibrium Points', transform=ax1.transAxes,
         verticalalignment='top', fontsize=10, bbox=dict(boxstyle="round,pad=0.3",
         facecolor="white", alpha=0.8))
ax1.grid(True)
ax1.set_xlim(-Horizontal_axis, Horizontal_axis)
ax1.set_ylim(-Vertical_axis, Vertical_axis)

# -------------------------------
# Bottom left: Divergence
# -------------------------------
ax2 = fig.add_subplot(gs[1, 0])
div_plot = ax2.contourf(B, R, divergence, 50, cmap="RdYlBu")
plt.colorbar(div_plot, ax=ax2, label='Divergence')
ax2.set_xlabel("β [rad]")
ax2.set_ylabel("r [rad/s]")
ax2.set_title("Divergence (∇·F)")
ax2.grid(True)
ax2.set_xlim(-Horizontal_axis, Horizontal_axis)
ax2.set_ylim(-Vertical_axis, Vertical_axis)

# -------------------------------
# Bottom right: Curl
# -------------------------------
ax3 = fig.add_subplot(gs[1, 1])
curl_plot = ax3.contourf(B, R, curl, 50, cmap='RdYlBu')
plt.colorbar(curl_plot, ax=ax3, label='Curl')
ax3.set_xlabel("β [rad]")
ax3.set_ylabel("r [rad/s]")
ax3.set_title("Curl (∇×F)")
ax3.grid(True)
ax3.set_xlim(-Horizontal_axis, Horizontal_axis)
ax3.set_ylim(-Vertical_axis, Vertical_axis)

plt.tight_layout()
plt.show()

# Print some statistics

print(f"Divergence range: [{divergence.min():.3f}, {divergence.max():.3f}]")
print(f"Curl range: [{curl.min():.3f}, {curl.max():.3f}]")
