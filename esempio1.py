# 3D Magnetic Field Around a Long Straight Wire — Stylish Dark Mode
# Modernized aesthetics:
# - Dark background, white labels/ticks
# - Neon rings (L1–L3) with soft "glow"
# - Metal-like wire cylinder
# - Bright tracers with short fading trails
# - Tangent B arrows whose length ∝ |B| = μ0|I|/(2πr)
# - Sliders: I, r1–r3, camera elev/azim
#
# Tip: drag with mouse to rotate the 3D view; use sliders for fine control.
#
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from collections import deque
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------- Physics ----------
mu0 = 4*np.pi*1e-7

def B_phi_mag(I, r):
    r = np.maximum(np.asarray(r), 1e-9)
    return mu0*np.abs(I)/(2*np.pi*r)

# ---------- Scene bounds & init ----------
Rmax, Zmax = 3.2, 2.2
I0 = 6.0
r1_0, r2_0, r3_0 = 0.8, 1.6, 2.4
elev0, azim0 = 24, -40

fig = plt.figure(figsize=(8.2, 7.6))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.08, right=0.98, bottom=0.24, top=0.90)

# ---------- Dark Theme ----------
fig.patch.set_facecolor("black")
ax.set_facecolor("black")
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.label.set_color("white")
    axis.set_pane_color((0,0,0,0))
ax.tick_params(colors="white")
ax.xaxis.line.set_color((0,0,0,0))
ax.yaxis.line.set_color((0,0,0,0))
ax.zaxis.line.set_color((0,0,0,0))

ax.grid(False)
ax.set_xlim(-Rmax, Rmax); ax.set_ylim(-Rmax, Rmax); ax.set_zlim(-Zmax, Zmax)
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
ax.view_init(elev=elev0, azim=azim0)
ax.set_title("Campo magnetico di un filo lungo (3D) — stile dark", color="white", pad=14)

# ---------- Wire (metal-like cylinder) ----------
wire_R = 0.12
z = np.linspace(-Zmax, Zmax, 70)
theta = np.linspace(0, 2*np.pi, 90)
Theta, Z = np.meshgrid(theta, z)
Xc = wire_R*np.cos(Theta)
Yc = wire_R*np.sin(Theta)
# soft metal look
ax.plot_surface(Xc, Yc, Z, color="#9aa0a6", alpha=0.35, linewidth=0, zorder=1)

# Current direction arrow and symbol
arrow = ax.quiver(0, 0, -0.4, 0, 0, 0.9, length=1, normalize=False, color="#ffcc66")
sym = ax.text(0, 0, 0.06, "⊙", ha='center', va='center', fontsize=18, fontweight='bold', color="white")

# ---------- Neon ring helper ----------
def glow_ring(r, base_color="#62E9FF", levels=4, alpha0=0.9):
    """Draw a ring with multiple strokes to fake a glow effect."""
    t = np.linspace(0, 2*np.pi, 600)
    x, y, z = r*np.cos(t), r*np.sin(t), np.zeros_like(t)
    lines = []
    for i in range(levels):
        lw = 2.6 - 0.5*i
        alpha = max(0.1, alpha0 - 0.22*i)
        line, = ax.plot(x, y, z, color=base_color, linewidth=lw, alpha=alpha, zorder=2+i)
        lines.append(line)
    return lines

# Create rings with glow
L1_lines = glow_ring(r1_0, base_color="#62E9FF")  # cyan
L2_lines = glow_ring(r2_0, base_color="#A78BFA")  # violet
L3_lines = glow_ring(r3_0, base_color="#FB923C")  # orange
tL1 = ax.text(r1_0+0.10, 0.10, 0.0, "L1", color="white")
tL2 = ax.text(r2_0+0.10, 0.10, 0.0, "L2", color="white")
tL3 = ax.text(r3_0+0.10, 0.10, 0.0, "L3", color="white")

# ---------- Tracers with fading trails ----------
def make_tracer(color):
    pt, = ax.plot([0], [0], [0], marker='o', markersize=8, color=color, linestyle='')
    trail, = ax.plot([], [], [], color=color, linewidth=2, alpha=0.8)
    return pt, trail, deque(maxlen=28)

p1, tr1, hist1 = make_tracer("#62E9FF")  # cyan
p2, tr2, hist2 = make_tracer("#A78BFA")  # violet
p3, tr3, hist3 = make_tracer("#FB923C")  # orange

# Local B arrows (tangent) — recreate per frame
q1 = ax.quiver([], [], [], [], [], [], color="#62E9FF")
q2 = ax.quiver([], [], [], [], [], [], color="#A78BFA")
q3 = ax.quiver([], [], [], [], [], [], color="#FB923C")

# ---------- Sliders ----------
axI   = plt.axes([0.08, 0.16, 0.84, 0.03], facecolor="#111111")
axr1  = plt.axes([0.08, 0.12, 0.84, 0.03], facecolor="#111111")
axr2  = plt.axes([0.08, 0.08, 0.84, 0.03], facecolor="#111111")
axr3  = plt.axes([0.08, 0.04, 0.84, 0.03], facecolor="#111111")
axEl  = plt.axes([0.08, 0.20, 0.40, 0.03], facecolor="#111111")
axAz  = plt.axes([0.52, 0.20, 0.40, 0.03], facecolor="#111111")

sI  = Slider(axI,  "I [A]", -12.0, 12.0, valinit=I0,  valstep=0.1, color="#6666ff")
sr1 = Slider(axr1, "r1 [m]", 0.30, 2.9,  valinit=r1_0, valstep=0.01, color="#62E9FF")
sr2 = Slider(axr2, "r2 [m]", 0.50, 3.0,  valinit=r2_0, valstep=0.01, color="#A78BFA")
sr3 = Slider(axr3, "r3 [m]", 0.70, 3.1,  valinit=r3_0, valstep=0.01, color="#FB923C")
sEl = Slider(axEl, "elev",   -5,   60,   valinit=elev0, valstep=1,    color="#999999")
sAz = Slider(axAz, "azim",   -180, 180,  valinit=azim0, valstep=1,    color="#999999")

for lab in (axI, axr1, axr2, axr3, axEl, axAz):
    lab.tick_params(colors="white")
    lab.xaxis.label.set_color("white")
    lab.yaxis.label.set_color("white")

# ---------- Animation state ----------
a1, a2, a3 = 0.0, 1.1, 2.3

def update_ring(lines, r):
    t = np.linspace(0, 2*np.pi, 600)
    x, y, z = r*np.cos(t), r*np.sin(t), np.zeros_like(t)
    for ln in lines:
        ln.set_data(x, y)
        ln.set_3d_properties(z)

def set_point(line, x, y, z):
    line.set_data([x], [y])
    line.set_3d_properties([z])

def update_trail(trail, hist):
    if len(hist) >= 2:
        X, Y, Z = zip(*hist)
        trail.set_data(X, Y)
        trail.set_3d_properties(Z)

def remake_quiver(old, x, y, z, u, v, w, color):
    old.remove()
    return ax.quiver([x], [y], [z], [u], [v], [w], length=1, normalize=False, color=color)

def step(frame):
    global a1, a2, a3, q1, q2, q3, arrow

    I  = sI.val
    r1, r2, r3 = sr1.val, sr2.val, sr3.val
    elev, azim = sEl.val, sAz.val

    # angular speeds ~ I/r (sign from I)
    k = 1.1
    dt = 0.035
    a1 = (a1 + k*I/max(r1,1e-6)*dt) % (2*np.pi)
    a2 = (a2 + k*I/max(r2,1e-6)*dt) % (2*np.pi)
    a3 = (a3 + k*I/max(r3,1e-6)*dt) % (2*np.pi)

    # positions on z=0
    x1, y1, z1 = r1*np.cos(a1), r1*np.sin(a1), 0.0
    x2, y2, z2 = r2*np.cos(a2), r2*np.sin(a2), 0.0
    x3, y3, z3 = r3*np.cos(a3), r3*np.sin(a3), 0.0

    set_point(p1, x1, y1, z1); set_point(p2, x2, y2, z2); set_point(p3, x3, y3, z3)

    # trails
    hist1.append((x1,y1,z1)); hist2.append((x2,y2,z2)); hist3.append((x3,y3,z3))
    update_trail(tr1, hist1); update_trail(tr2, hist2); update_trail(tr3, hist3)

    # tangent directions
    t1 = np.array([-np.sin(a1), np.cos(a1), 0.0])
    t2 = np.array([-np.sin(a2), np.cos(a2), 0.0])
    t3 = np.array([-np.sin(a3), np.cos(a3), 0.0])
    s1 = 0.7*B_phi_mag(I, r1)*np.sign(I)
    s2 = 0.7*B_phi_mag(I, r2)*np.sign(I)
    s3 = 0.7*B_phi_mag(I, r3)*np.sign(I)
    q1 = remake_quiver(q1, x1, y1, z1, t1[0]*s1, t1[1]*s1, t1[2]*s1, "#62E9FF")
    q2 = remake_quiver(q2, x2, y2, z2, t2[0]*s2, t2[1]*s2, t2[2]*s2, "#A78BFA")
    q3 = remake_quiver(q3, x3, y3, z3, t3[0]*s3, t3[1]*s3, t3[2]*s3, "#FB923C")

    # update glow rings & labels
    update_ring(L1_lines, r1); update_ring(L2_lines, r2); update_ring(L3_lines, r3)
    tL1.set_position((r1+0.10, 0.10)); tL1.set_3d_properties(0.0)
    tL2.set_position((r2+0.10, 0.10)); tL2.set_3d_properties(0.0)
    tL3.set_position((r3+0.10, 0.10)); tL3.set_3d_properties(0.0)

    # arrow & symbol
    arrow.remove()
    dirlen = 0.9 if I >= 0 else -0.9
    arrow = ax.quiver(0, 0, -0.4, 0, 0, dirlen, length=1, normalize=False, color="#ffcc66")
    sym.set_text("⊙" if I >= 0 else "⊗")

    # camera
    ax.view_init(elev=elev, azim=azim)

    return p1, p2, p3, tr1, tr2, tr3, q1, q2, q3, *L1_lines, *L2_lines, *L3_lines, tL1, tL2, tL3, arrow, sym

ani = FuncAnimation(fig, step, interval=30, blit=False)

# --- Optional: to export a video (uncomment) ---
# ani.save("campo_magnetico_dark.mp4", fps=30, dpi=180, bitrate=6000)

plt.show()
