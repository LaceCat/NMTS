"""
浓密机自然沉降动画
从空机 (x=1e-6) 开始，只开进料泵 (Q_uf=0, Qf=40, Cf=0.35)
沉降到底流浓度 C_uf = 0.66
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.animation as animation
from matplotlib import colors

def c2d(c):
    rho_0, rho_w = 4.27, 1.0
    return rho_0 / (rho_0 - (rho_0 - rho_w) * c)

def d2c(d):
    rho_0, rho_w = 4.27, 1.0
    if abs(d) < 1e-10:
        return 0.0
    return rho_0 * (d - rho_w) / ((rho_0 - 1) * d)

# ====== 参数 ======
D1=0.547678836441774; D2=0.476947700914306
D3=0.940451926648319*2; D4=0.952633313167728
v1=1/40*50+0.25; v2=16.5914198572464
v3=1.14944107415579*2; v4=1.73671299891156*1.5
rv1=7.50829943915578e-06; rv2=8.48008041543593e-06
rv3=10.62486693809181e-07; rv4=13.58704316829464e-07
z=5.79; zf=4; n=10; A=np.pi*15**2; detaz=z/n
deltat=1/60

def step_fn(Q_uf, Qf, Cf, state):
    qu_A=Q_uf/A; qf_A=Qf/A; qe_A=(qf_A-qu_A)/A
    xf=c2d(Cf)*1e6; qushang=qu_A*0.8
    x=state.copy(); xn=np.zeros(10)
    xn[0]=x[0]-qushang*x[0]+(-v1*np.exp(-rv1*x[0])*x[0]/detaz+D1*(x[1]-x[0])/detaz**2)*deltat
    xn[1]=x[1]+(qushang*x[0]-qushang*x[1])+((qe_A*x[2]-qe_A*x[1]+v1*np.exp(-rv1*x[0])*x[0]-v1*np.exp(-rv1*x[1])*x[1])/detaz+(D1*(x[2]-x[1])-D1*(x[1]-x[0]))/detaz**2)*deltat
    xn[2]=x[2]+(qushang*x[1]-qushang*x[2])+((qe_A*x[3]-qe_A*x[2]+v1*np.exp(-rv1*x[1])*x[1]-v1*np.exp(-rv1*x[2])*x[2])/detaz+(D1*(x[3]-x[2])-D1*(x[2]-x[1]))/detaz**2)*deltat
    xn[3]=x[3]+(qushang*x[2]-qushang*x[3])+((qe_A*x[4]-qe_A*x[3]+v1*np.exp(-rv1*x[2])*x[2]-v1*np.exp(-rv1*x[3])*x[3])/detaz+(D1*(x[4]-x[3])-D1*(x[3]-x[2]))/detaz**2)*deltat
    xn[4]=x[4]+(qushang*x[3]-qushang*x[4])+((qe_A*x[5]-qe_A*x[4]+v1*np.exp(-rv1*x[3])*x[3]-v1*np.exp(-rv1*x[4])*x[4])/detaz+(D1*(x[5]-x[4])-D1*(x[4]-x[3]))/detaz**2)*deltat
    xn[5]=x[5]+(qushang*x[4]-qushang*x[5])+((qf_A*xf-qe_A*x[5]-qushang*x[5])/detaz)*deltat+(v1*np.exp(-rv1*x[4])*x[4]-v2*np.exp(-rv2*x[5])*x[5])+((D3*(x[6]-x[5])-D2*(x[5]-x[4]))/detaz**2)*deltat
    xn[6]=x[6]+((qushang*x[5]-qu_A*x[6]+v3*np.exp(-rv3*x[5])*x[5]-v3*np.exp(-rv3*x[6])*x[6])/detaz+(D3*(x[7]-x[6])-D3*(x[6]-x[5]))/detaz**2)*deltat
    xn[7]=x[7]+((qu_A*x[6]-qu_A*x[7]+v3*np.exp(-rv3*x[6])*x[6]-v3*np.exp(-rv3*x[7])*x[7])/detaz+(D3*(x[8]-x[7])-D3*(x[7]-x[6]))/detaz**2)*deltat
    xn[8]=x[8]+((qu_A*x[7]-qu_A*x[8]+v3*np.exp(-rv3*x[7])*x[7]-v4*np.exp(-rv4*x[8])*x[8])/detaz+(D4*(x[9]-x[8])-D3*(x[8]-x[7]))/detaz**2)*deltat
    xn[9]=x[9]+(qu_A*x[8]-qu_A*x[9])+((v4*np.exp(-rv4*x[8])*x[8])/detaz-D4*(x[9]-x[9])/detaz**2)*deltat
    return xn

# ====== 运行沉降模拟 ======
print("Running simulation...")
state = np.full(10, 1e-6, dtype=np.float64)
history = []
max_steps = 200
for s in range(1, max_steps + 1):
    state = step_fn(0.0, 40.0, 0.35, state)
    concs = np.array([max(0.0, d2c(x/1e6)) for x in state])
    c_uf = concs[-1]
    history.append((s, state.copy(), concs, c_uf))
    if abs(c_uf - 0.66) <= 0.01 * 0.66:
        print(f"Converged at step {s}, C_uf={c_uf:.4f}")
        break

n_frames = len(history)
print(f"Total frames: {n_frames}")

# ====== 创建动画 ======
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle('Thickener Natural Settling (Q_uf=0, Qf=40, Cf=0.35)', fontsize=14, fontweight='bold')

# Custom colormap: water -> slurry -> thick
cmap = colors.LinearSegmentedColormap.from_list('slurry', [
    '#e8f4f8',  # water (C=0)
    '#87ceeb',  # light
    '#4682b4',  # medium
    '#cd853f',  # thick
    '#8b4513',  # very thick (C>0.6)
])

# Thickener visual parameters
tank_width = 3.0
tank_height = 5.79
feed_level = 4.0
layer_height = tank_height / n

# --- Left: Thickener cross-section ---
ax1.set_xlim(-2, 2)
ax1.set_ylim(-0.5, tank_height + 1)
ax1.set_aspect('equal')
ax1.set_title('Thickener Cross-Section')
ax1.set_ylabel('Height (m)')
ax1.set_xticks([])

# Draw tank walls
ax1.plot([-tank_width/2, -tank_width/2], [0, tank_height], 'k-', linewidth=2)
ax1.plot([tank_width/2, tank_width/2], [0, tank_height], 'k-', linewidth=2)
ax1.plot([-tank_width/2, tank_width/2], [0, 0], 'k-', linewidth=2)

# Feed inlet
ax1.plot([-0.15, 0.15], [feed_level, feed_level], 'g-', linewidth=3)
ax1.text(0, feed_level + 0.15, 'Feed (Qf=40, Cf=0.35)', ha='center', fontsize=9, color='green')

# Underflow outlet label
ax1.text(0, -0.3, 'Underflow (Q_uf=0)', ha='center', fontsize=9, color='red')

# Layer rectangles (will be updated)
rects = []
texts = []
for i in range(n):
    y = i * layer_height
    rect = FancyBboxPatch((-tank_width/2, y), tank_width, layer_height,
                          boxstyle="round,pad=0.01", facecolor='#e8f4f8',
                          edgecolor='gray', linewidth=0.5)
    ax1.add_patch(rect)
    rects.append(rect)
    txt = ax1.text(0, y + layer_height/2, '', ha='center', va='center',
                   fontsize=7, fontweight='bold')
    texts.append(txt)

# Info text
info_text = ax1.text(0, tank_height + 0.5, '', ha='center', fontsize=10,
                     fontweight='bold', color='navy')

# --- Right: Concentration profile ---
ax2.set_title('Concentration Profile')
ax2.set_xlabel('Volume Fraction C')
ax2.set_ylabel('Layer')
ax2.set_xlim(0, 0.8)
ax2.set_ylim(-0.5, 10.5)
ax2.set_yticks(range(10))
ax2.set_yticklabels([f'L{i+1}' for i in range(10)])
ax2.axvline(x=0.66, color='red', linestyle='--', alpha=0.7, label='Target C_uf=0.66')
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)

profile_line, = ax2.plot([], [], 'bo-', linewidth=2, markersize=6)
profile_text = ax2.text(0.02, 9.5, '', fontsize=9, va='top')

def animate(frame_idx):
    step, state_raw, concs, c_uf = history[frame_idx]

    # Update thickener visual
    for i in range(n):
        c = min(concs[i], 0.7)
        rects[i].set_facecolor(cmap(c / 0.7))
        if concs[i] > 0.01:
            texts[i].set_text(f'C={concs[i]:.3f}')
        else:
            texts[i].set_text('')

    info_text.set_text(f'Step {step} min | C_uf = {c_uf:.4f} | Target: 0.66')

    # Update profile
    layers = np.arange(10)
    profile_line.set_data(concs, layers)
    profile_text.set_text(f'Step {step}\nC_uf = {c_uf:.4f}')

    return rects + texts + [info_text, profile_line, profile_text]

anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=100, blit=False)

output_path = 'F:/毕设/claude-code/settling_animation.gif'
anim.save(output_path, writer='pillow', fps=8)
print(f"Animation saved to: {output_path}")
