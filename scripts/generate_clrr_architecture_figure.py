"""Generate publication-quality vector PDF for Figure 1: CLRR Architecture.
Zero external dependencies other than matplotlib.
Clean, modern, aesthetic layout matching top-tier ACL standards.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set publication style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9

fig, ax = plt.subplots(figsize=(10, 4.2), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 42)
ax.axis('off')

# Color palette (Modern academic slate/teal/amber)
c_layer = '#E2E8F0'
c_layer_border = '#475569'
c_stream = '#0F172A'
c_skip = '#0D9488'      # Teal for CLRR
c_sg = '#DC2626'        # Red for stop-gradient
c_align = '#D97706'     # Amber for Latent Alignment
c_box_bg = '#F8FAFC'

# Subplot (a): Standard Sequential Transformer Stack
rect_a = patches.FancyBboxPatch((2, 2), 26, 38, boxstyle="round,pad=0.5", ec="#CBD5E1", fc=c_box_bg, ls="--")
ax.add_patch(rect_a)
ax.text(15, 38, "(a) Sequential Backbone", weight='bold', ha='center', fontsize=9.5, color='#1E293B')

for idx, y in enumerate([8, 16, 24, 32]):
    layer_name = f"Layer {idx+1}"
    lp = patches.FancyBboxPatch((5, y-2.2), 20, 4.4, boxstyle="round,pad=0.2", ec=c_layer_border, fc=c_layer)
    ax.add_patch(lp)
    ax.text(15, y, layer_name, ha='center', va='center', fontsize=8.5, weight='bold', color='#1E293B')
    if idx < 3:
        ax.annotate('', xy=(15, y+5.6), xytext=(15, y+2.4),
                    arrowprops=dict(arrowstyle="->", color=c_stream, lw=1.5))

ax.text(15, 4.2, r"Input $X = (x_1, \dots, x_L)$", ha='center', fontsize=8, style='italic', color='#475569')

# Subplot (b): Proposed Cross-Layer Residual Rewiring (CLRR)
rect_b = patches.FancyBboxPatch((32, 2), 34, 38, boxstyle="round,pad=0.5", ec=c_skip, fc='#F0FDFA', lw=1.2)
ax.add_patch(rect_b)
ax.text(49, 38, r"(b) Cross-Layer Residual Rewiring ($d=2$)", weight='bold', ha='center', fontsize=9.5, color='#0F766E')

for idx, y in enumerate([8, 16, 24, 32]):
    layer_name = f"Layer {idx+1}"
    lp = patches.FancyBboxPatch((36, y-2.2), 20, 4.4, boxstyle="round,pad=0.2", ec=c_skip, fc='#CCFBF1', lw=1.2)
    ax.add_patch(lp)
    ax.text(46, y, layer_name, ha='center', va='center', fontsize=8.5, weight='bold', color='#0F766E')
    if idx < 3:
        ax.annotate('', xy=(46, y+5.6), xytext=(46, y+2.4),
                    arrowprops=dict(arrowstyle="->", color=c_stream, lw=1.5))

# Skip connection from Layer 1 to Layer 3 (d=2)
# Arrow goes out of Layer 1, arches right, passes through Stop-Grad box, into Layer 3
ax.annotate('', xy=(59, 24), xytext=(56, 8),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.4", color=c_skip, lw=2.0))
# Stop-gradient label box
sg_box = patches.FancyBboxPatch((60.5, 15), 5.2, 3.2, boxstyle="round,pad=0.1", ec=c_sg, fc='#FEE2E2', lw=1.2)
ax.add_patch(sg_box)
ax.text(63.1, 16.6, r"sg$(\cdot)$", ha='center', va='center', fontsize=7.5, weight='bold', color=c_sg)
ax.text(63.1, 12.5, r"$+ \alpha \cdot h'_{i-2}$", ha='center', fontsize=7.5, weight='bold', color=c_skip)

# Formula banner at bottom of (b)
ax.text(49, 3.5, r"$h'_i = h_i + \alpha \cdot \mathrm{sg}(h'_{i-d})$  ($\Delta\theta = 0$)", 
        ha='center', fontsize=8.5, weight='bold', color='#0F766E',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=c_skip, lw=0.8))

# Subplot (c): Latent Semantic Regularization (LRA)
rect_c = patches.FancyBboxPatch((70, 2), 28, 38, boxstyle="round,pad=0.5", ec=c_align, fc='#FFFBEB', lw=1.2)
ax.add_patch(rect_c)
ax.text(84, 38, "(c) Latent Semantic Alignment", weight='bold', ha='center', fontsize=9.5, color='#B45309')

# Source Encoder output
src_box = patches.FancyBboxPatch((73, 24), 10, 4.5, boxstyle="round,pad=0.2", ec=c_skip, fc='#CCFBF1', lw=1)
ax.add_patch(src_box)
ax.text(78, 26.2, r"Source $H_X$" "\n(Amis)", ha='center', va='center', fontsize=7.5, weight='bold')

# Target Encoder output (Stop-gradient)
tgt_box = patches.FancyBboxPatch((85, 24), 11, 4.5, boxstyle="round,pad=0.2", ec=c_sg, fc='#FEE2E2', lw=1)
ax.add_patch(tgt_box)
ax.text(90.5, 26.2, r"sg$(H_Y)$" "\n(Mandarin)", ha='center', va='center', fontsize=7.5, weight='bold', color=c_sg)

# Pooling arrows down to embeddings
ax.annotate('', xy=(78, 17), xytext=(78, 23.8), arrowprops=dict(arrowstyle="->", color='#64748B', lw=1.2))
ax.annotate('', xy=(90.5, 17), xytext=(90.5, 23.8), arrowprops=dict(arrowstyle="->", color='#64748B', lw=1.2))

# Pooled vectors
z_src = patches.FancyBboxPatch((74, 13.5), 8, 3.2, boxstyle="round,pad=0.2", ec='#64748B', fc='white', lw=1)
ax.add_patch(z_src)
ax.text(78, 15.1, r"$\bar{z}_X$", ha='center', va='center', fontsize=8, weight='bold')

z_tgt = patches.FancyBboxPatch((86.5, 13.5), 8, 3.2, boxstyle="round,pad=0.2", ec='#64748B', fc='white', lw=1)
ax.add_patch(z_tgt)
ax.text(90.5, 15.1, r"$\bar{z}_Y$", ha='center', va='center', fontsize=8, weight='bold')

# Alignment objective
align_circ = patches.Circle((84.25, 8.5), radius=3.2, ec=c_align, fc='#FEF3C7', lw=1.5)
ax.add_patch(align_circ)
ax.text(84.25, 8.5, r"$\cos(\cdot)$", ha='center', va='center', fontsize=8, weight='bold', color=c_align)

ax.annotate('', xy=(82, 10.5), xytext=(78, 13.3), arrowprops=dict(arrowstyle="->", color=c_align, lw=1.2))
ax.annotate('', xy=(86.5, 10.5), xytext=(90.5, 13.3), arrowprops=dict(arrowstyle="->", color=c_align, lw=1.2))

ax.text(84, 3.5, r"$\mathcal{L}_{\mathrm{align}} = 1 - \bar{z}_X^\top \bar{z}_Y$", 
        ha='center', fontsize=8.5, weight='bold', color='#B45309',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=c_align, lw=0.8))

plt.tight_layout()
output_path = "docs/CLRR-paper/latex/figures/clrr_architecture.pdf"
plt.savefig(output_path, bbox_inches='tight', pad_inches=0.05)
print(f"Successfully generated clean vector PDF architecture diagram at: {output_path}")
