"""Generate simplified network topology visualization for 2x2 grid."""
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# Load topology
script_dir = os.path.dirname(os.path.abspath(__file__))
topo_path = os.path.join(script_dir, 'network_topology.json')
with open(topo_path, 'r', encoding='utf-8') as f:
    topo = json.load(f)

intersections = topo['intersections']

fig, ax = plt.subplots(1, 1, figsize=(10, 9))

# Scale: divide by 100 for display
SCALE = 100

for inter_id, inter_data in intersections.items():
    x = inter_data['position']['x'] / SCALE
    y = inter_data['position']['y'] / SCALE
    
    # Draw intersection node
    circle = plt.Circle((x, y), 0.5, color='#4A90D9', ec='#2C5F8A', linewidth=2, zorder=5)
    ax.add_patch(circle)
    
    # Label: intersection ID
    short_id = inter_id.replace('intersection_', '')
    ax.text(x, y, short_id, ha='center', va='center', fontsize=9,
            fontweight='bold', color='white', zorder=6)
    
    # Draw connections to neighbors
    for direction, neighbor in inter_data['neighbors'].items():
        n_id = neighbor.get('neighbor_id')
        if n_id is None:
            # Draw boundary stub extending outward
            stub_length = 1.5
            dx = {'E': stub_length, 'W': -stub_length, 'N': 0, 'S': 0}[direction]
            dy = {'E': 0, 'W': 0, 'N': stub_length, 'S': -stub_length}[direction]
            ax.plot([x, x + dx], [y, y + dy], color='#CCCCCC', lw=3, zorder=2, solid_capstyle='round')
            # Add small arrow
            ax.annotate('', xy=(x + dx*0.9, y + dy*0.9), xytext=(x + dx*0.5, y + dy*0.5),
                        arrowprops=dict(arrowstyle='->', color='#666666', lw=1.5), zorder=3)
            continue
        
        n_data = intersections[n_id]
        nx = n_data['position']['x'] / SCALE
        ny = n_data['position']['y'] / SCALE
        
        # Draw single bidirectional road
        road_dx = nx - x
        road_dy = ny - y
        road_len = (road_dx**2 + road_dy**2)**0.5
        if road_len == 0:
            continue
        
        # Draw line between intersections
        ax.plot([x, nx], [y, ny], color='#CCCCCC', lw=3, zorder=2, solid_capstyle='round')
        
        # Add directional arrow at midpoint
        mid_x = (x + nx) / 2
        mid_y = (y + ny) / 2
        arrow_dx = (nx - x) / road_len * 0.3
        arrow_dy = (ny - y) / road_len * 0.3
        ax.annotate('', xy=(mid_x + arrow_dx, mid_y + arrow_dy), 
                    xytext=(mid_x - arrow_dx, mid_y - arrow_dy),
                    arrowprops=dict(arrowstyle='->', color='#666666', lw=1.5),
                    zorder=3)
        
        # Road info label
        dist = neighbor['distance_m']
        speed = neighbor['speed_limit_mps']
        my_exit = neighbor['my_exit_direction']
        their_entry = neighbor['their_entry_direction']
        
        # Position label perpendicular to road
        perp_x = -road_dy / road_len * 0.8
        perp_y = road_dx / road_len * 0.8
        lx = mid_x + perp_x
        ly = mid_y + perp_y
        
        info_text = f'{dist}m | {speed:.1f}m/s\n{my_exit}→{their_entry}'
        ax.text(lx, ly, info_text, ha='center', va='center', fontsize=8,
                color='#555555', zorder=4,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='#DDDDDD', alpha=0.9))

# Legend
legend_elements = [
    mpatches.Patch(color='#4A90D9', label='Intersection'),
    plt.Line2D([0], [0], color='#CCCCCC', lw=3, label='Road segment'),
    plt.Line2D([0], [0], color='#666666', lw=1.5, marker='>', label='Traffic direction'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

ax.set_xlim(-1.5, 5.5)
ax.set_ylim(-1.5, 9.5)
ax.set_aspect('equal')
ax.set_xlabel('X (×100m)', fontsize=10)
ax.set_ylabel('Y (×100m)', fontsize=10)
ax.set_title('2×2 Grid Network Topology', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.2, linestyle='--')

plt.tight_layout()
out_path = os.path.join(script_dir, 'network_topology.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved topology diagram to: {out_path}")
