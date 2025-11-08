"""
Model Schema Demo Test Script
Test the model schema visualization feature
"""
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime


def draw_test_schema():
    """Test schema drawing with a sample 3-layer model."""
    
    # Create figure (larger to accommodate more information)
    fig = Figure(figsize=(10, 9), dpi=100, facecolor='white')
    ax = fig.add_subplot(111)
    ax.set_facecolor('white')
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)  # Extended y-axis for more info space
    
    # Sample model parameters
    num_layers = 3
    hidden_sizes = [50, 30, 20]
    input_size = 1
    output_size = 1
    
    # Colors
    input_color = '#3498db'
    hidden_color = '#2ecc71'
    output_color = '#e74c3c'
    text_color = 'black'
    line_color = '#7f8c8d'
    info_box_color = '#e8f4f8'
    metrics_box_color = '#fff4e6'
    
    # Calculate positions
    total_width = 8
    layer_spacing = total_width / (num_layers + 2)
    start_x = 1
    center_y = 7.5  # Moved up to make room for info below
    node_height = 2.25  # 50% increase from 1.5
    node_width = 1.2  # 50% increase from 0.8
    
    # Draw Input Layer (50% wider boxes)
    x = start_x
    ax.add_patch(plt.Rectangle((x - node_width/2, center_y - node_height/2), node_width, node_height,
                               facecolor=input_color, edgecolor=line_color, linewidth=2))
    ax.text(x, center_y, 'INPUT', ha='center', va='center', 
           fontsize=13, weight='bold', color='white')
    ax.text(x, center_y - node_height/2 - 0.3, f'Size: {input_size}',
           ha='center', va='top', fontsize=11, color=text_color)
    
    prev_x = x
    x += layer_spacing
    
    # Draw Hidden Layers (50% wider boxes)
    for i, hidden_size in enumerate(hidden_sizes, 1):
        # Draw arrow
        ax.annotate('', xy=(x - node_width/2 - 0.1, center_y), xytext=(prev_x + node_width/2 + 0.1, center_y),
                   arrowprops=dict(arrowstyle='->', lw=2, color=line_color))
        
        # Draw layer box
        ax.add_patch(plt.Rectangle((x - node_width/2, center_y - node_height/2), node_width, node_height,
                                  facecolor=hidden_color, edgecolor=line_color, linewidth=2))
        ax.text(x, center_y + 0.3, f'HIDDEN {i}', ha='center', va='center',
               fontsize=12, weight='bold', color='white')
        ax.text(x, center_y - 0.3, f'{hidden_size}', ha='center', va='center',
               fontsize=14, weight='bold', color='white')
        
        # Layer info below
        ax.text(x, center_y - node_height/2 - 0.3, f'Neurons: {hidden_size}',
               ha='center', va='top', fontsize=10, color=text_color)
        
        # Recurrent connection (self-loop) - larger circle
        circle = plt.Circle((x + node_width/2 + 0.3, center_y + node_height/2 + 0.3), 0.2,
                           fill=False, edgecolor=hidden_color, linewidth=2)
        ax.add_patch(circle)
        ax.annotate('', xy=(x + node_width/2 - 0.1, center_y + node_height/2), 
                   xytext=(x + node_width/2 + 0.3, center_y + node_height/2 + 0.15),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color=hidden_color))
        
        prev_x = x
        x += layer_spacing
    
    # Draw Output Layer (50% wider box)
    ax.annotate('', xy=(x - node_width/2 - 0.1, center_y), xytext=(prev_x + node_width/2 + 0.1, center_y),
               arrowprops=dict(arrowstyle='->', lw=2, color=line_color))
    
    ax.add_patch(plt.Rectangle((x - node_width/2, center_y - node_height/2), node_width, node_height,
                               facecolor=output_color, edgecolor=line_color, linewidth=2))
    ax.text(x, center_y, 'OUTPUT', ha='center', va='center',
           fontsize=13, weight='bold', color='white')
    ax.text(x, center_y - node_height/2 - 0.3, f'Size: {output_size}',
           ha='center', va='top', fontsize=11, color=text_color)
    
    # Add title (moved to new y position)
    title_text = f"RNN Architecture: {num_layers}-Layer (Deep/Stacked RNN)"
    ax.text(5, 13.5, title_text, ha='center', va='center',
           fontsize=16, weight='bold', color=text_color)
    
    # Add architecture info box
    info_text = f"Input: {input_size} → "
    info_text += " → ".join([str(s) for s in hidden_sizes])
    info_text += f" → Output: {output_size}"
    
    ax.text(5, 12.8, info_text, ha='center', va='center',
           fontsize=12, color=text_color,
           bbox=dict(boxstyle='round,pad=0.5', facecolor=info_box_color, alpha=0.8))
    
    # === MODEL PARAMETERS SECTION ===
    model_info = "MODEL PARAMETERS\n"
    model_info += "Total Parameters: 4,571 | Input Size: 1 | Output Size: 1\n"
    model_info += f"Hidden Layers: {num_layers} | Hidden Sizes: {hidden_sizes}\n"
    model_info += "Activation: tanh | Dropout: 0.3 | Optimizer: ADAM\n"
    model_info += "Sequence Length: 20 | Learning Rate: 0.0100 | LR Schedule: cosine\n"
    model_info += "Gradient Clip: 5.0 | Current LR: 0.008765"
    
    ax.text(5, 3.5, model_info, ha='center', va='center',
           fontsize=11, color=text_color,
           bbox=dict(boxstyle='round,pad=0.5', facecolor=info_box_color, alpha=0.8))
    
    # === TRAINING METRICS SECTION === (sample data with advanced metrics and data generation)
    metrics_info = "TRAINING METRICS & DATA GENERATION\n"
    metrics_info += "Data Type: Sine Wave | Samples: 1000 | Training Sequences: 800\n"
    metrics_info += "Frequency: 2.50 | Noise: 0.050 | Epochs: 100 | Final Loss: 0.002345\n"
    metrics_info += "Convergence: 87.5/100 | Plateau: No | Gradient: Healthy\n"
    metrics_info += "MSE: 0.002345 | RMSE: 0.048427 | MAE: 0.035678 | R²: 0.9245\n"
    metrics_info += "Grad Mean: 0.000123 | Grad Max: 0.012345 | Vanishing: 0 | Exploding: 0\n"
    metrics_info += "Weight Mean: 0.001234 | Weight Std: 0.234567 | Dead Neurons: 0\n"
    metrics_info += "Avg Loss: 0.003456 | Min Loss: 0.001234 | Loss Std: 0.000789"
    
    ax.text(5, 1.0, metrics_info, ha='center', va='center',
           fontsize=10, color=text_color,
           bbox=dict(boxstyle='round,pad=0.5', facecolor=metrics_box_color, alpha=0.8))
    
    # Add legend (moved to new position)
    ax.text(0.5, 12.2, '→ Forward Pass', ha='left', va='center',
           fontsize=10, color=text_color)
    ax.text(0.5, 11.8, '↻ Recurrent Connection (Time)', ha='left', va='center',
           fontsize=10, color=text_color)
    
    # Add timestamp at bottom left corner
    from datetime import datetime
    timestamp_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ax.text(0.3, 0.2, timestamp_text, ha='left', va='bottom',
           fontsize=9, color=text_color, style='italic',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.6))
    
    return fig


def main():
    """Main test function."""
    print("=" * 60)
    print("Model Schema Demo Test")
    print("=" * 60)
    
    print("\n📊 Creating test schema...")
    fig = draw_test_schema()
    
    print("✅ Schema created successfully!")
    
    # Save test schema
    output_file = "test_model_schema.png"
    fig.tight_layout()
    fig.savefig(output_file, facecolor='white', edgecolor='none', 
                bbox_inches='tight', dpi=150)
    
    print(f"\n💾 Schema saved to: {output_file}")
    print("\n🎨 Schema details:")
    print("   • Input Layer: Blue (1 neuron)")
    print("   • Hidden Layer 1: Green (50 neurons)")
    print("   • Hidden Layer 2: Green (30 neurons)")
    print("   • Hidden Layer 3: Green (20 neurons)")
    print("   • Output Layer: Red (1 neuron)")
    print("   • Total Parameters: 4,571")
    
    print("\n✨ Features demonstrated:")
    print("   ✓ Multi-layer architecture visualization")
    print("   ✓ Recurrent connections (self-loops)")
    print("   ✓ Forward pass arrows")
    print("   ✓ Layer information boxes")
    print("   ✓ Model parameters display")
    print("   ✓ Sample notes at bottom")
    
    print("\n" + "=" * 60)
    print("Test completed successfully! ✅")
    print(f"Check {output_file} to see the result.")
    print("=" * 60)
    
    # Display the figure
    plt.figure(figsize=(10, 7))
    plt.imshow(plt.imread(output_file))
    plt.axis('off')
    plt.title("Generated Model Schema", fontsize=16, weight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
