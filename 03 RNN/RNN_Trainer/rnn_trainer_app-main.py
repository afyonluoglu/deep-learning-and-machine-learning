"""
RNN Training and Visualization Application
Professional CustomTkinter-based GUI for RNN learning
"""
import customtkinter as ctk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import threading
import time
from tkinter import filedialog, messagebox
import os
import json
from datetime import datetime

from rnn_model import RNNModel
from data_generator import DataGenerator


class RNNTrainerApp(ctk.CTk):
    """Main application class for RNN Trainer."""
    
    def __init__(self):
        super().__init__()
        
        # Configure window
        self.title("RNN Trainer - Recurrent Neural Network Learning Platform")
        
        # Set theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Set window size and center
        window_width = 1400
        window_height = 900
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Initialize variables
        self.model = None
        self.training_data = None
        self.training_targets = None
        self.test_data = None
        self.current_data_raw = None
        self.custom_data_raw = None  # For custom loaded data
        self.custom_data_normalized = None
        self.data_min = 0
        self.data_max = 1
        self.is_training = False
        self.training_thread = None
        
        # Create outputs directory
        self.outputs_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        os.makedirs(self.outputs_dir, exist_ok=True)
        
        # Create UI
        self.create_widgets()
        
        # Update status
        self.update_status("Ready. Configure parameters and generate data to start.")
    
    def create_widgets(self):
        """Create all UI widgets."""
        # Main container with grid
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Left panel - Controls
        self.create_control_panel()
        
        # Right panel - Visualization
        self.create_visualization_panel()
        
        # Bottom status bar
        self.create_status_bar()
    
    def create_control_panel(self):
        """Create left control panel."""
        control_frame = ctk.CTkScrollableFrame(self, width=350, corner_radius=10)
        control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Title
        title_label = ctk.CTkLabel(
            control_frame, 
            text="RNN Training Controls",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.pack(pady=10)
        
        # Model Parameters Section
        self.create_model_parameters_section(control_frame)
        
        # Data Generation Section
        self.create_data_generation_section(control_frame)
        
        # Training Section
        self.create_training_section(control_frame)
        
        # Model Management Section
        self.create_model_management_section(control_frame)
        
        # Help Button
        help_btn = ctk.CTkButton(
            control_frame,
            text="📖 Help & Documentation",
            command=self.show_help,
            height=40,
            font=ctk.CTkFont(size=14)
        )
        help_btn.pack(pady=10, padx=10, fill="x")
    
    def create_model_parameters_section(self, parent):
        """Create model parameters section."""
        frame = ctk.CTkFrame(parent, corner_radius=10)
        frame.pack(pady=10, padx=10, fill="x")
        
        label = ctk.CTkLabel(
            frame,
            text="🔧 Model Parameters",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        label.pack(pady=5)
        
        # Hidden Size
        self.hidden_size_label = ctk.CTkLabel(frame, text="Hidden Units (Neurons): 20")
        self.hidden_size_label.pack()
        self.hidden_size_slider = ctk.CTkSlider(
            frame, from_=5, to=100, number_of_steps=19,
            command=lambda v: self.hidden_size_label.configure(text=f"Hidden Units (Neurons): {int(v)}")
        )
        self.hidden_size_slider.set(20)
        self.hidden_size_slider.pack(pady=5, padx=10, fill="x")
        
        # Learning Rate
        self.lr_label = ctk.CTkLabel(frame, text="Learning Rate: 0.01")
        self.lr_label.pack()
        self.lr_slider = ctk.CTkSlider(
            frame, from_=0.001, to=0.1, number_of_steps=99,
            command=lambda v: self.lr_label.configure(text=f"Learning Rate: {v:.4f}")
        )
        self.lr_slider.set(0.01)
        self.lr_slider.pack(pady=5, padx=10, fill="x")
        
        # Sequence Length
        self.seq_len_label = ctk.CTkLabel(frame, text="Sequence Length: 20")
        self.seq_len_label.pack()
        self.seq_len_slider = ctk.CTkSlider(
            frame, from_=5, to=50, number_of_steps=9,
            command=lambda v: self.seq_len_label.configure(text=f"Sequence Length: {int(v)}")
        )
        self.seq_len_slider.set(20)
        self.seq_len_slider.pack(pady=5, padx=10, fill="x")
        
        # Activation Function
        ctk.CTkLabel(frame, text="Activation Function:").pack()
        self.activation_var = ctk.StringVar(value="tanh")
        activation_menu = ctk.CTkOptionMenu(
            frame,
            values=["tanh", "relu"],
            variable=self.activation_var
        )
        activation_menu.pack(pady=5, padx=10, fill="x")
        
        # Dropout Rate
        self.dropout_label = ctk.CTkLabel(frame, text="Dropout Rate: 0.0 (Off)")
        self.dropout_label.pack()
        self.dropout_slider = ctk.CTkSlider(
            frame, from_=0.0, to=0.9, number_of_steps=18,
            command=lambda v: self.dropout_label.configure(
                text=f"Dropout Rate: {v:.2f} {'(Off)' if v < 0.01 else '(Regularization)'}"
            )
        )
        self.dropout_slider.set(0.0)
        self.dropout_slider.pack(pady=5, padx=10, fill="x")
        
        # Number of Hidden Layers (NEW!)
        self.num_layers_label = ctk.CTkLabel(frame, text="Hidden Layers: 1 (Single)")
        self.num_layers_label.pack(pady=(10, 0))
        self.num_layers_slider = ctk.CTkSlider(
            frame, from_=1, to=5, number_of_steps=4,
            command=lambda v: self.update_layers_label(int(v))
        )
        self.num_layers_slider.set(1)
        self.num_layers_slider.pack(pady=5, padx=10, fill="x")
        
        # Hidden sizes for each layer (NEW!)
        ctk.CTkLabel(frame, text="Layer Sizes (comma-separated):").pack()
        self.layer_sizes_entry = ctk.CTkEntry(frame, placeholder_text="e.g., 30,20,10")
        self.layer_sizes_entry.pack(pady=5, padx=10, fill="x")
        
        # Info label
        self.layer_info_label = ctk.CTkLabel(
            frame, 
            text="💡 Leave empty to use Hidden Units for all layers",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        self.layer_info_label.pack(pady=2)
        
        # Optimizer Type
        ctk.CTkLabel(frame, text="Optimizer:").pack(pady=(10, 0))
        self.optimizer_var = ctk.StringVar(value="adam")
        optimizer_menu = ctk.CTkOptionMenu(
            frame,
            values=["sgd", "momentum", "adam", "rmsprop"],
            variable=self.optimizer_var
        )
        optimizer_menu.pack(pady=5, padx=10, fill="x")
        
        # Learning Rate Schedule (NEW!)
        ctk.CTkLabel(frame, text="LR Schedule:").pack()
        self.lr_schedule_var = ctk.StringVar(value="constant")
        lr_schedule_menu = ctk.CTkOptionMenu(
            frame,
            values=["constant", "step", "exponential", "cosine", "reduce_on_plateau", "cyclical", "warmup_decay"],
            variable=self.lr_schedule_var
        )
        lr_schedule_menu.pack(pady=5, padx=10, fill="x")
        
        # Initialize Model Button
        init_btn = ctk.CTkButton(
            frame,
            text="Initialize Model",
            command=self.initialize_model,
            fg_color="green",
            hover_color="darkgreen"
        )
        init_btn.pack(pady=10, padx=10, fill="x")
    
    def create_data_generation_section(self, parent):
        """Create data generation section."""
        frame = ctk.CTkFrame(parent, corner_radius=10)
        frame.pack(pady=10, padx=10, fill="x")
        
        label = ctk.CTkLabel(
            frame,
            text="📊 Data Generation",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        label.pack(pady=5)
        
        # Wave Type
        ctk.CTkLabel(frame, text="Wave Type:").pack()
        self.wave_type_var = ctk.StringVar(value="Sine Wave")
        wave_menu = ctk.CTkOptionMenu(
            frame,
            values=[
                "Sine Wave", "Cosine Wave", "Square Wave",
                "Sawtooth Wave", "Triangular Wave", "Mixed Waves",
                "Exponential", "Polynomial", "Random Walk",
                "ARMA", "Damped Oscillation"
            ],
            variable=self.wave_type_var
        )
        wave_menu.pack(pady=5, padx=10, fill="x")
        
        # Number of Samples
        self.n_samples_label = ctk.CTkLabel(frame, text="Samples: 500")
        self.n_samples_label.pack()
        self.n_samples_slider = ctk.CTkSlider(
            frame, from_=100, to=2000, number_of_steps=19,
            command=lambda v: self.n_samples_label.configure(text=f"Samples: {int(v)}")
        )
        self.n_samples_slider.set(500)
        self.n_samples_slider.pack(pady=5, padx=10, fill="x")
        
        # Frequency
        self.frequency_label = ctk.CTkLabel(frame, text="Frequency: 1.0")
        self.frequency_label.pack()
        self.frequency_slider = ctk.CTkSlider(
            frame, from_=0.1, to=5.0, number_of_steps=49,
            command=lambda v: self.frequency_label.configure(text=f"Frequency: {v:.1f}")
        )
        self.frequency_slider.set(1.0)
        self.frequency_slider.pack(pady=5, padx=10, fill="x")
        
        # Noise Level
        self.noise_label = ctk.CTkLabel(frame, text="Noise Level: 0.05")
        self.noise_label.pack()
        self.noise_slider = ctk.CTkSlider(
            frame, from_=0.0, to=0.5, number_of_steps=50,
            command=lambda v: self.noise_label.configure(text=f"Noise Level: {v:.2f}")
        )
        self.noise_slider.set(0.05)
        self.noise_slider.pack(pady=5, padx=10, fill="x")
        
        # Generate Button
        generate_btn = ctk.CTkButton(
            frame,
            text="Generate Data",
            command=self.generate_data,
            fg_color="purple",
            hover_color="darkviolet"
        )
        generate_btn.pack(pady=10, padx=10, fill="x")
    
    def create_training_section(self, parent):
        """Create training section."""
        frame = ctk.CTkFrame(parent, corner_radius=10)
        frame.pack(pady=10, padx=10, fill="x")
        
        label = ctk.CTkLabel(
            frame,
            text="🎓 Training",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        label.pack(pady=5)
        
        # Epochs
        self.epochs_label = ctk.CTkLabel(frame, text="Epochs: 50")
        self.epochs_label.pack()
        self.epochs_slider = ctk.CTkSlider(
            frame, from_=10, to=500, number_of_steps=49,
            command=lambda v: self.epochs_label.configure(text=f"Epochs: {int(v)}")
        )
        self.epochs_slider.set(50)
        self.epochs_slider.pack(pady=5, padx=10, fill="x")
        
        # Train/Stop Buttons
        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.pack(pady=5, padx=10, fill="x")
        
        self.train_btn = ctk.CTkButton(
            btn_frame,
            text="▶ Start Training",
            command=self.start_training,
            fg_color="orange",
            hover_color="darkorange"
        )
        self.train_btn.pack(side="left", expand=True, padx=5)
        
        self.stop_btn = ctk.CTkButton(
            btn_frame,
            text="⏹ Stop",
            command=self.stop_training,
            fg_color="red",
            hover_color="darkred",
            state="disabled"
        )
        self.stop_btn.pack(side="right", expand=True, padx=5)
        
        # Test Prediction Button
        test_btn = ctk.CTkButton(
            frame,
            text="🔮 Test Prediction",
            command=self.test_prediction,
            fg_color="teal",
            hover_color="darkcyan"
        )
        test_btn.pack(pady=10, padx=10, fill="x")
        
        # Advanced Metrics Display (NEW!)
        metrics_label = ctk.CTkLabel(
            frame,
            text="📊 Advanced Metrics",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        metrics_label.pack(pady=(15, 5))
        
        # Metrics text display
        self.metrics_text = ctk.CTkTextbox(frame, height=120, width=300)
        self.metrics_text.pack(pady=5, padx=10, fill="x")
        self.metrics_text.insert("1.0", "Train model to see metrics...")
        self.metrics_text.configure(state="disabled")
        
        # Gradient Health Display (NEW!)
        gradient_label = ctk.CTkLabel(
            frame,
            text="🔍 Gradient Health",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        gradient_label.pack(pady=(10, 5))
        
        self.gradient_status_label = ctk.CTkLabel(
            frame,
            text="Status: Not trained",
            font=ctk.CTkFont(size=12)
        )
        self.gradient_status_label.pack()
        
        # Convergence Score Display (NEW!)
        convergence_label = ctk.CTkLabel(
            frame,
            text="⚡ Training Status",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        convergence_label.pack(pady=(10, 5))
        
        self.convergence_label = ctk.CTkLabel(
            frame,
            text="Convergence: --/100",
            font=ctk.CTkFont(size=12)
        )
        self.convergence_label.pack()
        
        self.plateau_label = ctk.CTkLabel(
            frame,
            text="Plateau: --",
            font=ctk.CTkFont(size=12)
        )
        self.plateau_label.pack()
    
    def create_model_management_section(self, parent):
        """Create model management section."""
        frame = ctk.CTkFrame(parent, corner_radius=10)
        frame.pack(pady=10, padx=10, fill="x")
        
        label = ctk.CTkLabel(
            frame,
            text="💾 Model Management",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        label.pack(pady=5)
        
        # Save Model
        save_btn = ctk.CTkButton(
            frame,
            text="Save Model",
            command=self.save_model,
            fg_color="blue",
            hover_color="darkblue"
        )
        save_btn.pack(pady=5, padx=10, fill="x")
        
        # Load Model
        load_btn = ctk.CTkButton(
            frame,
            text="Load Model",
            command=self.load_model,
            fg_color="indigo",
            hover_color="purple"
        )
        load_btn.pack(pady=5, padx=10, fill="x")
        
        # Model Info
        info_btn = ctk.CTkButton(
            frame,
            text="Model Info",
            command=self.show_model_info
        )
        info_btn.pack(pady=5, padx=10, fill="x")
        
        # Model Schema Button (NEW!)
        schema_btn = ctk.CTkButton(
            frame,
            text="📊 Model Schema",
            command=self.show_model_schema,
            fg_color="darkorange",
            hover_color="orange"
        )
        schema_btn.pack(pady=5, padx=10, fill="x")
        
        # Custom Data Section
        custom_label = ctk.CTkLabel(
            frame,
            text="📁 Custom Data",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        custom_label.pack(pady=(10, 5))
        
        # Load Custom Data Button
        load_custom_btn = ctk.CTkButton(
            frame,
            text="📂 Load CSV Data",
            command=self.load_custom_data,
            fg_color="darkgreen",
            hover_color="green"
        )
        load_custom_btn.pack(pady=5, padx=10, fill="x")
        
        # Future Prediction Button
        predict_future_btn = ctk.CTkButton(
            frame,
            text="🔮 Predict Future Values",
            command=self.predict_future_values,
            fg_color="purple",
            hover_color="darkviolet"
        )
        predict_future_btn.pack(pady=5, padx=10, fill="x")
    
    def create_visualization_panel(self):
        """Create right visualization panel."""
        viz_frame = ctk.CTkFrame(self, corner_radius=10)
        viz_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        viz_frame.grid_rowconfigure(0, weight=1)
        viz_frame.grid_rowconfigure(1, weight=1)
        viz_frame.grid_columnconfigure(0, weight=1)
        
        # Top: Data and Prediction Plot
        self.create_data_plot(viz_frame)
        
        # Bottom: Loss Plot
        self.create_loss_plot(viz_frame)
    
    def create_data_plot(self, parent):
        """Create data and prediction plot with toolbar."""
        frame = ctk.CTkFrame(parent, corner_radius=10)
        frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        # Title and save button container
        header_frame = ctk.CTkFrame(frame, fg_color="transparent")
        header_frame.pack(fill="x", pady=5)
        
        title = ctk.CTkLabel(
            header_frame,
            text="Time Series Data & Predictions",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title.pack(side="left", padx=10)
        
        # Save button
        save_btn = ctk.CTkButton(
            header_frame,
            text="💾 Save Graph",
            command=self.save_data_plot,
            width=100,
            height=28
        )
        save_btn.pack(side="right", padx=10)
        
        # Create matplotlib figure
        self.data_fig = Figure(figsize=(8, 4), dpi=100)
        self.data_ax = self.data_fig.add_subplot(111)
        self.data_ax.set_xlabel("Time Step")
        self.data_ax.set_ylabel("Value")
        self.data_ax.grid(True, alpha=0.3)
        
        # Canvas
        self.data_canvas = FigureCanvasTkAgg(self.data_fig, frame)
        self.data_canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        
        # Toolbar for zoom/pan
        toolbar_frame = ctk.CTkFrame(frame, fg_color="transparent")
        toolbar_frame.pack(fill="x", padx=5, pady=5)
        self.data_toolbar = NavigationToolbar2Tk(self.data_canvas, toolbar_frame)
    
    def create_loss_plot(self, parent):
        """Create loss plot with toolbar."""
        frame = ctk.CTkFrame(parent, corner_radius=10)
        frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        
        # Title and save button container
        header_frame = ctk.CTkFrame(frame, fg_color="transparent")
        header_frame.pack(fill="x", pady=5)
        
        title = ctk.CTkLabel(
            header_frame,
            text="Training Loss",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title.pack(side="left", padx=10)
        
        # Save button
        save_btn = ctk.CTkButton(
            header_frame,
            text="💾 Save Graph",
            command=self.save_loss_plot,
            width=100,
            height=28
        )
        save_btn.pack(side="right", padx=10)
        
        # Create matplotlib figure
        self.loss_fig = Figure(figsize=(8, 4), dpi=100)
        self.loss_ax = self.loss_fig.add_subplot(111)
        self.loss_ax.set_xlabel("Iteration")
        self.loss_ax.set_ylabel("Loss")
        self.loss_ax.set_yscale("log")
        self.loss_ax.grid(True, alpha=0.3)
        
        # Canvas
        self.loss_canvas = FigureCanvasTkAgg(self.loss_fig, frame)
        self.loss_canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        
        # Toolbar for zoom/pan
        toolbar_frame = ctk.CTkFrame(frame, fg_color="transparent")
        toolbar_frame.pack(fill="x", padx=5, pady=5)
        self.loss_toolbar = NavigationToolbar2Tk(self.loss_canvas, toolbar_frame)
        self.loss_canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
    
    def create_status_bar(self):
        """Create status bar."""
        self.status_bar = ctk.CTkLabel(
            self,
            text="Ready",
            anchor="w",
            height=30,
            corner_radius=5,
            fg_color=("gray80", "gray20")
        )
        self.status_bar.grid(row=1, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="ew")
    
    def update_status(self, message):
        """Update status bar message."""
        self.status_bar.configure(text=f"Status: {message}")
        self.update()
    
    def initialize_model(self):
        """Initialize RNN model with current parameters."""
        try:
            hidden_size = int(self.hidden_size_slider.get())
            learning_rate = self.lr_slider.get()
            sequence_length = int(self.seq_len_slider.get())
            activation = self.activation_var.get()
            dropout_rate = self.dropout_slider.get()
            optimizer_type = self.optimizer_var.get()
            lr_schedule = self.lr_schedule_var.get()
            num_layers = int(self.num_layers_slider.get())
            
            # Parse layer sizes if provided
            hidden_sizes = None
            layer_sizes_text = self.layer_sizes_entry.get().strip()
            if layer_sizes_text:
                try:
                    hidden_sizes = [int(x.strip()) for x in layer_sizes_text.split(',')]
                    if len(hidden_sizes) < num_layers:
                        hidden_sizes = hidden_sizes + [hidden_size] * (num_layers - len(hidden_sizes))
                except ValueError:
                    messagebox.showwarning("Warning", "Invalid layer sizes format. Using Hidden Units for all layers.")
                    hidden_sizes = None
            
            self.model = RNNModel(
                input_size=1,
                hidden_size=hidden_size,
                output_size=1,
                learning_rate=learning_rate,
                sequence_length=sequence_length,
                activation=activation,
                dropout_rate=dropout_rate,
                optimizer_type=optimizer_type,
                lr_schedule=lr_schedule,
                num_layers=num_layers,
                hidden_sizes=hidden_sizes
            )
            
            # Build architecture string
            arch_str = " → ".join([str(s) for s in self.model.hidden_sizes]) if num_layers > 1 else str(hidden_size)
            layer_type = "Multi-layer" if num_layers > 1 else "Single-layer"
            
            self.update_status(f"Model initialized: {layer_type} RNN ({arch_str}), "
                             f"lr={learning_rate:.4f}, seq_len={sequence_length}, "
                             f"activation={activation}, dropout={dropout_rate:.2f}, "
                             f"optimizer={optimizer_type}, lr_schedule={lr_schedule}")
            
            # Reset metrics display
            self.metrics_text.configure(state="normal")
            self.metrics_text.delete("1.0", "end")
            self.metrics_text.insert("1.0", "Model initialized. Train to see metrics...")
            self.metrics_text.configure(state="disabled")
            
            self.gradient_status_label.configure(text="Status: Not trained")
            self.convergence_label.configure(text="Convergence: --/100")
            self.plateau_label.configure(text="Plateau: --")
            
            messagebox.showinfo("Success", f"Model initialized successfully!\n"
                                         f"Architecture: {layer_type} RNN\n"
                                         f"Hidden Units (neurons): {arch_str}\n"
                                         f"Optimizer: {optimizer_type.upper()}\n"
                                         f"LR Schedule: {lr_schedule}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize model: {str(e)}")
    
    def generate_data(self):
        """Generate training data based on selected parameters."""
        try:
            wave_type = self.wave_type_var.get()
            n_samples = int(self.n_samples_slider.get())
            frequency = self.frequency_slider.get()
            noise_level = self.noise_slider.get()
            
            generator = DataGenerator()
            
            # Generate data based on type
            if wave_type == "Sine Wave":
                data = generator.generate_sine_wave(n_samples, frequency, 1.0, noise_level)
            elif wave_type == "Cosine Wave":
                data = generator.generate_cosine_wave(n_samples, frequency, 1.0, noise_level)
            elif wave_type == "Square Wave":
                data = generator.generate_square_wave(n_samples, frequency, 1.0, noise_level)
            elif wave_type == "Sawtooth Wave":
                data = generator.generate_sawtooth_wave(n_samples, frequency, 1.0, noise_level)
            elif wave_type == "Triangular Wave":
                data = generator.generate_triangular_wave(n_samples, frequency, 1.0, noise_level)
            elif wave_type == "Mixed Waves":
                data = generator.generate_mixed_waves(n_samples, [1.0, 2.0], [1.0, 0.5], noise_level)
            elif wave_type == "Exponential":
                data = generator.generate_exponential(n_samples, 0.01, noise_level)
            elif wave_type == "Polynomial":
                data = generator.generate_polynomial(n_samples, [0, 0.5, 0.1], noise_level)
            elif wave_type == "Random Walk":
                data = generator.generate_random_walk(n_samples, 0.1)
            elif wave_type == "ARMA":
                data = generator.generate_arma(n_samples, [0.5], [0.3], noise_level)
            elif wave_type == "Damped Oscillation":
                data = generator.generate_damped_oscillation(n_samples, frequency, 0.1, 1.0, noise_level)
            
            # Store raw data
            self.current_data_raw = data.copy()
            
            # Normalize
            normalized_data, self.data_min, self.data_max = generator.normalize_data(data)
            
            # Create sequences
            seq_len = int(self.seq_len_slider.get())
            X, y = generator.create_sequences(normalized_data, seq_len)
            
            # Split into train and test
            split_idx = int(0.8 * len(X))
            self.training_data = X[:split_idx]
            self.training_targets = y[:split_idx]
            self.test_data = normalized_data
            
            # Plot data
            self.plot_data()
            
            self.update_status(f"Generated {wave_type}: {n_samples} samples, "
                             f"{len(self.training_data)} training sequences")
            
            messagebox.showinfo("Success", f"Data generated successfully!\n"
                                         f"Training sequences: {len(self.training_data)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate data: {str(e)}")
    
    def plot_data(self):
        """Plot current data."""
        if self.current_data_raw is None:
            return
        
        self.data_ax.clear()
        self.data_ax.plot(self.current_data_raw, 'b-', label='Data', linewidth=1.5)
        self.data_ax.set_xlabel("Time Step")
        self.data_ax.set_ylabel("Value")
        self.data_ax.legend()
        self.data_ax.grid(True, alpha=0.3)
        self.data_canvas.draw()
    
    def start_training(self):
        """Start model training in separate thread."""
        if self.model is None:
            messagebox.showwarning("Warning", "Please initialize the model first!")
            return
        
        if self.training_data is None:
            messagebox.showwarning("Warning", "Please generate training data first!")
            return
        
        if self.is_training:
            messagebox.showwarning("Warning", "Training is already in progress!")
            return
        
        self.is_training = True
        self.train_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        
        # Start training thread
        self.training_thread = threading.Thread(target=self.train_model, daemon=True)
        self.training_thread.start()
    
    def train_model(self):
        """Train the model."""
        try:
            epochs = int(self.epochs_slider.get())
            
            # Reshape data for training
            X_train = self.training_data.reshape(len(self.training_data), -1, 1)
            y_train = self.training_targets.reshape(len(self.training_targets), -1, 1)
            
            for epoch in range(epochs):
                if not self.is_training:
                    break
                
                # Train one epoch
                avg_loss = self.model.train_epoch(
                    X_train.reshape(-1, 1),
                    y_train.reshape(-1, 1)
                )
                
                # Update plots and metrics every 5 epochs
                if epoch % 5 == 0:
                    self.after(0, self.update_training_plots)
                    self.after(0, self.update_advanced_metrics)  # NEW!
                    
                    # Get current LR if available
                    current_lr = "N/A"
                    if hasattr(self.model, 'get_parameters'):
                        params = self.model.get_parameters()
                        current_lr = f"{params.get('current_lr', self.model.learning_rate):.6f}"
                    
                    self.after(0, self.update_status, 
                             f"Training... Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, LR: {current_lr}")
                
                time.sleep(0.01)  # Small delay to prevent UI freezing
            
            self.after(0, self.training_complete)
            
        except Exception as e:
            self.after(0, messagebox.showerror, "Error", f"Training failed: {str(e)}")
            self.after(0, self.training_complete)
    
    def update_training_plots(self):
        """Update plots during training."""
        if self.model and len(self.model.loss_history) > 0:
            # Update loss plot
            self.loss_ax.clear()
            self.loss_ax.plot(self.model.loss_history, 'r-', linewidth=1)
            self.loss_ax.set_xlabel("Iteration")
            self.loss_ax.set_ylabel("Loss")
            self.loss_ax.set_yscale("log")
            self.loss_ax.grid(True, alpha=0.3)
            self.loss_canvas.draw()
    
    def update_advanced_metrics(self):
        """Update advanced metrics display (NEW!)"""
        if not self.model:
            return
        
        try:
            # Get comprehensive metrics if available
            if hasattr(self.model, 'get_comprehensive_metrics') and self.training_data is not None:
                X_train = self.training_data.reshape(len(self.training_data), -1, 1)
                y_train = self.training_targets.reshape(len(self.training_targets), -1, 1)
                
                metrics = self.model.get_comprehensive_metrics(
                    X_train.reshape(-1, 1), 
                    y_train.reshape(-1, 1)
                )
                
                # Update metrics text
                self.metrics_text.configure(state="normal")
                self.metrics_text.delete("1.0", "end")
                
                metrics_str = f"MSE:   {metrics.get('mse', 0):.6f}\n"
                metrics_str += f"RMSE:  {metrics.get('rmse', 0):.6f}\n"
                metrics_str += f"MAE:   {metrics.get('mae', 0):.6f}\n"
                metrics_str += f"MAPE:  {metrics.get('mape', 0):.2f}%\n"
                metrics_str += f"R²:    {metrics.get('r2', 0):.4f}\n"
                
                # Interpret R²
                r2 = metrics.get('r2', 0)
                if r2 > 0.9:
                    metrics_str += "Quality: ✅ Excellent"
                elif r2 > 0.7:
                    metrics_str += "Quality: ✅ Good"
                elif r2 > 0.5:
                    metrics_str += "Quality: ⚠️  Moderate"
                else:
                    metrics_str += "Quality: ❌ Poor"
                
                self.metrics_text.insert("1.0", metrics_str)
                self.metrics_text.configure(state="disabled")
            
            # Get gradient health if available
            if hasattr(self.model, 'get_gradient_health'):
                grad_health = self.model.get_gradient_health()
                status = grad_health.get('status', 'Unknown')
                
                # Color code status
                if 'healthy' in status.lower():
                    status_text = f"✅ {status}"
                elif 'vanishing' in status.lower():
                    status_text = f"⚠️  {status}"
                elif 'exploding' in status.lower():
                    status_text = f"❌ {status}"
                else:
                    status_text = status
                
                self.gradient_status_label.configure(text=f"Status: {status_text}")
            
            # Get training status if available
            if hasattr(self.model, 'get_training_status'):
                training_status = self.model.get_training_status()
                convergence = training_status.get('convergence_score', 0)
                plateau = training_status.get('plateau_detected', False)
                
                self.convergence_label.configure(text=f"Convergence: {convergence:.1f}/100")
                
                if plateau:
                    self.plateau_label.configure(text="Plateau: ⚠️  Detected")
                else:
                    self.plateau_label.configure(text="Plateau: ✅ No")
                
        except Exception as e:
            # Silently fail if metrics not available
            pass
    
    def training_complete(self):
        """Called when training is complete."""
        self.is_training = False
        self.train_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        
        # Final metrics update
        self.update_advanced_metrics()
        
        self.update_status("Training complete!")
        messagebox.showinfo("Training Complete", "Model training finished successfully!")
    
    def stop_training(self):
        """Stop training."""
        self.is_training = False
        self.update_status("Training stopped by user.")
    
    def test_prediction(self):
        """Test model predictions."""
        if self.model is None:
            messagebox.showwarning("Warning", "Please initialize and train the model first!")
            return
        
        if self.test_data is None:
            messagebox.showwarning("Warning", "Please generate data first!")
            return
        
        try:
            # Make predictions
            predictions = self.model.predict(self.test_data)
            
            # Denormalize
            generator = DataGenerator()
            predictions_denorm = generator.denormalize_data(
                predictions, self.data_min, self.data_max
            )
            
            # Plot
            self.data_ax.clear()
            self.data_ax.plot(self.current_data_raw, 'b-', label='Actual', linewidth=1.5, alpha=0.7)
            self.data_ax.plot(predictions_denorm, 'r--', label='Predicted', linewidth=1.5)
            self.data_ax.set_xlabel("Time Step")
            self.data_ax.set_ylabel("Value")
            self.data_ax.legend()
            self.data_ax.grid(True, alpha=0.3)
            self.data_ax.set_title("Model Predictions vs Actual Data")
            self.data_canvas.draw()
            
            # Calculate MSE
            mse = np.mean((self.current_data_raw[:len(predictions_denorm)] - predictions_denorm) ** 2)
            
            self.update_status(f"Prediction complete. MSE: {mse:.6f}")
            messagebox.showinfo("Prediction", f"Prediction complete!\nMSE: {mse:.6f}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
    def save_model(self):
        """Save model to file."""
        if self.model is None:
            messagebox.showwarning("Warning", "No model to save!")
            return
        
        try:
            filepath = filedialog.asksaveasfilename(
                defaultextension=".pkl",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
                initialdir=os.path.dirname(__file__)
            )
            
            if filepath:
                self.model.save_model(filepath)
                
                # Also save configuration
                config_path = filepath.replace('.pkl', '_config.json')
                config = {
                    'data_min': float(self.data_min),
                    'data_max': float(self.data_max),
                    'wave_type': self.wave_type_var.get()
                }
                
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=4)
                
                messagebox.showinfo("Success", f"Model saved to:\n{filepath}")
                self.update_status(f"Model saved: {os.path.basename(filepath)}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save model: {str(e)}")
    
    def load_model(self):
        """Load model from file."""
        try:
            filepath = filedialog.askopenfilename(
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
                initialdir=os.path.dirname(__file__)
            )
            
            if filepath:
                self.model = RNNModel.load_model(filepath)
                
                # Load configuration if exists
                config_path = filepath.replace('.pkl', '_config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    self.data_min = config.get('data_min', 0)
                    self.data_max = config.get('data_max', 1)
                
                # Update UI sliders and labels
                params = self.model.get_parameters()
                
                # Update sliders
                self.hidden_size_slider.set(params['hidden_size'])
                self.lr_slider.set(params['learning_rate'])
                self.seq_len_slider.set(params['sequence_length'])
                self.dropout_slider.set(params.get('dropout_rate', 0.0))
                self.activation_var.set(params['activation'])
                
                # Update optimizer and LR schedule if available
                if 'optimizer_type' in params:
                    self.optimizer_var.set(params['optimizer_type'])
                if 'lr_schedule_type' in params:
                    self.lr_schedule_var.set(params['lr_schedule_type'])
                
                # Update num_layers if available
                if 'num_layers' in params:
                    self.num_layers_slider.set(params['num_layers'])
                    self.update_layers_label(params['num_layers'])
                
                # Update layer sizes if available
                if 'hidden_sizes' in params and params['hidden_sizes']:
                    layer_sizes_str = ','.join([str(s) for s in params['hidden_sizes']])
                    self.layer_sizes_entry.delete(0, 'end')
                    self.layer_sizes_entry.insert(0, layer_sizes_str)
                
                # Update labels
                self.hidden_size_label.configure(text=f"Hidden Units (Neurons): {params['hidden_size']}")
                self.lr_label.configure(text=f"Learning Rate: {params['learning_rate']:.4f}")
                self.seq_len_label.configure(text=f"Sequence Length: {params['sequence_length']}")
                dropout_val = params.get('dropout_rate', 0.0)
                self.dropout_label.configure(
                    text=f"Dropout Rate: {dropout_val:.2f} {'(Off)' if dropout_val < 0.01 else '(Regularization)'}"
                )
                
                # Update loss plot if available
                if len(self.model.loss_history) > 0:
                    self.update_training_plots()
                
                # Update advanced metrics
                self.update_advanced_metrics()
                
                messagebox.showinfo("Success", f"Model loaded from:\n{filepath}\n"
                                             f"Optimizer: {params.get('optimizer_type', 'sgd').upper()}\n"
                                             f"LR Schedule: {params.get('lr_schedule_type', 'constant')}")
                self.update_status(f"Model loaded: {os.path.basename(filepath)}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def show_model_info(self):
        """Show model information."""
        if self.model is None:
            messagebox.showwarning("Warning", "No model initialized!")
            return
        
        params = self.model.get_parameters()
        
        # Format final loss properly
        if self.model.epoch_losses:
            final_loss = f"{self.model.epoch_losses[-1]:.6f}"
        else:
            final_loss = "N/A"
        
        # Build architecture string
        num_layers = self.model.num_layers
        layer_type = "Multi-layer (Stacked/Deep RNN)" if num_layers > 1 else "Single-layer RNN"
        layer_sizes = " → ".join([str(s) for s in self.model.hidden_sizes])
        
        info = f"""
Model Information:
═══════════════════════════════

Architecture:
  • Type: {layer_type}
  • Number of Layers: {num_layers}
  • Layer Sizes: {layer_sizes}
  • Input Size: {params['input_size']}
  • Output Size: {params['output_size']}
  • Total Parameters: {params['total_parameters']:,}

Hyperparameters:
  • Learning Rate: {params['learning_rate']:.6f}
  • Sequence Length: {params['sequence_length']}
  • Activation: {params['activation']}
  • Dropout Rate: {params.get('dropout_rate', 0.0):.3f}
  • Optimizer: {params.get('optimizer_type', 'sgd').upper()}
  • LR Schedule: {params.get('lr_schedule_type', 'constant')}
  • Current LR: {params.get('current_lr', params['learning_rate']):.6f}

Training History:
  • Total Iterations: {len(self.model.loss_history)}
  • Total Epochs: {len(self.model.epoch_losses)}
  • Final Loss: {final_loss}

Layer Details:
"""
        
        # Add layer details
        for i, size in enumerate(self.model.hidden_sizes, 1):
            info += f"  • Layer {i}: {size}  (Neurons)\n"
        
        messagebox.showinfo("Model Information", info)
    
    def show_model_schema(self):
        """Show model architecture schema in a new window with visualization."""
        if self.model is None:
            messagebox.showwarning("Warning", "No model initialized!")
            return
        
        # Create new window
        schema_window = ctk.CTkToplevel(self)
        schema_window.title("Model Architecture Schema")
        
        # Keep window on top and grab focus
        schema_window.attributes('-topmost', True)
        schema_window.focus_force()
        schema_window.grab_set()  # Make it modal
        
        # Set window size (larger to accommodate more information)
        window_width = 1000
        window_height = 850
        screen_width = schema_window.winfo_screenwidth()
        screen_height = schema_window.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        schema_window.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Create main frame
        main_frame = ctk.CTkFrame(schema_window)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Title
        title_label = ctk.CTkLabel(
            main_frame,
            text="🧠 RNN Model Architecture Schema",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=10)
        
        # Create matplotlib figure for schema (larger to show more information)
        fig = Figure(figsize=(9, 8), dpi=100, facecolor='#2b2b2b')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#2b2b2b')
        
        # Draw the schema
        self.draw_model_schema(ax)
        
        # Create canvas
        canvas_frame = ctk.CTkFrame(main_frame)
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        canvas = FigureCanvasTkAgg(fig, canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Notes section
        notes_label = ctk.CTkLabel(
            main_frame,
            text="📝 Add Notes (optional):",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        notes_label.pack(pady=(10, 5))
        
        notes_text = ctk.CTkTextbox(main_frame, height=80)
        notes_text.pack(fill="x", padx=10, pady=5)
        notes_text.insert("1.0", "Add your notes about this model here...")
        
        # Buttons frame
        button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        button_frame.pack(pady=10)
        
        # Save button
        def save_schema():
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"model_schema_{timestamp}.png"
                filepath = os.path.join(self.outputs_dir, filename)
                
                # Get notes
                notes = notes_text.get("1.0", "end-1c").strip()
                
                # Create new figure with notes if provided (larger to accommodate more info)
                if notes and notes != "Add your notes about this model here...":
                    save_fig = Figure(figsize=(10, 10), dpi=150, facecolor='white')
                    save_ax = save_fig.add_subplot(111)
                    save_ax.set_facecolor('white')
                    
                    # Draw schema
                    self.draw_model_schema(save_ax, for_save=True)
                    
                    # Add notes at the bottom
                    notes_y = -0.15
                    save_ax.text(0.5, notes_y, f"Notes: {notes}", 
                               transform=save_ax.transAxes,
                               fontsize=9, ha='center', va='top',
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8),
                               wrap=True)
                else:
                    save_fig = Figure(figsize=(10, 9), dpi=150, facecolor='white')
                    save_ax = save_fig.add_subplot(111)
                    save_ax.set_facecolor('white')
                    self.draw_model_schema(save_ax, for_save=True)
                
                save_fig.tight_layout()
                save_fig.savefig(filepath, facecolor='white', edgecolor='none', bbox_inches='tight')
                
                messagebox.showinfo("Success", f"Schema saved to:\n{filepath}")
                self.update_status(f"Model schema saved: {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save schema: {str(e)}")
        
        save_btn = ctk.CTkButton(
            button_frame,
            text="💾 Save Schema as PNG",
            command=save_schema,
            fg_color="green",
            hover_color="darkgreen",
            width=200,
            height=40
        )
        save_btn.pack(side="left", padx=5)
        
        # Close button
        close_btn = ctk.CTkButton(
            button_frame,
            text="❌ Close",
            command=schema_window.destroy,
            fg_color="red",
            hover_color="darkred",
            width=150,
            height=40
        )
        close_btn.pack(side="left", padx=5)
        
        # Store references
        schema_window.fig = fig
        schema_window.canvas = canvas
    
    def draw_model_schema(self, ax, for_save=False):
        """Draw the model architecture schema on the given axes with comprehensive information."""
        ax.clear()
        ax.axis('off')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 14)  # Extended for more information
        
        params = self.model.get_parameters()
        num_layers = self.model.num_layers
        hidden_sizes = self.model.hidden_sizes
        
        # Colors
        if for_save:
            input_color = '#3498db'
            hidden_color = '#2ecc71'
            output_color = '#e74c3c'
            text_color = 'black'
            line_color = '#7f8c8d'
            box_color = 'lightblue'
            info_box_color = '#fff3cd'
            metrics_box_color = '#d4edda'
        else:
            input_color = '#3498db'
            hidden_color = '#2ecc71'
            output_color = '#e74c3c'
            text_color = 'white'
            line_color = '#95a5a6'
            box_color = '#34495e'
            info_box_color = '#3a3a3a'
            metrics_box_color = '#2a4a2a'
        
        # Calculate positions with 50% wider boxes
        total_width = 8
        layer_spacing = total_width / (num_layers + 2)  # +2 for input and output
        
        # Starting x position
        start_x = 1
        
        # Y positions - moved down to make room for info
        center_y = 7.5
        node_height = 2.25  # 50% taller (1.5 * 1.5)
        node_width = 1.2    # 50% wider (0.8 * 1.5)
        
        # Draw Input Layer
        x = start_x
        ax.add_patch(plt.Rectangle((x - node_width/2, center_y - node_height/2), node_width, node_height,
                                   facecolor=input_color, edgecolor=line_color, linewidth=2))
        ax.text(x, center_y, 'INPUT', ha='center', va='center', 
               fontsize=13, weight='bold', color='white')
        ax.text(x, center_y - node_height/2 - 0.3, f'Size: {params["input_size"]}',
               ha='center', va='top', fontsize=11, color=text_color)
        
        prev_x = x
        x += layer_spacing
        
        # Draw Hidden Layers (with 50% wider boxes)
        for i, hidden_size in enumerate(hidden_sizes, 1):
            # Draw arrow
            ax.annotate('', xy=(x - node_width/2 - 0.1, center_y), xytext=(prev_x + node_width/2 + 0.1, center_y),
                       arrowprops=dict(arrowstyle='->', lw=2, color=line_color))
            
            # Draw layer box (50% wider)
            ax.add_patch(plt.Rectangle((x - node_width/2, center_y - node_height/2), node_width, node_height,
                                      facecolor=hidden_color, edgecolor=line_color, linewidth=2))
            ax.text(x, center_y + 0.4, f'HIDDEN {i}', ha='center', va='center',
                   fontsize=12, weight='bold', color='white')
            ax.text(x, center_y - 0.4, f'{hidden_size}', ha='center', va='center',
                   fontsize=14, weight='bold', color='white')
            
            # Layer info below
            ax.text(x, center_y - node_height/2 - 0.3, f'Neurons: {hidden_size}',
                   ha='center', va='top', fontsize=10, color=text_color)
            
            # Recurrent connection (self-loop) - adjusted for larger box
            circle = plt.Circle((x + node_width/2 + 0.3, center_y + node_height/2 + 0.3), 0.2,
                               fill=False, edgecolor=hidden_color, linewidth=2)
            ax.add_patch(circle)
            ax.annotate('', xy=(x + node_width/2 - 0.1, center_y + node_height/2), 
                       xytext=(x + node_width/2 + 0.3, center_y + node_height/2 + 0.1),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color=hidden_color))
            
            prev_x = x
            x += layer_spacing
        
        # Draw Output Layer (50% wider)
        ax.annotate('', xy=(x - node_width/2 - 0.1, center_y), xytext=(prev_x + node_width/2 + 0.1, center_y),
                   arrowprops=dict(arrowstyle='->', lw=2, color=line_color))
        
        ax.add_patch(plt.Rectangle((x - node_width/2, center_y - node_height/2), node_width, node_height,
                                   facecolor=output_color, edgecolor=line_color, linewidth=2))
        ax.text(x, center_y, 'OUTPUT', ha='center', va='center',
               fontsize=13, weight='bold', color='white')
        ax.text(x, center_y - node_height/2 - 0.3, f'Size: {params["output_size"]}',
               ha='center', va='top', fontsize=11, color=text_color)
        
        # Add title
        title_text = f"RNN Architecture: {num_layers}-Layer"
        if num_layers > 1:
            title_text += f" (Deep/Stacked RNN)"
        ax.text(5, 13.5, title_text, ha='center', va='center',
               fontsize=16, weight='bold', color=text_color)
        
        # Add architecture info box
        info_text = f"Input: {params['input_size']} → "
        info_text += " → ".join([str(s) for s in hidden_sizes])
        info_text += f" → Output: {params['output_size']}"
        
        ax.text(5, 12.8, info_text, ha='center', va='center',
               fontsize=12, color=text_color,
               bbox=dict(boxstyle='round,pad=0.5', facecolor=info_box_color, alpha=0.8))
        
        # === MODEL PARAMETERS SECTION === (y: 2.8 - 1.8)
        model_info = "MODEL PARAMETERS\n"
        model_info += f"Total Parameters: {params['total_parameters']:,} | "
        model_info += f"Input Size: {params['input_size']} | "
        model_info += f"Output Size: {params['output_size']}\n"

        model_info += f"Hidden Layers: {num_layers} | "
        model_info += f"Hidden Sizes: {hidden_sizes}\n"
        
        model_info += f"Activation: {params['activation']} | "
        model_info += f"Dropout: {params.get('dropout_rate', 0.0):.2f} | "
        model_info += f"Optimizer: {params.get('optimizer_type', 'sgd').upper()}\n"
        
        model_info += f"Sequence Length: {params['sequence_length']} | "
        model_info += f"Learning Rate: {params['learning_rate']:.4f} | "
        model_info += f"LR Schedule: {params.get('lr_schedule', 'constant')}\n"
        
        model_info += f"Gradient Clip: {params.get('gradient_clip', 5.0):.1f} | "
        model_info += f"Current LR: {params.get('current_lr', params['learning_rate']):.6f}"
        
        ax.text(5, 3.5, model_info, ha='center', va='center',
               fontsize=11, color=text_color,
               bbox=dict(boxstyle='round,pad=0.5', facecolor=info_box_color, alpha=0.8))
        
        # === TRAINING METRICS SECTION === (y: 1.3 - 0.3) - Only if model is trained
        if hasattr(self, 'model') and self.model is not None and hasattr(self.model, 'loss_history') and len(self.model.loss_history) > 0:
            metrics_info = "TRAINING METRICS & DATA GENERATION\n"
            
            # Data Generation Information
            if hasattr(self, 'wave_type_var'):
                metrics_info += f"Data Type: {self.wave_type_var.get()} | "
            if hasattr(self, 'current_data_raw') and self.current_data_raw is not None:
                metrics_info += f"Samples: {len(self.current_data_raw)} | "
            if hasattr(self, 'training_data') and self.training_data is not None:
                metrics_info += f"Training Sequences: {len(self.training_data)}\n"
            if hasattr(self, 'frequency_slider'):
                metrics_info += f"Frequency: {self.frequency_slider.get():.2f} | "
            if hasattr(self, 'noise_slider'):
                metrics_info += f"Noise: {self.noise_slider.get():.3f} | "
            
            # Training Status
            metrics_info += f"Epochs: {len(self.model.loss_history)} | "
            metrics_info += f"Final Loss: {self.model.loss_history[-1]:.6f}\n"
            
            # Add training status if available
            if hasattr(self.model, 'get_training_status'):
                train_status = self.model.get_training_status()
                convergence = train_status.get('convergence_score', 0)
                plateau = train_status.get('plateau_detected', False)
                metrics_info += f"Convergence: {convergence:.1f}/100 | "
                metrics_info += f"Plateau: {'Yes' if plateau else 'No'} | "
            
            # Add gradient health if available
            if hasattr(self.model, 'get_gradient_health'):
                grad_health = self.model.get_gradient_health()
                status = grad_health.get('status', 'Unknown')
                metrics_info += f"Gradient: {status}\n"
            else:
                metrics_info += "\n"
            
            # Add comprehensive metrics from model if available
            if hasattr(self.model, 'get_comprehensive_metrics') and hasattr(self, 'training_data') and self.training_data is not None:
                try:
                    X_train = self.training_data.reshape(len(self.training_data), -1, 1)
                    y_train = self.training_targets.reshape(len(self.training_targets), -1, 1)
                    comp_metrics = self.model.get_comprehensive_metrics(
                        X_train.reshape(-1, 1), 
                        y_train.reshape(-1, 1)
                    )
                    metrics_info += f"MSE: {comp_metrics.get('mse', 0):.6f} | "
                    metrics_info += f"RMSE: {comp_metrics.get('rmse', 0):.6f} | "
                    metrics_info += f"MAE: {comp_metrics.get('mae', 0):.6f} | "
                    metrics_info += f"R²: {comp_metrics.get('r2', 0):.4f}\n"
                except:
                    pass
            
            # Add gradient monitor stats if available
            if hasattr(self, 'gradient_monitor') and self.gradient_monitor:
                grad_stats = self.gradient_monitor.get_statistics()
                if grad_stats:
                    metrics_info += f"Grad Mean: {grad_stats.get('mean_gradient', 0.0):.6f} | "
                    metrics_info += f"Grad Max: {grad_stats.get('max_gradient', 0.0):.6f} | "
                    metrics_info += f"Vanishing: {grad_stats.get('vanishing_count', 0)} | "
                    metrics_info += f"Exploding: {grad_stats.get('exploding_count', 0)}\n"
            
            # Add weight analyzer stats if available
            if hasattr(self, 'weight_analyzer') and self.weight_analyzer:
                weight_stats = self.weight_analyzer.get_statistics()
                if weight_stats:
                    metrics_info += f"Weight Mean: {weight_stats.get('mean', 0.0):.6f} | "
                    metrics_info += f"Weight Std: {weight_stats.get('std', 0.0):.6f} | "
                    metrics_info += f"Dead Neurons: {weight_stats.get('dead_neurons', 0)}\n"
            
            # Add training monitor stats if available
            if hasattr(self, 'training_monitor') and self.training_monitor:
                train_stats = self.training_monitor.get_statistics()
                if train_stats:
                    metrics_info += f"Avg Loss: {train_stats.get('mean_loss', 0.0):.6f} | "
                    metrics_info += f"Min Loss: {train_stats.get('min_loss', 0.0):.6f} | "
                    metrics_info += f"Loss Std: {train_stats.get('std_loss', 0.0):.6f}"
            
            ax.text(5, 1.0, metrics_info, ha='center', va='center',
                   fontsize=10, color=text_color,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=metrics_box_color, alpha=0.8))
        
        # Add legend for arrows at bottom left
        ax.text(0.5, 12.2, '→ Forward Pass', ha='left', va='center',
               fontsize=12, color=text_color)
        ax.text(0.5, 11.8, '↻ Recurrent Connection (Time)', ha='left', va='center',
               fontsize=12, color=text_color)
        
        # Add timestamp at bottom left corner
        timestamp_text = f"{datetime.now().strftime('%d-%m-%Y  (%H:%M)')}"
        ax.text(0.0, 0.2, timestamp_text, ha='left', va='bottom',
               fontsize=12, color=text_color, style='italic',
               bbox=dict(boxstyle='round,pad=0.3', facecolor=box_color, alpha=0.6))
    
    def save_data_plot(self):
        """Save data plot with parameters."""
        if self.current_data_raw is None:
            messagebox.showwarning("Warning", "No data to save! Generate data first.")
            return
        
        try:
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data_plot_{timestamp}.png"
            filepath = os.path.join(self.outputs_dir, filename)
            
            # Create a new figure for saving (larger and with parameters)
            fig = Figure(figsize=(12, 8), dpi=150)
            ax = fig.add_subplot(111)
            
            # Plot data
            ax.clear()
            if hasattr(self.data_ax, 'lines') and len(self.data_ax.lines) > 0:
                for line in self.data_ax.lines:
                    ax.plot(line.get_xdata(), line.get_ydata(), 
                           label=line.get_label(), 
                           color=line.get_color(),
                           linestyle=line.get_linestyle(),
                           linewidth=line.get_linewidth())
            
            ax.set_xlabel("Time Step", fontsize=12)
            ax.set_ylabel("Value", fontsize=12)
            ax.set_title("Time Series Data & Predictions", fontsize=14, weight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add parameters text
            params_text = self._get_parameters_text()
            fig.text(0.02, 0.02, params_text, fontsize=9, family='monospace',
                    verticalalignment='bottom', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.5))
            
            # Save
            fig.tight_layout()
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            messagebox.showinfo("Success", f"Plot saved to:\n{filepath}")
            self.update_status(f"Data plot saved: {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save plot: {str(e)}")
    
    def save_loss_plot(self):
        """Save loss plot with parameters."""
        if self.model is None or len(self.model.loss_history) == 0:
            messagebox.showwarning("Warning", "No training data to save! Train model first.")
            return
        
        try:
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"loss_plot_{timestamp}.png"
            filepath = os.path.join(self.outputs_dir, filename)
            
            # Create a new figure for saving
            fig = Figure(figsize=(12, 8), dpi=150)
            ax = fig.add_subplot(111)
            
            # Plot loss
            ax.plot(self.model.loss_history, 'r-', linewidth=1.5, label='Training Loss')
            ax.set_xlabel("Iteration", fontsize=12)
            ax.set_ylabel("Loss", fontsize=12)
            ax.set_title("Training Loss History", fontsize=14, weight='bold')
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add parameters text
            params_text = self._get_parameters_text()
            fig.text(0.02, 0.02, params_text, fontsize=9, family='monospace',
                    verticalalignment='bottom', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.5))
            
            # Save
            fig.tight_layout()
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            messagebox.showinfo("Success", f"Plot saved to:\n{filepath}")
            self.update_status(f"Loss plot saved: {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save plot: {str(e)}")
    
    def _get_parameters_text(self):
        """Get formatted parameters text for saving."""
        lines = []
        lines.append("=" * 60)
        lines.append("TRAINING PARAMETERS")
        lines.append("=" * 60)
        
        # Model parameters
        if self.model:
            params = self.model.get_parameters()
            lines.append(f"Hidden Units:      {params['hidden_size']}")
            lines.append(f"Learning Rate:     {params['learning_rate']:.6f}")
            lines.append(f"Sequence Length:   {params['sequence_length']}")
            lines.append(f"Activation:        {params['activation']}")
            lines.append(f"Dropout Rate:      {params.get('dropout_rate', 0.0):.3f}")
            lines.append(f"Optimizer:         {params.get('optimizer_type', 'sgd').upper()}")
            lines.append(f"LR Schedule:       {params.get('lr_schedule_type', 'constant')}")
            lines.append(f"Current LR:        {params.get('current_lr', params['learning_rate']):.6f}")
            lines.append(f"Total Parameters:  {params['total_parameters']:,}")
        
        # Data parameters
        lines.append(f"\nWave Type:         {self.wave_type_var.get()}")
        lines.append(f"Samples:           {int(self.n_samples_slider.get())}")
        lines.append(f"Frequency:         {self.frequency_slider.get():.2f}")
        lines.append(f"Noise Level:       {self.noise_slider.get():.3f}")
        
        # Training info
        if self.model and len(self.model.epoch_losses) > 0:
            lines.append(f"\nEpochs Trained:    {len(self.model.epoch_losses)}")
            lines.append(f"Final Loss:        {self.model.epoch_losses[-1]:.6f}")
        
        lines.append(f"\nTimestamp:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def load_custom_data(self):
        """Load custom time series data from CSV file."""
        try:
            filepath = filedialog.askopenfilename(
                title="Select CSV File with Time Series Data",
                filetypes=[
                    ("CSV files", "*.csv"),
                    ("Text files", "*.txt"),
                    ("All files", "*.*")
                ],
                initialdir=os.path.dirname(__file__)
            )
            
            if not filepath:
                return
            
            # Read CSV file
            import csv
            data_list = []
            
            with open(filepath, 'r', encoding='utf-8') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    # Skip empty rows
                    if not row:
                        continue
                    # Try to convert first column to float
                    try:
                        value = float(row[0])
                        data_list.append(value)
                    except (ValueError, IndexError):
                        # Skip header or invalid rows
                        continue
            
            if len(data_list) < 10:
                messagebox.showerror("Error", "CSV file must contain at least 10 numeric values!")
                return
            
            # Store raw data
            self.custom_data_raw = np.array(data_list)
            
            # Normalize data
            generator = DataGenerator()
            self.custom_data_normalized, self.data_min, self.data_max = generator.normalize_data(
                self.custom_data_raw
            )
            
            # Store as test data
            self.test_data = self.custom_data_normalized
            self.current_data_raw = self.custom_data_raw
            
            # Visualize
            self.data_ax.clear()
            self.data_ax.plot(self.custom_data_raw, 'b-', label='Custom Data', linewidth=1.5)
            self.data_ax.set_xlabel("Time Step")
            self.data_ax.set_ylabel("Value")
            self.data_ax.set_title("Custom Time Series Data")
            self.data_ax.legend()
            self.data_ax.grid(True, alpha=0.3)
            self.data_canvas.draw()
            
            messagebox.showinfo(
                "Success",
                f"Loaded {len(data_list)} data points from CSV!\n\n"
                f"Min: {self.data_min:.4f}\n"
                f"Max: {self.data_max:.4f}\n\n"
                f"You can now use 'Predict Future Values' to forecast."
            )
            
            self.update_status(f"Custom data loaded: {len(data_list)} points from {os.path.basename(filepath)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV: {str(e)}")
    
    def predict_future_values(self):
        """Predict future values based on loaded custom data."""
        if self.model is None:
            messagebox.showwarning("Warning", "Please initialize or load a trained model first!")
            return
        
        if self.custom_data_raw is None and self.current_data_raw is None:
            messagebox.showwarning(
                "Warning",
                "Please load custom data (CSV) or generate data first!"
            )
            return
        
        # Create dialog to ask how many steps to predict
        dialog = ctk.CTkInputDialog(
            text="How many future steps to predict?",
            title="Future Prediction"
        )
        n_steps_str = dialog.get_input()
        
        if not n_steps_str:
            return
        
        try:
            n_steps = int(n_steps_str)
            if n_steps < 1 or n_steps > 100:
                messagebox.showerror("Error", "Please enter a number between 1 and 100!")
                return
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number!")
            return
        
        try:
            # Use the last sequence_length points as seed
            seq_len = self.model.sequence_length
            
            # Get data to use
            if self.custom_data_normalized is not None:
                data_to_use = self.custom_data_normalized
                raw_data = self.custom_data_raw
            else:
                data_to_use = self.test_data
                raw_data = self.current_data_raw
            
            if len(data_to_use) < seq_len:
                messagebox.showerror(
                    "Error",
                    f"Data must have at least {seq_len} points (model's sequence length)!"
                )
                return
            
            # Take last seq_len points as seed
            seed = data_to_use[-seq_len:]
            
            # Predict future sequence
            predictions_normalized = self.model.predict_sequence(seed, n_steps)
            
            # Denormalize predictions
            generator = DataGenerator()
            predictions = generator.denormalize_data(
                predictions_normalized,
                self.data_min,
                self.data_max
            )
            
            # Plot results
            self.data_ax.clear()
            
            # Historical data
            self.data_ax.plot(
                range(len(raw_data)),
                raw_data,
                'b-',
                label='Historical Data',
                linewidth=2
            )
            
            # Seed region (what model uses to predict)
            seed_start = len(raw_data) - seq_len
            self.data_ax.axvline(
                x=seed_start,
                color='orange',
                linestyle='--',
                alpha=0.5,
                label='Prediction Start'
            )
            
            # Future predictions
            future_x = range(len(raw_data), len(raw_data) + n_steps)
            self.data_ax.plot(
                future_x,
                predictions[-n_steps:],
                'r--',
                label=f'Predicted ({n_steps} steps)',
                linewidth=2,
                marker='o',
                markersize=4
            )
            
            self.data_ax.set_xlabel("Time Step")
            self.data_ax.set_ylabel("Value")
            self.data_ax.set_title(f"Future Prediction: Next {n_steps} Values")
            self.data_ax.legend()
            self.data_ax.grid(True, alpha=0.3)
            self.data_canvas.draw()
            
            # Show predictions in message box
            pred_text = "Predicted Future Values:\n" + "="*30 + "\n"
            for i, val in enumerate(predictions[-n_steps:], 1):
                pred_text += f"Step +{i}: {val[0]:.4f}\n"
            
            messagebox.showinfo("Future Prediction", pred_text)
            self.update_status(f"Predicted {n_steps} future values")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}\n\n{type(e).__name__}")
    
    def show_help(self):
        """Show help window."""
        help_window = ctk.CTkToplevel(self)
        help_window.title("RNN Trainer - Help & Documentation")
        
        # Keep window on top and grab focus
        help_window.attributes('-topmost', True)
        help_window.focus_force()
        help_window.grab_set()  # Make it modal
        
        # Set help window size and center
        window_width = 800
        window_height = 600
        screen_width = help_window.winfo_screenwidth()
        screen_height = help_window.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        help_window.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Create scrollable text
        text_frame = ctk.CTkScrollableFrame(help_window)
        text_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Read help text from file
        help_file = os.path.join(os.path.dirname(__file__), 'rnn_help.txt')
        try:
            with open(help_file, 'r', encoding='utf-8') as f:
                help_text = f.read()
        except FileNotFoundError:
            help_text = """
RNN TRAINER - HELP

Help file not found. Please ensure rnn_help.txt is in the same directory as this application.

For documentation, see README.md in the project folder.
            """
        
        help_label = ctk.CTkLabel(
            text_frame,
            text=help_text,
            font=ctk.CTkFont(family="Courier New", size=12),
            justify="left",
            anchor="w"
        )
        help_label.pack(fill="both", expand=True, padx=10, pady=10)
    
    def update_layers_label(self, num_layers):
        """Update the layers label based on slider value."""
        layer_type = "Single" if num_layers == 1 else f"Multi-layer ({num_layers})"
        self.num_layers_label.configure(text=f"Hidden Layers: {num_layers} ({layer_type})")


def main():
    """Main entry point."""
    app = RNNTrainerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
