import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import json
import glob
import os
from pathlib import Path
import skrf as rf
import seaborn as sns
from scipy import signal
import warnings
warnings.filterwarnings('ignore')


def load_configuration(plot_config_data, kpi_config_data=None):
    if not plot_config_data:
        raise ValueError("Plot configuration data is required.")
    if kpi_config_data is None:
        kpi_config_data = {}
    return plot_config_data, kpi_config_data


def load_excel_data(excel_files):
    if not excel_files:
        raise FileNotFoundError("SParam_Summary.xlsx not found in the provided folders.")
    per_file_data = pd.read_excel(excel_files[0], sheet_name='Per_File')
    summary_data = pd.read_excel(excel_files[0], sheet_name='Summary')
    return {'Per_File': per_file_data, 'Summary': summary_data}


def load_s2p_files(s2p_files):
    if not s2p_files:
        raise FileNotFoundError("No *.s2p files found in the provided folders.")
    networks = {}
    for file_path in s2p_files:
        try:
            network = rf.Network(file_path)
            filename = os.path.basename(file_path)
            networks[filename] = network
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
    return networks


def load_simulation_data(sim_s2p_files, s11_sigma_files, s21_sigma_files):
    sim_data = {}
    if sim_s2p_files:
        try:
            sim_data['nominal'] = rf.Network(sim_s2p_files[0])
        except Exception as e:
            print(f"Warning: Could not load simulation S2P: {e}")
    if s11_sigma_files:
        try:
            sim_data['s11_sigma'] = pd.read_csv(s11_sigma_files[0])
        except Exception as e:
            print(f"Warning: Could not load S11 sigma data: {e}")
    if s21_sigma_files:
        try:
            sim_data['s21_sigma'] = pd.read_csv(s21_sigma_files[0])
        except Exception as e:
            print(f"Warning: Could not load S21 sigma data: {e}")
    return sim_data


def apply_frequency_shift(network, shift_mhz):
    if shift_mhz == 0:
        return network
    shifted_network = network.copy()
    original_freq_hz = network.frequency.f
    shifted_freq_hz = original_freq_hz + shift_mhz * 1e6
    new_frequency = rf.Frequency.from_f(shifted_freq_hz, unit='Hz')
    shifted_network = rf.Network(frequency=new_frequency, s=network.s, name=network.name)
    return shifted_network


# For brevity, we'll define a helper function to generate each plot

def generate_plot(networks, plot_config, sim_data, freq_shifts, s_param_config, param_type, save_folder, plot_filename):
    plt.figure(figsize=(12, 8))
    for filename, network in networks.items():
        for shift in freq_shifts:
            shifted_net = apply_frequency_shift(network, shift)
            freq_mhz = shifted_net.frequency.f / 1e6
            if param_type == 'S11':
                y_values = 20 * np.log10(np.abs(shifted_net.s[:, 0, 0]))
            elif param_type == 'S22':
                y_values = 20 * np.log10(np.abs(shifted_net.s[:, 1, 1]))
            elif param_type == 'S21':
                y_values = 20 * np.log10(np.abs(shifted_net.s[:, 1, 0]))
            label = f"{filename}"
            if shift != 0:
                label += f" (shift: {shift:+.1f}MHz)"
            plt.plot(freq_mhz, y_values, alpha=0.7, label=label)

    if sim_data and 'nominal' in sim_data and plot_config['simulation_settings']['include_simulation']:
        sim_freq_mhz = sim_data['nominal'].frequency.f / 1e6
        if param_type == 'S11':
            sim_y_values = 20 * np.log10(np.abs(sim_data['nominal'].s[:, 0, 0]))
        elif param_type == 'S22':
            sim_y_values = 20 * np.log10(np.abs(sim_data['nominal'].s[:, 1, 1]))
        elif param_type == 'S21':
            sim_y_values = 20 * np.log10(np.abs(sim_data['nominal'].s[:, 1, 0]))
        plt.plot(sim_freq_mhz, sim_y_values, 'r--', linewidth=2, label='Simulation Nominal')

    config = s_param_config[param_type.lower() + '_return_loss' if param_type != 'S21' else 's21_insertion_loss']
    plt.xlim(config['x_axis']['min'], config['x_axis']['max'])
    plt.ylim(config['y_axis']['min'], config['y_axis']['max'])
    plt.xlabel(f"Frequency ({config['x_axis']['unit']})")
    plt.ylabel(f"{param_type} ({config['y_axis']['unit']})")
    plt.title(f"{param_type} vs Frequency")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plot_path = os.path.join(save_folder, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return plot_filename



def create_s_parameter_plots(networks, plot_config, kpi_config, sim_data=None, save_folder='.'): 
    plots_created = []
    freq_shifts = plot_config['frequency_shifts']['shifts'] if plot_config['frequency_shifts']['enabled'] else [0]
    s_param_config = plot_config['axis_ranges']['s_parameter_plots']

    # S11 Return Loss Plot
    plots_created.append(generate_plot(networks, plot_config, sim_data, freq_shifts, s_param_config, 'S11', save_folder, 'S11_Return_Loss.png'))

    # S22 Return Loss Plot
    plots_created.append(generate_plot(networks, plot_config, sim_data, freq_shifts, s_param_config, 'S22', save_folder, 'S22_Return_Loss.png'))

    # S21 Insertion Loss Plot
    plots_created.append(generate_plot(networks, plot_config, sim_data, freq_shifts, s_param_config, 'S21', save_folder, 'S21_Insertion_Loss.png'))

    # S21 Rejection Loss Plot
    s_param_config['s21_insertion_loss'] = s_param_config['s21_rejection_loss']  # Using rejection config for this plot
    plots_created.append(generate_plot(networks, plot_config, sim_data, freq_shifts, s_param_config, 'S21', save_folder, 'S21_Rejection_Loss.png'))

    return plots_created
def create_statistical_plots(excel_data, plot_config, save_folder='.'):
    """Create individual box plots for each parameter from Excel statistical data and save in the provided folder"""
    plots_created = []
    
    # Get the Per_File sheet which contains individual measurements
    per_file_df = excel_data['Per_File']
    
    # Get column names excluding 'File' column
    parameter_columns = [col for col in per_file_df.columns if col != 'File']
    
    for param in parameter_columns:
        # Get data for this parameter
        param_values = per_file_df[param].dropna()
        
        if len(param_values) == 0:
            continue
            
        # Create individual box plot for this parameter
        plt.figure(figsize=(10, 6))
        
        # Create box plot
        box_data = [param_values.values]
        bp = plt.boxplot(box_data, labels=[param], patch_artist=True)
        
        # Customize box plot appearance
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][0].set_alpha(0.7)
        
        # Add individual data points
        y_values = param_values.values
        x_values = [1] * len(y_values)  # All points at x=1 for single parameter
        plt.scatter(x_values, y_values, alpha=0.6, color='red', s=30, zorder=3)
        
        plt.ylabel("Value")
        
        # Set title and labels
        plt.title(f"{param} - Statistical Distribution")
        plt.xlabel("Parameter")
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_val = param_values.mean()
        std_val = param_values.std()
        min_val = param_values.min()
        max_val = param_values.max()
        
        stats_text = f"Mean: {mean_val:.3f}\nStd: {std_val:.3f}\nMin: {min_val:.3f}\nMax: {max_val:.3f}"
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save individual plot to the provided folder
        filename = f"BoxPlot_{param}.png"
        plot_path = os.path.join(save_folder, filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        plots_created.append(filename)
        
        print(f"Created box plot: {plot_path}")
    
    return plots_created


def create_histogram_plots(excel_data, plot_config, save_folder='.'):
    """Create individual histogram plots for each parameter and save in the provided folder"""
    plots_created = []
    
    # Get the Per_File sheet which contains individual measurements
    per_file_df = excel_data['Per_File']
    
    # Get column names excluding 'File' column
    parameter_columns = [col for col in per_file_df.columns if col != 'File']
    
    for param in parameter_columns:
        # Get data for this parameter
        param_values = per_file_df[param].dropna()
        
        if len(param_values) == 0:
            continue
            
        # Create individual histogram for this parameter
        plt.figure(figsize=(10, 6))
        
        # Create histogram
        values = param_values.values
        plt.hist(values, bins=min(20, max(5, len(values)//3)), alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add vertical lines for mean and limits
        mean_val = values.mean()
        plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        
        # Add ±3σ lines
        std_val = values.std()
        if std_val > 0:  # Only add sigma lines if std deviation is not zero
            plt.axvline(mean_val + 3*std_val, color='orange', linestyle=':', linewidth=2, label=f'+3σ: {mean_val + 3*std_val:.3f}')
            plt.axvline(mean_val - 3*std_val, color='orange', linestyle=':', linewidth=2, label=f'-3σ: {mean_val - 3*std_val:.3f}')
        
        # Set title and labels
        plt.title(f"{param} - Distribution Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        # Save individual plot to the provided folder
        filename = f"Histogram_{param}.png"
        plot_path = os.path.join(save_folder, filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        plots_created.append(filename)
        
        print(f"Created histogram: {plot_path}")
    
    return plots_created
def create_main_execution(
        plot_config_data,
        excel_files,
        s2p_files,
        sim_s2p_files=None,
        s11_sigma_files=None,
        s21_sigma_files=None,
        kpi_config_data=None,
        save_folder='.'
):
    try:
        plot_config, kpi_config = load_configuration(plot_config_data, kpi_config_data)
        excel_data = load_excel_data(excel_files)
        networks = load_s2p_files(s2p_files)
        sim_data = load_simulation_data(sim_s2p_files, s11_sigma_files, s21_sigma_files)

        all_plots = []

        # Generate S-parameter plots
        s_param_plots = create_s_parameter_plots(networks, plot_config, kpi_config, sim_data, save_folder)
        all_plots.extend(s_param_plots)

        # Generate statistical plots
        statistical_plots = create_statistical_plots(excel_data, plot_config, save_folder)
        all_plots.extend(statistical_plots)

        # Generate histogram plots
        histogram_plots = create_histogram_plots(excel_data, plot_config, save_folder)
        all_plots.extend(histogram_plots)

        return all_plots

    except Exception as e:
        print(f"Error during main execution: {e}")
        return []
