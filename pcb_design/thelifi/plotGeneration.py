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

def generate_plot(networks, plot_config, sim_data, freq_shifts, s_param_config, kpi_config, param_type, save_folder, plot_filename):
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

    # ----------- SPECLINE BLOCKS -----------
    if param_type == 'S11':
        rl_colors = ['red', 'darkred', 'crimson']
        for i, rl_spec in enumerate(kpi_config['KPIs'].get('RL', [])):
            freq_start = rl_spec['range'][0] / 1e6
            freq_end = rl_spec['range'][1] / 1e6
            lsl_value = -abs(rl_spec['LSL'])
            color = rl_colors[i % len(rl_colors)]
            freq_range = np.linspace(freq_start, freq_end, 100)
            if plot_config.get('plot_settings', {}).get('show_spec_lines', False):
                plt.plot(freq_range, [lsl_value] * len(freq_range), color=color, linestyle='--', linewidth=2, label=f'{rl_spec["name"]} LSL: {lsl_value} dB')
            plt.axvspan(freq_start, freq_end, alpha=0.05, color=color)
    elif param_type == 'S22':
        rl_colors = ['red', 'darkred', 'crimson']
        for i, rl_spec in enumerate(kpi_config['KPIs'].get('RL', [])):
            freq_start = rl_spec['range'][0] / 1e6
            freq_end = rl_spec['range'][1] / 1e6
            lsl_value = -abs(rl_spec['LSL'])
            color = rl_colors[i % len(rl_colors)]
            freq_range = np.linspace(freq_start, freq_end, 100)
            if plot_config.get('plot_settings', {}).get('show_spec_lines', False):
                plt.plot(freq_range, [lsl_value] * len(freq_range), color=color, linestyle='--', linewidth=2, label=f'{rl_spec["name"]} LSL: {lsl_value} dB')
            plt.axvspan(freq_start, freq_end, alpha=0.05, color=color)
    elif param_type == 'S21' and plot_filename == 'S21_Insertion_Loss.png':
        il_colors = ['green', 'darkgreen', 'lime']
        for i, il_spec in enumerate(kpi_config['KPIs'].get('IL', [])):
            freq_start = il_spec['range'][0] / 1e6
            freq_end = il_spec['range'][1] / 1e6
            usl_value = -abs(il_spec['USL'])  # Always negative for loss
            color = il_colors[i % len(il_colors)]
            freq_range = np.linspace(freq_start, freq_end, 100)
            if plot_config.get('plot_settings', {}).get('show_spec_lines', False):
                plt.plot(freq_range, [usl_value] * len(freq_range), color=color, linestyle='--', linewidth=2, label=f'{il_spec["name"]} USL: {usl_value} dB')
            plt.axvspan(freq_start, freq_end, alpha=0.05, color=color)
    elif param_type == 'S21' and plot_filename == 'S21_Rejection_Loss.png':
        rej_colors = ['orange', 'purple', 'brown', 'pink']
        for i, sb_spec in enumerate(kpi_config.get('StopBands', [])):
            freq_start = sb_spec['range'][0] / 1e6
            freq_end = sb_spec['range'][1] / 1e6
            lsl_value = -abs(sb_spec['LSL'])  # Always negative for rejection
            color = rej_colors[i % len(rej_colors)]
            freq_range = np.linspace(freq_start, freq_end, 100)
            if plot_config.get('plot_settings', {}).get('show_spec_lines', False):
                plt.plot(freq_range, [lsl_value] * len(freq_range), color=color, linestyle='--', linewidth=2, label=f'{sb_spec["name"]} LSL: {lsl_value} dB')
            plt.axvspan(freq_start, freq_end, alpha=0.1, color=color)

    # ----------- END SPECLINE BLOCKS -----------

    if sim_data and 'nominal' in sim_data and plot_config['simulation_settings']['include_simulation']:
        sim_freq_mhz = sim_data['nominal'].frequency.f / 1e6
        if param_type == 'S11':
            sim_y_values = 20 * np.log10(np.abs(sim_data['nominal'].s[:, 0, 0]))
        elif param_type == 'S22':
            sim_y_values = 20 * np.log10(np.abs(sim_data['nominal'].s[:, 1, 1]))
        elif param_type == 'S21':
            sim_y_values = 20 * np.log10(np.abs(sim_data['nominal'].s[:, 1, 0]))
        plt.plot(sim_freq_mhz, sim_y_values, 'r--', linewidth=2, label='Simulation Nominal')

    config = s_param_config[param_type.lower() + '_return_loss' if param_type != 'S21' else ('s21_insertion_loss' if plot_filename == 'S21_Insertion_Loss.png' else 's21_rejection_loss')]
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
    plots_created.append(generate_plot(networks, plot_config, sim_data, freq_shifts, s_param_config, kpi_config, 'S11', save_folder, 'S11_Return_Loss.png'))

    # S22 Return Loss Plot
    plots_created.append(generate_plot(networks, plot_config, sim_data, freq_shifts, s_param_config, kpi_config, 'S22', save_folder, 'S22_Return_Loss.png'))

    # S21 Insertion Loss Plot
    plots_created.append(generate_plot(networks, plot_config, sim_data, freq_shifts, s_param_config, kpi_config, 'S21', save_folder, 'S21_Insertion_Loss.png'))

    # S21 Rejection Loss Plot
    plots_created.append(generate_plot(networks, plot_config, sim_data, freq_shifts, s_param_config, kpi_config, 'S21', save_folder, 'S21_Rejection_Loss.png'))

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
        #mean_val = values.mean()
        #plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        
        # Add ±3σ lines
        #std_val = values.std()
        #if std_val > 0:  # Only add sigma lines if std deviation is not zero
        #    plt.axvline(mean_val + 3*std_val, color='orange', linestyle=':', linewidth=2, label=f'+3σ: {mean_val + 3*std_val:.3f}')
        #    plt.axvline(mean_val - 3*std_val, color='orange', linestyle=':', linewidth=2, label=f'-3σ: {mean_val - 3*std_val:.3f}')
        
        # Set title and labels
        plt.title(f"{param} - Distribution Histogram")
        #plt.xlabel("Value")
        #plt.ylabel("Frequency")

        plt.xlabel(param)
        plt.ylabel("Number of units")

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

# --- ADVANCED PLOTS: Fix GD Curve Y-axis auto-scaling if needed ---
def create_advanced_plots(plot_config, kpi_config, networks, freq_shifts, save_folder='.'):
    plots_created = []


    # -------- Group Delay Plot --------
    def phase_and_gd(network):
        f_hz = network.frequency.f  # Hz
        ang = np.unwrap(np.angle(network.s[:, 1, 0]))  # S21 phase
        df = np.gradient(f_hz)
        # Avoid division by zero or very small df if frequency points are identical
        gd_s = -np.gradient(ang) / (2 * np.pi * (df + 1e-12)) # Add small epsilon to prevent division by zero
        return ang, gd_s * 1e9  # ns

    plt.figure(figsize=(12, 6))
    for fname, nt in networks.items():
        for s in freq_shifts:
            nt_shift = apply_frequency_shift(nt, s)
            _, gd_ns = phase_and_gd(nt_shift)
            plt.plot(nt_shift.frequency.f / 1e6, gd_ns, alpha=.7, label=f"{fname} {s:+.1f} MHz" if s != 0 else fname)

    # Add GD spec limits
    gd_colors = ['purple', 'darkviolet', 'magenta']
    if 'GD' in kpi_config['KPIs']:
        for i, gd_spec in enumerate(kpi_config['KPIs']['GD']):
            freq_start_hz = gd_spec['range'][0]
            freq_end_hz = gd_spec['range'][1]
            freq_start_mhz = freq_start_hz / 1e6
            freq_end_mhz = freq_end_hz / 1e6
            color = gd_colors[i % len(gd_colors)]
            
            freq_range_mhz = np.linspace(freq_start_mhz, freq_end_mhz, 100)
            if plot_config.get('plot_settings', {}).get('show_spec_lines', False):
                if 'USL' in gd_spec:
                    plt.plot(freq_range_mhz, [gd_spec['USL']] * len(freq_range_mhz), 
                            color=color, linestyle='--', linewidth=2, 
                            label=f'{gd_spec["name"]} USL: {gd_spec["USL"]} ns')
                if 'LSL' in gd_spec:
                    plt.plot(freq_range_mhz, [gd_spec['LSL']] * len(freq_range_mhz), 
                            color=color, linestyle=':', linewidth=2, 
                            label=f'{gd_spec["name"]} LSL: {gd_spec["LSL"]} ns')
            plt.axvspan(freq_start_mhz, freq_end_mhz, alpha=0.05, color=color)


    cfg = plot_config['axis_ranges']['advanced_plots']['group_delay']
    plt.xlim(cfg['x_axis']['min'], cfg['x_axis']['max'])
    plt.ylim(cfg['y_axis']['min'], cfg['y_axis']['max'])
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("GD (ns)")
    plt.title("Group Delay (continuous)")
    plt.grid(True, alpha=.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    gd_plot = os.path.join(save_folder, "GD_Curve.png")
    plt.savefig(gd_plot, dpi=300)
    plt.close()
    plots_created.append("GD_Curve.png")

    # -------- Linear Phase Deviation Plot --------
    plt.figure(figsize=(12, 6))
    LPD_FREQ_LOW, LPD_FREQ_HIGH = kpi_config['KPIs']['LPD_MIN'][0]['range']

    for fname, nt in networks.items():
        for s in freq_shifts:
            nt_shift = apply_frequency_shift(nt, s)
            phase, _ = phase_and_gd(nt_shift)
            f_hz = nt_shift.frequency.f
            mask = (f_hz >= LPD_FREQ_LOW) & (f_hz <= LPD_FREQ_HIGH)

            if np.sum(mask) > 2:
                f_fit = f_hz[mask]
                phase_fit = phase[mask]
                A = np.vstack([f_fit, np.ones_like(f_fit)]).T
                slope, intercept = np.linalg.lstsq(A, phase_fit, rcond=None)[0]
                linear_phase = slope * f_hz + intercept
                phase_deviation_deg = np.degrees(phase - linear_phase)
                lpd_band = phase_deviation_deg[mask]
                lpd_normalized = lpd_band - lpd_band.min()
                f_band_mhz = f_hz[mask] / 1e6
                label = f"{fname} {s:+.1f} MHz" if s != 0 else fname
                plt.plot(f_band_mhz, lpd_normalized, alpha=.7, label=label)

    # LPD Spec lines (MIN and MAX)
    lpd_colors = ['red', 'darkred', 'crimson', 'orange', 'darkorange', 'orangered']
    color_index = 0
    for i, lpd_spec in enumerate(kpi_config['KPIs'].get('LPD_MIN', [])):
        freq_start = lpd_spec['range'][0] / 1e6
        freq_end = lpd_spec['range'][1] / 1e6
        lsl_value = lpd_spec['LSL']
        color = lpd_colors[color_index % len(lpd_colors)]
        color_index += 1
        freq_range = np.linspace(freq_start, freq_end, 100)
        if plot_config.get('plot_settings', {}).get('show_spec_lines', False):
            plt.plot(freq_range, [lsl_value] * len(freq_range), color=color, linestyle='--', linewidth=2, label=f'{lpd_spec["name"]} LSL: {lpd_spec["LSL"]} deg')
        plt.axvspan(freq_start, freq_end, alpha=0.05, color=color)
    for i, lpd_spec in enumerate(kpi_config['KPIs'].get('LPD_MAX', [])):
        freq_start = lpd_spec['range'][0] / 1e6
        freq_end = lpd_spec['range'][1] / 1e6
        usl_value = lpd_spec['USL']
        color = lpd_colors[color_index % len(lpd_colors)]
        color_index += 1
        freq_range = np.linspace(freq_start, freq_end, 100)
        if plot_config.get('plot_settings', {}).get('show_spec_lines', False):
            plt.plot(freq_range, [usl_value] * len(freq_range), color=color, linestyle='--', linewidth=2, label=f'{lpd_spec["name"]} USL: {lpd_spec["USL"]} deg')
        plt.axvspan(freq_start, freq_end, alpha=0.05, color=color)

    cfg = plot_config['axis_ranges']['advanced_plots']['linear_phase_deviation']
    plt.xlim(cfg['x_axis']['min'], cfg['x_axis']['max'])
    plt.ylim(0, 4)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("LPD (deg)")
    plt.title("Linear-Phase Deviation (continuous)")
    plt.grid(True, alpha=.3)
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    lpd_plot = os.path.join(save_folder, "LPD_Curve.png")
    plt.savefig(lpd_plot, dpi=300, bbox_inches='tight')
    plt.close()
    plots_created.append("LPD_Curve.png")

    # -------- Flatness Scatter Plot --------
    plt.figure(figsize=(12, 6))
    colors = ['blue', 'red', 'green', 'orange', 'purple']

    for band_idx, band in enumerate(kpi_config['KPIs'].get('Flat', [])):
        lo, hi = band['range']
        unit_numbers = []
        flatness_values = []

        unit_num = 1
        for fname, nt in networks.items():
            f = nt.frequency.f
            s21 = 20 * np.log10(np.abs(nt.s[:, 1, 0]))
            mask = (f >= lo) & (f <= hi)

            if np.any(mask):
                flat = s21[mask].max() - s21[mask].min()
                unit_numbers.append(unit_num)
                flatness_values.append(flat)

            unit_num += 1

        plt.scatter(unit_numbers, flatness_values, color=colors[band_idx % len(colors)], alpha=0.7, s=50, label=f"{band['name']} ({lo / 1e6:.0f}-{hi / 1e6:.0f} MHz)")
        if plot_config.get('plot_settings', {}).get('show_spec_lines', False):
            if 'USL' in band:
                plt.axhline(y=band['USL'], color=colors[band_idx % len(colors)], linestyle='--', alpha=0.5, label=f"{band['name']} USL: {band['USL']} dB")

    plt.xlabel("Unit Number")
    plt.ylabel("Flatness (dB)")
    plt.title("S21 Flatness vs Unit Number")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    flat_plot = os.path.join(save_folder, "Flatness_Scatter.png")
    plt.savefig(flat_plot, dpi=300, bbox_inches='tight')
    plt.close()
    plots_created.append("Flatness_Scatter.png")

    # -------- GD Variation Scatter Plot --------
    plt.figure(figsize=(12, 6))
    for band_idx, band in enumerate(kpi_config['KPIs'].get('GDV', [])):
        lo, hi = band['range']
        unit_numbers = []
        gdv_values = []

        unit_num = 1
        for fname, nt in networks.items():
            f = nt.frequency.f
            _, gd_ns = phase_and_gd(nt)
            mask = (f >= lo) & (f <= hi)

            if np.any(mask):
                gdv = np.ptp(gd_ns[mask])
                unit_numbers.append(unit_num)
                gdv_values.append(gdv)

            unit_num += 1

        plt.scatter(unit_numbers, gdv_values, color=colors[band_idx % len(colors)], alpha=0.7, s=50, label=f"{band['name']} ({lo / 1e6:.0f}-{hi / 1e6:.0f} MHz)")
        if plot_config.get('plot_settings', {}).get('show_spec_lines', False):
            if 'USL' in band:
                plt.axhline(y=band['USL'], color=colors[band_idx % len(colors)], linestyle='--', alpha=0.5, label=f"{band['name']} USL: {band['USL']} ns")

    plt.xlabel("Unit Number")
    plt.ylabel("GD Variation (ns)")
    plt.title("Group Delay Variation vs Unit Number")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    gdv_plot = os.path.join(save_folder, "GDV_Scatter.png")
    plt.savefig(gdv_plot, dpi=300, bbox_inches='tight')
    plt.close()
    plots_created.append("GDV_Scatter.png")

    return plots_created

def generate_s_parameter_plots_only(plot_config_data, excel_files, s2p_files, sim_s2p_files=None, s11_sigma_files=None, s21_sigma_files=None, kpi_config_data=None, save_folder='.'):
    try:
        plot_config, kpi_config = load_configuration(plot_config_data, kpi_config_data)
        if not kpi_config or 'KPIs' not in kpi_config:
            raise ValueError("kpi_config_data must be provided and contain a 'KPIs' key.")
        networks = load_s2p_files(s2p_files)
        sim_data = load_simulation_data(sim_s2p_files, s11_sigma_files, s21_sigma_files)

        all_plots = []

        # Generate S-parameter plots
        s_param_plots = create_s_parameter_plots(networks, plot_config, kpi_config, sim_data, save_folder)
        all_plots.extend(s_param_plots)

        # Generate advanced plots
        freq_shifts = plot_config['frequency_shifts']['shifts'] if plot_config['frequency_shifts']['enabled'] else [0]
        advanced_plots = create_advanced_plots(plot_config, kpi_config, networks, freq_shifts, save_folder)
        all_plots.extend(advanced_plots)

        return all_plots

    except Exception as e:
        print(f"Error during S-Parameter and advanced plot generation: {e}")
        return []



def generate_statistical_and_histogram_plots_only(plot_config_data, excel_files, save_folder='.'):
    try:
        plot_config, _ = load_configuration(plot_config_data)
        excel_data = load_excel_data(excel_files)

        all_plots = []

        # Generate statistical plots
        statistical_plots = create_statistical_plots(excel_data, plot_config, save_folder)
        all_plots.extend(statistical_plots)

        # Generate histogram plots
        histogram_plots = create_histogram_plots(excel_data, plot_config, save_folder)
        all_plots.extend(histogram_plots)

        return all_plots

    except Exception as e:
        print(f"Error during statistical/histogram plot generation: {e}")
        return []