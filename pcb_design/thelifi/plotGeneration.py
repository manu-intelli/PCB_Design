import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
import json
import glob
import os
from pathlib import Path
import skrf as rf
import seaborn as sns
from scipy import signal
import warnings

warnings.filterwarnings("ignore")


def load_configuration(plot_config_data, kpi_config_data=None):
    """
    Load and validate plot and KPI configuration data.

    Args:
        plot_config_data (dict): Plot configuration dictionary (required).
        kpi_config_data (dict, optional): KPI configuration dictionary. Defaults to empty dict if not provided.

    Returns:
        tuple: (plot_config_data, kpi_config_data)

    Raises:
        ValueError: If plot_config_data is not provided.
    """
    if not plot_config_data:
        raise ValueError("Plot configuration data is required.")
    if kpi_config_data is None:
        kpi_config_data = {}
    return plot_config_data, kpi_config_data


def load_excel_data(excel_files):
    """
    Load 'Per_File' and 'Summary' sheets from the first Excel file in the list.

    Args:
        excel_files (list): List of Excel file paths.

    Returns:
        dict: Dictionary with keys 'Per_File' and 'Summary' containing DataFrames.

    Raises:
        FileNotFoundError: If no Excel files are provided.
        ValueError: If required sheets are missing in the Excel file.
    """
    if not excel_files:
        raise FileNotFoundError(
            "SParam_Summary.xlsx not found in the provided folders."
        )
    try:
        per_file_data = pd.read_excel(excel_files[0], sheet_name="Per_File")
        summary_data = pd.read_excel(excel_files[0], sheet_name="Summary")
    except ValueError as e:
        raise ValueError(f"Required sheet missing in Excel file: {e}")
    return {"Per_File": per_file_data, "Summary": summary_data}


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
    """
    Load simulation S2P and sigma CSV files.

    Args:
        sim_s2p_files (list): List of simulation S2P file paths.
        s11_sigma_files (list): List of S11 sigma CSV file paths.
        s21_sigma_files (list): List of S21 sigma CSV file paths.

    Returns:
        dict: Dictionary with keys 'nominal', 's11_sigma', 's21_sigma' as available.
    """
    sim_data = {}
    if sim_s2p_files:
        try:
            sim_data["nominal"] = rf.Network(sim_s2p_files[0])
        except Exception as e:
            print(f"Warning: Could not load simulation S2P: {e}")
    if s11_sigma_files:
        try:
            sim_data["s11_sigma"] = pd.read_csv(s11_sigma_files[0])
        except Exception as e:
            print(f"Warning: Could not load S11 sigma data: {e}")
    if s21_sigma_files:
        try:
            sim_data["s21_sigma"] = pd.read_csv(s21_sigma_files[0])
        except Exception as e:
            print(f"Warning: Could not load S21 sigma data: {e}")
    return sim_data


def apply_frequency_shift(network, shift_mhz):
    """
    Apply a frequency shift to a scikit-rf Network object.

    Args:
        network (rf.Network): The original network.
        shift_mhz (float): Frequency shift in MHz.

    Returns:
        rf.Network: New network with shifted frequency.
    """
    if shift_mhz == 0:
        return network
    shifted_network = network.copy()
    original_freq_hz = network.frequency.f
    shifted_freq_hz = original_freq_hz * (1 + shift_mhz / 100.0)
    new_frequency = rf.Frequency.from_f(shifted_freq_hz, unit="Hz")
    shifted_network = rf.Network(
        frequency=new_frequency, s=network.s, name=network.name
    )
    return shifted_network


def calculate_freq_shifts(plot_config):
    """
    Calculate frequency shifts, always including 0 MHz.

    Args:
        plot_config (dict): Plot configuration dictionary.

    Returns:
        list: List of frequency shifts (MHz), always starting with 0.
    """
    if plot_config["frequency_shifts"]["enabled"]:
        user_shifts = plot_config["frequency_shifts"]["shifts"]
        freq_shifts = [0] + [s for s in user_shifts if s != 0]
        max_allowed = plot_config["frequency_shifts"]["max_shifts"]
        if len(freq_shifts) > max_allowed:
            print(
                f"Warning: Too many shifts specified. Using first {max_allowed} shifts including 0."
            )
            freq_shifts = freq_shifts[:max_allowed]
    else:
        freq_shifts = [0]
    return freq_shifts


# =============================================================================
# INDIVIDUAL S-PARAMETER PLOT FUNCTIONS
# =============================================================================
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from PIL import Image

def extract_unique_ids(all_labels):
    from os.path import commonprefix

    simplified_labels = []

    # Get only actual file labels, exclude shifts
    file_labels = [label for label in all_labels if label not in ['+Shift', '-Shift']]

    # Find the common prefix across all file labels
    common_text = os.path.commonprefix(file_labels)

    for label in all_labels:
        if label in ['+Shift', '-Shift']:
            simplified_labels.append(label)
        else:
            # Unique part is what comes after the common prefix
            unique_id = label.replace(common_text, '').strip('_').strip('-').strip()

            # Optional: If the unique part is too big, just take the first 5-6 characters
            if len(unique_id) > 6:
                unique_id = unique_id[:6]

            # Take first 4 letters from common text as prefix
            prefix = common_text[:4]
            simplified_labels.append(f"{prefix}_{unique_id}")

    return simplified_labels

def generate_s11_return_loss_plot(
    networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder
):
    """Generate lightweight S11 Return Loss Plot with file names and shift lines in legend."""
    plt.figure(figsize=(20, 9))

    num_files = len(networks)
    color_norm = mcolors.Normalize(vmin=0, vmax=num_files - 1)
    scalar_map = cm.ScalarMappable(norm=color_norm, cmap="tab20")

    all_handles = []
    all_labels = []

    positive_shift_plotted = False
    negative_shift_plotted = False

    for idx, (filename, network) in enumerate(networks.items()):
        unique_color = scalar_map.to_rgba(idx)

        # Plot file name once with nominal (center) line
        shifted_net = apply_frequency_shift(network, 0)
        freq_mhz = shifted_net.frequency.f / 1e6
        s11_db = 20 * np.log10(np.abs(shifted_net.s[:, 0, 0]))
        (line,) = plt.plot(freq_mhz, s11_db, alpha=0.7, color=unique_color)
        all_handles.append(line)
        all_labels.append(filename)


    # You already have all_labels filled
    


        # Plot shifted lines without individual file names
        for shift in freq_shifts:
            if shift == 0:
                continue

            shifted_net = apply_frequency_shift(network, shift)
            freq_mhz = shifted_net.frequency.f / 1e6
            s11_db = 20 * np.log10(np.abs(shifted_net.s[:, 0, 0]))

            if shift > 0:
                (line,) = plt.plot(
                    freq_mhz, s11_db, alpha=0.7, color=unique_color, linestyle="--"
                )
                if not positive_shift_plotted:
                    all_handles.append(line)
                    all_labels.append("+Shift")
                    positive_shift_plotted = True
            else:
                (line,) = plt.plot(
                    freq_mhz, s11_db, alpha=0.7, color=unique_color, linestyle=":"
                )
                if not negative_shift_plotted:
                    all_handles.append(line)
                    all_labels.append("-Shift")
                    negative_shift_plotted = True
    
    all_labels = extract_unique_ids(all_labels)

    # Sigma curves if enabled
    if (
        plot_config["sigma_settings"]["enabled"]
        and plot_config["sigma_settings"]["sigma_values"]
    ):
        print("Calculating S11 sigma curves...")
        first_network = list(networks.values())[0]
        ref_freq_hz = first_network.frequency.f
        ref_freq_mhz = ref_freq_hz / 1e6

        s11_data_collection = []
        for filename, network in networks.items():
            shifted_net = apply_frequency_shift(network, 0)
            if len(shifted_net.frequency.f) == len(ref_freq_hz):
                s11_db = 20 * np.log10(np.abs(shifted_net.s[:, 0, 0]))
                s11_data_collection.append(s11_db)
            else:
                freq_mhz = shifted_net.frequency.f / 1e6
                s11_db = 20 * np.log10(np.abs(shifted_net.s[:, 0, 0]))
                s11_interp = np.interp(ref_freq_mhz, freq_mhz, s11_db)
                s11_data_collection.append(s11_interp)

        if len(s11_data_collection) > 1:
            s11_array = np.array(s11_data_collection)
            s11_mean = np.mean(s11_array, axis=0)
            s11_std = np.std(s11_array, axis=0, ddof=1)

            sigma_colors = ["purple", "orange", "brown"]
            sigma_styles = ["-", "--", ":"]

            for i, sigma_val in enumerate(
                plot_config["sigma_settings"]["sigma_values"]
            ):
                if i >= 3:
                    break
                color = sigma_colors[i % len(sigma_colors)]
                style = sigma_styles[i % len(sigma_styles)]

                s11_upper = s11_mean + sigma_val * s11_std
                s11_lower = s11_mean - sigma_val * s11_std

                (upper_line,) = plt.plot(
                    ref_freq_mhz,
                    s11_upper,
                    color=color,
                    linestyle=style,
                    linewidth=2.5,
                    alpha=0.8,
                    label=f"Mean+{sigma_val}σ S11",
                )
                (lower_line,) = plt.plot(
                    ref_freq_mhz,
                    s11_lower,
                    color=color,
                    linestyle=style,
                    linewidth=2.5,
                    alpha=0.8,
                    label=f"Mean-{sigma_val}σ S11",
                )

                all_handles.append(upper_line)
                all_labels.append(f"Mean+{sigma_val}σ S11")
                all_handles.append(lower_line)
                all_labels.append(f"Mean-{sigma_val}σ S11")

                if i == 0:
                    (mean_line,) = plt.plot(
                        ref_freq_mhz,
                        s11_mean,
                        color="black",
                        linestyle="-",
                        linewidth=2,
                        alpha=0.9,
                        label="Mean S11",
                    )
                    all_handles.append(mean_line)
                    all_labels.append("Mean S11")
        else:
            print("Warning: Need at least 2 measurement files for sigma calculations.")

    # Simulation overlay if available
    if (
        sim_data
        and "nominal" in sim_data
        and plot_config["simulation_settings"]["include_simulation"]
    ):
        sim_freq_mhz = sim_data["nominal"].frequency.f / 1e6
        sim_s11_db = 20 * np.log10(np.abs(sim_data["nominal"].s[:, 0, 0]))
        (sim_line,) = plt.plot(
            sim_freq_mhz, sim_s11_db, "r--", linewidth=2, label="Simulation Nominal"
        )
        all_handles.append(sim_line)
        all_labels.append("Simulation Nominal")

    # RL spec limits for S11
    rl_colors = ["red", "darkred", "crimson"]
    for i, rl_spec in enumerate(kpi_config["KPIs"].get("RL", [])):
        freq_start = rl_spec["range"][0] / 1e6
        freq_end = rl_spec["range"][1] / 1e6
        lsl_value = -abs(rl_spec["LSL"])
        color = rl_colors[i % len(rl_colors)]
        freq_range = np.linspace(freq_start, freq_end, 100)

        if plot_config.get("plot_settings", {}).get("show_spec_lines", False):
            (rl_line,) = plt.plot(
                freq_range,
                [lsl_value] * len(freq_range),
                color=color,
                linestyle="--",
                linewidth=2,
                label=f'{rl_spec["name"]} LSL: {lsl_value} dB',
            )
            all_handles.append(rl_line)
            all_labels.append(f'{rl_spec["name"]} LSL: {lsl_value} dB')

        plt.axvspan(freq_start, freq_end, alpha=0.05, color=color)

    # Configure plot
    config = plot_config["axis_ranges"]["s_parameter_plots"]["s11_return_loss"]
    plt.xlim(config["x_axis"]["min"], config["x_axis"]["max"])
    plt.ylim(config["y_axis"]["min"], config["y_axis"]["max"])
    plt.xlabel(f"Frequency ({config['x_axis']['unit']})")
    plt.ylabel(f"S11 ({config['y_axis']['unit']})")
    plt.title("S11 Return Loss vs Frequency")
    plt.grid(True, alpha=0.3)

    plt.legend(
        all_handles,
        all_labels,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize="small",
        ncol=2,
        borderaxespad=0.1,
    )

    plt.tight_layout()

    # Save temporary PNG
    temp_png_path = os.path.join(save_folder, "temp_s11_plot.png")
    plt.savefig(temp_png_path, dpi=100, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    # Compress and save as lightweight JPG
    final_path = os.path.join(save_folder, "S11_Return_Loss.jpg")
    image = Image.open(temp_png_path)
    image = image.convert("RGB")  # Ensure JPG format
    image.save(final_path, format="JPEG", quality=70, optimize=True)

    # Remove temp PNG
    os.remove(temp_png_path)

    return "S11_Return_Loss.jpg"

def generate_s22_return_loss_plot(networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder):
    """Generate lightweight S22 Return Loss Plot with file names and shift lines in legend."""
    plt.figure(figsize=(20, 9))

    num_files = len(networks)
    color_norm = mcolors.Normalize(vmin=0, vmax=num_files - 1)
    scalar_map = cm.ScalarMappable(norm=color_norm, cmap="tab20")

    all_handles = []
    all_labels = []

    positive_shift_plotted = False
    negative_shift_plotted = False

    for idx, (filename, network) in enumerate(networks.items()):
        unique_color = scalar_map.to_rgba(idx)

        # Plot file name once with nominal (center) line
        shifted_net = apply_frequency_shift(network, 0)
        freq_mhz = shifted_net.frequency.f / 1e6
        s22_db = 20 * np.log10(np.abs(shifted_net.s[:, 1, 1]))
        (line,) = plt.plot(freq_mhz, s22_db, alpha=0.7, color=unique_color)
        all_handles.append(line)
        all_labels.append(filename)

        # Plot shifted lines without individual file names
        for shift in freq_shifts:
            if shift == 0:
                continue

            shifted_net = apply_frequency_shift(network, shift)
            freq_mhz = shifted_net.frequency.f / 1e6
            s22_db = 20 * np.log10(np.abs(shifted_net.s[:, 1, 1]))

            if shift > 0:
                (line,) = plt.plot(freq_mhz, s22_db, alpha=0.7, color=unique_color, linestyle="--")
                if not positive_shift_plotted:
                    all_handles.append(line)
                    all_labels.append("+Shift")
                    positive_shift_plotted = True
            else:
                (line,) = plt.plot(freq_mhz, s22_db, alpha=0.7, color=unique_color, linestyle=":")
                if not negative_shift_plotted:
                    all_handles.append(line)
                    all_labels.append("-Shift")
                    negative_shift_plotted = True
    all_labels = extract_unique_ids(all_labels)
    # Sigma curves if enabled
    if plot_config["sigma_settings"]["enabled"] and plot_config["sigma_settings"]["sigma_values"]:
        print("Calculating S22 sigma curves...")
        first_network = list(networks.values())[0]
        ref_freq_hz = first_network.frequency.f
        ref_freq_mhz = ref_freq_hz / 1e6

        s22_data_collection = []
        for filename, network in networks.items():
            shifted_net = apply_frequency_shift(network, 0)
            if len(shifted_net.frequency.f) == len(ref_freq_hz):
                s22_db = 20 * np.log10(np.abs(shifted_net.s[:, 1, 1]))
                s22_data_collection.append(s22_db)
            else:
                freq_mhz = shifted_net.frequency.f / 1e6
                s22_db = 20 * np.log10(np.abs(shifted_net.s[:, 1, 1]))
                s22_interp = np.interp(ref_freq_mhz, freq_mhz, s22_db)
                s22_data_collection.append(s22_interp)

        if len(s22_data_collection) > 1:
            s22_array = np.array(s22_data_collection)
            s22_mean = np.mean(s22_array, axis=0)
            s22_std = np.std(s22_array, axis=0, ddof=1)

            sigma_colors = ["purple", "orange", "brown"]
            sigma_styles = ["-", "--", ":"]

            for i, sigma_val in enumerate(plot_config["sigma_settings"]["sigma_values"]):
                if i >= 3:
                    break
                color = sigma_colors[i % len(sigma_colors)]
                style = sigma_styles[i % len(sigma_styles)]

                s22_upper = s22_mean + sigma_val * s22_std
                s22_lower = s22_mean - sigma_val * s22_std

                (upper_line,) = plt.plot(ref_freq_mhz, s22_upper, color=color, linestyle=style, linewidth=2.5, alpha=0.8, label=f"Mean+{sigma_val}σ S22")
                (lower_line,) = plt.plot(ref_freq_mhz, s22_lower, color=color, linestyle=style, linewidth=2.5, alpha=0.8, label=f"Mean-{sigma_val}σ S22")

                all_handles.append(upper_line)
                all_labels.append(f"Mean+{sigma_val}σ S22")
                all_handles.append(lower_line)
                all_labels.append(f"Mean-{sigma_val}σ S22")

                if i == 0:
                    (mean_line,) = plt.plot(ref_freq_mhz, s22_mean, color="black", linestyle="-", linewidth=2, alpha=0.9, label="Mean S22")
                    all_handles.append(mean_line)
                    all_labels.append("Mean S22")
        else:
            print("Warning: Need at least 2 measurement files for sigma calculations.")

    # Simulation overlay if available
    if sim_data and "nominal" in sim_data and plot_config["simulation_settings"]["include_simulation"]:
        sim_freq_mhz = sim_data["nominal"].frequency.f / 1e6
        sim_s22_db = 20 * np.log10(np.abs(sim_data["nominal"].s[:, 1, 1]))
        (sim_line,) = plt.plot(sim_freq_mhz, sim_s22_db, "r--", linewidth=2, label="Simulation Nominal")
        all_handles.append(sim_line)
        all_labels.append("Simulation Nominal")

    # RL spec limits for S22
    rl_colors = ["red", "darkred", "crimson"]
    for i, rl_spec in enumerate(kpi_config["KPIs"].get("RL", [])):
        freq_start = rl_spec["range"][0] / 1e6
        freq_end = rl_spec["range"][1] / 1e6
        lsl_value = -abs(rl_spec["LSL"])
        color = rl_colors[i % len(rl_colors)]
        freq_range = np.linspace(freq_start, freq_end, 100)

        if plot_config.get("plot_settings", {}).get("show_spec_lines", False):
            (rl_line,) = plt.plot(freq_range, [lsl_value] * len(freq_range), color=color, linestyle="--", linewidth=2, label=f'{rl_spec["name"]} LSL: {lsl_value} dB')
            all_handles.append(rl_line)
            all_labels.append(f'{rl_spec["name"]} LSL: {lsl_value} dB')

        plt.axvspan(freq_start, freq_end, alpha=0.05, color=color)

    # Configure plot
    config = plot_config["axis_ranges"]["s_parameter_plots"]["s22_return_loss"]
    plt.xlim(config["x_axis"]["min"], config["x_axis"]["max"])
    plt.ylim(config["y_axis"]["min"], config["y_axis"]["max"])
    plt.xlabel(f"Frequency ({config['x_axis']['unit']})")
    plt.ylabel(f"S22 ({config['y_axis']['unit']})")
    plt.title("S22 Return Loss vs Frequency")
    plt.grid(True, alpha=0.3)

    plt.legend(all_handles, all_labels, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small", ncol=2, borderaxespad=0.1)

    plt.tight_layout()

    # Save temporary PNG
    temp_png_path = os.path.join(save_folder, "temp_s22_plot.png")
    plt.savefig(temp_png_path, dpi=100, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    # Compress and save as lightweight JPG
    final_path = os.path.join(save_folder, "S22_Return_Loss.jpg")
    image = Image.open(temp_png_path)
    image = image.convert("RGB")  # Ensure JPG format
    image.save(final_path, format="JPEG", quality=70, optimize=True)

    # Remove temp PNG
    os.remove(temp_png_path)

    return "S22_Return_Loss.jpg"

def generate_s21_insertion_loss_plot(networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder):
    """Generate lightweight S21 Insertion Loss Plot with file names and shift lines in legend."""
    plt.figure(figsize=(20, 9))

    num_files = len(networks)
    color_norm = mcolors.Normalize(vmin=0, vmax=num_files - 1)
    scalar_map = cm.ScalarMappable(norm=color_norm, cmap="tab20")

    all_handles = []
    all_labels = []

    positive_shift_plotted = False
    negative_shift_plotted = False

    for idx, (filename, network) in enumerate(networks.items()):
        unique_color = scalar_map.to_rgba(idx)

        # Plot file name once with nominal (center) line
        shifted_net = apply_frequency_shift(network, 0)
        freq_mhz = shifted_net.frequency.f / 1e6
        s21_db = 20 * np.log10(np.abs(shifted_net.s[:, 1, 0]))
        (line,) = plt.plot(freq_mhz, s21_db, alpha=0.7, color=unique_color)
        all_handles.append(line)
        all_labels.append(filename)

        # Plot shifted lines without individual file names
        for shift in freq_shifts:
            if shift == 0:
                continue

            shifted_net = apply_frequency_shift(network, shift)
            freq_mhz = shifted_net.frequency.f / 1e6
            s21_db = 20 * np.log10(np.abs(shifted_net.s[:, 1, 0]))

            if shift > 0:
                (line,) = plt.plot(freq_mhz, s21_db, alpha=0.7, color=unique_color, linestyle="--")
                if not positive_shift_plotted:
                    all_handles.append(line)
                    all_labels.append("+Shift")
                    positive_shift_plotted = True
            else:
                (line,) = plt.plot(freq_mhz, s21_db, alpha=0.7, color=unique_color, linestyle=":")
                if not negative_shift_plotted:
                    all_handles.append(line)
                    all_labels.append("-Shift")
                    negative_shift_plotted = True
    all_labels = extract_unique_ids(all_labels)
    # Sigma curves if enabled
    if plot_config["sigma_settings"]["enabled"] and plot_config["sigma_settings"]["sigma_values"]:
        print("Calculating S21 sigma curves...")
        first_network = list(networks.values())[0]
        ref_freq_hz = first_network.frequency.f
        ref_freq_mhz = ref_freq_hz / 1e6

        s21_data_collection = []
        for filename, network in networks.items():
            shifted_net = apply_frequency_shift(network, 0)
            if len(shifted_net.frequency.f) == len(ref_freq_hz):
                s21_db = 20 * np.log10(np.abs(shifted_net.s[:, 1, 0]))
                s21_data_collection.append(s21_db)
            else:
                freq_mhz = shifted_net.frequency.f / 1e6
                s21_db = 20 * np.log10(np.abs(shifted_net.s[:, 1, 0]))
                s21_interp = np.interp(ref_freq_mhz, freq_mhz, s21_db)
                s21_data_collection.append(s21_interp)

        if len(s21_data_collection) > 1:
            s21_array = np.array(s21_data_collection)
            s21_mean = np.mean(s21_array, axis=0)
            s21_std = np.std(s21_array, axis=0, ddof=1)

            sigma_colors = ["purple", "orange", "brown"]
            sigma_styles = ["-", "--", ":"]

            for i, sigma_val in enumerate(plot_config["sigma_settings"]["sigma_values"]):
                if i >= 3:
                    break
                color = sigma_colors[i % len(sigma_colors)]
                style = sigma_styles[i % len(sigma_styles)]

                s21_upper = s21_mean + sigma_val * s21_std
                s21_lower = s21_mean - sigma_val * s21_std

                (upper_line,) = plt.plot(ref_freq_mhz, s21_upper, color=color, linestyle=style, linewidth=2.5, alpha=0.8, label=f"Mean+{sigma_val}σ S21")
                (lower_line,) = plt.plot(ref_freq_mhz, s21_lower, color=color, linestyle=style, linewidth=2.5, alpha=0.8, label=f"Mean-{sigma_val}σ S21")

                all_handles.append(upper_line)
                all_labels.append(f"Mean+{sigma_val}σ S21")
                all_handles.append(lower_line)
                all_labels.append(f"Mean-{sigma_val}σ S21")

                if i == 0:
                    (mean_line,) = plt.plot(ref_freq_mhz, s21_mean, color="black", linestyle="-", linewidth=2, alpha=0.9, label="Mean S21")
                    all_handles.append(mean_line)
                    all_labels.append("Mean S21")
        else:
            print("Warning: Need at least 2 measurement files for sigma calculations.")

    # Simulation overlay if available
    if sim_data and "nominal" in sim_data and plot_config["simulation_settings"]["include_simulation"]:
        sim_freq_mhz = sim_data["nominal"].frequency.f / 1e6
        sim_s21_db = 20 * np.log10(np.abs(sim_data["nominal"].s[:, 1, 0]))
        (sim_line,) = plt.plot(sim_freq_mhz, sim_s21_db, "r--", linewidth=2, label="Simulation Nominal")
        all_handles.append(sim_line)
        all_labels.append("Simulation Nominal")

    # Simulation sigma curves if available
    if (sim_data and "nominal" in sim_data and "s21_sigma" in sim_data and plot_config["simulation_settings"]["include_simulation"]
            and plot_config["sigma_settings"]["enabled"] and plot_config["sigma_settings"]["sigma_values"]):
        print("Adding S21 simulation sigma curves...")
        sim_freq_mhz = sim_data["nominal"].frequency.f / 1e6
        sim_s21_db = 20 * np.log10(np.abs(sim_data["nominal"].s[:, 1, 0]))
        s21_sigma_df = sim_data["s21_sigma"]

        if "Freq (MHz)" in s21_sigma_df.columns and any("stdev" in col.lower() for col in s21_sigma_df.columns):
            stdev_col = next(col for col in s21_sigma_df.columns if "stdev" in col.lower())
            sigma_freq_mhz = s21_sigma_df["Freq (MHz)"].values
            sigma_stdev_db = s21_sigma_df[stdev_col].values
            sim_s21_sigma = np.interp(sim_freq_mhz, sigma_freq_mhz, sigma_stdev_db)
            sim_sigma_colors = ["purple", "darkviolet", "mediumorchid"]

            for i, sigma_val in enumerate(plot_config["sigma_settings"]["sigma_values"]):
                if i >= 3:
                    break
                color = sim_sigma_colors[i % len(sim_sigma_colors)]
                sim_s21_upper = sim_s21_db + sigma_val * sim_s21_sigma
                sim_s21_lower = sim_s21_db - sigma_val * sim_s21_sigma

                (upper_line,) = plt.plot(sim_freq_mhz, sim_s21_upper, color=color, linestyle="-.", linewidth=2, alpha=0.8, label=f"Sim Mean+{sigma_val}σ S21")
                (lower_line,) = plt.plot(sim_freq_mhz, sim_s21_lower, color=color, linestyle="-.", linewidth=2, alpha=0.8, label=f"Sim Mean-{sigma_val}σ S21")

                all_handles.append(upper_line)
                all_labels.append(f"Sim Mean+{sigma_val}σ S21")
                all_handles.append(lower_line)
                all_labels.append(f"Sim Mean-{sigma_val}σ S21")
        else:
            print("Warning: S21 sigma data format not recognized. Expected 'Freq (MHz)' and 'stdev*' columns.")

    # IL spec limits
    il_colors = ["green", "darkgreen", "lime"]
    for i, il_spec in enumerate(kpi_config["KPIs"].get("IL", [])):
        freq_start = il_spec["range"][0] / 1e6
        freq_end = il_spec["range"][1] / 1e6
        usl_value = -abs(il_spec["USL"])
        color = il_colors[i % len(il_colors)]
        freq_range = np.linspace(freq_start, freq_end, 100)

        if plot_config.get("plot_settings", {}).get("show_spec_lines", False):
            (il_line,) = plt.plot(freq_range, [usl_value] * len(freq_range), color=color, linestyle="--", linewidth=2, label=f'{il_spec["name"]} USL: {usl_value} dB')
            all_handles.append(il_line)
            all_labels.append(f'{il_spec["name"]} USL: {usl_value} dB')

        plt.axvspan(freq_start, freq_end, alpha=0.05, color=color)

    # Configure plot
    config = plot_config["axis_ranges"]["s_parameter_plots"]["s21_insertion_loss"]
    plt.xlim(config["x_axis"]["min"], config["x_axis"]["max"])
    plt.ylim(config["y_axis"]["min"], config["y_axis"]["max"])
    plt.xlabel(f"Frequency ({config['x_axis']['unit']})")
    plt.ylabel(f"S21 ({config['y_axis']['unit']})")
    plt.title("S21 Insertion Loss vs Frequency (Pass-band)")
    plt.grid(True, alpha=0.3)

    plt.legend(all_handles, all_labels, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small", ncol=2, borderaxespad=0.1)

    plt.tight_layout()

    # Save temporary PNG
    temp_png_path = os.path.join(save_folder, "temp_s21_plot.png")
    plt.savefig(temp_png_path, dpi=100, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    # Compress and save as lightweight JPG
    final_path = os.path.join(save_folder, "S21_Insertion_Loss.jpg")
    image = Image.open(temp_png_path)
    image = image.convert("RGB")  # Ensure JPG format
    image.save(final_path, format="JPEG", quality=70, optimize=True)

    # Remove temp PNG
    os.remove(temp_png_path)

    return "S21_Insertion_Loss.jpg"

def generate_s21_rejection_loss_plot(networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder):
    """Generate lightweight S21 Rejection Loss Plot with simplified legend (file names and shifts)."""
    plt.figure(figsize=(20, 9))

    num_files = len(networks)
    color_norm = mcolors.Normalize(vmin=0, vmax=num_files - 1)
    scalar_map = cm.ScalarMappable(norm=color_norm, cmap="tab20")

    all_handles = []
    all_labels = []

    positive_shift_plotted = False
    negative_shift_plotted = False

    for idx, (filename, network) in enumerate(networks.items()):
        unique_color = scalar_map.to_rgba(idx)

        # Plot file name once with nominal line
        shifted_net = apply_frequency_shift(network, 0)
        freq_mhz = shifted_net.frequency.f / 1e6
        s21_db = 20 * np.log10(np.abs(shifted_net.s[:, 1, 0]))
        (line,) = plt.plot(freq_mhz, s21_db, alpha=0.7, color=unique_color)
        all_handles.append(line)
        all_labels.append(filename)

        # Plot shifted lines with common +Shift / -Shift labels
        for shift in freq_shifts:
            if shift == 0:
                continue

            shifted_net = apply_frequency_shift(network, shift)
            freq_mhz = shifted_net.frequency.f / 1e6
            s21_db = 20 * np.log10(np.abs(shifted_net.s[:, 1, 0]))

            if shift > 0:
                (line,) = plt.plot(freq_mhz, s21_db, alpha=0.7, color=unique_color, linestyle="--")
                if not positive_shift_plotted:
                    all_handles.append(line)
                    all_labels.append("+Shift")
                    positive_shift_plotted = True
            else:
                (line,) = plt.plot(freq_mhz, s21_db, alpha=0.7, color=unique_color, linestyle=":")
                if not negative_shift_plotted:
                    all_handles.append(line)
                    all_labels.append("-Shift")
                    negative_shift_plotted = True
    all_labels = extract_unique_ids(all_labels)
    # Sigma curves if enabled
    if plot_config["sigma_settings"]["enabled"] and plot_config["sigma_settings"]["sigma_values"]:
        print("Calculating S21 rejection loss sigma curves...")
        first_network = list(networks.values())[0]
        ref_freq_hz = first_network.frequency.f
        ref_freq_mhz = ref_freq_hz / 1e6

        s21_data_collection = []
        for filename, network in networks.items():
            shifted_net = apply_frequency_shift(network, 0)
            if len(shifted_net.frequency.f) == len(ref_freq_hz):
                s21_db = 20 * np.log10(np.abs(shifted_net.s[:, 1, 0]))
                s21_data_collection.append(s21_db)
            else:
                freq_mhz = shifted_net.frequency.f / 1e6
                s21_db = 20 * np.log10(np.abs(shifted_net.s[:, 1, 0]))
                s21_interp = np.interp(ref_freq_mhz, freq_mhz, s21_db)
                s21_data_collection.append(s21_interp)

        if len(s21_data_collection) > 1:
            s21_array = np.array(s21_data_collection)
            s21_mean = np.mean(s21_array, axis=0)
            s21_std = np.std(s21_array, axis=0, ddof=1)

            sigma_colors = ["purple", "orange", "brown"]
            sigma_styles = ["-", "--", ":"]

            for i, sigma_val in enumerate(plot_config["sigma_settings"]["sigma_values"]):
                if i >= 3:
                    break
                color = sigma_colors[i % len(sigma_colors)]
                style = sigma_styles[i % len(sigma_styles)]

                s21_upper = s21_mean + sigma_val * s21_std
                s21_lower = s21_mean - sigma_val * s21_std

                (upper_line,) = plt.plot(ref_freq_mhz, s21_upper, color=color, linestyle=style, linewidth=1.5, alpha=0.8, label=f"Mean+{sigma_val}σ S21 Rej")
                (lower_line,) = plt.plot(ref_freq_mhz, s21_lower, color=color, linestyle=style, linewidth=1.5, alpha=0.8, label=f"Mean-{sigma_val}σ S21 Rej")

                all_handles.append(upper_line)
                all_labels.append(f"Mean+{sigma_val}σ S21 Rej")
                all_handles.append(lower_line)
                all_labels.append(f"Mean-{sigma_val}σ S21 Rej")

                if i == 0:
                    (mean_line,) = plt.plot(ref_freq_mhz, s21_mean, color="black", linestyle="-", linewidth=2, alpha=0.9, label="Mean S21 Rej")
                    all_handles.append(mean_line)
                    all_labels.append("Mean S21 Rej")
        else:
            print("Warning: Need at least 2 measurement files for sigma calculations.")

    # Simulation overlay if available
    if sim_data and "nominal" in sim_data and plot_config["simulation_settings"]["include_simulation"]:
        sim_freq_mhz = sim_data["nominal"].frequency.f / 1e6
        sim_s21_db = 20 * np.log10(np.abs(sim_data["nominal"].s[:, 1, 0]))
        (sim_line,) = plt.plot(sim_freq_mhz, sim_s21_db, "r--", linewidth=2, label="Simulation Nominal")
        all_handles.append(sim_line)
        all_labels.append("Simulation Nominal")

    # Simulation sigma curves if available
    if (sim_data and "nominal" in sim_data and "s21_sigma" in sim_data and plot_config["simulation_settings"]["include_simulation"]
            and plot_config["sigma_settings"]["enabled"] and plot_config["sigma_settings"]["sigma_values"]):
        print("Adding S21 simulation sigma curves...")
        sim_freq_mhz = sim_data["nominal"].frequency.f / 1e6
        sim_s21_db = 20 * np.log10(np.abs(sim_data["nominal"].s[:, 1, 0]))
        s21_sigma_df = sim_data["s21_sigma"]

        if "Freq (MHz)" in s21_sigma_df.columns and any("stdev" in col.lower() for col in s21_sigma_df.columns):
            stdev_col = next(col for col in s21_sigma_df.columns if "stdev" in col.lower())
            sigma_freq_mhz = s21_sigma_df["Freq (MHz)"].values
            sigma_stdev_db = s21_sigma_df[stdev_col].values
            sim_s21_sigma = np.interp(sim_freq_mhz, sigma_freq_mhz, sigma_stdev_db)
            sim_sigma_colors = ["purple", "darkviolet", "mediumorchid"]

            for i, sigma_val in enumerate(plot_config["sigma_settings"]["sigma_values"]):
                if i >= 3:
                    break
                color = sim_sigma_colors[i % len(sim_sigma_colors)]
                sim_s21_upper = sim_s21_db + sigma_val * sim_s21_sigma
                sim_s21_lower = sim_s21_db - sigma_val * sim_s21_sigma

                (upper_line,) = plt.plot(sim_freq_mhz, sim_s21_upper, color=color, linestyle="-.", linewidth=2, alpha=0.8, label=f"Sim Mean+{sigma_val}σ S21 Rej")
                (lower_line,) = plt.plot(sim_freq_mhz, sim_s21_lower, color=color, linestyle="-.", linewidth=2, alpha=0.8, label=f"Sim Mean-{sigma_val}σ S21 Rej")

                all_handles.append(upper_line)
                all_labels.append(f"Sim Mean+{sigma_val}σ S21 Rej")
                all_handles.append(lower_line)
                all_labels.append(f"Sim Mean-{sigma_val}σ S21 Rej")
        else:
            print("Warning: S21 sigma data format not recognized. Expected 'Freq (MHz)' and 'stdev*' columns.")

    # Add stopband rejection spec limits
    rej_colors = ["orange", "purple", "brown", "pink"]
    for i, sb_spec in enumerate(kpi_config.get("StopBands", [])):
        freq_start = sb_spec["range"][0] / 1e6
        freq_end = sb_spec["range"][1] / 1e6
        lsl_value = -abs(sb_spec["LSL"])
        color = rej_colors[i % len(rej_colors)]
        freq_range = np.linspace(freq_start, freq_end, 100)

        if plot_config.get("plot_settings", {}).get("show_spec_lines", False):
            (rej_line,) = plt.plot(freq_range, [lsl_value] * len(freq_range), color=color, linestyle="--", linewidth=2, label=f'{sb_spec["name"]} LSL: {lsl_value} dB')
            all_handles.append(rej_line)
            all_labels.append(f'{sb_spec["name"]} LSL: {lsl_value} dB')

        plt.axvspan(freq_start, freq_end, alpha=0.1, color=color)

    # Configure plot
    config = plot_config["axis_ranges"]["s_parameter_plots"]["s21_rejection_loss"]
    plt.xlim(config["x_axis"]["min"], config["x_axis"]["max"])
    plt.ylim(config["y_axis"]["min"], config["y_axis"]["max"])
    plt.xlabel(f"Frequency ({config['x_axis']['unit']})")
    plt.ylabel(f"S21 ({config['y_axis']['unit']})")
    plt.title("S21 Rejection Loss vs Frequency (Stop-band)")
    plt.grid(True, alpha=0.3)

    plt.legend(all_handles, all_labels, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small", ncol=2, borderaxespad=0.1)

    plt.tight_layout()

    # Save temporary PNG
    temp_png_path = os.path.join(save_folder, "temp_s21_rejection_plot.png")
    plt.savefig(temp_png_path, dpi=100, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    # Compress and save as lightweight JPG
    final_path = os.path.join(save_folder, "S21_Rejection_Loss.jpg")
    image = Image.open(temp_png_path)
    image = image.convert("RGB")
    image.save(final_path, format="JPEG", quality=70, optimize=True)

    os.remove(temp_png_path)

    return "S21_Rejection_Loss.jpg"


# =============================================================================
# INDIVIDUAL ADVANCED PLOT FUNCTIONS
# =============================================================================


def generate_group_delay_plot(networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder):
    """Generate Group Delay Plot with simplified legend (file names and shifts)."""

    def phase_and_gd(network):
        f_hz = network.frequency.f
        ang = np.unwrap(np.angle(network.s[:, 1, 0]))
        df = np.gradient(f_hz)
        gd_s = -np.gradient(ang) / (2 * np.pi * (df + 1e-12))
        return ang, gd_s * 1e9  # Convert to ns

    plt.figure(figsize=(20, 9))

    num_files = len(networks)
    color_norm = mcolors.Normalize(vmin=0, vmax=num_files - 1)
    scalar_map = cm.ScalarMappable(norm=color_norm, cmap="tab20")

    all_handles = []
    all_labels = []

    positive_shift_plotted = False
    negative_shift_plotted = False

    for idx, (filename, nt) in enumerate(networks.items()):
        unique_color = scalar_map.to_rgba(idx)

        # Plot nominal file (label the file name)
        nt_shift = apply_frequency_shift(nt, 0)
        _, gd_ns = phase_and_gd(nt_shift)
        (line,) = plt.plot(nt_shift.frequency.f / 1e6, gd_ns, alpha=0.7, color=unique_color)
        all_handles.append(line)
        all_labels.append(filename)

        # Plot shifted files (common legend for +Shift and -Shift)
        for s in freq_shifts:
            if s == 0:
                continue

            nt_shift = apply_frequency_shift(nt, s)
            _, gd_ns = phase_and_gd(nt_shift)

            if s > 0:
                (line,) = plt.plot(nt_shift.frequency.f / 1e6, gd_ns, alpha=0.7, color=unique_color, linestyle="--")
                if not positive_shift_plotted:
                    all_handles.append(line)
                    all_labels.append("+Shift")
                    positive_shift_plotted = True
            else:
                (line,) = plt.plot(nt_shift.frequency.f / 1e6, gd_ns, alpha=0.7, color=unique_color, linestyle=":")
                if not negative_shift_plotted:
                    all_handles.append(line)
                    all_labels.append("-Shift")
                    negative_shift_plotted = True
    all_labels = extract_unique_ids(all_labels)
    # Sigma curves if enabled
    if plot_config["sigma_settings"]["enabled"] and plot_config["sigma_settings"]["sigma_values"]:
        print("Calculating Group Delay sigma curves...")
        first_network = list(networks.values())[0]
        ref_freq_hz = first_network.frequency.f
        ref_freq_mhz = ref_freq_hz / 1e6
        gd_data_collection = []

        for filename, network in networks.items():
            shifted_net = apply_frequency_shift(network, 0)
            _, gd_ns = phase_and_gd(shifted_net)
            if len(shifted_net.frequency.f) == len(ref_freq_hz):
                gd_data_collection.append(gd_ns)
            else:
                freq_mhz = shifted_net.frequency.f / 1e6
                gd_interp = np.interp(ref_freq_mhz, freq_mhz, gd_ns)
                gd_data_collection.append(gd_interp)

        if len(gd_data_collection) > 1:
            gd_array = np.array(gd_data_collection)
            gd_mean = np.mean(gd_array, axis=0)
            gd_std = np.std(gd_array, axis=0, ddof=1)

            sigma_colors = ["purple", "orange", "brown"]
            sigma_styles = ["-", "--", ":"]

            for i, sigma_val in enumerate(plot_config["sigma_settings"]["sigma_values"]):
                if i >= 3:
                    break
                color = sigma_colors[i % len(sigma_colors)]
                style = sigma_styles[i % len(sigma_styles)]
                gd_upper = gd_mean + sigma_val * gd_std
                gd_lower = gd_mean - sigma_val * gd_std

                (upper_line,) = plt.plot(ref_freq_mhz, gd_upper, color=color, linestyle=style, linewidth=1.5, alpha=0.8, label=f"Mean+{sigma_val}σ GD")
                (lower_line,) = plt.plot(ref_freq_mhz, gd_lower, color=color, linestyle=style, linewidth=1.5, alpha=0.8, label=f"Mean-{sigma_val}σ GD")

                all_handles.append(upper_line)
                all_labels.append(f"Mean+{sigma_val}σ GD")
                all_handles.append(lower_line)
                all_labels.append(f"Mean-{sigma_val}σ GD")

                if i == 0:
                    (mean_line,) = plt.plot(ref_freq_mhz, gd_mean, color="black", linestyle="-", linewidth=2, alpha=0.9, label="Mean GD")
                    all_handles.append(mean_line)
                    all_labels.append("Mean GD")
        else:
            print("Warning: Need at least 2 measurement files for sigma calculations.")

    # KPI spec limits
    gd_colors = ["purple", "darkviolet", "magenta"]
    if "GD" in kpi_config["KPIs"]:
        for i, gd_spec in enumerate(kpi_config["KPIs"]["GD"]):
            freq_start_mhz = gd_spec["range"][0] / 1e6
            freq_end_mhz = gd_spec["range"][1] / 1e6
            color = gd_colors[i % len(gd_colors)]
            freq_range_mhz = np.linspace(freq_start_mhz, freq_end_mhz, 100)

            if plot_config.get("plot_settings", {}).get("show_spec_lines", False):
                if "USL" in gd_spec:
                    (usl_line,) = plt.plot(freq_range_mhz, [gd_spec["USL"]] * len(freq_range_mhz), color=color, linestyle="--", linewidth=2, label=f'{gd_spec["name"]} USL: {gd_spec["USL"]} ns')
                    all_handles.append(usl_line)
                    all_labels.append(f'{gd_spec["name"]} USL: {gd_spec["USL"]} ns')
                if "LSL" in gd_spec:
                    (lsl_line,) = plt.plot(freq_range_mhz, [gd_spec["LSL"]] * len(freq_range_mhz), color=color, linestyle=":", linewidth=2, label=f'{gd_spec["name"]} LSL: {gd_spec["LSL"]} ns')
                    all_handles.append(lsl_line)
                    all_labels.append(f'{gd_spec["name"]} LSL: {gd_spec["LSL"]} ns')

            plt.axvspan(freq_start_mhz, freq_end_mhz, alpha=0.05, color=color)

    cfg = plot_config["axis_ranges"]["advanced_plots"]["group_delay"]
    plt.xlim(cfg["x_axis"]["min"], cfg["x_axis"]["max"])
    plt.ylim(cfg["y_axis"]["min"], cfg["y_axis"]["max"])
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("GD (ns)")
    plt.title("Group Delay (continuous)")
    plt.grid(True, alpha=0.3)

    plt.legend(all_handles, all_labels, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small", ncol=2, borderaxespad=0.1)
    plt.tight_layout()

    temp_png_path = os.path.join(save_folder, "temp_gd_curve.png")
    plt.savefig(temp_png_path, dpi=100, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    final_path = os.path.join(save_folder, "GD_Curve.jpg")
    image = Image.open(temp_png_path)
    image = image.convert("RGB")
    image.save(final_path, format="JPEG", quality=70, optimize=True)
    os.remove(temp_png_path)

    return "GD_Curve.jpg"


def generate_linear_phase_deviation_plot(networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder):
    """Generate Linear Phase Deviation Plot with simplified legend."""

    def phase_and_gd(network):
        f_hz = network.frequency.f
        ang = np.unwrap(np.angle(network.s[:, 1, 0]))
        df = np.gradient(f_hz)
        gd_s = -np.gradient(ang) / (2 * np.pi * (df + 1e-12))
        return ang, gd_s * 1e9

    plt.figure(figsize=(12, 6))
    LPD_FREQ_LOW, LPD_FREQ_HIGH = kpi_config["KPIs"]["LPD_MIN"][0]["range"]

    all_handles = []
    all_labels = []

    positive_shift_plotted = False
    negative_shift_plotted = False

    for fname, nt in networks.items():
        # Plot nominal (no shift) with file name
        nt_shift = apply_frequency_shift(nt, 0)
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
            f_band_mhz = f_hz[mask] / 1e6
            (line,) = plt.plot(f_band_mhz, lpd_band, alpha=0.7)
            all_handles.append(line)
            all_labels.append(fname)

        # Plot shifted curves with common shift legend
        for s in freq_shifts:
            if s == 0:
                continue

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
                f_band_mhz = f_hz[mask] / 1e6

                if s > 0:
                    (line,) = plt.plot(f_band_mhz, lpd_band, alpha=0.7, linestyle="--")
                    if not positive_shift_plotted:
                        all_handles.append(line)
                        all_labels.append("+Shift")
                        positive_shift_plotted = True
                else:
                    (line,) = plt.plot(f_band_mhz, lpd_band, alpha=0.7, linestyle=":")
                    if not negative_shift_plotted:
                        all_handles.append(line)
                        all_labels.append("-Shift")
                        negative_shift_plotted = True
    all_labels = extract_unique_ids(all_labels)
    # Add sigma curves for Linear Phase Deviation if enabled
    if plot_config["sigma_settings"]["enabled"] and plot_config["sigma_settings"]["sigma_values"]:
        print("Calculating Linear Phase Deviation sigma curves...")

        first_network = list(networks.values())[0]
        ref_freq_hz = first_network.frequency.f
        ref_freq_mhz = ref_freq_hz / 1e6

        mask = (ref_freq_hz >= LPD_FREQ_LOW) & (ref_freq_hz <= LPD_FREQ_HIGH)
        ref_freq_band_hz = ref_freq_hz[mask]
        ref_freq_band_mhz = ref_freq_band_hz / 1e6

        lpd_data_collection = []
        for filename, network in networks.items():
            shifted_net = apply_frequency_shift(network, 0)
            phase, _ = phase_and_gd(shifted_net)
            f_hz = shifted_net.frequency.f

            net_mask = (f_hz >= LPD_FREQ_LOW) & (f_hz <= LPD_FREQ_HIGH)

            if np.sum(net_mask) > 2:
                f_fit = f_hz[net_mask]
                phase_fit = phase[net_mask]

                A = np.vstack([f_fit, np.ones_like(f_fit)]).T
                slope, intercept = np.linalg.lstsq(A, phase_fit, rcond=None)[0]

                linear_phase = slope * f_hz + intercept
                phase_deviation_rad = phase - linear_phase
                phase_deviation_deg = np.degrees(phase_deviation_rad)

                lpd_band = phase_deviation_deg[net_mask]

                if len(f_hz[net_mask]) == len(ref_freq_band_hz):
                    lpd_data_collection.append(lpd_band)
                else:
                    freq_band_mhz = f_hz[net_mask] / 1e6
                    lpd_interp = np.interp(ref_freq_band_mhz, freq_band_mhz, lpd_band)
                    lpd_data_collection.append(lpd_interp)

        if len(lpd_data_collection) > 1:
            lpd_array = np.array(lpd_data_collection)
            lpd_mean = np.mean(lpd_array, axis=0)
            lpd_std = np.std(lpd_array, axis=0, ddof=1)

            sigma_colors = ["purple", "orange", "brown"]
            sigma_styles = ["-", "--", ":"]

            for i, sigma_val in enumerate(plot_config["sigma_settings"]["sigma_values"]):
                if i >= 3:
                    break

                color = sigma_colors[i % len(sigma_colors)]
                style = sigma_styles[i % len(sigma_styles)]

                lpd_upper = lpd_mean + sigma_val * lpd_std
                lpd_lower = lpd_mean - sigma_val * lpd_std

                (upper_line,) = plt.plot(ref_freq_band_mhz, lpd_upper, color=color, linestyle=style, linewidth=1.5, alpha=0.8)
                (lower_line,) = plt.plot(ref_freq_band_mhz, lpd_lower, color=color, linestyle=style, linewidth=1.5, alpha=0.8)

                all_handles.append(upper_line)
                all_labels.append(f"Mean+{sigma_val}σ LPD")
                all_handles.append(lower_line)
                all_labels.append(f"Mean-{sigma_val}σ LPD")

                if i == 0:
                    (mean_line,) = plt.plot(ref_freq_band_mhz, lpd_mean, color="black", linestyle="-", linewidth=2, alpha=0.9)
                    all_handles.append(mean_line)
                    all_labels.append("Mean LPD")
        else:
            print("Warning: Need at least 2 measurement files for sigma calculations")

    # LPD Spec lines
    lpd_colors = ["red", "darkred", "crimson", "orange", "darkorange", "orangered"]
    color_index = 0

    for i, lpd_spec in enumerate(kpi_config["KPIs"].get("LPD_MIN", [])):
        freq_start = lpd_spec["range"][0] / 1e6
        freq_end = lpd_spec["range"][1] / 1e6
        lsl_value = lpd_spec["LSL"]
        color = lpd_colors[color_index % len(lpd_colors)]
        color_index += 1
        freq_range = np.linspace(freq_start, freq_end, 100)
        if plot_config.get("plot_settings", {}).get("show_spec_lines", False):
            (lsl_line,) = plt.plot(freq_range, [lsl_value] * len(freq_range), color=color, linestyle="--", linewidth=2)
            all_handles.append(lsl_line)
            all_labels.append(f'{lpd_spec["name"]} LSL: {lsl_value} deg')
        plt.axvspan(freq_start, freq_end, alpha=0.05, color=color)

    for i, lpd_spec in enumerate(kpi_config["KPIs"].get("LPD_MAX", [])):
        freq_start = lpd_spec["range"][0] / 1e6
        freq_end = lpd_spec["range"][1] / 1e6
        usl_value = lpd_spec["USL"]
        color = lpd_colors[color_index % len(lpd_colors)]
        color_index += 1
        freq_range = np.linspace(freq_start, freq_end, 100)
        if plot_config.get("plot_settings", {}).get("show_spec_lines", False):
            (usl_line,) = plt.plot(freq_range, [usl_value] * len(freq_range), color=color, linestyle="--", linewidth=2)
            all_handles.append(usl_line)
            all_labels.append(f'{lpd_spec["name"]} USL: {usl_value} deg')
        plt.axvspan(freq_start, freq_end, alpha=0.05, color=color)

    cfg = plot_config["axis_ranges"]["advanced_plots"]["linear_phase_deviation"]
    plt.xlim(cfg["x_axis"]["min"], cfg["x_axis"]["max"])
    plt.ylim(cfg["y_axis"]["min"], cfg["y_axis"]["max"])
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("LPD (deg)")
    plt.title("Linear-Phase Deviation (continuous)")
    plt.grid(True, alpha=0.3)

    plt.legend(all_handles, all_labels, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small", ncol=2, borderaxespad=0.0)
    plt.tight_layout()

    plot_path = os.path.join(save_folder, "LPD_Curve.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    return "LPD_Curve.png"

def generate_flatness_scatter_plot(networks, plot_config, kpi_config, save_folder):
    """Generate Flatness Scatter Plot with simplified legend and varying point sizes"""

    plt.figure(figsize=(12, 6))
    colors = ["blue", "red", "green", "orange", "purple"]

    all_handles = []
    all_labels = []

    for band_idx, band in enumerate(kpi_config["KPIs"].get("Flat", [])):
        lo, hi = band["range"]
        unit_numbers = []
        flatness_values = []
        point_sizes = []

        unit_num = 1
        for fname, nt in networks.items():
            f = nt.frequency.f
            s21 = 20 * np.log10(np.abs(nt.s[:, 1, 0]))
            mask = (f >= lo) & (f <= hi)

            if np.any(mask):
                flat = s21[mask].max() - s21[mask].min()
                unit_numbers.append(unit_num)
                flatness_values.append(flat)
                # Assign point size based on unit number (gives visual difference)
                point_sizes.append(40 + (unit_num * 5))

            unit_num += 1

        scatter = plt.scatter(
            unit_numbers,
            flatness_values,
            color=colors[band_idx % len(colors)],
            alpha=0.7,
            s=point_sizes  # varying point sizes
        )
        all_handles.append(scatter)
        all_labels.append(f"{band['name']} ({lo / 1e6:.0f}-{hi / 1e6:.0f} MHz)")

        if plot_config.get("plot_settings", {}).get("show_spec_lines", False):
            if "USL" in band:
                usl_line = plt.axhline(
                    y=band["USL"],
                    color=colors[band_idx % len(colors)],
                    linestyle="--",
                    alpha=0.5
                )
                all_handles.append(usl_line)
                all_labels.append(f"{band['name']} USL: {band['USL']} dB")

    plt.xlabel("Unit Number")
    plt.ylabel("Flatness (dB)")
    plt.title("S21 Flatness vs Unit Number (Variable Point Sizes)")
    plt.grid(True, alpha=0.3)
    plt.legend(all_handles, all_labels, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    plot_path = os.path.join(save_folder, "Flatness_Scatter.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    return "Flatness_Scatter.png"


def generate_gdv_scatter_plot(networks, plot_config, kpi_config, save_folder):
    """Generate GD Variation Scatter Plot with simplified legend"""
    
    def phase_and_gd(network):
        f_hz = network.frequency.f
        ang = np.unwrap(np.angle(network.s[:, 1, 0]))
        df = np.gradient(f_hz)
        gd_s = -np.gradient(ang) / (2 * np.pi * (df + 1e-12))
        return ang, gd_s * 1e9

    plt.figure(figsize=(12, 6))
    colors = ["blue", "red", "green", "orange", "purple"]

    all_handles = []
    all_labels = []

    for band_idx, band in enumerate(kpi_config["KPIs"].get("GDV", [])):
        lo, hi = band["range"]
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

        scatter = plt.scatter(
            unit_numbers,
            gdv_values,
            color=colors[band_idx % len(colors)],
            alpha=0.7,
            s=50
        )
        all_handles.append(scatter)
        all_labels.append(f"{band['name']} ({lo / 1e6:.0f}-{hi / 1e6:.0f} MHz)")

        if plot_config.get("plot_settings", {}).get("show_spec_lines", False):
            if "USL" in band:
                usl_line = plt.axhline(
                    y=band["USL"],
                    color=colors[band_idx % len(colors)],
                    linestyle="--",
                    alpha=0.5
                )
                all_handles.append(usl_line)
                all_labels.append(f"{band['name']} USL: {band['USL']} ns")

    plt.xlabel("Unit Number")
    plt.ylabel("GD Variation (ns)")
    plt.title("Group Delay Variation vs Unit Number")
    plt.grid(True, alpha=0.3)
    plt.legend(all_handles, all_labels, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    plot_path = os.path.join(save_folder, "GDV_Scatter.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    return "GDV_Scatter.png"


# =============================================================================
# INDIVIDUAL STATISTICAL PLOT FUNCTIONS
# =============================================================================


def generate_individual_box_plot(param_name, param_data, save_folder):
    """
    Generate and save an individual box plot for a parameter.

    Args:
        param_name (str): Name of the parameter.
        param_data (pd.Series): Data for the parameter.
        save_folder (str): Directory to save the plot.

    Returns:
        str: Filename of the saved plot.
    """
    plt.figure(figsize=(10, 6))
    box_data = [param_data.values]
    bp = plt.boxplot(box_data, labels=[param_name], patch_artist=True)
    bp["boxes"][0].set_facecolor("lightblue")
    bp["boxes"][0].set_alpha(0.7)

    # Add individual data points
    y_values = param_data.values
    x_values = [1] * len(y_values)
    plt.scatter(x_values, y_values, alpha=0.6, color="red", s=30, zorder=3)

    plt.ylabel("Value")
    plt.title(f"{param_name} - Statistical Distribution")
    plt.xlabel("Parameter")
    plt.grid(True, alpha=0.3)

    # Add statistics text
    mean_val = param_data.mean()
    std_val = param_data.std()
    min_val = param_data.min()
    max_val = param_data.max()
    stats_text = f"Mean: {mean_val:.3f}\nStd: {std_val:.3f}\nMin: {min_val:.3f}\nMax: {max_val:.3f}"
    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    filename = f"BoxPlot_{param_name}.png"
    plot_path = os.path.join(save_folder, filename)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    return filename


def generate_individual_histogram_plot(param_name, param_data, save_folder):
    """
    Generate and save an individual histogram plot for a parameter.

    Args:
        param_name (str): Name of the parameter.
        param_data (pd.Series): Data for the parameter.
        save_folder (str): Directory to save the plot.

    Returns:
        str: Filename of the saved plot.
    """
    plt.figure(figsize=(10, 6))
    values = param_data.values
    plt.hist(
        values,
        bins=min(20, max(5, len(values) // 3)),
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
    )

    plt.title(f"{param_name} - Distribution Histogram")
    plt.xlabel(param_name)
    plt.ylabel("Number of units")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    filename = f"Histogram_{param_name}.png"
    plot_path = os.path.join(save_folder, filename)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    return filename


# =============================================================================
# CATEGORY MASTER FUNCTIONS
# =============================================================================


def create_s_parameter_plots(
    networks, plot_config, kpi_config, sim_data=None, save_folder="."
):
    """Create S-parameter continuous curve plots"""
    plots_created = []

    freq_shifts = calculate_freq_shifts(plot_config)

    plots_created.append(
        generate_s11_return_loss_plot(
            networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder
        )
    )
    plots_created.append(
        generate_s22_return_loss_plot(
            networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder
        )
    )
    plots_created.append(
        generate_s21_insertion_loss_plot(
            networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder
        )
    )
    plots_created.append(
        generate_s21_rejection_loss_plot(
            networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder
        )
    )

    return plots_created


def create_advanced_plots(
    plot_config, kpi_config, networks, freq_shifts, save_folder="."
):
    """Create advanced plots"""
    plots_created = []

    plots_created.append(
        generate_group_delay_plot(
            networks, plot_config, kpi_config, {}, freq_shifts, save_folder
        )
    )
    plots_created.append(
        generate_linear_phase_deviation_plot(
            networks, plot_config, kpi_config, {}, freq_shifts, save_folder
        )
    )
    plots_created.append(
        generate_flatness_scatter_plot(networks, plot_config, kpi_config, save_folder)
    )
    plots_created.append(
        generate_gdv_scatter_plot(networks, plot_config, kpi_config, save_folder)
    )

    return plots_created


# =============================================================================
# MAIN API FUNCTIONS
# =============================================================================


def generate_s_parameter_plots_only(
    plot_config_data,
    excel_files,
    s2p_files,
    sim_s2p_files=None,
    s11_sigma_files=None,
    s21_sigma_files=None,
    kpi_config_data=None,
    save_folder=".",
):
    """Generate S-parameter plots only"""
    try:
        plot_config, kpi_config = load_configuration(plot_config_data, kpi_config_data)
        if not kpi_config or "KPIs" not in kpi_config:
            raise ValueError(
                "kpi_config_data must be provided and contain a 'KPIs' key."
            )
        networks = load_s2p_files(s2p_files)
        sim_data = load_simulation_data(sim_s2p_files, s11_sigma_files, s21_sigma_files)

        all_plots = []

        # Generate S-parameter plots
        s_param_plots = create_s_parameter_plots(
            networks, plot_config, kpi_config, sim_data, save_folder
        )
        all_plots.extend(s_param_plots)

        # Generate advanced plots
        freq_shifts = calculate_freq_shifts(plot_config)
        advanced_plots = create_advanced_plots(
            plot_config, kpi_config, networks, freq_shifts, save_folder
        )
        all_plots.extend(advanced_plots)

        return all_plots

    except Exception as e:
        print(f"Error during S-Parameter and advanced plot generation: {e}")
        return []


def generate_all_plots(
    plot_config_data,
    excel_files,
    s2p_files,
    sim_s2p_files=None,
    s11_sigma_files=None,
    s21_sigma_files=None,
    kpi_config_data=None,
    save_folder=".",
):
    """Generate ALL plots - calls all individual plot functions"""
    try:
        # Load all required data once
        plot_config, kpi_config = load_configuration(plot_config_data, kpi_config_data)
        if not kpi_config or "KPIs" not in kpi_config:
            raise ValueError(
                "kpi_config_data must be provided and contain a 'KPIs' key."
            )
        networks = load_s2p_files(s2p_files)
        sim_data = load_simulation_data(sim_s2p_files, s11_sigma_files, s21_sigma_files)
        excel_data = load_excel_data(excel_files)
        freq_shifts = calculate_freq_shifts(plot_config)

        all_plots = []

        # S-Parameter Plots (4 plots)
        all_plots.append(
            generate_s11_return_loss_plot(
                networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder
            )
        )
        all_plots.append(
            generate_s22_return_loss_plot(
                networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder
            )
        )
        all_plots.append(
            generate_s21_insertion_loss_plot(
                networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder
            )
        )
        all_plots.append(
            generate_s21_rejection_loss_plot(
                networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder
            )
        )

        # Advanced Plots (4 plots)
        all_plots.append(
            generate_group_delay_plot(
                networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder
            )
        )
        all_plots.append(
            generate_linear_phase_deviation_plot(
                networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder
            )
        )
        all_plots.append(
            generate_flatness_scatter_plot(
                networks, plot_config, kpi_config, save_folder
            )
        )
        all_plots.append(
            generate_gdv_scatter_plot(networks, plot_config, kpi_config, save_folder)
        )

        # Statistical Plots - Box Plots (one for each parameter)
        parameter_columns = [
            col for col in excel_data["Per_File"].columns if col != "File"
        ]
        for param in parameter_columns:
            param_data = excel_data["Per_File"][param].dropna()
            if len(param_data) > 0:
                all_plots.append(
                    generate_individual_box_plot(param, param_data, save_folder)
                )

        # Statistical Plots - Histograms (one for each parameter)
        for param in parameter_columns:
            param_data = excel_data["Per_File"][param].dropna()
            if len(param_data) > 0:
                all_plots.append(
                    generate_individual_histogram_plot(param, param_data, save_folder)
                )

        return all_plots

    except Exception as e:
        print(f"Error during all plots generation: {e}")
        return []


# =============================================================================
# ULTRA-GRANULAR INDIVIDUAL PLOT FUNCTIONS (Optional)
# =============================================================================


def generate_single_s11_plot(
    plot_config_data,
    s2p_files,
    sim_s2p_files=None,
    s11_sigma_files=None,
    kpi_config_data=None,
    save_folder=".",
):
    """Generate only S11 return loss plot"""
    try:
        plot_config, kpi_config = load_configuration(plot_config_data, kpi_config_data)
        if not kpi_config or "KPIs" not in kpi_config:
            raise ValueError(
                "kpi_config_data must be provided and contain a 'KPIs' key."
            )
        networks = load_s2p_files(s2p_files)
        sim_data = load_simulation_data(sim_s2p_files, s11_sigma_files, [])
        freq_shifts = calculate_freq_shifts(plot_config)

        return [
            generate_s11_return_loss_plot(
                networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder
            )
        ]

    except Exception as e:
        print(f"Error during S11 plot generation: {e}")
        return []


def generate_single_s22_plot(
    plot_config_data,
    s2p_files,
    sim_s2p_files=None,
    kpi_config_data=None,
    save_folder=".",
):
    """Generate only S22 return loss plot"""
    try:
        plot_config, kpi_config = load_configuration(plot_config_data, kpi_config_data)
        if not kpi_config or "KPIs" not in kpi_config:
            raise ValueError(
                "kpi_config_data must be provided and contain a 'KPIs' key."
            )
        networks = load_s2p_files(s2p_files)
        sim_data = load_simulation_data(sim_s2p_files, [], [])
        freq_shifts = calculate_freq_shifts(plot_config)

        return [
            generate_s22_return_loss_plot(
                networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder
            )
        ]

    except Exception as e:
        print(f"Error during S22 plot generation: {e}")
        return []


def generate_single_s21_il_plot(
    plot_config_data,
    s2p_files,
    sim_s2p_files=None,
    s21_sigma_files=None,
    kpi_config_data=None,
    save_folder=".",
):
    """Generate only S21 insertion loss plot"""
    try:
        plot_config, kpi_config = load_configuration(plot_config_data, kpi_config_data)
        if not kpi_config or "KPIs" not in kpi_config:
            raise ValueError(
                "kpi_config_data must be provided and contain a 'KPIs' key."
            )
        networks = load_s2p_files(s2p_files)
        sim_data = load_simulation_data(sim_s2p_files, [], s21_sigma_files)
        freq_shifts = calculate_freq_shifts(plot_config)

        return [
            generate_s21_insertion_loss_plot(
                networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder
            )
        ]

    except Exception as e:
        print(f"Error during S21 IL plot generation: {e}")
        return []


def generate_single_s21_rej_plot(
    plot_config_data,
    s2p_files,
    sim_s2p_files=None,
    s21_sigma_files=None,
    kpi_config_data=None,
    save_folder=".",
):
    """Generate only S21 rejection loss plot"""
    try:
        plot_config, kpi_config = load_configuration(plot_config_data, kpi_config_data)
        if not kpi_config or "KPIs" not in kpi_config:
            raise ValueError(
                "kpi_config_data must be provided and contain a 'KPIs' key."
            )
        networks = load_s2p_files(s2p_files)
        sim_data = load_simulation_data(sim_s2p_files, [], s21_sigma_files)
        freq_shifts = calculate_freq_shifts(plot_config)

        return [
            generate_s21_rejection_loss_plot(
                networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder
            )
        ]

    except Exception as e:
        print(f"Error during S21 rejection plot generation: {e}")
        return []


def generate_single_gd_plot(
    plot_config_data, s2p_files, kpi_config_data=None, save_folder="."
):
    """Generate only Group Delay plot"""
    try:
        plot_config, kpi_config = load_configuration(plot_config_data, kpi_config_data)
        if not kpi_config or "KPIs" not in kpi_config:
            raise ValueError(
                "kpi_config_data must be provided and contain a 'KPIs' key."
            )
        networks = load_s2p_files(s2p_files)
        freq_shifts = calculate_freq_shifts(plot_config)

        return [
            generate_group_delay_plot(
                networks, plot_config, kpi_config, {}, freq_shifts, save_folder
            )
        ]

    except Exception as e:
        print(f"Error during GD plot generation: {e}")
        return []


def generate_single_lpd_plot(
    plot_config_data, s2p_files, kpi_config_data=None, save_folder="."
):
    """Generate only Linear Phase Deviation plot"""
    try:
        plot_config, kpi_config = load_configuration(plot_config_data, kpi_config_data)
        if not kpi_config or "KPIs" not in kpi_config:
            raise ValueError(
                "kpi_config_data must be provided and contain a 'KPIs' key."
            )
        networks = load_s2p_files(s2p_files)
        freq_shifts = calculate_freq_shifts(plot_config)

        return [
            generate_linear_phase_deviation_plot(
                networks, plot_config, kpi_config, {}, freq_shifts, save_folder
            )
        ]

    except Exception as e:
        print(f"Error during LPD plot generation: {e}")
        return []


def generate_single_flatness_plot(
    plot_config_data, s2p_files, kpi_config_data=None, save_folder="."
):
    """Generate only Flatness scatter plot"""
    try:
        plot_config, kpi_config = load_configuration(plot_config_data, kpi_config_data)
        if not kpi_config or "KPIs" not in kpi_config:
            raise ValueError(
                "kpi_config_data must be provided and contain a 'KPIs' key."
            )
        networks = load_s2p_files(s2p_files)

        return [
            generate_flatness_scatter_plot(
                networks, plot_config, kpi_config, save_folder
            )
        ]

    except Exception as e:
        print(f"Error during Flatness plot generation: {e}")
        return []


def generate_single_gdv_plot(
    plot_config_data, s2p_files, kpi_config_data=None, save_folder="."
):
    """Generate only GDV scatter plot"""
    try:
        plot_config, kpi_config = load_configuration(plot_config_data, kpi_config_data)
        if not kpi_config or "KPIs" not in kpi_config:
            raise ValueError(
                "kpi_config_data must be provided and contain a 'KPIs' key."
            )
        networks = load_s2p_files(s2p_files)

        return [
            generate_gdv_scatter_plot(networks, plot_config, kpi_config, save_folder)
        ]

    except Exception as e:
        print(f"Error during GDV plot generation: {e}")
        return []


def create_statistical_plots(excel_data, plot_config, save_folder="."):
    """Create individual box plots for each parameter from Excel statistical data and save in the provided folder"""
    plots_created = []

    # Get the Per_File sheet which contains individual measurements
    per_file_df = excel_data["Per_File"]

    # Get column names excluding 'File' column
    parameter_columns = [col for col in per_file_df.columns if col != "File"]

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
        bp["boxes"][0].set_facecolor("lightblue")
        bp["boxes"][0].set_alpha(0.7)

        # Add individual data points
        y_values = param_values.values
        x_values = [1] * len(y_values)  # All points at x=1 for single parameter
        plt.scatter(x_values, y_values, alpha=0.6, color="red", s=30, zorder=3)

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
        plt.text(
            0.02,
            0.98,
            stats_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()

        # Save individual plot to the provided folder
        filename = f"BoxPlot_{param}.png"
        plot_path = os.path.join(save_folder, filename)
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        plots_created.append(filename)

        print(f"Created box plot: {plot_path}")

    return plots_created


def create_histogram_plots(excel_data, plot_config, save_folder="."):
    """Create individual histogram plots for each parameter and save in the provided folder"""
    plots_created = []

    # Get the Per_File sheet which contains individual measurements
    per_file_df = excel_data["Per_File"]

    # Get column names excluding 'File' column
    parameter_columns = [col for col in per_file_df.columns if col != "File"]

    for param in parameter_columns:
        # Get data for this parameter
        param_values = per_file_df[param].dropna()

        if len(param_values) == 0:
            continue

        # Create individual histogram for this parameter
        plt.figure(figsize=(10, 6))

        # Create histogram
        values = param_values.values
        plt.hist(
            values,
            bins=min(20, max(5, len(values) // 3)),
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
        )

        # Add vertical lines for mean and limits
        # mean_val = values.mean()
        # plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')

        # Add ±3σ lines
        # std_val = values.std()
        # if std_val > 0:  # Only add sigma lines if std deviation is not zero
        #    plt.axvline(mean_val + 3*std_val, color='orange', linestyle=':', linewidth=2, label=f'+3σ: {mean_val + 3*std_val:.3f}')
        #    plt.axvline(mean_val - 3*std_val, color='orange', linestyle=':', linewidth=2, label=f'-3σ: {mean_val - 3*std_val:.3f}')

        # Set title and labels
        plt.title(f"{param} - Distribution Histogram")
        # plt.xlabel("Value")
        # plt.ylabel("Frequency")

        plt.xlabel(param)
        plt.ylabel("Number of units")

        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()

        # Save individual plot to the provided folder
        filename = f"Histogram_{param}.png"
        plot_path = os.path.join(save_folder, filename)
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        plots_created.append(filename)

        print(f"Created histogram: {plot_path}")

    return plots_created


def generate_selected_s_parameter_plots(
    plot_config_data,
    excel_files,
    s2p_files,
    sim_s2p_files=None,
    s11_sigma_files=None,
    s21_sigma_files=None,
    kpi_config_data=None,
    save_folder=".",
):
    """
    Generate only the S-parameter and advanced plots requested in parameters_for_statistical_plots.
    """
    try:
        plot_config, kpi_config = load_configuration(plot_config_data, kpi_config_data)
        if not kpi_config or "KPIs" not in kpi_config:
            raise ValueError(
                "kpi_config_data must be provided and contain a 'KPIs' key."
            )
        networks = load_s2p_files(s2p_files)
        sim_data = load_simulation_data(sim_s2p_files, s11_sigma_files, s21_sigma_files)

        all_plots = []

        # Only generate requested S-parameter plots
        requested = set(
            [p.lower() for p in plot_config.get("parameters_for_statistical_plots", [])]
        )

        freq_shifts = calculate_freq_shifts(plot_config)

        # S-parameter plots
        if "s11" in requested:
            all_plots.append(
                generate_s11_return_loss_plot(
                    networks,
                    plot_config,
                    kpi_config,
                    sim_data,
                    freq_shifts,
                    save_folder,
                )
            )
        if "s22" in requested:
            all_plots.append(
                generate_s22_return_loss_plot(
                    networks,
                    plot_config,
                    kpi_config,
                    sim_data,
                    freq_shifts,
                    save_folder,
                )
            )
        if "s21_insertion" in requested:
            all_plots.append(
                generate_s21_insertion_loss_plot(
                    networks,
                    plot_config,
                    kpi_config,
                    sim_data,
                    freq_shifts,
                    save_folder,
                )
            )
        if "s21_rejection" in requested:
            all_plots.append(
                generate_s21_rejection_loss_plot(
                    networks,
                    plot_config,
                    kpi_config,
                    sim_data,
                    freq_shifts,
                    save_folder,
                )
            )
        # Advanced plots
        if "group_delay" in requested:
            all_plots.append(
                generate_group_delay_plot(
                    networks,
                    plot_config,
                    kpi_config,
                    sim_data,
                    freq_shifts,
                    save_folder,
                )
            )
        if "linear_phase_deviation" in requested:
            all_plots.append(
                generate_linear_phase_deviation_plot(
                    networks,
                    plot_config,
                    kpi_config,
                    sim_data,
                    freq_shifts,
                    save_folder,
                )
            )
        if "flatness" in requested:
            all_plots.append(
                generate_flatness_scatter_plot(
                    networks, plot_config, kpi_config, save_folder
                )
            )
        if "group_delay_variation" in requested:
            all_plots.append(
                generate_gdv_scatter_plot(
                    networks, plot_config, kpi_config, save_folder
                )
            )

        return all_plots

    except Exception as e:
        print(f"Error during selected S-Parameter and advanced plot generation: {e}")
        return []


def generate_selected_statistical_and_histogram_plots(
    plot_config_data, excel_files, save_folder="."
):
    try:
        plot_config, _ = load_configuration(plot_config_data)
        excel_data = load_excel_data(excel_files)

        all_plots = []

        # Generate statistical plots
        statistical_plots = create_statistical_plots(
            excel_data, plot_config, save_folder
        )
        all_plots.extend(statistical_plots)

        # Generate histogram plots
        histogram_plots = create_histogram_plots(excel_data, plot_config, save_folder)
        all_plots.extend(histogram_plots)

        return all_plots

    except Exception as e:
        print(f"Error during statistical/histogram plot generation: {e}")
        return []
