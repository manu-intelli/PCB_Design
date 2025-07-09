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
        ValueError: If plot_config_data is not provided or is empty.
        TypeError: If either input is not a dictionary.
        Exception: For any unexpected errors.
    """
    try:
        if not plot_config_data:
            raise ValueError("Plot configuration data is required.")
        if not isinstance(plot_config_data, dict):
            raise TypeError("plot_config_data must be a dictionary.")

        if kpi_config_data is None:
            kpi_config_data = {}
        elif not isinstance(kpi_config_data, dict):
            raise TypeError("kpi_config_data must be a dictionary if provided.")

        return plot_config_data, kpi_config_data

    except (ValueError, TypeError):
        raise  # Re-raise known validation errors
    except Exception as e:
        # Raise a generic error with context, preserving traceback
        raise RuntimeError("An unexpected error occurred in load_configuration.") from e


def load_excel_data(excel_files):
    """
    Load 'Per_File' and 'Summary' sheets from the first Excel file in the list.

    Args:
        excel_files (list): List of Excel file paths.

    Returns:
        dict: Dictionary with keys 'Per_File' and 'Summary' containing pandas DataFrames.

    Raises:
        FileNotFoundError: If the file list is empty.
        TypeError: If input is not a list.
        ValueError: If required sheets are missing in the Excel file.
        RuntimeError: If any other unexpected error occurs.
    """
    try:
        if not isinstance(excel_files, list):
            raise TypeError("excel_files must be a list of file paths.")
        
        if not excel_files:
            raise FileNotFoundError("SParam_Summary.xlsx not found in the provided folders.")

        per_file_data = pd.read_excel(excel_files[0], sheet_name="Per_File")
        summary_data = pd.read_excel(excel_files[0], sheet_name="Summary")

        return {"Per_File": per_file_data, "Summary": summary_data}

    except (FileNotFoundError, ValueError, TypeError):
        raise  # Re-raise expected exceptions as-is for caller to handle

    except Exception as e:
        raise RuntimeError("An unexpected error occurred in load_excel_data.") from e


def load_s2p_files(s2p_files):
    """
    Load S2P files and return a dictionary of rf.Network objects.

    Args:
        s2p_files (list): List of file paths to .s2p files.

    Returns:
        dict: Dictionary where keys are filenames and values are rf.Network objects.

    Raises:
        FileNotFoundError: If the input list is empty.
        TypeError: If input is not a list.
        RuntimeError: If an unexpected error occurs during loading.
    """
    try:
        if not isinstance(s2p_files, list):
            raise TypeError("s2p_files must be a list of file paths.")
        
        if not s2p_files:
            raise FileNotFoundError("No *.s2p files found in the provided folders.")

        networks = {}
        for file_path in s2p_files:
            try:
                network = rf.Network(file_path)
                filename = os.path.basename(file_path)
                networks[filename] = network
            except Exception as e:
                # Log this at a higher level if needed
                continue  # Skip problematic file but continue loading others

        return networks

    except (FileNotFoundError, TypeError):
        raise
    except Exception as e:
        raise RuntimeError("An unexpected error occurred in load_s2p_files.") from e


def load_simulation_data(sim_s2p_files, s11_sigma_files, s21_sigma_files):
    """
    Load simulation S2P and sigma CSV files.

    Args:
        sim_s2p_files (list): List of simulation S2P file paths.
        s11_sigma_files (list): List of S11 sigma CSV file paths.
        s21_sigma_files (list): List of S21 sigma CSV file paths.

    Returns:
        dict: Dictionary containing available data with keys:
              - 'nominal': rf.Network object from simulation S2P file.
              - 's11_sigma': DataFrame from S11 sigma CSV file.
              - 's21_sigma': DataFrame from S21 sigma CSV file.

    Raises:
        TypeError: If any input is not a list.
        RuntimeError: If unexpected errors occur during loading.
    """
    try:
        if not all(isinstance(lst, list) for lst in [sim_s2p_files, s11_sigma_files, s21_sigma_files]):
            raise TypeError("All input arguments must be lists of file paths.")

        sim_data = {}

        if sim_s2p_files:
            try:
                sim_data["nominal"] = rf.Network(sim_s2p_files[0])
            except Exception:
                pass  # Log upstream if needed

        if s11_sigma_files:
            try:
                sim_data["s11_sigma"] = pd.read_csv(s11_sigma_files[0])
            except Exception:
                pass  # Log upstream if needed

        if s21_sigma_files:
            try:
                sim_data["s21_sigma"] = pd.read_csv(s21_sigma_files[0])
            except Exception:
                pass  # Log upstream if needed

        return sim_data

    except TypeError:
        raise
    except Exception as e:
        raise RuntimeError("An unexpected error occurred in load_simulation_data.") from e



def apply_frequency_shift(network, shift_mhz):
    """
    Apply a frequency shift to a scikit-rf Network object.

    Args:
        network (rf.Network): The original network object.
        shift_mhz (float): Frequency shift in MHz.

    Returns:
        rf.Network: A new Network object with shifted frequencies.

    Raises:
        TypeError: If `network` is not an instance of rf.Network.
        ValueError: If `shift_mhz` is not a float or int.
        RuntimeError: For unexpected errors during frequency shifting.
    """
    try:
        if not isinstance(network, rf.Network):
            raise TypeError("Expected `network` to be an instance of rf.Network.")

        if not isinstance(shift_mhz, (int, float)):
            raise ValueError("`shift_mhz` must be a numeric value (int or float).")

        if shift_mhz == 0:
            return network

        shifted_network = network.copy()
        original_freq_hz = network.frequency.f
        shifted_freq_hz = original_freq_hz * (1 + shift_mhz / 100.0)
        new_frequency = rf.Frequency.from_f(shifted_freq_hz, unit="Hz")

        return rf.Network(
            frequency=new_frequency,
            s=network.s,
            name=network.name
        )

    except (TypeError, ValueError):
        raise
    except Exception as e:
        raise RuntimeError("An unexpected error occurred in apply_frequency_shift.") from e



def calculate_freq_shifts(plot_config):
    """
    Calculate frequency shifts from plot configuration, always including 0 MHz.

    Args:
        plot_config (dict): Plot configuration dictionary. Must contain the key
                            'frequency_shifts' with keys 'enabled', 'shifts', and 'max_shifts'.

    Returns:
        list: List of frequency shifts (in MHz), always starting with 0.

    Raises:
        TypeError: If `plot_config` is not a dictionary.
        KeyError: If required keys are missing in `plot_config`.
        ValueError: If frequency shifts are not a list or max_shifts is not an integer.
        RuntimeError: If any unexpected error occurs.
    """
    try:
        if not isinstance(plot_config, dict):
            raise TypeError("plot_config must be a dictionary.")

        freq_shift_config = plot_config["frequency_shifts"]

        if not isinstance(freq_shift_config.get("shifts", []), list):
            raise ValueError("`shifts` must be a list.")
        if not isinstance(freq_shift_config.get("max_shifts", 0), int):
            raise ValueError("`max_shifts` must be an integer.")

        if freq_shift_config.get("enabled", False):
            user_shifts = freq_shift_config["shifts"]
            freq_shifts = [0] + [s for s in user_shifts if s != 0]
            max_allowed = freq_shift_config["max_shifts"]

            if len(freq_shifts) > max_allowed:
                # Trim the list to allowed size, always including 0 first
                freq_shifts = freq_shifts[:max_allowed]
        else:
            freq_shifts = [0]

        return freq_shifts

    except (TypeError, KeyError, ValueError):
        raise
    except Exception as e:
        raise RuntimeError("An unexpected error occurred in calculate_freq_shifts.") from e



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
    """
    Extract simplified unique IDs from a list of full labels, preserving shift indicators.

    Args:
        all_labels (list): List of label strings including filenames and optional '+Shift'/'-Shift'.

    Returns:
        list: Simplified label list with shortened unique identifiers.

    Raises:
        TypeError: If input is not a list or contains non-string elements.
        RuntimeError: If an unexpected error occurs.
    """
    try:
        if not isinstance(all_labels, list):
            raise TypeError("all_labels must be a list of strings.")
        if not all(isinstance(label, str) for label in all_labels):
            raise TypeError("All elements in all_labels must be strings.")

        simplified_labels = []

        # Get only actual file labels, exclude shifts
        file_labels = [label for label in all_labels if label not in ['+Shift', '-Shift']]
        common_text = os.path.commonprefix(file_labels)

        for label in all_labels:
            if label in ['+Shift', '-Shift']:
                simplified_labels.append(label)
            else:
                # Unique part is what comes after the common prefix
                unique_id = label.replace(common_text, '').strip('_- ').strip()

                # Optionally shorten long identifiers
                if len(unique_id) > 6:
                    unique_id = unique_id[:6]

                # Take first 4 characters from common prefix
                prefix = common_text[:4]
                simplified_labels.append(f"{prefix}_{unique_id}")

        return simplified_labels

    except (TypeError, ValueError):
        raise
    except Exception as e:
        raise RuntimeError("An unexpected error occurred in extract_unique_ids.") from e


def generate_s11_return_loss_plot(
    networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder
):
    """
    Generate S11 Return Loss Plot from a set of networks, including optional frequency shifts,
    sigma overlays, and simulation data. The plot is saved as a compressed JPEG image.

    Args:
        networks (dict): Dictionary of filename -> rf.Network.
        plot_config (dict): Plot configuration.
        kpi_config (dict): KPI configuration including RL specs.
        sim_data (dict): Optional simulation data with 'nominal' key.
        freq_shifts (list): List of frequency shifts in MHz.
        save_folder (str): Directory to save the generated plot.

    Returns:
        str: Path to the saved S11 return loss plot JPG image.

    Raises:
        ValueError: For missing or malformed configurations.
        RuntimeError: If any unexpected error occurs.
    """
    try:
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

            shifted_net = apply_frequency_shift(network, 0)
            freq_mhz = shifted_net.frequency.f / 1e6
            s11_db = 20 * np.log10(np.abs(shifted_net.s[:, 0, 0]))
            (line,) = plt.plot(freq_mhz, s11_db, alpha=0.7, color=unique_color)
            all_handles.append(line)
            all_labels.append(filename)

            for shift in freq_shifts:
                if shift == 0:
                    continue

                shifted_net = apply_frequency_shift(network, shift)
                freq_mhz = shifted_net.frequency.f / 1e6
                s11_db = 20 * np.log10(np.abs(shifted_net.s[:, 0, 0]))

                if shift > 0 and not positive_shift_plotted:
                    (line,) = plt.plot(freq_mhz, s11_db, alpha=0.7, color=unique_color, linestyle="--")
                    all_handles.append(line)
                    all_labels.append("+Shift")
                    positive_shift_plotted = True
                elif shift < 0 and not negative_shift_plotted:
                    (line,) = plt.plot(freq_mhz, s11_db, alpha=0.7, color=unique_color, linestyle=":")
                    all_handles.append(line)
                    all_labels.append("-Shift")
                    negative_shift_plotted = True

        all_labels = extract_unique_ids(all_labels)

        # Sigma Curves
        sigma_cfg = plot_config.get("sigma_settings", {})
        if sigma_cfg.get("enabled") and sigma_cfg.get("sigma_values"):
            first_network = list(networks.values())[0]
            ref_freq_hz = first_network.frequency.f
            ref_freq_mhz = ref_freq_hz / 1e6

            s11_data_collection = []
            for network in networks.values():
                shifted_net = apply_frequency_shift(network, 0)
                if len(shifted_net.frequency.f) == len(ref_freq_hz):
                    s11_data_collection.append(20 * np.log10(np.abs(shifted_net.s[:, 0, 0])))
                else:
                    freq_mhz = shifted_net.frequency.f / 1e6
                    s11_db = 20 * np.log10(np.abs(shifted_net.s[:, 0, 0]))
                    interp_s11 = np.interp(ref_freq_mhz, freq_mhz, s11_db)
                    s11_data_collection.append(interp_s11)

            if len(s11_data_collection) > 1:
                s11_array = np.array(s11_data_collection)
                s11_mean = np.mean(s11_array, axis=0)
                s11_std = np.std(s11_array, axis=0, ddof=1)

                sigma_colors = ["purple", "orange", "brown"]
                sigma_styles = ["-", "--", ":"]
                for i, sigma_val in enumerate(sigma_cfg["sigma_values"]):
                    if i >= 3:
                        break

                    s11_upper = s11_mean + sigma_val * s11_std
                    s11_lower = s11_mean - sigma_val * s11_std
                    color = sigma_colors[i % len(sigma_colors)]
                    style = sigma_styles[i % len(sigma_styles)]

                    (line_upper,) = plt.plot(ref_freq_mhz, s11_upper, color=color, linestyle=style, linewidth=2.5, alpha=0.8)
                    (line_lower,) = plt.plot(ref_freq_mhz, s11_lower, color=color, linestyle=style, linewidth=2.5, alpha=0.8)
                    all_handles += [line_upper, line_lower]
                    all_labels += [f"Mean+{sigma_val}σ S11", f"Mean-{sigma_val}σ S11"]

                    if i == 0:
                        (mean_line,) = plt.plot(ref_freq_mhz, s11_mean, color="black", linestyle="-", linewidth=2, alpha=0.9)
                        all_handles.append(mean_line)
                        all_labels.append("Mean S11")

        # Simulation Overlay
        if sim_data and "nominal" in sim_data and plot_config.get("simulation_settings", {}).get("include_simulation"):
            sim_freq_mhz = sim_data["nominal"].frequency.f / 1e6
            sim_s11_db = 20 * np.log10(np.abs(sim_data["nominal"].s[:, 0, 0]))
            (sim_line,) = plt.plot(sim_freq_mhz, sim_s11_db, "r--", linewidth=2)
            all_handles.append(sim_line)
            all_labels.append("Simulation Nominal")

        # KPI spec lines
        for i, rl_spec in enumerate(kpi_config.get("KPIs", {}).get("RL", [])):
            freq_start = rl_spec["range"][0] / 1e6
            freq_end = rl_spec["range"][1] / 1e6
            lsl_value = -abs(rl_spec["LSL"])
            color = ["red", "darkred", "crimson"][i % 3]

            if plot_config.get("plot_settings", {}).get("show_spec_lines", False):
                (rl_line,) = plt.plot(np.linspace(freq_start, freq_end, 100), [lsl_value]*100, color=color, linestyle="--", linewidth=2)
                all_handles.append(rl_line)
                all_labels.append(f'{rl_spec["name"]} LSL: {lsl_value} dB')

            plt.axvspan(freq_start, freq_end, alpha=0.05, color=color)

        # Axis Configuration
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

        # Save image
        temp_png_path = os.path.join(save_folder, "temp_s11_plot.png")
        final_path = os.path.join(save_folder, "S11_Return_Loss.jpg")

        plt.savefig(temp_png_path, dpi=100, bbox_inches="tight", pad_inches=0.1)
        plt.close()

        image = Image.open(temp_png_path).convert("RGB")
        image.save(final_path, format="JPEG", quality=70, optimize=True)
        os.remove(temp_png_path)

        return final_path

    except (KeyError, ValueError, TypeError):
        raise  # Raise known configuration/data issues
    except Exception as e:
        raise RuntimeError("An unexpected error occurred in generate_s11_return_loss_plot.") from e


def generate_s22_return_loss_plot(networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder):
    """
    Generate S22 Return Loss Plot with frequency shifts, sigma overlays, and simulation data.
    Saves the final plot as a compressed JPG image.

    Args:
        networks (dict): Dictionary mapping filename to rf.Network.
        plot_config (dict): Plotting configuration including axis ranges and sigma settings.
        kpi_config (dict): KPI spec limits to overlay on the plot.
        sim_data (dict): Dictionary with optional 'nominal' simulation network.
        freq_shifts (list): List of frequency shifts to apply.
        save_folder (str): Directory to save the generated plot.

    Returns:
        str: Path to the saved S22 Return Loss JPG image.

    Raises:
        KeyError, TypeError, ValueError: For missing or invalid configuration values.
        RuntimeError: If an unexpected error occurs.
    """
    try:
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

            shifted_net = apply_frequency_shift(network, 0)
            freq_mhz = shifted_net.frequency.f / 1e6
            s22_db = 20 * np.log10(np.abs(shifted_net.s[:, 1, 1]))
            (line,) = plt.plot(freq_mhz, s22_db, alpha=0.7, color=unique_color)
            all_handles.append(line)
            all_labels.append(filename)

            for shift in freq_shifts:
                if shift == 0:
                    continue

                shifted_net = apply_frequency_shift(network, shift)
                freq_mhz = shifted_net.frequency.f / 1e6
                s22_db = 20 * np.log10(np.abs(shifted_net.s[:, 1, 1]))

                linestyle = "--" if shift > 0 else ":"
                (line,) = plt.plot(freq_mhz, s22_db, alpha=0.7, color=unique_color, linestyle=linestyle)

                if shift > 0 and not positive_shift_plotted:
                    all_handles.append(line)
                    all_labels.append("+Shift")
                    positive_shift_plotted = True
                elif shift < 0 and not negative_shift_plotted:
                    all_handles.append(line)
                    all_labels.append("-Shift")
                    negative_shift_plotted = True

        all_labels = extract_unique_ids(all_labels)

        # Sigma Curves
        sigma_cfg = plot_config.get("sigma_settings", {})
        if sigma_cfg.get("enabled") and sigma_cfg.get("sigma_values"):
            first_network = list(networks.values())[0]
            ref_freq_hz = first_network.frequency.f
            ref_freq_mhz = ref_freq_hz / 1e6

            s22_data_collection = []
            for network in networks.values():
                shifted_net = apply_frequency_shift(network, 0)
                if len(shifted_net.frequency.f) == len(ref_freq_hz):
                    s22_db = 20 * np.log10(np.abs(shifted_net.s[:, 1, 1]))
                    s22_data_collection.append(s22_db)
                else:
                    freq_mhz = shifted_net.frequency.f / 1e6
                    s22_db = 20 * np.log10(np.abs(shifted_net.s[:, 1, 1]))
                    interp = np.interp(ref_freq_mhz, freq_mhz, s22_db)
                    s22_data_collection.append(interp)

            if len(s22_data_collection) > 1:
                s22_array = np.array(s22_data_collection)
                s22_mean = np.mean(s22_array, axis=0)
                s22_std = np.std(s22_array, axis=0, ddof=1)

                sigma_colors = ["purple", "orange", "brown"]
                sigma_styles = ["-", "--", ":"]

                for i, sigma_val in enumerate(sigma_cfg["sigma_values"]):
                    if i >= 3:
                        break
                    color = sigma_colors[i % len(sigma_colors)]
                    style = sigma_styles[i % len(sigma_styles)]

                    upper = s22_mean + sigma_val * s22_std
                    lower = s22_mean - sigma_val * s22_std

                    (line_u,) = plt.plot(ref_freq_mhz, upper, color=color, linestyle=style, linewidth=2.5, alpha=0.8)
                    (line_l,) = plt.plot(ref_freq_mhz, lower, color=color, linestyle=style, linewidth=2.5, alpha=0.8)
                    all_handles += [line_u, line_l]
                    all_labels += [f"Mean+{sigma_val}σ S22", f"Mean-{sigma_val}σ S22"]

                    if i == 0:
                        (line_m,) = plt.plot(ref_freq_mhz, s22_mean, color="black", linestyle="-", linewidth=2, alpha=0.9)
                        all_handles.append(line_m)
                        all_labels.append("Mean S22")

        # Simulation Overlay
        if sim_data and "nominal" in sim_data and plot_config.get("simulation_settings", {}).get("include_simulation"):
            sim_freq_mhz = sim_data["nominal"].frequency.f / 1e6
            sim_s22_db = 20 * np.log10(np.abs(sim_data["nominal"].s[:, 1, 1]))
            (sim_line,) = plt.plot(sim_freq_mhz, sim_s22_db, "r--", linewidth=2)
            all_handles.append(sim_line)
            all_labels.append("Simulation Nominal")

        # Spec lines
        for i, rl_spec in enumerate(kpi_config.get("KPIs", {}).get("RL", [])):
            freq_start = rl_spec["range"][0] / 1e6
            freq_end = rl_spec["range"][1] / 1e6
            lsl_value = -abs(rl_spec["LSL"])
            color = ["red", "darkred", "crimson"][i % 3]

            if plot_config.get("plot_settings", {}).get("show_spec_lines", False):
                freq_range = np.linspace(freq_start, freq_end, 100)
                (rl_line,) = plt.plot(freq_range, [lsl_value] * len(freq_range), color=color, linestyle="--", linewidth=2)
                all_handles.append(rl_line)
                all_labels.append(f'{rl_spec["name"]} LSL: {lsl_value} dB')

            plt.axvspan(freq_start, freq_end, alpha=0.05, color=color)

        # Axis Config
        config = plot_config["axis_ranges"]["s_parameter_plots"]["s22_return_loss"]
        plt.xlim(config["x_axis"]["min"], config["x_axis"]["max"])
        plt.ylim(config["y_axis"]["min"], config["y_axis"]["max"])
        plt.xlabel(f"Frequency ({config['x_axis']['unit']})")
        plt.ylabel(f"S22 ({config['y_axis']['unit']})")
        plt.title("S22 Return Loss vs Frequency")
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

        # Save as JPG
        temp_png = os.path.join(save_folder, "temp_s22_plot.png")
        final_jpg = os.path.join(save_folder, "S22_Return_Loss.jpg")

        plt.savefig(temp_png, dpi=100, bbox_inches="tight", pad_inches=0.1)
        plt.close()

        image = Image.open(temp_png).convert("RGB")
        image.save(final_jpg, format="JPEG", quality=70, optimize=True)
        os.remove(temp_png)

        return final_jpg

    except (KeyError, TypeError, ValueError) as e:
        raise e  # Allow known issues to propagate for higher-level handling
    except Exception as e:
        raise RuntimeError("Unexpected error in generate_s22_return_loss_plot.") from e


def generate_s21_insertion_loss_plot(networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder):
    """
    Generate S21 Insertion Loss Plot with shifts, sigma curves, simulation data, and spec overlays.
    
    Args:
        networks (dict): Dictionary of filename -> rf.Network objects.
        plot_config (dict): Configuration for plotting behavior and ranges.
        kpi_config (dict): KPI specs including USL/LSL ranges.
        sim_data (dict): Optional simulation data and sigma.
        freq_shifts (list): List of frequency shifts to apply (MHz).
        save_folder (str): Folder where image will be saved.

    Returns:
        str: Path to the saved JPG image.
    """
    try:
        plt.figure(figsize=(20, 9))
        num_files = len(networks)
        color_norm = mcolors.Normalize(vmin=0, vmax=num_files - 1)
        scalar_map = cm.ScalarMappable(norm=color_norm, cmap="tab20")

        all_handles, all_labels = [], []
        positive_shift_plotted = False
        negative_shift_plotted = False

        for idx, (filename, network) in enumerate(networks.items()):
            unique_color = scalar_map.to_rgba(idx)

            shifted_net = apply_frequency_shift(network, 0)
            freq_mhz = shifted_net.frequency.f / 1e6
            s21_db = 20 * np.log10(np.abs(shifted_net.s[:, 1, 0]))
            (line,) = plt.plot(freq_mhz, s21_db, alpha=0.7, color=unique_color)
            all_handles.append(line)
            all_labels.append(filename)

            for shift in freq_shifts:
                if shift == 0:
                    continue

                shifted_net = apply_frequency_shift(network, shift)
                freq_mhz = shifted_net.frequency.f / 1e6
                s21_db = 20 * np.log10(np.abs(shifted_net.s[:, 1, 0]))
                linestyle = "--" if shift > 0 else ":"

                (line,) = plt.plot(freq_mhz, s21_db, alpha=0.7, color=unique_color, linestyle=linestyle)
                if shift > 0 and not positive_shift_plotted:
                    all_handles.append(line)
                    all_labels.append("+Shift")
                    positive_shift_plotted = True
                elif shift < 0 and not negative_shift_plotted:
                    all_handles.append(line)
                    all_labels.append("-Shift")
                    negative_shift_plotted = True

        all_labels = extract_unique_ids(all_labels)

        # Sigma curves
        sigma_cfg = plot_config.get("sigma_settings", {})
        if sigma_cfg.get("enabled") and sigma_cfg.get("sigma_values"):
            ref_net = list(networks.values())[0]
            ref_freq_hz = ref_net.frequency.f
            ref_freq_mhz = ref_freq_hz / 1e6

            s21_data_collection = []
            for network in networks.values():
                shifted_net = apply_frequency_shift(network, 0)
                freq_hz = shifted_net.frequency.f
                s21_db = 20 * np.log10(np.abs(shifted_net.s[:, 1, 0]))

                if len(freq_hz) == len(ref_freq_hz):
                    s21_data_collection.append(s21_db)
                else:
                    freq_mhz = freq_hz / 1e6
                    interp = np.interp(ref_freq_mhz, freq_mhz, s21_db)
                    s21_data_collection.append(interp)

            if len(s21_data_collection) > 1:
                s21_array = np.array(s21_data_collection)
                s21_mean = np.mean(s21_array, axis=0)
                s21_std = np.std(s21_array, axis=0, ddof=1)

                sigma_colors = ["purple", "orange", "brown"]
                sigma_styles = ["-", "--", ":"]

                for i, sigma_val in enumerate(sigma_cfg["sigma_values"]):
                    if i >= 3:
                        break
                    color = sigma_colors[i % len(sigma_colors)]
                    style = sigma_styles[i % len(sigma_styles)]

                    upper = s21_mean + sigma_val * s21_std
                    lower = s21_mean - sigma_val * s21_std

                    (u_line,) = plt.plot(ref_freq_mhz, upper, color=color, linestyle=style, linewidth=2.5, alpha=0.8)
                    (l_line,) = plt.plot(ref_freq_mhz, lower, color=color, linestyle=style, linewidth=2.5, alpha=0.8)
                    all_handles += [u_line, l_line]
                    all_labels += [f"Mean+{sigma_val}σ S21", f"Mean-{sigma_val}σ S21"]

                    if i == 0:
                        (m_line,) = plt.plot(ref_freq_mhz, s21_mean, color="black", linestyle="-", linewidth=2, alpha=0.9)
                        all_handles.append(m_line)
                        all_labels.append("Mean S21")

        # Simulation nominal
        if sim_data and "nominal" in sim_data and plot_config.get("simulation_settings", {}).get("include_simulation"):
            sim_net = sim_data["nominal"]
            sim_freq_mhz = sim_net.frequency.f / 1e6
            sim_s21_db = 20 * np.log10(np.abs(sim_net.s[:, 1, 0]))
            (sim_line,) = plt.plot(sim_freq_mhz, sim_s21_db, "r--", linewidth=2)
            all_handles.append(sim_line)
            all_labels.append("Simulation Nominal")

        # Simulation sigma overlay
        if (
            sim_data and "nominal" in sim_data and "s21_sigma" in sim_data
            and sigma_cfg.get("enabled") and sigma_cfg.get("sigma_values")
        ):
            s21_sigma_df = sim_data["s21_sigma"]
            if "Freq (MHz)" in s21_sigma_df.columns:
                stdev_col = next((col for col in s21_sigma_df.columns if "stdev" in col.lower()), None)
                if stdev_col:
                    sigma_freq = s21_sigma_df["Freq (MHz)"].values
                    sigma_vals = s21_sigma_df[stdev_col].values
                    sim_sigma = np.interp(sim_freq_mhz, sigma_freq, sigma_vals)

                    sim_colors = ["purple", "darkviolet", "mediumorchid"]
                    for i, sigma_val in enumerate(sigma_cfg["sigma_values"]):
                        if i >= 3:
                            break
                        color = sim_colors[i % len(sim_colors)]
                        upper = sim_s21_db + sigma_val * sim_sigma
                        lower = sim_s21_db - sigma_val * sim_sigma

                        (u_line,) = plt.plot(sim_freq_mhz, upper, color=color, linestyle="-.", linewidth=2, alpha=0.8)
                        (l_line,) = plt.plot(sim_freq_mhz, lower, color=color, linestyle="-.", linewidth=2, alpha=0.8)
                        all_handles += [u_line, l_line]
                        all_labels += [f"Sim Mean+{sigma_val}σ S21", f"Sim Mean-{sigma_val}σ S21"]

        # IL Spec limits
        for i, il_spec in enumerate(kpi_config.get("KPIs", {}).get("IL", [])):
            freq_start = il_spec["range"][0] / 1e6
            freq_end = il_spec["range"][1] / 1e6
            usl_value = -abs(il_spec["USL"])
            color = ["green", "darkgreen", "lime"][i % 3]
            freq_range = np.linspace(freq_start, freq_end, 100)

            if plot_config.get("plot_settings", {}).get("show_spec_lines", False):
                (il_line,) = plt.plot(freq_range, [usl_value] * len(freq_range), color=color, linestyle="--", linewidth=2)
                all_handles.append(il_line)
                all_labels.append(f'{il_spec["name"]} USL: {usl_value} dB')

            plt.axvspan(freq_start, freq_end, alpha=0.05, color=color)

        # Axis config and title
        config = plot_config["axis_ranges"]["s_parameter_plots"]["s21_insertion_loss"]
        plt.xlim(config["x_axis"]["min"], config["x_axis"]["max"])
        plt.ylim(config["y_axis"]["min"], config["y_axis"]["max"])
        plt.xlabel(f"Frequency ({config['x_axis']['unit']})")
        plt.ylabel(f"S21 ({config['y_axis']['unit']})")
        plt.title("S21 Insertion Loss vs Frequency (Pass-band)")
        plt.grid(True, alpha=0.3)

        plt.legend(all_handles, all_labels, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small", ncol=2, borderaxespad=0.1)
        plt.tight_layout()

        # Save
        temp_png = os.path.join(save_folder, "temp_s21_plot.png")
        final_jpg = os.path.join(save_folder, "S21_Insertion_Loss.jpg")
        plt.savefig(temp_png, dpi=100, bbox_inches="tight", pad_inches=0.1)
        plt.close()

        img = Image.open(temp_png).convert("RGB")
        img.save(final_jpg, format="JPEG", quality=70, optimize=True)
        os.remove(temp_png)

        return final_jpg

    except (KeyError, ValueError, TypeError) as e:
        raise e
    except Exception as e:
        raise RuntimeError("Error in generate_s21_insertion_loss_plot") from e


def generate_s21_rejection_loss_plot(networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder):
    """
    Generate S21 Rejection Loss Plot with shifts, sigma curves, simulation overlays, and spec limits.
    """

    try:
        plt.figure(figsize=(20, 9))
        num_files = len(networks)
        color_norm = mcolors.Normalize(vmin=0, vmax=num_files - 1)
        scalar_map = cm.ScalarMappable(norm=color_norm, cmap="tab20")

        all_handles, all_labels = [], []
        positive_shift_plotted = False
        negative_shift_plotted = False

        for idx, (filename, network) in enumerate(networks.items()):
            unique_color = scalar_map.to_rgba(idx)
            shifted_net = apply_frequency_shift(network, 0)
            freq_mhz = shifted_net.frequency.f / 1e6
            s21_db = 20 * np.log10(np.abs(shifted_net.s[:, 1, 0]))
            (line,) = plt.plot(freq_mhz, s21_db, alpha=0.7, color=unique_color)
            all_handles.append(line)
            all_labels.append(filename)

            for shift in freq_shifts:
                if shift == 0:
                    continue
                shifted_net = apply_frequency_shift(network, shift)
                freq_mhz = shifted_net.frequency.f / 1e6
                s21_db = 20 * np.log10(np.abs(shifted_net.s[:, 1, 0]))
                linestyle = "--" if shift > 0 else ":"

                (line,) = plt.plot(freq_mhz, s21_db, alpha=0.7, color=unique_color, linestyle=linestyle)
                if shift > 0 and not positive_shift_plotted:
                    all_handles.append(line)
                    all_labels.append("+Shift")
                    positive_shift_plotted = True
                elif shift < 0 and not negative_shift_plotted:
                    all_handles.append(line)
                    all_labels.append("-Shift")
                    negative_shift_plotted = True

        all_labels = extract_unique_ids(all_labels)

        # Sigma curves
        sigma_cfg = plot_config.get("sigma_settings", {})
        if sigma_cfg.get("enabled") and sigma_cfg.get("sigma_values"):
            ref_net = list(networks.values())[0]
            ref_freq_hz = ref_net.frequency.f
            ref_freq_mhz = ref_freq_hz / 1e6
            s21_data_collection = []

            for net in networks.values():
                shifted = apply_frequency_shift(net, 0)
                freq_hz = shifted.frequency.f
                s21_db = 20 * np.log10(np.abs(shifted.s[:, 1, 0]))

                if len(freq_hz) == len(ref_freq_hz):
                    s21_data_collection.append(s21_db)
                else:
                    freq_mhz = freq_hz / 1e6
                    interp = np.interp(ref_freq_mhz, freq_mhz, s21_db)
                    s21_data_collection.append(interp)

            if len(s21_data_collection) > 1:
                s21_array = np.array(s21_data_collection)
                s21_mean = np.mean(s21_array, axis=0)
                s21_std = np.std(s21_array, axis=0, ddof=1)

                sigma_colors = ["purple", "orange", "brown"]
                sigma_styles = ["-", "--", ":"]

                for i, sigma_val in enumerate(sigma_cfg["sigma_values"]):
                    if i >= 3:
                        break
                    color = sigma_colors[i % len(sigma_colors)]
                    style = sigma_styles[i % len(sigma_styles)]

                    upper = s21_mean + sigma_val * s21_std
                    lower = s21_mean - sigma_val * s21_std

                    (u_line,) = plt.plot(ref_freq_mhz, upper, color=color, linestyle=style, linewidth=1.5, alpha=0.8)
                    (l_line,) = plt.plot(ref_freq_mhz, lower, color=color, linestyle=style, linewidth=1.5, alpha=0.8)

                    all_handles += [u_line, l_line]
                    all_labels += [f"Mean+{sigma_val}σ S21 Rej", f"Mean-{sigma_val}σ S21 Rej"]

                    if i == 0:
                        (m_line,) = plt.plot(ref_freq_mhz, s21_mean, color="black", linestyle="-", linewidth=2, alpha=0.9)
                        all_handles.append(m_line)
                        all_labels.append("Mean S21 Rej")

        # Simulation overlay
        if sim_data and "nominal" in sim_data and plot_config.get("simulation_settings", {}).get("include_simulation"):
            sim_net = sim_data["nominal"]
            sim_freq_mhz = sim_net.frequency.f / 1e6
            sim_s21_db = 20 * np.log10(np.abs(sim_net.s[:, 1, 0]))
            (sim_line,) = plt.plot(sim_freq_mhz, sim_s21_db, "r--", linewidth=2)
            all_handles.append(sim_line)
            all_labels.append("Simulation Nominal")

        # Simulation sigma overlay
        if (
            sim_data and "s21_sigma" in sim_data
            and sigma_cfg.get("enabled") and sigma_cfg.get("sigma_values")
        ):
            df = sim_data["s21_sigma"]
            if "Freq (MHz)" in df.columns:
                stdev_col = next((col for col in df.columns if "stdev" in col.lower()), None)
                if stdev_col:
                    sigma_freq = df["Freq (MHz)"].values
                    sigma_vals = df[stdev_col].values
                    sim_freq_mhz = sim_data["nominal"].frequency.f / 1e6
                    sim_s21_db = 20 * np.log10(np.abs(sim_data["nominal"].s[:, 1, 0]))
                    sim_sigma = np.interp(sim_freq_mhz, sigma_freq, sigma_vals)
                    sim_colors = ["purple", "darkviolet", "mediumorchid"]

                    for i, sigma_val in enumerate(sigma_cfg["sigma_values"]):
                        if i >= 3:
                            break
                        color = sim_colors[i % len(sim_colors)]
                        upper = sim_s21_db + sigma_val * sim_sigma
                        lower = sim_s21_db - sigma_val * sim_sigma

                        (u_line,) = plt.plot(sim_freq_mhz, upper, color=color, linestyle="-.", linewidth=2, alpha=0.8)
                        (l_line,) = plt.plot(sim_freq_mhz, lower, color=color, linestyle="-.", linewidth=2, alpha=0.8)

                        all_handles += [u_line, l_line]
                        all_labels += [f"Sim Mean+{sigma_val}σ S21 Rej", f"Sim Mean-{sigma_val}σ S21 Rej"]

        # Rejection spec lines
        for i, sb in enumerate(kpi_config.get("StopBands", [])):
            freq_start = sb["range"][0] / 1e6
            freq_end = sb["range"][1] / 1e6
            lsl = -abs(sb["LSL"])
            color = ["orange", "purple", "brown", "pink"][i % 4]
            freq_range = np.linspace(freq_start, freq_end, 100)

            if plot_config.get("plot_settings", {}).get("show_spec_lines", False):
                (spec_line,) = plt.plot(freq_range, [lsl] * len(freq_range), color=color, linestyle="--", linewidth=2)
                all_handles.append(spec_line)
                all_labels.append(f'{sb["name"]} LSL: {lsl} dB')

            plt.axvspan(freq_start, freq_end, alpha=0.1, color=color)

        # Final axis and legend
        config = plot_config["axis_ranges"]["s_parameter_plots"]["s21_rejection_loss"]
        plt.xlim(config["x_axis"]["min"], config["x_axis"]["max"])
        plt.ylim(config["y_axis"]["min"], config["y_axis"]["max"])
        plt.xlabel(f"Frequency ({config['x_axis']['unit']})")
        plt.ylabel(f"S21 ({config['y_axis']['unit']})")
        plt.title("S21 Rejection Loss vs Frequency (Stop-band)")
        plt.grid(True, alpha=0.3)

        plt.legend(all_handles, all_labels, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small", ncol=2, borderaxespad=0.1)
        plt.tight_layout()

        # Save final output
        temp_png = os.path.join(save_folder, "temp_s21_rejection_plot.png")
        final_jpg = os.path.join(save_folder, "S21_Rejection_Loss.jpg")
        plt.savefig(temp_png, dpi=100, bbox_inches="tight", pad_inches=0.1)
        plt.close()

        img = Image.open(temp_png).convert("RGB")
        img.save(final_jpg, format="JPEG", quality=70, optimize=True)
        os.remove(temp_png)

        return final_jpg

    except Exception as e:
        raise RuntimeError("Error in generate_s21_rejection_loss_plot") from e


# =============================================================================
# INDIVIDUAL ADVANCED PLOT FUNCTIONS
# =============================================================================


def generate_group_delay_plot(networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder):
    """Generate Group Delay Plot with file names, shift curves, sigma curves, and spec limits."""

    def phase_and_gd(network):
        f_hz = network.frequency.f
        ang = np.unwrap(np.angle(network.s[:, 1, 0]))
        df = np.gradient(f_hz)
        gd_s = -np.gradient(ang) / (2 * np.pi * (df + 1e-12))
        return ang, gd_s * 1e9  # in ns

    plt.figure(figsize=(20, 9))
    num_files = len(networks)
    color_norm = mcolors.Normalize(vmin=0, vmax=num_files - 1)
    scalar_map = cm.ScalarMappable(norm=color_norm, cmap="tab20")

    all_handles, all_labels = [], []
    positive_shift_plotted = negative_shift_plotted = False

    for idx, (filename, net) in enumerate(networks.items()):
        color = scalar_map.to_rgba(idx)

        # Nominal plot
        shifted_net = apply_frequency_shift(net, 0)
        freq_mhz = shifted_net.frequency.f / 1e6
        _, gd_ns = phase_and_gd(shifted_net)
        (line,) = plt.plot(freq_mhz, gd_ns, alpha=0.7, color=color)
        all_handles.append(line)
        all_labels.append(filename)

        # Frequency shifts
        for shift in freq_shifts:
            if shift == 0:
                continue
            shifted_net = apply_frequency_shift(net, shift)
            freq_mhz = shifted_net.frequency.f / 1e6
            _, gd_ns = phase_and_gd(shifted_net)
            style = "--" if shift > 0 else ":"
            (line,) = plt.plot(freq_mhz, gd_ns, alpha=0.7, color=color, linestyle=style)
            if shift > 0 and not positive_shift_plotted:
                all_handles.append(line)
                all_labels.append("+Shift")
                positive_shift_plotted = True
            elif shift < 0 and not negative_shift_plotted:
                all_handles.append(line)
                all_labels.append("-Shift")
                negative_shift_plotted = True

    all_labels = extract_unique_ids(all_labels)

    # Sigma curves
    if plot_config["sigma_settings"]["enabled"] and plot_config["sigma_settings"]["sigma_values"]:
        print("Calculating Group Delay sigma curves...")
        ref_net = list(networks.values())[0]
        ref_freq = ref_net.frequency.f
        ref_freq_mhz = ref_freq / 1e6
        gd_all = []

        for net in networks.values():
            shifted = apply_frequency_shift(net, 0)
            _, gd_ns = phase_and_gd(shifted)
            if len(shifted.frequency.f) == len(ref_freq):
                gd_all.append(gd_ns)
            else:
                gd_interp = np.interp(ref_freq_mhz, shifted.frequency.f / 1e6, gd_ns)
                gd_all.append(gd_interp)

        if len(gd_all) > 1:
            gd_arr = np.array(gd_all)
            gd_mean = np.mean(gd_arr, axis=0)
            gd_std = np.std(gd_arr, axis=0, ddof=1)

            colors, styles = ["purple", "orange", "brown"], ["-", "--", ":"]
            for i, sigma in enumerate(plot_config["sigma_settings"]["sigma_values"][:3]):
                upper = gd_mean + sigma * gd_std
                lower = gd_mean - sigma * gd_std
                (u,) = plt.plot(ref_freq_mhz, upper, color=colors[i], linestyle=styles[i], linewidth=1.5, alpha=0.8)
                (l,) = plt.plot(ref_freq_mhz, lower, color=colors[i], linestyle=styles[i], linewidth=1.5, alpha=0.8)
                all_handles += [u, l]
                all_labels += [f"Mean+{sigma}σ GD", f"Mean-{sigma}σ GD"]
                if i == 0:
                    (m,) = plt.plot(ref_freq_mhz, gd_mean, color="black", linestyle="-", linewidth=2, alpha=0.9)
                    all_handles.append(m)
                    all_labels.append("Mean GD")
        else:
            print("Warning: At least 2 measurement files are required for GD sigma calculation.")

    # KPI spec lines
    if "GD" in kpi_config["KPIs"]:
        colors = ["purple", "darkviolet", "magenta"]
        for i, spec in enumerate(kpi_config["KPIs"]["GD"]):
            f_start = spec["range"][0] / 1e6
            f_end = spec["range"][1] / 1e6
            color = colors[i % len(colors)]
            f_range = np.linspace(f_start, f_end, 100)

            if plot_config.get("plot_settings", {}).get("show_spec_lines", False):
                if "USL" in spec:
                    (line,) = plt.plot(f_range, [spec["USL"]] * len(f_range), color=color, linestyle="--", linewidth=2)
                    all_handles.append(line)
                    all_labels.append(f'{spec["name"]} USL: {spec["USL"]} ns')
                if "LSL" in spec:
                    (line,) = plt.plot(f_range, [spec["LSL"]] * len(f_range), color=color, linestyle=":", linewidth=2)
                    all_handles.append(line)
                    all_labels.append(f'{spec["name"]} LSL: {spec["LSL"]} ns')

            plt.axvspan(f_start, f_end, alpha=0.05, color=color)

    # Axis config and save
    cfg = plot_config["axis_ranges"]["advanced_plots"]["group_delay"]
    plt.xlim(cfg["x_axis"]["min"], cfg["x_axis"]["max"])
    plt.ylim(cfg["y_axis"]["min"], cfg["y_axis"]["max"])
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("GD (ns)")
    plt.title("Group Delay vs Frequency")
    plt.grid(True, alpha=0.3)
    plt.legend(all_handles, all_labels, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small", ncol=2)
    plt.tight_layout()

    temp_png = os.path.join(save_folder, "temp_gd_curve.png")
    final_jpg = os.path.join(save_folder, "GD_Curve.jpg")
    plt.savefig(temp_png, dpi=100, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    img = Image.open(temp_png).convert("RGB")
    img.save(final_jpg, format="JPEG", quality=70, optimize=True)
    os.remove(temp_png)

    return final_jpg


def generate_linear_phase_deviation_plot(networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder):
    """
    Generate Linear Phase Deviation (LPD) plot across frequency with support for:
    - Shifted networks
    - Sigma deviation envelopes
    - Spec limits (USL/LSL)
    - JPEG image export

    Returns:
        str: Path to saved plot image (PNG)
    """

    def phase_and_gd(network):
        f_hz = network.frequency.f
        ang = np.unwrap(np.angle(network.s[:, 1, 0]))
        df = np.gradient(f_hz)
        gd_s = -np.gradient(ang) / (2 * np.pi * (df + 1e-12))
        return ang, gd_s * 1e9  # GD in ns

    try:
        plt.figure(figsize=(16, 7))
        LPD_FREQ_LOW, LPD_FREQ_HIGH = kpi_config["KPIs"]["LPD_MIN"][0]["range"]

        all_handles, all_labels = [], []
        positive_shift_plotted = negative_shift_plotted = False

        # Plot all networks with and without shifts
        for fname, nt in networks.items():
            for shift in [0] + [s for s in freq_shifts if s != 0]:
                shifted_net = apply_frequency_shift(nt, shift)
                phase, _ = phase_and_gd(shifted_net)
                f_hz = shifted_net.frequency.f
                mask = (f_hz >= LPD_FREQ_LOW) & (f_hz <= LPD_FREQ_HIGH)

                if np.sum(mask) > 2:
                    f_fit = f_hz[mask]
                    phase_fit = phase[mask]
                    A = np.vstack([f_fit, np.ones_like(f_fit)]).T
                    slope, intercept = np.linalg.lstsq(A, phase_fit, rcond=None)[0]
                    linear_phase = slope * f_hz + intercept
                    lpd = np.degrees(phase - linear_phase)
                    lpd_band = lpd[mask]
                    f_band_mhz = f_hz[mask] / 1e6

                    if shift == 0:
                        (line,) = plt.plot(f_band_mhz, lpd_band, alpha=0.7)
                        all_handles.append(line)
                        all_labels.append(fname)
                    elif shift > 0 and not positive_shift_plotted:
                        (line,) = plt.plot(f_band_mhz, lpd_band, alpha=0.7, linestyle="--")
                        all_handles.append(line)
                        all_labels.append("+Shift")
                        positive_shift_plotted = True
                    elif shift < 0 and not negative_shift_plotted:
                        (line,) = plt.plot(f_band_mhz, lpd_band, alpha=0.7, linestyle=":")
                        all_handles.append(line)
                        all_labels.append("-Shift")
                        negative_shift_plotted = True

        all_labels = extract_unique_ids(all_labels)

        # Sigma LPD calculation
        sigma_cfg = plot_config.get("sigma_settings", {})
        if sigma_cfg.get("enabled") and sigma_cfg.get("sigma_values"):
            print("Calculating LPD sigma curves...")
            ref_net = list(networks.values())[0]
            ref_freq = ref_net.frequency.f
            mask = (ref_freq >= LPD_FREQ_LOW) & (ref_freq <= LPD_FREQ_HIGH)
            ref_freq_band_hz = ref_freq[mask]
            ref_freq_band_mhz = ref_freq_band_hz / 1e6

            lpd_collection = []
            for net in networks.values():
                shifted = apply_frequency_shift(net, 0)
                phase, _ = phase_and_gd(shifted)
                f_hz = shifted.frequency.f
                mask_net = (f_hz >= LPD_FREQ_LOW) & (f_hz <= LPD_FREQ_HIGH)

                if np.sum(mask_net) > 2:
                    f_fit = f_hz[mask_net]
                    phase_fit = phase[mask_net]
                    A = np.vstack([f_fit, np.ones_like(f_fit)]).T
                    slope, intercept = np.linalg.lstsq(A, phase_fit, rcond=None)[0]
                    linear_phase = slope * f_hz + intercept
                    lpd_deg = np.degrees(phase - linear_phase)
                    lpd_band = lpd_deg[mask_net]

                    if len(f_hz[mask_net]) == len(ref_freq_band_hz):
                        lpd_collection.append(lpd_band)
                    else:
                        lpd_interp = np.interp(ref_freq_band_mhz, f_hz[mask_net] / 1e6, lpd_band)
                        lpd_collection.append(lpd_interp)

            if len(lpd_collection) > 1:
                lpd_array = np.array(lpd_collection)
                lpd_mean = np.mean(lpd_array, axis=0)
                lpd_std = np.std(lpd_array, axis=0, ddof=1)

                colors, styles = ["purple", "orange", "brown"], ["-", "--", ":"]
                for i, sigma_val in enumerate(sigma_cfg["sigma_values"][:3]):
                    upper = lpd_mean + sigma_val * lpd_std
                    lower = lpd_mean - sigma_val * lpd_std

                    (u,) = plt.plot(ref_freq_band_mhz, upper, color=colors[i], linestyle=styles[i], linewidth=1.5, alpha=0.8)
                    (l,) = plt.plot(ref_freq_band_mhz, lower, color=colors[i], linestyle=styles[i], linewidth=1.5, alpha=0.8)

                    all_handles += [u, l]
                    all_labels += [f"Mean+{sigma_val}σ LPD", f"Mean-{sigma_val}σ LPD"]

                    if i == 0:
                        (m,) = plt.plot(ref_freq_band_mhz, lpd_mean, color="black", linestyle="-", linewidth=2, alpha=0.9)
                        all_handles.append(m)
                        all_labels.append("Mean LPD")

            else:
                print("Warning: Need at least 2 measurement files for sigma calculations.")

        # Spec lines
        color_cycle = ["red", "darkred", "crimson", "orange", "darkorange", "orangered"]
        spec_index = 0

        for side, label_key in [("LPD_MIN", "LSL"), ("LPD_MAX", "USL")]:
            for spec in kpi_config["KPIs"].get(side, []):
                f_start = spec["range"][0] / 1e6
                f_end = spec["range"][1] / 1e6
                val = spec[label_key]
                color = color_cycle[spec_index % len(color_cycle)]
                spec_index += 1
                f_range = np.linspace(f_start, f_end, 100)

                if plot_config.get("plot_settings", {}).get("show_spec_lines", False):
                    (line,) = plt.plot(f_range, [val] * len(f_range), color=color, linestyle="--", linewidth=2)
                    all_handles.append(line)
                    all_labels.append(f'{spec["name"]} {label_key}: {val} deg')

                plt.axvspan(f_start, f_end, alpha=0.05, color=color)

        # Final plot config and save
        cfg = plot_config["axis_ranges"]["advanced_plots"]["linear_phase_deviation"]
        plt.xlim(cfg["x_axis"]["min"], cfg["x_axis"]["max"])
        plt.ylim(cfg["y_axis"]["min"], cfg["y_axis"]["max"])
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("LPD (deg)")
        plt.title("Linear Phase Deviation (LPD) vs Frequency")
        plt.grid(True, alpha=0.3)
        plt.legend(all_handles, all_labels, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small", ncol=2)
        plt.tight_layout()

        path = os.path.join(save_folder, "LPD_Curve.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

        return path

    except Exception as e:
        raise RuntimeError("Error in generate_linear_phase_deviation_plot") from e


def generate_flatness_scatter_plot(networks, plot_config, kpi_config, save_folder):
    """
    Generate Flatness Scatter Plot:
    - Flatness (max - min S21) across frequency bands
    - Unit-wise scatter plot with varying point sizes
    - USL spec lines per band if configured
    - Saves plot as PNG

    Returns:
        str: Path to the saved flatness scatter plot image
    """

    try:
        plt.figure(figsize=(12, 6))
        colors = ["blue", "red", "green", "orange", "purple", "brown", "magenta"]

        all_handles = []
        all_labels = []

        for band_idx, band in enumerate(kpi_config["KPIs"].get("Flat", [])):
            lo, hi = band["range"]
            color = colors[band_idx % len(colors)]

            unit_numbers = []
            flatness_values = []
            point_sizes = []

            for unit_idx, (fname, nt) in enumerate(networks.items(), start=1):
                f_hz = nt.frequency.f
                s21_db = 20 * np.log10(np.abs(nt.s[:, 1, 0]))
                mask = (f_hz >= lo) & (f_hz <= hi)

                if np.any(mask):
                    flat = s21_db[mask].max() - s21_db[mask].min()
                    unit_numbers.append(unit_idx)
                    flatness_values.append(flat)
                    point_sizes.append(40 + (unit_idx * 5))  # Emphasize later units slightly

            # Scatter plot for the band
            scatter = plt.scatter(
                unit_numbers,
                flatness_values,
                color=color,
                alpha=0.7,
                s=point_sizes,
                label=f"{band['name']} ({lo/1e6:.0f}-{hi/1e6:.0f} MHz)"
            )
            all_handles.append(scatter)
            all_labels.append(f"{band['name']} ({lo / 1e6:.0f}-{hi / 1e6:.0f} MHz)")

            # Add USL spec line if present
            if plot_config.get("plot_settings", {}).get("show_spec_lines", False) and "USL" in band:
                (line,) = plt.axhline(
                    y=band["USL"],
                    color=color,
                    linestyle="--",
                    linewidth=2,
                    alpha=0.5
                )
                all_handles.append(line)
                all_labels.append(f"{band['name']} USL: {band['USL']} dB")

        plt.xlabel("Unit Number")
        plt.ylabel("Flatness (dB)")
        plt.title("S21 Flatness vs Unit Number (Variable Point Sizes)")
        plt.grid(True, alpha=0.3)
        plt.legend(all_handles, all_labels, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
        plt.tight_layout()

        plot_path = os.path.join(save_folder, "Flatness_Scatter.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        return plot_path

    except Exception as e:
        raise RuntimeError("Error in generate_flatness_scatter_plot") from e


def generate_gdv_scatter_plot(networks, plot_config, kpi_config, save_folder):
    """
    Generate Group Delay Variation (GDV) Scatter Plot:
    - Unit-wise scatter of peak-to-peak group delay variation within each defined band
    - Optional USL spec lines and colored markers per band
    - Saves as a high-resolution PNG image

    Returns:
        str: Path to saved GDV scatter plot image
    """

    def phase_and_gd(network):
        f_hz = network.frequency.f
        ang = np.unwrap(np.angle(network.s[:, 1, 0]))
        df = np.gradient(f_hz)
        gd_s = -np.gradient(ang) / (2 * np.pi * (df + 1e-12))
        return ang, gd_s * 1e9  # Convert to nanoseconds

    try:
        plt.figure(figsize=(12, 6))
        colors = ["blue", "red", "green", "orange", "purple", "brown", "magenta"]

        all_handles = []
        all_labels = []

        for band_idx, band in enumerate(kpi_config["KPIs"].get("GDV", [])):
            lo, hi = band["range"]
            color = colors[band_idx % len(colors)]

            unit_numbers = []
            gdv_values = []

            for unit_idx, (fname, nt) in enumerate(networks.items(), start=1):
                f_hz = nt.frequency.f
                _, gd_ns = phase_and_gd(nt)
                mask = (f_hz >= lo) & (f_hz <= hi)

                if np.any(mask):
                    gdv = np.ptp(gd_ns[mask])  # Peak-to-peak variation
                    unit_numbers.append(unit_idx)
                    gdv_values.append(gdv)

            scatter = plt.scatter(
                unit_numbers,
                gdv_values,
                color=color,
                alpha=0.7,
                s=50,
                label=f"{band['name']} ({lo / 1e6:.0f}-{hi / 1e6:.0f} MHz)"
            )
            all_handles.append(scatter)
            all_labels.append(f"{band['name']} ({lo / 1e6:.0f}-{hi / 1e6:.0f} MHz)")

            # Draw spec line if enabled
            if plot_config.get("plot_settings", {}).get("show_spec_lines", False) and "USL" in band:
                (line,) = plt.axhline(
                    y=band["USL"],
                    color=color,
                    linestyle="--",
                    linewidth=2,
                    alpha=0.5
                )
                all_handles.append(line)
                all_labels.append(f"{band['name']} USL: {band['USL']} ns")

        # Plot styling
        plt.xlabel("Unit Number")
        plt.ylabel("Group Delay Variation (ns)")
        plt.title("Group Delay Variation vs Unit Number")
        plt.grid(True, alpha=0.3)
        plt.legend(all_handles, all_labels, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
        plt.tight_layout()

        plot_path = os.path.join(save_folder, "GDV_Scatter.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        return plot_path

    except Exception as e:
        raise RuntimeError("Error in generate_gdv_scatter_plot") from e


# =============================================================================
# INDIVIDUAL STATISTICAL PLOT FUNCTIONS
# =============================================================================


def generate_individual_box_plot(param_name, param_data, save_folder):
    """
    Generate and save a box plot with statistical overlay for a single parameter.

    Args:
        param_name (str): Name of the parameter (used in title and filename).
        param_data (pd.Series): Parameter values to plot.
        save_folder (str): Directory to save the image.

    Returns:
        str: Full path to the saved PNG image.
    """

    try:
        plt.figure(figsize=(10, 6))

        # Draw box plot
        box_data = [param_data.values]
        bp = plt.boxplot(box_data, labels=[param_name], patch_artist=True)
        bp["boxes"][0].set_facecolor("lightblue")
        bp["boxes"][0].set_alpha(0.7)

        # Plot individual data points
        plt.scatter([1] * len(param_data), param_data, color="red", alpha=0.6, s=30, zorder=3)

        # Axis labels and title
        plt.xlabel("Parameter")
        plt.ylabel("Value")
        plt.title(f"{param_name} - Statistical Distribution")
        plt.grid(True, alpha=0.3)

        # Statistics annotation box
        stats_text = (
            f"Mean: {param_data.mean():.3f}\n"
            f"Std: {param_data.std():.3f}\n"
            f"Min: {param_data.min():.3f}\n"
            f"Max: {param_data.max():.3f}"
        )
        plt.text(
            0.02, 0.98, stats_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()

        # Save plot
        filename = f"BoxPlot_{param_name.replace(' ', '_')}.png"
        plot_path = os.path.join(save_folder, filename)
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        return plot_path

    except Exception as e:
        raise RuntimeError(f"Error generating box plot for {param_name}") from e


def generate_individual_histogram_plot(param_name, param_data, save_folder):
    """
    Generate and save a histogram plot for a parameter.

    Args:
        param_name (str): Name of the parameter.
        param_data (pd.Series): Data for the parameter.
        save_folder (str): Directory to save the plot.

    Returns:
        str: Full path to the saved histogram plot.
    """

    try:
        plt.figure(figsize=(10, 6))
        values = param_data.values

        # Create histogram
        bins = min(20, max(5, len(values) // 3))
        plt.hist(
            values,
            bins=bins,
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
            label=f"N = {len(values)}"
        )

        # Plot styling
        plt.title(f"{param_name} - Distribution Histogram")
        plt.xlabel(param_name)
        plt.ylabel("Number of Units")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        # Save to file
        filename = f"Histogram_{param_name.replace(' ', '_')}.png"
        plot_path = os.path.join(save_folder, filename)
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        return plot_path

    except Exception as e:
        raise RuntimeError(f"Error generating histogram plot for {param_name}") from e



# =============================================================================
# CATEGORY MASTER FUNCTIONS
# =============================================================================


def create_s_parameter_plots(
    networks, plot_config, kpi_config, sim_data=None, save_folder="."
):
    """
    Create and save all major S-parameter plots (S11, S22, S21 IL, S21 Rejection).

    Args:
        networks (dict): Dictionary of filename → network object (e.g., from scikit-rf).
        plot_config (dict): Configuration dictionary for plot styling and limits.
        kpi_config (dict): Configuration dictionary for KPI specs like LSL/USL.
        sim_data (dict, optional): Simulation network and sigma data. Default is None.
        save_folder (str, optional): Path to save all plots. Default is current directory.

    Returns:
        list: List of filenames of the successfully saved plots.
    """
    plots_created = []

    try:
        freq_shifts = calculate_freq_shifts(plot_config)
    except Exception as e:
        print(f"[Error] Failed to calculate frequency shifts: {e}")
        return plots_created

    # Plot generation with error handling
    try:
        plots_created.append(
            generate_s11_return_loss_plot(
                networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder
            )
        )
    except Exception as e:
        print(f"[Error] S11 plot failed: {e}")

    try:
        plots_created.append(
            generate_s22_return_loss_plot(
                networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder
            )
        )
    except Exception as e:
        print(f"[Error] S22 plot failed: {e}")

    try:
        plots_created.append(
            generate_s21_insertion_loss_plot(
                networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder
            )
        )
    except Exception as e:
        print(f"[Error] S21 insertion loss plot failed: {e}")

    try:
        plots_created.append(
            generate_s21_rejection_loss_plot(
                networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder
            )
        )
    except Exception as e:
        print(f"[Error] S21 rejection loss plot failed: {e}")

    return plots_created


def create_advanced_plots(
    plot_config, kpi_config, networks, freq_shifts, save_folder="."
):
    """
    Create and save advanced S-parameter plots.

    This includes:
        - Group Delay plot
        - Linear Phase Deviation plot
        - Flatness Scatter plot
        - GDV (Group Delay Variation) Scatter plot

    Args:
        plot_config (dict): Plot configuration dictionary (axis limits, styles, etc.).
        kpi_config (dict): KPI configuration dictionary (spec limits, ranges).
        networks (dict): Dictionary of filename → network object.
        freq_shifts (list): List of frequency shifts to apply (e.g., [0, +shift, -shift]).
        save_folder (str): Directory to save the plots. Default is current folder.

    Returns:
        list: Filenames of successfully created plots.
    """
    plots_created = []

    try:
        plots_created.append(
            generate_group_delay_plot(
                networks, plot_config, kpi_config, {}, freq_shifts, save_folder
            )
        )
    except Exception as e:
        print(f"[Error] Group Delay plot failed: {e}")

    try:
        plots_created.append(
            generate_linear_phase_deviation_plot(
                networks, plot_config, kpi_config, {}, freq_shifts, save_folder
            )
        )
    except Exception as e:
        print(f"[Error] Linear Phase Deviation plot failed: {e}")

    try:
        plots_created.append(
            generate_flatness_scatter_plot(
                networks, plot_config, kpi_config, save_folder
            )
        )
    except Exception as e:
        print(f"[Error] Flatness Scatter plot failed: {e}")

    try:
        plots_created.append(
            generate_gdv_scatter_plot(
                networks, plot_config, kpi_config, save_folder
            )
        )
    except Exception as e:
        print(f"[Error] GDV Scatter plot failed: {e}")

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
    """
    Generate all S-parameter and advanced plots based on input configuration and data files.

    Args:
        plot_config_data (dict): Dictionary for plot configuration (axis, styles, sigma, etc.).
        excel_files (list): Excel files for reference or future expansion (currently unused).
        s2p_files (list): List of measured S2P Touchstone files.
        sim_s2p_files (list, optional): List of simulated S2P files.
        s11_sigma_files (list, optional): CSV or Excel files containing S11 sigma curve data.
        s21_sigma_files (list, optional): CSV or Excel files containing S21 sigma curve data.
        kpi_config_data (dict, optional): KPI specs configuration.
        save_folder (str, optional): Folder to save the generated plots. Defaults to current folder.

    Returns:
        list: List of saved plot filenames (JPG/PNG).
    """
    try:
        # Load configuration
        plot_config, kpi_config = load_configuration(plot_config_data, kpi_config_data)
        if not kpi_config or "KPIs" not in kpi_config:
            raise ValueError("kpi_config_data must be provided and contain a 'KPIs' key.")

        # Load measurement and simulation data
        networks = load_s2p_files(s2p_files)
        sim_data = load_simulation_data(sim_s2p_files, s11_sigma_files, s21_sigma_files)

        all_plots = []

        # S-parameter continuous plots (S11, S22, S21 IL, S21 Rej)
        s_param_plots = create_s_parameter_plots(
            networks, plot_config, kpi_config, sim_data, save_folder
        )
        all_plots.extend(s_param_plots)

        # Advanced plots (GD, LPD, Flatness, GDV)
        freq_shifts = calculate_freq_shifts(plot_config)
        advanced_plots = create_advanced_plots(
            plot_config, kpi_config, networks, freq_shifts, save_folder
        )
        all_plots.extend(advanced_plots)

        return all_plots

    except Exception as e:
        print(f"[Error] Failed to generate S-parameter plots: {e}")
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
    """
    Generate all required plots:
    - S-parameter continuous plots (S11, S22, S21 IL, S21 Rej)
    - Advanced plots (Group Delay, LPD, Flatness, GDV)
    - Statistical plots (Box plots, Histograms)

    Args:
        plot_config_data (dict): Plot configuration dictionary.
        excel_files (list): Excel files for statistical parameter data.
        s2p_files (list): Measured S2P files.
        sim_s2p_files (list, optional): Simulated S2P files.
        s11_sigma_files (list, optional): Sigma data for S11.
        s21_sigma_files (list, optional): Sigma data for S21.
        kpi_config_data (dict, optional): KPI configuration dictionary.
        save_folder (str, optional): Directory to save plots.

    Returns:
        list: List of all generated plot filenames.
    """
    try:
        # Load configurations and data
        plot_config, kpi_config = load_configuration(plot_config_data, kpi_config_data)
        if not kpi_config or "KPIs" not in kpi_config:
            raise ValueError("kpi_config_data must contain a 'KPIs' key.")

        networks = load_s2p_files(s2p_files)
        sim_data = load_simulation_data(sim_s2p_files, s11_sigma_files, s21_sigma_files)
        excel_data = load_excel_data(excel_files)
        freq_shifts = calculate_freq_shifts(plot_config)

        all_plots = []

        # S-Parameter Plots
        all_plots.extend([
            generate_s11_return_loss_plot(networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder),
            generate_s22_return_loss_plot(networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder),
            generate_s21_insertion_loss_plot(networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder),
            generate_s21_rejection_loss_plot(networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder),
        ])

        # Advanced Plots
        all_plots.extend([
            generate_group_delay_plot(networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder),
            generate_linear_phase_deviation_plot(networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder),
            generate_flatness_scatter_plot(networks, plot_config, kpi_config, save_folder),
            generate_gdv_scatter_plot(networks, plot_config, kpi_config, save_folder),
        ])

        # Statistical Plots (Box + Histogram)
        parameter_columns = [col for col in excel_data["Per_File"].columns if col != "File"]

        for param in parameter_columns:
            param_data = excel_data["Per_File"][param].dropna()
            if len(param_data) > 0:
                all_plots.append(generate_individual_box_plot(param, param_data, save_folder))
                all_plots.append(generate_individual_histogram_plot(param, param_data, save_folder))

        return all_plots

    except Exception as e:
        print(f"[Error] Failed to generate all plots: {e}")
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
    """
    Create individual box plots for each parameter from Excel statistical data and save in the provided folder.

    Args:
        excel_data (dict): Dictionary containing DataFrames loaded from Excel sheets.
        plot_config (dict): Plot configuration settings.
        save_folder (str): Directory where plots will be saved.

    Returns:
        list: List of filenames of the generated box plots.
    """
    plots_created = []

    try:
        if "Per_File" not in excel_data:
            raise KeyError("Missing 'Per_File' sheet in Excel data.")

        per_file_df = excel_data["Per_File"]
        parameter_columns = [col for col in per_file_df.columns if col.lower() != "file"]

        for param in parameter_columns:
            try:
                param_values = per_file_df[param].dropna()

                if len(param_values) == 0:
                    continue

                plt.figure(figsize=(10, 6))
                box_data = [param_values.values]
                bp = plt.boxplot(box_data, labels=[param], patch_artist=True)
                bp["boxes"][0].set_facecolor("lightblue")
                bp["boxes"][0].set_alpha(0.7)

                x_values = [1] * len(param_values)
                plt.scatter(x_values, param_values.values, alpha=0.6, color="red", s=30, zorder=3)

                plt.ylabel("Value")
                plt.xlabel("Parameter")
                plt.title(f"{param} - Statistical Distribution")
                plt.grid(True, alpha=0.3)

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

                filename = f"BoxPlot_{param}.png"
                plot_path = os.path.join(save_folder, filename)
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()

                plots_created.append(filename)
                print(f"Created box plot: {plot_path}")
            except Exception as param_err:
                print(f"[Warning] Skipped plotting for '{param}' due to error: {param_err}")
                plt.close()  # Ensure no figure remains open

    except Exception as e:
        print(f"[Error] Failed to create statistical plots: {e}")

    return plots_created


def create_histogram_plots(excel_data, plot_config, save_folder="."):
    """
    Create individual histogram plots for each parameter and save in the provided folder.

    Args:
        excel_data (dict): Dictionary containing DataFrames loaded from Excel sheets.
        plot_config (dict): Plot configuration settings.
        save_folder (str): Directory where plots will be saved.

    Returns:
        list: List of filenames of the generated histogram plots.
    """
    plots_created = []

    try:
        if "Per_File" not in excel_data:
            raise KeyError("Missing 'Per_File' sheet in Excel data.")

        per_file_df = excel_data["Per_File"]
        parameter_columns = [col for col in per_file_df.columns if col.lower() != "file"]

        for param in parameter_columns:
            try:
                param_values = per_file_df[param].dropna()
                if len(param_values) == 0:
                    continue

                plt.figure(figsize=(10, 6))

                values = param_values.values
                plt.hist(
                    values,
                    bins=min(20, max(5, len(values) // 3)),
                    alpha=0.7,
                    color="skyblue",
                    edgecolor="black",
                )

                plt.title(f"{param} - Distribution Histogram")
                plt.xlabel(param)
                plt.ylabel("Number of units")
                plt.grid(True, alpha=0.3)
                plt.legend()

                plt.tight_layout()

                filename = f"Histogram_{param}.png"
                plot_path = os.path.join(save_folder, filename)
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()

                plots_created.append(filename)
                print(f"Created histogram: {plot_path}")

            except Exception as param_err:
                print(f"[Warning] Skipped histogram for '{param}' due to error: {param_err}")
                plt.close()  # Ensure no plot remains open in case of failure

    except Exception as e:
        print(f"[Error] Failed to create histogram plots: {e}")

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
            raise ValueError("kpi_config_data must be provided and contain a 'KPIs' key.")

        networks = load_s2p_files(s2p_files)
        sim_data = load_simulation_data(sim_s2p_files, s11_sigma_files, s21_sigma_files)

        all_plots = []
        requested = set(
            [p.lower() for p in plot_config.get("parameters_for_statistical_plots", [])]
        )
        freq_shifts = calculate_freq_shifts(plot_config)

        # S-parameter plots
        if "s11" in requested:
            try:
                all_plots.append(generate_s11_return_loss_plot(
                    networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder
                ))
            except Exception as e:
                print(f"[Warning] Failed to generate S11 plot: {e}")

        if "s22" in requested:
            try:
                all_plots.append(generate_s22_return_loss_plot(
                    networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder
                ))
            except Exception as e:
                print(f"[Warning] Failed to generate S22 plot: {e}")

        if "s21_insertion" in requested:
            try:
                all_plots.append(generate_s21_insertion_loss_plot(
                    networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder
                ))
            except Exception as e:
                print(f"[Warning] Failed to generate S21 Insertion Loss plot: {e}")

        if "s21_rejection" in requested:
            try:
                all_plots.append(generate_s21_rejection_loss_plot(
                    networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder
                ))
            except Exception as e:
                print(f"[Warning] Failed to generate S21 Rejection Loss plot: {e}")

        # Advanced plots
        if "group_delay" in requested:
            try:
                all_plots.append(generate_group_delay_plot(
                    networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder
                ))
            except Exception as e:
                print(f"[Warning] Failed to generate Group Delay plot: {e}")

        if "linear_phase_deviation" in requested:
            try:
                all_plots.append(generate_linear_phase_deviation_plot(
                    networks, plot_config, kpi_config, sim_data, freq_shifts, save_folder
                ))
            except Exception as e:
                print(f"[Warning] Failed to generate Linear Phase Deviation plot: {e}")

        if "flatness" in requested:
            try:
                all_plots.append(generate_flatness_scatter_plot(
                    networks, plot_config, kpi_config, save_folder
                ))
            except Exception as e:
                print(f"[Warning] Failed to generate Flatness plot: {e}")

        if "group_delay_variation" in requested:
            try:
                all_plots.append(generate_gdv_scatter_plot(
                    networks, plot_config, kpi_config, save_folder
                ))
            except Exception as e:
                print(f"[Warning] Failed to generate Group Delay Variation plot: {e}")

        return all_plots

    except Exception as e:
        print(f"[Error] Failed to generate selected S-parameter and advanced plots: {e}")
        return []


def generate_selected_statistical_and_histogram_plots(
    plot_config_data, excel_files, save_folder="."
):
    try:
        plot_config, _ = load_configuration(plot_config_data)
        excel_data = load_excel_data(excel_files)

        all_plots = []

        # Generate statistical plots
        try:
            statistical_plots = create_statistical_plots(
                excel_data, plot_config, save_folder
            )
            all_plots.extend(statistical_plots)
        except Exception as e:
            print(f"[Warning] Failed to generate statistical (box) plots: {e}")

        # Generate histogram plots
        try:
            histogram_plots = create_histogram_plots(
                excel_data, plot_config, save_folder
            )
            all_plots.extend(histogram_plots)
        except Exception as e:
            print(f"[Warning] Failed to generate histogram plots: {e}")

        return all_plots

    except Exception as e:
        print(f"[Error] Failed during overall statistical/histogram plot generation: {e}")
        return []
