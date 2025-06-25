import os

def generate_summary(s2p_path, plot_path, kpi_data):
    # 🚀 Dummy process: Save kpi_data to an Excel file for now
    import pandas as pd

    # Prepare paths
    summary_excel_path = os.path.join(s2p_path, 'SParam_Summary.xlsx')

    # Convert JSON to DataFrame (for example)
    df = pd.json_normalize(kpi_data)

    # Save to Excel
    with pd.ExcelWriter(summary_excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Summary', index=False)

    return summary_excel_path
