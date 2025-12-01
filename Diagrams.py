import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import os
import numpy as np

def generate_site_reports(df: pd.DataFrame, output_folder="site_reports"):
    """
    Splits data by 'site_national_code' and generates a PDF report 
    containing Histograms and Box Plots for each site.
    """
    # 1. Create directory
    os.makedirs(output_folder, exist_ok=True)

    # 2. Identify numeric columns (excluding IDs)
    numeric_cols = df.select_dtypes(include="number").columns
    # Filter out likely ID columns or constant columns
    numeric_cols = [c for c in numeric_cols if df[c].nunique() > 1 and not c.lower().endswith('id')]
    
    # 3. Get unique sites
    if 'site_national_code' not in df.columns:
        raise ValueError("Column 'site_national_code' not found in DataFrame.")
        
    sites = df['site_national_code'].unique()
    print(f"Found {len(sites)} sites: {sites}")

    # 4. Iterate through each site
    for site in sites:
        print(f"Processing Site: {site}...")
        
        # Filter data
        site_df = df[df['site_national_code'] == site]
        
        if site_df.empty:
            continue
            
        # Define PDF path
        pdf_path = os.path.join(output_folder, f"Report_{site}.pdf")
        
        with PdfPages(pdf_path) as pdf:
            # --- PART A: HISTOGRAMS ---
            # Loop in batches of 9 for 3x3 grid
            for page_num, i in enumerate(range(0, len(numeric_cols), 9)):
                cols_batch = numeric_cols[i:i+9]
                
                fig, axes = plt.subplots(3, 3, figsize=(15, 12))
                axes = axes.flatten()
                
                for ax, col in zip(axes, cols_batch):
                    data = site_df[col].dropna()
                    
                    if len(data) == 0:
                        ax.text(0.5, 0.5, "No Data", ha='center')
                        continue

                    # Discrete vs Continuous logic
                    if data.nunique() < 10:
                        ax.hist(data, bins=data.nunique(), color='skyblue', edgecolor='black')
                    else:
                        ax.hist(data, bins=30, color='steelblue', alpha=0.7)
                    
                    ax.set_title(col, fontsize=9, fontweight='bold')
                    ax.tick_params(axis='x', rotation=45, labelsize=8)
                    ax.grid(axis='y', linestyle='--', alpha=0.5)

                # Clean up empty axes
                for j in range(len(cols_batch), 9):
                    axes[j].axis("off")

                plt.suptitle(f"Site {site} - Histograms (Page {page_num + 1})", fontsize=16)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                pdf.savefig(fig)
                plt.close(fig)
                
            # --- PART B: BOX PLOTS ---
            # Loop in batches of 9 for 3x3 grid
            for page_num, i in enumerate(range(0, len(numeric_cols), 9)):
                cols_batch = numeric_cols[i:i+9]
                
                fig, axes = plt.subplots(3, 3, figsize=(15, 12))
                axes = axes.flatten()
                
                for ax, col in zip(axes, cols_batch):
                    data = site_df[col].dropna()
                    
                    if len(data) == 0:
                        ax.text(0.5, 0.5, "No Data", ha='center')
                        continue

                    # Horizontal Box Plot
                    sns.boxplot(x=data, ax=ax, color='lightgreen')
                    
                    ax.set_title(col, fontsize=9, fontweight='bold')
                    ax.tick_params(axis='x', rotation=45, labelsize=8)

                # Clean up empty axes
                for j in range(len(cols_batch), 9):
                    axes[j].axis("off")

                plt.suptitle(f"Site {site} - Box Plots (Page {page_num + 1})", fontsize=16)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                pdf.savefig(fig)
                plt.close(fig)

        print(f"Saved report: {pdf_path}")

# --- Usage ---
df = pd.read_csv("Normal_Data.csv")
generate_site_reports(df)