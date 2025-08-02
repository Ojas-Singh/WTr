"""
Report generation and result summarization.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
import os

def load_results(run_dir: str) -> List[Dict]:
    """Load all result files from a run directory."""
    results = []
    run_path = Path(run_dir)
    
    # Find all result.json files in evaluation subdirectories
    for eval_dir in run_path.glob("eval_*"):
        result_file = eval_dir / "result.json"
        if result_file.exists():
            try:
                with open(result_file, 'r') as f:
                    result = json.load(f)
                    results.append(result)
            except Exception as e:
                print(f"Warning: Could not load {result_file}: {e}")
    
    return results

def create_summary_table(results: List[Dict], top_n: int = 10) -> pd.DataFrame:
    """Create summary table of top results."""
    
    if not results:
        return pd.DataFrame()
    
    # Sort by 10K barrier
    sorted_results = sorted(results, key=lambda r: r.get('deltaG_dagger_10K', float('inf')))
    top_results = sorted_results[:top_n]
    
    # Create DataFrame
    data = []
    for i, result in enumerate(top_results):
        descriptors = result.get('descriptors', {})
        
        row = {
            'Rank': i + 1,
            'Surface_ID': result.get('surface_id', 'Unknown'),
            'ΔE‡ (eV)': result.get('deltaE_dagger', 0.0),
            'ΔG‡_10K (eV)': result.get('deltaG_dagger_10K', 0.0),
            'ΔG‡_20K (eV)': result.get('deltaG_dagger_20K', 0.0),
            'k_10K (s⁻¹)': result.get('rate_10K', 0.0),
            'k_20K (s⁻¹)': result.get('rate_20K', 0.0),
            'E_field_proj (V/Å)': descriptors.get('efield_proj', 0.0),
            'Donors': descriptors.get('donor_count', 0),
            'Acceptors': descriptors.get('acceptor_count', 0),
            'Wire_length (Å)': descriptors.get('wire_length', 0.0),
            'Strain': descriptors.get('strain_penalty', 0.0)
        }
        data.append(row)
    
    return pd.DataFrame(data)

def plot_barrier_distribution(results: List[Dict], output_path: str):
    """Plot distribution of barrier heights."""
    if not results:
        return
    
    barriers_10K = [r.get('deltaG_dagger_10K', 0.0) for r in results]
    barriers_20K = [r.get('deltaG_dagger_20K', 0.0) for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 10K barriers
    ax1.hist(barriers_10K, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('ΔG‡ at 10K (eV)')
    ax1.set_ylabel('Count')
    ax1.set_title('Barrier Distribution (10K)')
    ax1.grid(True, alpha=0.3)
    
    # 20K barriers
    ax2.hist(barriers_20K, bins=20, alpha=0.7, color='red', edgecolor='black')
    ax2.set_xlabel('ΔG‡ at 20K (eV)')
    ax2.set_ylabel('Count')
    ax2.set_title('Barrier Distribution (20K)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_descriptor_correlations(results: List[Dict], output_path: str):
    """Plot correlations between descriptors and barriers."""
    if not results:
        return
    
    # Extract data
    data = []
    for result in results:
        descriptors = result.get('descriptors', {})
        row = {
            'barrier_10K': result.get('deltaG_dagger_10K', 0.0),
            'efield_proj': descriptors.get('efield_proj', 0.0),
            'donors': descriptors.get('donor_count', 0),
            'acceptors': descriptors.get('acceptor_count', 0),
            'wire_length': descriptors.get('wire_length', 0.0),
            'strain': descriptors.get('strain_penalty', 0.0)
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    if df.empty:
        return
    
    # Create correlation plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    descriptors = ['efield_proj', 'donors', 'wire_length', 'strain']
    labels = ['E-field projection (V/Å)', 'Donor count', 'Wire length (Å)', 'Strain penalty']
    
    for i, (desc, label) in enumerate(zip(descriptors, labels)):
        if i < len(axes) and desc in df.columns:
            axes[i].scatter(df[desc], df['barrier_10K'], alpha=0.6, s=30)
            axes[i].set_xlabel(label)
            axes[i].set_ylabel('ΔG‡ at 10K (eV)')
            axes[i].grid(True, alpha=0.3)
            
            # Calculate correlation
            try:
                corr = df[desc].corr(df['barrier_10K'])
                axes[i].set_title(f'Correlation: {corr:.3f}')
            except:
                axes[i].set_title('Correlation: N/A')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def generate_html_report(summary_df: pd.DataFrame, run_dir: str, 
                        plot_files: List[str], output_path: str):
    """Generate HTML report."""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>WTr Results Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 30px 0; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .plot {{ text-align: center; margin: 20px 0; }}
            .plot img {{ max-width: 800px; width: 100%; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>WTr Results Report</h1>
            <p><strong>Run Directory:</strong> {run_dir}</p>
            <p><strong>Total Configurations:</strong> {len(summary_df)}</p>
            <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>Top Surface Configurations</h2>
            {summary_df.to_html(index=False, table_id="results_table")}
        </div>
        
        <div class="section">
            <h2>Visualizations</h2>
    """
    
    for plot_file in plot_files:
        if os.path.exists(plot_file):
            plot_name = os.path.basename(plot_file)
            html_content += f"""
            <div class="plot">
                <h3>{plot_name.replace('_', ' ').replace('.png', '').title()}</h3>
                <img src="{plot_name}" alt="{plot_name}">
            </div>
            """
    
    html_content += """
        </div>
        
        <div class="section">
            <h2>Summary Statistics</h2>
            <ul>
    """
    
    if not summary_df.empty:
        best_barrier = summary_df['ΔG‡_10K (eV)'].min()
        mean_barrier = summary_df['ΔG‡_10K (eV)'].mean()
        best_rate = summary_df['k_10K (s⁻¹)'].max()
        
        html_content += f"""
                <li><strong>Best Barrier (10K):</strong> {best_barrier:.3f} eV</li>
                <li><strong>Mean Barrier (10K):</strong> {mean_barrier:.3f} eV</li>
                <li><strong>Best Rate (10K):</strong> {best_rate:.2e} s⁻¹</li>
        """
    
    html_content += """
            </ul>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)

def generate_report(run_dir: str, top_n: int = 10, output_file: str = "report.html"):
    """Generate comprehensive report from WTr results."""
    
    print(f"Loading results from {run_dir}...")
    results = load_results(run_dir)
    
    if not results:
        print("No results found in run directory.")
        return
    
    print(f"Found {len(results)} result files")
    
    # Create summary table
    summary_df = create_summary_table(results, top_n)
    
    # Generate plots
    report_dir = Path(output_file).parent
    report_dir.mkdir(exist_ok=True)
    
    plot_files = []
    
    try:
        # Barrier distribution plot
        barrier_plot = report_dir / "barrier_distribution.png"
        plot_barrier_distribution(results, str(barrier_plot))
        plot_files.append(str(barrier_plot))
        
        # Descriptor correlations
        corr_plot = report_dir / "descriptor_correlations.png"
        plot_descriptor_correlations(results, str(corr_plot))
        plot_files.append(str(corr_plot))
        
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")
    
    # Save CSV summary
    csv_file = report_dir / "summary.csv"
    summary_df.to_csv(csv_file, index=False)
    print(f"Summary table saved to {csv_file}")
    
    # Generate HTML report
    generate_html_report(summary_df, run_dir, plot_files, output_file)
    print(f"HTML report generated: {output_file}")
    
    return summary_df
