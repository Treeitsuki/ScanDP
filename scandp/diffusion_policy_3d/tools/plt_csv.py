import os 
import pandas as pd
import seaborn as sns
import argparse
import matplotlib.pyplot as plt
from matplotlib import rcParams

def plot_coverage_from_csv(file_name):
    csv_path = os.path.join(os.getcwd(), "scandp/data/logs/csv", file_name)

    data = pd.read_csv(csv_path)
    steps = data['step']
    coverage = data['coverage']
    file_name = os.path.splitext(file_name)[0]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, coverage, color='tab:blue', label='Coverage', linewidth=2)
    plt.title(f'{file_name}', fontsize=20)
    plt.xlabel('Step', fontsize=20)
    plt.ylabel('Coverage', fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_all_csv_in_folder(target='bunny', y_axis='coverage', highlight_keywords=["spconv"], exclude_keywords=["dp3*_550", "idp3"]):
    if y_axis not in ['coverage', 'path_length']:
        raise ValueError("y_axis must be either 'coverage' or 'path_length'")
    
    if exclude_keywords is None:
        exclude_keywords = []

    sns.set_palette("pastel")

    folder_path = os.path.join(os.getcwd(), "scandp/data/logs/csv/", target)
    csv_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.csv') and not any(keyword in f for keyword in exclude_keywords)],
        reverse=True
    )  # Sort files in reverse alphabetical order

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.family'] = 'Times New Roman'

    plt.figure(figsize=(10, 8))
    for file_name in csv_files:
        csv_path = os.path.join(folder_path, file_name)
        data = pd.read_csv(csv_path)
        steps = data['step']
        y_values = data[y_axis]
        label = os.path.splitext(file_name)[0]

        # Check if the file name contains any of the highlight keywords
        if any(keyword in file_name for keyword in highlight_keywords):
            plt.plot(steps, y_values, label=label, linewidth=8, color="red", linestyle='-')
        else:
            plt.plot(steps, y_values, label=label, linewidth=8, linestyle='--')

    plt.xlabel('Steps', fontsize=55)
    if y_axis == 'coverage':
        plt.ylabel('Coverage [%]', fontsize=55)
    elif y_axis == 'path_length':
        plt.ylabel('Path Length [m]', fontsize=55)
    plt.xticks(fontsize=55)
    plt.yticks(fontsize=55)  
    plt.grid(True, linestyle='--', alpha=0.9)
    plt.legend(fontsize=30, loc='lower right')
    plt.tight_layout()
    
    # Save the plot as a PDF
    output_folder = os.path.join(os.getcwd(), "scandp/data/logs/plots", target)
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"{target}_{y_axis}.pdf")
    plt.savefig(output_file, format='pdf')
    print(f"Plot saved as {output_file}")
    plt.show()

def plot_all_csv_in_folder_path_length(target='bunny', y_axis='coverage', highlight_keywords=["spconv"], exclude_keywords= ["dp3*_550", "idp3"]):
    if y_axis not in ['coverage', 'path_length']:
        raise ValueError("y_axis must be either 'coverage' or 'path_length'")
    
    if exclude_keywords is None:
        exclude_keywords = []

    sns.set_palette("pastel")

    # folder_path = os.path.join(os.getcwd(), "scandp/data/logs/csv/", target)
    folder_path = "/home/cvl/cvl/ScanDP/scandp/data/logs/output/csv_[0, -1.5, 1]_1.5"
    csv_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.csv') and not any(keyword in f for keyword in exclude_keywords)],
        reverse=True
    )  # Sort files in reverse alphabetical order

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.family'] = 'Times New Roman'

    plt.figure(figsize=(10, 8))
    for file_name in csv_files:
        csv_path = os.path.join(folder_path, file_name)
        data = pd.read_csv(csv_path)
        steps = data['path_length']
        y_values = data[y_axis]
        label = os.path.splitext(file_name)[0]
        # Check if the file name contains any of the highlight keywords
        if any(keyword in file_name for keyword in highlight_keywords):
            line, = plt.plot(steps, y_values, label=label, linewidth=8, color="red", linestyle='-')
        else:
            line, = plt.plot(steps, y_values, label=label, linewidth=8, linestyle='--')
        
        # Plot a point for the last row of the CSV with the same color as the line
        plt.scatter(steps.iloc[-1], y_values.iloc[-1], color=line.get_color(), s=400, zorder=5)

    plt.xlabel('Path Length [m]', fontsize=55)
    if y_axis == 'coverage':
        plt.ylabel('Coverage [%]', fontsize=55)
    elif y_axis == 'path_length':
        plt.ylabel('Path Length [m]', fontsize=55)
    plt.xticks(fontsize=55)
    plt.yticks(fontsize=55)  
    plt.grid(True, linestyle='--', alpha=0.9)
    plt.legend(fontsize=30, loc='lower right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # plot_coverage_from_csv('spot_spconv_550.csv')
    parser = argparse.ArgumentParser(description="Plot CSV data.")
    parser.add_argument('--target', type=str, default='bunny', help='Target name')
    parser.add_argument('--y_axis', type=str, default='coverage', choices=['coverage', 'path_length'], help='Y-axis data to plot')
    args = parser.parse_args()

    # plot_all_csv_in_folder(target=args.target, y_axis=args.y_axis)
    plot_all_csv_in_folder_path_length(target=args.target, y_axis=args.y_axis)