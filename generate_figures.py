# generate_figures.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid", palette="muted")
os.makedirs("./figures", exist_ok=True)

def generate_fig1():
    """Fig 1: Accuracy Drop vs. Speedup"""
    df = pd.read_csv("./results/main_evaluation.csv")
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x="Speedup", y="Accuracy Drop (%)", hue="Method", marker="o", linewidth=2.5)
    plt.title("Fig 1: Accuracy Drop vs. Speedup (ResNet-50)", fontsize=14, fontweight='bold')
    plt.ylabel("Accuracy Drop (%)", fontsize=12)
    plt.xlabel("Target Speedup", fontsize=12)
    plt.tight_layout()
    plt.savefig("./figures/Fig1_Accuracy_vs_Speedup.png", dpi=300)
    plt.close()

def generate_fig2():
    """Fig 2: Pruning Time Comparison"""
    df = pd.read_csv("./results/main_evaluation.csv")
    df_2x = df[df["Speedup"] == "2x"]
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df_2x, x="Method", y="Pruning Time (s)", palette="viridis")
    plt.yscale("log")
    plt.title("Fig 2: Pruning Time Comparison (Log Scale)", fontsize=14, fontweight='bold')
    plt.ylabel("Pruning Time (Seconds, Log Scale)", fontsize=12)
    plt.tight_layout()
    plt.savefig("./figures/Fig2_Pruning_Time.png", dpi=300)
    plt.close()

def generate_fig3():
    """Fig 3: Layer-wise Pruning Distribution"""
    df = pd.read_csv("./results/ablation_layerwise.csv")
    df_melt = df.melt(id_vars=["Layer"], var_name="Method", value_name="Pruned (%)")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melt, x="Layer", y="Pruned (%)", hue="Method", palette="Set2")
    plt.xticks(rotation=45, ha='right')
    plt.title("Fig 3: Layer-wise Pruning Distribution (Structural Preservation)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("./figures/Fig3_Layerwise_Distribution.png", dpi=300)
    plt.close()

def generate_fig5():
    """Fig 5: Ablation - Centrality Measures"""
    df = pd.read_csv("./results/ablation_centrality.csv")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="Time (s)", y="Accuracy Drop (%)", hue="Centrality", s=200, palette="deep")
    
    for i in range(df.shape[0]):
        plt.text(df["Time (s)"][i] + 0.05, df["Accuracy Drop (%)"][i], df["Centrality"][i], fontsize=10)
        
    plt.title("Fig 5: Centrality Ablation (Accuracy vs Time)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("./figures/Fig5_Centrality_Ablation.png", dpi=300)
    plt.close()

def generate_fig6():
    """Fig 6: Ablation - Sampling Ratio vs Variance"""
    df = pd.read_csv("./results/ablation_sampling.csv")
    fig, ax1 = plt.subplots(figsize=(8, 6))

    color = 'tab:red'
    ax1.set_xlabel('Sampling Ratio')
    ax1.set_ylabel('Accuracy Drop (%)', color=color)
    ax1.plot(df["Sample Ratio"], df["Accuracy Drop (%)"], color=color, marker='o', linewidth=2.5)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Empirical Variance', color=color)  
    ax2.plot(df["Sample Ratio"], df["Variance"], color=color, marker='s', linestyle='--', linewidth=2.5)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  
    plt.title("Fig 6: Importance Sampling Ratio vs Accuracy/Variance", fontsize=14, fontweight='bold')
    plt.savefig("./figures/Fig6_Sampling_Variance.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Generating figures...")
    generate_fig1()
    generate_fig2()
    generate_fig3()
    generate_fig5()
    generate_fig6()
    print("Figures successfully generated in ./figures/")
