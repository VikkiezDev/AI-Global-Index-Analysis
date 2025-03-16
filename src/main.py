import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("/data/AI_index_db.csv")

# Data Cleaning
# Checking for missing values
df = df.dropna()

# Checking for duplicates
df = df.drop_duplicates()

# Convert data types if necessary
df["Total score"] = df["Total score"].astype(float)

# Summary Statistics
print(df.describe())

# Correlation Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

# AI Score Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Total score"], bins=20, kde=True, color='blue')
plt.title("Distribution of AI Scores")
plt.xlabel("AI Score")
plt.ylabel("Count")
plt.show()

# AI Score by Income Group
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Income group", y="Total score", palette="Set2")
plt.title("AI Score by Income Group")
plt.xlabel("Income Group")
plt.ylabel("Total AI Score")
plt.show()

# AI Score by Political Regime
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Political regime", y="Total score", palette="Set3")
plt.title("AI Score by Political Regime")
plt.xlabel("Political Regime")
plt.ylabel("Total AI Score")
plt.xticks(rotation=15)
plt.show()

# AI Score by Region
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="Region", y="Total score", estimator=lambda x: x.mean(), palette="coolwarm")
plt.title("Average AI Score by Region")
plt.xlabel("Region")
plt.ylabel("Average AI Score")
plt.xticks(rotation=15)
plt.show()

# Categorizing countries as Developed or Emerging
df["Economic Status"] = df["Income group"].apply(lambda x: "Developed" if x == "High" else "Emerging")

# AI Score Comparison: Developed vs Emerging
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="Economic Status", y="Total score", palette="muted")
plt.title("AI Score Comparison: Developed vs. Emerging Economies")
plt.xlabel("Economic Status")
plt.ylabel("Total AI Score")
plt.show()

# Radar Chart: Developed vs Emerging AI Readiness
def radar_chart(df):
    import numpy as np
    
    developed_avg = df[df["Economic Status"] == "Developed"].mean(numeric_only=True)
    emerging_avg = df[df["Economic Status"] == "Emerging"].mean(numeric_only=True)
    
    indicators = ["Talent", "Infrastructure", "Operating Environment", "Research", "Development", "Government Strategy", "Commercial"]
    developed_values = [developed_avg[ind] for ind in indicators]
    emerging_values = [emerging_avg[ind] for ind in indicators]
    
    angles = np.linspace(0, 2 * np.pi, len(indicators), endpoint=False).tolist()
    developed_values += developed_values[:1]
    emerging_values += emerging_values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, developed_values, color="blue", alpha=0.25, label="Developed Economies")
    ax.fill(angles, emerging_values, color="red", alpha=0.25, label="Emerging Economies")
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(indicators, fontsize=10)
    ax.set_yticklabels([])
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    plt.title("AI Development Gaps: Developed vs. Emerging Economies")
    plt.show()

radar_chart(df)

# Additional Useful Plots
# 1. Government Strategy vs. Commercial AI Activity
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Government Strategy", y="Commercial", hue="Economic Status", palette="coolwarm")
plt.title("Government Strategy vs. Commercial AI Activity")
plt.xlabel("Government Strategy Score")
plt.ylabel("Commercial AI Activity Score")
plt.show()

# 2. AI Research vs. Development
plt.figure(figsize=(8, 5))
sns.regplot(data=df, x="Research", y="Development", scatter_kws={"alpha":0.6}, line_kws={"color":"red"})
plt.title("AI Research vs. AI Development")
plt.xlabel("AI Research Score")
plt.ylabel("AI Development Score")
plt.show()

# 3. Infrastructure vs. AI Score
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Infrastructure", y="Total score", hue="Economic Status", palette="Set1")
plt.title("Infrastructure vs. AI Score")
plt.xlabel("Infrastructure Score")
plt.ylabel("Total AI Score")
plt.show()
