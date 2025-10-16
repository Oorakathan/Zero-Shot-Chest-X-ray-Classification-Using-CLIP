"""
Data Analysis for NIH Chest X-ray Dataset (Data_Entry_2017.csv)
This script performs comprehensive exploratory data analysis on the medical imaging dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_data(file_path):
    """Load the dataset from CSV file"""
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully with {len(df)} records\n")
    return df

def basic_info(df):
    """Display basic information about the dataset"""
    print("="*80)
    print("BASIC DATASET INFORMATION")
    print("="*80)
    print(f"\nDataset Shape: {df.shape}")
    print(f"Number of Images: {df.shape[0]:,}")
    print(f"Number of Features: {df.shape[1]}")
    
    print("\nColumn Names and Types:")
    print(df.dtypes)
    
    print("\nFirst Few Rows:")
    print(df.head())
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    print("\nBasic Statistics:")
    print(df.describe())
    print("\n")

def analyze_findings(df):
    """Analyze disease findings in the dataset"""
    print("="*80)
    print("DISEASE FINDINGS ANALYSIS")
    print("="*80)
    
    # Split findings and count each disease
    all_findings = []
    for findings in df['Finding Labels']:
        diseases = findings.split('|')
        all_findings.extend(diseases)
    
    finding_counts = Counter(all_findings)
    
    print(f"\nTotal unique findings: {len(finding_counts)}")
    print("\nDisease Distribution:")
    for disease, count in finding_counts.most_common():
        percentage = (count / len(df)) * 100
        print(f"  {disease:25s}: {count:6,} ({percentage:5.2f}%)")
    
    # Create visualization
    plt.figure(figsize=(14, 6))
    diseases = [item[0] for item in finding_counts.most_common()]
    counts = [item[1] for item in finding_counts.most_common()]
    
    plt.bar(diseases, counts, color='steelblue', edgecolor='black')
    plt.xlabel('Disease/Finding', fontsize=12, fontweight='bold')
    plt.ylabel('Count', fontsize=12, fontweight='bold')
    plt.title('Distribution of Diseases/Findings in NIH Chest X-ray Dataset', 
              fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('disease_distribution.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: disease_distribution.png")
    
    return finding_counts

def analyze_patients(df):
    """Analyze patient demographics"""
    print("\n" + "="*80)
    print("PATIENT DEMOGRAPHICS")
    print("="*80)
    
    print(f"\nTotal Unique Patients: {df['Patient ID'].nunique():,}")
    print(f"Average Images per Patient: {len(df) / df['Patient ID'].nunique():.2f}")
    
    # Age analysis
    print("\nAge Statistics:")
    print(f"  Mean Age: {df['Patient Age'].mean():.1f} years")
    print(f"  Median Age: {df['Patient Age'].median():.1f} years")
    print(f"  Age Range: {df['Patient Age'].min()}-{df['Patient Age'].max()} years")
    print(f"  Std Dev: {df['Patient Age'].std():.2f} years")
    
    # Gender analysis
    print("\nGender Distribution:")
    gender_counts = df['Patient Gender'].value_counts()
    for gender, count in gender_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {gender}: {count:,} ({percentage:.2f}%)")
    
    # Create visualizations
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Age distribution
    axes[0].hist(df['Patient Age'], bins=30, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Age', fontweight='bold')
    axes[0].set_ylabel('Frequency', fontweight='bold')
    axes[0].set_title('Age Distribution', fontweight='bold')
    axes[0].axvline(df['Patient Age'].mean(), color='red', linestyle='--', 
                    label=f'Mean: {df["Patient Age"].mean():.1f}')
    axes[0].legend()
    
    # Gender distribution
    gender_counts.plot(kind='bar', ax=axes[1], color=['#ff9999', '#66b3ff'], 
                       edgecolor='black')
    axes[1].set_xlabel('Gender', fontweight='bold')
    axes[1].set_ylabel('Count', fontweight='bold')
    axes[1].set_title('Gender Distribution', fontweight='bold')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    
    # View Position
    view_counts = df['View Position'].value_counts()
    view_counts.plot(kind='bar', ax=axes[2], color='lightgreen', edgecolor='black')
    axes[2].set_xlabel('View Position', fontweight='bold')
    axes[2].set_ylabel('Count', fontweight='bold')
    axes[2].set_title('X-ray View Position Distribution', fontweight='bold')
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.savefig('patient_demographics.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: patient_demographics.png")

def analyze_comorbidities(df):
    """Analyze co-occurring diseases"""
    print("\n" + "="*80)
    print("CO-MORBIDITY ANALYSIS")
    print("="*80)
    
    # Count number of findings per image
    df['Finding Count'] = df['Finding Labels'].apply(lambda x: len(x.split('|')))
    
    print("\nNumber of Findings per Image:")
    finding_dist = df['Finding Count'].value_counts().sort_index()
    for count, freq in finding_dist.items():
        percentage = (freq / len(df)) * 100
        print(f"  {count} finding(s): {freq:,} images ({percentage:.2f}%)")
    
    print(f"\nAverage findings per image: {df['Finding Count'].mean():.2f}")
    
    # Most common disease combinations
    print("\nTop 20 Most Common Finding Combinations:")
    combo_counts = df['Finding Labels'].value_counts().head(20)
    for i, (combo, count) in enumerate(combo_counts.items(), 1):
        percentage = (count / len(df)) * 100
        print(f"  {i:2d}. {combo:50s}: {count:5,} ({percentage:5.2f}%)")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    finding_dist.plot(kind='bar', color='coral', edgecolor='black')
    plt.xlabel('Number of Findings per Image', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.title('Distribution of Number of Findings per Image', fontweight='bold')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('comorbidity_distribution.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: comorbidity_distribution.png")

def analyze_image_properties(df):
    """Analyze image dimensions and properties"""
    print("\n" + "="*80)
    print("IMAGE PROPERTIES ANALYSIS")
    print("="*80)
    
    # Extract width and height (they appear to be in the column names)
    # The format seems to be: OriginalImage[Width,Height]
    print("\nImage dimension analysis would require parsing the specific columns.")
    print("Based on the data structure, images have varying dimensions.")
    
    # View Position analysis
    print("\nView Position Distribution:")
    view_counts = df['View Position'].value_counts()
    for view, count in view_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {view}: {count:,} ({percentage:.2f}%)")

def generate_summary_report(df, finding_counts):
    """Generate a comprehensive summary report"""
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    print(f"""
Dataset Overview:
-----------------
• Total Images: {len(df):,}
• Unique Patients: {df['Patient ID'].nunique():,}
• Images per Patient: {len(df) / df['Patient ID'].nunique():.2f}
• Date Range (Follow-up): {df['Follow-up #'].min()} to {df['Follow-up #'].max()}

Patient Demographics:
---------------------
• Age Range: {df['Patient Age'].min()}-{df['Patient Age'].max()} years
• Mean Age: {df['Patient Age'].mean():.1f} years
• Gender: M={len(df[df['Patient Gender']=='M']):,}, F={len(df[df['Patient Gender']=='F']):,}

Disease Findings:
-----------------
• Unique Findings: {len(finding_counts)}
• Most Common: {finding_counts.most_common(1)[0][0]} ({finding_counts.most_common(1)[0][1]:,} cases)
• No Finding Cases: {finding_counts.get('No Finding', 0):,} ({(finding_counts.get('No Finding', 0)/len(df)*100):.2f}%)

Image Views:
------------
• PA (Posterior-Anterior): {len(df[df['View Position']=='PA']):,}
• AP (Anterior-Posterior): {len(df[df['View Position']=='AP']):,}
""")

def main():
    """Main function to run all analyses"""
    print("\n" + "="*80)
    print("NIH CHEST X-RAY DATASET - COMPREHENSIVE DATA ANALYSIS")
    print("="*80 + "\n")
    
    # Load data
    csv_path = 'Data_Entry_2017.csv'
    df = load_data(csv_path)
    
    # Run all analyses
    basic_info(df)
    finding_counts = analyze_findings(df)
    analyze_patients(df)
    analyze_comorbidities(df)
    analyze_image_properties(df)
    generate_summary_report(df, finding_counts)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated Files:")
    print("  1. disease_distribution.png")
    print("  2. patient_demographics.png")
    print("  3. comorbidity_distribution.png")
    print("\n")

if __name__ == "__main__":
    main()
