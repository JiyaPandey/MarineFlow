# MarineFlow Data Preprocessing - Phase 2
import pandas as pd
import numpy as np

print("🚢 MarineFlow Data Preprocessing Started...")
print("="*50)

# Load dataset
print("Loading dataset...")
df = pd.read_csv('marineflow_demurrage_synth.csv')

print(f"✅ Dataset loaded successfully!")
print(f"📊 Shape: {df.shape}")
print(f"📋 Columns: {len(df.columns)} features")

print("\n📋 Column names:")
print(df.columns.tolist())

print("\n📊 Dataset Info:")
print(f"   • Total records: {len(df):,}")
print(f"   • Total features: {len(df.columns)}")

print("\n🎯 Target Variable Analysis:")
print("Demurrage Flag Distribution:")
print(df['demurrage_flag'].value_counts())

print("\n✅ BASIC ANALYSIS COMPLETE!")