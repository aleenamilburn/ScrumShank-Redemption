import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_excel("MarchData.xlsx", sheet_name="cleaned")

# Extract fiscal year
df["Fiscal.Year"] = df["Ordered.Date"].dt.year

# Aggregate by fiscal year, category, and vendor
agg_data = df.groupby(["Fiscal.Year", "PO.Category.Description", "Vendor.Name"])["Line.Total"].sum().reset_index()

# Convert Fiscal Year to datetime format
agg_data["Fiscal.Year"] = pd.to_datetime(agg_data["Fiscal.Year"], format="%Y")

# Check data structure
print(agg_data.head())
