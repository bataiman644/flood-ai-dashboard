import geopandas as gpd

# Load shapefile
df = gpd.read_file("wlf_nhr_fl_dfomasterlist_20190418.shp")

print(df.head())   # see data
print(df.columns)  # see columns

# Remove map geometry (important)
df = df.drop(columns="geometry")

# Save as CSV
df.to_csv("real_flood_data.csv", index=False)

print("✅ Converted successfully!")