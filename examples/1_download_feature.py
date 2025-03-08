import quilt3 as q3
import requests
import os

# Create data directory if it doesn't exist
data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)
print(f"Saving files to {os.path.abspath(data_dir)}")

# Define output paths
profiles_output_path = os.path.join(data_dir, "profiles_wellpos_cc_var_mad_outlier_featselect_sphering_harmony.parquet")
compound_output_path = os.path.join(data_dir, "compound.csv.gz")

# Download profiles data from S3 bucket
b = q3.Bucket("s3://cellpainting-gallery")
# Download [[https://docs.quiltdata.com/api-reference/bucket#bucket.fetch]]
print("Downloading profiles data from S3 bucket...")
b.fetch("cpg0016-jump-assembled/source_all/workspace/profiles/jump-profiling-recipe_2024_0224e0f/ALL/profiles_wellpos_cc_var_mad_outlier_featselect_sphering_harmony/profiles_wellpos_cc_var_mad_outlier_featselect_sphering_harmony.parquet", profiles_output_path)

# Download compound.csv.gz from GitHub
github_url = "https://github.com/jump-cellpainting/datasets/raw/refs/heads/main/metadata/compound.csv.gz"

print(f"Downloading compound data from {github_url}...")
response = requests.get(github_url, stream=True)
response.raise_for_status()  # Raise an exception for HTTP errors

# Save the downloaded file
with open(compound_output_path, 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

print(f"Successfully downloaded compound data to {compound_output_path}")

# Verify both files exist
if os.path.exists(profiles_output_path) and os.path.exists(compound_output_path):
    print("All required files have been downloaded successfully.")
    print(f"Profiles file size: {os.path.getsize(profiles_output_path) / (1024*1024):.2f} MB")
    print(f"Compound file size: {os.path.getsize(compound_output_path) / (1024*1024):.2f} MB")

