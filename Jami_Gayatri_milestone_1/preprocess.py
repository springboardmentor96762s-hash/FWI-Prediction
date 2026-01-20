import pandas as pd

blocks = []
current_region = None

with open("data.csv", "r") as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    line = lines[i].strip()
    if "Region Dataset" in line:
        # Extract region name
        current_region = line.replace(" Region Dataset", "").strip()
        i += 1  # Next line is header
        header = lines[i].strip().split(",")  # adjust if space-separated
        i += 1
        # Collect rows until next region line or end
        rows = []
        while i < len(lines) and "Region Dataset" not in lines[i]:
            if lines[i].strip():  # skip empty lines
                rows.append(lines[i].strip().split(","))
            i += 1
        df_block = pd.DataFrame(rows, columns=header)
        df_block["Region"] = current_region
        blocks.append(df_block)
    else:
        i += 1

# Combine all regions
df = pd.concat(blocks, ignore_index=True)

# Remove unnamed columns
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# Handle day/month/year
for col in ["day", "month", "year"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df[col] = df[col].astype("category")

# Convert other object columns to numeric if possible
for col in df.columns:
    if df[col].dtype == "object" and col not in ["Region", "day", "month", "year"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# Save cleaned CSV
df.to_csv("clean_data.csv", index=False)

print("✅ Preprocessing complete. Region kept, day/month/year categorical with 0, no NaNs.")
