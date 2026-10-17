import pandas as pd

excel_path = "test_set1.xlsx"

df = pd.read_excel(excel_path)

pairs = []

for _, row in df.iterrows():

    error_value = row["Error"]
    correction_value = row["Correction"]

    if pd.isna(error_value) or pd.isna(correction_value):
        continue

    errors = [x.strip() for x in str(error_value).split(",")]
    corrections = [x.strip() for x in str(correction_value).split(",")]

    # Pair errors and corrections by position
    if len(errors) == len(corrections):
        for error, correction in zip(errors, corrections):

            pairs.append({
                "Sentence No.": row["No."],
                "Error": error,
                "Correction": correction,
                "Sentence": row["Sentence"]
            })

    else:
        print(
            f"WARNING: No. {row['No.']} has "
            f"{len(errors)} errors but {len(corrections)} corrections"
        )

result = pd.DataFrame(pairs)

# Remove exact duplicate pairs
result = result.drop_duplicates(
    subset=["Error", "Correction"]
).reset_index(drop=True)

result.to_excel(
    "individual_error_pairs.xlsx",
    index=False
)

print("\n================================")
print("INDIVIDUAL ERROR PAIRS CREATED")
print("================================")

print("Unique Error → Correction pairs:", len(result))

print("\nPairs:")
print(
    result[
        ["Sentence No.", "Error", "Correction"]
    ].to_string(index=False)
)

print("\nSaved as: individual_error_pairs.xlsx")