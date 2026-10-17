import requests
import pandas as pd

# Excel file
excel_path = "test_set1.xlsx"

# Read the 100 test sentences
df = pd.read_excel(excel_path)

print("Total test sentences:", len(df))

results = []

for index, row in df.iterrows():

    sentence = str(row["Sentence"])

    try:
        response = requests.post(
            "http://127.0.0.1:5000/process",
            json={"text": sentence}
        )

        data = response.json()

        original = sentence
        corrected = data.get("corrected_result", "")

        results.append({
            "No.": row["No."],
            "Original": original,
            "Expected Correction": row["Correction"],
            "SayarCheck Output": corrected
        })

        print(f"Test {index + 1}/100 completed")

    except Exception as e:
        print(f"Error on test {index + 1}: {e}")

# Save results
result_df = pd.DataFrame(results)

result_df.to_excel(
    "test_100_results.xlsx",
    index=False
)

print("\n================================")
print("100-SENTENCE TEST COMPLETED")
print("================================")
print("Results saved to: test_100_results.xlsx")