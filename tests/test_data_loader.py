# Test Excel loading
excel_path = PROJECT_ROOT / "data" / "raw" / "test_data.xlsx"

df.to_excel(excel_path, index=False)

excel_df = load_data(excel_path)

print("\nExcel test:")
print(excel_df)

print("\nExcel loaded successfully!")