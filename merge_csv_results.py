import os
import sys
import pandas as pd


def summarize_csv_files(directory):
    # List all CSV files in the directory
    # csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    csv_files = sorted([f for f in os.listdir(directory) if f.endswith('.csv')])

    if not csv_files:
        print("No CSV files found in the directory.")
        return

    summary_data = []
    column_headers = None

    for file in csv_files:
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)

        # Extract the scene name from the file name
        scene_name = file.split("_")[0]

        # Get the average (last row)
        average_row = df.iloc[-1].copy()
        average_row.iloc[0] = scene_name  # Set the first column as the scene name

        # Store column headers if not already stored
        if column_headers is None:
            column_headers = df.columns

        summary_data.append(average_row)

    # Create a summary DataFrame
    summary_df = pd.DataFrame(summary_data, columns=column_headers)

    # Compute the overall average and append it to the summary DataFrame
    overall_average = summary_df.iloc[:, 1:].astype(float).mean(axis=0)
    overall_average = pd.Series([summary_df.columns[0]] + overall_average.tolist(), index=summary_df.columns)
    overall_average[0] = "Overall_Average"

    summary_df = pd.concat([summary_df, pd.DataFrame([overall_average])], ignore_index=True)

    # Write the summary DataFrame to a new CSV file
    summary_csv_path = os.path.join(directory, "summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    print(f"Summary CSV file created at: {summary_csv_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python summarize_csv.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    summarize_csv_files(directory)
