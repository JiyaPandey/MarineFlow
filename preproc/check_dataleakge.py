import pandas as pd

def check_leakage(train_path, val_path, test_path, id_col='voyage_id_target_encoded', time_col='depart_ts'):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    train_ids = set(train_df[id_col])
    val_ids = set(val_df[id_col])
    test_ids = set(test_df[id_col])

    print(f"Train size: {len(train_ids)}, Validation size: {len(val_ids)}, Test size: {len(test_ids)}")

    train_val_overlap = train_ids.intersection(val_ids)
    train_test_overlap = train_ids.intersection(test_ids)
    val_test_overlap = val_ids.intersection(test_ids)

    if train_val_overlap:
        print(f"Data leakage detected! Overlap between train and validation: {train_val_overlap}")
    else:
        print("No train-validation ID overlap detected.")

    if train_test_overlap:
        print(f"Data leakage detected! Overlap between train and test: {train_test_overlap}")
    else:
        print("No train-test ID overlap detected.")

    if val_test_overlap:
        print(f"Data leakage detected! Overlap between validation and test: {val_test_overlap}")
    else:
        print("No validation-test ID overlap detected.")

    train_time_max = pd.to_datetime(train_df[time_col]).max()
    val_time_min = pd.to_datetime(val_df[time_col]).min()
    val_time_max = pd.to_datetime(val_df[time_col]).max()
    test_time_min = pd.to_datetime(test_df[time_col]).min()

    if train_time_max >= val_time_min:
        print("Warning: Train and validation time overlap detected.")
    else:
        print("No train-validation time overlap detected.")

    if val_time_max >= test_time_min:
        print("Warning: Validation and test time overlap detected.")
    else:
        print("No validation-test time overlap detected.")


if __name__ == "__main__":
    # Fill in your real CSV file paths here
    train_csv_path = 'csvs\marineflow_train.csv'
    val_csv_path = 'csvs\marineflow_validation.csv'
    test_csv_path = 'csvs\marineflow_test.csv'

    check_leakage(train_csv_path, val_csv_path, test_csv_path)
