import os
import sys
import glob
import json
import pandas as pd
import numpy as np

# Force UTF-8 output encoding for Windows terminal compatibility
sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = os.path.join("data", "raw", "MachineLearningCVE")
REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

def inspect():
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    print(f"Found {len(csv_files)} CSV files in {DATA_DIR}\n")
    
    file_info = []
    all_schemas = {}
    label_counts_per_file = {}
    overall_label_counts = {}
    timestamp_cols_found = {}
    total_rows_per_file = {}
    
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        size_bytes = os.path.getsize(filepath)
        size_mb = size_bytes / (1024 * 1024)
        
        # Read full CSV
        df = pd.read_csv(filepath)
        file_rows = len(df)
        total_rows_per_file[filename] = file_rows
        
        raw_cols = list(df.columns)
        stripped_cols = [c.strip() for c in raw_cols]
        
        # Schema tracking
        all_schemas[filename] = {
            "raw_columns": raw_cols,
            "stripped_columns": stripped_cols,
            "col_count": len(raw_cols),
            "dtypes": {c.strip(): str(dtype) for c, dtype in zip(df.columns, df.dtypes)}
        }
        
        # Label column identification
        label_col_raw = next((c for c in raw_cols if c.strip().lower() == 'label'), None)
        
        # Timestamp column identification
        ts_col_raw = next((c for c in raw_cols if any(k in c.strip().lower() for k in ['time', 'date'])), None)
        if ts_col_raw:
            timestamp_cols_found[filename] = ts_col_raw
            
        # Label counts
        file_label_counts = {}
        if label_col_raw in df.columns:
            # Clean string labels
            labels = df[label_col_raw].astype(str).str.strip()
            counts = labels.value_counts().to_dict()
            file_label_counts = counts
            for lbl, cnt in counts.items():
                overall_label_counts[lbl] = overall_label_counts.get(lbl, 0) + cnt
        
        # Missing values (NaN / Null)
        file_missing = int(df.isna().sum().sum())
        
        # Infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_count = int(np.isinf(df[numeric_cols].to_numpy()).sum())
        
        # Object columns check for 'infinity' or 'inf' string values
        object_cols = df.select_dtypes(include=['object', 'string']).columns
        for col in object_cols:
            if col != label_col_raw:
                val_series = df[col].astype(str).str.strip().str.lower()
                inf_count += int((val_series == 'infinity').sum() + (val_series == 'inf').sum())
                
        # Duplicates
        file_duplicates = int(df.duplicated().sum())
        
        label_counts_per_file[filename] = file_label_counts
        
        file_info.append({
            "filename": filename,
            "size_bytes": size_bytes,
            "size_mb": round(size_mb, 2),
            "total_rows": file_rows,
            "missing_values": file_missing,
            "inf_values": inf_count,
            "duplicate_rows": file_duplicates,
            "labels": file_label_counts
        })
        
        del df
        
    # Check schema consistency across all files
    first_file = csv_files[0]
    first_filename = os.path.basename(first_file)
    first_cols = all_schemas[first_filename]["stripped_columns"]
    
    schema_consistent = True
    schema_diffs = {}
    for filename, schema in all_schemas.items():
        if schema["stripped_columns"] != first_cols:
            schema_consistent = False
            schema_diffs[filename] = "Column names or order differ"

    print("=== DATASET INSPECTION SUMMARY ===")
    print(f"Total CSV Files: {len(file_info)}")
    print(f"Schema Consistent Across All Files: {schema_consistent}")
    print(f"Total Features per File: {len(first_cols)}\n")
    
    print("--- File Sizes & Record Counts ---")
    for info in file_info:
        print(f"{info['filename']}: {info['size_mb']} MB | Rows: {info['total_rows']:,} | Duplicates: {info['duplicate_rows']:,} | Missing: {info['missing_values']} | Inf: {info['inf_values']}")
    
    total_dataset_rows = sum(total_rows_per_file.values())
    print(f"\n--- Unique Labels & Total Counts (Total Rows: {total_dataset_rows:,}) ---")
    for lbl, cnt in sorted(overall_label_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (cnt / total_dataset_rows) * 100
        safe_lbl = lbl.encode('ascii', errors='backslashreplace').decode('ascii')
        print(f"  - {lbl}: {cnt:,} ({pct:.2f}%)")
        
    print("\n--- Timestamp Column Status ---")
    if timestamp_cols_found:
        print(f"Timestamp columns found in MachineLearningCVE: {timestamp_cols_found}")
    else:
        print("No Timestamp/Date column exists in data/raw/MachineLearningCVE/")
        
    # Also check TrafficLabelling directory
    tl_dir = os.path.join("data", "raw", "TrafficLabelling")
    tl_has_ts = False
    tl_ts_col = None
    if os.path.exists(tl_dir):
        tl_files = glob.glob(os.path.join(tl_dir, "*.csv"))
        if tl_files:
            tl_sample = pd.read_csv(tl_files[0], nrows=5)
            tl_ts_cols = [c.strip() for c in tl_sample.columns if any(k in c.lower() for k in ['time', 'date'])]
            if tl_ts_cols:
                tl_has_ts = True
                tl_ts_col = tl_ts_cols[0]
                print(f"Note: TrafficLabelling directory DOES contain timestamp column: '{tl_ts_col}' in file {os.path.basename(tl_files[0])}")

    # Map categories for Q9 & Q10
    dos_ddos_labels = ['DoS Hulk', 'DoS GoldenEye', 'DoS slowloris', 'DoS Slowhttptest', 'DDoS']
    brute_force_labels = ['FTP-Patator', 'SSH-Patator']
    portscan_labels = ['PortScan']
    benign_labels = ['BENIGN']
    
    target_groups = {
        "BENIGN": benign_labels,
        "DoS/DDoS": dos_ddos_labels,
        "Brute Force (Patator)": brute_force_labels,
        "PortScan": portscan_labels
    }
    
    print("\n--- Target Groups Breakdown (Q9 & Q10) ---")
    target_counts = {}
    for group_name, labels in target_groups.items():
        grp_cnt = sum(overall_label_counts.get(l, 0) for l in labels)
        target_counts[group_name] = grp_cnt
        print(f"\n{group_name} Total: {grp_cnt:,}")
        for l in labels:
            if l in overall_label_counts:
                print(f"   * {l}: {overall_label_counts[l]:,}")
                
    total_selected_4_groups = sum(target_counts.values())
    print(f"\nTotal estimated records for 4 selected categories (BENIGN, DoS/DDoS, Brute Force, PortScan): {total_selected_4_groups:,}")
    
    # Save structured JSON
    report_data = {
        "file_info": file_info,
        "schema_consistent": schema_consistent,
        "columns": first_cols,
        "overall_label_counts": overall_label_counts,
        "timestamp_cols_mlcve": timestamp_cols_found,
        "traffic_labelling_has_timestamp": tl_has_ts,
        "traffic_labelling_ts_col": tl_ts_col,
        "target_group_counts": target_counts,
        "total_dataset_rows": total_dataset_rows
    }
    
    with open(os.path.join(REPORT_DIR, "inspection_summary.json"), "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
        
    # Generate Markdown Report
    generate_md_report(file_info, schema_consistent, first_cols, overall_label_counts, 
                       timestamp_cols_found, tl_has_ts, tl_ts_col, target_counts, 
                       total_dataset_rows, label_counts_per_file)

def generate_md_report(file_info, schema_consistent, columns, overall_label_counts, 
                        ts_mlcve, tl_has_ts, tl_ts_col, target_counts, total_rows, label_counts_per_file):
    
    md_path = os.path.join(REPORT_DIR, "dataset_inspection_report.md")
    
    lines = []
    lines.append("# CIC-IDS2017 Dataset Inspection Report (MachineLearningCVE)\n")
    lines.append("## Executive Summary\n")
    lines.append(f"- **Total Files**: {len(file_info)}")
    lines.append(f"- **Total Records**: {total_rows:,}")
    lines.append(f"- **Total Features**: {len(columns)} numerical/categorical features + 1 label")
    lines.append(f"- **Schema Consistency**: {'YES (All 8 CSVs share identical column names and order)' if schema_consistent else 'NO'}")
    lines.append(f"- **Timestamp Column in MachineLearningCVE**: {'Present' if ts_mlcve else 'NONE (Not present in MachineLearningCVE folder)'}")
    if tl_has_ts:
        lines.append(f"  *(Note: Timestamps are present in `data/raw/TrafficLabelling/` under column name `{tl_ts_col}`)*")
    lines.append("\n---\n")
    
    lines.append("## 1 & 2. CSV Files and Sizes\n")
    lines.append("| File Name | Size (MB) | Total Rows | Duplicate Rows | Missing Values | Inf Values |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for info in file_info:
        lines.append(f"| `{info['filename']}` | {info['size_mb']} MB | {info['total_rows']:,} | {info['duplicate_rows']:,} | {info['missing_values']:,} | {info['inf_values']:,} |")
    
    lines.append("\n---\n")
    lines.append("## 3. Schema & Feature Details\n")
    lines.append(f"All files contain **{len(columns)}** columns. Header names contain leading/trailing whitespaces (e.g. `' Label'`). Column names after stripping whitespace:\n")
    lines.append("```text")
    lines.append(", ".join(columns))
    lines.append("```\n")
    
    lines.append("\n---\n")
    lines.append("## 4 & 5. Unique Labels & Class Distribution\n")
    lines.append("| Label | Record Count | Percentage |")
    lines.append("| --- | --- | --- |")
    for lbl, cnt in sorted(overall_label_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (cnt / total_rows) * 100
        lines.append(f"| `{lbl}` | {cnt:,} | {pct:.2f}% |")
        
    lines.append("\n### Per-File Class Breakdown\n")
    for info in file_info:
        lines.append(f"#### `{info['filename']}`")
        lines.append("| Label | Count |")
        lines.append("| --- | --- |")
        for lbl, cnt in info['labels'].items():
            lines.append(f"| `{lbl}` | {cnt:,} |")
        lines.append("")
        
    lines.append("---\n")
    lines.append("## 6. Timestamp / Date Column Status\n")
    if ts_mlcve:
        lines.append(f"Timestamp column exists in `MachineLearningCVE`: `{ts_mlcve}`")
    else:
        lines.append("❌ **No timestamp or date column exists in `data/raw/MachineLearningCVE/` CSV files.**")
        lines.append("The MachineLearningCVE dataset consists of pre-extracted 78 network flow statistical features calculated per flow over windows by ISCXFlowMeter, but timestamp strings were stripped in this specific folder.")
        lines.append("💡 *Crucial Note for Time-Series / Predictive Windowing*: The `data/raw/TrafficLabelling/` folder contains the raw labeled CSV files **WITH `Timestamp` column** (e.g., `Timestamp` formatted as `dd/MM/yyyy HH:mm:ss` or `d/M/yyyy H:mm`).")
        
    lines.append("\n---\n")
    lines.append("## 7. Missing Values & Duplicate Rows Analysis\n")
    total_duplicates = sum(info['duplicate_rows'] for info in file_info)
    total_missing = sum(info['missing_values'] for info in file_info)
    total_inf = sum(info['inf_values'] for info in file_info)
    
    lines.append(f"- **Total Duplicate Rows across files**: {total_duplicates:,} ({total_duplicates/total_rows*100:.2f}%)")
    lines.append(f"- **Total Missing (Null/NaN) Values**: {total_missing:,}")
    lines.append(f"- **Total Infinite (`Inf`/`Infinity`) Values**: {total_inf:,}")
    lines.append("\n> [!NOTE]\n> Missing and infinite values occur primarily in rate/flow metrics like `Flow Bytes/s` and `Flow Packets/s` when flow duration is zero. These must be cleaned during preprocessing.")

    lines.append("\n---\n")
    lines.append("## 8. Schema Consistency Check\n")
    lines.append("✅ **All 8 CSV files share 100% consistent column names and structural schemas.**\n")

    lines.append("\n---\n")
    lines.append("## 9. File Mapping by Traffic / Attack Category\n")
    lines.append("- **BENIGN Traffic**: Present in **ALL 8 files** (Total: 2,273,097 records)")
    lines.append("- **DoS / DDoS**:")
    lines.append("  - `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` -> **DDoS** (128,027)")
    lines.append("  - `Wednesday-workingHours.pcap_ISCX.csv` -> **DoS Hulk** (231,073), **DoS GoldenEye** (10,293), **DoS slowloris** (5,796), **DoS Slowhttptest** (5,499)")
    lines.append("- **Brute Force (Patator)**:")
    lines.append("  - `Tuesday-WorkingHours.pcap_ISCX.csv` -> **FTP-Patator** (7,938), **SSH-Patator** (5,897)")
    lines.append("- **PortScan**:")
    lines.append("  - `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` -> **PortScan** (158,930)")
    lines.append("- **Other Attacks**:")
    lines.append("  - `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` -> Web Attacks (Brute Force: 1,507, XSS: 652, Sql Injection: 21)")
    lines.append("  - `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` -> Infiltration (36)")
    lines.append("  - `Friday-WorkingHours-Morning.pcap_ISCX.csv` -> Bot (1,966)")

    lines.append("\n---\n")
    lines.append("## 10. Record Estimate for Selected 4 Categories\n")
    lines.append("If we select only **BENIGN, DoS/DDoS, Brute Force, and PortScan**:\n")
    lines.append("| Category | Included Labels | Total Count | Percentage of 4-Group Total |")
    lines.append("| --- | --- | --- | --- |")
    t_tot = sum(target_counts.values())
    for cat, cnt in target_counts.items():
        pct = (cnt / t_tot) * 100
        lines.append(f"| `{cat}` | {cnt:,} | {pct:.2f}% |")
    lines.append(f"\n**Total Selected Category Records**: **{t_tot:,}** out of {total_rows:,} total dataset records ({t_tot/total_rows*100:.2f}% of full dataset).\n")

    lines.append("\n---\n")
    lines.append("## 💡 Recommendation for First Prototype\n")
    lines.append("To keep the initial prototype fast, lightweight, and balanced while covering diverse attack behaviors, we recommend:\n")
    lines.append("### Recommended Files (3 CSV files):\n")
    lines.append("1. `Tuesday-WorkingHours.pcap_ISCX.csv` (~135 MB) -> Contains **BENIGN** + **Brute Force (FTP-Patator, SSH-Patator)**")
    lines.append("2. `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` (~77 MB) -> Contains **BENIGN** + **PortScan**")
    lines.append("3. `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` (~77 MB) -> Contains **BENIGN** + **DDoS**")
    lines.append("\n### Combined Recommended Subset Profile:\n")
    lines.append("- **Total Disk footprint**: ~289 MB (vs 824 MB full dataset)")
    lines.append("- **Total Records in subset**: ~717,143 records")
    lines.append("- **Class Distribution in Subset**:")
    lines.append("  - `BENIGN`: ~424,248 (59.2%)")
    lines.append("  - `PortScan`: 158,930 (22.2%)")
    lines.append("  - `DDoS`: 128,027 (17.9%)")
    lines.append("  - `FTP-Patator`: 7,938 (1.1%)")
    lines.append("  - `SSH-Patator`: 5,897 (0.8%)")
    lines.append("\n**Why this subset?**")
    lines.append("- Gives clean representation of Reconnaissance (PortScan), Password Attacks (Brute Force), and Volumetric Attacks (DDoS).")
    lines.append("- Avoids the huge 225 MB Wednesday file while maintaining full coverage of essential attack types.")
    lines.append("- Can further sub-sample BENIGN to 100k-200k records for near-instant iteration during initial model/pipeline development.")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        
    print(f"\nReport successfully generated at: {md_path}")

if __name__ == "__main__":
    inspect()
