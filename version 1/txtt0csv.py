import pandas as pd

# Standard NSL-KDD Column Names
col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label", "difficulty"]

def convert_to_csv(input_file, output_file):
    print(f"Reading {input_file}...")
    # Read txt file (it acts like a CSV without headers)
    df = pd.read_csv(input_file, names=col_names)
    
    # Save to actual CSV format
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

# Usage
convert_to_csv('KDDTrain+.txt', 'KDDTrain+.csv')
convert_to_csv('KDDTest+.txt', 'KDDTest+.csv')