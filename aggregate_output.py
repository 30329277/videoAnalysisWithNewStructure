import os
import glob
from datetime import datetime

def aggregate_txt_files():
    output_dir = "output"
    txt_files = glob.glob(os.path.join(output_dir, "*.txt"))
    if not txt_files:
        print("No .txt files found in the output directory.")
        return
    
    # Generate the filename based on the current date and time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"total_{timestamp}_output_.txt")

    with open(output_file, 'w') as outfile:
        for txt_file in txt_files:
            with open(txt_file, 'r') as infile:
                outfile.write(infile.read() + '\n')  # Add content from each file followed by a newline

    print(f"All files aggregated into {output_file}")
