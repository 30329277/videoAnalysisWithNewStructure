import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from utils import format_time

def analyze_and_chart():
    output_dir = "output"
    txt_files = glob.glob(os.path.join(output_dir, "*.txt"))

    for file in txt_files:
        print(f"Processing file: {file}")
        person_times = {}
        with open(file, 'r') as f:
            for line in f:
                if "person" in line and "detected from" in line:
                    parts = line.split("detected from")
                    count_str = parts[0].strip()
                    times_str = parts[1].strip()

                    count = int(count_str.split()[0])
                    start_time_str, end_time_str = times_str.split("to")
                    start_time = datetime.strptime(start_time_str.strip(), '%Y-%m-%d %H:%M:%S')
                    end_time = datetime.strptime(end_time_str.strip(), '%Y-%m-%d %H:%M:%S')
                    duration = (end_time - start_time).total_seconds()

                    person_times[count] = person_times.get(count, 0) + duration

        # Correct negative durations to 0
        for count, duration in person_times.items():
            if duration < 0:
                print(f"Warning: Negative duration detected for {count} person(s). Correcting to 0.")
                person_times[count] = 0

        print(f"Person times: {person_times}")  # Debugging output

        df = pd.DataFrame(list(person_times.items()), columns=['Number of People', 'Total Time (seconds)'])

        total_time = df['Total Time (seconds)'].sum()

        def format_autopct(pct):
            total_sec = int(round(pct * total_time / 100.0))
            time_str = format_time(total_sec)  # Use your format_time function
            return f'{pct:.1f}%\n({time_str})'

        # Set the figure size and adjust the layout
        fig, ax = plt.subplots(figsize=(6, 6))
        plt.title(f"Person Detection Distribution - {os.path.basename(file)}", pad=15)

        # Create pie chart with adjusted font size
        wedges, texts, autotexts = ax.pie(df['Total Time (seconds)'], labels=df['Number of People'],
                                          autopct=format_autopct, startangle=90,
                                          textprops={'fontsize': 9})
        
        # Set font size of labels inside the pie
        for autotext in autotexts:
            autotext.set_fontsize(8)

        # Move total time text closer to the pie chart
        plt.text(0.5, -0.1, f"Total Time: {format_time(total_time)}",
                 ha='center', va='center', transform=ax.transAxes, fontsize=10)

        # Save and close the figure with tighter layout
        plt.savefig(file.replace(".txt", ".png"), bbox_inches='tight', pad_inches=0.2)
        plt.close()

if __name__ == "__main__":
    print(f"Current working directory: {os.getcwd()}")
    output_dir = "output"
    txt_files = glob.glob(os.path.join(output_dir, "*.txt"))
    print(f"Looking for .txt files in: {output_dir}")
    print(f"Found .txt files: {txt_files}")
    analyze_and_chart()
