import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import os
from video_analysis import detect_people_in_video, format_time, model, transform
import utils
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import cv2
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

root = tk.Tk()
root.title("视频分析工具")

def browse_file():
    filename = filedialog.askopenfilename(initialdir=os.getcwd(), title="选择视频文件",
                                          filetypes=(("Video Files", "*.mp4;*.avi;*.mov;*.mts"), ("All Files", "*.*")))
    file_path_var.set(filename)

def analyze_video():
    video_path = file_path_var.get()
    if not video_path:
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, "请选择视频文件")
        return

    try:
        detected_times, active_times, fps = detect_people_in_video(video_path, model, transform, int(config['Model']['target_label_id']))
        if detected_times == -1:  # Check for MTS conversion failure
            result_text.delete("1.0", tk.END)
            result_text.insert(tk.END, "MTS文件转换失败。")
            return

        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, f"检测到人物时间: {format_time(detected_times)}\n")
        result_text.insert(tk.END, f"活动时间: {format_time(active_times)}\n")

        generate_time_ranges_and_plot(video_path, fps)

    except Exception as e:
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, f"发生错误: {e}")



def generate_time_ranges_and_plot(video_path, fps):
    # Replace this with your actual person detection results
    # Example data (replace with your actual data) - suitable for a pie chart
    labels = ['0 People', '1 Person', '2 People', '3 People']
    sizes = [20, 30, 40, 10] # Example: percentage of time with 0, 1, 2, or 3 people


    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_title("Distribution of People Detected")

    # Embed the plot in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)



file_path_var = tk.StringVar()
file_path_label = ttk.Label(root, text="视频路径:")
file_path_label.grid(row=0, column=0, sticky=tk.W)
file_path_entry = ttk.Entry(root, textvariable=file_path_var, width=50)
file_path_entry.grid(row=0, column=1, padx=5)
browse_button = ttk.Button(root, text="浏览", command=browse_file)
browse_button.grid(row=0, column=2)

analyze_button = ttk.Button(root, text="分析视频", command=analyze_video)
analyze_button.grid(row=1, column=1, pady=10)

result_text = tk.Text(root, wrap=tk.WORD, height=10)
result_text.grid(row=2, column=0, columnspan=3, padx=5, pady=5)

# Frame for the Matplotlib plot
plot_frame = tk.Frame(root)
plot_frame.grid(row=3, column=0, columnspan=3, padx=5, pady=5)



root.mainloop()