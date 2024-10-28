import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import os
import cv2  # Import cv2 for video duration calculation
from video_analysis import detect_people_in_video, format_time, model, transform, COCO_INSTANCE_CATEGORY_NAMES, device
import utils
import chart_generator  # 导入 add_chart 模块
import aggregate_output  # Import the new aggregation module

root = tk.Tk()
root.title("视频分析工具")

# --- Variable declarations ---
directory_path = tk.StringVar()
video_files = []
video_checkboxes = []
result_text = tk.StringVar()
mass_result = tk.StringVar()

# --- Functions ---
def browse_directory():
    directory_path.set(filedialog.askdirectory())
    update_video_list()

def update_video_list():
    global video_files, video_checkboxes
    video_files = [f for f in os.listdir(directory_path.get()) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mts'))]
    for widget in root.winfo_children():
        if isinstance(widget, ttk.Frame) and widget.winfo_name() == "video_list":
            widget.destroy()

    frame_video_list = ttk.Frame(root, name="video_list")
    frame_video_list.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

    video_checkboxes = []
    for i, file in enumerate(video_files):
        var = tk.BooleanVar()
        checkbox = ttk.Checkbutton(frame_video_list, text=file, variable=var)
        checkbox.grid(row=i, column=0, sticky="w")
        result_label = ttk.Label(frame_video_list, text="")
        result_label.grid(row=i, column=1, sticky="w")
        video_checkboxes.append((var, file, result_label))


def analyze_video(video_path, result_label):
    try:
        detected_times, active_times, fps = detect_people_in_video(video_path, model, transform, 1)  # 1 is the label ID for 'person'

        # 获取视频时长
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        video_duration = total_frames / fps if fps > 0 else 0  # Handle cases with zero fps
        video.release()
        video_duration_formatted = format_time(video_duration)

        if detected_times == -1:
            result_label.config(text="视频处理失败")
            return

        result_text.set(f"累计检测到人员时间：{format_time(detected_times)}\n视频总时长：{video_duration_formatted}")
        result_label.config(text=f"检测到人员时间：{format_time(detected_times)}, 视频时长：{video_duration_formatted}")
    except Exception as e:
        result_label.config(text=f"错误：{e}")


def select_deselect_all(select):
    for var, _, _ in video_checkboxes:
        var.set(select)

def add_chart():
    aggregate_output.aggregate_txt_files()  # Aggregate all .txt files
    chart_generator.analyze_and_chart()  # 调用 chart_generator 中的 analyze_and_chart 函数

# --- GUI setup ---
frame_directory = ttk.Frame(root)
frame_directory.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

directory_label = ttk.Label(frame_directory, text="选择视频目录：")
directory_label.grid(row=0, column=0, padx=(0, 10))

directory_entry = ttk.Entry(frame_directory, textvariable=directory_path, width=40)
directory_entry.grid(row=0, column=1, padx=(0, 10))

browse_button = ttk.Button(frame_directory, text="浏览", command=browse_directory)
browse_button.grid(row=0, column=2)


frame_result = ttk.Frame(root)
frame_result.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

result_label = ttk.Label(frame_result, textvariable=result_text)
result_label.grid(row=0, column=0, sticky="w")

# --- Part 3: Mass analysis ---
frame_mass_analysis = ttk.Frame(root)
frame_mass_analysis.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

select_all_button = ttk.Button(frame_mass_analysis, text="全选", command=lambda: select_deselect_all(True))
select_all_button.grid(row=0, column=0, padx=(0, 10))

deselect_all_button = ttk.Button(frame_mass_analysis, text="全不选", command=lambda: select_deselect_all(False))
deselect_all_button.grid(row=0, column=1, padx=(0, 10))

analyze_all_button = ttk.Button(frame_mass_analysis, text="分析所有选定的视频", command=lambda: analyze_all_videos())
analyze_all_button.grid(row=0, column=2, padx=(0, 10))

# 新增“增加饼图”按钮
add_chart_button = ttk.Button(frame_mass_analysis, text="增加饼图", command=add_chart)
add_chart_button.grid(row=0, column=3, padx=(0, 10))

mass_result_label = ttk.Label(frame_mass_analysis, textvariable=mass_result)
mass_result_label.grid(row=0, column=4)

def analyze_all_videos():
    selected_videos_data = [
        (os.path.join(directory_path.get(), file), result_label)
        for var, file, result_label in video_checkboxes
        if var.get() and os.path.isfile(os.path.join(directory_path.get(), file))
    ]
    if not selected_videos_data:
        mass_result.set("没有选择任何视频。")
        return

    mass_result.set("正在分析所有选定的视频...")
    cumulative_detected_time = 0
    total_video_duration = 0

    for video_path, result_label in selected_videos_data:
        try:
            # Get video duration using cv2
            video = cv2.VideoCapture(video_path)
            fps = video.get(cv2.CAP_PROP_FPS)
            total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
            video_duration = total_frames / fps if fps > 0 else 0  # Handle cases with zero fps
            video.release()

            detected_times, active_times, _ = detect_people_in_video(video_path, model, transform, 1)
            if detected_times != -1:
                cumulative_detected_time += detected_times
            total_video_duration += video_duration

            # 计算人员检测时间占比
            percentage = (detected_times / video_duration) * 100 if video_duration > 0 else 0

            result_label.config(text=f"检测到人员时间：{format_time(detected_times)}, 视频时长：{format_time(video_duration)}, 占比：{percentage:.2f}%") # 修改了这里
            
        except Exception as e:
            result_label.config(text=f"错误：{e}")
            continue #Skip to the next video if an error occurs


    percentage = (cumulative_detected_time / total_video_duration) * 100 if total_video_duration > 0 else 0

    result_text.set(f"累计检测到人员时间：{format_time(cumulative_detected_time)}\n"
                     f"所有视频总时长：{format_time(total_video_duration)}\n"
                     f"人员检测时间占比：{percentage:.2f}%")
    mass_result.set("所有选定视频分析完成。")



root.mainloop()
