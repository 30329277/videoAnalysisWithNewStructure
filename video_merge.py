import os
import subprocess
import shutil  # 用于文件操作

input_dir = r"split_video"  # 使用原始字符串，替换为你的输入文件夹路径
output_dir = r"merge_video"
os.makedirs(output_dir, exist_ok=True)

# ffmpeg 可执行文件路径
ffmpeg_path = r"ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"  # 请替换为你的ffmpeg路径

def merge_videos(input_directory, output_directory, sort_method=1):
    """
    合并指定目录下的视频文件。

    Args:
        input_directory: 输入视频所在的目录。
        output_directory: 输出合并后视频的目录。
        sort_method: 排序方法：
            1: 按名称顺序
            2: 按名称逆序
            3: 按文件修改时间顺序
            4: 按文件修改时间逆序
    """

    video_files = [f for f in os.listdir(input_directory) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mts'))]

    if not video_files:
        print(f"在 {input_directory} 中找不到视频文件。")
        return

    if sort_method == 1:
        video_files.sort()
    elif sort_method == 2:
        video_files.sort(reverse=True)
    elif sort_method == 3:
        video_files.sort(key=lambda x: os.path.getmtime(os.path.join(input_directory, x)))
    elif sort_method == 4:
        video_files.sort(key=lambda x: os.path.getmtime(os.path.join(input_directory, x)), reverse=True)
    else:
        print("无效的排序方法。使用默认的名称顺序排序。")


    # 创建一个临时文件来存储视频文件列表
    temp_list_file = os.path.join(output_directory, "temp_list.txt")
    with open(temp_list_file, "w") as f:
        for file in video_files:
            # 使用os.path.abspath获取绝对路径，避免相对路径问题
            absolute_file_path = os.path.abspath(os.path.join(input_directory, file))
            f.write(f"file '{absolute_file_path}'\n") # 使用绝对路径

    # 构建输出视频路径
    output_video_path = os.path.join(output_directory, "merged_video.mp4")

    # 构建 ffmpeg 命令
    command = [
        ffmpeg_path,
        "-f", "concat",
        "-safe", "0",  # 允许使用绝对路径
        "-i", temp_list_file,
        "-c", "copy",
        output_video_path
    ]

    # 执行命令
    try:
        subprocess.run(command, check=True)
        print(f"视频已成功合并到 {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"视频合并失败: {e}")

    # 删除临时文件
    os.remove(temp_list_file)



# 调用函数进行视频合并，使用默认的按名称顺序排序
merge_videos(input_dir, output_dir)


# 示例：按修改时间逆序排序
# merge_videos(input_dir, output_dir, sort_method=1)