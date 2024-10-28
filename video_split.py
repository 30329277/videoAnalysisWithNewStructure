import os
import subprocess

# 输入视频路径和输出目录
input_video_path = r"data\03331.MTS"  # 使用原始字符串，替换为你的 MTS 视频路径
output_dir = "split_video"
os.makedirs(output_dir, exist_ok=True)

# 定义每个视频片段的时长（单位：秒）
segment_duration = 3 * 60  # 3分钟

# 定义每隔多少秒保存一帧画面
frame_interval = 3 * 60  # 每5秒保存一帧

# ffmpeg 可执行文件路径
ffmpeg_path = r"ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"

def split_video(input_path, output_directory, segment_length, frame_interval):
    # 获取视频文件名（不包括扩展名）
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # 构建输出模板路径 (视频)
    output_video_template = os.path.join(output_directory, f"{base_name}_%03d.mp4")
    
    # 构建输出模板路径 (图片)
    output_image_template = os.path.join(output_directory, f"{base_name}_%03d.jpg")

    # 构建 ffmpeg 命令 (分割视频)
    command_video = [
        ffmpeg_path,
        "-i", input_path,
        "-c", "copy",
        "-map", "0:0",
        "-map", "0:1",
        "-segment_time", str(segment_length),
        "-f", "segment",
        "-reset_timestamps", "1",
        output_video_template
    ]

    # 构建 ffmpeg 命令 (提取帧)
    command_image = [
        ffmpeg_path,
        "-i", input_path,
        "-vf", f"fps=1/{frame_interval}",  # 使用vf过滤器设置帧率
        output_image_template
    ]
    
    # 执行命令
    try:
        subprocess.run(command_video, check=True)
        print(f"视频已成功分割并保存在 {output_directory} 文件夹中")

        subprocess.run(command_image, check=True)
        print(f"图片已成功提取并保存在 {output_directory} 文件夹中")

    except subprocess.CalledProcessError as e:
        print(f"处理失败: {e}")

# 调用函数进行视频分割和帧提取
split_video(input_video_path, output_dir, segment_duration, frame_interval)