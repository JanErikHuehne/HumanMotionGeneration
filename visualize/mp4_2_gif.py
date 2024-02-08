from moviepy.editor import VideoFileClip

# Replace 'path/to/video.mp4' with the path to your MP4 file
input_path = r'E:\ADL4CV\HumanMotionGeneration\user_output\fixed_length\user_input_2_seed10\sample00_rep01_0.mp4'
# Replace 'output.gif' with the desired output GIF file name
output_path = r'E:\ADL4CV\HumanMotionGeneration\user_output\fixed_length\user_input_2_seed10\sample00_rep01_0.gif'

# Load the video file
clip = VideoFileClip(input_path)

# Optionally, you can resize the clip to reduce the output GIF size
# clip = clip.resize(width=480)  # You can adjust the width as needed

# Write the GIF file
clip.write_gif(output_path, fps=20)  # You can adjust the fps (frames per second) as needed