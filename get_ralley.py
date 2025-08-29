import cv2
import numpy as np
from tqdm import tqdm
import kernal
import util
# Input and output files
input_file = "src/bwf_2.mp4"
bg_file = "src/bwf_2_background.png"
output_file = "src/bwf_2_ralley.mp4"

# Open input video
cap = cv2.VideoCapture(input_file)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()




# Video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"FPS={fps}, Size={width}x{height}, Frames={total_frames}")

# Define codec & create VideoWriter
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or "XVID"
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

# Move to start frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

background = cv2.imread(bg_file)

count = 0
# Loop through and save frames
for f in tqdm(range(0, total_frames), desc="Processing frames"):
    ret, frame = cap.read()
    if not ret:
        break            # to uint8 for display/diff
    fg = cv2.absdiff(frame, background)
    fg_gray = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
    if fg_gray.sum() < width*height*30:
        out.write(frame)
        count += 1


cap.release()
out.release()
cv2.destroyAllWindows()
