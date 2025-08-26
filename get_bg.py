import cv2
import numpy as np
import kernal
import util
# Input and output files
input_file = "bwf_1.mp4"
output_file = "bwf_1_ralley.mp4"
output_bg_file = "bwf_1_background.png"

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

# Get Background Model
ret, frame = cap.read()
bg_model = np.float32(frame)
alpha = 0.5  # your 0.1 update rate
track = np.zeros((3, 2), dtype=np.int32)  # to store tracked positions
all_track = np.zeros((total_frames, 2), dtype=np.int32)

if not ret:
    print("Error: Could not read background model.")
    cap.release()
    out.release()
    exit()

count = 0
background = None
# Loop through and save frames
for f in range(0, total_frames):
    ret, frame = cap.read()
    if not ret:
        break
    bg_u8 = cv2.convertScaleAbs(bg_model)               # to uint8 for display/diff
    fg = cv2.absdiff(frame, bg_u8)
    cv2.accumulateWeighted(frame, bg_model, alpha)

    fg_gray = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
    fg_clip = fg_gray.clip(0, 255)
    if fg_clip.sum() < 1000000:
        count += 1
        if background is None:
            background = frame.astype(np.float32)
        else:
            background = cv2.addWeighted(background, (count-1)/count, frame.astype(np.float32), 1/count, 0)
        out.write(frame)          # save frame to output

cv2.imwrite(output_bg_file, background.astype(np.uint8))

cap.release()
out.release()
cv2.destroyAllWindows()
