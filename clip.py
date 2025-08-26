import cv2

# Input and output files
input_file = "bwf_1.mp4"
output_file = "bwf_1_clip_4.mp4"

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

# Choose start and end times in seconds
start_sec = 510   
end_sec   = 1050  

start_frame = int(start_sec * fps)
end_frame   = int(end_sec * fps)

if start_frame >= total_frames or end_frame > total_frames or start_frame >= end_frame or start_frame < 0:
    print("Error: Invalid start or end time.")
    cap.release()
    out.release()
    exit()

# Move to start frame
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# Loop through and save frames
for f in range(start_frame, end_frame):
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)          # save frame to output

# Release video capture
cap.release()
out.release()
cv2.destroyAllWindows()