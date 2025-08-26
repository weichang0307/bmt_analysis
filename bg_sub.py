import cv2
import numpy as np
import kernal
import util
# Input and output files
input_file = "bwf_1_clip_3_ralley.mp4"
output_file = "bwf_1_clip_3_bg_sub.mp4"

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
background = cv2.imread("bwf_1_background.png")
alpha = 0.15  # your 0.1 update rate
track = np.zeros((2, 2), dtype=np.int32)  # to store tracked positions
all_track = np.zeros((total_frames, 2), dtype=np.int32)

if not ret:
    print("Error: Could not read background model.")
    cap.release()
    out.release()
    exit()

td = 100
tracking = 0
# Loop through and save frames
for f in range(0, total_frames):
    ret, frame = cap.read()
    if not ret:
        break
    org = frame.copy()
    frame = cv2.absdiff(frame, background)
    cv2.accumulateWeighted(frame, bg_model, alpha)
    bg_u8 = cv2.convertScaleAbs(bg_model)               # to uint8 for display/diff
    fg = cv2.absdiff(frame, bg_u8)
    vel = track[0] - track[1]

    kernel = kernal.gaussian_kernel()  # Create a Gaussian kernel
    result = cv2.filter2D(cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY).astype(np.float32), ddepth=-1, kernel=kernel)  # smooth the fg mask


    result = np.clip(result, 0, None)  # shift to be non-negative


    H, W = result.shape
    cy, cx = track[0]+track[0]-track[1] # center point
    sigma = 10       # controls spread

    Y, X = np.ogrid[:H, :W]
    dist2 = (X - cx)**2 + (Y - cy)**2
    mask = np.exp(-dist2 / (2 * sigma**2)).astype(np.float32)

    #result *= mask



    pos = np.unravel_index(result.argmax(), result.shape)
    print(result.max())
    result *= 255.0 / max(result.max(), 50)  # scale to [0, 255]
    result = result.astype(np.uint8)

    
    if dist2[pos] < td and result[pos] > 200:
        track[1] = track[0]
        track[0] = pos
        cv2.circle(org, pos[::-1], 10, (0, 255, 0), 2)

     
    else:
        td *= 1.04
    cv2.imshow("result", result)
    cv2.imshow("frame", org)
    cv2.imshow("bg_u8", frame)
    cv2.imshow("fg", fg)
    cv2.imshow("mask", (mask*255).astype(np.uint8))
    cv2.imshow("kernel", ((kernel+0.5)*255).astype(np.uint8))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# # write vedio with tracking
# cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
# for f in range(0, total_frames):
#     ret, frame = cap.read()
#     if not ret:
#         break
#     if np.linalg.norm(all_track[f]) > 0:
#         cv2.circle(frame, all_track[f][::-1], 10, (0, 255, 0), 2)
#     out.write(frame)          # save frame to output
#     cv2.imshow("Tracking...", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

cap.release()
out.release()
cv2.destroyAllWindows()


