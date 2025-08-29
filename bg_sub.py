import cv2
import numpy as np
import kernal
import util
# Input and output files
input_file = "src/bwf_2_ralley.mp4"
output_file = "src/bwf_2_bg_sub.mp4"
bg_file = "src/bwf_2_background.png"
court_data = "src/bwf_2_court.csv"

loaded = np.loadtxt(court_data, delimiter=",")
boxl = loaded[0]
boxr = loaded[1]
net_width = (boxr[2] + boxr[0])/2 - (boxl [2] + boxl[0])/2 
background = cv2.imread(bg_file)
net_width = background.shape[1]//2  # approximate if court data not available
kernel_ = kernal.gaussian_kernel(net_width/130)
print(kernel_)
print("Court width in pixels:", net_width)



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

alpha = 0.15  # your 0.1 update rate
track = np.zeros((2, 2), dtype=np.int32)  # to store tracked positions
track[0] = track[1] = (height//2, width//2)
all_track = np.zeros((total_frames, 2), dtype=np.int32)

if not ret:
    print("Error: Could not read background model.")
    cap.release()
    out.release()
    exit()

td = 1
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

    result = cv2.filter2D(cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY).astype(np.float32), ddepth=-1, kernel=kernel_)  # smooth the fg mask


    result = np.clip(result, 0, None)  # shift to be non-negative


    H, W = result.shape
    cy, cx = track[0]+track[0]-track[1] # center point
    sigma = width/60      # controls spread

    Y, X = np.ogrid[:H, :W]
    dist2 = (X - cx)**2 + (Y - cy)**2
    mask = np.exp(-dist2 / (2 * sigma**2)).astype(np.float32)

    #result *= mask



    pos = np.unravel_index(result.argmax(), result.shape)
    print(result.max())
    result *= 255.0 / max(result.max(), net_width/10)  # scale to [0, 255]
    result = result.astype(np.uint8)

    
    if np.sqrt(dist2[pos]) < td * net_width / 10 and result[pos] > 200:
        td = 1
        track[1] = track[0]
        track[0] = pos
        cv2.circle(org, pos[::-1], 10, (0, 255, 0), 2)
    else:
        td += 1
    cv2.imshow("result", result)
    cv2.imshow("frame", org)
    cv2.imshow("bg_u8", frame)
    cv2.imshow("fg", fg)
    cv2.imshow("mask", (mask*255).astype(np.uint8))

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


