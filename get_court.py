import cv2
import numpy as np
from tqdm import tqdm

input_file = "src/bwf_2_background.png"
output_file = "src/bwf_2_court.png"
output_data_file = "src/bwf_2_court.csv"

frame = cv2.imread(input_file)
out = frame.copy()
for i in tqdm(range(frame.shape[0])):
    row = frame[i]
    for pixel in row:
        if np.all(pixel > [200, 200, 200]):
            pixel[:] = [pixel.min(), pixel.min(), pixel.min()]
        else:
            pixel[:] = [0, 0, 0]
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

cv2.imshow("gray", gray)
cv2.waitKey(1000)

lines = cv2.HoughLinesP(gray, rho=frame.shape[1]//100, theta=np.pi/360, threshold=100,
                        minLineLength=frame.shape[1]//10, maxLineGap=frame.shape[1]//100)
if lines is None:
    print("No lines found")
    lines = []

ms = []
shift = []
for l in lines:
    x1,y1,x2,y2 = l[0]
    x1 = float(x1)
    y1 = float(y1)
    x2 = float(x2)
    y2 = float(y2)
    if y1 > y2:
        l[0] = (x2,y2,x1,y1)
    m = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
    if abs(m) < 1:
        m = None
    b = (y1 - m * x1) if m != None else 0
    s = - b/m if m != None else None
    ms.append(m)
    shift.append(s)

boxl = [-1,-1,-1,-1]
boxr = [-1,-1,-1,-1]
for i in range(len(shift)):
    if shift[i] is None:
        continue
    if shift[i] < min([s for s in shift if s is not None]) + 5:
        if lines[i][0][1] < boxl[1] or boxl[1] == -1:
            boxl[0] = lines[i][0][0]
            boxl[1] = lines[i][0][1]
        if lines[i][0][3] > boxl[3] or boxl[3] == -1:
            boxl[2] = lines[i][0][2]
            boxl[3] = lines[i][0][3]
    elif shift[i] > max([s for s in shift if s is not None]) - 5:
        if lines[i][0][1] < boxr[1] or boxr[1] == -1:
            boxr[0] = lines[i][0][0]
            boxr[1] = lines[i][0][1]
        if lines[i][0][3] > boxr[3] or boxr[3] == -1:
            boxr[2] = lines[i][0][2]
            boxr[3] = lines[i][0][3]


cv2.line(out, (boxl[0],boxl[1]), (boxl[2],boxl[3]), (0,255,0), 2)
cv2.line(out, (boxr[0],boxr[1]), (boxr[2],boxr[3]), (0,255,0), 2)
cv2.line(out, (boxl[0],boxl[1]), (boxr[0],boxr[1]), (0,255,0), 2)
cv2.line(out, (boxl[2],boxl[3]), (boxr[2],boxr[3]), (0,255,0), 2)

cv2.imwrite(output_file, out.astype(np.uint8))
np.savetxt(output_data_file, [boxl, boxr], delimiter=",", fmt="%.3f")