import random
import cv2
import numpy as np
from itertools import combinations
import pyautogui
from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import KMeans
from tensorboard.compat.tensorflow_stub.dtypes import int32
from collections import Counter
from sklearn.cluster import DBSCAN
import math


label_path = r"C:\Users\34765\Desktop\zouruiwen\label\V0_F55.txt"
image_path = r"C:\Users\34765\Desktop\zouruiwen\image\V0_F55.png"
# label_path = r"C:\Users\34765\Desktop\zouruiwen\label\V40_F196.txt"
# image_path = r"C:\Users\34765\Desktop\zouruiwen\image\V40_F196.png"

lb=-25
ub=25

def compute_intersection(x1, y1, x2, y2, x3, y3, x4, y4):

    a1=(x1 - x2)*(y3 - y4)
    a2=(y1 - y2)*(x3 - x4)

    denominator = a1 - a2

    if abs(math.atan(a1)-math.atan(a2)) < 0.4:
        return None  # 平行或重合，没有交点

    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denominator
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denominator

    return (px, py)

def point_list(gray,t):
    edges = cv2.Canny(gray, 150,180, apertureSize=3)
    H, W = gray.shape

    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    # 可选：闭运算（更平滑）
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    lines = cv2.HoughLinesP(dilated, 2, np.pi / 180, threshold=40,
                             minLineLength=min(W,H)//5, maxLineGap=1)

    intersections=[]

    v=30+t
    if lines is not None:
        for line in lines:
            cv2.line(gray, (line[0,0],line[0,1]), (line[0,2],line[0,3]), 0, 1)
        # 计算所有直线的交点
        for (l1, l2) in combinations([l for l in lines], 2):
            x1,y1,x2,y2=l1[0]
            x3, y3, x4, y4 = l2[0]
            pt = compute_intersection(x1, y1, x2, y2, x3, y3, x4, y4)
            if pt and (0<pt[0]<v  or W-v<pt[0]<W or 0<pt[1]<v or H-v<pt[1]<H):
                cv2.circle(gray, tuple(map(int, pt)), 5, 0,thickness=2)
                intersections.append(pt)
    db = DBSCAN(eps=10, min_samples=3)
    labels = db.fit_predict(intersections)
    mx = max(labels)+1
    centers = [[0,0] for _ in range(mx)]
    num = [0]*mx
    for idx,val in enumerate(labels):
        centers[val][0] += intersections[idx][0]
        centers[val][1] += intersections[idx][1]
        num[val] += 1

    for i in range(mx):
        centers[i][0]//=num[i]
        centers[i][1]//=num[i]

    # if len(intersections)<4:
    #     return None, None
    # kmeans = KMeans(n_clusters=4, random_state=0)
    # kmeans.fit(intersections)

    # 获取聚类中心
    # centers = kmeans.cluster_centers_
    pts = []
    for (p1,p2) in combinations(centers, 2):
        dis = math.dist(p1,p2)
        if dis<max(W,H)+20:
            pts.append([p1,p2])

    mncen=100

    cor = [None]*4
    pts = np.array(pts)
    o=np.array((W/2,H/2))
    for (seg1,seg2) in combinations(pts, 2):
        o1=(seg1[0]+seg1[1])/2
        o2=(seg2[0]+seg2[1])/2
        dis = math.dist(o1,o2)+math.dist(o,o1)+math.dist(o,o2)
        if dis<mncen and math.dist(seg1[0],seg2[0])*math.dist(seg1[1],seg2[1])>W*H/5:
            mncen=dis
            cor=[seg1[0],seg2[0],seg1[1],seg2[1]]
            print(cor)

    if cor[0] is None:
        return None, None
    pts = np.array(cor,np.int32)
    extension_H, extension_W = max(2*t,(np.max(pts[:, 1])-W+5)*2), max(2*t,(np.max(pts[:, 0])-H+5)*2)
    pts+=np.array((extension_W//2,extension_H//2))
    mask = np.ones((H + extension_H, W + extension_W), np.uint8)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    mask = mask[extension_H//2:-extension_H//2, extension_W//2:-extension_W//2]

    # edges+=mask
    cv2.imshow("gray", edges)
    cv2.waitKey(0)
    return centers, mask





with open(label_path, 'r') as f:
    data = f.readlines()


img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
winH, winW = img.shape

# 解析前两个点的坐标（忽略类别）
for idx,line in enumerate(data):
    parts = line.strip().split()
    xc = float(parts[1])
    yc = float(parts[2])
    w = float(parts[3])
    h = float(parts[4])

    x1, x2 = xc - 0.5 * w, xc + 0.5 * w
    y1, y2 = yc - 0.5 * h, yc + 0.5 * h


    center_x_pixel = xc * winW
    center_y_pixel = yc * winH
    width_pixel = w * winW
    height_pixel = h * winH

    ep = 40
    x_min = int(center_x_pixel - width_pixel / 2)-ep
    y_min = int(center_y_pixel - height_pixel / 2)-ep
    x_max = int(center_x_pixel + width_pixel / 2)+ep
    y_max = int(center_y_pixel + height_pixel / 2)+ep

    winimg = img[y_min:y_max, x_min:x_max]
    pts, mask = point_list(winimg,t=20)
    if pts is None:
        print(f"{image_path}第{idx+1}个窗口解析失败")
        continue
    cv2.imshow("Mask", mask)
    cv2.waitKey(0)

# 居中显示窗口
# screen_w, screen_h = pyautogui.size()
# img_h, img_w = img.shape[:2]
# winx = max((screen_w - img_w) // 2, 0)
# winy = max((screen_h - img_h) // 2, 0)
# cv2.namedWindow("img", cv2.WINDOW_NORMAL)
# cv2.moveWindow("img", winx, winy)
# cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
# cv2.moveWindow("Mask", winx + 50, winy + 50)
# cv2.imshow("img", img)
# cv2.imshow("Mask", mask)
# cv2.waitKey(0)
cv2.destroyAllWindows()
