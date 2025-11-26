import cv2, numpy as np
from sklearn.cluster import KMeans

img = cv2.imread("Dog.jpg"); h,w = img.shape[:2]; X = img.reshape(-1,3)
k = KMeans(n_clusters=16).fit(X)
out = k.cluster_centers_[k.labels_].astype("uint8").reshape(h,w,3)
cv2.imwrite("compressed.png", out)