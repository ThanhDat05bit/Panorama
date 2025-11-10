
import cv2
import numpy as np

img1 = cv2.imread('room3.jpg')
img2 = cv2.imread('room4.jpg')

img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
# detector = cv2.ORB_create()

keypoints1, descriptors1 = sift.detectAndCompute(img1_gray, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2_gray, None)

print('number of keypoints1: ',len(keypoints1))
print('number of descriptors1: ',len(descriptors1))

print('number of keypoints2: ',len(keypoints2))
print('number of descriptors2: ',len(descriptors2))

# matcher = cv2.BFMatcher()
matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)


# matches = matcher.match(descriptors1, descriptors2)
matches = matcher.knnMatch(descriptors1,descriptors2,k=2)

goods=[]
for m, n in matches:
    if m.distance < 0.6*n.distance:
        goods.append(m)




img = cv2.drawMatches(img1,keypoints1,
                img2,keypoints2,
                goods,None)

if len(goods)>4:
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in goods]).reshape(-1,1,2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in goods]).reshape(-1,1,2)

    H, mask=cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0)
    height,width,_=img2.shape
    panorama = cv2.warpPerspective(img1,H,((width+img2.shape[1]),height))
    panorama[0:height,0:width]=img2

    panorama = cv2.resize(panorama,(1000,500))
    cv2.imshow('Anh panorama',panorama)
    cv2.waitKey(0)


# h,w,_=img.shape
# img = cv2.resize(img,(w//2,h//2))

# cv2.imshow('Anh so khop dac trung',img)
# cv2.waitKey(0)



