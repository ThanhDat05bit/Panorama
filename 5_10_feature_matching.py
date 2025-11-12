import cv2
import numpy as np

img1 = cv2.imread('room3.jpg')
img2 = cv2.imread('room4.jpg')

img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(img1_gray, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2_gray, None)

print('Số keypoints ảnh 1:', len(keypoints1))
print('Số keypoints ảnh 2:', len(keypoints2))

matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.6 * n.distance:
        good_matches.append(m)

print('Số match tốt:', len(good_matches))

if len(good_matches) > 4:
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    height, width, _ = img2.shape

    warped_img1 = cv2.warpPerspective(img1, H, (width + img2.shape[1], height))
    img2_extended = np.zeros_like(warped_img1)
    img2_extended[0:height, 0:width] = img2

    def blend_pyramid(img1, img2, mask, levels=5):
        gp1 = [img1.copy()]
        gp2 = [img2.copy()]
        gp_mask = [mask.copy()]
        for i in range(levels):
            img1 = cv2.pyrDown(img1)
            img2 = cv2.pyrDown(img2)
            mask = cv2.pyrDown(mask)
            gp1.append(img1)
            gp2.append(img2)
            gp_mask.append(mask)
        lp1 = [gp1[-1]]
        lp2 = [gp2[-1]]
        for i in range(levels - 1, 0, -1):
            GE1 = cv2.pyrUp(gp1[i])
            GE2 = cv2.pyrUp(gp2[i])
            L1 = cv2.subtract(gp1[i - 1], GE1)
            L2 = cv2.subtract(gp2[i - 1], GE2)
            lp1.append(L1)
            lp2.append(L2)
        LS = []
        for l1, l2, gm in zip(lp1, lp2, gp_mask[::-1]):
            ls = l1 * gm + l2 * (1 - gm)
            LS.append(ls)
        blended = LS[0]
        for i in range(1, levels):
            blended = cv2.pyrUp(blended)
            blended = cv2.add(blended, LS[i])
        return blended

    mask = np.zeros((height, width + img2.shape[1], 3), dtype=np.float32)
    mask[:, :width] = 1

    panorama = blend_pyramid(
        warped_img1.astype(np.float32) / 255.0,
        img2_extended.astype(np.float32) / 255.0,
        mask, levels=5
    )
    panorama = (panorama * 255).astype(np.uint8)

    panorama = cv2.resize(panorama, (1000, 500))

    cv2.imshow('Panorama Blended', panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Không đủ điểm đặc trưng để ghép ảnh.")
