import cv2
import numpy as np

def linear_mask(w, h, overlap_min, overlap_max):
    """Tạo mask tuyến tính cho blending vùng overlap"""
    mask = np.zeros((h, w), np.float32)
    for x in range(overlap_min, overlap_max):
        alpha = (x - overlap_min) / (overlap_max - overlap_min + 1e-6)
        mask[:, x] = alpha
    mask[:, :overlap_min] = 0.0
    mask[:, overlap_max:] = 1.0
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    return np.dstack([mask]*3)

def multiband_blend(img1, img2, mask, levels=6):
    img1_f = img1.astype(np.float32)
    img2_f = img2.astype(np.float32)

    gp1 = [img1_f]
    gp2 = [img2_f]
    gpm = [mask]

    for _ in range(levels):
        gp1.append(cv2.pyrDown(gp1[-1]))
        gp2.append(cv2.pyrDown(gp2[-1]))
        gpm.append(cv2.pyrDown(gpm[-1]))

    lp1, lp2 = [], []
    for i in range(len(gp1)-1):
        size = (gp1[i].shape[1], gp1[i].shape[0])
        lp1.append(gp1[i] - cv2.pyrUp(gp1[i+1], dstsize=size))
        lp2.append(gp2[i] - cv2.pyrUp(gp2[i+1], dstsize=size))

    lp1.append(gp1[-1])
    lp2.append(gp2[-1])

    LS = []
    gpm_rev = gpm[::-1]
    for l1, l2, gm in zip(lp1, lp2, gpm_rev):
        if gm.shape[:2] != l1.shape[:2]:
            gm = cv2.resize(gm, (l1.shape[1], l1.shape[0]))
        LS.append(l1 * (1 - gm) + l2 * gm)

    blended = LS[-1]
    for i in range(len(LS)-2, -1, -1):
        blended = cv2.pyrUp(blended, dstsize=(LS[i].shape[1], LS[i].shape[0]))
        blended = blended + LS[i]

    return np.clip(blended, 0, 255).astype(np.uint8)
# Ghép Nhiều ảnh
# load ảnh
#  Folder chứa ảnh
folder = "imgs"  # đổi theo folder 
image_files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))])

# Load tất cả ảnh
images = [cv2.imread(os.path.join(folder, f)) for f in image_files]

if len(images) < 2:
    raise RuntimeError("Can it nhat 2 anh de ghep.")

# Ghép nhiều ảnh 
panorama = images[0]

for i in range(1, len(images)):
    img1 = panorama 
    img2 = images[i] 

    print(f"Dang ghep anh {i+1}/{len(images)} ...")

img1 = cv2.imread('anh1.jpg')
img2 = cv2.imread('anh2.jpg')

img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

keypoints1, descriptors1 = sift.detectAndCompute(img1_gray, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2_gray, None)

matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

goods=[]
for m, n in matches:
    if m.distance < 0.6*n.distance:
        goods.append(m)

if len(goods) < 4:
    raise RuntimeError("Không đủ match!")

src_pts = np.float32([keypoints1[m.queryIdx].pt for m in goods]).reshape(-1,1,2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in goods]).reshape(-1,1,2)

H, maskH = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# ===== Canvas =====
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

corners_img1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
warped_corners = cv2.perspectiveTransform(corners_img1, H)

all_pts = np.concatenate((warped_corners, np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)), axis=0)

xmin, ymin = np.int32(all_pts.min(axis=0).ravel() - 5)
xmax, ymax = np.int32(all_pts.max(axis=0).ravel() + 5)

T = np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]])
canvas_w, canvas_h = xmax - xmin, ymax - ymin

warp_img1 = cv2.warpPerspective(img1, T @ H, (canvas_w, canvas_h))

canvas_img2 = np.zeros_like(warp_img1)
canvas_img2[-ymin:-ymin+h2, -xmin:-xmin+w2] = img2

#  Mask tuyến tính trong vùng overlap 
overlap = (np.any(warp_img1 != 0, axis=2) & np.any(canvas_img2 != 0, axis=2))
ys, xs = np.where(overlap == True)
xmin_overlap, xmax_overlap = xs.min(), xs.max()

mask = linear_mask(canvas_w, canvas_h, xmin_overlap, xmax_overlap)

#  Multi-band blending 
final = multiband_blend(warp_img1, canvas_img2, mask, levels=7)

# Crop viền đen 
gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
coords = cv2.findNonZero(thresh)
x,y,w,h = cv2.boundingRect(coords)
final_crop = final[y:y+h, x:x+w]

# Show result
final_resize = cv2.resize(final_crop, (1000, 500))
cv2.imshow('Panorama', final_resize)
cv2.waitKey(0)
cv2.destroyAllWindows()

