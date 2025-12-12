import cv2
import numpy as np
import os
import re
import gc
import gradio as gr

MAX_W = 1200        
ORB_N = 10000       
FOCAL_LENGTH = 1200 

def cylindrical_warp(img, f):
    h, w = img.shape[:2]
    K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])
    y, x = np.indices((h, w))
    X = np.linalg.inv(K).dot(np.stack([x, y, np.ones_like(x)], -1).reshape(h*w, 3).T).T
    B = K.dot(np.stack([np.sin(X[:,0]), X[:,1], np.cos(X[:,0])], -1).reshape(h*w, 3).T).T
    B = (B[:, :-1] / B[:, [-1]]).reshape(h, w, 2)
    img_cyl = cv2.remap(img, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    gray = cv2.cvtColor(img_cyl, cv2.COLOR_BGR2GRAY)
    cnts, _ = cv2.findContours(cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        img_cyl = img_cyl[y:y+h, x:x+w]
    return img_cyl

def get_homography(img1, img2):
    orb = cv2.ORB_create(nfeatures=ORB_N)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None: return None
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
    
    good = matches[:max(int(len(matches)*0.20), 10)]
    if len(good) < 4: return None
    
    src = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    dst = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    
    M, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    
    if M is None: return None
    return np.vstack([M, [0,0,1]])

def compensate(img1, img2, m1, m2):
    k = np.ones((5,5), np.uint8)
    ov = cv2.bitwise_and(cv2.erode(m1, k), cv2.erode(m2, k))
    if not cv2.countNonZero(ov): return img2
    
    mu1 = cv2.mean(img1, mask=ov)[:3]
    mu2 = cv2.mean(img2, mask=ov)[:3]
    
    res = img2.copy()
    for i in range(3):
        if mu2[i] > 10 and mu1[i] > 10:
            gain = mu1[i] / mu2[i]
            res[:,:,i] = np.clip(img2[:,:,i] * np.clip(gain, 0.9, 1.1), 0, 255)
    return res

def build_gaussian_pyramid(img, levels):
    G = img.copy()
    gp = [G]
    for i in range(levels):
        G = cv2.pyrDown(G)
        gp.append(G)
    return gp

def build_laplacian_pyramid(gp, levels):
    lp = [gp[levels-1]]
    for i in range(levels-1, 0, -1):
        GE = cv2.pyrUp(gp[i])
        rows, cols = gp[i-1].shape[:2]
        GE = cv2.resize(GE, (cols, rows))
        L = cv2.subtract(gp[i-1], GE)
        lp.append(L)
    return lp

def multi_band_blend(img1, img2, m1, m2, levels=4):
    m1f = m1.astype(np.float32) / 255.0
    m2f = m2.astype(np.float32) / 255.0
    
    d1 = cv2.distanceTransform(m1, cv2.DIST_L2, 3)
    d2 = cv2.distanceTransform(m2, cv2.DIST_L2, 3)
    sum_d = d1 + d2
    sum_d[sum_d == 0] = 1.0
    alpha = d1 / sum_d
    
    gp_alpha = build_gaussian_pyramid(alpha, levels)
    gp_img1 = build_gaussian_pyramid(img1.astype(np.float32), levels)
    lp_img1 = build_laplacian_pyramid(gp_img1, levels)
    gp_img2 = build_gaussian_pyramid(img2.astype(np.float32), levels)
    lp_img2 = build_laplacian_pyramid(gp_img2, levels)
    
    LS = []
    for l1, l2, mask in zip(lp_img1, lp_img2, gp_alpha[::-1]):
        rows, cols = l1.shape[:2]
        if mask.shape[:2] != (rows, cols):
            mask = cv2.resize(mask, (cols, rows))
        mask_3c = np.dstack([mask]*3)
        ls = l1 * mask_3c + l2 * (1.0 - mask_3c)
        LS.append(ls)
        
    ls_reconstruct = LS[0]
    for i in range(1, levels):
        ls_reconstruct = cv2.pyrUp(ls_reconstruct)
        rows, cols = LS[i].shape[:2]
        ls_reconstruct = cv2.resize(ls_reconstruct, (cols, rows))
        ls_reconstruct = cv2.add(ls_reconstruct, LS[i])
        
    return np.clip(ls_reconstruct, 0, 255).astype(np.uint8)

def straighten_shear_robust(img, cents):
    if len(cents) < 2: return img
    cx = np.array([c[0] for c in cents])
    cy = np.array([c[1] for c in cents])
    slope, intercept = np.polyfit(cx, cy, 1)
    
    if abs(slope) < 0.001: return img

    print(f"Robust Shear Correction (Slope: {slope:.5f})...")
    h, w = img.shape[:2]
    M = np.float32([[1, 0, 0], [-slope, 1, 0]])
    
    corners = np.float32([[0,0], [w,0], [w,h], [0,h]])
    new_corners = cv2.transform(np.array([corners]), M)[0] 
    
    min_x = new_corners[:,0].min(); max_x = new_corners[:,0].max()
    min_y = new_corners[:,1].min(); max_y = new_corners[:,1].max()
    new_w = int(max_x - min_x); new_h = int(max_y - min_y)
    
    M[0, 2] -= min_x; M[1, 2] -= min_y
    return cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

def crop_to_bounding_box(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return img

    c = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(c)
    
    print(f" -> Giữ lại vùng: x={x}, y={y}, w={w}, h={h}")

    return img[y:y+h, x:x+w]

def process_images(files):
    try:
        if not files or len(files) < 2:
            return None, "Cần tải ít nhất 2 ảnh!"

        key = lambda v: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', os.path.basename(v))]
        paths = sorted([f.name for f in files], key=key)
        
        imgs_raw = []
        print(f"--- Processing {len(paths)} images ---")
        
        for p in paths:
            i = cv2.imread(p)
            if i is None: continue
            h, w = i.shape[:2]
            scale = MAX_W / w if w > MAX_W else 1.0
            if scale < 1.0:
                i = cv2.resize(i, None, fx=scale, fy=scale)
            imgs_raw.append(i)
        
        if len(imgs_raw) < 2: return None, "Không đọc được đủ ảnh."

        imgs = [cylindrical_warp(i, FOCAL_LENGTH) for i in imgs_raw]
        curr = imgs[0]
        cents = [(curr.shape[1]/2, curr.shape[0]/2)]
        
        print("\nStitching started...")
        num_imgs = len(imgs)
        
        for i in range(1, num_imgs):
            print(f"... Ghép ảnh {i+1}/{num_imgs}")
            img_new = imgs[i]
            
            H = get_homography(curr, img_new)
            
            if H is None:
                tail = int(curr.shape[1] * 0.5) 
                Hr = get_homography(curr[:, -tail:], img_new)
                if Hr is None: continue
                H = np.array([[1, 0, curr.shape[1] - tail], [0, 1, 0], [0, 0, 1]]).dot(Hr)
            
            h1, w1 = curr.shape[:2]; h2, w2 = img_new.shape[:2]
            corners_new = cv2.perspectiveTransform(np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2), H).reshape(-1,2)
            pts = np.concatenate(([[0,0],[0,h1],[w1,h1],[w1,0]], corners_new))
            xmin, ymin = np.int32(pts.min(0) - 0.5); xmax, ymax = np.int32(pts.max(0) + 0.5)
            
            T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float32)
            H_final = T.dot(H)
            
            cents = [(cx-xmin, cy-ymin) for cx, cy in cents]
            new_center = cv2.perspectiveTransform(np.array([[[w2/2, h2/2]]], dtype=np.float32), H_final)[0,0]
            cents.append(tuple(new_center))
            
            wb = cv2.warpPerspective(curr, T, (xmax-xmin, ymax-ymin))
            wn = cv2.warpPerspective(img_new, H_final, (xmax-xmin, ymax-ymin))
            
            mb = cv2.threshold(cv2.cvtColor(wb, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)[1]
            mn = cv2.threshold(cv2.cvtColor(wn, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)[1]
            
            wn_comp = compensate(wb, wn, mb, mn)
            curr = multi_band_blend(wb, wn_comp, mb, mn, levels=4)
            gc.collect()

        print("Đang khử trôi dọc (Robust Shear)...")
        final = straighten_shear_robust(curr, cents)
        
        final = crop_to_bounding_box(final)
        
        return cv2.cvtColor(final, cv2.COLOR_BGR2RGB), "Hoàn tất!"
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return None, f"Lỗi: {str(e)}"

with gr.Blocks(title="Panorama") as demo:
    gr.Markdown("# 📸 Panorama")
    
   
    with gr.Row():
        
        with gr.Column(scale=1):
            file_input = gr.File(file_count="multiple", label="Input Images",height=300)
        
  
        with gr.Column(scale=1):
            btn_run = gr.Button("STITCH NOW", variant="primary")
            status_text = gr.Textbox(label="Log", interactive=False) 

   
    image_output = gr.Image(label="Result", type="numpy", interactive=False)

    btn_run.click(process_images, inputs=file_input, outputs=[image_output, status_text])

if __name__ == "__main__":
    demo.launch(share=True)
