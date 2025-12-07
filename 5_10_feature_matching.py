import cv2
import numpy as np
import os
import re
import gc
import gradio as gr

# --- CẤU HÌNH ---
MAX_W = 1200       # Giới hạn chiều ngang để xử lý nhanh hơn
ORB_N = 10000      # Số lượng điểm đặc trưng tối đa
FOCAL_LENGTH = 1500 # Tiêu cự giả định (Tăng lên để giảm độ cong ảnh)

# --- 1. KỸ THUẬT: WARP PERSPECTIVE / CYLINDRICAL ---
def cylindrical_warp(img, f):
    h, w = img.shape[:2]
    K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])
    y, x = np.indices((h, w))
    X = np.linalg.inv(K).dot(np.stack([x, y, np.ones_like(x)], -1).reshape(h*w, 3).T).T
    B = K.dot(np.stack([np.sin(X[:,0]), X[:,1], np.cos(X[:,0])], -1).reshape(h*w, 3).T).T
    B = (B[:, :-1] / B[:, [-1]]).reshape(h, w, 2)
    img_cyl = cv2.remap(img, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    # Cắt bỏ viền đen thừa
    gray = cv2.cvtColor(img_cyl, cv2.COLOR_BGR2GRAY)
    cnts, _ = cv2.findContours(cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        img_cyl = img_cyl[y:y+h, x:x+w]
    return img_cyl

# --- 2. KỸ THUẬT: SIFT/ORB & RANSAC ---
def get_homography(img1, img2):
    orb = cv2.ORB_create(nfeatures=ORB_N)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None: return None
    
    # Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
    
    # Lấy top 20% matches tốt nhất
    good = matches[:int(len(matches)*0.2)]
    if len(good) < 5: return None
    
    src = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    dst = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    
    # RANSAC để lọc nhiễu
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)
    return np.vstack([M, [0,0,1]]) if M is not None else None

# --- 3. KỸ THUẬT: EXPOSURE COMPENSATION ---
def compensate(img1, img2, m1, m2):
    # Tìm vùng chồng lấn (overlap)
    k = np.ones((5,5), np.uint8)
    ov = cv2.bitwise_and(cv2.erode(m1, k), cv2.erode(m2, k))
    if not cv2.countNonZero(ov): return img2
    
    # Tính trung bình màu vùng chồng lấn
    mu1 = cv2.mean(img1, mask=ov)[:3]
    mu2 = cv2.mean(img2, mask=ov)[:3]
    
    res = img2.copy()
    for i in range(3):
        if mu2[i] > 10 and mu1[i] > 10:
            # Điều chỉnh độ lợi (Gain compensation)
            gain = mu1[i] / mu2[i]
            res[:,:,i] = np.clip(img2[:,:,i] * np.clip(gain, 0.8, 1.2), 0, 255)
    return res

# --- 4. KỸ THUẬT: MULTI-BAND BLENDING (Pyramid Blending) ---
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
        # Resize bắt buộc để khớp kích thước do sai số làm tròn của pyrDown
        rows, cols = gp[i-1].shape[:2]
        GE = cv2.resize(GE, (cols, rows))
        L = cv2.subtract(gp[i-1], GE)
        lp.append(L)
    return lp

def multi_band_blend(img1, img2, m1, m2, levels=4):
    # Chuyển mask sang float để làm trọng số
    m1f = m1.astype(np.float32) / 255.0
    m2f = m2.astype(np.float32) / 255.0
    
    # Distance transform để tạo mask mềm mịn (seam finding đơn giản)
    d1 = cv2.distanceTransform(m1, cv2.DIST_L2, 3)
    d2 = cv2.distanceTransform(m2, cv2.DIST_L2, 3)
    sum_d = d1 + d2
    sum_d[sum_d == 0] = 1.0
    alpha = d1 / sum_d
    
    # 1. Tạo Gaussian Pyramid cho mask Alpha
    gp_alpha = build_gaussian_pyramid(alpha, levels)
    
    # 2. Tạo Laplacian Pyramid cho 2 ảnh
    gp_img1 = build_gaussian_pyramid(img1.astype(np.float32), levels)
    lp_img1 = build_laplacian_pyramid(gp_img1, levels)
    gp_img2 = build_gaussian_pyramid(img2.astype(np.float32), levels)
    lp_img2 = build_laplacian_pyramid(gp_img2, levels)
    
    # 3. Trộn (Blend) từng tầng
    LS = []
    for l1, l2, mask in zip(lp_img1, lp_img2, gp_alpha[::-1]):
        rows, cols = l1.shape[:2]
        if mask.shape[:2] != (rows, cols):
            mask = cv2.resize(mask, (cols, rows))
        mask_3c = np.dstack([mask]*3)
        ls = l1 * mask_3c + l2 * (1.0 - mask_3c)
        LS.append(ls)
        
    # 4. Tái tạo ảnh (Reconstruct)
    ls_reconstruct = LS[0]
    for i in range(1, levels):
        ls_reconstruct = cv2.pyrUp(ls_reconstruct)
        rows, cols = LS[i].shape[:2]
        ls_reconstruct = cv2.resize(ls_reconstruct, (cols, rows))
        ls_reconstruct = cv2.add(ls_reconstruct, LS[i])
        
    return np.clip(ls_reconstruct, 0, 255).astype(np.uint8)

# --- 5. KỸ THUẬT: STRAIGHTENING (NẮN THẲNG) ---
def straighten(img, cents):
    if len(cents) < 2: return img
    pts = np.array(cents)
    
    # Dùng fitLine thay vì lstsq để chống nhiễu tốt hơn
    [vx, vy, x, y] = cv2.fitLine(pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
    angle = np.degrees(np.arctan2(vy, vx))[0]
    
    print(f" -> Góc nghiêng phát hiện: {angle:.2f} độ")
    
    # Chỉ xoay nếu nghiêng đáng kể (> 0.5 độ)
    if abs(angle) < 0.5: return img
    
    h, w = img.shape[:2]
    # Xoay ngược chiều kim đồng hồ để bù góc nghiêng
    M = cv2.getRotationMatrix2D((w//2, h//2), -angle, 1.0)
    
    # Tính lại kích thước bounding box để không bị mất góc ảnh
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))
    M[0, 2] += (nw / 2) - w / 2
    M[1, 2] += (nh / 2) - h / 2
    
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LANCZOS4)

# --- LOGIC CHÍNH ---
def process_images(files):
    try:
        if not files or len(files) < 2:
            return None, "Cần tải ít nhất 2 ảnh!"

        # Sắp xếp file theo tên số (1.jpg, 2.jpg...)
        key = lambda v: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', os.path.basename(v))]
        paths = sorted([f.name for f in files], key=key)
        
        imgs_raw = []
        print(f"--- Đang xử lý {len(paths)} ảnh... ---")
        
        for p in paths:
            i = cv2.imread(p)
            if i is None: continue
            h, w = i.shape[:2]
            # Resize nếu ảnh quá lớn để tránh tràn RAM
            imgs_raw.append(cv2.resize(i, None, fx=MAX_W/w, fy=MAX_W/w) if w > MAX_W else i)
        
        if len(imgs_raw) < 2: return None, "Lỗi đọc ảnh."

        # Warp trụ
        imgs = [cylindrical_warp(i, FOCAL_LENGTH) for i in imgs_raw]
        curr = imgs[0]
        cents = [(imgs[0].shape[1]/2, imgs[0].shape[0]/2)]
        
        print("\nBắt đầu ghép...")
        num_imgs = len(imgs)
        
        for i in range(1, num_imgs):
            print(f"... Đang ghép ảnh {i+1}/{num_imgs}")
            img_new = imgs[i]
            
            # Tìm Homography
            H = get_homography(curr, img_new)
            
            # Nếu không tìm thấy, thử cắt lấy phần đuôi ảnh trước để tìm lại (overlap area)
            if H is None:
                tail = int(curr.shape[1] * 0.5) # Lấy 50% ảnh cuối
                Hr = get_homography(curr[:, -tail:], img_new)
                if Hr is None:
                    print(f" -> CẢNH BÁO: Không khớp được ảnh {i+1}, bỏ qua."); continue
                H = np.array([[1, 0, curr.shape[1] - tail], [0, 1, 0], [0, 0, 1]]).dot(Hr)
            
            # Tính kích thước canvas mới
            h1, w1 = curr.shape[:2]; h2, w2 = img_new.shape[:2]
            corners_new = cv2.perspectiveTransform(np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2), H).reshape(-1,2)
            pts = np.concatenate(([[0,0],[0,h1],[w1,h1],[w1,0]], corners_new))
            xmin, ymin = np.int32(pts.min(0) - 0.5); xmax, ymax = np.int32(pts.max(0) + 0.5)
            
            # Ma trận dịch chuyển
            T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float32)
            H_final = T.dot(H)
            
            # Cập nhật tâm ảnh để nắn thẳng sau này
            new_center = cv2.perspectiveTransform(np.array([[[w2/2, h2/2]]], dtype=np.float32), H_final)[0,0]
            cents = [(cx-xmin, cy-ymin) for cx, cy in cents] + [tuple(new_center)]
            
            # Warp ảnh
            wb = cv2.warpPerspective(curr, T, (xmax-xmin, ymax-ymin))
            wn = cv2.warpPerspective(img_new, H_final, (xmax-xmin, ymax-ymin))
            
            # Tạo mask
            mb = cv2.threshold(cv2.cvtColor(wb, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)[1]
            mn = cv2.threshold(cv2.cvtColor(wn, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)[1]
            
            # Cân bằng sáng (Exposure Compensation)
            wn_compensated = compensate(wb, wn, mb, mn)
            
            # Ghép đa tần số (Multi-band Blending)
            curr = multi_band_blend(wb, wn_compensated, mb, mn, levels=4)
            
            gc.collect() # Dọn RAM

        print("Đang nắn thẳng (Straightening)...")
        final = straighten(curr, cents)
        
        # Crop bỏ viền đen lần cuối
        gray_final = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_final, 1, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(thresh)
        final = final[y:y+h, x:x+w]
        
        final_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
        return final_rgb, "Ghép ảnh thành công! (Đã dùng Multi-band Blending)"
        
    except Exception as e:
        print(f"LỖI: {str(e)}")
        return None, f"Lỗi: {str(e)}"

# --- GIAO DIỆN GRADIO ---
with gr.Blocks(title="Panorama Stitcher Pro") as demo:
    gr.Markdown("# 📸 Panorama Stitcher (Full Tech)")
    gr.Markdown("Đã tích hợp: SIFT/ORB, RANSAC, Warp, Exposure Comp, Multi-band Blending.")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(file_count="multiple", label="Tải ảnh lên (Thứ tự trái -> phải)")
            btn_run = gr.Button("Bắt đầu Ghép", variant="primary")
            status_text = gr.Textbox(label="Log", interactive=False) 
        with gr.Column(scale=3):
            image_output = gr.Image(label="Kết quả Panorama", type="numpy", interactive=False)

    btn_run.click(process_images, inputs=file_input, outputs=[image_output, status_text])

if __name__ == "__main__":
    demo.launch(share=True)
