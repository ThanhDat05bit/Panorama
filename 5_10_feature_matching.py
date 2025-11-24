import cv2, numpy as np, os, gc

# ==============================================================================
# CẤU HÌNH (Max Detail + De-Ghosting)
# ==============================================================================
MIN_MATCH, RANSAC, LOWE = 8, 4.0, 0.70
BLEND_LVL, BORDER = 3, 3
MAX_PIX = 300_000_000 

# ==============================================================================
# CÁC HÀM HỖ TRỢ
# ==============================================================================
def crop(img):
    _, t = cv2.threshold(cv2.cvtColor(img, 6), 1, 255, 0); c = cv2.findNonZero(t)
    if c is None: return img
    x, y, w, h = cv2.boundingRect(c); return img[y:y+h, x:x+w]

def get_mask(img):
    h, w = img.shape[:2]; m = np.ones((h, w), np.uint8) * 255
    m[:BORDER,:]=0; m[-BORDER:,:]=0; m[:,:BORDER]=0; m[:,-BORDER:]=0
    d = cv2.distanceTransform(m, cv2.DIST_L2, 3); return (d/d.max())**0.5 if d.max()>0 else d

def match_exp(ref, tgt, wm_r, wm_t):
    s = 4; ov = (wm_r[::s,::s]>0.01) & (wm_t[::s,::s]>0.01)
    if np.sum(ov)<100: return tgt
    d = np.clip(np.mean(cv2.cvtColor(ref[::s,::s],44)[:,:,0][ov]) - np.mean(cv2.cvtColor(tgt[::s,::s],44)[:,:,0][ov]), -20, 20)
    return cv2.LUT(tgt, np.array([i+d for i in range(256)]).clip(0,255).astype("uint8")) if abs(d)>1 else tgt

# ==============================================================================
# HÀM BLEND CẢI TIẾN: TÍCH HỢP XÓA BÓNG MA (SEAM CUTTING)
# ==============================================================================
def blend(can, img, wc, wi):
    # 1. Tìm vùng giao nhau (ROI)
    y, x = np.where(wi > 0)
    if not len(y): return can
    ym, yM, xm, xM = y.min(), y.max(), x.min(), x.max()
    
    # Mở rộng vùng đệm để chạy Pyramid
    p = 2**BLEND_LVL
    ym, yM, xm, xM = max(0, ym-p), min(can.shape[0], yM+p), max(0, xm-p), min(can.shape[1], xM+p)
    
    # Cắt dữ liệu ra để xử lý
    rc = can[ym:yM, xm:xM]
    ri = img[ym:yM, xm:xM]
    rwc = wc[ym:yM, xm:xM]
    rwi = wi[ym:yM, xm:xM]
    
    if rc.shape[0] < p or rc.shape[1] < p: return can

    # --- [NEW] LOGIC XÓA BÓNG MA (DEGHOSTING) ---
    # Tính sự khác biệt giữa ảnh cũ và ảnh mới tại vùng giao thoa
    # Chỉ tính ở những nơi cả 2 đều có dữ liệu (rwc > 0 và rwi > 0)
    overlap = (rwc > 0) & (rwi > 0)
    if np.sum(overlap) > 0:
        diff = np.mean(np.abs(rc.astype(np.float32) - ri.astype(np.float32)), axis=2)
        
        # Ngưỡng phát hiện bóng ma (lớn hơn 30 là coi như vật thể khác nhau/chuyển động)
        is_ghost = (diff > 30.0) & overlap
        
        # Winner Takes All: Pixel nào gần tâm hơn (weight lớn hơn) thì giữ lại, xóa pixel kia
        # Nếu Canvas mạnh hơn: Xóa trọng số của Image
        mask_kill_i = is_ghost & (rwc > rwi)
        rwi[mask_kill_i] = 0
        
        # Nếu Image mạnh hơn: Xóa trọng số của Canvas
        mask_kill_c = is_ghost & (rwi >= rwc)
        rwc[mask_kill_c] = 0
    # --------------------------------------------

    # Chuẩn bị Padding cho Pyramid
    hp, wp = (BLEND_LVL-(rc.shape[0]%BLEND_LVL))%BLEND_LVL, (BLEND_LVL-(rc.shape[1]%BLEND_LVL))%BLEND_LVL
    
    # Copy có border
    rc = cv2.copyMakeBorder(rc,0,hp,0,wp,2)
    ri = cv2.copyMakeBorder(ri,0,hp,0,wp,2)
    rwc = cv2.copyMakeBorder(rwc,0,hp,0,wp,0,value=0)
    rwi = cv2.copyMakeBorder(rwi,0,hp,0,wp,0,value=0)
    
    # Multi-band Blending
    G1, G2, MW = [rc.astype(float)], [ri.astype(float)], rwc+rwi
    MW[MW<1e-5]=1.0; A = rwi/MW # Alpha mask lúc này đã sắc nét ở vùng bóng ma (do bước de-ghosting trên)
    
    for _ in range(BLEND_LVL): G1.append(cv2.pyrDown(G1[-1])); G2.append(cv2.pyrDown(G2[-1]))
    L1 = [G1[i]-cv2.pyrUp(G1[i+1],dstsize=G1[i].shape[:2][::-1]) for i in range(BLEND_LVL)] + [G1[-1]]
    L2 = [G2[i]-cv2.pyrUp(G2[i+1],dstsize=G2[i].shape[:2][::-1]) for i in range(BLEND_LVL)] + [G2[-1]]
    
    GA = [A]
    for _ in range(BLEND_LVL): GA.append(cv2.pyrDown(GA[-1]))
    
    R = [L1[i]*(1.0-GA[i][...,None]) + L2[i]*GA[i][...,None] for i in range(len(L1))]
    res = R[-1]
    for i in range(len(R)-2,-1,-1): res = cv2.add(cv2.pyrUp(res, dstsize=R[i].shape[:2][::-1]), R[i])
    
    # Ghi đè lại vào Canvas
    can[ym:yM, xm:xM] = np.clip(res,0,255).astype(np.uint8)[:rc.shape[0]-hp, :rc.shape[1]-wp]
    
    # Cập nhật lại mask gốc (wc) bằng max weight mới (để lần ghép sau biết chỗ nào là ảnh chính)
    wc[ym:yM, xm:xM] = np.maximum(wc[ym:yM, xm:xM], wi[ym:yM, xm:xM]) # Cập nhật đơn giản cho vòng sau
    
    return can

# ==============================================================================
# CÁC HÀM KHÁC (GIỮ NGUYÊN)
# ==============================================================================
def get_H(i1, i2):
    s = cv2.SIFT_create(10000)
    k1, d1 = s.detectAndCompute(cv2.cvtColor(i1,6),None)
    k2, d2 = s.detectAndCompute(cv2.cvtColor(i2,6),None)
    if d1 is None or d2 is None: return None, 0
    m = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50)).knnMatch(d1, d2, k=2)
    g = [x for x, y in m if x.distance < LOWE * y.distance]
    if len(g) < MIN_MATCH: return None, 0
    p1 = np.float32([k2[m.trainIdx].pt for m in g]).reshape(-1,1,2)
    p2 = np.float32([k1[m.queryIdx].pt for m in g]).reshape(-1,1,2)
    M, mask = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=RANSAC)
    return (np.vstack([M, [0,0,1]]), np.sum(mask)) if M is not None else (None, 0)

def pipeline(imgs):
    n = len(imgs); sc = np.zeros(n)
    for i in range(n):
        for j in range(max(0,i-2), min(n,i+3)):
            if i!=j:
                h,w = imgs[i].shape[:2]; s = 400/w
                _, c = get_H(cv2.resize(imgs[i],(0,0),fx=s,fy=s), cv2.resize(imgs[j],(0,0),fx=s,fy=s))
                if c: sc[i]+=c
    cid = np.argmax(sc); H_g = [None]*n; H_g[cid] = np.eye(3); q = [cid]; vis = {cid}
    
    while q:
        cur = q.pop(0)
        for i in sorted(range(n), key=lambda x: abs(x-cur)):
            if i not in vis:
                H, c = get_H(imgs[cur], imgs[i])
                if H is not None and c >= MIN_MATCH: H_g[i] = H_g[cur]@H; vis.add(i); q.append(i)
    
    pts = np.concatenate([cv2.perspectiveTransform(np.float32([[0,0],[0,imgs[i].shape[0]],[imgs[i].shape[1],imgs[i].shape[0]],[imgs[i].shape[1],0]]).reshape(-1,1,2), H) for i, H in enumerate(H_g) if H is not None])
    mn, mx = np.int32(pts.min(0).ravel()-0.5), np.int32(pts.max(0).ravel()+0.5); cw, ch = mx-mn
    T = np.array([[1,0,-mn[0]],[0,1,-mn[1]],[0,0,1]], dtype=float)
    if cw*ch > MAX_PIX: s = (MAX_PIX/(cw*ch))**0.5; cw, ch = int(cw*s), int(ch*s); T = np.diag([s,s,1])@T
    
    can = np.zeros((ch, cw, 3), np.uint8); wm = np.zeros((ch, cw), np.float32)
    can = cv2.warpPerspective(imgs[cid], T@H_g[cid], (cw, ch), flags=4)
    wm = cv2.warpPerspective(get_mask(imgs[cid]), T@H_g[cid], (cw, ch), flags=1)
    
    for _, i in sorted([(np.linalg.norm(H_g[i][:2,2]), i) for i, H in enumerate(H_g) if H is not None and i!=cid]):
        H = T@H_g[i]
        wi = cv2.warpPerspective(get_mask(imgs[i]), H, (cw, ch), flags=1)
        # Warp ảnh mới
        im = match_exp(can, cv2.warpPerspective(imgs[i], H, (cw, ch), flags=4), wm, wi)
        
        # GỌI HÀM BLEND MỚI
        # Lưu ý: Ta không cần gc.collect() quá nhiều ở đây nếu RAM đủ, nhưng nên có nếu ảnh to
        can = blend(can, im, wm, wi)
        
        # Cập nhật mask tổng (wm) sau khi đã merge
        # Lưu ý: Hàm blend đã xử lý de-ghosting trên cục bộ rwc, rwi
        # Ở đây ta cập nhật wm toàn cục
        wm = np.maximum(wm, wi) 
        gc.collect()
        
    return can

def post_process(img):
    h, w = img.shape[:2]; _, t = cv2.threshold(cv2.cvtColor(img, 6), 1, 255, 0)
    idx = [np.where(t[:,x]>0)[0] for x in range(w)]
    valid = [(x, (i[0]+i[-1])/2) for x, i in enumerate(idx) if len(i)>h*0.2]
    if valid:
        poly = np.poly1d(np.polyfit([v[0] for v in valid], [v[1] for v in valid], 2)); res = np.zeros_like(img)
        for x in range(w):
            sh = int(h/2 - poly(x)); col = img[:,x]
            if sh > 0: res[sh:,x] = col[:h-sh]
            elif sh < 0: res[:h+sh,x] = col[-sh:]
            else: res[:,x] = col
        img = res
    img = crop(img); _, m = cv2.threshold(cv2.cvtColor(img, 6), 1, 255, 1) 
    img = cv2.inpaint(img, m, 3, cv2.INPAINT_TELEA) if np.sum(m) > 100 else img
    return np.clip(cv2.addWeighted(img, 2.0, cv2.GaussianBlur(img, (0,0), 3.0), -1.0, 0), 0, 255).astype(np.uint8)

if __name__ == '__main__':
    imgs = [cv2.imread(f"img3/{f}") for f in sorted(os.listdir("img3")) if f.endswith(('.jpg','.png'))]
    if len(imgs) > 1:
        res = pipeline(imgs)
        if res is not None: cv2.imwrite("result_deghost.jpg", post_process(res), [cv2.IMWRITE_JPEG_QUALITY, 100])
