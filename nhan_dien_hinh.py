# -*- coding: utf-8 -*-
"""
NHẬN DIỆN HÌNH DẠNG BẰNG PHÁT HIỆN BIÊN (không dùng OpenCV / NumPy)
Pipeline: Ảnh -> Xám -> Làm mờ -> Sobel/Prewitt/Laplacian -> Otsu -> Giãn nở
          -> Gán nhãn vùng -> Phân loại (tròn, tam giác, vuông, chữ nhật)
Pillow + tkinter chỉ dùng để mở/lưu/hiển thị ảnh và dựng giao diện.
"""
import math
import tkinter as tk
from collections import deque
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageDraw, ImageFont, ImageTk

MAX_SIZE = 400  # thu nhỏ ảnh để Python thuần chạy nhanh

# ---------------------------------------------------------------- 1. ĐỌC ẢNH
def load_rgb(path):
    img = Image.open(path).convert("RGB")
    img.thumbnail((MAX_SIZE, MAX_SIZE))
    return img


# ---------------------------------------------------------------- 2. ẢNH XÁM
def to_gray(img):
    w, h = img.size
    px = img.load()
    gray = [[0] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            r, g, b = px[x, y]
            gray[y][x] = int(0.299 * r + 0.587 * g + 0.114 * b)
    return gray


# ---------------------------------------------------------------- 3. TÍCH CHẬP + BIÊN
KERNELS = {
    "Sobel": (
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],   # Gx
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],   # Gy
    ),
    "Prewitt": (
        [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
        [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
    ),
    "Laplacian": ([[0, 1, 0], [1, -4, 1], [0, 1, 0]], None),
}


def convolve3(m, k):
    """Tích chập 3x3, viền ngoài để bằng 0."""
    h, w = len(m), len(m[0])
    out = [[0] * w for _ in range(h)]
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            s = 0
            for j in range(3):
                row = m[y + j - 1]
                kj = k[j]
                for i in range(3):
                    s += row[x + i - 1] * kj[i]
            out[y][x] = s
    return out


def blur(gray):
    """Làm mờ trung bình 3x3 để giảm nhiễu."""
    box = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    c = convolve3(gray, box)
    return [[v // 9 for v in row] for row in c]


def edge_magnitude(gray, method):
    kx, ky = KERNELS[method]
    gx = convolve3(gray, kx)
    h, w = len(gray), len(gray[0])
    if ky is None:  # Laplacian
        mag = [[abs(v) for v in row] for row in gx]
    else:           # Sobel / Prewitt: sqrt(Gx^2 + Gy^2)
        gy = convolve3(gray, ky)
        mag = [[math.sqrt(gx[y][x] ** 2 + gy[y][x] ** 2) for x in range(w)]
               for y in range(h)]
    # bỏ 2 pixel viền (viền bị sai do tích chập/làm mờ)
    for y in range(h):
        for x in range(w):
            if x < 2 or y < 2 or x >= w - 2 or y >= h - 2:
                mag[y][x] = 0
    mx = max(max(r) for r in mag) or 1
    return [[int(v * 255 / mx) for v in row] for row in mag]


# ---------------------------------------------------------------- 4. NHỊ PHÂN HÓA (OTSU)
def otsu_threshold(m):
    hist = [0] * 256
    for row in m:
        for v in row:
            hist[v] += 1
    total = sum(hist)
    sum_all = sum(i * hist[i] for i in range(256))
    sum_b = w_b = 0
    best_var, best_t = -1, 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return best_t


def binarize(m, t):
    return [[1 if v > t else 0 for v in row] for row in m]


def dilate(b):
    """Giãn nở 3x3 để nối các nét biên bị đứt."""
    h, w = len(b), len(b[0])
    out = [[0] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            if b[y][x]:
                for j in range(max(0, y - 1), min(h, y + 2)):
                    for i in range(max(0, x - 1), min(w, x + 2)):
                        out[j][i] = 1
    return out


# ---------------------------------------------------------------- 5. GÁN NHÃN VÙNG
def find_regions(edge, min_area=150):
    """
    Coi các pixel KHÔNG phải biên là vùng. Vùng chạm viền ảnh là nền -> bỏ.
    Vùng nằm trọn trong ảnh (bị đường biên bao quanh) chính là phần trong của hình.
    """
    h, w = len(edge), len(edge[0])
    seen = [[False] * w for _ in range(h)]
    regions = []
    for sy in range(h):
        for sx in range(w):
            if edge[sy][sx] or seen[sy][sx]:
                continue
            q = deque([(sx, sy)])
            seen[sy][sx] = True
            pix = []
            touch = False
            while q:
                x, y = q.popleft()
                pix.append((x, y))
                if x == 0 or y == 0 or x == w - 1 or y == h - 1:
                    touch = True
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if 0 <= nx < w and 0 <= ny < h and not seen[ny][nx] and not edge[ny][nx]:
                        seen[ny][nx] = True
                        q.append((nx, ny))
            if not touch and len(pix) >= min_area:
                xs = [p[0] for p in pix]
                ys = [p[1] for p in pix]
                regions.append({"area": len(pix), "pix": pix,
                                "x0": min(xs), "y0": min(ys), "x1": max(xs), "y1": max(ys)})
    return regions


# ---------------------------------------------------------------- 6. HÌNH HỌC + PHÂN LOẠI
# Cách làm KHÔNG phụ thuộc góc xoay và chịu được méo nhẹ:
#   biên vùng -> bao lồi (convex hull) -> xấp xỉ đa giác (Douglas-Peucker)
#   -> độ tròn 4πA/P² và số đỉnh
def boundary_points(pix):
    s = set(pix)
    return [(x, y) for (x, y) in pix
            if any((x + dx, y + dy) not in s for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))]


def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points):
    """Andrew monotone chain."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def dist_point_line(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = math.hypot(dx, dy)
    if n == 0:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    return abs(dx * (p[1] - a[1]) - dy * (p[0] - a[0])) / n


def rdp(points, eps):
    """Douglas-Peucker cho một đường hở."""
    if len(points) < 3:
        return points
    a, b = points[0], points[-1]
    idx, dmax = 0, 0
    for i in range(1, len(points) - 1):
        d = dist_point_line(points[i], a, b)
        if d > dmax:
            dmax, idx = d, i
    if dmax > eps:
        left = rdp(points[:idx + 1], eps)
        right = rdp(points[idx:], eps)
        return left[:-1] + right
    return [a, b]


def approx_polygon(hull, eps):
    """Douglas-Peucker cho đa giác kín: tách tại điểm xa nhất so với đỉnh đầu."""
    n = len(hull)
    if n < 4:
        return hull
    a = hull[0]
    k = max(range(n), key=lambda i: (hull[i][0] - a[0]) ** 2 + (hull[i][1] - a[1]) ** 2)
    c1 = rdp(hull[:k + 1], eps)
    c2 = rdp(hull[k:] + [hull[0]], eps)
    return c1[:-1] + c2[:-1]


def poly_perimeter(poly):
    return sum(math.dist(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly)))


def poly_area(poly):
    s = 0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


# ---- Các ngưỡng có thể chỉnh -------------------------------------------------
ANGLE_TOL = 12          # góc tứ giác lệch tối đa so với 90° để là vuông/chữ nhật (độ)
SQUARE_RATIO = 0.85     # cạnh ngắn/cạnh dài >= giá trị này -> hình vuông
MERGE_RATIO = 0.08      # cạnh ngắn hơn 8% chu vi -> coi là đỉnh thừa, gộp lại
MIN_TURN = 22           # đỉnh có góc rẽ nhỏ hơn (độ) gần như thẳng hàng -> đỉnh thừa, xóa
SOLIDITY_MIN = 0.92     # diện tích vùng / diện tích bao lồi; thấp hơn -> khuyết/mất góc -> bỏ
ELLIPSE_CV = 0.015      # độ lệch bán kính chuẩn hóa tối đa để được coi là tròn/elip
ELLIPSE_CV_PX = 0.4     # cộng thêm ELLIPSE_CV_PX / (bán trục nhỏ) vì hình nhỏ nhiễu pixel nhiều hơn
CIRCLE_RATIO = 0.92     # trục nhỏ/trục lớn >= giá trị này -> tròn, nhỏ hơn -> elip
ELLIPSE_MIN_RATIO = 0.45  # dẹt hơn nữa -> quá méo -> bỏ
# -------------------------------------------------------------------------------

VN_NAME = {
    "Hinh tron": "Hình tròn",
    "Elip": "Elip",
    "Tam giac": "Tam giác",
    "Hinh vuong": "Hình vuông",
    "Hinh chu nhat": "Hình chữ nhật",
    "Tu giac": "Tứ giác",
    "Ngu giac": "Ngũ giác",
    "Luc giac": "Lục giác",
    "That giac": "Thất giác",
    "Bat giac": "Bát giác",
    "Cuu giac": "Cửu giác",
    "Thap giac": "Thập giác",
}
POLY_KEY = {5: "Ngu giac", 6: "Luc giac", 7: "That giac",
            8: "Bat giac", 9: "Cuu giac", 10: "Thap giac"}
COLORS = {
    "Hinh tron": (220, 0, 0), "Elip": (200, 0, 100), "Tam giac": (0, 150, 0),
    "Hinh vuong": (0, 0, 255), "Hinh chu nhat": (255, 140, 0), "Tu giac": (150, 0, 150),
    "Ngu giac": (0, 150, 150), "Luc giac": (120, 120, 0), "That giac": (90, 60, 200),
    "Bat giac": (160, 80, 0), "Cuu giac": (0, 100, 160), "Thap giac": (100, 100, 100),
}


def ellipse_stats(pix, pts):
    """
    Khớp elip theo mô-men bậc 2 (không phụ thuộc góc xoay).
    Trả về (cv, tỉ lệ trục nhỏ/trục lớn, bán trục nhỏ). cv là độ lệch chuẩn của bán kính chuẩn hóa
    của các điểm biên: càng nhỏ thì biên càng giống elip.
    """
    n = len(pix)
    cx = sum(p[0] for p in pix) / n
    cy = sum(p[1] for p in pix) / n
    sxx = sum((p[0] - cx) ** 2 for p in pix) / n
    syy = sum((p[1] - cy) ** 2 for p in pix) / n
    sxy = sum((p[0] - cx) * (p[1] - cy) for p in pix) / n
    half = (sxx + syy) / 2
    root = math.sqrt(((sxx - syy) / 2) ** 2 + sxy ** 2)
    l1, l2 = half + root, half - root
    if l2 <= 1e-6 or not pts:
        return None
    th = 0.5 * math.atan2(2 * sxy, sxx - syy)
    ct, st = math.cos(th), math.sin(th)
    a, b = 2 * math.sqrt(l1), 2 * math.sqrt(l2)
    rs = []
    for x, y in pts:
        dx, dy = x - cx, y - cy
        u = (dx * ct + dy * st) / a
        v = (-dx * st + dy * ct) / b
        rs.append(math.hypot(u, v))
    mean = sum(rs) / len(rs)
    std = math.sqrt(sum((r - mean) ** 2 for r in rs) / len(rs))
    return std / mean, b / a, b


def classify(r):
    """
    Trả về (khóa tên hình, đa giác để vẽ). Khóa = None nghĩa là KHÔNG TÍNH.
    1) Vùng bị khuyết/lõm (mất góc)  -> bỏ
    2) Biên khớp elip: tròn (trục gần bằng nhau) / elip / quá dẹt -> bỏ
    3) Còn lại: đa giác, đặt tên theo số đỉnh (tam, tứ, ngũ, lục, thất...)
    """
    pix = r["pix"]
    pts = boundary_points(pix)
    hull = convex_hull(pts)
    if len(hull) < 3:
        return None, hull
    per = poly_perimeter(hull)
    # số pixel ước lượng nếu vùng đặc hoàn toàn theo bao lồi (công thức Pick)
    hull_pixels = poly_area(hull) + per / 2 + 1
    if len(pix) / hull_pixels < SOLIDITY_MIN:
        return None, hull

    es = ellipse_stats(pix, pts)
    if es is not None and es[0] <= ELLIPSE_CV + ELLIPSE_CV_PX / es[2]:
        ratio = es[1]
        if ratio >= CIRCLE_RATIO:
            return "Hinh tron", hull
        if ratio >= ELLIPSE_MIN_RATIO:
            return "Elip", hull
        return None, hull

    poly = approx_polygon(hull, max(2.0, 0.03 * per))
    # xóa đỉnh thừa gần thẳng hàng (góc rẽ nhỏ) do pixel hóa
    while len(poly) > 3:
        n = len(poly)
        turns = []
        for i in range(n):
            a, b, c = poly[i - 1], poly[i], poly[(i + 1) % n]
            h1 = math.atan2(b[1] - a[1], b[0] - a[0])
            h2 = math.atan2(c[1] - b[1], c[0] - b[0])
            d = abs(math.degrees(h2 - h1)) % 360
            turns.append(min(d, 360 - d))
        i = turns.index(min(turns))
        if turns[i] >= MIN_TURN:
            break
        poly = poly[:i] + poly[i + 1:]
    # gộp đỉnh thừa (góc nhọn bị bo tròn sinh đỉnh sát nhau)
    while len(poly) > 3:
        n = len(poly)
        sides = [math.dist(poly[i], poly[(i + 1) % n]) for i in range(n)]
        i = sides.index(min(sides))
        if sides[i] >= MERGE_RATIO * sum(sides):
            break
        j = (i + 1) % n
        mid = ((poly[i][0] + poly[j][0]) / 2, (poly[i][1] + poly[j][1]) / 2)
        poly = [mid if k == i else poly[k] for k in range(n) if k != j]

    n = len(poly)
    if n == 3:
        return "Tam giac", poly
    if n == 4:
        max_dev = 0
        for i in range(4):
            a, b, c = poly[i - 1], poly[i], poly[(i + 1) % 4]
            v1 = (a[0] - b[0], a[1] - b[1])
            v2 = (c[0] - b[0], c[1] - b[1])
            cosang = (v1[0] * v2[0] + v1[1] * v2[1]) / (math.hypot(*v1) * math.hypot(*v2) + 1e-9)
            ang = math.degrees(math.acos(max(-1, min(1, cosang))))
            max_dev = max(max_dev, abs(ang - 90))
        if max_dev > ANGLE_TOL:
            return "Tu giac", poly
        s = [math.dist(poly[i], poly[(i + 1) % 4]) for i in range(4)]
        w, h = (s[0] + s[2]) / 2, (s[1] + s[3]) / 2
        return ("Hinh vuong" if min(w, h) / max(w, h) >= SQUARE_RATIO else "Hinh chu nhat"), poly
    if n in POLY_KEY:
        return POLY_KEY[n], poly
    return None, poly


def get_font():
    """Thử nạp font có dấu tiếng Việt; không có thì dùng chữ không dấu."""
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "tahoma.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, 14), True
        except Exception:
            pass
    return ImageFont.load_default(), False


def run_pipeline(img, method):
    gray = to_gray(img)
    smooth = blur(gray)
    mag = edge_magnitude(smooth, method)
    t = otsu_threshold(mag)
    binary = binarize(mag, t)
    binary = dilate(binary)
    if method == "Laplacian":      # biên Laplacian hay bị hở 1-2 px -> giãn thêm 1 lần
        binary = dilate(binary)

    regions = find_regions(binary)
    result = img.copy()
    draw = ImageDraw.Draw(result)
    font, has_vn = get_font()
    counts = {k: 0 for k in VN_NAME}
    skipped = 0
    for r in regions:
        key, poly = classify(r)
        if key is None:
            skipped += 1          # quá méo / mất góc -> không tính
            continue
        counts[key] += 1
        col = COLORS[key]
        draw.line(poly + [poly[0]], fill=col, width=2)
        label = VN_NAME[key] if has_vn else key
        draw.text((r["x0"], max(0, r["y0"] - 18)), label, fill=col, font=font)

    gimg = Image.new("L", img.size)
    gimg.putdata([v for row in gray for v in row])
    eimg = Image.new("L", img.size)
    eimg.putdata([255 if v else 0 for row in binary for v in row])
    return gimg, eimg, result, counts, skipped


# ---------------------------------------------------------------- GIAO DIỆN
class App:
    def __init__(self, root):
        self.root = root
        root.title("Nhận diện hình dạng - Xử lý ảnh")
        self.img = None
        self.result = None

        bar = tk.Frame(root)
        bar.pack(pady=6)
        tk.Button(bar, text="Mở ảnh", command=self.open_image).pack(side="left", padx=4)
        self.method = ttk.Combobox(bar, values=list(KERNELS), state="readonly", width=10)
        self.method.current(0)
        self.method.pack(side="left", padx=4)
        tk.Button(bar, text="Xử lý", command=self.process).pack(side="left", padx=4)
        tk.Button(bar, text="Lưu kết quả", command=self.save).pack(side="left", padx=4)

        panels = tk.Frame(root)
        panels.pack()
        self.labels = []
        for title in ("Ảnh xám", "Biên (nhị phân)", "Kết quả"):
            f = tk.Frame(panels)
            f.pack(side="left", padx=4)
            tk.Label(f, text=title).pack()
            lb = tk.Label(f)
            lb.pack()
            self.labels.append(lb)

        self.info = tk.Label(root, text="", justify="left", font=("Arial", 11))
        self.info.pack(pady=6)

    def show(self, idx, pil_img):
        tk_img = ImageTk.PhotoImage(pil_img)
        self.labels[idx].configure(image=tk_img)
        self.labels[idx].image = tk_img  # giữ tham chiếu

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Ảnh", "*.png *.jpg *.jpeg *.bmp")])
        if path:
            self.img = load_rgb(path)
            self.show(2, self.img)

    def process(self):
        if self.img is None:
            messagebox.showwarning("Thiếu ảnh", "Hãy mở ảnh trước.")
            return
        self.info.config(text="Đang xử lý...")
        self.root.update()
        gimg, eimg, res, counts, skipped = run_pipeline(self.img, self.method.get())
        self.result = res
        self.show(0, gimg)
        self.show(1, eimg)
        self.show(2, res)
        lines = [f"{VN_NAME[k]}: {v}" for k, v in counts.items() if v > 0]
        txt = "Tìm thấy:  " + ("   |   ".join(lines) if lines else "không có hình nào")
        if skipped:
            txt += f"\n(Bỏ qua {skipped} vùng quá méo hoặc mất góc)"
        self.info.config(text=txt)

    def save(self):
        if self.result is None:
            messagebox.showwarning("Chưa có kết quả", "Hãy xử lý ảnh trước.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png"), ("JPG", "*.jpg")])
        if path:
            self.result.save(path)
            messagebox.showinfo("Đã lưu", path)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
