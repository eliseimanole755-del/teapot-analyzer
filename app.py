from flask import Flask, request, jsonify
import io, requests, numpy as np, cv2
from PIL import Image

app = Flask(__name__)

def fetch_img(url):
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")

def to_bin(im, size=168):
    im = im.resize((size, size), Image.BILINEAR)
    g  = np.array(im.convert("L"), dtype=np.float32)
    thr = 0.9 * g.mean()
    bw  = (g < thr).astype(np.uint8)
    return bw

def draw_mask(size=64, stroke=None, cw=True):
    if stroke is None: stroke = max(8, int(size*0.18))
    img = np.zeros((size,size), np.uint8)
    cx, cy = size//2, size//2
    L, hook = int(size*0.30), int(size*0.22)
    t = stroke
    img[cy-L:cy+L, cx-t//2:cx+t//2] = 1
    img[cy-t//2:cy+t//2, cx-L:cx+L] = 1
    if cw:
        img[cy-L-t//2:cy-L+t//2, cx:cx+hook] = 1
        img[cy:cy+hook, cx+L-t//2:cx+L+t//2] = 1
        img[cy+L-t//2:cy+L+t//2, cx-hook:cx] = 1
        img[cy-hook:cy, cx-L-t//2:cx-L+t//2] = 1
    else:
        img[cy-L-t//2:cy-L+t//2, cx-hook:cx] = 1
        img[cy:cy+hook, cx-L-t//2:cx-L+t//2] = 1
        img[cy+L-t//2:cy+L+t//2, cx:cx+hook] = 1
        img[cy-hook:cy, cx+L-t//2:cx+L+t//2] = 1
    return img

def make_masks():
    base = [draw_mask(64, cw=True), draw_mask(64, cw=False)]
    outs = []
    for b in base:
        for angle in [0, 90]:
            M = cv2.getRotationMatrix2D((32,32), angle, 1.0)
            rot = cv2.warpAffine(b*255, M, (64,64), flags=cv2.INTER_NEAREST)
            for s in [0.9, 1.15]:
                sz = max(18, int(64*s))
                outs.append(cv2.resize(rot, (sz,sz), interpolation=cv2.INTER_NEAREST)//255)
    return outs

MASKS = make_masks()

def match_score(bw, mask, stride=6):
    H, W = bw.shape
    mH, mW = mask.shape
    if mH>=H or mW>=W: return 0.0
    ones = mask.sum() or 1
    best = 0.0
    for y in range(0, H-mH+1, stride):
        for x in range(0, W-mW+1, stride):
            patch = bw[y:y+mH, x:x+mW]
            hit = (patch & mask).sum()
            fp  = (patch & (1-mask)).sum()
            s = (hit/ones) - 0.25*(fp/ones)
            if s > best:
                best = s
                if best >= 0.74:
                    return float(best)
    return float(best)

@app.route("/teapot/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json(force=True)
        items = data.get("items", [])
    except Exception:
        return jsonify({"error":"bad json"}), 400

    results=[]
    for it in items:
        _id = str(it.get("id",""))
        url = it.get("url","")
        try:
            im = fetch_img(url)
            bw = to_bin(im, 168)
            best = 0.0
            for m in MASKS:
                s = match_score(bw, m)
                if s > best: best = s
                if best >= 0.74: break
            results.append({"id": _id, "suspect": best>=0.74, "score": round(best, 3)})
        except Exception:
            results.append({"id": _id, "suspect": False, "score": 0.0, "error":"download_or_process_failed"})

    resp = jsonify({"results": results})
    resp.headers["Access-Control-Allow-Origin"]  = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp

@app.route("/")
def home():
    return "Teapot Analyzer API is running."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
