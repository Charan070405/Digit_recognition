# app_svm.py
import os
import uuid
import base64
from flask import Flask, request, render_template, jsonify, redirect, url_for, send_from_directory
from PIL import Image, ImageOps
import numpy as np
import joblib
import scipy.ndimage as ndi

app = Flask(__name__, static_folder="static", template_folder="templates")

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load PCA + SVM trained on MNIST (per-digit)
pca = joblib.load("pca_model.pkl")
svm = joblib.load("svm_model.pkl")


# -------------------------
# Helpers: MNIST-style preprocessing
# -------------------------
def center_and_resize_digit(img, size=28, target_box=20, pad=4):
    """
    img: PIL grayscale image (digit crop, white-on-black expected or inverted later)
    Resize to target_box x target_box preserving aspect ratio, then paste
    centered into `size x size` with `pad` margin (so target_box + 2*pad = size).
    """
    # ensure grayscale
    im = img.convert("L")
    arr = np.array(im)

    # Binarize mildly to find bounding content
    # If background is lighter than ink (means white bg with black ink), invert
    if arr.mean() > 127:
        im = ImageOps.invert(im)
        arr = np.array(im)

    # find content bbox
    nz = np.where(arr > 0)
    if nz[0].size == 0:
        # empty input: return blank 28x28
        return Image.new("L", (size, size), color=0)

    y_min, y_max = nz[0].min(), nz[0].max()
    x_min, x_max = nz[1].min(), nz[1].max()

    crop = im.crop((x_min, y_min, x_max + 1, y_max + 1))

    # resize crop so longest side == target_box
    w, h = crop.size
    if w > h:
        new_w = target_box
        new_h = max(1, int(round((target_box / w) * h)))
    else:
        new_h = target_box
        new_w = max(1, int(round((target_box / h) * w)))

    crop = crop.resize((new_w, new_h), Image.LANCZOS)

    # create new image and paste centered
    out = Image.new("L", (size, size), color=0)  # background black
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    out.paste(crop, (left, top))

    return out


def preprocess_for_model(pil_img):
    """
    Input: PIL Image (single-digit crop).
    Output: 1x784 numpy float32, ready for PCA & SVM.
    """
    # ensure white-on-black: after center_and_resize_digit result is black background, white digit
    img28 = center_and_resize_digit(pil_img, size=28, target_box=20, pad=4)
    arr = np.array(img28).astype("float32")
    # normalize to 0-255 as integers (model expects same scale as training)
    # flatten row-major
    return arr.reshape(1, -1)


# -------------------------
# Segmentation: connected components + merge small gaps
# -------------------------
def segment_digits(pil_image):
    """
    Input: PIL grayscale image (full input). Output: list of PIL images (digit crops),
    ordered left-to-right.
    Uses connected-component labeling to find strokes and then merges components that are close
    horizontally (so multi-stroke digits remain one).
    """
    gray = pil_image.convert("L")
    arr = np.array(gray)

    # Binarize: digit pixels should be 1
    thresh = arr.mean()
    binary = (arr < thresh).astype(np.uint8)  # assumes foreground darker? we'll handle white-on-black variant
    # If your canvas is white-on-black (white strokes on black background), do inverse:
    # detect whether foreground should be 1 by checking which polarity has more sparse pixels
    # If binary sum is huge (many ones), invert
    if binary.sum() > (binary.size * 0.5):
        binary = 1 - binary

    # label connected components
    labeled, num = ndi.label(binary)

    boxes = []
    for lab in range(1, num + 1):
        ys, xs = np.where(labeled == lab)
        if xs.size == 0 or ys.size == 0:
            continue
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        # filter tiny noise
        if w < 6 and h < 6:
            continue
        boxes.append((x1, y1, x2, y2))

    if not boxes:
        # no components: return full image as single digit
        return [pil_image.convert("L")]

    # sort by x (left-to-right)
    boxes = sorted(boxes, key=lambda b: b[0])

    # merge boxes that are very close horizontally (to combine multi-stroke digits)
    merged = []
    for box in boxes:
        if not merged:
            merged.append(box)
            continue
        px1, py1, px2, py2 = merged[-1]
        x1, y1, x2, y2 = box
        gap = x1 - px2
        # tolerance depends on widths: allow small gaps to be merged
        avgw = max( (px2-px1+1), (x2-x1+1) )
        if gap <= max(8, int(0.15 * avgw)):
            # merge
            nx1 = min(px1, x1)
            ny1 = min(py1, y1)
            nx2 = max(px2, x2)
            ny2 = max(py2, y2)
            merged[-1] = (nx1, ny1, nx2, ny2)
        else:
            merged.append(box)

    # Build PIL crops and sort left->right
    digits = []
    for (x1, y1, x2, y2) in merged:
        # expand a little padding
        pad = 4
        x0 = max(0, x1 - pad)
        y0 = max(0, y1 - pad)
        x3 = min(arr.shape[1] - 1, x2 + pad)
        y3 = min(arr.shape[0] - 1, y2 + pad)
        crop_arr = arr[y0:y3+1, x0:x3+1]
        # if original image had white-on-black strokes (i.e. high values), convert to 0/255 format:
        # we want white digit on black background for MNIST-like processing. We'll create PIL image accordingly:
        # In our binarization, digit pixels are 1 in binary variable; create image from binary to ensure correct polarity:
        # But crop_arr contains grayscale; we reconstruct from binary mask:
        mask = (crop_arr < thresh).astype(np.uint8) * 255
        pil_crop = Image.fromarray(mask).convert("L")
        digits.append(pil_crop)

    return digits


# -------------------------
# Flask routes
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/predict", methods=["POST"])
def predict():
    """Upload image: segment digits, predict per-digit, combine."""
    if "file" not in request.files:
        return redirect(url_for("index"))
    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("index"))

    ext = os.path.splitext(file.filename)[1]
    filename = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    pil_img = Image.open(save_path).convert("L")
    digit_crops = segment_digits(pil_img)

    preds = []
    for crop in digit_crops:
        x = preprocess_for_model(crop)
        x_pca = pca.transform(x)
        pred = int(svm.predict(x_pca)[0])
        preds.append(str(pred))

    number = "".join(preds) if preds else "N/A"

    return render_template("result.html", label=number, image_url=url_for("uploaded_file", filename=filename))


@app.route("/predict_canvas", methods=["POST"])
def predict_canvas():
    """Canvas: receive base64, save, segment, predict, return JSON."""
    data = request.form.get("image")
    if not data:
        return jsonify({"error": "No image"}), 400

    img_bytes = base64.b64decode(data.replace("data:image/png;base64,", ""))
    filename = f"{uuid.uuid4().hex}.png"
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(save_path, "wb") as f:
        f.write(img_bytes)

    pil_img = Image.open(save_path).convert("L")
    digit_crops = segment_digits(pil_img)

    preds = []
    for crop in digit_crops:
        x = preprocess_for_model(crop)
        x_pca = pca.transform(x)
        pred = int(svm.predict(x_pca)[0])
        preds.append(str(pred))

    number = "".join(preds) if preds else "N/A"

    return jsonify({"label": number, "image_url": url_for("uploaded_file", filename=filename)})


@app.route("/result_from_canvas")
def result_from_canvas():
    label = request.args.get("label", "N/A")
    image_url = request.args.get("image", "")
    return render_template("result.html", label=label, image_url=image_url)


if __name__ == "__main__":
    app.run(debug=True)
