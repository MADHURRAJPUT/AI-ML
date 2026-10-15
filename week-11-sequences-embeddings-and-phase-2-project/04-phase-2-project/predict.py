"""Phase 2 project — classify your own image files with a trained run.

    python predict.py --run resnet-fashion photo1.jpg photo2.png
    python predict.py --run resnet-fashion --demo        # uses a few test images instead

Real photos look nothing like Fashion-MNIST (centred, grey, dark background, 28x28) — the
preprocessing below tries to bridge that, and the mismatch is itself a lesson in distribution shift.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.torch_data import FASHION_CLASSES, image_tensors  # noqa: E402
from model import MODELS  # noqa: E402


def load_image(path):
    """Any image -> (1, 1, 28, 28) float in [0, 1], object bright on a dark background."""
    import numpy as np
    from PIL import Image, ImageOps

    img = ImageOps.grayscale(Image.open(path))
    background = img.getpixel((0, 0))                               # pad with the photo's own background
    img = ImageOps.pad(img, (max(img.size),) * 2, color=background)  # square, keep aspect ratio
    img = img.resize((28, 28), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32)).view(1, 1, 28, 28) / 255.0
    if x[..., :3, :].mean() > 0.5:                                  # light background -> invert
        x = 1.0 - x
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="resnet-fashion")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("images", nargs="*")
    args = ap.parse_args()
    run = HERE / "runs" / args.run
    cfg = json.loads((run / "config.json").read_text())
    model = MODELS[cfg["model"]]()
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu"))
    model.eval()
    classes = FASHION_CLASSES if cfg["dataset"] == "fashion" else [str(i) for i in range(10)]
    s = cfg["normalisation"]

    if args.demo:
        X, y = image_tensors(cfg["dataset"], train=False, limit=5)
        items = [(f"test image #{i} (true: {classes[y[i]]})", X[i:i + 1]) for i in range(5)]
    else:
        items = [(p, load_image(p)) for p in args.images]
    if not items:
        ap.error("give image paths or --demo")
    with torch.inference_mode():
        for name, x in items:
            p = F.softmax(model((x - s["mean"]) / s["std"]), 1)[0]
            top = p.topk(3)
            print(f"{name}: " + ", ".join(f"{classes[i]} {v:.2f}" for v, i in zip(top.values.tolist(),
                                                                                top.indices.tolist())))


if __name__ == "__main__":
    main()
