#!/usr/bin/env python3
"""
cw_ssim_ssim_simple_preprocess.py

Load the Olivetti faces, apply a simple per-image standardization
(zero mean, unit variance + rescale to [0,1]), then compute CW‑SSIM
and classic SSIM between I1 and four test pairs, displaying both
results in 2×2 montages.

Requires:
    pip install dtcwt scikit-learn numpy matplotlib scikit-image
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
import dtcwt
from skimage.metrics import structural_similarity as ssim


def preprocess(img):
    """
    Simple conventional preprocessing:
      1. zero-mean, unit-variance
      2. rescale to [0,1]
    """
    img = (img - img.mean()) / img.std()
    img = (img - img.min()) / (img.max() - img.min())
    return img


def compute_cwssim(img1, img2, cw_levels=4, cw_K=1e-8):
    """
    Compute the CW‑SSIM index between two preprocessed images via DT‑CWT.
    """
    t = dtcwt.Transform2d()
    c1 = t.forward(img1, nlevels=cw_levels)
    c2 = t.forward(img2, nlevels=cw_levels)
    scores = []
    for sb1, sb2 in zip(c1.highpasses, c2.highpasses):
        v1 = sb1.reshape(-1, sb1.shape[2])
        v2 = sb2.reshape(-1, sb2.shape[2])
        num = np.abs(np.sum(v1 * np.conj(v2), axis=1))
        den = np.sqrt(np.sum(np.abs(v1)**2, axis=1) * np.sum(np.abs(v2)**2, axis=1))
        scores.append(np.mean((num + cw_K) / (den + cw_K)))
    return float(np.mean(scores))


def compute_ssim(img1, img2):
    """
    Compute classic SSIM between two preprocessed images.
    """
    score, _ = ssim(
        img1, img2,
        full=True,
        data_range=img2.max() - img2.min(),
        multichannel=False
    )
    return float(score)


def make_montage(images, pairs, scores, title):
    """
    Draw a 2×2 montage of concatenated face pairs + their similarity scores.
    """
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle(title, fontsize=16)
    for idx, ((label, i, j), sc) in enumerate(zip(pairs, scores)):
        ax = axes.flat[idx]
        comp = np.hstack([images[i], images[j]])
        ax.imshow(comp, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        ax.set_title(f"{label}\nidxs ({i},{j})  score: {sc:.4f}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def main():
    # 1) Load Olivetti faces
    faces = fetch_olivetti_faces(shuffle=False)
    raw = faces.images   # (400,64,64)

    # 2) Preprocess all faces simply
    images = [preprocess(img) for img in raw]

    # 3) Define reference & test indices
    I1, I2 = 1, 3        # same person
    I3, I4 = 21, 34      # different persons
    pairs = [
        ("I1 vs I1", I1, I1),
        ("I1 vs I2", I1, I2),
        ("I1 vs I3", I1, I3),
        ("I1 vs I4", I1, I4),
    ]

    # 4) Compute both metrics
    cw_scores   = [compute_cwssim(images[i], images[j]) for _, i, j in pairs]
    ssim_scores = [compute_ssim(images[i], images[j])   for _, i, j in pairs]

    # 5) Plot montages
    cw_fig   = make_montage(images, pairs, cw_scores,
                            title="CW‑SSIM (simple preprocess)")
    ssim_fig = make_montage(images, pairs, ssim_scores,
                            title="Classic SSIM (simple preprocess)")

    # 6) Save & show
    cw_fig.savefig("cw_ssim_simple.png", dpi=150)
    ssim_fig.savefig("ssim_simple.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()