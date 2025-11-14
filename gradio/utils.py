from transformers.image_utils import load_image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
from transformers import AutoImageProcessor, AutoModel
from huggingface_hub import login
import os
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m",token=os.environ.get("HF_TOKEN"))
model = AutoModel.from_pretrained(
    "facebook/dinov3-vits16-pretrain-lvd1689m",
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa",
    token=os.environ.get("HF_TOKEN")
)
model.eval()


def display_image(img, rows,cols):
    W, H = img.size
    patch_w = W / rows
    patch_h = H / cols

    plt.figure(figsize=(8,8))
    plt.imshow(img)

    # Draw vertical lines
    for i in range(1, rows):
        plt.axvline(i * patch_w, color='white', linestyle='--', linewidth=0.8)

    # Draw horizontal lines
    for i in range(1, cols):
        plt.axhline(i * patch_h, color='white', linestyle='--', linewidth=0.8)

    plt.axis('off')
    plt.show()


def get_patch_embeddings(img, ps=16, device="cuda"):
    inputs = processor(images=img, return_tensors="pt").to(device, torch.float16) # preprocessing for image include scaling, normalization etc
    B, C, H, W = inputs["pixel_values"].shape
    rows, cols = H // ps, W // ps # image of size 224x224, patch size = 16x16, hence image has 14x14 patches

    with torch.no_grad():
        out = model(**inputs)

    hs = out.last_hidden_state.squeeze(0).detach().cpu().numpy()

    # remove CLS + any non-patch token
    n_patches = rows * cols
    patch_embs = hs[-n_patches:, :].reshape(rows, cols, -1)

    # flatten and normalize
    X = patch_embs.reshape(-1, patch_embs.shape[-1])
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    return patch_embs, Xn, rows, cols # list of normalized patch vectors

def compute_patch_similarity(patch_embs, patch_embs_norm, row, col):
    rows, cols, dim = patch_embs.shape
    patch_idx = row * cols + col  # flatten index

    sim = patch_embs_norm @ patch_embs_norm[patch_idx] # cosine similarity via dot product
    sim_map = sim.reshape(rows, cols)
    sim_map = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min() + 1e-8)
    return sim_map

def overlay_similarity(img, sim_map, alpha=0.5, cmap="hot"):
    W, H = img.size

    # Expand sim_map (14x14) to full resolution via Kronecker upsampling
    sim_map_resized = np.kron(sim_map, np.ones((H // sim_map.shape[0], W // sim_map.shape[1])))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    ax.imshow(sim_map_resized, cmap=cmap, alpha=alpha)

    patch_w = W / sim_map.shape[1]
    patch_h = H / sim_map.shape[0]
    for i in range(1, sim_map.shape[1]):
        ax.axvline(i * patch_w, color='white', linestyle='--', linewidth=0.8)
    for i in range(1, sim_map.shape[0]):
        ax.axhline(i * patch_h, color='white', linestyle='--', linewidth=0.8)

    ax.axis('off')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    overlay_img = Image.open(buf)

    return overlay_img

# img = Image.open("two-cats.jpg")
# patch_embs,patch_embs_norm,rows,cols= get_patch_embeddings(img,ps=16, device=device)
# display_image(img,rows,cols)
# sim_map = compute_patch_similarity(patch_embs, patch_embs_norm, 7, 7)
# result_img = overlay_similarity(img,sim_map)
# plt.imshow(result_img)
# plt.savefig("overlay_result.png")
# plt.show()