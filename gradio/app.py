from PIL import Image
import gradio as gr
from utils import get_patch_embeddings, compute_patch_similarity, overlay_similarity, device

selected_patch = {"row": 0, "col": 0}  

def init_states(img, alpha):
    if img is None:
        return gr.update(value=None), None
    patch_embs, patch_embs_norm, rows, cols = get_patch_embeddings(img, ps=16, device=device)
    
    sim_map = compute_patch_similarity(patch_embs, patch_embs_norm, 0, 0)
    result_img = overlay_similarity(img, sim_map, alpha=alpha, cmap="hot")

    state = {
        "img": img,
        "patch_embs": patch_embs,
        "patch_embs_norm": patch_embs_norm,
        "grid_size": rows,
        "alpha": alpha,
        "overlay_img":result_img,
    }

    return state, result_img

def store_patch(evt, state):
    if state is None or evt is None:
        return state

    rows = state["grid_size"]  
    cols = rows
    overlay_img = state["overlay_img"]
    overlay_W, overlay_H = overlay_img.size
    x_click, y_click = evt.index     # coordinates from click event

    # Map click coordinates to original patch grid
    col = int(x_click / overlay_W * cols)
    row = int(y_click / overlay_H * rows)

    # Clamp to valid range
    col = min(max(col, 0), cols - 1)
    row = min(max(row, 0), rows - 1)

    # Store in global or state dictionary
    selected_patch["row"] = row
    selected_patch["col"] = col


    return state


def reload_overlay(evt: gr.SelectData,state,alpha):
    if state is None:
        return None
    store_patch(evt, state)
    row, col = selected_patch["row"], selected_patch["col"]
    img = state["img"]
    patch_embs = state["patch_embs"]
    patch_embs_norm = state["patch_embs_norm"]
    sim_map = compute_patch_similarity(patch_embs, patch_embs_norm, row, col)
    result_img = overlay_similarity(img, sim_map, alpha=alpha, cmap="hot")
    return result_img

SAMPLE_IMAGES = {
    "Dog": "dog.jpg",
    "Cats": "cats.png",
    "Fruits": "fruits.png",
    "Brick Kiln": "brick-kiln.png",
    "People crossing street": "people.png",
    "Medical scan": "brain-tumor.png",
}

def load_sample(choice):
    if choice and choice in SAMPLE_IMAGES:
        return Image.open(SAMPLE_IMAGES[choice])
    return None

with gr.Blocks() as demo:
    state_store = gr.State()

    gr.Markdown("""
    <h1 style="font-size:36px; font-weight:bold;">Patch Similarity Visualizer</h1>
    <p style="font-size:18px;"> This tool visualizes similarity between image patches using DINOv3 embeddings. To use: <br>
     <ul style="font-size:18px;">
        <li>Upload an image in the <strong>left box</strong> or choose a sample image.</li>
        <li>Click anywhere in the <strong>right box</strong> to select a patch.</li>
        <li>View the similarity of the selected patch with all other patches in the image.</li>
    </ul>
    """)

    with gr.Row():
        img_choice = gr.Dropdown(
            choices=list(SAMPLE_IMAGES.keys()),
            label="Choose a sample image",
            value=None,
            interactive=True
        )

        alpha_slider = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.5,
            step=0.05,
            label="Overlay Transparency (alpha)",
            interactive=True
        )
    with gr.Row():
        img_input = gr.Image(
                type="pil",
                label="Or upload your own image"
            )
    
        output_img = gr.Image(
            type="pil",
            label="Similarity Overlay",
            interactive=True
        )

    img_choice.change(
        fn=load_sample,
        inputs=[img_choice],
        outputs=[img_input]
    )

    img_input.change(
        fn=init_states,
        inputs=[img_input, alpha_slider],
        outputs=[state_store, output_img]
    )

    output_img.select(
        fn=reload_overlay,
        inputs=[state_store, alpha_slider],
        outputs=[output_img]
    )


demo.launch()