# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2025 Stefan Baumann et al., CompVis @ LMU Munich

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
from torchvision.utils import flow_to_image
import numpy as np
from einops import rearrange, repeat
import fire
import gradio as gr
from jaxtyping import UInt8, Float
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle, ConnectionPatch
from tqdm.auto import trange

from flow_poke.model import FlowPokeTransformer, FlowPokeTransformer_Base, make_axial_pos_2d, MixtureSameFamily


@dataclass
class ModelState:
    model: FlowPokeTransformer
    device: torch.device


@dataclass
class SampleState:
    image: UInt8[np.ndarray, "h w c"]
    d_img: dict[str, torch.Tensor]
    pokes: list[tuple[tuple[float, float], tuple[float, float]]] | None = field(default_factory=list)
    query: tuple[float, float] | None = None

    def add_point_ui(self, point: tuple[float, float]):
        if self.query is None:
            self.query = point
        else:
            self.pokes.append((self.query, point))
            self.query = None


def render_input(inputs: SampleState) -> UInt8[np.ndarray, "h w c"]:
    fig = plt.figure(figsize=(5, 5), frameon=False, dpi=(inputs.image.shape[0] / 5))
    ax = plt.gca()
    plt.imshow(inputs.image, extent=(0, 1, 1, 0), origin="upper", aspect="equal")
    plt.axis("off")
    plt.tight_layout()
    plt.margins(0, 0)
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)

    for poke in inputs.pokes:
        # If there is ~no movement, draw a circle to prevent an invisible arrow
        if np.linalg.norm(np.array(poke[0]) - np.array(poke[1]), ord=2) <= 0.01:
            arrow = Circle(xy=poke[0], radius=0.01, facecolor="black", edgecolor="white")
            ax.add_patch(arrow)
        else:
            arrow = FancyArrowPatch(
                poke[0],
                poke[1],
                arrowstyle="simple,head_length=0.8,head_width=0.8",
                facecolor="black",
                edgecolor="white",
                linewidth=1,
                mutation_scale=20,
                shrinkA=0,
                shrinkB=0,
            )
            ax.add_patch(arrow)

    if inputs.query is not None:
        ax.scatter(inputs.query[0], inputs.query[1], color="C1", marker="x", s=150, linewidths=3)

    fig.canvas.draw()
    img = np.array(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return img


@torch.no_grad()
def predict_flow(
    inputs: SampleState,
    model: ModelState,
    prediction_mode: Literal["parallel", "autoregressive"],
    flow_resolution: int = 256,
    ar_downsampling_factor: int = 8,
    enable_profiling: bool = False,
) -> UInt8[torch.Tensor, "h w c"]:
    assert inputs.query is None, "Query point should not be specified for flow prediction."

    # Prepare inputs
    if len(inputs.pokes) > 0:
        pokes: Float[torch.Tensor, "b l t c"] = torch.tensor(inputs.pokes, device=model.device)[None]
    else:
        pokes = torch.empty((1, 0, 2, 2), dtype=torch.float32, device=model.device)
    poke_pos: Float[torch.Tensor, "b l c"] = pokes[:, :, 0, :]
    poke_flow: Float[torch.Tensor, "b l c"] = pokes[:, :, 1, :] - pokes[:, :, 0, :]
    query_pos: Float[torch.Tensor, "b l c"] = make_axial_pos_2d(flow_resolution, flow_resolution, device=model.device)[
        None
    ]

    with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16):
        if prediction_mode == "parallel":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            pred: MixtureSameFamily = model.model.predict_parallel(
                poke_pos=poke_pos, poke_flow=poke_flow, query_pos=query_pos, camera_static=True, d_img=inputs.d_img
            )
            end.record()
            torch.cuda.synchronize()
            gr.Info(
                f"Delay: {start.elapsed_time(end):,.1f} ms ({math.prod(query_pos.shape[:-1]) / (start.elapsed_time(end) / 1e3):,.0f} queries / second)",
                duration=2,
            )
            flow: Float[torch.Tensor, "b h w c"] = rearrange(
                pred.mean, "b (h w) c -> b c h w", h=flow_resolution, w=flow_resolution
            )
        elif prediction_mode == "autoregressive":
            if ar_downsampling_factor != 1:
                ar_flow_resolution = flow_resolution // ar_downsampling_factor
                query_pos_ar = make_axial_pos_2d(
                    ar_flow_resolution,
                    ar_flow_resolution,
                    device=model.device,
                )[None]
            else:
                query_pos_ar = query_pos
            flow: Float[torch.Tensor, "b h w c"] = rearrange(
                model.model.predict_autoregressive(
                    poke_pos=poke_pos,
                    poke_flow=poke_flow,
                    query_pos=query_pos_ar,
                    camera_static=True,
                    d_img=inputs.d_img,
                    randomize_order=True,
                ),
                "b (h w) c -> b c h w",
                h=ar_flow_resolution,
                w=ar_flow_resolution,
            )
            # If we did downsampling for the AR part, upsample using parallel prediction for postprocessing
            if ar_downsampling_factor != 1:
                flow = rearrange(
                    model.model.predict_parallel(
                        poke_pos=torch.cat([poke_pos, query_pos_ar], dim=1),
                        poke_flow=torch.cat([poke_flow, rearrange(flow, "b c h w -> b (h w) c")], dim=1),
                        query_pos=query_pos,
                        camera_static=True,
                        d_img=inputs.d_img,
                    ).mean,
                    "b (h w) c -> b c h w",
                    h=flow_resolution,
                    w=flow_resolution,
                )
        else:
            raise ValueError(f"Invalid prediction mode: {prediction_mode}")

    return rearrange(flow_to_image(flow[0].float().cpu()), "c h w -> h w c").byte()


@torch.no_grad()
def predict_distribution(inputs: SampleState, model: ModelState) -> UInt8[torch.Tensor, "h w c"]:
    assert inputs.query is not None, "Query point should be specified for distribution prediction."

    # Prepare inputs
    if len(inputs.pokes) > 0:
        pokes: Float[torch.Tensor, "b l t c"] = torch.tensor(inputs.pokes, device=model.device)[None]
    else:
        pokes = torch.empty((1, 0, 2, 2), dtype=torch.float32, device=model.device)
    poke_pos: Float[torch.Tensor, "b l c"] = pokes[:, :, 0, :]
    poke_flow: Float[torch.Tensor, "b l c"] = pokes[:, :, 1, :] - pokes[:, :, 0, :]
    query_pos: Float[torch.Tensor, "b l c"] = torch.tensor(inputs.query, device=model.device)[None, None]

    with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        pred: MixtureSameFamily = model.model.predict_parallel(
            poke_pos=poke_pos, poke_flow=poke_flow, query_pos=query_pos, camera_static=True, d_img=inputs.d_img
        )
        end.record()
        torch.cuda.synchronize()
        gr.Info(f"Delay: {start.elapsed_time(end):,.1f} ms", duration=2)

    grid_resolution = 256

    # We want to compute a density grid from the predicted distribution
    # This can (in theory) be done exactly via the distribution's cdf
    # Unfortunately, there is no closed form for the cdf of a multivariate normal,
    # so we're using a numerical approximation that we implemented in the our distribution wrapper under the hood
    # Besides that, we're just computing the cdf for all corners of our grid cells we'd like to compute the density for
    # and then get the per-cell density via the difference
    def render_density_grid(
        dist,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        grid_resolution: int = grid_resolution,
        normalize: bool = False,
    ) -> Float[torch.Tensor, "grid_resolution grid_resolution"]:
        x_edges = torch.linspace(x_min, x_max, grid_resolution + 1, device=model.device)
        y_edges = torch.linspace(y_min, y_max, grid_resolution + 1, device=model.device)
        x_grid, y_grid = torch.meshgrid(x_edges, y_edges, indexing="ij")
        corners = torch.stack([x_grid.T.reshape(-1), y_grid.T.reshape(-1)], dim=-1)
        cdf_corners = dist.cdf(corners).reshape(grid_resolution + 1, grid_resolution + 1)

        # The probability mass in each cell is given via the difference of the cdf at the corners
        grid = (
            (cdf_corners[1:, 1:] - cdf_corners[:-1, 1:] - cdf_corners[1:, :-1] + cdf_corners[:-1, :-1])
            .clamp_min(0)
            .cpu()
        )
        if normalize:
            grid = grid / grid.sum()
        return grid

    # If you intend to use the rest of this method as a reference implementation, consider whether spending the time
    # it'll take you to understand the remainder of this method, which effectively is just plotting code, is worth it

    # Query the motion distribution on the motion grid that corresponds to all the points in the image
    x_min, x_max, y_min, y_max = 0 - inputs.query[0], 1 - inputs.query[0], 0 - inputs.query[1], 1 - inputs.query[1]
    m = render_density_grid(pred[0, 0], x_min, x_max, y_min, y_max, normalize=True)

    # Determine zoom-in bounds such that the majority of the mass is contained
    w_start = (
        max((m.sum(dim=0).cumsum(dim=0) >= 0.01).nonzero().min() - 1, 0) / grid_resolution * (x_max - x_min) + x_min
    )
    w_end = (
        min((m.sum(dim=0).cumsum(dim=0) >= 0.99).nonzero().min() + 2, grid_resolution)
        / grid_resolution
        * (x_max - x_min)
        + x_min
    )
    h_start = (
        max((m.sum(dim=1).cumsum(dim=0) >= 0.01).nonzero().min() - 1, 0) / grid_resolution * (y_max - y_min) + y_min
    )
    h_end = (
        min((m.sum(dim=1).cumsum(dim=0) >= 0.99).nonzero().min() + 2, grid_resolution)
        / grid_resolution
        * (y_max - y_min)
        + y_min
    )
    # Make sure to also include query point
    w_start, w_end = min(w_start, 0), max(w_end, 0)
    h_start, h_end = min(h_start, 0), max(h_end, 0)
    # Zoom in to smallest enclosing square
    w_center, h_center = (w_start + w_end) / 2, (h_start + h_end) / 2
    radius = max(w_end - w_start, h_end - h_start) / 2
    w_start, w_end = w_center - radius, w_center + radius
    h_start, h_end = h_center - radius, h_center + radius
    m_zoom = render_density_grid(pred[0, 0], w_start, w_end, h_start, h_end)

    n_components = pred.mixture_distribution.probs.size(-1)
    fig, axs = plt.subplots(
        nrows=n_components,
        ncols=3,
        figsize=(12, 5),
        dpi=200,
        frameon=False,
        gridspec_kw={
            "left": 0,
            "bottom": 0.025,
            "right": 1,
            "top": 0.9,
            "width_ratios": [1, 1, 2 / n_components],
            "hspace": 0,
            "wspace": 0.02,
        },
    )

    # Column 1: input image + density overlay
    gs = axs[0, 0].get_gridspec()
    for ax in axs[:, 0]:
        ax.remove()
    ax0 = fig.add_subplot(gs[:, 0])
    ax0.set_title("Prediction", fontsize=18)
    ax0.imshow(inputs.image, extent=[x_min, x_max, y_max, y_min])
    ax0.imshow(m, cmap="viridis", extent=[x_min, x_max, y_max, y_min], alpha=0.5)
    ax0.scatter(0, 0, color="C1", marker="x", s=50, linewidths=1.5)
    zoom_rect_kwargs = dict(fill=False, edgecolor="C2", linewidth=2, clip_on=False)
    rect_zoom = Rectangle(xy=(w_start, h_start), height=h_end - h_start, width=w_end - w_start, **zoom_rect_kwargs)
    ax0.add_patch(rect_zoom)
    ax0.axis("off")

    # Column 2: density zoom-in
    for ax in axs[:, 1]:
        ax.remove()
    ax1 = fig.add_subplot(gs[:, 1])
    ax1.set_title("Density Zoom-in", fontsize=18)
    ax1.imshow(m_zoom, cmap="viridis", extent=[w_start, w_end, h_end, h_start])
    ax1.scatter(0, 0, color="C1", marker="x", s=50, linewidths=1.5)
    rect_zoom_2 = Rectangle(xy=(w_start, h_start), height=h_end - h_start, width=w_end - w_start, **zoom_rect_kwargs)
    ax1.add_patch(rect_zoom_2)
    conn_patch_kwargs = dict(coordsA="data", coordsB="data", axesA=ax0, axesB=ax1, color="C2", linewidth=2)
    conn_1 = ConnectionPatch(xyA=(w_end, h_start), xyB=(w_start, h_start), **conn_patch_kwargs)
    ax1.add_artist(conn_1)
    conn_2 = ConnectionPatch(xyA=(w_end, h_end), xyB=(w_start, h_end), **conn_patch_kwargs)
    ax1.add_artist(conn_2)
    ax1.axis("off")

    # Column 3: density zoom-in for every mode/component, ordered by mixture weights
    i_cs = sorted(range(n_components), key=lambda i_c: pred[0, 0].mixture_distribution.probs[i_c], reverse=True)
    axs[0, 2].set_title("Modes", fontsize=18)
    for i, i_c in enumerate(i_cs):
        prob = pred[0, 0].mixture_distribution.probs[i_c].item()
        axs[i, 2].set_xlim(0, 1)
        axs[i, 2].set_ylim(0, 1)

        axs[i, 2].imshow(
            render_density_grid(
                pred[0, 0].component_distribution[i_c],
                w_start,
                w_end,
                h_start,
                h_end,
                grid_resolution=grid_resolution // 4,
            ),
            cmap="viridis",
            extent=[1, 2, 1, 0],
        )
        axs[i, 2].text(0.55, 0.5, f"{prob:.2f} ·", fontsize=18, ha="center", va="center")
        # Overlay with white relative to how likely the component is to visualize contribution
        axs[i, 2].imshow(np.ones((1, 1, 3)), extent=[0, 2, 1, 0], alpha=(1 - prob) ** 2, zorder=1000)

        rect_zoom_3 = Rectangle(xy=(1, 0), height=1, width=1, fill=False, edgecolor="C2", linewidth=2)
        axs[i, 2].add_patch(rect_zoom_3)

        axs[i, 2].set_xlim(0, 2)
        axs[i, 2].set_ylim(1, 0)
        axs[i, 2].set_aspect("equal")
        axs[i, 2].axis("off")

    fig.canvas.draw()
    m_vis = torch.from_numpy(np.array(fig.canvas.buffer_rgba())[..., :3])
    plt.close(fig)

    return m_vis.byte()


def predict(
    inputs: SampleState,
    model: ModelState,
    prediction_mode: Literal["parallel", "autoregressive"] = "parallel",
    flow_resolution: int = 256,
    ar_downsampling_factor: int = 8,
    enable_profiling: bool = False,
) -> UInt8[np.ndarray, "h w c"]:
    assert inputs.image is not None, "Image input is required for prediction."
    if inputs.query is None:
        img = predict_flow(
            inputs,
            model,
            flow_resolution=flow_resolution,
            prediction_mode=prediction_mode,
            ar_downsampling_factor=ar_downsampling_factor,
            enable_profiling=enable_profiling,
        )
    else:
        img = predict_distribution(inputs, model)
    return img.numpy()


def demo(
    checkpoint: str | None = None,  # Will be automatically downloaded from torch.hub() if not specified
    device: str = "cuda",
    compile: bool = False,  # Faster inference, at the cost of compilation time whenever a prediction config is first encountered
    warmup_compiled_paths: bool = False,
    # Gradio settings
    share: bool = False,
    server_name: str = "0.0.0.0",
    server_port: int = 55555,
):
    with gr.Blocks() as demo:
        gr.Markdown("## Flow Poke Transformer Demo")
        with gr.Row():
            with gr.Accordion("Instructions", open=True):
                gr.Markdown(
                    """
1. Upload an image in the first column (images get resized & center-cropped automatically)
2. Specify pokes & the query by clicking on the image (in the first column). Start by specifying all pokes by clicking on the start and end points.
    - If you want to predict the distribution for a query point, click on the position where the query should be placed.
    - If you want to perform a dense flow prediction, do not add a query point.
3. Click on the `Predict` button to predict the motion distribution of the query point. This might take longer for the first time due to compilation, but should be faster afterwards.
"""
                )

        if checkpoint is not None:
            model: FlowPokeTransformer = FlowPokeTransformer_Base()
            checkpoint = torch.load(checkpoint, weights_only=True)
            model.load_state_dict(checkpoint["model"])
        else:
            model = torch.hub.load(".", "fpt_base", source="local")
        model.eval()
        model.to(device)
        model.requires_grad_(False)

        model.transformer.reset_kv_cache(1, 128**2)
        if compile:
            model.image_embedder.model = torch.compile(model.image_embedder.model, dynamic=False)

            model.predict_parallel = torch.compile(model.predict_parallel, mode="reduce-overhead", dynamic=False)
            model.forward = torch.compile(model.forward, mode="reduce-overhead", dynamic=False, fullgraph=False)

            # Compile often-occurring paths ahead of time instead of making the user wait for compilation when hitting them for the first time
            # This has to happen in the same thread as the inference later for torch.compile() with cudagraphs to work (as of PyTorch ~2.8)
            # TODO: this seems to work somewhat, but also somewhat not. Look into how to make it work reliably.
            if warmup_compiled_paths:
                global warmed_up
                warmed_up = False

                @torch.no_grad()
                def warmup(*args, **kwargs):
                    global warmed_up
                    if warmed_up:
                        return
                    warmed_up = True
                    print("Warming up compiled paths...")
                    with torch.no_grad():
                        print("Warming up model.embed_image()...")
                        for _ in range(3):
                            _d_img = {
                                k: v.clone()
                                for k, v in model.embed_image(
                                    torch.randn((1, 3, 448, 448), device=device).clip(-1, 1)
                                ).items()
                            }
                        print("Warming up model.predict_parallel() for dense prediction...")
                        for n_p in trange(5, desc=f"Warming up model.predict_parallel()"):
                            for _ in range(3):
                                # Distribution prediction
                                model.predict_parallel(
                                    poke_pos=torch.randn((1, n_p, 2), device=device),
                                    poke_flow=torch.randn((1, n_p, 2), device=device),
                                    query_pos=torch.randn((1, 1, 2), device=device),
                                    camera_static=True,
                                    d_img=_d_img,
                                )
                                # Dense prediction
                                model.predict_parallel(
                                    poke_pos=torch.randn((1, n_p, 2), device=device),
                                    poke_flow=torch.randn((1, n_p, 2), device=device),
                                    query_pos=torch.randn((1, 256 * 256, 2), device=device),
                                    camera_static=True,
                                    d_img=_d_img,
                                )
                    print("Done warming up compiled paths.")

                demo.load(warmup)

        gr_model = gr.State(ModelState(model=model, device=torch.device(device)))

        with gr.Row():
            # Input & poke/query interaction handling
            with gr.Column():
                inputs = gr.State(None)
                image_input = gr.Image(
                    label="Input Image", interactive=True, type="numpy", image_mode="RGB", format="png"
                )

                @torch.no_grad()
                def preprocess_image(
                    image_input: UInt8[np.ndarray, "h w c"] | None, inputs: SampleState, model: ModelState
                ):
                    if image_input is None:
                        return None, None
                    H, W, C = image_input.shape
                    assert C == 3, f"Expected 3 channels in the image, got {C} channels."
                    target_size: int = model.model.image_embedder.image_size
                    if H != target_size or W != target_size:
                        # Center-crop & resize
                        min_size = min(H, W)
                        start_h = (H - min_size) // 2
                        start_w = (W - min_size) // 2
                        image_input = image_input[start_h : start_h + min_size, start_w : start_w + min_size, :]
                        image_input = np.array(
                            Image.fromarray(image_input).resize((target_size, target_size), Image.BICUBIC)
                        )
                    inputs = SampleState(
                        image=image_input,
                        # Embed image ahead of time, as this will not change after the upload
                        d_img={
                            k: v.clone()
                            for k, v in model.model.embed_image(
                                rearrange(torch.from_numpy(image_input), "h w c -> 1 c h w")
                                .float()
                                .div(127.5)
                                .sub(1.0)
                                .to(model.device)
                            ).items()
                        },
                    )
                    return image_input, inputs

                image_input.upload(
                    preprocess_image, inputs=[image_input, inputs, gr_model], outputs=[image_input, inputs]
                )

                def input_click(image_input, inputs: SampleState, evt: gr.SelectData):
                    if image_input is None:
                        return image_input, inputs
                    # (x, y) indexing
                    inputs.add_point_ui((evt.index[0] / image_input.shape[1], evt.index[1] / image_input.shape[0]))
                    return render_input(inputs), inputs

                image_input.select(input_click, inputs=[image_input, inputs], outputs=[image_input, inputs])

            # Output visualization
            with gr.Column():
                image_output = gr.Image(label="Prediction", type="numpy", image_mode="RGB", format="png")

                predict_button = gr.Button("Predict", variant="primary")
                with gr.Accordion("Advanced Settings"):
                    dense_resolution = gr.Dropdown(
                        label="Dense Prediction Resolution",
                        choices=[32, 64, 128, 256],
                        value=256,
                    )
                    dense_prediction_mode = gr.Dropdown(
                        label="Dense Prediction Mode",
                        choices=["parallel", "autoregressive"],
                        value="parallel",
                    )
                    ar_downsampling_factor = gr.Dropdown(
                        label="AR Downsampling Factor",
                        info="Determines at which resolution the autoregressive prediction is performed, after which it is upsampled using parallel predictions for efficiency.",
                        choices=[1, 2, 4, 8, 16, 32, 64],
                        value=8,
                    )
                predict_button.click(
                    predict,
                    inputs=[inputs, gr_model, dense_prediction_mode, dense_resolution, ar_downsampling_factor],
                    outputs=image_output,
                    show_progress=True,
                )

    demo.launch(share=share, server_name=server_name, server_port=server_port, debug=True)


if __name__ == "__main__":
    # Allow TF32, make model go brrrrrrrrrr
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Enable benchmarking for matmuls
    torch.backends.cudnn.benchmark = True
    # Increase the compilation cache size, so that we don't stop compiling when hitting many different paths
    torch._dynamo.config.cache_size_limit = max(2**12, torch._dynamo.config.cache_size_limit)

    # This seems to be crucial for compilation with cudagraphs to be stable (as of torch ~2.8) with gradio's multithreading
    # If you do inference using a normal script, leave these enabled, as it'll likely give you slight speedups
    # Also, if you run out of memory, consider enabling these again; as it might help with that, although you might pay
    # with the aforementioned stability issues
    torch._inductor.config.triton.cudagraph_trees = False

    fire.Fire(demo)
