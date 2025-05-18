import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

CONFIG_PATH = "configs/sam2.1/sam2.1_hiera_t.yaml"
CHECKPOINT_PATH = "checkpoints/sam2.1_hiera_tiny.pt"

# モデルは起動時に一度だけ読み込む
_model = build_sam2(CONFIG_PATH, CHECKPOINT_PATH)

def run_auto_mask(
    image,
    points_per_side,
    points_per_batch,
    pred_iou_thresh,
    stability_score_thresh,
    stability_score_offset,
    crop_n_layers,
    box_nms_thresh,
    crop_n_points_downscale_factor,
    min_mask_region_area,
    use_m2m
):
    mask_generator = SAM2AutomaticMaskGenerator(
        _model,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        stability_score_offset=stability_score_offset,
        crop_n_layers=crop_n_layers,
        box_nms_thresh=box_nms_thresh,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        min_mask_region_area=min_mask_region_area,
        use_m2m=use_m2m,
    )
    masks = mask_generator.generate(image)
    h, w = image.shape[:2]
    mask_color_img = np.zeros((h, w, 3), dtype=np.uint8)
    rng = np.random.default_rng(42)
    for m in masks:
        color = rng.integers(0, 256, size=3, dtype=np.uint8)
        seg = m['segmentation']
        mask_color_img[seg] = color
    return masks, Image.fromarray(mask_color_img)

def add_point(image, evt: gr.SelectData, points):
    if image is None or evt is None:
        return None, points
    new_points = points + [[evt.index[0], evt.index[1]]]
    img = Image.fromarray(image.astype(np.uint8)) if isinstance(image, np.ndarray) else image
    draw = ImageDraw.Draw(img)
    for x, y in new_points:
        r = 5
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0), outline=(255, 0, 0))
    return img, new_points

def reset_points():
    return [], None, None

def predict_mask(image, points, masks_state):
    if image is None or not points or not masks_state:
        return None
    masks = masks_state
    h, w = image.shape[:2]
    mask_color_img = np.zeros((h, w, 3), dtype=np.uint8)
    rng = np.random.default_rng(42)
    for i, pt in enumerate(points):
        x, y = int(pt[0]), int(pt[1])
        found = False
        for m in masks:
            seg = m['segmentation']
            if seg[y, x]:
                color = rng.integers(0, 256, size=3, dtype=np.uint8)
                mask_color_img[seg] = color
                found = True
                break
        if not found:
            pass
    return Image.fromarray(mask_color_img)

with gr.Blocks() as demo:
    gr.Markdown(
        """
# SAM2 Gradio Demo

画像をアップロード→「自動マスク推論」ボタンで全領域分割・プレビュー→クリック点推論で任意パーツ抽出

---

- 画像をアップロードした後、「自動マスク推論」ボタンでパーツ分割（test_run.pyと同様）
- 自動マスクの粒度や分割挙動はスライダーで細かく調整できます
- 2列目に自動マスクプレビュー（全パーツ色分け）が表示されます
- 画像上でクリックした点を含むパーツだけが色分けされます
- 複数点クリックで複数パーツを同時に抽出・色分けできます
- 「リセット」で選択点をクリアします

---
"""
    )
    points_state = gr.State([])
    masks_state = gr.State(None)
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="numpy", label="画像をアップロード")
        with gr.Column():
            auto_mask_img = gr.Image(type="pil", label="自動マスクプレビュー")
            points_per_side = gr.Slider(
                8, 128, value=32, step=1,
                label="points_per_side（サンプリング密度）",
                info="上げると細かく分割されるが計算が重くなる。下げると大きなパーツになりやすい"
            )
            points_per_batch = gr.Slider(
                1, 128, value=16, step=1,
                label="points_per_batch（バッチサイズ）",
                info="上げると高速化するがメモリ消費が増える"
            )
            pred_iou_thresh = gr.Slider(
                0.0, 1.0, value=0.6, step=0.01,
                label="pred_iou_thresh（IoU閾値）",
                info="上げると高品質なマスクのみ残る。下げると粗いマスクも残る"
            )
            stability_score_thresh = gr.Slider(
                0.0, 1.0, value=0.85, step=0.01,
                label="stability_score_thresh（安定度閾値）",
                info="上げると安定したマスクのみ残る。下げると不安定なマスクも残る"
            )
            stability_score_offset = gr.Slider(
                0.0, 1.0, value=0.7, step=0.01,
                label="stability_score_offset（安定度計算オフセット）",
                info="上げると安定度スコアが厳しくなる"
            )
            crop_n_layers = gr.Slider(
                0, 4, value=2, step=1,
                label="crop_n_layers（クロップ層数）",
                info="上げると細かい領域も分割されやすい"
            )
            box_nms_thresh = gr.Slider(
                0.0, 1.0, value=0.7, step=0.01,
                label="box_nms_thresh（NMS IoUしきい値）",
                info="上げると重複領域が減る。下げると重複が増える"
            )
            crop_n_points_downscale_factor = gr.Slider(
                1, 8, value=4, step=1,
                label="crop_n_points_downscale_factor（クロップごとの点数減衰）",
                info="上げるとクロップごとのサンプリング点が減り粗くなる"
            )
            min_mask_region_area = gr.Slider(
                0, 1000, value=5, step=1,
                label="min_mask_region_area（最小マスク面積）",
                info="上げると小さい領域が除外される"
            )
            use_m2m = gr.Checkbox(
                value=True,
                label="use_m2m（1ステップリファイン有効）",
                info="ONでマスクの品質が向上する場合がある"
            )
            auto_mask_btn = gr.Button("自動マスク推論")
        with gr.Column():
            preview_img = gr.Image(type="pil", label="クリック点プレビュー")
            infer_btn = gr.Button("推論（選択パーツのみ色分け）")
            reset_btn = gr.Button("リセット")
        with gr.Column():
            mask_img = gr.Image(type="pil", label="クリックマスクプレビュー")

    # 自動マスク推論ボタンでマスク生成＋プレビュー
    def on_auto_mask(
        image, points_per_side, points_per_batch, pred_iou_thresh, stability_score_thresh,
        stability_score_offset, crop_n_layers, box_nms_thresh, crop_n_points_downscale_factor,
        min_mask_region_area, use_m2m
    ):
        if image is None:
            return None, None
        masks, auto_mask_img = run_auto_mask(
            image, points_per_side, points_per_batch, pred_iou_thresh, stability_score_thresh,
            stability_score_offset, crop_n_layers, box_nms_thresh, crop_n_points_downscale_factor,
            min_mask_region_area, use_m2m
        )
        return masks, auto_mask_img

    auto_mask_btn.click(
        fn=on_auto_mask,
        inputs=[
            input_img, points_per_side, points_per_batch, pred_iou_thresh, stability_score_thresh,
            stability_score_offset, crop_n_layers, box_nms_thresh, crop_n_points_downscale_factor,
            min_mask_region_area, use_m2m
        ],
        outputs=[masks_state, auto_mask_img],
    )

    # 画像クリックで点追加＆プレビュー更新
    input_img.select(
        fn=add_point,
        inputs=[input_img, points_state],
        outputs=[preview_img, points_state],
    )

    # 推論ボタンで選択領域のみ色分け
    infer_btn.click(
        fn=predict_mask,
        inputs=[input_img, points_state, masks_state],
        outputs=mask_img,
    )

    # リセットボタンで座標リストとプレビュー初期化
    reset_btn.click(
        fn=reset_points,
        inputs=[],
        outputs=[points_state, preview_img, mask_img],
    )

if __name__ == "__main__":
    demo.launch()