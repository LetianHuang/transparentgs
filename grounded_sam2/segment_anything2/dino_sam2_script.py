import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ['CURL_CA_BUNDLE'] = ''
import sys
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))
import shutil

import numpy as np
import cv2
import torch
from PIL import Image
import tqdm

# Segment Anything 2
from sam2.build_sam import build_sam2_video_predictor
# Segment Anything 1
from segment_anything import (
    sam_model_registry,
    SamPredictor
)

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

import argparse

SAM2_CHECKPOINT = "./checkpoints/sam2_hiera_large.pt"
MODEL_CFG = "sam2_hiera_l.yaml"

SAM_CHECKPOINT = "./checkpoints/sam_vit_h_4b8939.pth"
GROUNDED_CHECKPOINT = "./checkpoints/groundingdino_swint_ogc.pth"


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def mask2boxes(mask):
    ys, xs = np.where(mask)
    x_min, x_max, y_min, y_max = xs.min(), xs.max(), ys.min(), ys.max()
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

def get_grounding_boxes(input_dir, text_prompt, box_threshold=0.3, text_threshold=0.25, slices=None, is_refine=False):
    config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"

    model = load_model(config_file, GROUNDED_CHECKPOINT, device="cuda")
    sam_version = "vit_h"
    predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=SAM_CHECKPOINT).to("cuda"))
    
    results = []

    to_do = sorted(os.listdir(input_dir), key=lambda x : x if x[:-4].isdigit() else x[4:])
    # print(f"to_do: {to_do}")

    if slices is not None:
        to_do = to_do[slices]

    for idx, imagename in enumerate(tqdm.tqdm(to_do, desc="GroundingDINO: ")):
        # print(f"imagename: {imagename}")
        image_path = os.path.join(input_dir, imagename)

        # load image
        image_pil, image = load_image(image_path)
        # load model

        # run grounding dino model
        boxes_filt, pred_phrases = get_grounding_output(
            model, image, text_prompt, box_threshold, text_threshold, device="cuda"
        )

        ################ fine the maximum value and the corresponding index of phrases ################################
        index = pred_phrases.index(max(pred_phrases))
        boxes_filt = boxes_filt[index:index + 1]
        pred_phrases = pred_phrases[index:index + 1]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to("cuda")


        # It is strongly recommended to set `is_refine` True; otherwise, some unexpected issues with 'point scale' may occur.
        if is_refine:
            predictor.set_image(image) 
            masks, _, _ = predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes,
                multimask_output = False,
            )

            results += [(idx, mask2boxes(masks[0][0].detach().cpu().numpy()))]

            # print(f"transformed_boxes.detach().cpu().numpy()[0]: {transformed_boxes.detach().cpu().numpy()[0]}, results[-1]: {results[-1]}")
        else:
            results += [(idx, transformed_boxes.detach().cpu().numpy()[0])]

    print("finish grounding")
    print(f"results: {results}")
    return results

def step(input_path, output_path, text_prompt, box_threshold=0.3, text_threshold=0.25, slices=None, is_refine=False):
    input_dir = input_path
    raw_filenames = sorted(os.listdir(input_dir))
    video_dir = os.path.join(*os.path.split(input_path)[:-1], "xxxxxxxtempxxxxxxx")
    os.makedirs(video_dir, exist_ok=True)
    output_dir = output_path

    predictor = build_sam2_video_predictor(MODEL_CFG, SAM2_CHECKPOINT, device="cuda")

    for i, filename in enumerate(sorted(os.listdir(input_dir), key=lambda x : x if x[:-4].isdigit() else x[4:])):
        # print(f"filename: {filename}")
        if not os.path.exists(os.path.join(video_dir, f"{i:04}.jpg")):
            shutil.copyfile(os.path.join(input_dir, filename), os.path.join(video_dir, f"{i:04}.jpg"))
    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    # frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))


    inference_state = predictor.init_state(video_path=video_dir)

    predictor.reset_state(inference_state)

    grounding_boxes = get_grounding_boxes(video_dir, text_prompt=text_prompt, box_threshold=box_threshold, text_threshold=text_threshold, slices=slices, is_refine=is_refine)

    for ann_frame_idx, box in grounding_boxes:
        # print(f"ann_frame_idx: {ann_frame_idx}, box: {box}")
        # print(f"box: {box}")
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=4,
            box=box,
        )

    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    os.makedirs(output_dir, exist_ok=True)

    for out_frame_idx in range(0, len(frame_names)):
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            mask = out_mask[0, ...]
            img = Image.open(os.path.join(video_dir, frame_names[out_frame_idx]))
            img = np.array(img)
            img[~mask] = 0
            img = np.concatenate([img, mask[..., None] * 255], -1)
            Image.fromarray(mask).save(os.path.join(output_dir, raw_filenames[out_frame_idx]))
    
    shutil.rmtree(video_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str) # input images directory
    parser.add_argument('--output_path', type=str) # output masks directory
    parser.add_argument('--text_prompt', type=str) # text prompt
    parser.add_argument('--box_threshold', default=0.3, type=float) # box threshold
    parser.add_argument('--text_threshold', default=0.25, type=float) # text threshold

    # GroundingDINO slices
    parser.add_argument('--dino_start', default=0, type=int)
    parser.add_argument('--dino_stop', default=None, type=int)
    parser.add_argument('--dino_step', default=1, type=int)
    parser.add_argument('--is_refine', action="store_true", help="refine grounding-dino boxes")

    opt = parser.parse_args()

    # It is strongly recommended to set `is_refine` True; otherwise, some unexpected issues with 'point scale' may occur.
    opt.is_refine = True

    step(opt.input_path, opt.output_path, opt.text_prompt, opt.box_threshold, opt.text_threshold, slices=slice(opt.dino_start, opt.dino_stop, opt.dino_step), is_refine=opt.is_refine)