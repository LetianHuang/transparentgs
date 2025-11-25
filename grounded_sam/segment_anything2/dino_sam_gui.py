import cv2

import argparse
import os
import sys
import os
os.environ['CURL_CA_BUNDLE'] = ''
import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))


# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


# segment anything
from segment_anything import sam_model_registry, SamPredictor

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

class GroundingDINOSamGUI:
    def __init__(self, input_path, output_path, text_prompt, box_threshold=0.3, text_threshold=0.25, start_index=0, guiname="GroundingDINOSamGUI"):
        self.input_path = input_path
        self.output_path = output_path
        self.guiname = guiname

        self.text_prompt = text_prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        self.filenames = sorted(os.listdir(self.input_path))
        self.index = start_index

        self.load_sam()
        self.load_dino()

        self.set_img(os.path.join(self.input_path, self.filenames[self.index]))

    def load_sam(self):
        model_type = "vit_h"

        device = "cuda"

        sam = sam_model_registry[model_type](checkpoint=SAM_CHECKPOINT)
        sam.to(device=device)

        self.predictor = SamPredictor(sam)
    
    def load_dino(self):
        config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"

        self.dino = load_model(config_file, GROUNDED_CHECKPOINT, device="cuda")

    def set_img(self, imgpath):
        image = cv2.imread(imgpath)  
        self.img = image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.st_image = image
        self.predictor.set_image(image)

        image_pil, image = load_image(imgpath)

        self.dino_img = image
        self.dino_imgpil = image_pil


    def predict_points(self, input_point, input_label):
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        mask_input = logits[np.argmax(scores), :, :]
        masks, _, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            mask_input=mask_input[None, :, :],
            multimask_output=False,
        )
        return masks[0]
    
    def predict_grounding(self):
        image_pil = self.dino_imgpil
        text_prompt = self.text_prompt
        box_threshold = self.box_threshold
        text_threshold = self.text_threshold

        boxes_filt, pred_phrases = get_grounding_output(
            self.dino, self.dino_img, text_prompt, box_threshold, text_threshold, device="cuda"
        )

        index = pred_phrases.index(max(pred_phrases))
        boxes_filt = boxes_filt[index:index + 1]
        pred_phrases = pred_phrases[index:index + 1]

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes_filt, self.st_image.shape[:2]).to("cuda")

        return transformed_boxes
    
    def predict_grounded_points(self, input_point, input_label):
        transformed_boxes = self.predict_grounding()

        if len(input_point) == 0:
            # print(f"transformed_boxes: {transformed_boxes}")
            # tensor([[380.5636,  28.3196, 608.6262, 475.2975]], device='cuda:0')
            masks, _, _ = self.predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes,
                multimask_output = False,
            )
            return masks[0][0].detach().cpu().numpy()
        else:
            return self.predict_points(input_point, input_label)


    def start_gui(self):
        img = self.img.copy()
        coords = []
        labels = []
        mask = self.predict_grounded_points(np.array(coords), np.array(labels))
        mask = np.stack([mask, mask, mask], -1)
        mask = (mask * 255).astype(np.uint8)


        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                coords.append([x, y])
                labels.append(1) 
                cv2.drawMarker(img, (x, y), (0, 255, 0), markerType=cv2.MARKER_STAR, markerSize=20, thickness=2)
            elif event == cv2.EVENT_RBUTTONDOWN: 
                coords.append([x, y])
                labels.append(0) 
                cv2.drawMarker(img, (x, y), (0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=20, thickness=2)
            
            cv2.imshow(self.guiname, np.concatenate([img, mask], 1))

        cv2.namedWindow(self.guiname)
        cv2.setMouseCallback(self.guiname, mouse_callback)

        while True:
            cv2.imshow(self.guiname, np.concatenate([img, mask], 1))
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): # quit with the key 'q'
                break
            elif key == ord('m'): # segment with sam
                mask = self.predict_grounded_points(np.array(coords), np.array(labels))
                mask = np.stack([mask, mask, mask], -1)
                mask = (mask * 255).astype(np.uint8)
            elif key == ord('b') and len(coords) > 0: # rollback (point prompts)
                coords.pop()
                labels.pop()
                img = self.img.copy()
                for coord, label in zip(coords, labels):
                    x, y = coord
                    if label == 1:
                        cv2.drawMarker(img, (x, y), (0, 255, 0), markerType=cv2.MARKER_STAR, markerSize=20, thickness=2)
                    else:
                        cv2.drawMarker(img, (x, y), (0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=20, thickness=2)
            elif key == ord('s'): # save the mask
                cv2.imwrite(os.path.join(self.output_path, self.filenames[self.index]), mask)
            elif key == ord('n'): # get the next image
                self.index += 1
                self.set_img(os.path.join(self.input_path, self.filenames[self.index]))
                
                img = self.img.copy()
                coords = []
                labels = []
                mask = self.predict_grounded_points(np.array(coords), np.array(labels))
                mask = np.stack([mask, mask, mask], -1)
                mask = (mask * 255).astype(np.uint8)
            elif key == ord('p'): # get the previous image
                self.index -= 1
                self.set_img(os.path.join(self.input_path, self.filenames[self.index]))
                img = self.img.copy()
                mask = np.zeros_like(img)
                coords = []
                labels = []

        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str) # input images directory
    parser.add_argument('--output_path', type=str) # output masks directory
    parser.add_argument('--start_index', default=0, type=int) # the starting index of views to be segmented
    parser.add_argument('--text_prompt', type=str) # text prompt
    parser.add_argument('--box_threshold', default=0.3, type=float) # box threshold
    parser.add_argument('--text_threshold', default=0.25, type=float) # text threshold

    opt = parser.parse_args()

    os.makedirs(opt.output_path, exist_ok=True)

    GroundingDINOSamGUI(opt.input_path, opt.output_path, text_prompt=opt.text_prompt, box_threshold=opt.box_threshold, text_threshold=opt.text_threshold, start_index=opt.start_index).start_gui()