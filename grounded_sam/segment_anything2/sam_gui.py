import numpy as np
import cv2
import os
import argparse

from segment_anything import sam_model_registry, SamPredictor

class SamGUI:
    def __init__(self, input_path, output_path, start_index=0, guiname="SamGUI"):
        self.input_path = input_path
        self.output_path = output_path
        self.guiname = guiname

        self.filenames = sorted(os.listdir(self.input_path))
        self.index = start_index

        self.load_sam()
        self.set_img(os.path.join(self.input_path, self.filenames[self.index]))

    def load_sam(self):
        sam_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"
        model_type = "vit_h"

        device = "cuda"

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.predictor = SamPredictor(sam)

    def set_img(self, imgpath):
        image = cv2.imread(imgpath)  
        self.img = image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image)

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

    def start_gui(self):
        img = self.img.copy()
        mask = np.zeros_like(img)
        coords = []
        labels = []

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
            elif key == ord('m') and len(coords) > 0: # segment with sam
                mask = self.predict_points(np.array(coords), np.array(labels))
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
                mask = np.zeros_like(img)
                coords = []
                labels = []
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

    opt = parser.parse_args()

    SamGUI(opt.input_path, opt.output_path, opt.start_index).start_gui()