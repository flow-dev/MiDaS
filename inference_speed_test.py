"""
python inference_speed_test.py \
    --variant dpt_hybrid \
    --precision float16
"""

import argparse

import cv2
import torch
from torchvision.transforms import Compose
from tqdm import tqdm

from midas.dpt_depth import DPTDepthModel
from midas.midas_net_custom import MidasNet_small
from midas.transforms import NormalizeImage, PrepareForNet, Resize

torch.backends.cudnn.benchmark = True

class InferenceSpeedTest:
    def __init__(self):
        self.parse_args()
        self.init_model()
        self.loop()
        
    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--variant', type=str, required=True, choices=['dpt_large', 'dpt_hybrid','midas_v21_small'])
        parser.add_argument('--model_weights', type=str, default=None) 
        parser.add_argument('--precision', type=str, default='float16')
        self.args = parser.parse_args()
        
    def init_model(self):
        self.device = 'cuda'
        self.precision = {'float32': torch.float32, 'float16': torch.float16}[self.args.precision]

        default_models = {
            "midas_v21_small": "weights/midas_v21_small-70d6b9c8.pt",
            "dpt_large": "weights/dpt_large-midas-2f21e586.pt",
            "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
        }

        if self.args.model_weights is None:
            self.args.model_weights = default_models[self.args.variant]
            model_path = self.args.model_weights

        if self.args.variant == "dpt_large": # DPT-Large
            self.model = DPTDepthModel(
                path=model_path,
                backbone="vitl16_384",
                non_negative=True,
            )
            self.net_w, self.net_h = 384, 384
            resize_mode = "minimal"
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        elif self.args.variant == "dpt_hybrid": #DPT-Hybrid
            self.model = DPTDepthModel(
                path=model_path,
                backbone="vitb_rn50_384",
                non_negative=True,
            )
            self.net_w, self.net_h = 384, 384
            resize_mode="minimal"
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif self.args.variant == "midas_v21_small":
            self.model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
            self.net_w, self.net_h = 256, 256
            resize_mode="upper_bound"
            normalization = NormalizeImage(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        else:
            print(f"model_type '{self.args.variant}' not implemented, use: --model_type large")
            assert False

        self.transform = Compose(
            [
                Resize(
                    self.net_w,
                    self.net_h,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method=resize_mode,
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                normalization,
                PrepareForNet(),
            ]
        )

        self.model = self.model.to(device=self.device, dtype=self.precision).eval()
        #self.model = torch.jit.script(self.model)
        #self.model = torch.jit.freeze(self.model)
    
    def loop(self):
        src = torch.randn((1, 3, self.net_h, self.net_w), device=self.device, dtype=self.precision)
        with torch.no_grad():
            for _ in tqdm(range(1000)):
                prediction = self.model.forward(src)
                torch.cuda.synchronize()

if __name__ == '__main__':
    InferenceSpeedTest()
