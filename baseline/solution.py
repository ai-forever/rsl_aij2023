import os

from tqdm import tqdm
from glob import glob
import pandas as pd
import numpy as np
import cv2
from mmaction.datasets.transforms.loading import DecordInit, SampleFrames, DecordDecode
from mmaction.datasets.transforms.processing import Resize, CenterCrop
from mmaction.datasets.transforms.formatting import FormatShape, PackActionInputs
from mmaction.datasets.transforms.wrappers import PytorchVideoWrapper
from mmaction.apis import inference_recognizer, init_recognizer
from mmengine.dataset import Compose
from mmcv.transforms import BaseTransform


CHECKPOINT = "mvit32.2_small.pth"
CONFIG = "mvit32.2_small_config.py"
DATASET_DIR = "dataset"
OUTPUT_FILE = "predicts.csv"
DEVICE = "cuda:0"


class SquarePadding(BaseTransform):
    def __init__(self, out_shape):
        self.out_shape = out_shape

    def transform(self, results):
        images = results['imgs']
        in_shape = results['img_shape']
        out_shape = self.out_shape
        padding = (int((out_shape[1] - in_shape[1]) / 2), int((out_shape[0] - in_shape[0]) / 2))
        pad_func = lambda x: cv2.copyMakeBorder(x, padding[1], padding[1], padding[0], padding[0], cv2.BORDER_CONSTANT, value=114)
        
        padded_images = [pad_func(img) for img in images]
        results['imgs'] = padded_images
        results['img_shape'] = out_shape
        return results


if __name__ == "__main__":
    videos = glob(os.path.join(DATASET_DIR, "*.mp4"))
    
    shape = (300, 300)

    test_pipeline = Compose([
        DecordInit(io_backend='disk'),
        SampleFrames(
            clip_len=32, 
            frame_interval=2,
            num_clips=1,
            test_mode=True,
            out_of_bound_opt='repeat_last'
        ),
        DecordDecode(),
        Resize(scale=shape),
        SquarePadding(out_shape=shape),
        CenterCrop(crop_size=224),
        FormatShape(input_format='NCTHW'),
        PackActionInputs(),
    ])
    
    model = init_recognizer(CONFIG, CHECKPOINT, device=DEVICE)
    model.eval()
    
    names = []
    predicts = []
    for video in tqdm(videos):
        name = os.path.basename(video).replace(".mp4", "")
        names.append(name)
        predicted = inference_recognizer(model, video, test_pipeline)
        predicted_class = int(predicted.pred_labels.item)
        predicts.append(predicted_class)

    result_df = pd.DataFrame.from_dict({"attachment_id": names, "class_indx": predicts})
    
    result_df.to_csv(OUTPUT_FILE, sep="\t", index=False)
