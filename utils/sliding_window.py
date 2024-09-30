import cv2
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from PIL import Image
import time
import torch
from torch import nn
from PIL import Image as PILImg
from tqdm import trange, tqdm
from utils.changed_fn import area, annotate, get_object_proposal
from utils.img_utils import masks_to_bboxes
from utils.inference_utils import compute_similarity, stableMatching, getColor, create_instances, nms, apply_nms, get_features

# Code from https://github.com/niconielsen32/tiling-window-detection/blob/main/tiling.py

# *Mine*
# Image Width: 8192 | Image Height:6144 | Slice Height: 2048 | Slice Width: 1536

def get_slice_bboxes(
    image_height: int,
    image_width: int,
    slice_height: Optional[int] = None,
    slice_width: Optional[int] = None,
    auto_slice_resolution: bool = True, 
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
) -> List[List[int]]:
    """
    This functions process the height and width of the image and create box's to slice the image.
    @Params:
        image_height: Height of the image
        image_width: Width of the image
        slice_height: Height of the slice
        slice_width: Width of the slice
        auto_slice_resolution: ---
        overlap_height_ratio: Overlaping area in height
        overlap_width_ratio: Overlaping area in width
    @Returns:
        slice_bboxes: List of box's to slice the image
    """

    slice_bboxes = []
    y_max = y_min = 0

    if slice_height and slice_width:  # Overlap height and width by 20%
        y_overlap = int(overlap_height_ratio * slice_height)
        x_overlap = int(overlap_width_ratio * slice_width)
    else:
        raise ValueError("Compute type is not auto and slice width and height are not provided.")

    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + slice_height
        while x_max < image_width:
            x_max = x_min + slice_width
            if y_max > image_height or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height, y_max)
                xmin = max(0, xmax - slice_width)
                ymin = max(0, ymax - slice_height)
                slice_bboxes.append([xmin, ymin, xmax, ymax])
            else:
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes


class SlicedImage: # Class to save the sliced image and top left corner cordinate of the image
    def __init__(self, image, starting_pixel):
        self.image = image
        self.starting_pixel = starting_pixel


class SliceImageResult: # Sliced Image Class
    def __init__(self, original_image_size: List[int], image_dir: Optional[str] = None):
        self.original_image_height = original_image_size[0]
        self.original_image_width = original_image_size[1]
        self.image_dir = image_dir

        self._sliced_image_list: List[SlicedImage] = []

    def add_sliced_image(self, sliced_image: SlicedImage):
        """
        Adds a sliced image object to the list
        """
        if not isinstance(sliced_image, SlicedImage):
            raise TypeError("sliced_image must be a SlicedImage instance")

        self._sliced_image_list.append(sliced_image)

    @property
    def sliced_image_list(self):
        """
        Returns a list of sliced images objects
        """
        return self._sliced_image_list

    @property
    def images(self):
        """
        Returns a list of sliced images
        """
        images = []
        for sliced_image in self._sliced_image_list:
            images.append(sliced_image.image)
        return images

    @property
    def starting_pixels(self) -> List[int]:
        """
        Returns a list of starting pixels of the sliced images
        """
        starting_pixels = []
        for sliced_image in self._sliced_image_list:
            starting_pixels.append(sliced_image.starting_pixel)
        return starting_pixels

    @property
    def filenames(self) -> List[int]:
        """
        Returns a list of filenames of the sliced images
        """
        filenames = []
        for sliced_image in self._sliced_image_list:
            filenames.append(sliced_image.coco_image.file_name)
        return filenames

    def __getitem__(self, i):
        """
        Returns a sliced image object result
        """
        def _prepare_ith_dict(i):
            return {
                "image": self.images[i],
                "starting_pixel": self.starting_pixels[i],
            }

        if isinstance(i, np.ndarray):
            i = i.tolist()

        if isinstance(i, int):
            return _prepare_ith_dict(i)
        elif isinstance(i, slice):
            start, stop, step = i.indices(len(self))
            return [_prepare_ith_dict(i) for i in range(start, stop, step)]
        elif isinstance(i, (tuple, list)):
            accessed_mapping = map(_prepare_ith_dict, i)
            return list(accessed_mapping)
        else:
            raise NotImplementedError(f"{type(i)}")

    def __len__(self):
        return len(self._sliced_image_list)


def slice_image(
    image: Union[str, Image.Image],
    slice_height: Optional[int] = None,
    slice_width: Optional[int] = None,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    auto_slice_resolution: bool = True,
    min_area_ratio: float = 0.1,
    out_ext: Optional[str] = None,
    verbose: bool = False,
) -> SliceImageResult:

    image_pil = image

    image_width, image_height = image_pil.size
    if not (image_width != 0 and image_height != 0):
        raise RuntimeError(f"invalid image size: {image_pil.size} for 'slice_image'.")
    slice_bboxes = get_slice_bboxes(
        image_height=image_height,
        image_width=image_width,
        auto_slice_resolution=auto_slice_resolution,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )

    n_ims = 0

    sliced_image_result = SliceImageResult(original_image_size=[image_height, image_width])

    image_pil_arr = np.asarray(image_pil)

    for slice_bbox in slice_bboxes:
        n_ims += 1

        tlx = slice_bbox[0]
        tly = slice_bbox[1]
        brx = slice_bbox[2]
        bry = slice_bbox[3]
        image_pil_slice = image_pil_arr[tly:bry, tlx:brx]

        slice_width = slice_bbox[2] - slice_bbox[0]
        slice_height = slice_bbox[3] - slice_bbox[1]

        sliced_image = SlicedImage(
            image=image_pil_slice, starting_pixel=[slice_bbox[0], slice_bbox[1]]
        )
        sliced_image_result.add_sliced_image(sliced_image)

    return sliced_image_result


def get_object_features_and_proposals(
        image_path, 
        all_windows, 
        threshold,
        slice_image_result, 
        gdino, 
        text_prompt, 
        SAM, 
        encoder,
        imsize,
        output_dir
    ):

    tag = "mask"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scene_features = []
    proposals = []
    width, height = PILImg.open(image_path).convert("RGB").size
    with torch.no_grad():
      for i, image_slice in tqdm(enumerate(slice_image_result)):

        window = PILImg.fromarray(image_slice['image'])
        start_x, start_y = image_slice['starting_pixel']
        bboxes, phrases, gdino_conf = gdino.predict(window, text_prompt)
        w, h = window.size
        xyxy = gdino.bbox_to_scaled_xyxy(bboxes, w, h).detach().cpu().numpy()
        # print(f"Window: {i} \nGdino bbox's: {xyxy.shape=}")
        # xyxy, masks = SAM.predict(window, xyxy)
        confs = gdino_conf.detach().cpu().numpy()
        torch.cuda.empty_cache()

        window_bboxs=[]
        
        # Threshold Check for bbox
        for k in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[k]

            s=area([x1, y1, x2, y2])
            b=area(all_windows[i])
            if int((s/b)*100)>threshold:
                continue
            window_bboxs.append([x1, y1, x2, y2])
        
        # Skipping Windows 
        if len(window_bboxs)==0:
            # print(f"Skipping this Window {i} as {len(window_bboxs)=} Bbox'x left after threshold")
            # print("==============================================================================================================================")
            continue

        # print(f"After threshold >> {len(window_bboxs)} Bbox's left")
        _, masks = SAM.predict(window, np.array(window_bboxs))
        masks = masks.squeeze(1)
        accurate_bboxs = masks_to_bboxes(masks)
        accurate_bboxs = torch.tensor(accurate_bboxs).cpu().numpy()
        masks=masks.cpu().numpy()
        # print(f"SAM output >> Accurate Bbox's Shape: {accurate_bboxs.shape} || masks Shape: {masks.shape}")

        ratio=0.25
        rois, sel_rois, cropped_imgs, cropped_masks = get_object_proposal(image_path, image_slice['image'], accurate_bboxs, masks, tag=tag, ratio=ratio, save_rois=False, output_dir=output_dir, save_proposal=False)

        # # rescaling bbox's
        for _ in range(len(sel_rois)):
            x1, y1, x2, y2 = accurate_bboxs[_]
            sel_rois[_]['bbox'] = [int((x1+start_x)*ratio), int((y1+start_y)*ratio), int((x2+start_x)*ratio)-int((x1+start_x)*ratio),  int((y2+start_y)*ratio)-int((y1+start_y)*ratio)]
            # print([int((x1+start_x)), int((y1+start_y)), int((x2+start_x)),  int((y2+start_y))])
   
        proposals.extend(sel_rois)

        # print(f"Object Proposal output >> croped imgs: {len(cropped_imgs)} || cropped masks: {len(cropped_masks)}")
        for j in range(len(cropped_imgs)):
            img = cropped_imgs[j]
            mask = cropped_masks[j]
            ffa_feature= get_features([img], [mask], encoder, device=device, img_size=imsize)
            scene_features.append(ffa_feature)
        
        # print(f"scene features after window {i}: {len(scene_features)}")
        # print(f"Proposals after window {i}: {len(proposals)}")
        # print("==============================================================================================================================")

    # print(f"scene features: {len(scene_features)} and each of shape: {scene_features[0].shape}")
    # print(f"Total running time: {end_time - start_time} seconds")

    for i, e in enumerate(proposals):
        e['roi_id'] = i
        e['scale'] = ratio
        e['image_height']=height*ratio
        e['image_width']=width*ratio

    scene_features = torch.cat(scene_features, dim=0)
    scene_features = nn.functional.normalize(scene_features, dim=1, p=2)
    # print("proposals (sel_rois): ", len(proposals))
    # print("scene features: ", scene_features.shape) # Shape (n, 1024)
    return scene_features, proposals