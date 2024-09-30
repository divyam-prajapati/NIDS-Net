
import numpy as np
from PIL import Image
import cv2
import supervision as sv
from PIL import Image as PILImg
from absl import app, logging
import os
from pycocotools import mask as maskUtils
import json
from utils.data_utils import gen_square_crops


def annotate(image_source, boxes, logits, phrases):
    """
    Annotate image with bounding boxes, logits, and phrases.

    Parameters:
    - image_source (PIL.Image.Image): Input image source.
    - boxes (torch.tensor): Bounding boxes in xyxy format.
    - logits (list): List of confidence logits.
    - phrases (list): List of phrases.

    Returns:
    - PIL.Image: Annotated image.
    """
    try:
        detections = sv.Detections(xyxy=np.array(boxes))
        labels = [
            f"{phrase} {logit:.2f}"
            for phrase, logit
            in zip(phrases, logits)
        ]
        box_annotator = sv.BoxAnnotator(thickness=2)
        img_pil = PILImg.fromarray(box_annotator.annotate(scene=np.array(image_source), detections=detections, labels=labels))
        return img_pil
    
    except Exception as e:
        logging.error(f"Error during annotation: {e}")
        raise e
    
def area(xyxy):
    x1, y1, x2, y2 = xyxy
    return (x2-x1)*(y2-y1)

def get_object_proposal(image_path, image_array, bboxs, masks, tag="mask", ratio=1.0, save_rois=True, output_dir='object_proposals', save_segm=False, save_proposal=False):
    """
    Get object proposals from the image according to the bounding boxes and masks.

    @param image_path:
    @param bboxs: numpy array, the bounding boxes of the objects [N, 4]
    @param masks: Boolean numpy array of shape [N, H, W], True for object and False for background
    @param tag: use mask or bbox to crop the object
    @param ratio: ratio to resize the image
    @param save_rois: if True, save the cropped object proposals
    @param output_dir: the folder to save the cropped object proposals
    @return: the cropped object proposals and the object proposals information
    """
    # raw_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    raw_image = image_array
    image_height, image_width = raw_image.shape[:-1]
    scene_name = os.path.basename(image_path).split('.')[0]
    sel_rois = []
    rois = []
    cropped_masks = []
    cropped_imgs = []
    # ratio = 0.25
    if ratio != 1.0:
        scene_image = cv2.resize(raw_image, (int(raw_image.shape[1] * ratio), int(raw_image.shape[0] * ratio)),
                               cv2.INTER_LINEAR)
    else:
        scene_image = raw_image
    # scene_image = cv2.resize(raw_image, (int(raw_image.shape[1] * ratio), int(raw_image.shape[0] * ratio)),
    #                          cv2.INTER_LINEAR)
    for ind in range(len(masks)):
        # bbox
        x0 = int(bboxs[ind][0])
        y0 = int(bboxs[ind][1])
        x1 = int(bboxs[ind][2])
        y1 = int(bboxs[ind][3])

        # load mask
        mask = masks[ind]
        # Assuming `mask` is your boolean numpy array with shape (H, W)
        rle = None
        if save_segm:
            rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
            rle['counts'] = rle['counts'].decode('ascii')  # If saving to JSON, ensure counts is a string
        cropped_mask = mask[y0:y1, x0:x1]
        cropped_mask = Image.fromarray(cropped_mask.astype(np.uint8) * 255)
        cropped_masks.append(cropped_mask)
        # show mask
        cropped_img = raw_image[y0:y1, x0:x1]
        cropped_img = Image.fromarray(cropped_img)
        # cropped_img.show()
        # cropped_mask.show()
        # try masked image
        # cropped_mask_array = np.array(cropped_mask).astype(bool)
        # cropped_masked_img = cropped_img * cropped_mask_array[:, :, None]
        # cropped_img = Image.fromarray(cropped_masked_img)

        cropped_imgs.append(cropped_img)

        # save roi region
        if save_rois:
            # invert background to white
            new_image = Image.new('RGB', size=(image_width, image_height), color=(255, 255, 255))
            new_image.paste(Image.fromarray(raw_image), (0, 0),
                            mask=Image.fromarray(mask).resize((image_width, image_height)))
            if tag == "mask":
                roi = gen_square_crops(new_image, [x0, y0, x1, y1])  # crop by mask
            elif tag == "bbox":
                roi = gen_square_crops(Image.fromarray(raw_image), [x0, y0, x1, y1])  # crop by bbox
            else:
                ValueError("Wrong tag!")

            rois.append(roi)
            os.makedirs(os.path.join(output_dir, scene_name), exist_ok=True)
            roi.save(os.path.join(output_dir, scene_name, scene_name + '_' + str(ind).zfill(3) + '.png'))

        # save bbox
        sel_roi = dict()
        sel_roi['roi_id'] = int(ind)
        sel_roi['image_id'] = int(scene_name.split('_')[-1])
        sel_roi['bbox'] = [int(x0 * ratio), int(y0 * ratio), int((x1 - x0) * ratio), int((y1 - y0) * ratio)]
        sel_roi['area'] = np.count_nonzero(mask)
        # if you need segmentation mask, uncomment the following line
        # sel_roi['mask'] = mask  # boolean numpy array. H X W
        sel_roi['roi_dir'] = os.path.join(output_dir, scene_name, scene_name + '_' + str(ind).zfill(3) + '.png')
        sel_roi['image_dir'] = image_path
        sel_roi['image_width'] = scene_image.shape[1]
        sel_roi['image_height'] = scene_image.shape[0]
        if save_segm:
            sel_roi['segmentation'] = rle  # Add RLE segmentation
        sel_roi['scale'] = int(1 / ratio)
        sel_rois.append(sel_roi)
    if save_proposal:
        with open(os.path.join(output_dir, 'proposals_on_' + scene_name + '.json'), 'w') as f:
            json.dump(sel_rois, f)
    return rois, sel_rois, cropped_imgs, cropped_masks