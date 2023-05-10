import glob
import os.path
import logging

from segment_anything import sam_model_registry, SamPredictor
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#using this script for masking photo
logging.basicConfig(filename="log/generate_masking_dataset.log", filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)


sam_checkpoint = "./checkpoint/sam_vit_b_01ec64.pth"
model_type = "vit_b"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device="cuda")
predictor = SamPredictor(sam)
input_point = np.array([[128, 150]])
input_label = np.array([1])


def main(source, target):

    source_files=glob.glob(f'{source}/Tomato*/*.JPG')
    total=len(source_files)
    success=0
    for img_path in source_files:
        try:
            pil_img=Image.open(img_path)
            np_img=np.array(pil_img)
            predictor.set_image(np_img)
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            path,base_name=os.path.split(img_path)
            kind=os.path.split(path)[1]
            path=os.path.join(target,kind)
            if not os.path.exists(path):
                os.makedirs(path)
            out_img=np.expand_dims(masks[0], axis=2)*np_img
            plt.imsave(os.path.join(path,base_name),out_img)
            success=success+1
        except:
            logging.warning(f'generate failed filename>>{img_path}')
            continue
    logging.info(f'total file num>>{total}\t, {success} successed')

if __name__ == '__main__':
    source='/media/xjw/ssk_data/plant/PlantVillage'
    target='/media/xjw/ssk_data/plant/plantvillage_masked'
    main(source,target)