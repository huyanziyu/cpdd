import os
import json
import torch
from PIL import Image
import torch.utils.data as data
import cv2
from pycocotools.coco import COCO
from train_utils import coco_remove_images_without_annotations, convert_coco_poly_mask
import pandas as pd  # 新增导入pandas

class CocoDetection(data.Dataset):
    def __init__(self, root, dataset="train", transforms=None, excel_path=None):
        super(CocoDetection, self).__init__()
        assert dataset in ["train", "val"], 'dataset must be in ["train", "val"]'
        anno_file = f"{dataset}.json"  # 根据您的文件命名修改
        self.img_root = os.path.join(root, "images")
        self.anno_path = os.path.join(root, "annotations", anno_file)

        assert os.path.exists(self.img_root), f"path '{self.img_root}' does not exist."
        assert os.path.exists(self.anno_path), f"file '{self.anno_path}' does not exist."

        self.transforms = transforms
        self.coco = COCO(self.anno_path)
        self.ids = list(sorted(self.coco.imgs.keys()))

        # 定义类别信息
        self.coco_classes = {
            0: "coal",
            1: "steel",
            2: "cement",
            3: "oil"
        }

        # 新增：加载统计特征
        if excel_path is not None:
            self.stat_features = pd.read_excel(excel_path).set_index('imageID').to_dict('index')
        else:
            self.stat_features = {}

    def parse_targets(self, img_id: int, coco_targets: list, w: int = None, h: int = None):
        assert w > 0
        assert h > 0

        # 只筛选出单个对象的情况
        anno = [obj for obj in coco_targets if obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]

        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # [xmin, ymin, w, h] -> [xmin, ymin, xmax, ymax]
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])

        segmentations = [obj["segmentation"] for obj in anno]
        masks = convert_coco_poly_mask(segmentations, h, w)

        # 筛选出合法的目标，即x_max>x_min且y_max>y_min
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        masks = masks[keep]
        area = area[keep]
        iscrowd = iscrowd[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["masks"] = masks
        target["image_id"] = torch.tensor([img_id])

        # for conversion to coco api
        target["area"] = area
        target["iscrowd"] = iscrowd

        return target

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(self.img_root, path)
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法加载图像: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        w, h, c = img.shape
        target = self.parse_targets(img_id, coco_target, w, h)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # 新增：获取统计特征
        stat_features = self.stat_features.get(img_id, {})
        stat_features_tensor = torch.tensor(list(stat_features.values()), dtype=torch.float32)

        return img, target, stat_features_tensor

    def __len__(self):
        return len(self.ids)

    def get_height_and_width(self, index):
        coco = self.coco
        img_id = self.ids[index]

        img_info = coco.loadImgs(img_id)[0]
        w = img_info["width"]
        h = img_info["height"]
        return h, w

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


if __name__ == '__main__':
    train = CocoDetection("/data/coco2017", dataset="train", excel_path="final_summary.xlsx.xlsx")
    print(len(train))
    t = train[0]
