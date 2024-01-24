import os
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from torchvision.transforms.functional import to_pil_image

# 指定类别数和模型路径
num_classes = 5
model_path = 'D:/pythondeeplearn/mask_rcnn/save_weights/model_49.pth'
image_folder = 'D:/pythondeeplearn/tif/images'
output_folder = 'D:/pythondeeplearn/tif/heatmaps'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 加载模型
model = maskrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)
checkpoint = torch.load(model_path)
model_state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
model.load_state_dict(model_state_dict)
model.eval()

# 修改模型以提取特定层的特征
class ModifiedMaskRCNN(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.backbone = original_model.backbone

    def forward(self, x):
        x = self.backbone(x)
        return x

modified_model = ModifiedMaskRCNN(model)

# 图像预处理
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 遍历文件夹中的所有图像
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    if os.path.isfile(image_path):
        # 加载图像
        original_image = Image.open(image_path)
        image_tensor = transform(original_image).unsqueeze(0)

        # 提取特征
        with torch.no_grad():
            features = modified_model(image_tensor)
        last_feature_map_key = list(features.keys())[-1]
        feature_map = features[last_feature_map_key][0, 0, :, :]

        # 处理特征图
        feature_map = feature_map.cpu().numpy()
        feature_map = (feature_map - np.min(feature_map)) / (np.max(feature_map) - np.min(feature_map))
        feature_map_resized = cv2.resize(feature_map, (original_image.width, original_image.height))
        heatmap = cv2.applyColorMap(np.uint8(255 * feature_map_resized), cv2.COLORMAP_JET)
        heatmap = to_pil_image(heatmap)
        superimposed_img = Image.blend(heatmap, original_image, alpha=0.6)

        # 保存热力图
        output_path = os.path.join(output_folder, image_name)
        superimposed_img.save(output_path)

print("All heatmaps have been saved to", output_folder)
