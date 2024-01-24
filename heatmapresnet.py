import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from torchvision.transforms.functional import to_pil_image
import os

# 加载ResNet50预训练模型
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-2]))  # 移除最后的全连接层和平均池化层
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 文件夹路径
image_folder = 'D:/pythondeeplearn/tif/images'
output_folder = 'D:/pythondeeplearn/tif/resnet_heatmaps'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历文件夹中的所有图像
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    if os.path.isfile(image_path):
        # 加载和预处理图像
        original_image = Image.open(image_path).convert('RGB')
        image_tensor = transform(original_image).unsqueeze(0)

        # 获取特征图
        with torch.no_grad():
            features = model(image_tensor)

        # 获取最后一个卷积层的输出
        feature_map = features[0].mean(0).cpu().numpy()

        # 将特征图转换为numpy数组并归一化
        feature_map = (feature_map - np.min(feature_map)) / (np.max(feature_map) - np.min(feature_map))

        # 调整热力图的尺寸以匹配原始图像
        feature_map_resized = cv2.resize(feature_map, (original_image.width, original_image.height))

        # 将热力图转换为伪彩色图像
        heatmap = cv2.applyColorMap(np.uint8(255 * feature_map_resized), cv2.COLORMAP_JET)

        # 将伪彩色图像转换为PIL图像
        heatmap = to_pil_image(heatmap)

        # 将热力图叠加到原始图像上
        superimposed_img = Image.blend(heatmap, original_image, alpha=0.6)

        # 保存热力图
        output_path = os.path.join(output_folder, image_name)
        superimposed_img.save(output_path)

print("All ResNet heatmaps have been saved to", output_folder)
