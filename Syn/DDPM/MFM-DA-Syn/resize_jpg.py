from PIL import Image
import os

# 设置文件夹路径
folder_path = '/ailab/public/pjlab-smarthealth03/leiwenhui/jjx/Syn/DDPM/few-shot-diffusion-master/REFUGE_Test_390'

# 获取文件夹内的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg'):
        # 获取图片的完整路径
        file_path = os.path.join(folder_path, filename)
        
        # 打开图片并调整大小
        with Image.open(file_path) as img:
            img_resized = img.resize((224, 224))
            
            # 保存调整后的图片，覆盖原图或保存为新文件
            img_resized.save(file_path)

print("所有图片已调整为224x224。")
