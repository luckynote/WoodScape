import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob
import time
import json
from models.resnet import ResnetEncoder
from models.semantic_decoder import SemanticDecoder

class SemanticInference:
    def __init__(self, model_dir, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_dir = model_dir
        self.models = {}
        self.args = self.create_dummy_args()
        self.load_models()
        
        # 定义ROI区域 (left, top, right, bottom)
        self.roi = (114, 110, 1176, 610)
        
        # 语义分割类别颜色编码
        self.color_encoding = self.semantic_color_encoding()
        self.alpha = 0.5  # 渲染透明度
    
    def semantic_color_encoding(self):
        """创建语义分割类别颜色编码"""
        # 这里使用与训练代码相同的颜色编码
        return {
            0: [0, 0, 0],        # 背景 - 黑色
            1: [255, 0, 0],      # 类别1 - 红色
            2: [0, 255, 0],      # 类别2 - 绿色
            3: [0, 0, 255],      # 类别3 - 蓝色
            4: [255, 255, 0],    # 类别4 - 黄色
            5: [255, 0, 255],    # 类别5 - 紫色
            6: [0, 255, 255],    # 类别6 - 青色
            7: [128, 0, 0],      # 类别7 - 深红
            8: [0, 128, 0],      # 类别8 - 深绿
            9: [0, 0, 128]       # 类别9 - 深蓝
        }
    
    def create_dummy_args(self):
        """创建推理所需的虚拟参数"""
        class Args:
            def __init__(self):
                self.network_layers = 18  # 修改为18层
                self.semantic_num_classes = 10  # 语义分割类别数
                
        return Args()
    
    def remove_module_prefix(self, state_dict):
        """移除权重键名中的'module.'前缀"""
        new_state_dict = {}
        for key, value in state_dict.items():
            # 移除'module.'前缀
            if key.startswith('module.'):
                new_key = key[7:]
            else:
                new_key = key
            new_state_dict[new_key] = value
        return new_state_dict
    
    def filter_model_keys(self, state_dict, model):
        """过滤掉不在模型中的额外键"""
        model_keys = set(model.state_dict().keys())
        state_dict_keys = set(state_dict.keys())
        
        # 找出不在模型中的额外键
        extra_keys = state_dict_keys - model_keys
        if extra_keys:
            print(f"发现额外键: {extra_keys}, 将被过滤掉")
        
        # 只保留模型需要的键
        filtered_state_dict = {}
        for key in state_dict_keys:
            if key in model_keys:
                filtered_state_dict[key] = state_dict[key]
        
        return filtered_state_dict
    
    def load_models(self):
        """加载所有预训练模型"""
        # 1. 加载编码器模型
        self.models["encoder"] = ResnetEncoder(
            num_layers=self.args.network_layers, 
            pretrained=False
        ).to(self.device)
        
        encoder_path = os.path.join(self.model_dir, "encoder.pth")
        encoder_state = torch.load(encoder_path, map_location=self.device)
        encoder_state = self.remove_module_prefix(encoder_state)
        
        # 过滤掉额外的键（如"height", "width"）
        encoder_state = self.filter_model_keys(encoder_state, self.models["encoder"])
        
        self.models["encoder"].load_state_dict(encoder_state)
        self.models["encoder"].eval()
        
        # 2. 加载语义分割解码器模型
        self.models["semantic"] = SemanticDecoder(
            self.models["encoder"].num_ch_enc,
            n_classes=self.args.semantic_num_classes
        ).to(self.device)
        
        semantic_path = os.path.join(self.model_dir, "semantic.pth")
        semantic_state = torch.load(semantic_path, map_location=self.device)
        semantic_state = self.remove_module_prefix(semantic_state)
        
        # 过滤掉额外的键
        semantic_state = self.filter_model_keys(semantic_state, self.models["semantic"])
        
        self.models["semantic"].load_state_dict(semantic_state)
        self.models["semantic"].eval()
        
        print(f"语义分割模型已从 {self.model_dir} 加载")
    
    def preprocess_image(self, image_path):
        """预处理输入图像，包括ROI裁剪"""
        # 打开图像
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # 保存原始尺寸
        
        # 裁剪ROI区域 (left, top, right, bottom)
        roi_image = image.crop(self.roi)
        roi_size = roi_image.size  # 保存ROI尺寸
        
        # WoodScape的标准预处理
        transform = transforms.Compose([
            transforms.Resize((288, 544)),  # 调整到模型输入尺寸
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], 
            #                      std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(roi_image).unsqueeze(0)  # 增加batch维度
        return input_tensor.to(self.device), original_size, roi_size, image
    
    def predict_semantic(self, input_tensor):
        """执行语义分割推理"""
        with torch.no_grad():
            # 通过编码器提取特征
            features = self.models["encoder"](input_tensor)
            
            # 通过语义分割解码器预测
            outputs = self.models["semantic"](features)
            
            # 获取最高分辨率的语义分割图 (scale=0)
            semantic_output = outputs["semantic", 0]
            
            # 获取每个像素的类别索引
            _, predictions = torch.max(semantic_output.data, 1)
            
        return predictions.squeeze(0).cpu()  # 移除batch维度
    
    def postprocess_semantic(self, predictions, original_size, roi_size, original_image):
        """后处理语义分割结果"""
        # 转换为numpy数组
        predictions_np = predictions.byte().numpy()
        
        # 调整回ROI尺寸
        predictions_roi = np.array(Image.fromarray(predictions_np).resize(
            roi_size, Image.NEAREST))
        
        # 创建全尺寸分割图（与原始图像相同大小）
        full_predictions = np.zeros((original_size[1], original_size[0]), dtype=np.uint8)
        
        # 将ROI分割图放置到正确位置
        left, top, right, bottom = self.roi
        full_predictions[top:bottom, left:right] = predictions_roi
        
        # 创建彩色渲染图
        color_image = np.array(original_image)
        for class_id, color in self.color_encoding.items():
            # 找到属于该类别的像素
            mask = full_predictions == class_id
            # 将原图对应位置的颜色与类别颜色混合
            color_image[mask] = (color_image[mask] * (1 - self.alpha) + 
                                 np.array(color) * self.alpha).astype(np.uint8)
        
        return full_predictions, color_image
    
    def save_results(self, predictions, color_image, output_dir, image_name):
        """保存分割结果和渲染图"""
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存原始分割图（单通道）
        seg_path = os.path.join(output_dir, f"{image_name}_seg.png")
        Image.fromarray(predictions).save(seg_path)
        
        # 保存彩色渲染图
        render_path = os.path.join(output_dir, f"{image_name}_render.png")
        Image.fromarray(color_image).save(render_path)
        
        return seg_path, render_path
    
    def process_single_image(self, image_path, output_dir):
        """处理单张图像"""
        # 预处理图像（包括ROI裁剪）
        input_tensor, original_size, roi_size, original_image = self.preprocess_image(image_path)
        
        # 执行语义分割推理
        start_time = time.time()
        predictions = self.predict_semantic(input_tensor)
        inference_time = time.time() - start_time
        print(f"语义分割推理完成，耗时: {inference_time:.4f}秒")
        
        # 后处理分割结果
        predictions, color_image = self.postprocess_semantic(
            predictions, original_size, roi_size, original_image)
        
        # 提取图像名称
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 保存结果
        seg_path, render_path = self.save_results(predictions, color_image, output_dir, image_name)
        print(f"分割结果已保存至: {seg_path}")
        print(f"渲染结果已保存至: {render_path}")
        
        return predictions, color_image
    
    def process_image_directory(self, input_dir, output_dir):
        """处理整个目录的图像"""
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有支持的图像格式
        image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(input_dir, f'*.{ext}')))
            image_paths.extend(glob.glob(os.path.join(input_dir, f'*.{ext.upper()}')))
        
        if not image_paths:
            print(f"在目录 {input_dir} 中未找到图像文件")
            return
        
        print(f"找到 {len(image_paths)} 张图像，开始批量处理...")
        
        total_time = 0
        for i, image_path in enumerate(image_paths):
            print(f"\n处理图像 {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            # 处理单张图像
            start_time = time.time()
            self.process_single_image(image_path, output_dir)
            process_time = time.time() - start_time
            total_time += process_time
            
            print(f"处理完成，耗时: {process_time:.4f}秒")
        
        avg_time = total_time / len(image_paths)
        print(f"\n所有图像处理完成！")
        print(f"总图像数: {len(image_paths)}")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"平均每张图像耗时: {avg_time:.4f}秒")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='语义分割推理')
    parser.add_argument('--model_dir', type=str, required=True, 
                        help='包含所有模型权重的目录')
    
    # 添加两种模式：单图像处理或目录处理
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image_path', type=str, 
                        help='输入图像路径（单图像模式）')
    group.add_argument('--image_dir', type=str, 
                        help='输入图像目录（批量模式）')
    
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出图像目录')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='计算设备')
    
    args = parser.parse_args()
    
    # 确保所有模型文件存在
    required_files = ["encoder.pth", "semantic.pth"]
    for file in required_files:
        if not os.path.exists(os.path.join(args.model_dir, file)):
            raise FileNotFoundError(f"找不到模型文件: {file}")
    
    for item in os.listdir(args.output_dir):
        item_path = os.path.join(args.output_dir, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
    
    # 创建推理器
    inference = SemanticInference(
        model_dir=args.model_dir,
        device=args.device
    )
    
    # 根据参数选择处理模式
    if args.image_path:
        # 单图像处理模式
        inference.process_single_image(
            image_path=args.image_path,
            output_dir=args.output_dir
        )
    elif args.image_dir:
        # 批量处理模式
        inference.process_image_directory(
            input_dir=args.image_dir,
            output_dir=args.output_dir
        )