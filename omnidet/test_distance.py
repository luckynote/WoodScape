import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import time
import glob
from models.resnet import ResnetEncoder
from models.normnet_decoder import NormDecoder
from train_utils.distance_utils import tensor2array

class DepthInference:
    def __init__(self, model_dir, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_dir = model_dir
        self.models = {}
        self.args = self.create_dummy_args()
        self.load_models()
        
        # 定义ROI区域 (left, top, right, bottom)
        self.roi = (114, 110, 1176, 610)
        
    def create_dummy_args(self):
        """创建推理所需的虚拟参数"""
        class Args:
            def __init__(self):
                self.network_layers = 18  # 修改为18层
                self.pose_network_layers = 18  # 位姿编码器层数
                self.pose_model_type = "separate"  # 位姿模型类型
                self.num_scales = 4  # 多尺度数量
                self.frame_idxs = [0]  # 只使用当前帧
                self.rotation_mode = 'euler'  # 旋转表示方式
                
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
        
        # 2. 加载规范化解码器模型
        self.models["norm"] = NormDecoder(
            self.models["encoder"].num_ch_enc
        ).to(self.device)
        
        norm_path = os.path.join(self.model_dir, "norm.pth")
        norm_state = torch.load(norm_path, map_location=self.device)
        norm_state = self.remove_module_prefix(norm_state)
        
        # 过滤掉额外的键
        norm_state = self.filter_model_keys(norm_state, self.models["norm"])
        
        self.models["norm"].load_state_dict(norm_state)
        self.models["norm"].eval()
        
        # 对于单图像推理，不需要加载位姿模型
        print(f"深度模型已从 {self.model_dir} 加载")
    
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
        return input_tensor.to(self.device), original_size, roi_size
    
    def predict_distances(self, input_tensor):
        """执行深度估计推理"""
        with torch.no_grad():
            # 通过编码器提取特征
            features = self.models["encoder"](input_tensor)
            
            # 通过规范化解码器预测深度
            outputs = self.models["norm"](features)
            
            # 获取最高分辨率的深度图 (scale=0)
            inv_depth = outputs[("norm", 0)]
            
            # 将逆深度转换为深度
            depth = 1.0 / inv_depth.clamp(min=1e-6)
            
        return depth.squeeze(0).squeeze(0).cpu()  # 移除batch和channel维度
    
    def postprocess_depth(self, depth_map, original_size, roi_size):
        """后处理深度图"""
        # 转换为numpy数组
        depth_np = depth_map.numpy()
        
        # 调整回ROI尺寸
        depth_roi = np.array(Image.fromarray(depth_np).resize(
            roi_size, Image.BILINEAR))
        
        # 创建全尺寸深度图（与原始图像相同大小）
        full_depth = np.zeros((original_size[1], original_size[0]), dtype=np.float32)
        
        # 将ROI深度图放置到正确位置
        left, top, right, bottom = self.roi
        full_depth[top:bottom, left:right] = depth_roi
        
        return full_depth
    
    def visualize_depth(self, depth_map, output_path=None):
        """可视化深度图并保存"""
        plt.figure(figsize=(12, 6))
        plt.imshow(depth_map, cmap='jet')  # 使用jet颜色映射
        plt.axis('off')
        plt.colorbar()
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            print(f"深度图已保存至: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def process_single_image(self, image_path, output_path=None):
        """处理单张图像"""
        # 预处理图像（包括ROI裁剪）
        input_tensor, original_size, roi_size = self.preprocess_image(image_path)
        
        # 执行深度推理
        start_time = time.time()
        depth_map = self.predict_distances(input_tensor)
        inference_time = time.time() - start_time
        print(f"深度推理完成，耗时: {inference_time:.4f}秒")
        
        # 后处理深度图（恢复ROI位置）
        depth_map = self.postprocess_depth(depth_map, original_size, roi_size)
        
        # 可视化并保存
        if output_path:
            self.visualize_depth(depth_map, output_path)
        
        return depth_map
    
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

        image_paths = sorted(image_paths)
    
        for i, image_path in enumerate(image_paths):
            print(f"\n处理图像 {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            # 创建输出路径
            filename = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{filename}_depth.png")
            
            # 处理单张图像
            start_time = time.time()
            depth_map = self.process_single_image(image_path, output_path)
            process_time = time.time() - start_time
            total_time += process_time
            
            print(f"处理完成，耗时: {process_time:.4f}秒")
        
        avg_time = total_time / len(image_paths)
        print(f"\n所有图像处理完成！")
        print(f"总图像数: {len(image_paths)}")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"平均每张图像耗时: {avg_time:.4f}秒")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='深度估计推理')
    parser.add_argument('--model_dir', type=str, required=True, 
                        help='包含所有模型权重的目录')
    
    # 添加两种模式：单图像处理或目录处理
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image_path', type=str, 
                        help='输入图像路径（单图像模式）')
    group.add_argument('--image_dir', type=str, 
                        help='输入图像目录（批量模式）')
    
    parser.add_argument('--output_path', type=str, default=None,
                        help='输出图像路径（单图像模式）')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出图像目录（批量模式）')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='计算设备')
    
    args = parser.parse_args()
    
    # 确保所有模型文件存在
    required_files = ["encoder.pth", "norm.pth"]
    for file in required_files:
        if not os.path.exists(os.path.join(args.model_dir, file)):
            raise FileNotFoundError(f"找不到模型文件: {file}")
    
    # 创建推理器
    inference = DepthInference(
        model_dir=args.model_dir,
        device=args.device
    )
    
    # 根据参数选择处理模式
    if args.image_path:
        # 单图像处理模式
        if not args.output_path:
            # 如果没有指定输出路径，使用默认路径
            filename = os.path.splitext(os.path.basename(args.image_path))[0]
            args.output_path = f"{filename}_depth.png"
        
        depth_map = inference.process_single_image(
            image_path=args.image_path,
            output_path=args.output_path
        )
    elif args.image_dir:
        # 批量处理模式
        if not args.output_dir:
            # 如果没有指定输出目录，使用默认目录
            args.output_dir = os.path.join(args.image_dir, "depth_results")
            print(f"未指定输出目录，使用默认目录: {args.output_dir}")

        for item in os.listdir(args.output_dir):
            item_path = os.path.join(args.output_dir, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
        
        
        inference.process_image_directory(
            input_dir=args.image_dir,
            output_dir=args.output_dir
        )