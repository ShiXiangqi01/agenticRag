import json
import os
import glob
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch


IMAGE_ANALYSIS_INSTRUCTION = """
# Your Role

You are an expert Pdf analysis assistant.

# Your Task

You will receive three image.The first image is the target image; the next three images are from the page before, the same page as, and the page after the target image, respectively.
You should accurately extract the text related to the target image based on the context provided by the other three images.
You also need to extract the name or title of the target image, if no name or title is found, you should give a reasonable guess according to the context.

# Output Format

Output all detected objects in JSON format with the following structure:
```json
[
    {
'        "name": "<NAME OF THE TARGET IMAGE>",'
'        "description": "<RELATED TEXT>",'
'        
    }
]
```
""".strip()


class OCRProcessor:
    """使用Qwen VL模型进行OCR处理的工具类"""
    
    def __init__(self, model_name="Qwen/Qwen3-VL-30B-A3B-Instruct"):
        """
        使用 transformers 加载 Qwen VL 模型。
        
        Args:
            model_name (str): HuggingFace 模型名称，默认为 "Qwen/Qwen3-VL-30B-A3B-Instruct"
                           可选: 
                           - "Qwen/Qwen3-VL-32B-Instruct" (最强，需要 ~64GB 显存)
                           - "Qwen/Qwen2-VL-7B-Instruct" (中等)
                           - "Qwen/Qwen2-VL-2B-Instruct" (更小更快)
        """
        print(f"正在加载 Qwen VL 模型: {model_name}")
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 加载模型和处理器
        print("加载 processor...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        print(f"加载模型到 {self.device}...")
        # 对于 32B 模型，使用 8bit 量化以节省显存
        print("检测到大模型，启用 8bit 量化...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            # device_map="auto",
            # load_in_8bit=True,
        )
        
        # 确保模型移到同一设备
        self.model = self.model.to(self.device)
        
        print(f"✓ 模型加载成功！设备: {self.device}")
    
    @staticmethod
    def load_extracted_images(images_dir="extracted_images", pages_dir="pages"):
        """
        从已提取的目录中加载图片和页面信息
        
        Args:
            images_dir (str): 图片目录
            pages_dir (str): 页面目录
            
        Returns:
            tuple: (page_list, image_info_list)
        """
        print(f"正在从目录加载已提取的图片...")
        print(f"  图片目录: {images_dir}")
        print(f"  页面目录: {pages_dir}")
        
        # 加载页面信息
        # 标准库中用于文件路径匹配的函数，可以根据通配符模式查找符合条件的文件。
        page_files = sorted(glob.glob(os.path.join(pages_dir, "page_*.jpg")))
        page_list = []
        for page_file in page_files:
            # 从文件名提取页码
            filename = os.path.basename(page_file)
            page_num = int(filename.split('_')[1].split('.')[0])
            page_list.append({
                "page_number": page_num,
                "page_path": page_file,
            })
        
        page_list.sort(key=lambda x: x['page_number'])
        print(f"✓ 加载了 {len(page_list)} 个页面")
        
        # 加载图片信息
        image_files = sorted(glob.glob(os.path.join(images_dir, "page_*_img_*.*")))
        image_info_list = []
        for image_file in image_files:
            # 从文件名提取信息: page_0_img_0.png
            filename = os.path.basename(image_file)
            parts = filename.split('_')
            page_num = int(parts[1])
            img_idx = int(parts[3].split('.')[0])
            
            image_info_list.append({
                "page_number": page_num,
                "image_index": img_idx,
                "image_path": image_file,
            })
        
        image_info_list.sort(key=lambda x: (x['page_number'], x['image_index']))
        print(f"✓ 加载了 {len(image_info_list)} 张图片")
        
        return page_list, image_info_list
    
    def build_messages(self, image_path, pre_page_path, next_page_path, cur_page_path):
        """
        构建多图像消息格式
        
        Args:
            image_path (str): 目标图片路径
            pre_page_path (str): 前一页图片路径
            next_page_path (str): 后一页图片路径
            cur_page_path (str): 当前页图片路径
            
        Returns:
            list: 消息列表
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "image", "image": pre_page_path},
                    {"type": "image", "image": cur_page_path},
                    {"type": "image", "image": next_page_path},
                    {"type": "text", "text": IMAGE_ANALYSIS_INSTRUCTION},
                ],
            }
        ]
        return messages
    
    def generate_image_description(self, image_path, pre_page_path, next_page_path, cur_page_path):
        """
        使用 Qwen3-VL 生成图片描述
        
        Args:
            image_path (str): 目标图片路径
            pre_page_path (str): 前一页图片路径
            next_page_path (str): 后一页图片路径
            cur_page_path (str): 当前页图片路径
            
        Returns:
            str: 生成的描述文本，失败返回None
        """
        try:
            print(f"正在分析图片: {image_path}")
            
            # 构建消息
            messages = self.build_messages(image_path, pre_page_path, next_page_path, cur_page_path)
            
            # 应用聊天模板
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # 处理视觉信息
            image_inputs, video_inputs = process_vision_info(messages)
            
            # 准备输入
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            # 生成输出
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    temperature=0.3,  # 控制生成文本的随机性 越大越创造
                    max_new_tokens=1024,
                    do_sample=True
                )
            
            # 解码输出
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            print(f"✓ 分析完成")
            return output_text
            
        except Exception as e:
            print(f"✗ 生成描述失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_images_with_context(self, page_list, images_info, pdf_name, part, output_json="image_descriptions.json"):
        """
        处理所有图片并生成描述
        
        Args:
            page_list (list): 页面信息列表
            images_info (list): 图片信息列表
            pdf_name (str): PDF文件名
            part (str): 部分标识
            output_json (str): 输出JSON文件路径
            
        Returns:
            list: 处理结果列表
        """
        results = []
                
        for i, img_info in enumerate(images_info):
            image_path = img_info["image_path"]
            page_num = img_info["page_number"]
            
            # 处理页码范围（避免超出边界）
            pre_page_idx = max(0, page_num - 1)
            next_page_idx = min(len(page_list) - 1, page_num + 1)
            
            pre_page_path = page_list[pre_page_idx]["page_path"]
            cur_page_path = page_list[page_num]["page_path"]
            next_page_path = page_list[next_page_idx]["page_path"]
            
            # 生成图片描述
            description_json = self.generate_image_description(
                image_path, pre_page_path, next_page_path, cur_page_path
            )
            
            if description_json is None:
                print(f"⚠ 跳过第 {page_num} 页的图片 {img_info['image_index']}")
                continue
            
            # 尝试解析 JSON 结果
            try:
                parsed_json = json.loads(description_json)
                image_name = parsed_json[0].get("name", "Unknown") if isinstance(parsed_json, list) and parsed_json else "Unknown"
                image_desc = parsed_json[0].get("description", description_json) if isinstance(parsed_json, list) and parsed_json else description_json
            except json.JSONDecodeError:
                # 如果不是 JSON 格式，直接使用原文本
                image_name = "Analysis Result"
                image_desc = description_json

            record = {
                "image_id": f"page_{img_info['page_number']}_img_{img_info['image_index']}",
                "image_name": image_name,
                "page_number": img_info["page_number"],
                "image_index": img_info["image_index"],
                "image_path": img_info["image_path"],
                "filename": pdf_name,
                "description": image_desc,
                "part": part
            }

            results.append(record)
            print(f"✓ 处理完成: {record['image_id']}")

        # 保存结果到JSON文件
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
                
        print(f"✓ 所有处理完成! 结果已保存到: {output_json}")
        return results


if __name__ == "__main__":
    # 使用已提取的图片进行OCR处理
    print("=" * 60)
    print("使用已提取的图片进行OCR处理")
    print("=" * 60)
    
    # 加载已提取的图片和页面信息
    page_list, image_info_list = OCRProcessor.load_extracted_images(
        images_dir="extracted_images",
        pages_dir="pages"
    )
    
    # 初始化OCR处理器并处理
    processor = OCRProcessor()
    results = processor.process_images_with_context(
        page_list=page_list,
        images_info=image_info_list,
        pdf_name="Stretching.pdf",
        part="Neck",
        output_json="qwen_vl_descriptions_with_context.json"
    )
    
    print("\n" + "=" * 60)
    print(f"完成！共处理 {len(results)} 张图片")
    print("=" * 60)
