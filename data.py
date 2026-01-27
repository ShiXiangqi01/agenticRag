import json
import fitz  # PyMuPDF
import os
from PIL import Image
import io
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor
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

class PDFImageProcessorWithContext:
    def __init__(self, model_name="Qwen/Qwen3-VL-32B-Instruct"):
        """
        使用 transformers 加载 Qwen VL 模型。
        
        Args:
            model_name (str): HuggingFace 模型名称，默认为 "Qwen/Qwen3-VL-32B-Instruct"
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
            device_map="auto",
            load_in_8bit=True  # 使用 8bit 量化
        )
        
        print(f"✓ 模型加载成功！设备: {self.device}")
    
    def extract_images_from_pdf(self, pdf_path, output_dir="extracted_images"):
        """

        Args:
            pdf_path (str): PDF文件路径
            output_dir (str): 图片输出目录

        Returns:
            list: 包含提取图片信息的字典列表
        """

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 打开PDF文件
        doc = fitz.open(pdf_path)
        image_info_list = []

        print(f"开始处理PDF: {pdf_path}, 共 {len(doc)} 页")

        page_list = []



        # 遍历每一页
        for page_num in range(len(doc)):
            page_path = f"pages/page_{page_num}.jpg"
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))  # 1.5倍分辨率
            pix.save(page_path)
            page_list.append({
                "page_number": page_num,
                "page_path": page_path,

            })

            # 获取该页的所有图片列表
            image_list = page.get_images(full=True)

            print(f"第 {page_num + 1} 页找到 {len(image_list)} 张图片")

            # 遍历该页的每张图片
            for image_index, img in enumerate(image_list):
                # 获取图片的xref（在PDF中的引用ID）
                xref = img[0]
                bbox = page.get_image_bbox(img)  # 获取图片在页面中的位置
        # 示例：排除高度小于20或位于底部10%区域的图片
                if bbox.height < 20 or bbox.y1 > page.rect.height * 0.9:
                    continue

                # 提取图片数据
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_size = len(image_bytes)

                # 生成唯一的图片文件名
                image_filename = f"page_{page_num}_img_{image_index}.{image_ext}"
                image_path = os.path.join(output_dir, image_filename)

                # 保存图片
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)

                # 收集图片信息
                image_info = {
                    "page_number": page_num,
                    "image_index": image_index,
                    "image_path": image_path,

                }

                image_info_list.append(image_info)

                print(f"  保存图片: {image_filename} (大小: {image_size} 字节)")

        doc.close()
        print(f"图片提取完成! 共提取 {len(image_info_list)} 张图片到目录: {output_dir}")
        return page_list, image_info_list
    
    
    
    def build_messages(self, image_path, pre_page_path, next_page_path, cur_page_path):
        """构建多图像消息格式"""
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
                    temperature=0.3, # 控制生成文本的随机性 越大越创造
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

    
    def process_image_with_context(self, pdf_path, pdf_name, part, output_json="image_descriptions.json"):

        page_list, images_info = self.extract_images_from_pdf(pdf_path)
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

        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
                
        print(f"✓ 所有处理完成! 结果已保存到: {output_json}")
        return results


if __name__ == "__main__":
    processor = PDFImageProcessorWithContext()
    pdf_file = "sample.pdf"  # 替换为你的PDF文件路径
    results = processor.process_image_with_context(pdf_path=pdf_file,output_json="qwen_vl_descriptions_with_context.json", pdf_name="sample.pdf", part="Part 1")