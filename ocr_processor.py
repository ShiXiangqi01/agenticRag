import json
import os
import glob
from ollama import Client


IMAGE_ANALYSIS_INSTRUCTION = """

You are a professional PDF and image analysis assistant specialized in extracting contextual information from document screenshots.

## Task Overview
You will receive 4 images:
1. **Target Image**: The specific region/element to analyze
2. **Context Images**: Full-page screenshots of the previous page, current page, and next page

## Your Objectives
1. **Analyze the target image**: Identify its content type (e.g., chart, diagram, table, code snippet, equation, illustration)
2. **Leverage context**: Use the surrounding pages to extract relevant captions, labels, section headings, or explanatory text associated with the target image
3. **Generate outputs**:
   - `name`: A concise, descriptive title (3-8 words) in English. Use semantic naming, NOT generic references like "Figure 40.1" or "image_001"
   - `description`: A clear English summary (1-3 sentences) that includes:
     • What the image depicts
     • Relevant contextual text from surrounding pages (e.g., captions, figure references, section context)

## Output Format Requirements
- Output **ONLY** a valid JSON object with exactly two keys: `name` and `description`
- Use double quotes for all JSON strings
- Do NOT include markdown code blocks, explanations, or additional text
- Ensure the JSON is parseable (no trailing commas, proper escaping)

## Example Output
{
    "name": 
    "description": 
}

## Quality Guidelines
✅ Prioritize accuracy: Only include text/context that is clearly associated with the target image
✅ Be concise: Avoid redundant or overly verbose descriptions
✅ Use technical terminology appropriately when the content is domain-specific
✅ If contextual text is ambiguous or unavailable, describe only what is visually evident in the target image

""".strip()


class OCRProcessor:
    """使用 Ollama 本地视觉模型进行OCR处理的工具类"""
    
    def __init__(self, model_name="qwen3.5:27b", ollama_host="http://127.0.0.1:11434"):
        """
        使用 Ollama 本地模型。
        
        Args:
            model_name (str): Ollama 本地模型名，例如 "qwen3.5:27b"
            ollama_host (str): Ollama 服务地址，默认 "http://127.0.0.1:11434"
        """
        print(f"正在连接 Ollama 模型: {model_name}")
        self.model_name = model_name
        self.ollama_host = ollama_host.rstrip("/")
        self.client = Client(host=self.ollama_host)

        try:
            available = self.client.list()
            model_names = {m.model for m in available.models}
            if self.model_name not in model_names:
                print(f"⚠ 当前 Ollama 未发现模型 {self.model_name}，请先执行: ollama pull {self.model_name}")
        except Exception as e:
            print(f"⚠ 无法校验本地模型列表: {e}")

        print(f"✓ Ollama 初始化完成: {self.ollama_host}")
    
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
        构建 Ollama 多图像消息格式
        
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
                "content": IMAGE_ANALYSIS_INSTRUCTION,
                "images": [
                    image_path,
                    pre_page_path,
                    cur_page_path,
                    next_page_path,
                ],
            }
        ]
        return messages
    
    def generate_image_description(self, image_path, pre_page_path, next_page_path, cur_page_path):
        """
        使用 Ollama 本地视觉模型生成图片描述
        
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
            
            messages = self.build_messages(image_path, pre_page_path, next_page_path, cur_page_path)
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": 0.2,
                    "num_predict": 4096,
                },
            )

            output_text = response.get("message", {}).get("content", "")
            
            if not output_text:
                print("✗ 模型返回为空")
                return None
            
            print(f"✓ 分析完成")
            return [output_text]
            
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

            print(f"image_path: {image_path}")
            print(f"pre_page_path: {pre_page_path}")
            print(f"cur_page_path: {cur_page_path}")
            print(f"next_page_path: {next_page_path}")

            # 生成图片描述r
            description_json = self.generate_image_description(
                image_path, pre_page_path, next_page_path, cur_page_path
            )
            
            if description_json is None:
                print(f"⚠ 跳过第 {page_num} 页的图片 {img_info['image_index']}")
                continue
            
            description = description_json[0].replace('```json','').replace('```','').strip()
            # 尝试解析 JSON 结果
            try:
                parsed_json = json.loads(description)
                image_name = parsed_json.get("name", "Unknown") if parsed_json else "Unknown"
                image_desc = parsed_json.get("description", description_json) if parsed_json else description_json
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
        images_dir="Database/images/Muscle_testing/Knee",
        pages_dir="pages"
    )
    
    # 初始化OCR处理器并处理
    processor = OCRProcessor()
    results = processor.process_images_with_context(
        page_list=page_list,
        images_info=image_info_list,
        pdf_name="Knee.pdf",
        part="Knee",
        output_json="qwen_vl_descriptions_with_context.json"
    )
    
    print("\n" + "=" * 60)
    print(f"完成！共处理 {len(results)} 张图片")
    print("=" * 60)
