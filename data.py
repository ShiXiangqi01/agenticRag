import json
import fitz  # PyMuPDF
import os
from PIL import Image
import io
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
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
    def __init__(self, model_name="Qwen/Qwen2.5-VL-3B-Instruct"):
        """
        初始化Qwen-VL模型和处理器
        """
        print("正在加载Qwen-VL模型...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 加载模型和处理器
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )

        print(f"模型加载完成，运行在: {self.device}")
    
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
    
    def build_message(self, image_path, pre_page_path, next_page_path, cur_page_path):

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {
                        "type": "image",
                        "image": pre_page_path,
                    },
                    {
                        "type": "image",
                        "image": cur_page_path
                    },
                    {
                        "type": "image",
                        "image": next_page_path,
                    },
                    {"type": "text", 
                     "text": IMAGE_ANALYSIS_INSTRUCTION},
                ],
            }
        ]
        return messages
    
    def generate_image_description(self, image_path, pre_page_path, next_page_path, cur_page_path ):

        messages = self.build_message(image_path, pre_page_path, next_page_path, cur_page_path)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**input) # max_new_tokens=128
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )


        print(output_text)
        return output_text

    
    def process_image_with_context(self, pdf_path, pdf_name, part, output_json="image_descriptions.json"):

        page_list, images_info = self.extract_images_from_pdf(pdf_path)
        results = []
                
        for i, img_info in enumerate(images_info):
            image_path = img_info["image_path"]
            page_num = img_info["page_num"]
            pre_page_path = page_list[page_num-1]["page_path"]
            cur_page_path = page_list[page_num]["page_path"]
            next_page_path = page_list[page_num+1]["page_path"]
            image_json = self.generate_image_description(image_path, pre_page_path, next_page_path, cur_page_path)

            record = {
                
                "image_id": f"page_{img_info['page_number']}_img_{img_info['image_index']}",
                "image_name": image_json.name,
                "page_number": img_info["page_number"],
                "image_index": img_info["image_index"],
                "image_path": img_info["image_path"],
                "filename": pdf_name,
                "description": image_json.description,
                "part": part


            }

            results.append(record)

        with open(output_json, 'w', encoding='utf-8') as f:

            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
                
        print(f"处理完成! 结果已保存到: {output_json}")
        return results


if __name__ == "__main__":
    processor = PDFImageProcessorWithContext()
    pdf_file = "sample.pdf"  # 替换为你的PDF文件路径
    results = processor.process_image_with_context(pdf_path=pdf_file,output_json="qwen_vl_descriptions_with_context.json")