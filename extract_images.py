import fitz  # PyMuPDF
import os


class PDFImageExtractor:
    """从PDF中提取图片和页面的工具类"""
    
    def __init__(self, output_dir="Database/images/Muscle_testing/Knee", pages_dir="pages"):
        """
        初始化图片提取器
        
        Args:
            output_dir (str): 提取的图片输出目录
            pages_dir (str): 页面图片输出目录
        """
        self.output_dir = output_dir
        self.pages_dir = pages_dir
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.pages_dir, exist_ok=True)
    
    def extract_images_from_pdf(self, pdf_path):
        """
        从PDF中提取所有图片和页面
        
        Args:
            pdf_path (str): PDF文件路径

        Returns:
            tuple: (page_list, image_info_list)
                - page_list: 包含每页信息的字典列表
                - image_info_list: 包含提取图片信息的字典列表
        """
        # 打开PDF文件
        doc = fitz.open(pdf_path)
        image_info_list = []
        page_list = []

        print(f"开始处理PDF: {pdf_path}, 共 {len(doc)} 页")

        # 遍历每一页
        for page_num in range(len(doc)):
            page_path = f"{self.pages_dir}/page_{page_num}.jpg"
            print(f"-------------{page_path}")
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))  # 1.5倍分辨率
            pix.save(page_path)
            print(f"保存页面图片: {page_path}")
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
                # if bbox.height < 20 or bbox.y1 > page.rect.height * 0.9:
                #     continue

                # 提取图片数据
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_size = len(image_bytes)

                # 生成唯一的图片文件名
                image_filename = f"page_{page_num}_img_{image_index}.{image_ext}"
                image_path = os.path.join(self.output_dir, image_filename)

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
        print(f"图片提取完成! 共提取 {len(image_info_list)} 张图片到目录: {self.output_dir}")
        return page_list, image_info_list


if __name__ == "__main__":
    # 测试代码
    extractor = PDFImageExtractor()
    pdf_file = "Database/pdf/Muscle_testing/Knee.pdf"  # 替换为你的PDF文件路径
    page_list, image_info_list = extractor.extract_images_from_pdf(pdf_file)
    print(f"\n提取了 {len(page_list)} 个页面和 {len(image_info_list)} 张图片")
