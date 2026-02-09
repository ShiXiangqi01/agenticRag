import base64
import json, psycopg2, sys, os


json_path = sys.argv[1] if len(sys.argv) > 1 else 'my_file.json'
table_name = sys.argv[2] if len(sys.argv) > 2 else 'my_table'

with open(json_path, encoding='utf-8') as f:
    data = json.load(f)
    if isinstance(data, dict):
        data = [data]

conn = psycopg2.connect(
    host = 'localhost', port = 5432,
    user = os.getenv('USER'),
    database = 'agenticDB'
)
cur = conn.cursor()

cur.execute(f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        image_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        image_name TEXT,
        page_number INT,
        image_index INT,
        image_path TEXT,
        file_name_id UUID,
        description TEXT,
        part TEXT,
        base64_image BYTEA

    )
"""
)

inserted = 0
failed = 0
skipped = 0

for item in data:
    
    image_path = item['image_path']
    if not image_path:
        skipped += 1
        print(f"⚠ 跳过没有 image_path 的记录")
        continue

    with open(image_path, 'rb') as img_file:
        image_bytes = img_file.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    if  not base64_image:
        print(f"✅ 处理图片: {image_path}")


    cur.execute(f"""
        INSERT INTO {table_name} (image_name, page_number, image_index,image_path,file_name_id,description,part, base64_image)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (item['image_name'], item['page_number'], item['image_index'], item['image_path'], "b9979c1e-e821-4800-bc43-d371a6308b09", item['description'], item['part'], base64_image))

    


conn.commit()
cur.close()
conn.close()
print(f"✅ 导入 {len(data)} 条记录到 {table_name}")