from typing import Callable
'''
一个标准的 Agent Callable 通常包含三部分：
    元数据（Schema）：名称、描述、参数定义（让 LLM 知道什么时候用、怎么用）。
    实现逻辑（Implementation）：实际执行的代码（Python 函数、Java 方法、API 请求）。
    返回结果（Observation）：执行后的结果，反馈给 LLM 进行下一步决策。
'''


class Tool:
    def __init__(self, name: str):
        self.name = name
        self.functions: dict[str, Callable] = {}

    def register_functions(self, functions: dict[str, Callable]) -> None:
        self.functions.update(functions)

    def __str__(self) -> str:
        return f"Tool(Name={self.name}, Functions={list(self.functions.keys())})"

    def __repr__(self) -> str:
        return str(self)
    

def get_sim_search_tool() -> Tool:

    from src.data.vector_db import VectorDB

    vdb = VectorDB()

    sim_search_tool = Tool(name="Similarity Search Tool")
    # because VectorDB is a singleton, the methods of the instance are always the same and we can use them in the dict
    sim_search_tool.register_functions(
        {
            "find_similar_images_img_according_to_users_image_query": vdb.find_similar_images_img_according_to_users_image_query,
            "find_similar_images_img_according_to_users_text_query": vdb.find_similar_images_img_according_to_users_text_query,
            "find_similar_images_description_according_to_users_text_query": vdb.find_similar_images_description_according_to_users_text_query,
            "find_similar_images_description_according_to_users_image_query": vdb.find_similar_images_description_according_to_users_image_query,
            "find_similar_texts_content_according_to_users_text_query": vdb.find_similar_texts_content_according_to_users_text_query,
            "find_similar_texts_content_according_to_images_description": vdb.find_similar_texts_content_according_to_images_description,
        }
    )

    return sim_search_tool

def get_lexical_search_tool() -> Tool:
    from src.data.vector_db import VectorDB

    vdb = VectorDB()
    lexical_search_tool = Tool(name="Lexical Search Tool")
    lexical_search_tool.register_functions(
        {
            "find_texts_content_relative_lexical_search": vdb.find_texts_content_relative_lexical_search,
            "find_images_relative_lexical_search": vdb.find_images_relative_lexical_search,
        }
    )
    return lexical_search_tool



def get_lookup_tool() -> Tool:
    from src.data.vector_db import VectorDB

    vdb = VectorDB()
    lookup_tool = Tool(name="DB Lookup Tool")
    lookup_tool.register_functions(
        {
            # For text
            "find_surrounding_chunks": vdb.find_surrounding_chunks,
            "get_text_record_by_vector_id": vdb.get_text_record_by_vector_id,
            # For image
            "get_image_record_by_vector_id": vdb.get_image_record_by_vector_id,
            # look up
            "list_all_body_parts": vdb.list_all_body_parts,
            "list_all_file_names": vdb.list_all_file_names,
            "list_all_character_names_in_one_file": vdb.list_all_character_names_in_one_file,
            
        }
    )
    return lookup_tool

def get_image_analysis_tool() -> Tool:
    from src.agent.tools.image_analyzer import ImageAnalyzer

    image_analysis_tool = Tool(name="Image Analysis Tool")
    image_analyzer = ImageAnalyzer()
    image_analysis_tool.register_functions(
        {
            "answer_question_about_image_record": image_analyzer.answer_question_about_image_in_agentic_image,
            "detect_objects_in_image_record": image_analyzer.detect_objects_in_image_record,
            "extract_text_from_image_record": image_analyzer.extract_text_from_image_record

        }
    )
    return image_analysis_tool
