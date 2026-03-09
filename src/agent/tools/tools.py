from typing import Callable

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
            "fundus_collection_title_similarity_search": vdb.fundus_collection_title_similarity_search,
            "fundus_collection_description_similarity_search": vdb.fundus_collection_description_similarity_search,
            "find_fundus_records_with_similar_image": vdb.find_fundus_records_with_similar_image,
            "find_fundus_records_with_images_similar_to_the_text_query": vdb.find_fundus_records_with_images_similar_to_the_text_query,
            "find_fundus_records_with_images_similar_to_user_image": vdb.find_fundus_records_with_images_similar_to_user_image,
            "find_fundus_records_with_titles_similar_to_the_text_query": vdb.find_fundus_records_with_titles_similar_to_the_text_query,
        }
    )

    return sim_search_tool
