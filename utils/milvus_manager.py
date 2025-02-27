from pymilvus import MilvusClient, DataType
import numpy as np
import concurrent.futures


class MilvusManager:
    def get_images_as_doc(self, images_with_vectors:list):
        
        images_data = []

        for i in range(len(images_with_vectors)):
            data = {
                "embedding": images_with_vectors[i]["embedding"],
                "doc_id": i,
                "filepath": images_with_vectors[i]["filepath"],
            }
            images_data.append(data)

        return images_data