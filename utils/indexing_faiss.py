import faiss
def vector_database(normalized_feature):
    """
    :param normalized_feature: provide normalized flatten feature

    Function store the feature into database 
    """
    index = faiss.IndexFlatIP(normalized_feature.shape[1])
    index.add(normalized_feature)
    return index

def search_VIT_database(vector_index,query_image,top_k=5):
    inputs = feature_extractor(images=query_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        query_embedding = outputs.last_hidden_state[:, 0, :].numpy().astype('float32')
        
    distances, indices = vector_index.search(query_embedding, top_k)  # k nearest neighbors
    return distances, indices
def search_database(vector_index,test_norm_feature,top_k = 5):
    distances, indices = vector_index.search(test_norm_feature, top_k)
    return distances, indices

