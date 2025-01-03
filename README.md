# image-similarity-search
Image Similarity Search, which enables you to retrieve similar images from a database based on an input image
In this approach used python,deep learning ,computer vision,faiss
# Install The Requirements
    pip install -r requirements.txt
# Run the script using the following command:
    python test.py [options]
  ## Arguments:
    -mt or --model_type: This is a required argument that specifies the type of model to use. Available options are resnet, autoencoder, siamesenet, or VIT. The default value is autoencoder.
    -i or --test_image_path: Type: str, Default: input/, Description: Path to the test image file.
    -o or --output_path: Type: str, Default: output/, Description: Directory where the output of the model will be saved
  ## EXAMPLE:
      python test.py -mt "autoencoder" -i "input/" -o "output/"
  # Result comparision:
      Tested the fashion database 150 test images and the metrics is:
      ![table 1](https://raw.githubusercontent.com/rishabhmohatta/image-similarity-search/main/images/Metrics.png)
    


      1. Autoencoder has the highest precision (77.64%) and accuracy (77.07%), indicating it performs well in distinguishing relevant results from non-relevant ones.
      2. Siamese network with triplets has better trade-off of Precision and recall
      3. Resnet has lower accuracy as it is not trained with the fashion-mnist dataset.It can be improved by training a model with own database rather then using pre trained weights.
      4. All model has a sample architecture they can be improved with adding complex network 
  # Remarks:
    0. Accuracy ,time,space (for storing faiss embedding) all three can be improved like( If we trained resnet50 with our sample dataset and then extract embeddings it can give good accurcay,Storing the ebedding in different format like binary files or compressed version,etc) 
    1.Can be improvise it better for user with many ways (The code is for evaluation Purpose)
    2.VIT approach implemetation tried but laptop resources doesn't support Sorry for inconvience :)
    3.There are many transformer approaches are present wher we can provide contextual data and image for better result like (CLIP)
    6. Used faiss indexing as it is faster than normally using cosine similarity or Euclidian distance. Its better for real time scrapped data set in terms of speed and accuracy , also it has few feature like IVF where speed of inferncing reduces more but its effect the accuracy. 

# Refernce Research paper and Blogs:
      --> In one Research paper if we can have a independent feature like tags for each image in dataset (like its for male or female,collar_type,etc) 
    we can relate them with embedding using corelation that will increase output accuracy 
        Paper link: https://arxiv.org/pdf/2308.16126v1 ("CorrEmbed: Evaluating Pre-trained Model Image Similarity Efficacy with a Novel Metric")
      --> For fashion related recent research paper link: https://arxiv.org/pdf/2306.02928v3("LRVS-Fashion: Extending Visual Search with Referring Instructions")  
      -->Blog: https://medium.com/@tapanbabbar/build-an-image-similarity-search-with-transformers-vit-clip-efficientnet-dino-v2-and-blip-2-5040d1848c00
              https://medium.com/pytorch/image-similarity-search-in-pytorch-1a744cf3469 
              https://medium.com/@tapanbabbar/build-an-image-similarity-search-with-transformers-vit-clip-efficientnet-dino-v2-and-blip-2-5040d1848c00
      -->Github refer link: https://github.com/bnsreenu/python_for_microscopists/tree/master/306%20-%20Content%20based%20image%20retrieval%E2%80%8B%20via%20feature%20extraction
      -->Referred Paper: 1. "Reducing the Dimensionality of Data with Neural Networks." Science, 313(5786), 504-507.
               2. Schroff, F., Kalenichenko, D., & Philbin, J. (2015). "FaceNet: A Unified Embedding for Face Recognition and Clustering." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 815-823.
               3. Johnson, J., Douze, M., & Jégou, H. (2017). "Billion-Scale Similarity Search with GPUs." Proceedings of the IEEE International Conference on Computer Vision (ICCV), 5390-5399.
               4. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., & Zisserman, A. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." Proceedings of the International Conference on Learning Representations (ICLR).
