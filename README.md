# image-similarity-search
Content based image similarity approach using python,deep learning ,computer vision
# Install The Requirements
    pip install -r requirements.txt
# Run the script using the following command:
    python inferencing.py [options]
  ## Arguments:
    -mt or --model_type: This is a required argument that specifies the type of model to use. Available options are resnet, autoencoder, siamesenet, or VIT. The default value is resnet.
    -train_ae or --train_autoencoder: This is a boolean flag that indicates whether to train the autoencoder. The default value is False.
    -train_sn or --train_siamesenet: This is a boolean flag that indicates whether to train the SiameseNet. The default value is False.
    -test or --testing: This boolean flag indicates if testing should be performed. The default value is True.
    -num_img or --no_of_test_image: This integer argument specifies the number of test images to use. The default value is 1.
    -vis or --visualize: This boolean flag enables visualization of results when set to True. The default value is False.
    -eval or --evaluation: This boolean flag determines whether evaluation should be performed after inference. The default value is True.
    -path_st or --siamese_model_path: This string argument specifies the path to the Siamese model file. The default path is models/Sample_siamese.h5.
    -path_ae or --autoencoder_model_path: This string argument specifies the path to the autoencoder model file. The default path is models/Sample_autoencoder.h5.
    -path_eval or --evalpath: This string argument specifies the path to the evaluation metrics file. The default path is eval_summary.csv
  ## EXAMPLE:
      python inferencing.py --model_type resnet --visualize True -eval True -num_img 10 [It will extract feature with resnet and visualize and display the test number of images and saved the evaluation metrics ]
      python inferencing.py --model_type siamesenet --train_sn True --no_of_test_image 5 --test True [It will train the siamese network on default dataset with default epochs and saved it into models and test 5 images with new model]
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
