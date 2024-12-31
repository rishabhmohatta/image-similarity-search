from models.getresnet50model import getResNet50Model
from models.autoencodermodel import AutoencoderModel
from models.siamese_model import SiameseTripletModel
from models.VisionTransformermodel import ViT
from utils.extract_feature import extract_resnet_feature,extract_autoencoder_feature,extract_siamese_feature,extract_vit_feature
from utils.indexing_faiss import vector_database,search_database
from utils.dataset import fashion_dataset,preprocess_dataset_VIT
from utils.Evaluation import get_labels
from utils.Visualize import plot_results
# from settings.constants import MODEL_TYPE,TRAIN_AUTOENCODER,TRAIN_SIAMESENET, TESTING,NO_OF_TEST_IMAGE,AUTOENCODER_MODEL_PATH,SIAMESE_MODEL_PATH,VISUALIZE,EVALUATION
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
import numpy as np
import pandas as pd
import faiss
import os

###Reading data fashion mnist
# MODEL_TYPE="siamesenet"
def main(MODEL_TYPE,TRAIN_AUTOENCODER,TRAIN_SIAMESENET,TESTING,NO_OF_TEST_IMAGE,VISUALIZE,
         EVALUATION,SIAMESE_MODEL_PATH,AUTOENCODER_MODEL_PATH,EVAL_PATH):
    if MODEL_TYPE !="VIT":
        x_train,y_train,x_test,y_test = fashion_dataset()
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
        print(x_train.shape,x_test.shape)
    else:
        fashion_mnist = preprocess_dataset_VIT(size=(224,224))
    query_image = x_test[:NO_OF_TEST_IMAGE]
    # TRAIN_AUTOENCODER = False
    # TRAIN_SIAMESENET = False
    # TESTING=True
    # If statment selecting network from which we need to extract feature and test image
    if MODEL_TYPE=="resnet":
        model = getResNet50Model()
        if TESTING:
            vector_index=faiss.read_index("Database/faiss_resnet_index.index")
            test_norm_feature = extract_resnet_feature(model,query_image)
            distance,indices = search_database(vector_index,test_norm_feature,top_k = 5)
            # print('Done',distance,indices)
        else:
            embeddings = extract_resnet_feature(model,x_train[:3000])
            vector_index = vector_database(embeddings)
            faiss.write_index(vector_index, "Database/faiss_resnet_index.index")
    elif MODEL_TYPE=="autoencoder":
        autoencoder = AutoencoderModel()
        # AUTOENCODER_MODEL_PATH = f"models/Sample_autoencoder.h5"
        if TRAIN_AUTOENCODER==False:
            encoder_model = autoencoder.load_model(AUTOENCODER_MODEL_PATH)
        else:
            autoencoder.train_autoencoder(x_train[:3000],x_val[:600],AUTOENCODER_MODEL_PATH,batch_size=256,epoch=20)
            encoder_model = autoencoder.load_model(AUTOENCODER_MODEL_PATH)
        if TESTING:
            vector_index=faiss.read_index("Database/faiss_autoencoder_index.index")
            test_norm_feature = extract_autoencoder_feature(encoder_model,query_image)
            distance,indices = search_database(vector_index,test_norm_feature,top_k = 5)
            # print('Done',distance,indices)
        else:
            embeddings = extract_autoencoder_feature(encoder_model,x_train[:3000])
            vector_index = vector_database(embeddings)
            faiss.write_index(vector_index, "Database/faiss_autoencoder_index.index")
    elif MODEL_TYPE=="siamesenet":
        input_shape = (28,28,1)
        siamese_triplet_model = SiameseTripletModel(input_shape)
        # SIAMESE_MODEL_PATH = f"models/Sample_siamese.h5"
        if TRAIN_SIAMESENET==False:
            siamese_model = siamese_triplet_model.load_model(SIAMESE_MODEL_PATH)
        else:
            siamese_triplet_model.train(x_train[:3000], y_val[:600],SIAMESE_MODEL_PATH)
            siamese_model = siamese_triplet_model.load_model(SIAMESE_MODEL_PATH)
        if TESTING:
            vector_index=faiss.read_index("Database/faiss_siamese_index.index")
            test_norm_feature = extract_siamese_feature(siamese_model,query_image)
            distance,indices = search_database(vector_index,test_norm_feature,top_k = 5)
            # print('Done',distance,indices)
        else:
            embeddings = extract_siamese_feature(siamese_model,x_train[:3000])
            # print(embeddings.shape)
            vector_index = vector_database(embeddings)
            faiss.write_index(vector_index, "Database/faiss_siamese_index.index")
    elif MODEL_TYPE=="VIT":
        vit_model = ViT()
        if TESTING:
            vector_index=faiss.read_index("Database/faiss_vit_index.index")
            test_norm_feature = extract_vit_feature(vit_model,query_image)
            distance,indices = search_VIT_database(vector_index,test_norm_feature,top_k = 5)
            # print('Done',distance,indices)
        else:
            embeddings = extract_vit_feature(vit_model,fashion_mnist[:3000])
            vector_index = vector_database(embeddings)
            faiss.write_index(vector_index, "Database/faiss_vit_index.index")
    if EVALUATION:
        ground_truth_labels, predictions_labels = get_labels(query_image,y_test,y_train,distance,indices)
        precision = precision_score(ground_truth_labels, predictions_labels, average='macro')
        recall = recall_score(ground_truth_labels, predictions_labels, average='macro')
        accuracy = accuracy_score(ground_truth_labels, predictions_labels)
        # print(precision,recall,accuracy)
        data = {"Model_type":MODEL_TYPE,
                "No_test_images":NO_OF_TEST_IMAGE,
                "Precision":precision*100,
                "Recall":recall*100,
                "Accuracy":accuracy*100
               }
        data = {k: [v] for k, v in data.items()}
        df = pd.DataFrame(data)
        if not os.path.isfile(EVAL_PATH):
            df.to_csv(EVAL_PATH, mode="w", header=True, index=False)
        
        else:
            df.to_csv(EVAL_PATH, mode="a", header=False, index=False)
    if VISUALIZE:
        plot_results(query_image,distance,x_train,y_train,indices)
    # pass
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Model configuration and testing parameters.')

    parser.add_argument('-mt','--model_type', type=str, choices=['resnet', 'autoencoder', 'siamesenet', 'VIT'],
                        default='resnet', help='Type of model to use.')
    parser.add_argument('-train_ae','--train_autoencoder', type=bool, default=False,
                        help='Flag to train autoencoder.')
    parser.add_argument('-train_st','--train_siamesenet', type=bool, default=False,
                        help='Flag to train SiameseNet.')
    parser.add_argument('-test','--testing', type=bool, default=True,
                        help='Flag to indicate if testing is to be performed.')
    parser.add_argument('-num_img','--no_of_test_image', type=int, default=1,
                        help='Number of test images to use.')
    parser.add_argument('-vis','--visualize', type=bool, default=False,
                        help='Flag to visualize results.')
    parser.add_argument('-eval','--evaluation', type=bool, default=True,
                        help='Flag to perform evaluation.')
    parser.add_argument('-path_st','--siamese_model_path', type=str, default='models/Sample_siamese.h5',
                        help='Path to the Siamese model file.')
    parser.add_argument('-path_ae','--autoencoder_model_path', type=str, default='models/Sample_autoencoder.h5',
                        help='Path to the autoencoder model file.')
    parser.add_argument('-path_eval','--evalpath', type=str, default='eval_summary.csv',
                        help='Path to the evaluation metrics file.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    # print(args)
    MODEL_TYPE=args.model_type  # ["resnet","autoencoder","siamesenet","VIT"]
    TRAIN_AUTOENCODER = args.train_autoencoder
    TRAIN_SIAMESENET = args.train_siamesenet
    TESTING=args.testing
    NO_OF_TEST_IMAGE = args.no_of_test_image
    VISUALIZE = args.visualize
    EVALUATION = args.evaluation
    SIAMESE_MODEL_PATH = args.siamese_model_path
    AUTOENCODER_MODEL_PATH =args.autoencoder_model_path 
    EVAL_PATH = args.evalpath
    main(MODEL_TYPE,TRAIN_AUTOENCODER,TRAIN_SIAMESENET,TESTING,NO_OF_TEST_IMAGE,
         VISUALIZE,EVALUATION,SIAMESE_MODEL_PATH,AUTOENCODER_MODEL_PATH,EVAL_PATH)