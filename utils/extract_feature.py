import numpy as np
def normalized_feature(flatten_feature):
    norm_feat = flatten_feature/np.linalg.norm(flatten_feature,axis=1, keepdims=True)
    return norm_feat
def extract_resnet_feature(model,images):
    resnet_feature = model.extract_feat(images)
    norm_resnet_feature = normalized_feature(resnet_feature)
    return norm_resnet_feature

def extract_autoencoder_feature(model,images):
    # if len(images.shape)<=3:
    #     encoder_feature = model.predict(np.expand_dims(images, axis=0))
    #     # query_norm_feature = normalized_feature(query_feature)
    # else:
    encoder_feature = model.predict(images)
    encoder_feature = encoder_feature.reshape((encoder_feature.shape[0], -1))
    norm_encoder_feature = normalized_feature(encoder_feature)
    return norm_encoder_feature
def extract_siamese_feature(model,images):
    siamese_feature = model.predict(images)
    norm_siamese_feature = normalized_feature(siamese_feature)
    return norm_siamese_feature

def extract_vit_feature(model,images):
    vit_feature = model.extract_feature(images)
    norm_vit_feature = normalized_feature(vit_feature)
    return norm_vit_feature