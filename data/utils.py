import os

import cv2
import numpy as np
from addict import Dict

attr_language_encoder = None


class Attr_Language_Encoder:
    def __init__(self, model_name='all-mpnet-base-v2'):
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device='cpu')
        self.attr_embeddings = Dict()

    def encode_attr(self, attr_name, attr_info):
        if attr_name in self.attr_embeddings:
            return np.copy(self.attr_embeddings[attr_name])
        else:
            attr_description = attr_info[attr_name].language_descriptions[0]
            embedding = self.model.encode([attr_description])[0]
            self.attr_embeddings[attr_name] = embedding
            return np.copy(embedding)


def resize_sequence(frames, img_size):
    resized_frames = []
    for i in range(frames.shape[0]):
        img = frames[i, :]
        img = cv2.resize(img,
                         (img_size, img_size),
                         interpolation=cv2.INTER_AREA)
        resized_frames.append(img)
    return np.array(resized_frames).astype(np.uint8)


def get_attr_rep(attr_name, attr_info, rep='one-hot'):
    """
    rep can be ['one-hot', 'language']
    """
    if rep == 'one-hot':
        n_attr = len(attr_info)
        attr_id = attr_info[attr_name].id
        attr_rep = [0 for _ in range(n_attr)]
        attr_rep[attr_id] = 1
        return np.array(attr_rep)
    elif rep == 'language':
        global attr_language_encoder
        if attr_language_encoder is None:
            attr_language_encoder = Attr_Language_Encoder()
        attr_emb = attr_language_encoder.encode_attr(attr_name, attr_info)
        return attr_emb
    else:
        raise NotImplementedError(f'unsupported rep: {rep}')
