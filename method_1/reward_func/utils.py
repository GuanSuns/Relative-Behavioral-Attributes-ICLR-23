import numpy as np


def to_attrs_vec(raw_attributes, encoded_attr_info):
    attr2id = {attr: encoded_attr_info[attr].id for attr in encoded_attr_info}
    n_attr = len(encoded_attr_info)

    attr_vectors = list()
    for raw_scores in raw_attributes:
        attr_vec = [0 for _ in range(n_attr)]
        for attr in raw_scores:
            attr_id = attr2id[attr]
            attr_min, attr_max = encoded_attr_info[attr].min, encoded_attr_info[attr].max
            attr_vec[attr_id] = (raw_scores[attr] - attr_min) / (attr_max - attr_min)
        attr_vectors.append(attr_vec)
    return np.array(attr_vectors).astype(np.float32)
