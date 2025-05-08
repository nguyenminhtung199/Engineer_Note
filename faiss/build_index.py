from faiss.contrib.ondisk import merge_ondisk
import faiss
import numpy as np
import gc
import os
DIM_FEATURE = 512  

def build_index_faiss(features, labels, path_save_index, efConstruction: int = 200, is_use_gpu: bool = False):
    features = np.array(features, dtype=np.float32)
    labels = np.int64(labels)
    if is_use_gpu:
        res = faiss.StandardGpuResources()
        clustering_index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(DIM_FEATURE))

    is_hnsw = True
    is_flat = False

    if len(features) > 1000000:
        index_name = "IVF262144_HNSW32,SQfp16"
        index = faiss.index_factory(DIM_FEATURE, index_name)
        print("      ++ training index {} with {}".format(index_name, features.shape))
    elif len(features) > 100000:
        index_name = "IVF4096_HNSW32,SQfp16"
        index = faiss.index_factory(DIM_FEATURE, index_name)
        print("      ++ training index {} with {}".format(index_name, features.shape))
    elif len(features) > 10000:
        print("here")
        index_name = "IVF1024,SQfp16"
        index = faiss.index_factory(DIM_FEATURE, index_name)
        print("      ++ training index {} with {}".format(index_name, features.shape))
        is_hnsw = False
    else:
        is_flat = True
        index_name = "IDMap,SQfp16"
        index = faiss.index_factory(DIM_FEATURE, index_name)
        print("      ++ training index FLAT {} with {}".format(index_name, features.shape))
        is_hnsw = False

    print("LEN FEATURE AFTER:", len(features))
    print("IS FLAT:", is_flat)
    print("IS HNSW:", is_hnsw)

    if not is_flat:
        index_ivf = faiss.extract_index_ivf(index)

    if is_use_gpu:
        print('      ++ Reset GPU index')
        clustering_index.reset()
        # garbage
        collected = gc.collect()

        index_ivf.clustering_index = clustering_index

    if is_hnsw:
        # Increase efConstruction
        print('      ++ Setting efConstruction with ', efConstruction)
        faiss.downcast_index(index_ivf.quantizer).hnsw.efConstruction = efConstruction
    print("train")
    # train
    index.train(features)

    print("ivfmodel")
    path_ivfmodel = path_save_index.replace(".bin", ".ivfmodel")
    faiss.write_index(index, path_ivfmodel)

    # on ram
    index.add_with_ids(features, labels)
    faiss.write_index(index, path_save_index)
    del index

    if not is_flat:
        print("merge")
        # merge ondisk
        index_trained = faiss.read_index(path_ivfmodel)
        path_ivfdata = path_save_index.replace(".bin", ".ivfdata")
        block_fnames = [path_save_index]
        merge_ondisk(index_trained, block_fnames, path_ivfdata)
        print("index")
        # save index merge ondisk
        faiss.write_index(index_trained, path_save_index)

    else:
        # save index
        index_trained = faiss.read_index(path_save_index)
        faiss.write_index(index_trained, path_save_index)

    del index_trained

    # remove ivfmodel
    if os.path.exists(path_ivfmodel):
        os.remove(path_ivfmodel)
