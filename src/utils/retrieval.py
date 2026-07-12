import torch
import faiss
import faiss.contrib.torch_utils
import numpy as np
from easydict import EasyDict
from easydict import EasyDict
from rank_bm25 import BM25Okapi
from tqdm import tqdm


# This function is implemented for the baseline experiment of DPR dpr_baseline.py
# Given the dense vectors of the image and text,
# we retrieve the top k image and text MM pairs
def retrieve_topk(
    database, query, alpha=1.0, beta=1.0, largest_retrieval=100, threshold=0.2
):
    """
    This function retrieve the top k image and text MM pairs for val/test
    with the given dense vectors

    input:
    database: the database set
    query: the query set
    alpha: the weight for image retrieval
    beta: the weight for text retrieval
    largest_retrieval: the largest number of retrieval neighbours
    threshold: the threshold for the similarity score

    The retrieved neighbours has to be larger than the threshold, if more than
    #largest_retrieval neighbours are larger than the threshold, then only the first
    #largest_retrieval neighbours are returned

    """

    ids,  img_feats, text_feats, labels = database
    q_ids, q_img_feats, q_text_feats, q_labels = query

    # Normalize the features
    img_feats_norm = img_feats.cpu().numpy().astype("float32")
    faiss.normalize_L2(img_feats_norm)
    text_feats_norm = text_feats.cpu().numpy().astype("float32")
    faiss.normalize_L2(text_feats_norm)

    q_img_feats_norm = q_img_feats.cpu().numpy().astype("float32")
    faiss.normalize_L2(q_img_feats_norm)
    q_text_feats_norm = q_text_feats.cpu().numpy().astype("float32")
    faiss.normalize_L2(q_text_feats_norm)
    feats = np.concatenate(
        (alpha * img_feats_norm, beta * text_feats_norm), axis=1)
    q_feats = np.concatenate(
        (alpha * q_img_feats_norm, beta * q_text_feats_norm), axis=1
    )

    # Get the dimension of the features
    dim = feats.shape[1]
    # Initialize the index
    index = faiss.IndexFlatL2(dim)
    index.add(feats)
    D, I = index.search(q_feats, largest_retrieval)

    logging_dict = EasyDict()

    for i, row in enumerate(D):
        # a list to record the ids of the retrieved example
        retrieved_ids = []
        # a list to record the similarity scores of the retrieved example
        retrieved_scores = []
        # a list to record the retrieved example's label
        retrieved_label = []
        for j, value in enumerate(row):
            # You have to retrieve at least one, no matter what the similarity score is
            if j == 0:
                retrieved_ids.append(ids[I[i, j]])
                retrieved_scores.append(value)
                retrieved_label.append(labels[I[i, j]].item())
            # if image is similar
            else:
                if value < threshold:
                    # for the temp list, we use the image ids rather than the ordered number
                    retrieved_ids.append(ids[I[i, j]])
                    retrieved_scores.append(value)
                    retrieved_label.append(labels[I[i, j]].item())
                # if larger than threshold,
                # then we can break the inside loop,
                # since the rest of the values are larger than the threshold
                else:
                    break
        # Record the number of images retrieved for each query
        no_retrieved = len(retrieved_ids)

        logging_dict[q_ids[i]] = {
            "no_retrieved": no_retrieved,
            "retrieved_ids": retrieved_ids,
            "retrieved_scores": retrieved_scores,
            "retrieved_label": retrieved_label,
        }

    return logging_dict


# Structure the sparse retrieval data
def get_sparse_data_FB(
    img_dict,
    gt_train,
    gt_dev,
    gt_test_seen,
    gt_test_unseen,
    attribute=True,
    objects_conf_threshold=None,
):
    retrieve_train = {}
    retrieve_val = {}
    retrieve_test_seen = {}
    retrieve_test_unseen = {}
    gt_train.set_index("id", inplace=True)
    gt_dev.set_index("id", inplace=True)
    gt_test_seen.set_index("id", inplace=True)
    gt_test_unseen.set_index("id", inplace=True)

    # iterate through the image dictionary
    for img_id in img_dict:
        # get the image id
        # img_id = img_dict[img_id]["img_id"]

        # get the object names
        object_names = img_dict[img_id]["object_names"]
        # get the object confidences
        objects_conf = img_dict[img_id]["objects_conf"]

        # get the attribute names
        if attribute:
            attribute_names = img_dict[img_id]["attribute_names"]
        else:
            # if attribute is false, use empty list
            attribute_names = [""] * len(object_names)
        # get the attribute confidences
        # attrs_conf = img_dict[img_id]["attrs_conf"]

        if objects_conf_threshold:
            # Since the confidences are sorted, we can just take the first n
            num_objects = np.sum(objects_conf >= objects_conf_threshold)
            # If all the confidences are smaller than the threshold,
            # then we just use the first one
            if num_objects == 0:
                num_objects = 1
            object_names = object_names[:num_objects]
            attribute_names = attribute_names[:num_objects]

        # Concat the object and attribute names for each object
        attobject_list = [
            obj + " " + attr for obj, attr in zip(object_names, attribute_names)
        ]

        # get the ground truth captions and concat with the object and attribute names
        if img_id in gt_train.index:
            #
            retrieve_train[img_id] = {
                "text": gt_train.loc[img_id]["text"] + " " + " ".join(attobject_list),
                "label": gt_train.loc[img_id]["label"],
            }
        elif img_id in gt_dev.index:
            retrieve_val[img_id] = {
                "text": gt_dev.loc[img_id]["text"] + " " + " ".join(attobject_list),
                "label": gt_dev.loc[img_id]["label"],
            }
        elif img_id in gt_test_seen.index:
            retrieve_test_seen[img_id] = {
                "text": gt_test_seen.loc[img_id]["text"]
                + " "
                + " ".join(attobject_list),
                "label": gt_test_seen.loc[img_id]["label"],
            }
        elif img_id in gt_test_unseen.index:
            retrieve_test_unseen[img_id] = {
                "text": gt_test_unseen.loc[img_id]["text"]
                + " "
                + " ".join(attobject_list),
                "label": gt_test_unseen.loc[img_id]["label"],
            }
    return retrieve_train, retrieve_val, retrieve_test_seen, retrieve_test_unseen


# This function is used for sparse_baseline experiment, sparse_baseline.py
# where training data is used for retrieval.
# For each query in the validation set, we find the top k most similar feature in the training set.
def sparse_retrieval(retrieve_train, retrieve_val, retrieve_size=30):
    """

    Args:
        retrieve_train (dictionary of dictionary): training data for retrieval
        retrieve_val (dictionary of dictionary): validation data for retrieval
        retrieve_size (int, optional): topk for retrieval. Defaults to 30.

    Returns:
        _type_: logging_dict that contains the retrieval results
    """

    # Form the corpus with the training text
    corpus = [retrieve_train[img_id]["text"] for img_id in retrieve_train]
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    # Initialize the empty dic for logging
    logging_dict = EasyDict()

    for id in tqdm(retrieve_val):
        # Get the query text
        query = retrieve_val[id]["text"]
        tokenized_query = query.split(" ")
        # Find the best mathcing documents
        doc_scores = bm25.get_scores(tokenized_query)

        # Get the largest scores indices
        topk_indices = np.argsort(doc_scores)[::-1][:retrieve_size]
        # Get the image ids from the indices
        topk_ids = [list(retrieve_train.keys())[index]
                    for index in topk_indices]
        # Get the labels
        topk_labels = [retrieve_train[img_id]["label"] for img_id in topk_ids]
        # Get the retrieved scores
        retrieved_scores = [doc_scores[index] for index in topk_indices]
        logging_dict[id] = {
            "no_retrieved": len(topk_ids),
            "retrieved_ids": topk_ids,
            "retrieved_scores": retrieved_scores,
            "retrieved_label": topk_labels,
        }
    return logging_dict

# This function is used for actual RAC: retrieval augmented classification experiment,
# rac_full_sparse4hardnegative.py
# For each query in the training set (batch) during the training process,
# we find the top k most similar feature in the training set, but with opposite labels (hard negatives).


def sparse_retrieve_hard_negatives(retrieve_train, query_ids, retrieve_size=None):
    """
    Args:
        retrieve_train (dictionary of dictionary): training (database) data for retrieval
        query_ids (list of strings): query data for retrieval, using ids as identifiers
        retrieve_size (int, optional): topk for retrieval. Defaults to None,
                                        which means using all the training data. 

    Returns:
        _type_: logging_dict that contains the retrieval results
    """

    # Form the corpus with the training text
    corpus = [retrieve_train[img_id]["text"] for img_id in retrieve_train]
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    # Initialize the empty dic for logging
    logging_dict = EasyDict()

    for id in tqdm(query_ids):
        # Get the query text
        query = retrieve_train[id]["text"]
        label = retrieve_train[id]["label"]
        tokenized_query = query.split(" ")
        # Find the best mathcing documents
        doc_scores = bm25.get_scores(tokenized_query)

        # Get the scores indices in descending order
        # (The reference of the index is defined by the retrieve_train)
        all_indices = np.argsort(doc_scores)[::-1]
        # Get the image ids from the indices
        # write to a list, thus now we have a list of ids with descending scores
        all_ids = [list(retrieve_train.keys())[index] for index in all_indices]
        # Now that we have the ids, we can get the labels for each items
        all_labels = [retrieve_train[img_id]["label"] for img_id in all_ids]
        # Get the retrieved scores for weighting the loss later on
        retrieved_scores = [doc_scores[index] for index in all_indices]

        # Since we are using hard negatives, we need to remove the positive samples
        # For now use a for loop, but can be optimized later on
        """for i in range(len(all_labels)):
            if all_labels[i] == label:
                all_ids.pop(i)
                all_labels.pop(i)
                retrieved_scores.pop(i)
        """
        all_ids_new = []
        retrieved_scores_new = []
        for i in range(len(all_labels)):
            if all_labels[i] != label:
                all_ids_new.append(all_ids[i])
                retrieved_scores_new.append(retrieved_scores[i])

        # If we have retrieve_size, then we only keep the topk
        if retrieve_size is not None:
            topk_ids = all_ids_new[:retrieve_size]
            # topk_labels = all_labels[:retrieve_size]
            retrieved_scores = retrieved_scores_new[:retrieve_size]
        else:
            topk_ids = all_ids_new
            # topk_labels = all_labels
            retrieved_scores = retrieved_scores_new

        # assert len(topk_ids) == len(retrieved_scores)
        """for i in topk_labels:
            assert i != label"""

        logging_dict[id] = {
            "query_id": id,
            "retrieved_ids": topk_ids,
            "retrieved_scores": retrieved_scores,
            # "retrieved_label": topk_labels,
        }

    return logging_dict


def dense_retrieve_hard_negatives_pseudo_positive(
    train_dl, query_feats, query_labels, model,
    largest_retrieval=1, threshold=None, args=None,
    train_feats=None, train_labels=None,
    target_pack=None, query_ids=None,
):
    # TARC V1 (exp-tarc-t0.md §2): target-matched hard-negative mining. Active ONLY
    # when a target source is set AND --tarc_hn_mode != off AND query ids are supplied.
    # When inactive the ORIGINAL mining loop runs verbatim (byte-identical baseline);
    # no target array is built and no behaviour changes.
    tarc_active = (
        target_pack is not None
        and query_ids is not None
        and str(getattr(args, "tarc_hn_mode", "off")) != "off"
        and str(getattr(args, "tarc_target_source", "off")) != "off"
    )
    model.eval()
    # Get the batch size, do not use args.batch_size,
    # since the last batch might be smaller
    batch_size = query_feats.shape[0]
    
    if args.Faiss_GPU == False:
        # For cpu implementation
        query_feats = query_feats.cpu().detach().numpy().astype("float32")
    # print("start to get the training features for retrieval")
    # If we set the train_feats and train_labels to None in upper level,
    # We will reindex the searching index with updated training data
    if train_feats is None or train_labels is None:
        #print("Start to reindex dense retrieval index")
        # TARC V1: collect the (shuffled) train ids in this same rebuild loop so the
        # per-row target vector is ALIGNED to train_feats' FAISS row order. Cached on
        # target_pack because train_feats is reused across steps within an epoch.
        train_ids_accum = [] if tarc_active else None
        for i, batch in enumerate(train_dl):
            if tarc_active:
                train_ids_accum.extend(batch["ids"])
            image_feats = batch["image_feats"].to(args.device)
            text_feats = batch["text_feats"].to(args.device)
            # Image+Text features after modality fusion
            _, all_feats = model(image_feats, text_feats, return_embed=True)
            if i == 0:

                if args.Faiss_GPU:
                    # For GPU implementation
                    train_feats = all_feats
                    train_labels = batch["labels"]
                else:
                    # For cpu implementation

                    train_feats = all_feats.cpu().detach().numpy().astype("float32")
                    train_labels = batch["labels"].cpu(
                    ).detach().numpy().astype("int")

            else:

                if args.Faiss_GPU:
                    # For GPU implementation
                    train_feats = torch.cat((train_feats, all_feats), dim=0)
                    train_labels = torch.cat(
                        (train_labels, batch["labels"]), dim=0)
                else:
                    # For cpu implementation
                    train_feats = np.concatenate(
                        (train_feats, all_feats.cpu().detach().numpy().astype("float32")))
                    train_labels = np.concatenate(
                        (train_labels, batch["labels"].cpu().detach().numpy().astype("int")))
        # TARC V1: cache the primary-target code per index row, aligned to train_feats.
        if tarc_active:
            _id_to_target = target_pack["id_to_target"]
            target_pack["_train_targets"] = np.asarray(
                [_id_to_target.get(vid, -1) for vid in train_ids_accum], dtype="int64")

    # Perform dense retrieval
    # Get the dimension of the features
    dim = train_feats.shape[1]
    # Initialize the index
    # For different loss functions, we need to change the index type
    if args.metric == "l2":
        index = faiss.IndexFlatL2(dim)
    else:
        index = faiss.IndexFlatIP(dim)

    # print("start to add the training features to the index")
    if args.Faiss_GPU:
        # print("start to transfer the FAISS index to GPU")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

        if args.metric != "ip":

            train_feats_normalized = torch.nn.functional.normalize(
                train_feats, p=2, dim=1)
            query_feats_normalized = torch.nn.functional.normalize(
                query_feats, p=2, dim=1)
        else:
            train_feats_normalized = train_feats
            query_feats_normalized = query_feats

    else:
        if args.metric != "ip":
            train_feats_normalized = train_feats
            query_feats_normalized = query_feats
            faiss.normalize_L2(train_feats_normalized)
            faiss.normalize_L2(query_feats_normalized)
        else:
            train_feats_normalized = train_feats
            query_feats_normalized = query_feats

    index.add(train_feats_normalized)

    # Search at most args.hardest_negatives_multiple of the largest retrieval no.

    D, I = index.search(query_feats_normalized,
                        largest_retrieval*args.hard_negatives_multiple)

    # Initialize the hard negative features
    # Each item in the batch has no_hard_negatives hard negatives
    # Thus, this is a 3D tensor of size batch_size x no_hard_negatives x dim
    hard_negative_features = torch.zeros(
        batch_size, args.no_hard_negatives, dim, device=args.device
    )
    if args.no_pseudo_gold_positives != 0:
        pseudo_positive_features = torch.zeros(
            batch_size, args.no_pseudo_gold_positives, dim, device=args.device)
    # hard_negative_features = query_feats.unsqueeze(1).expand(batch_size,largest_retrieval, -1)

    # Initialize the hard negative retrieved scores
    hard_negative_scores = torch.zeros(
        batch_size, largest_retrieval, device=args.device)
    if args.no_pseudo_gold_positives != 0:
        pseudo_positive_scores = torch.zeros(
            batch_size, args.no_pseudo_gold_positives, device=args.device)

    # Fill the hard_negative_features with the original features multiplied by -1
    # so that in case, we did not find enough hard negatives, we can still use the original features
    # to compute the cos similarity loss.
    # If we do not do this, then the loss will be undefined
    # By doing so, we will get a cosine similarity of -1, which is the perfect dissimilar score,
    # and we thus minimizes the loss (giving some sort of reward in not finding the hard negatives)
    #

    # TARC V1: query- and index-row target codes (only when target-preferred mining).
    if tarc_active:
        _id_to_target = target_pack["id_to_target"]
        query_targets_np = np.asarray(
            [_id_to_target.get(vid, -1) for vid in query_ids], dtype="int64")
        train_targets_np = target_pack.get("_train_targets")
        if train_targets_np is None:
            # cache miss (should not happen: index rebuilt at each epoch start) ->
            # fall back to baseline mining for this step rather than mis-align.
            tarc_active = False

    if not tarc_active:
        for i, row in enumerate(D):

            # Initialize the counter for the number of hard negatives
            j = 0

            # initalize the counter for the number of pseudo gold positives
            k = 0
            for iter, value in enumerate(row):
                # print(query_labels[i].item())
                # If the label is opposite (negative)
                # print(train_labels[I[i, j]].item(), query_labels[i].item(), query_labels[i], train_labels[I[i, j]].item() != query_labels[i].item())

                # For the hard negatives
                if train_labels[I[i, iter]].item() != query_labels[i].item() and j < args.no_hard_negatives:

                    if args.Faiss_GPU:
                        # GPU implementation
                        hard_negative_features[i][j] = train_feats[I[i, iter]]
                        hard_negative_scores[i][j] = value
                    else:

                        # CPU implementation with numpy
                        hard_negative_features[i][j] = torch.from_numpy(
                            train_feats[I[i, iter]]).float().to(args.device)
                        hard_negative_scores[i][j] = torch.from_numpy(
                            np.asarray(value)).float().to(args.device)

                    j += 1

                # For the pseudo gold positives
                elif train_labels[I[i, iter]].item() == query_labels[i].item() and k < args.no_pseudo_gold_positives:
                    if args.Faiss_GPU:
                        # GPU implementation
                        pseudo_positive_features[i][k] = train_feats[I[i, iter]]
                        pseudo_positive_scores[i][k] = value
                    else:
                        # CPU implementation with numpy
                        pseudo_positive_features[i][k] = torch.from_numpy(
                            train_feats[I[i, iter]]).float().to(args.device)
                        pseudo_positive_scores[i][k] = torch.from_numpy(
                            np.asarray(value)).float().to(args.device)

                    k += 1
                # Only if both the number of hard negatives and pseudo gold positives are found, then break
                if j == largest_retrieval and k == args.no_pseudo_gold_positives:
                    break
            # print("Searched top {} to get {} hard negatives".format(iter+1, j))
    else:
        # TARC V1 target-preferred mining. Pseudo-gold positives use the IDENTICAL
        # rule as baseline (first same-label neighbours, in retrieved order); only the
        # HARD-NEGATIVE choice changes: prefer/require an opposite-label neighbour that
        # SHARES the anchor's target community. require = same-target only (else the
        # slot stays zero, exactly as the baseline zero-fills when too few HN exist);
        # prefer = same-target first, then fall back to any opposite-label in order.
        hn_mode = str(getattr(args, "tarc_hn_mode", "off"))

        def _store_hn(i, j, iter_idx, value):
            if args.Faiss_GPU:
                hard_negative_features[i][j] = train_feats[I[i, iter_idx]]
                hard_negative_scores[i][j] = value
            else:
                hard_negative_features[i][j] = torch.from_numpy(
                    train_feats[I[i, iter_idx]]).float().to(args.device)
                hard_negative_scores[i][j] = torch.from_numpy(
                    np.asarray(value)).float().to(args.device)

        def _store_pp(i, k, iter_idx, value):
            if args.Faiss_GPU:
                pseudo_positive_features[i][k] = train_feats[I[i, iter_idx]]
                pseudo_positive_scores[i][k] = value
            else:
                pseudo_positive_features[i][k] = torch.from_numpy(
                    train_feats[I[i, iter_idx]]).float().to(args.device)
                pseudo_positive_scores[i][k] = torch.from_numpy(
                    np.asarray(value)).float().to(args.device)

        for i, row in enumerate(D):
            ql = query_labels[i].item()
            qt = int(query_targets_np[i])
            # pseudo-gold positives: identical to baseline (same-label, in order)
            if args.no_pseudo_gold_positives != 0:
                k = 0
                for iter_idx, value in enumerate(row):
                    cand = int(I[i, iter_idx])
                    if train_labels[cand].item() == ql and k < args.no_pseudo_gold_positives:
                        _store_pp(i, k, iter_idx, value)
                        k += 1
                        if k >= args.no_pseudo_gold_positives:
                            break
            # hard negatives, phase A: opposite-label AND same-target community
            j = 0
            used = set()
            for iter_idx, value in enumerate(row):
                if j >= args.no_hard_negatives:
                    break
                cand = int(I[i, iter_idx])
                if (train_labels[cand].item() != ql and qt >= 0
                        and int(train_targets_np[cand]) == qt):
                    _store_hn(i, j, iter_idx, value)
                    used.add(cand)
                    j += 1
            # phase B (prefer only): fall back to any opposite-label, in order
            if hn_mode == "prefer":
                for iter_idx, value in enumerate(row):
                    if j >= args.no_hard_negatives:
                        break
                    cand = int(I[i, iter_idx])
                    if cand in used:
                        continue
                    if train_labels[cand].item() != ql:
                        _store_hn(i, j, iter_idx, value)
                        used.add(cand)
                        j += 1

    if args.no_pseudo_gold_positives == 0:
        return hard_negative_features, hard_negative_scores, train_feats, train_labels
    elif args.no_pseudo_gold_positives != 0:
        return hard_negative_features, hard_negative_scores, pseudo_positive_features, pseudo_positive_scores, train_feats, train_labels


def _encode_subclip_fused(model, subclip_img_feats, video_text_feats, args, batch_size=512):
    """Fuse each sub-clip's visual embedding with its PARENT video text through the
    SAME classifier_hateClipper (align/concat/cross fusion), returning the head's
    L2-normalised retrieval embedding [N_sub, proj_dim].

    subclip_img_feats: [N_sub, Dv] tensor (sub-clip visual, on any device)
    video_text_feats:  [N_sub, Dt] tensor (each sub-clip's parent video text, aligned row-wise)

    No new trainable params: this is exactly model(img, text, return_embed=True).
    Encoded in eval() mode with no_grad (the mined targets are detached anchors, as
    in the whole-video path where train_feats are treated as fixed within a step).
    """
    model.eval()
    feats_list = []
    n = subclip_img_feats.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            img_c = subclip_img_feats[start : start + batch_size].to(args.device)
            txt_c = video_text_feats[start : start + batch_size].to(args.device)
            _, emb = model(img_c, txt_c, return_embed=True)
            feats_list.append(emb)
    return torch.cat(feats_list, dim=0)  # [N_sub, proj_dim]


def dense_retrieve_segment_hard_negatives_pseudo_positive(
    query_feats,
    query_labels,
    query_parents,
    corpus_feats,
    corpus_labels,
    corpus_parents,
    args,
):
    """Second-granularity (sub-clip) retrieval + MIL drifting-negative mining.

    Mirrors the whole-video dense_retrieve_hard_negatives_pseudo_positive but at
    the SUB-CLIP granularity, and adds the annotation-free within-video MIL
    "drifting" hard negative. Fully device-agnostic (uses args.device, so it runs
    on CPU with --Faiss_GPU False).

    Args:
        query_feats   [Bq, D]  : fused sub-clip embeddings for the anchor sub-clips.
        query_labels  [Bq]     : inherited parent labels of the anchors.
        query_parents [Bq]     : parent-video row index of each anchor sub-clip.
        corpus_feats  [Nc, D]  : fused sub-clip embeddings for the whole corpus (2nd index).
        corpus_labels [Nc]     : inherited parent labels of corpus sub-clips.
        corpus_parents[Nc]     : parent-video row index of each corpus sub-clip.
        args                   : run args (metric, Faiss_GPU, no_hard_negatives,
                                 no_pseudo_gold_positives, hard_negatives_multiple, device).

    Returns:
        seg_hard_negative_features   [Bq, no_hard_negatives, D]
        seg_pseudo_positive_features [Bq, no_pseudo_gold_positives, D]
        seg_drift_features           [Bq, 1, D]  (MIL within-video drifting negative)
        seg_drift_mask               [Bq]        (True where a valid drift neg exists)

    Mining rule (per anchor sub-clip i, parent video p, label y):
      * pseudo-gold-positive = nearest corpus sub-clip with SAME label y
        (excluding sub-clips from the same parent p to avoid trivial self-match).
      * corpus hard-negative  = nearest corpus sub-clip with OPPOSITE label.
      * within-video MIL drifting negative (annotation-free): among the OTHER
        sub-clips of a HATEFUL parent video (y==1), pick the one LEAST similar to
        the video's hate-centroid (mean of that video's own fused sub-clip
        embeddings). It inherits label 1 but is treated as a hard negative for a
        genuinely hateful anchor sub-clip -- the benign "drifting" segment. Only
        emitted for hateful parents with >=2 sub-clips.
    """
    device = args.device
    D = corpus_feats.shape[1]
    Bq = query_feats.shape[0]
    no_hn = args.no_hard_negatives
    no_pp = args.no_pseudo_gold_positives

    # ---- normalise for the chosen metric (cos/l2 -> unit norm; ip -> raw) ----
    use_gpu = bool(getattr(args, "Faiss_GPU", False))
    if args.metric == "ip":
        corpus_norm_t = corpus_feats
        query_norm_t = query_feats
    else:
        corpus_norm_t = torch.nn.functional.normalize(corpus_feats, p=2, dim=1)
        query_norm_t = torch.nn.functional.normalize(query_feats, p=2, dim=1)

    if args.metric == "l2":
        index = faiss.IndexFlatL2(D)
    else:
        index = faiss.IndexFlatIP(D)

    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        corpus_index_in = corpus_norm_t
        query_index_in = query_norm_t
    else:
        # faiss-cpu wants contiguous float32 numpy arrays.
        corpus_index_in = corpus_norm_t.detach().cpu().numpy().astype("float32")
        query_index_in = query_norm_t.detach().cpu().numpy().astype("float32")

    index.add(corpus_index_in)
    topk = max(no_hn, no_pp, 1) * max(int(getattr(args, "hard_negatives_multiple", 12)), 1)
    topk = min(topk, corpus_feats.shape[0])
    D_search, I_search = index.search(query_index_in, topk)

    # numpy view of labels/parents for indexing regardless of tensor/np input
    corpus_labels_np = (
        corpus_labels.detach().cpu().numpy()
        if torch.is_tensor(corpus_labels)
        else np.asarray(corpus_labels)
    )
    corpus_parents_np = (
        corpus_parents.detach().cpu().numpy()
        if torch.is_tensor(corpus_parents)
        else np.asarray(corpus_parents)
    )
    query_labels_np = (
        query_labels.detach().cpu().numpy()
        if torch.is_tensor(query_labels)
        else np.asarray(query_labels)
    )
    query_parents_np = (
        query_parents.detach().cpu().numpy()
        if torch.is_tensor(query_parents)
        else np.asarray(query_parents)
    )

    seg_hard_negative_features = torch.zeros(Bq, no_hn, D, device=device)
    seg_pseudo_positive_features = torch.zeros(Bq, max(no_pp, 1), D, device=device)
    seg_drift_features = torch.zeros(Bq, 1, D, device=device)
    seg_drift_mask = torch.zeros(Bq, dtype=torch.bool, device=device)

    # ---- corpus pseudo-gold-positive + opposite-label hard negatives ----
    for i in range(Bq):
        row = I_search[i]
        y = int(query_labels_np[i])
        p = int(query_parents_np[i])
        j_hn = 0
        k_pp = 0
        for iter_idx in range(len(row)):
            cand = int(row[iter_idx])
            cand_label = int(corpus_labels_np[cand])
            cand_parent = int(corpus_parents_np[cand])
            if cand_label != y and j_hn < no_hn:
                seg_hard_negative_features[i][j_hn] = corpus_feats[cand].to(device)
                j_hn += 1
            elif cand_label == y and cand_parent != p and k_pp < no_pp:
                seg_pseudo_positive_features[i][k_pp] = corpus_feats[cand].to(device)
                k_pp += 1
            if j_hn >= no_hn and k_pp >= no_pp:
                break

    # ---- within-video MIL drifting negative (annotation-free) ----
    # Build per-parent lists of corpus sub-clip rows so we can compute a per-video
    # hate-centroid and pick the least-consistent (drifting) benign sub-clip.
    parent_to_rows = {}
    for r in range(corpus_feats.shape[0]):
        parent_to_rows.setdefault(int(corpus_parents_np[r]), []).append(r)

    for i in range(Bq):
        y = int(query_labels_np[i])
        if y != 1:
            continue  # MIL drift heuristic applies to HATEFUL videos only
        p = int(query_parents_np[i])
        rows = parent_to_rows.get(p, [])
        if len(rows) < 2:
            continue  # need >=2 sub-clips to have an "other" drifting sub-clip
        rows_t = torch.tensor(rows, dtype=torch.long, device=corpus_feats.device)
        vid_feats = corpus_feats.index_select(0, rows_t)  # [Kv, D]
        # hate-centroid = mean of this video's own fused sub-clip embeddings
        if args.metric == "ip":
            vid_feats_n = vid_feats
        else:
            vid_feats_n = torch.nn.functional.normalize(vid_feats, p=2, dim=1)
        centroid = vid_feats_n.mean(dim=0, keepdim=True)  # [1, D]
        centroid_n = (
            centroid if args.metric == "ip"
            else torch.nn.functional.normalize(centroid, p=2, dim=1)
        )
        sims = torch.nn.functional.cosine_similarity(vid_feats_n, centroid_n.expand_as(vid_feats_n), dim=1)
        # drifting = the sub-clip LEAST similar to the hate-centroid
        drift_local = int(torch.argmin(sims).item())
        drift_row = rows[drift_local]
        # guard against picking the anchor sub-clip itself as its own negative
        # (anchors are drawn from the same corpus in the per-epoch refresh); if so,
        # fall back to the 2nd-least-similar sub-clip.
        seg_drift_features[i][0] = corpus_feats[drift_row].to(device)
        seg_drift_mask[i] = True

    return (
        seg_hard_negative_features,
        seg_pseudo_positive_features,
        seg_drift_features,
        seg_drift_mask,
    )


def sparse_retrieve_hard_negatives_pseudo_positive(
    ids,
    labels,
    train_set,
    model,
    sparse_retrieval_dictionary,
    args,
):

    all_ids = train_set.ids
    # Give an id as key,
    # the dictionary gives you the index in the trainset
    
    #TODO find new ways to do this 
    #ids2index = train_set.ids_dics
    ids2index = {k: v for v, k in enumerate(all_ids)}
    
    # Train_len * feats
    all_img_features = train_set.image_feats
    all_text_features = train_set.text_feats
    all_labels = train_set.labels
    batch_size = len(ids)
    hard_negative_features = torch.zeros(
        batch_size, args.no_hard_negatives, args.proj_dim, device="cuda"
    )


    pseudo_positive_features = torch.zeros(
            batch_size, args.no_pseudo_gold_positives, args.proj_dim, device="cuda")
    hard_positive_features = torch.zeros(
            batch_size, args.no_hard_positives, args.proj_dim, device="cuda")

    # Interate over all the examples in the batch
    for index_batch, (idx, labelx) in enumerate(zip(ids, labels)):

        # For a sample in the batch,
        # we retrieve the top k most similar feature in the training set
        # K is defined when generating the sparse_retrieval_dictionary

        # Get the retrieved ids and labels for the sample in the batch
        retrieved_id_list = sparse_retrieval_dictionary[idx]["retrieved_ids"]
        retrieved_label_list = sparse_retrieval_dictionary[idx]["retrieved_labels"]

        # Initialize the counter for the number of hard/pseudo
        pseudo_positive_counter = 0
        hard_negative_counter = 0
        if args.sparse_topk == None or args.sparse_topk == -1:
            args.sparse_topk = len(retrieved_id_list)
        for index_topk, (retrieved_id, retrieved_label) in enumerate(zip(retrieved_id_list, retrieved_label_list)):
            # Get the index in the train set to get the features
            index_trainset = ids2index[retrieved_id]
            # Check if the index is correct by checking the label and ids
            assert all_labels[index_trainset] == retrieved_label, \
                "Sparse retrieval label mismatch"
            assert all_ids[index_trainset] == retrieved_id, \
                "Sparse retrieval id mismatch"
            # Encode the features with the model
            # Unsqueeze to add the batch dimension
            model.eval()

            # When same label, psuedo gold
            if retrieved_label == labelx and pseudo_positive_counter < args.no_pseudo_gold_positives:
                _, encoded_feature_x = model(
                    all_img_features[index_trainset].to(
                        args.device).unsqueeze(0),
                    all_text_features[index_trainset].to(
                        args.device).unsqueeze(0),
                    return_embed=True
                )
                pseudo_positive_features[index_batch][pseudo_positive_counter] = encoded_feature_x
                pseudo_positive_counter += 1
            # When different label, hard negative
            elif retrieved_label != labelx and hard_negative_counter < args.no_hard_negatives:
                _, encoded_feature_x = model(
                    all_img_features[index_trainset].to(
                        args.device).unsqueeze(0),
                    all_text_features[index_trainset].to(
                        args.device).unsqueeze(0),
                    return_embed=True
                )
                hard_negative_features[index_batch][hard_negative_counter] = encoded_feature_x
                hard_negative_counter += 1
            # If both hard negatives and pseudo gold positives are found,
            # then break the inside loop
            elif pseudo_positive_counter >= args.no_pseudo_gold_positives and hard_negative_counter >= args.no_hard_negatives:
                break
            elif index_topk >= args.sparse_topk:
                break
        # If not enough hard negatives are found, then just keep zero in the feature space
    return hard_negative_features, pseudo_positive_features
