import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
from utils.retrieval import (
    dense_retrieve_hard_negatives_pseudo_positive,
    sparse_retrieve_hard_negatives_pseudo_positive,
    dense_retrieve_segment_hard_negatives_pseudo_positive,
    _encode_subclip_fused,
)


def compute_loss(batch,
                train_dl,
                model,
                args,
                train_set=None,
                sparse_retrieval_dictionary=None,
                train_feats=None,
                train_labels=None,
                segment_cache=None,
                aux_pack=None,
                cf_pack=None,
                target_pack=None,
                ):
    ids = batch["ids"]
    batch_size = len(ids)
    image_feats = batch["image_feats"].to(args.device)
    text_feats = batch["text_feats"].to(args.device)
    labels = batch["labels"].to(args.device)
    model.train()
    output, feats = model(image_feats, text_feats, return_embed=True)

    # We construct a matrix for label coincidences (Mask matrix for later loss computation)
    # 1 if the labels are the same (positive), 0 otherwise (negative)
    # The dimension would be batch_size x batch_size
    # This is used for the in-batch positive/negative mining

    # We construct it by stacking rows of the labels
    # then for ith row with label 0, we flip the label bit for whole row.

    # We can do this since, if the original label is 0 and
    # the target label is 0, then we have in-batch positive (1);
    # if the target label is 1, then we have in-batch negative (0)
    # Thus we flip the label for 0.

    # We first construct the inverse label, i.e., binary NOT operator on the label
    # Vectors of 1s and 0s of size batch_size
    labels = labels.bool()
    labels_inverse = ~labels
    # print(labels)
    # Matrix of size batch_size x batch_size
    label_matrix = torch.stack(
        [
            labels if labels[i] == True else labels_inverse
            for i in range(batch_size)
        ],
        axis=0,
    )
    # Bool to int conversion
    if args.no_pseudo_gold_positives == 0:
        label_matrix_positive = label_matrix.int()
    # print(label_matrix_positive)
    # FLip
    label_matrix_negative = (~label_matrix).int()
    # print(label_matrix_negative)
    # We then compute the number of in-batch positives and negatives per sample in the batch
    # vectors of sizes batch_size
    # Since the matrix is symmetric, use which dimension does not matter
    # -1 for minus the sample itself

    in_batch_positives_no = torch.sum(label_matrix, dim=1) - 1
    in_batch_negative_no = batch_size - in_batch_positives_no - 1

    # We then construct the similarity matrix by computing the
    # choice of loss function:
    # 1. cosine similarity
    # 2. Triplet loss
    # 3. Manhatten distance

    # We expand the feature matrix to a 3D tensor for vectorized computation
    # feats_expand Dimension: batch_size x feature_size x batch_size
    feats_expanded = feats.unsqueeze(
        2).expand(batch_size, -1, batch_size)

    if args.metric == "cos":
        cos = nn.CosineSimilarity(dim=1, eps=1e-8)

        # We compute the cosine similarity between each pair of features
        sim_matrix = cos(
            feats_expanded, feats_expanded.transpose(0, 2))
    elif args.metric == "ip":
        # might be wrong
        sim_matrix = torch.sum(
            feats_expanded * feats_expanded.transpose(0, 2), dim=1) / args.proj_dim
    elif args.metric == "l2":
        # l2 = nn.PairwiseDistance(p=2, eps=1e-8)
        # Poor vectorized implementation for pairwise distance
        # We use mse instead for vectorized computation

        """
        l2 = torch.nn.MSELoss(reduction='none')
        sim_matrix = l2(
            feats_expanded, feats_expanded.transpose(0, 2)).sum(dim=1) / args.proj_dim
        """
        # sim_matrix = torch.sum(torch.square((feats_expanded - feats_expanded.transpose(0, 2))), dim=1)
        sim_matrix = compute_l2(feats_expanded, feats_expanded.transpose(
            0, 2), normalise=args.norm_feats_loss, sum_dim=1, sqrt=args.l2_sqrt)
        # Add a negative sign here to account for the fact that
        # L2 is a distance measure, not a similarity measure, larger is more distant (dissimilar),
        # Where in similarity measure, larger is more similar

        # SQRT here gives NAN, thus we minimize the square of the L2 distance
        sim_matrix = - sim_matrix / args.proj_dim

    # The diagonal of the similarity matrix is 1,
    # which is the similarity of the same pair
    # Thus replace it with 0
    sim_matrix.fill_diagonal_(0)

    # We compute the loss matrix by multiplying the similarity matrix
    in_batch_negative_loss = sim_matrix * label_matrix_negative

    if args.no_pseudo_gold_positives == 0:
        in_batch_positives_loss = sim_matrix * label_matrix_positive
    else:
        # If we use pseudo gold positives, we do not use in batch positives
        # We set it to a matrix of zeros to make sure the contrastive loss can still use the same code
        in_batch_positives_loss = torch.zeros(
            batch_size, batch_size).to(args.device)
    # Wrong implementation since division by zero might happen
    """in_batch_loss = torch.sum(in_batch_negative_loss) / torch.sum(
        in_batch_negative_no
    ) - torch.sum(in_batch_positives_loss) / torch.sum(in_batch_positives_no)"""

    # We compute the loss by summing over the loss matrix

    # V1 implementation
    #  If there is no in_batch_negative, we set the loss to 0 to avoid division by zero
    """    if torch.sum(in_batch_negative_no) == 0:
        in_batch_negative_loss_sum = 0
    else:
        in_batch_negative_loss_sum = torch.sum(
            in_batch_negative_loss
        ) / torch.sum(in_batch_negative_no) 
    """

    # V2 implementation
    """in_batch_negative_loss_sum = torch.zeros(batch_size).to(args.device)
    in_batch_positives_loss_sum = torch.zeros(batch_size).to(args.device)
    for i in range(batch_size):
        if in_batch_negative_no[i] == 0:
            in_batch_negative_loss_sum[i] = 0
        else:
            in_batch_negative_loss_sum[i] = torch.sum(in_batch_negative_loss[i]) / in_batch_negative_no[i]"""

    # V3 implementation masking and replacing nan by zero
    """
    ## in_batch_negative_loss_sum = torch.sum(in_batch_negative_loss, dim=1) / in_batch_negative_no
    
    # Assert the number of samples from in_batch_negative_no and the number of non zero elements in in_batch_negative_loss are the same
    mask = in_batch_negative_loss != 0
    in_batch_negative_loss_sum = torch.sum(in_batch_negative_loss, dim=1) / mask.sum(dim=1)
    # Check if there is nan, if so replace it by zero
    in_batch_negative_loss_sum[torch.isnan(in_batch_negative_loss_sum)] = 0
    """
    # V4 implementation, doing the loss with the mask rather than detect if there is nan and set to zero:
    # Pick out the non-zero terms (gives 1), mask out the zero terms (gives 0)
    neg_mask = in_batch_negative_loss != 0

    # Dim batch_size, count the number of zeros for each sample in the batch,
    neg_zero_count = (neg_mask == 0).sum(dim=1)

    # However, if all the terms are zero, we will get nan due to zero division,
    # We will form a further mask to only operate on the sample with at least one non-zero term
    neg_zero_count_zero_mask = torch.zeros(batch_size, device=args.device) != in_batch_negative_no

    in_batch_negative_loss_sum = torch.zeros(batch_size, device=args.device)
    in_batch_negative_loss_sum[neg_zero_count_zero_mask] = torch.sum(
        in_batch_negative_loss[neg_zero_count_zero_mask], dim=1) / neg_mask.sum(dim=1)[neg_zero_count_zero_mask]

    # Only use in-batch positive if we do not use pseudo gold positive samples
    if args.no_pseudo_gold_positives == 0:
        # V1 implementation
        """if torch.sum(in_batch_positives_no) == 0:
            in_batch_positives_loss_sum = 0
        else:
            in_batch_positives_loss_sum = torch.sum(
                in_batch_positives_loss
            ) / torch.sum(in_batch_positives_no) """

        # V2 implementation
        """for i in range(batch_size):
            if in_batch_positives_no[i] == 0:
                in_batch_positives_loss_sum[i] = 0
            else:
                in_batch_positives_loss_sum[i] = torch.sum(in_batch_positives_loss[i]) / in_batch_positives_no[i]"""

        """# V3 implementation masking and replacing nan by zero
        ## in_batch_positives_loss_sum = torch.sum(in_batch_positives_loss, dim=1) / in_batch_positives_no
        mask = in_batch_positives_loss != 0
        in_batch_positives_loss_sum = torch.sum(in_batch_positives_loss, dim=1) / mask.sum(dim=1)
        # Check if there is nan, if so replace it by zero
        in_batch_positives_loss_sum[torch.isnan(in_batch_positives_loss_sum)] = 0"""
        # V4 implementation, doing the loss with the mask rather than detect if there is nan and set to zero:
        # Pick out the non-zero terms (gives 1), mask out the zero terms (gives 0)
        pos_mask = in_batch_positives_loss != 0
        # Dim batch_size, count the number of zeros for each sample in the batch,
        pos_zero_count = (pos_mask == 0).sum(dim=1)
        # However, if all the terms are zero, we will get nan due to zero division,
        # We will form a further mask to only operate on the sample with at least one non-zero term
        pos_zero_count_zero_mask = pos_zero_count != in_batch_positives_no

        in_batch_positives_loss_sum = torch.zeros(
            batch_size, device=args.device)
        in_batch_positives_loss_sum[pos_zero_count_zero_mask] = torch.sum(
            in_batch_positives_loss[pos_zero_count_zero_mask], dim=1) / pos_mask.sum(dim=1)[pos_zero_count_zero_mask]

    # If we use pseudo gold positives, we do not use in batch positives
    else:
        in_batch_positives_loss_sum = 0
    in_batch_loss = in_batch_negative_loss_sum - in_batch_positives_loss_sum

    # Sanity check

    """print("feature vector")
    print(feats.shape),
    print(feats)
    print("in-batch loss")
    print(in_batch_negative_loss.shape,
        in_batch_negative_no.shape, in_batch_negative_loss, in_batch_negative_no, torch.mean(in_batch_negative_loss, dim=1))
    
    print(in_batch_positives_loss.shape,
        in_batch_positives_no.shape, in_batch_positives_loss, in_batch_positives_no, torch.mean(in_batch_positives_loss, dim=1))
    
    print("in-batch positive loss sum:", in_batch_positives_loss_sum)
    print("in-batch negative loss sum:", in_batch_negative_loss_sum)
    print("in-batch loss sum:", in_batch_loss)
    """

    # ----------------- Hard Negative Retrieval and Pseudo Gold Positive -----------------
    # retrieve hard negatives and pseudo gold with Dense retrieval

    # Only hard negative
    #print("start to retrieve hard negatives and pseudo gold")
    if args.sparse_dictionary is None:
        if args.hard_negatives_loss and args.no_pseudo_gold_positives == 0:
            (
                hard_negative_features,
                hard_negative_scores,
                train_feats, 
                train_labels,
            ) = dense_retrieve_hard_negatives_pseudo_positive(
                train_dl,
                feats,
                labels,
                model,
                largest_retrieval=args.no_hard_negatives,
                args=args,
                train_feats=train_feats,
                train_labels=train_labels,
                target_pack=target_pack,
                query_ids=ids,
            )
        # Both hard negative and pseudo gold,
        # In default we will consider hard negative, which is key
        # to the good performance. 
        # But if we want to test without hard negative, this is also fine
        # We can just ignore the hard negative features and scores
        elif args.no_pseudo_gold_positives > 0:
            (
                hard_negative_features,
                hard_negative_scores,
                pseudo_positive_features,
                pseudo_positive_scores,
                train_feats, 
                train_labels,
            ) = dense_retrieve_hard_negatives_pseudo_positive(
                train_dl,
                feats,
                labels,
                model,
                largest_retrieval=args.no_pseudo_gold_positives,
                args=args,
                train_feats=train_feats,
                train_labels=train_labels,
                target_pack=target_pack,
                query_ids=ids,
            )
        else:
            pass
    # For sparse retrieval, 
    # we always retrieve both hard negatives and pseudo gold
    # Since no computation will be saved 
    # by only retrieving hard negatives/pseudo gold
    else:
        (   hard_negative_features,
            pseudo_positive_features,   
        )= sparse_retrieve_hard_negatives_pseudo_positive(
            ids,
            labels,
            train_set,
            model,
            sparse_retrieval_dictionary,
            args,
        )
                
            

    # for hard negative loss
    if args.hard_negatives_loss:
        # Now we have the hard negatives features, we compute the loss

        # hard_negative_scores size batch_size, largest_retrieval

        # We compute the similarity matrix between the hard negatives and the original features
        # The dimension of hard_negative_features is batch_size x no_hard_negatives x dim
        # The dimension of original feats is batch_size x dim
        # We thus need to expand the original feats to batch_size x no_hard_negatives x embed_dim/hidden_dim
        feats_expanded = feats.unsqueeze(1).expand(
            batch_size, args.no_hard_negatives, -1
        )

        # The returned hard_negative_features might contain all zero embeddings for some samples,
        # We need to discard them in the loss computation
        # What we need to do is to construct a mask to zero out the loss for those samples

        # For simplicity, we only check if the first dimension is zero in the feature embedding
        # The mask is batch_size x no_hard_negatives, 1 if embedding non zero, 0 if embedding zero,
        # Thus we can multiply the mask with the loss.
        #zeroLoss_mask = hard_negative_features[:, :, 0] != 0

        # 2024.12.07 update, the above method is not correct, since the first dimension can be zero for some samples
        # Instead, we will sum the sum of the value of the embeddings
        # If the sum is zero, then we will set the mask to zero
        zeroLoss_mask = torch.sum(hard_negative_features, dim=2) != 0

        # Compute Hard negative loss, at the third dimension feature dimension(dim=2)

        if args.metric == "cos":
            # Compute loss
            # Loss is batch_size x no_hard_negatives
            # print(hard_negative_scores)
            # we compute the cosine similarity
            cos_hard = nn.CosineSimilarity(dim=2, eps=1e-8)
            hard_loss = zeroLoss_mask * cos_hard(
                feats_expanded, hard_negative_features)
            # print(hard_loss.shape)
            # print(hard_loss)
        elif args.metric == "ip":
            # Compute loss
            # Loss is batch_size x no_hard_negatives
            hard_loss = zeroLoss_mask * torch.sum(
                feats_expanded * hard_negative_features, dim=2
            ) / args.proj_dim

        elif args.metric == "l2":

            """
            l2_hard = torch.nn.MSELoss(reduction='none')
            hard_loss = l2_hard(feats_expanded,
                                hard_negative_features).sum(dim=2)
            """
            # hard_loss = zeroLoss_mask * torch.sum(torch.square((feats_expanded - hard_negative_features)), dim=2)
            hard_loss = compute_l2(feats_expanded, hard_negative_features,
                                   normalise=args.norm_feats_loss, sum_dim=2, sqrt=args.l2_sqrt)
            hard_loss *= zeroLoss_mask
            """print("feats_expanded:", feats_expanded)
            print("hard negative features:", hard_negative_features)
            print("hard negative features shape:", hard_negative_features.shape)
            print("hard loss:", hard_loss)"""
            # SQRT gives NAN, thus we minimize the square of the L2 distance
            hard_loss = - hard_loss / args.proj_dim

        # For contrastive loss, we take mean during the loss computation
        if args.loss != "contrastive":
            # Hard loss batch_size * no_hard_neg -> batch_size
            hard_loss = torch.sum(hard_loss, dim=1)
            """print("hard loss")
            print(hard_loss.shape)
            print(hard_loss)"""

    # If not using hard negative, set to 0
    else:
        #hard_loss = 0
        hard_loss = torch.tensor([0.0], device=args.device)

    # for pseudo gold loss
    if args.no_pseudo_gold_positives != 0:
        # Now we have the pseudo gold positive features, we compute the loss
        # pseudo_positive_scores size: batch_size, args.no_pseudo_gold_positives

        feats_expanded = feats.unsqueeze(1).expand(
            batch_size, args.no_pseudo_gold_positives, -1
        )
        if args.metric == "cos":
            # Compute loss
            # Loss is batch_size x no_pseudo_gold_positives
            # print(pseudo_positive_scores)
            # we compute the cosine similarity
            cos_pseudo_gold = nn.CosineSimilarity(dim=2, eps=1e-8)
            pseudo_gold_loss = cos_pseudo_gold(
                feats_expanded, pseudo_positive_features)
            # print(pseudo_gold_loss.shape)
            # print(pseudo_gold_loss)
        elif args.metric == "ip":
            # Compute loss
            # Loss is batch_size x no_hard_negatives
            pseudo_gold_loss = torch.sum(
                feats_expanded * pseudo_positive_features, dim=2
            ) / args.proj_dim

        elif args.metric == "l2":

            # pseudo_gold_loss = torch.sum(torch.square((feats_expanded - pseudo_positive_features)), dim=2)
            pseudo_gold_loss = compute_l2(
                feats_expanded, pseudo_positive_features, normalise=args.norm_feats_loss, sum_dim=2, sqrt=args.l2_sqrt)
            """print("feats_expanded:", feats_expanded)
            print("Pseudo Positive Feats:", pseudo_positive_features)
            print("Pseudo Positive Feats Shape:", pseudo_positive_features.shape)
            print("Pseiudo Gold Loss:", pseudo_gold_loss)"""
            # SQRT gives NAN, thus we minimize the square of the L2 distance
            pseudo_gold_loss = - pseudo_gold_loss / args.proj_dim

        # For contrastive loss, we take mean during the loss computation
        if args.loss != "contrastive":

            pseudo_gold_loss = torch.mean(pseudo_gold_loss, dim=1)
            """print("pseudo_gold loss")
            print(pseudo_gold_loss.shape)
            print(pseudo_gold_loss)"""

    # if not using psedo gold, set to 0
    else:
        #pseudo_gold_loss = 0
        # use tensor zero instead
        pseudo_gold_loss = torch.tensor([0.0], device=args.device)

    # ----------------- P5: counterfactual twin as one extra hard negative -----------------
    # For each anchor (TRAIN positive) with a verified sanitized twin, push the anchor away
    # from twin_fused = model(anchor_img, sanitized_text). Same sign/scale as a mined hard
    # negative (cosine added into hard_loss, inside the triplet relu). cf_negs off (or no
    # cf_pack) -> EXACT no-op. Defined for the triplet/naive video recipe only.
    if getattr(args, "cf_negs", False) and cf_pack is not None:
        if args.loss == "contrastive":
            raise NotImplementedError(
                "cf_negs is defined for the triplet/naive video recipe")
        cf_sim = compute_cf_negative_sim(feats, batch, model, cf_pack, args)  # [B]
        hard_loss = hard_loss + cf_sim

    if args.loss == "naive":
        # Take mean on batch-sample level
        total_loss = torch.mean(in_batch_loss + hard_loss - pseudo_gold_loss)
    elif args.loss == "triplet":
        total_loss = torch.mean(torch.relu(
            in_batch_loss + hard_loss - pseudo_gold_loss + args.triplet_margin))

        # Don't use if statement, rather, we can use a relu
        # if total_loss < 0:
        #    total_loss = torch.tensor([0.0], requires_grad=True).to(args.device)
    elif args.loss == "contrastive":

        # Dim Batch size * Batch size
        # Pick out the non-zero terms (gives 1), mask out the zero terms (gives 0)
        neg_mask = in_batch_negative_loss != 0
        # Dim batch_size, count the number of zeros for each sample in the batch,
        # Since exponential of zero gives 1, we will delete the the number of zeros to discard the zero term
        neg_zero_count = (neg_mask == 0).sum(dim=1)
        # However, if all the terms are zero, we will get nan due to zero division,
        # We will form a further mask to only operate on the sample with at least one non-zero term
        # neg_zero_count_zero_mask = neg_zero_count == 0
        # Above is incorrect, we need to get the mask for samples in the batch with all examples zero
        neg_zero_count_zero_mask = torch.zeros(batch_size, device=args.device) != in_batch_negative_no
        in_batch_negative_loss_tmp = torch.zeros(
            batch_size, device=args.device)
        #in_batch_negative_loss_tmp[neg_zero_count_zero_mask] = (torch.exp(in_batch_negative_loss[neg_zero_count_zero_mask]).sum(
        #    dim=1) - neg_zero_count[neg_zero_count_zero_mask]) / (neg_mask.sum(dim=1))[neg_zero_count_zero_mask]
        
        in_batch_negative_loss_tmp[neg_zero_count_zero_mask] = (torch.exp(in_batch_negative_loss[neg_zero_count_zero_mask]).sum(
            dim=1) - neg_zero_count[neg_zero_count_zero_mask])
        in_batch_negative_loss = in_batch_negative_loss_tmp
        """print(in_batch_negative_no)
        print(neg_zero_count)
        print(neg_zero_count_zero_mask)
        print(neg_mask.sum(dim=1))
        print((neg_mask.sum(dim=1))[neg_zero_count_zero_mask])"""

        if args.no_hard_negatives != 0:
            # Dim batch size x no_hard_negatives
            hard_neg_mask = hard_loss != 0

            # Dim batch_size
            hard_zero_count = (hard_neg_mask == 0).sum(dim=1)
            # Constract this matrix to avoid zero division error
            # hard_zero_count_zero_mask = hard_zero_count == 0
            hard_zero_count_zero_mask = hard_zero_count != args.no_hard_negatives
            # initialise all zero matrix for hard loss
            hard_loss_tmp = torch.zeros(batch_size, device=args.device)
            # We need to count the number of zero terms to discard them in the loss computation,
            # Since zero terms gives exp(0) = 1, we will delete the the number of zeros to discard the zero term
            hard_loss_tmp[hard_zero_count_zero_mask] = (torch.exp(hard_loss[hard_zero_count_zero_mask]).sum(
                dim=1) - hard_zero_count[hard_zero_count_zero_mask]) / (hard_neg_mask.sum(dim=1))[hard_zero_count_zero_mask]
            hard_loss = hard_loss_tmp

        """print(hard_zero_count)
        print(hard_zero_count_zero_mask)
        print((hard_neg_mask.sum(dim=1)))
        print((hard_neg_mask.sum(dim=1))[hard_zero_count_zero_mask])"""

        # If we dont have pseudo gold positives, we use the in batch positives
        if args.no_pseudo_gold_positives == 0:

            """loss = - torch.log(torch.mean(torch.exp(in_batch_positives_loss), dim=1) / (torch.mean(torch.exp(in_batch_negative_loss),
                            dim=1) + torch.mean(torch.exp(in_batch_positives_loss), dim=1) + torch.mean(torch.exp(hard_loss), dim=1)))"""
            """pos_mask = in_batch_positives_loss != 0
            pos_zero_count = (pos_mask == 0).sum(dim=1)

            pos_zero_count_zero_mask = pos_zero_count != in_batch_positives_no

            in_batch_positives_loss_tmp = torch.zeros(
                batch_size, device=args.device)
            in_batch_positives_loss_tmp[pos_zero_count_zero_mask] = (torch.exp(in_batch_positives_loss[pos_zero_count_zero_mask]).sum(
                dim=1) - pos_zero_count[pos_zero_count_zero_mask]) / (pos_mask.sum(dim=1))[pos_zero_count_zero_mask]
            in_batch_positives_loss = in_batch_positives_loss_tmp"""
            in_batch_positives_loss = torch.mean(torch.exp(in_batch_positives_loss), dim=1)
            loss = - torch.log(in_batch_positives_loss /
                               (in_batch_negative_loss + in_batch_positives_loss + hard_loss))
        # If we have pseudo gold positives, we use the pseudo gold positives rather than the in batch positives
        else:

            """loss = - torch.log(torch.mean(torch.exp(pseudo_gold_loss), dim=1) / (torch.mean(torch.exp(hard_loss),
                            dim=1) + torch.mean(torch.exp(pseudo_gold_loss), dim=1) + torch.mean(torch.exp(in_batch_negative_loss), dim=1)))"""

            pseudo_gold_loss = torch.mean(torch.exp(pseudo_gold_loss), dim=1)

            loss = - torch.log(pseudo_gold_loss / (hard_loss +
                               pseudo_gold_loss + in_batch_negative_loss))

        """print("Loss:", loss) 
        print("Hard Loss:", hard_loss)
        #print("In Batch Positives Loss:", in_batch_positives_loss)
        print("In Batch Negative Loss:", in_batch_negative_loss)
        print("Pseudo Gold Loss:", pseudo_gold_loss)"""

        total_loss = torch.mean(loss)
    if args.hybrid_loss:
        if args.pos_weight_value != None:
            lossFn_classifier = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([args.pos_weight_value], device=args.device))
        else:
            lossFn_classifier = nn.BCEWithLogitsLoss()
        loss_classifier = lossFn_classifier(
                output, labels.float().reshape(-1, 1))
        #total_loss = (total_loss + loss_classifier * args.ce_weight) / (1 + args.ce_weight)
        total_loss = total_loss * (1-args.ce_weight) + loss_classifier * args.ce_weight
    else:
        loss_classifier = 0

    # ----------------- Segment-RGCL (multi-granularity) additive term -----------------
    # L = L_whole_video + lambda_seg * L_segment.
    # lambda_seg == 0 (or no segment cache) -> EXACT no-op, identical to baseline.
    lambda_seg = float(getattr(args, "lambda_seg", 0.0))
    if lambda_seg > 0 and segment_cache is not None:
        # Pass the batch's whole-video fused embeddings so the segment loss can,
        # in `driftneg` mode, use each sub-clip's OWN parent-video rep as a clean
        # positive (see compute_segment_loss). `feats` is grad-tracked.
        seg_loss = compute_segment_loss(
            batch, model, args, segment_cache, whole_video_feats=feats)
        total_loss = total_loss + lambda_seg * seg_loss

    # ----------------- P4: archive-field auxiliary distillation term -----------------
    # L = L_main + lambda_aux * sum_field aux_field. lambda_aux == 0 (or no aux_pack)
    # -> EXACT no-op, identical to baseline. Reuses the SAME grad-tracked whole-video
    # fused embedding `feats` from the forward pass above (no second forward, so no
    # extra dropout RNG is consumed and the lambda_aux=0 path is byte-identical).
    lambda_aux = float(getattr(args, "lambda_aux", 0.0))
    if lambda_aux > 0 and aux_pack is not None:
        aux_loss = compute_aux_loss(feats, batch["ids"], aux_pack, args)
        total_loss = total_loss + lambda_aux * aux_loss

    # ----------------- TARC V3: intra-target separation regulariser -----------------
    # L = L_main + lambda_tarc * L_tarc (exp-tarc-t0.md §2 V3). lambda_tarc == 0 (or no
    # target_pack) -> EXACT no-op, identical to baseline. Reuses the SAME grad-tracked
    # whole-video fused embedding `feats` from the forward pass above (no second forward,
    # no extra dropout RNG), so the lambda_tarc=0 path is byte-identical (mirrors P4).
    lambda_tarc = float(getattr(args, "lambda_tarc", 0.0))
    if lambda_tarc > 0 and target_pack is not None:
        tarc_loss = compute_target_loss(feats, batch["ids"], target_pack, labels, args)
        total_loss = total_loss + lambda_tarc * tarc_loss

    return total_loss, torch.mean(in_batch_loss), torch.mean(hard_loss), torch.mean(pseudo_gold_loss), loss_classifier, train_feats, train_labels


def compute_aux_loss(feats, batch_ids, aux_pack, args):
    """P4 archive-field distillation loss on the whole-video fused embedding.

    For every field, a small linear head maps `feats` [B, proj_dim] to the field's
    schema targets (CE for the single-label explicitness field, BCE for the multi-label
    modality / mechanism / target_group fields). Samples whose archive is missing /
    unparseable are masked out of the aux loss ONLY (never the main loss). Returns the
    SUM over fields of the per-field mean loss over valid samples.
    """
    module = aux_pack["module"]
    id_to_row = aux_pack["id_to_row"]
    specs = aux_pack["specs"]
    targets = aux_pack["targets"]
    valids = aux_pack["valids"]
    device = args.device
    rows = torch.as_tensor([id_to_row[i] for i in batch_ids],
                           dtype=torch.long, device=device)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    total = torch.zeros((), device=device)
    for field, head in module.items():
        v = valids[field].index_select(0, rows)          # [B] bool
        if int(v.sum().item()) == 0:
            continue
        logits = head(feats)[v]                            # [Bv, dim]
        tgt = targets[field].index_select(0, rows)[v]
        if specs[field]["type"] == "single":
            total = total + ce(logits, tgt.long())
        else:
            total = total + bce(logits, tgt.float())
    return total


def compute_target_loss(feats, batch_ids, target_pack, labels, args):
    """TARC V3 intra-target separation regulariser (exp-tarc-t0.md §2 V3).

    Within each target community T present in the batch with BOTH a hateful and a
    benign example, push that community's hate centroid and benign centroid apart:

        L_tarc = mean_T relu( margin + sim(mu_{T,hate}, mu_{T,benign}) )

    where mu_{T,.} are the batch's per-target per-label mean fused embeddings and sim
    is the run's metric (_pair_similarity, higher == more similar). Driving L_tarc down
    drives that similarity below -margin, i.e. SEPARATES the same-community hate/benign
    centroids ("same community, opposite intent" becomes explicit geometry).

    Sign note (load-bearing). The pre-registration wrote `relu(m - d(mu_hate,mu_benign))`
    naming d = _pair_similarity, but _pair_similarity is a SIMILARITY (higher == closer),
    so relu(m - sim) would PULL the centroids together -- the opposite of the stated
    intent. To realise "push apart by a margin" we hinge on similarity directly:
    relu(margin + sim), which reaches 0 once the two centroids are at similarity
    <= -margin. Recorded as a deliberate deviation from the literal formula in §9.

    Centroids are means of the grad-tracked `feats`; no new params, no RNG. Targets
    with code < 0 (no community) and targets missing either class in the batch are
    skipped. Returns a scalar (0 when no eligible target is present in the batch).
    """
    device = args.device
    id_to_target = target_pack["id_to_target"]
    codes = torch.as_tensor(
        [id_to_target.get(vid, -1) for vid in batch_ids],
        dtype=torch.long, device=device)                       # [B]
    is_hate = labels.to(device).bool()                         # [B]
    margin = float(getattr(args, "triplet_margin", 0.1))
    terms = []
    for t in torch.unique(codes[codes >= 0]).tolist():
        sel = (codes == t)
        hate_sel = sel & is_hate
        benign_sel = sel & (~is_hate)
        if int(hate_sel.sum().item()) == 0 or int(benign_sel.sum().item()) == 0:
            continue
        mu_hate = feats[hate_sel].mean(dim=0, keepdim=True)    # [1, D] grad-tracked
        mu_benign = feats[benign_sel].mean(dim=0, keepdim=True)
        sim = _pair_similarity(mu_hate, mu_benign, args)       # [1]
        terms.append(torch.relu(margin + sim).reshape(()))
    if not terms:
        return torch.zeros((), device=device)
    return torch.stack(terms).mean()


def compute_cf_negative_sim(feats, batch, model, cf_pack, args):
    """P5: per-anchor counterfactual-twin negative similarity, returned as [B] (0 where the
    anchor has no verified twin). twin_fused = model(anchor real img_feats, sanitized twin
    text_feats); the returned cosine is added into hard_loss as one extra hard negative.

    The twin forward is wrapped in a CPU+CUDA RNG save/restore so it does NOT perturb the
    main training RNG stream (dropout / shuffle): the cf_negs run then differs from the floor
    ONLY by this added negative's gradient (the main forward + hard-neg mining draws are
    byte-identical to the floor).
    """
    device = args.device
    ids = batch["ids"]
    B = len(ids)
    out = torch.zeros(B, device=device)
    id_to_row = cf_pack["id_to_row"]
    valid = cf_pack["valid"]
    rows, bidx = [], []
    for i, vid in enumerate(ids):
        r = id_to_row.get(vid)
        if r is not None and bool(valid[r]):
            rows.append(r)
            bidx.append(i)
    if not rows:
        return out
    rows_t = torch.as_tensor(rows, dtype=torch.long, device=device)
    bidx_t = torch.as_tensor(bidx, dtype=torch.long, device=device)
    anchor_img = batch["image_feats"].to(device).index_select(0, bidx_t)   # [n, Dv] real
    twin_text = cf_pack["twin_text"].index_select(0, rows_t)               # [n, Dt] sanitized

    is_cuda = str(device).startswith("cuda")
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state() if is_cuda else None
    _, twin_fused = model(anchor_img, twin_text, return_embed=True)        # [n, D] grad-tracked
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state)
    torch.set_rng_state(cpu_state)

    anchor_fused = feats.index_select(0, bidx_t)                           # [n, D] grad-tracked
    sim = _pair_similarity(anchor_fused, twin_fused, args)                 # [n]
    out = out.index_copy(0, bidx_t, sim)
    return out


def _pair_similarity(anchor, other, args):
    """Row-wise similarity between anchor [B, D] and other [B, D] under args.metric.
    Returns [B]; larger == more similar (l2 is negated). Mirrors the whole-video path."""
    if args.metric == "cos":
        return nn.functional.cosine_similarity(anchor, other, dim=1, eps=1e-8)
    elif args.metric == "ip":
        return torch.sum(anchor * other, dim=1) / args.proj_dim
    else:  # l2
        l2 = compute_l2(anchor, other, normalise=args.norm_feats_loss, sum_dim=1, sqrt=args.l2_sqrt)
        return - l2 / args.proj_dim


def compute_segment_loss(batch, model, args, segment_cache, whole_video_feats=None):
    """Sub-clip-granularity retrieval-guided contrastive loss (Delta 1).

    For every sub-clip whose PARENT video is in this batch, we:
      1. fuse (sub-clip visual, PARENT video text) through the SAME
         classifier_hateClipper -> fused sub-clip embedding (no new params);
      2. build/refresh the 2nd FAISS index over the batch's sub-clip corpus and
         mine, per anchor sub-clip: pseudo-gold-positive (nearest same-label
         sub-clip), opposite-label hard negative, and the within-video MIL
         drifting hard negative (annotation-free);
      3. apply the SAME contrastive/triplet objective form as the whole-video term.

    seg_mode (args.seg_mode, default "full") selects the variant:
      * "full"     -- original behaviour: pull each anchor sub-clip toward the
        label-inherited pseudo-gold positive (nearest same-label sub-clip), push
        it from the within-video drifting hard-neg and the opposite-label hard-neg.
      * "driftneg" -- DROP the noisy label-inherited pseudo-gold positive.
        Diagnosis: sub-clips inherit the whole-video label, but most sub-clips of
        a hateful video are actually benign, so "nearest same-label sub-clip"
        positives corrupt the space. Instead we use each sub-clip's OWN parent
        whole-video fused embedding as a CLEAN positive (a real, uncontroversial
        target) and push the sub-clip away from BOTH the within-video drifting
        hard-neg and the opposite-label hard-neg. The within-video drifting
        hard-neg (well-motivated MIL signal) is retained; only the poison
        (inherited positive) is removed. Requires `whole_video_feats`.
      * "milmax"   -- represent each parent video by its MOST-hateful sub-clip
        (max predicted hate logit among its sub-clips) and contrast at that single
        representative only (pull representative toward parent-video positive if
        available, else its label-inherited pseudo-gold positive; push from
        opposite-label hard-neg).

    segment_cache is a dict:
      {
        "subclip_img_feats": [TotalSub, Dv]   (all splits' sub-clips for train corpus),
        "subclip_parent"   : [TotalSub]       (row index into the whole-video train cache),
        "labels"           : [TotalSub]       (inherited),
        "parent_id_to_row" : {video_id -> whole-video train row index},
        "video_text_feats" : [V_train, Dt]    (whole-video train text feats, indexed by parent row),
      }
    Only sub-clips whose parent is present in the current batch are used as anchors;
    the corpus is the batch's own sub-clips (per-step 2nd index), matching the
    whole-video path's per-epoch index over currently-seen embeddings.

    whole_video_feats [B, proj_dim] : the batch's grad-tracked whole-video fused
      embeddings (row-aligned with batch["ids"]). Used by "driftneg"/"milmax" as
      the clean per-video positive.
    """
    device = args.device
    seg_mode = str(getattr(args, "seg_mode", "full"))
    batch_ids = batch["ids"]
    parent_id_to_row = segment_cache["parent_id_to_row"]
    subclip_img = segment_cache["subclip_img_feats"]
    subclip_parent = segment_cache["subclip_parent"]
    subclip_label = segment_cache["labels"]
    video_text = segment_cache["video_text_feats"]

    import numpy as _np
    # numpy views for masking
    sc_parent_np = (
        subclip_parent.detach().cpu().numpy()
        if torch.is_tensor(subclip_parent) else _np.asarray(subclip_parent)
    )

    # Rows of the sub-clip cache whose parent video is in this batch.
    batch_parent_rows = set()
    for vid in batch_ids:
        row = parent_id_to_row.get(vid, None)
        if row is not None:
            batch_parent_rows.add(int(row))
    if len(batch_parent_rows) == 0:
        return torch.tensor(0.0, device=device)

    keep = _np.isin(sc_parent_np, list(batch_parent_rows))
    keep_idx = _np.nonzero(keep)[0]
    if keep_idx.shape[0] == 0:
        return torch.tensor(0.0, device=device)

    keep_t = torch.as_tensor(keep_idx, dtype=torch.long)
    sc_img = subclip_img.index_select(0, keep_t).to(device).float()   # [Ns, Dv]
    sc_parent = subclip_parent.index_select(0, keep_t).to(device)      # [Ns]
    sc_label = subclip_label.index_select(0, keep_t).to(device)        # [Ns]

    # PARENT video text per sub-clip (video text is SHARED across sub-clips).
    parent_text = video_text.index_select(0, sc_parent.cpu()).to(device).float()  # [Ns, Dt]

    # Map each sub-clip's parent whole-video row -> that video's position in the
    # current batch, so driftneg/milmax can use the grad-tracked whole-video fused
    # embedding as a CLEAN per-video positive. Rows aligned with batch["ids"].
    own_video_target = None      # [Ns, proj_dim] parent whole-video fused embed
    own_video_valid = None       # [Ns] bool: parent present in this batch
    if whole_video_feats is not None:
        row_to_batchidx = {}
        for bidx, vid in enumerate(batch_ids):
            prow = parent_id_to_row.get(vid, None)
            if prow is not None:
                row_to_batchidx[int(prow)] = bidx
        sc_parent_kept_np = subclip_parent.index_select(
            0, keep_t).detach().cpu().numpy()
        map_idx = _np.array(
            [row_to_batchidx.get(int(p), -1) for p in sc_parent_kept_np],
            dtype=_np.int64)
        own_video_valid = torch.as_tensor(map_idx >= 0, device=device)
        # Clamp -1 -> 0 for a safe gather; invalidated via own_video_valid.
        gather_idx = torch.as_tensor(
            _np.clip(map_idx, 0, None), dtype=torch.long, device=whole_video_feats.device)
        own_video_target = whole_video_feats.index_select(
            0, gather_idx).to(device)  # [Ns, proj_dim], grad-tracked

    # Fuse through the SAME head -> fused sub-clip embeddings (retrieval space).
    # No new trainable params: exactly model(img, text, return_embed=True).
    model.train()
    _, sc_feats = model(sc_img, parent_text, return_embed=True)  # [Ns, proj_dim], grad-tracked

    # Non-zero (decodable) sub-clip mask: a zero-vector guard yields all-zero visual.
    valid_mask = (torch.sum(sc_img, dim=1) != 0)

    # ---- seg_mode in {consensus, selfscore}: pseudo-labelled segment loss ----
    # Sub-clip pseudo-labels (ROLE_*) are assigned OUTSIDE the step (EM E-step in
    # run_rac.py, per-round index rebuild) and stored in segment_cache. Only
    # confident sub-clips (margin >= tau) enter the contrastive term; the
    # noisy-MIL-positive cell (Y_v=hate, vote=benign) is NEVER a positive and
    # instead supplies the within-video drifting hard negative.
    if seg_mode in ("consensus", "selfscore"):
        return _compute_pseudo_role_segment_loss(
            sc_feats, sc_parent, segment_cache, keep_t, valid_mask, args)

    # Build the 2nd FAISS index over a DETACHED copy of the fused sub-clip corpus
    # (targets are fixed anchors, as in the whole-video path). Mine per anchor.
    corpus_feats = sc_feats.detach()
    (
        seg_hard,
        seg_pos,
        seg_drift,
        seg_drift_mask,
    ) = dense_retrieve_segment_hard_negatives_pseudo_positive(
        query_feats=corpus_feats,
        query_labels=sc_label,
        query_parents=sc_parent,
        corpus_feats=corpus_feats,
        corpus_labels=sc_label,
        corpus_parents=sc_parent,
        args=args,
    )

    Ns = sc_feats.shape[0]

    # ---- pseudo-gold-positive similarity (want HIGH) ----
    # use first mined positive. seg_pos always has >=1 column (see mining fn),
    # but the video config sets no_pseudo_gold_positives==1.
    if seg_pos.shape[1] >= 1:
        pos_target = seg_pos[:, 0, :]                   # [Ns, D]
        pos_valid = (torch.sum(pos_target, dim=1) != 0)
        pos_sim = _pair_similarity(sc_feats, pos_target, args)  # [Ns]
    else:
        pos_target = torch.zeros(Ns, sc_feats.shape[1], device=device)
        pos_valid = torch.zeros(Ns, dtype=torch.bool, device=device)
        pos_sim = torch.zeros(Ns, device=device)

    # ---- opposite-label hard-negative similarity (want LOW) ----
    if seg_hard.shape[1] >= 1:
        hn_target = seg_hard[:, 0, :]                   # [Ns, D]
        hn_valid = (torch.sum(hn_target, dim=1) != 0)
        hn_sim = _pair_similarity(sc_feats, hn_target, args)
    else:
        hn_target = torch.zeros(Ns, sc_feats.shape[1], device=device)
        hn_valid = torch.zeros(Ns, dtype=torch.bool, device=device)
        hn_sim = torch.zeros(Ns, device=device)

    # ---- within-video MIL drifting hard-negative similarity (want LOW) ----
    drift_target = seg_drift[:, 0, :]                   # [Ns, D]
    drift_valid = seg_drift_mask & (torch.sum(drift_target, dim=1) != 0)
    drift_sim = _pair_similarity(sc_feats, drift_target, args)

    # ---- own parent whole-video fused embedding as CLEAN positive (want HIGH) ----
    # (driftneg/milmax). Detach the target so the sub-clip is pulled toward the
    # video-level rep without dragging the (separately-optimised) whole-video
    # embedding toward its own sub-clips; the whole-video term already anchors it.
    if own_video_target is not None:
        own_sim = _pair_similarity(sc_feats, own_video_target.detach(), args)  # [Ns]
        own_valid = own_video_valid & (torch.sum(own_video_target, dim=1) != 0)
    else:
        own_sim = torch.zeros(Ns, device=device)
        own_valid = torch.zeros(Ns, dtype=torch.bool, device=device)

    margin = float(getattr(args, "triplet_margin", 0.1))
    per_anchor = torch.zeros(Ns, device=device)
    term_count = torch.zeros(Ns, device=device)

    if seg_mode == "driftneg":
        # DROP the noisy label-inherited pseudo-gold positive. Positive = anchor's
        # OWN parent whole-video fused embedding (clean, real target). Push from
        # BOTH the within-video drifting hard-neg and the opposite-label hard-neg.
        if own_video_target is None:
            # No whole-video positive available -> degrade to a pure push-only
            # margin repulsion from the two negatives (still well-defined).
            m1 = (valid_mask & hn_valid)
            if m1.any():
                t1 = torch.relu(hn_sim + margin)
                per_anchor = per_anchor + torch.where(m1, t1, torch.zeros_like(t1))
                term_count = term_count + m1.float()
            m2 = (valid_mask & drift_valid)
            if m2.any():
                t2 = torch.relu(drift_sim + margin)
                per_anchor = per_anchor + torch.where(m2, t2, torch.zeros_like(t2))
                term_count = term_count + m2.float()
        else:
            # own-positive vs opposite-label hard-negative triplet
            m1 = (valid_mask & own_valid & hn_valid)
            if m1.any():
                t1 = torch.relu(hn_sim - own_sim + margin)
                per_anchor = per_anchor + torch.where(m1, t1, torch.zeros_like(t1))
                term_count = term_count + m1.float()
            # own-positive vs within-video drifting hard-negative triplet (MIL delta)
            m2 = (valid_mask & own_valid & drift_valid)
            if m2.any():
                t2 = torch.relu(drift_sim - own_sim + margin)
                per_anchor = per_anchor + torch.where(m2, t2, torch.zeros_like(t2))
                term_count = term_count + m2.float()

    elif seg_mode == "milmax":
        # Represent each parent video by its MOST-hateful sub-clip (max predicted
        # hate logit) and contrast at that single representative only.
        # output_sc is the classifier logit per sub-clip.
        output_sc, _ = model(sc_img, parent_text, return_embed=True)  # logits [Ns, 1]
        hate_logit = output_sc.reshape(-1)  # [Ns]
        # Positive per representative: own parent whole-video fused embed if
        # available, else the label-inherited pseudo-gold positive.
        rep_pos_sim = own_sim if own_video_target is not None else pos_sim
        rep_pos_valid = own_valid if own_video_target is not None else pos_valid
        # Select, per parent, the sub-clip with the max hate logit as representative.
        sc_parent_np = sc_parent.detach().cpu().numpy()
        rep_mask = torch.zeros(Ns, dtype=torch.bool, device=device)
        for p in set(int(x) for x in sc_parent_np):
            rows = _np.nonzero(sc_parent_np == p)[0]
            if rows.shape[0] == 0:
                continue
            rows_t = torch.as_tensor(rows, dtype=torch.long, device=device)
            local_max = int(torch.argmax(hate_logit.index_select(0, rows_t)).item())
            rep_mask[int(rows[local_max])] = True
        # rep-positive vs opposite-label hard-negative triplet, only at representatives
        m1 = (rep_mask & valid_mask & rep_pos_valid & hn_valid)
        if m1.any():
            t1 = torch.relu(hn_sim - rep_pos_sim + margin)
            per_anchor = per_anchor + torch.where(m1, t1, torch.zeros_like(t1))
            term_count = term_count + m1.float()

    else:
        # seg_mode == "full": ORIGINAL behaviour. Pull to label-inherited
        # pseudo-gold-positive, push from both hard negatives (corpus
        # opposite-label + within-video drifting).
        m1 = (valid_mask & pos_valid & hn_valid)
        if m1.any():
            t1 = torch.relu(hn_sim - pos_sim + margin)
            per_anchor = per_anchor + torch.where(m1, t1, torch.zeros_like(t1))
            term_count = term_count + m1.float()
        m2 = (valid_mask & pos_valid & drift_valid)
        if m2.any():
            t2 = torch.relu(drift_sim - pos_sim + margin)
            per_anchor = per_anchor + torch.where(m2, t2, torch.zeros_like(t2))
            term_count = term_count + m2.float()

    denom = term_count.sum()
    if denom.item() == 0:
        return torch.tensor(0.0, device=device)
    seg_loss = per_anchor.sum() / denom
    return seg_loss


def _compute_pseudo_role_segment_loss(sc_feats, sc_parent, segment_cache, keep_t,
                                      valid_mask, args):
    """Segment loss for seg_mode in {consensus, selfscore} (DESIGN_iter3 SS2).

    Uses the per-sub-clip pseudo-label ROLES precomputed by the EM E-step
    (utils/consensus.py) and stored in segment_cache:
      segment_cache["pseudo_role"]   LongTensor [TotalSub] in {-1,0,1,2,3}
      segment_cache["pseudo_margin"] FloatTensor [TotalSub]

    Loss structure (mirrors seg_mode=full, ONLY the label source changes):
      * anchors  = confident sub-clips (ROLE_POS=1 / ROLE_NEG=0) of batch parents;
      * pseudo-gold positive = nearest same-pseudo-label sub-clip from another
        video (mined on the batch sub-clip corpus, as in full);
      * hard negative = nearest opposite-pseudo-label sub-clip;
      * drifting hard negative (consensus replacement of the MIL centroid
        heuristic): for a ROLE_POS anchor, the same-video ROLE_DRIFT sub-clip
        (Y_v=hate but vote=benign, i.e. the semantically drifting benign
        segment). Enabled by --consensus_use_drift.
      * ROLE_CONFLICT (Y_v=benign, vote=hate): ignored by default; with
        --consensus_conflict hardneg those sub-clips join the mining corpus
        with pseudo-label 0 (extra confusable hard negatives).
    """
    device = args.device
    roles_all = segment_cache.get("pseudo_role", None)
    margins_all = segment_cache.get("pseudo_margin", None)
    if roles_all is None or margins_all is None:
        # E-step has not populated pseudo-labels (should not happen in the EM
        # driver); contribute nothing rather than train on inherited noise.
        return torch.tensor(0.0, device=device)

    sc_role = roles_all.index_select(0, keep_t).to(device)          # [Ns]
    sc_pmargin = margins_all.index_select(0, keep_t).to(device)     # [Ns]

    conf_mask = ((sc_role == 0) | (sc_role == 1)) & valid_mask
    if int(conf_mask.sum().item()) == 0:
        return torch.tensor(0.0, device=device)

    conflict_mode = str(getattr(args, "consensus_conflict", "ignore"))
    corpus_mask = conf_mask.clone()
    if conflict_mode == "hardneg":
        corpus_mask = corpus_mask | ((sc_role == 3) & valid_mask)

    q_idx = torch.nonzero(conf_mask).reshape(-1)                    # anchors
    c_idx = torch.nonzero(corpus_mask).reshape(-1)                  # mining corpus
    # pseudo binary label: ROLE_POS -> 1, ROLE_NEG / ROLE_CONFLICT -> 0
    pseudo_label = (sc_role == 1).long()

    corpus_feats = sc_feats.detach()
    (
        seg_hard,
        seg_pos,
        _unused_drift,
        _unused_drift_mask,
    ) = dense_retrieve_segment_hard_negatives_pseudo_positive(
        query_feats=corpus_feats.index_select(0, q_idx),
        query_labels=pseudo_label.index_select(0, q_idx),
        query_parents=sc_parent.index_select(0, q_idx),
        corpus_feats=corpus_feats.index_select(0, c_idx),
        corpus_labels=pseudo_label.index_select(0, c_idx),
        corpus_parents=sc_parent.index_select(0, c_idx),
        args=args,
    )

    anchor_feats = sc_feats.index_select(0, q_idx)                  # grad-tracked
    Nq = anchor_feats.shape[0]

    # pseudo-gold positive (want HIGH)
    if seg_pos.shape[1] >= 1:
        pos_target = seg_pos[:, 0, :]
        pos_valid = (torch.sum(pos_target, dim=1) != 0)
        pos_sim = _pair_similarity(anchor_feats, pos_target, args)
    else:
        pos_valid = torch.zeros(Nq, dtype=torch.bool, device=device)
        pos_sim = torch.zeros(Nq, device=device)

    # opposite-pseudo-label hard negative (want LOW)
    if seg_hard.shape[1] >= 1:
        hn_target = seg_hard[:, 0, :]
        hn_valid = (torch.sum(hn_target, dim=1) != 0)
        hn_sim = _pair_similarity(anchor_feats, hn_target, args)
    else:
        hn_valid = torch.zeros(Nq, dtype=torch.bool, device=device)
        hn_sim = torch.zeros(Nq, device=device)

    # consensus drifting hard negative: same-video ROLE_DRIFT sub-clip
    use_drift = bool(getattr(args, "consensus_use_drift", True))
    drift_target = torch.zeros(Nq, sc_feats.shape[1], device=device)
    drift_ok = torch.zeros(Nq, dtype=torch.bool, device=device)
    if use_drift:
        drift_rows = torch.nonzero((sc_role == 2) & valid_mask).reshape(-1)
        if drift_rows.numel() > 0:
            # per parent, keep the highest-margin (most confidently benign) drift
            best = {}
            for r in drift_rows.tolist():
                p = int(sc_parent[r].item())
                m = float(sc_pmargin[r].item())
                if p not in best or m > best[p][1]:
                    best[p] = (r, m)
            anchor_roles = sc_role.index_select(0, q_idx)
            for j in range(Nq):
                if int(anchor_roles[j].item()) != 1:
                    continue  # drift repulsion only for hateful (POS) anchors
                p = int(sc_parent[q_idx[j]].item())
                if p in best:
                    drift_target[j] = corpus_feats[best[p][0]]
                    drift_ok[j] = True
    drift_sim = _pair_similarity(anchor_feats, drift_target, args)

    margin = float(getattr(args, "triplet_margin", 0.1))
    per_anchor = torch.zeros(Nq, device=device)
    term_count = torch.zeros(Nq, device=device)

    # pseudo-positive vs opposite-pseudo-label hard-negative triplet
    m1 = (pos_valid & hn_valid)
    if m1.any():
        t1 = torch.relu(hn_sim - pos_sim + margin)
        per_anchor = per_anchor + torch.where(m1, t1, torch.zeros_like(t1))
        term_count = term_count + m1.float()
    # pseudo-positive vs within-video consensus-drift triplet (POS anchors only)
    m2 = (pos_valid & drift_ok)
    if m2.any():
        t2 = torch.relu(drift_sim - pos_sim + margin)
        per_anchor = per_anchor + torch.where(m2, t2, torch.zeros_like(t2))
        term_count = term_count + m2.float()

    denom = term_count.sum()
    if denom.item() == 0:
        return torch.tensor(0.0, device=device)
    return per_anchor.sum() / denom


def compute_l2(feats_1, feats_2, normalise=False, sum_dim=1, sqrt=False, eps=1e-5):
    """Compute L2 loss."""
    l2_loss = 0
    if normalise:
        feats_1 = torch.nn.functional.normalize(feats_1, dim=sum_dim)
        feats_2 = torch.nn.functional.normalize(feats_2, dim=sum_dim)
    if not sqrt:
        l2_loss = torch.sum(torch.square((feats_1 - feats_2)), dim=sum_dim)
    else:
        l2_loss = torch.sqrt(torch.sum(torch.square(
            (feats_1 - feats_2)), dim=sum_dim) + torch.finfo(torch.float32).tiny)

    return l2_loss


def compute_ip(feats_1, feats_2, normalise=False, sum_dim=1):
    return None
