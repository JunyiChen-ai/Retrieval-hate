import torch.nn as nn
import torch
from model.evaluate_rac import retrieve_evaluate_RAC_, final_evaluation
from model.classifier import classifier_hateClipper
from model.loss import compute_loss
import argparse
import wandb
from data_loader.rac_dataloader import CLIP2Dataloader
from data_loader.dataset import load_feats_from_CLIP

from utils.metrics import eval_and_save_epoch_end, compute_metrics_retrieval
from tqdm import tqdm
import numpy as np
import os

import json


def parse_args():

    arg_parser = argparse.ArgumentParser()

    # <----------------- Data Configs ----------------->
    arg_parser.add_argument(
        "--path", type=str, default="./data/")
    arg_parser.add_argument(
        "--output_path", type=str, default="./logging/"
    )
    arg_parser.add_argument("--model", type=str, default="")

    arg_parser.add_argument("--dataset", type=str, default="FB")

    # The threshold for the similarity score for RAC
    arg_parser.add_argument("--similarity_threshold", type=float, default=-1.)
    arg_parser.add_argument("--fusion_mode", type=str, default="concat")

    arg_parser.add_argument(
        "--topk", type=int, default=5, help="Retrieve at most k pairs for validation"
    )

    arg_parser.add_argument("--majority_voting", type=str, default="mean",
                            help="Choose the majority voting method, options are mean, arithmetic, geometric, learned")

    # ----------------- Loss Configs -----------------

    # The loss function for the model is a combination of two parts:
    # Metric class and loss class, both need to be specified

    arg_parser.add_argument("--metric", type=str, default="cos",
                            help="Choose the metric for similarity score, options are cos, ip, l2")
    """
    cos: cosine similarity
    ip: inner product
    l2: l2 distance
    if we use a certain type of metric, we will also use the same criterion for dense retrieval
    """

    arg_parser.add_argument("--loss", type=str, default="naive",
                            help="Choose to use which loss function, options are naive, triplet, contrastive")

    arg_parser.add_argument("--triplet_margin", type=float, default=0.1,
                            help="The margin for triplet loss, epsilon")

    arg_parser.add_argument("--norm_feats_loss", type=lambda x: (str(x).lower() == "true"), default=False,
                            help="Whether to normalize the feature fpr computing loss ")

    # Do sqrt for L2
    arg_parser.add_argument("--l2_sqrt", type=lambda x: (str(x).lower() == "true"), default=False,
                            help="Whether to do square root for L2 loss ")

    arg_parser.add_argument("--hybrid_loss", type=lambda x: (str(x).lower() == "true"), default=False,
                            help="Whether to use logistic loss for the model")
    arg_parser.add_argument("--ce_weight", type=float, default=0.5,
                            help="The weight for the cross entropy loss")
    arg_parser.add_argument("--pos_weight_value", type=float, default=None,
                            help="The weight for the positive samples in the cross entropy loss")

    # <----------------- Model Configs ----------------->
    arg_parser.add_argument('--num_layers', type=int, default=3)

    # MLP dimension for general
    arg_parser.add_argument('--proj_dim', type=int, default=1024)

    # For hateclipper
    # the pre-modality fusion feature projection dimension
    arg_parser.add_argument('--map_dim', type=int, default=1024)

    arg_parser.add_argument('--dropout', type=float, nargs=3,
                            default=[0.1, 0.4, 0.2],
                            help="Set drop probabilities for map, fusion, pre_output")

    arg_parser.add_argument("--batch_norm", type=lambda x: (str(x).lower() == "true"),
                            default=False, help="Whether to use batch norm for Mapping Network")
    arg_parser.add_argument("--last_layer", type=str, default="none",
                            help="Choose the last layer for the model, options are none, sigmoid, tanh")
    # ----------------- Training Configs -----------------
    arg_parser.add_argument("--epochs", type=int, default=5)
    # batch size also sets the number of in_batch positive and in_batch negative
    # we can set limit to the size of in_batch samples
    arg_parser.add_argument("--batch_size", type=int, default=128)
    arg_parser.add_argument("--lr", type=float, default=0.0001)
    arg_parser.add_argument("--weight_decay", type=float, default=0.0001)
    arg_parser.add_argument(
        "--lr_scheduler",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Using LR scheduler or not",
    )
    arg_parser.add_argument("--num_workers", type=int, default=24)
    # default set to zero to match the number of in_batch samples

    arg_parser.add_argument("--grad_clip", type=float,
                            default=0.1, help="Gradient clipping")

    # <----------------- Psuedo Gold Positive Configs ----------------->

    arg_parser.add_argument("--no_pseudo_gold_positives", type=int, default=1)

    # <----------------- Hard Negative Configs ----------------->
    # we need to experiment with different settings here:
    # set a limit ot the number of hard negatives to be retrieved
    # set a threshold for the hard negatives,
    # use single threshold or both of the above threhsolding

    arg_parser.add_argument(
        "--in_batch_loss",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Using in batch loss for model training",
    ) 
    
    
    arg_parser.add_argument(
        "--hard_negatives_loss",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Using hard negative loss for model training",
    )

    arg_parser.add_argument("--no_hard_negatives", type=int, default=1)
    arg_parser.add_argument("--no_hard_positives", type=int, default=0)

    arg_parser.add_argument(
        "--hard_negatives_multiple",
        type=int,
        default=12,
        help="The value times the no_hard_negatives is the\
                            number of most similar retrieved pairs hard negatives to be retrieved for each sample",
    )
    arg_parser.add_argument(
        "--Faiss_GPU", type=lambda x: (str(x).lower() == "true"), default=False,
        help="Whether to use GPU for Faiss")

    arg_parser.add_argument(
        "--reindex_every_step",
        type=lambda x: (str(x).lower() == "true"), default=False,
        help="Whether to reindex the faiss index every step for dense retrieval")

    # For sparse hard negative
    # If the sparse dictionary file is not None, we will use sparse retrieval,
    # otherwise, dense retrieval is used as default when the dictioary file is None
    arg_parser.add_argument(
        "--sparse_dictionary",
        type=str,
        default=None,
        help="The name of the file of the sparse retrieval dictionary",
    )
    arg_parser.add_argument(
        "--use_attribute",
        default=True,
        type=lambda x: (str(x).lower() == "true"),
        help="Whether to use attribute for object detection in sparse data",
    )
    arg_parser.add_argument(
        "--sparse_topk",
        type=int,
        default=None,
        help="The number of topk retrieved samples for sparse retrieval",
    )
    
    arg_parser.add_argument(
        "--eval_retrieval",
        default=True,
        type=lambda x: (str(x).lower() == "true"),
        help="Using retrieval evaluation",
    )

    # <----------------- Logging Configs ----------------->
    arg_parser.add_argument("--log_interval", type=int, default=10)
    arg_parser.add_argument(
        "--final_eval",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Doing the final eval or not",
    )

    arg_parser.add_argument("--exp_comment", type=str, default="",
                            help="Optional comment for the experiment")

    arg_parser.add_argument("--group_name", type=str, default="RAC_TEST",
                            help=" Name for the wandb group")

    arg_parser.add_argument("--seed", type=int, default=0)

    arg_parser.add_argument("--device", type=str, default="cuda")
    arg_parser.add_argument("--visualise_embed", type=bool, default=False)
    arg_parser.add_argument("--force", type=lambda x: (str(x).lower() == "true"),
                            default=False, help="Whether to force the run or not")
    arg_parser.add_argument(
        "--warmup", type=int, default=5,
        help="Minimum epoch index (0-based) eligible for best-epoch selection. "
             "Epochs 0..(warmup-1) are ignored when choosing the best checkpoint. "
             "If no epoch >= warmup exists (run too short), falls back to all epochs."
    )
    arg_parser.add_argument(
        "--save_embed",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Save the embedding or not",
    )

    # <----------------- Multi-granularity (segment-RGCL) Configs ----------------->
    arg_parser.add_argument(
        "--lambda_seg", type=float, default=0.0,
        help="Weight of the sub-clip (segment) RGCL loss term. "
             "0 == baseline (whole-video only), exact no-op. Paper default 0.5.")
    arg_parser.add_argument(
        "--num_subclips", type=int, default=4,
        help="Number of sub-clips (K) per video for the segment cache lookup.")
    arg_parser.add_argument(
        "--subclip_cache", type=str, default="auto",
        help="Path to the TRAIN sub-clip cache .pt, or 'auto' to derive it as "
             "<path>/CLIP_Embedding/<dataset>/train_subclipK<K>_<model>.pt. "
             "Only loaded when --lambda_seg > 0.")
    arg_parser.add_argument(
        "--seg_mode", type=str, default="full",
        choices=["full", "driftneg", "milmax"],
        help="Variant of the segment-RGCL loss (only active when --lambda_seg > 0):\n"
             "  full     = pseudo-gold positive (nearest same-label sub-clip) + "
             "within-video drifting hard-neg + opposite-label hard-neg (original).\n"
             "  driftneg = DROP the label-inherited pseudo-gold positive; use the "
             "anchor's OWN whole-video fused embedding as the (clean) positive and "
             "push it away from BOTH the within-video drifting hard-neg and the "
             "opposite-label hard-neg. Motivated by: inherited positives are noisy "
             "(most sub-clips of a hateful video are benign) while the drifting "
             "hard-neg is well-founded.\n"
             "  milmax   = represent each video by its most-hateful sub-clip "
             "(max hate logit) and contrast at that representative only.")

    args = arg_parser.parse_args()

    return args


def model_pass(
    train_dl,
    evaluate_dl,
    test_seen_dl,
    model,
    epochs=0,
    log_interval=10,
    args=None,
    artifacts=None,
    train_set=None,
    sparse_dict=None,
    segment_cache=None,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    num_training_steps = args.epochs * len(train_dl)
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,  # Maximum number of iterations.
            eta_min=1e-5,
        )  # Minimum learning rate.
    global_step = -1

    # Best model criterion
    best_acc = 0.0
    best_roc = 0.0
    best_epoch_path = None
    # Per-epoch record for warmup-floor fallback: list of (epoch, select_acc, select_roc, ckpt_path)
    all_epoch_records = []
    warmup = args.warmup  # only epochs >= warmup are eligible for best-epoch selection
    for epoch in tqdm(range(epochs)):
        # train_feats, train_labels is used for dense retrieval for
        # hard negatives and pseudo gold positives
        # When we pass in none, we will force the system
        # to reindex the dense vector embeddings
        # After every epoch, we reindex the dense vector embeddings
        train_feats = None
        train_labels = None
        for step, batch in enumerate(train_dl):
            # Reindex the dense vector embeddings,
            # If we force reindex every step or if it is the first 3 epochs
            if args.reindex_every_step:
                #print("Reindex every step")
                train_feats = None
                train_labels = None
            
            global_step += 1
            (
                total_loss,
                in_batch_loss,
                hard_loss,
                pseudo_gold_loss,
                cross_entropy,
                train_feats,
                train_labels,
            ) = compute_loss(
                batch,
                train_dl,
                model,
                args,
                train_set=train_set,
                sparse_retrieval_dictionary=sparse_dict,
                train_feats=train_feats,
                train_labels=train_labels,
                segment_cache=segment_cache,
            )
            """if args.sparse_dictionary is None and (args.no_hard_negatives != 0 and args.no_pseudo_gold_positives != 0):
                # Only for dense retrieval
                train_feats = train_feats.detach()
                train_labels = train_labels.detach()"""
            if not((args.no_hard_negatives == 0 and args.no_pseudo_gold_positives == 0) or args.sparse_dictionary is not None):
                # The CPU-FAISS retrieval path (Faiss_GPU=False) returns these as
                # numpy arrays, which have no .detach(); only detach real tensors.
                if torch.is_tensor(train_feats):
                    train_feats = train_feats.detach()
                if torch.is_tensor(train_labels):
                    train_labels = train_labels.detach()
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip)
            optimizer.step()
            if args.lr_scheduler:
                scheduler.step()
            if step % log_interval == 0:

                # acc, roc, pre, recall, f1 = compute_metrics_retrieval_baseline(logging_dict, evaluate_labels)
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        step,
                        len(train_dl),
                        100.0 * step / len(train_dl),
                        total_loss.item(),
                    )
                )
                if args.loss != "contrastive":
                    hard_loss_val = hard_loss.item() if args.hard_negatives_loss else 0
                    in_batch_loss_val = in_batch_loss.item() if type(
                        in_batch_loss) != int else in_batch_loss
                    pseudo_gold_loss_val = pseudo_gold_loss.item(
                    ) if args.no_pseudo_gold_positives != 0 else 0
                else:
                    # For contrastive loss, we do not have hard negative loss, we only have total loss
                    hard_loss_val = torch.mean(hard_loss).item(
                    ) if args.hard_negatives_loss else 0
                    in_batch_loss_val = torch.mean(in_batch_loss).item() if type(
                        in_batch_loss) != int else in_batch_loss
                    pseudo_gold_loss_val = torch.mean(pseudo_gold_loss).item(
                    ) if args.no_pseudo_gold_positives != 0 else 0

        if args.eval_retrieval:
            logging_dict, evaluate_labels = retrieve_evaluate_RAC_(
                train_dl,
                evaluate_dl,
                model,
                largest_retrieval=args.topk,
                threshold=args.similarity_threshold,
                args=args,
                eval_name="dev",
                epoch=epoch,
            )

            acc, roc, pre, recall, f1, prediction, labels, macro_val = compute_metrics_retrieval(
                logging_dict, evaluate_labels, majority_voting=args.majority_voting, topk=args.topk, use_sim=True
            )

            logging_dict_test, test_labels = retrieve_evaluate_RAC_(
                train_dl,
                test_seen_dl,
                model,
                largest_retrieval=args.topk,
                threshold=args.similarity_threshold,
                args=args,
                eval_name="test",
                epoch=epoch,
            )

            acc_test, roc_test, pre_test, recall_test, f1_test, prediction, labels, macro_test = compute_metrics_retrieval(
                logging_dict_test, test_labels, majority_voting=args.majority_voting, topk=args.topk, use_sim=True
            )
        else:
            acc, roc, pre, recall, f1 = 0, 0, 0, 0, 0
            acc_test, roc_test, pre_test, recall_test, f1_test = 0, 0, 0, 0, 0
            macro_val = {"acc": 0, "macro_f1": 0, "macro_pre": 0, "macro_recall": 0, "roc": 0}
            macro_test = {"acc": 0, "macro_f1": 0, "macro_pre": 0, "macro_recall": 0, "roc": 0}
            

        if args.hybrid_loss:
            # logging at the end of each epoch
            (acc_, roc_, pre_, recall_, f1_, eval_loss_), _ = eval_and_save_epoch_end(
                args, artifacts, train_dl, evaluate_dl, test_seen_dl, model, epoch)


            
        # Print out the summary of the epoch.
        # Binary-positive P/R/F1 (legacy) ...
        print(
            "Val_Retrieval Epoch  {} acc: {:.4f} roc: {:.4f} \
pre: {:.4f} recall: {:.4f} f1: {:.4f}".format(
                epoch,
                acc,
                roc,
                pre,
                recall,
                f1,
            )
        )
        print(
            "Test_Retrieval Epoch {} acc: {:.4f} roc: {:.4f} \
pre: {:.4f} recall: {:.4f} f1: {:.4f}".format(
                epoch,
                acc_test,
                roc_test,
                pre_test,
                recall_test,
                f1_test,
            )
        )
        # ... and macro-averaged metrics (GOAL headline: macro-F1).
        print(
            "Val_Retrieval Epoch  {} macroF1: {:.4f} macroP: {:.4f} macroR: {:.4f} acc: {:.4f} roc: {:.4f}".format(
                epoch,
                macro_val["macro_f1"],
                macro_val["macro_pre"],
                macro_val["macro_recall"],
                macro_val["acc"],
                macro_val["roc"],
            )
        )
        print(
            "Test_Retrieval Epoch {} macroF1: {:.4f} macroP: {:.4f} macroR: {:.4f} acc: {:.4f} roc: {:.4f}".format(
                epoch,
                macro_test["macro_f1"],
                macro_test["macro_pre"],
                macro_test["macro_recall"],
                macro_test["acc"],
                macro_test["roc"],
            )
        )
        # print new line
        print(" ")
        # Save the model if the val criterion is the best so far.
        # Model selection is by the *retrieval* metric (Val_Retrieval acc,
        # tie-broken by ROC), NOT the hybrid classifier-head dev accuracy.
        # This guarantees the saved checkpoint == the genuinely best retrieval
        # epoch (the kNN-vote head is the primary metric for this method).
        select_acc = acc
        select_roc = roc

        # Save a checkpoint for every epoch so the fallback path can load any epoch.
        epoch_ckpt_path = args.output_path + \
            "/ckpt/epoch_model_{}_{}.pt".format(epoch, str(select_acc))
        torch.save(model.state_dict(), epoch_ckpt_path)
        all_epoch_records.append((epoch, select_acc, select_roc, epoch_ckpt_path))

        # Warmup floor: only epochs >= warmup are eligible for best-epoch selection.
        if epoch >= warmup:
            is_best = (select_acc > best_acc) or (
                select_acc == best_acc and select_roc > best_roc)
            if is_best:
                print("Current Epoch Val_Retrieval acc: ", select_acc,
                      "roc: ", select_roc, "Best model so far, saving...")
                best_acc = select_acc
                best_roc = select_roc

                # Delete the previous best model
                #if best_epoch_path is not None:
                #    if os.path.exists(best_epoch_path):
                #        os.remove(best_epoch_path)

                best_epoch_path = args.output_path + \
                    "/ckpt/best_model_{}_{}.pt".format(epoch, str(select_acc))

                torch.save(
                    model.state_dict(),
                    best_epoch_path
                )

        # If last epoch or early stop, save the model
        if epoch == args.epochs - 1:
            print("Last Epoch, saving...")
            torch.save(
                model.state_dict(),
                args.output_path +
                "/ckpt/last_model_{}_{}.pt".format(epoch, acc)
            )

    # Warmup-floor fallback: if no epoch >= warmup was ever selected (run was too
    # short, i.e. args.epochs <= warmup), fall back to the global best over ALL epochs.
    if best_epoch_path is None and all_epoch_records:
        print("WARNING: no epoch >= warmup ({}) reached; falling back to best epoch "
              "over all {} epochs.".format(warmup, len(all_epoch_records)))
        fb_epoch, fb_acc, fb_roc, fb_ckpt = max(
            all_epoch_records, key=lambda r: (r[1], r[2]))
        best_epoch_path = args.output_path + \
            "/ckpt/best_model_{}_{}.pt".format(fb_epoch, str(fb_acc))
        import shutil
        shutil.copy(fb_ckpt, best_epoch_path)
        print("Fallback selected epoch {} (Val_Retrieval acc: {}, roc: {})".format(
            fb_epoch, fb_acc, fb_roc))

    return model, best_epoch_path


def main(args):

    # <----------------- Name the model ----------------->

    if args.metric == "cos":
        loss_str = "cosSim"
    elif args.metric == "ip":
        loss_str = "innerProduct"
    elif args.metric == "l2":
        loss_str = "L2"

    if args.loss == "naive":
        loss_str += "_naive"
    elif args.loss == "triplet":
        loss_str += "_triplet"
    elif args.loss == "contrastive":
        loss_str += "_contrastive"

    hard_negative_name = "_hard_negative_{}".format(
        args.no_hard_negatives)
    
    if args.no_pseudo_gold_positives!=0 and args.no_hard_positives !=0:
        positive_name = "_PseudoGold_positive_{}_hard_positive_{}".format(
            args.no_pseudo_gold_positives, args.no_hard_positives)
    elif args.no_pseudo_gold_positives!=0:
        positive_name = "_PseudoGold_positive_{}".format(
            args.no_pseudo_gold_positives)
    elif args.no_hard_positives !=0:
        positive_name = "_hard_positive_{}".format(
            args.no_hard_positives)
    else:
        positive_name = "inbatch_positive"
    # group_name = "RAC_FB_{}_{}_{}_dense_hard_negative".format(
    #    args.fusion_mode, args.model, loss_str) if args.hard_negatives_loss else "RAC_FB_{}_{}_{}".format(args.fusion_mode, args.model, loss_str)

    # we use the group name from args
    group_name = args.group_name
    exp_name = "RAC_lr{}_Bz{}_Ep{}_{}_drop{}_topK{}_{}{}_seed{}{}{}{}".format(
        args.lr,
        args.batch_size,
        args.epochs,
        loss_str,
        args.dropout,
        args.topk,
        positive_name,
        hard_negative_name,
        args.seed,
        "_hybrid_loss" if args.hybrid_loss else "",
        args.exp_comment,
        "_{}".format(args.sparse_dictionary) if args.sparse_dictionary is not None else "",
    )
    # Construct output path
    args.output_path = os.path.join(
        args.output_path, "Retrieval", args.dataset, args.group_name, exp_name, "")
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        os.makedirs(args.output_path + "/ckpt/")
    else:
        if not args.force:
            print(args.force)
            # Abort avoid overwriting
            raise Exception("Output path already exists, aborting...")

    
    print(args)

    # <----------------- Load the data ----------------->
    if args.dataset == "FB":
        train, dev, test_seen, test_unseen = load_feats_from_CLIP(
            os.path.join(args.path, "CLIP_Embedding"), "FB", args.model
        )
    else:
        train, dev, test_seen = load_feats_from_CLIP(
            os.path.join(args.path, "CLIP_Embedding"), args.dataset, args.model
        )

    (train_dl, dev_dl, test_seen_dl), (
        train_set,
        _,
        _,
    ) = CLIP2Dataloader(
        train,
        dev,
        test_seen,
        batch_size=args.batch_size,
        return_dataset=True,
        normalize=False,
    )

    # The data loader contains a dictionary with the following keys:
    # "ids" - the id of the sample
    # "image_feats"
    # "text_feats"
    # "labels" - the label of the sample

    # Load the sparse retrieval dictionary if the path is not None
    if args.sparse_dictionary is not None:
        sparse_dict = {}
        for line in open(
            os.path.join(
                args.path,
                "Sparse_Retrieval_Dict",
                args.dataset,
                args.sparse_dictionary+".json"
            ), "r"
        ):
            subdict = json.loads(line)
            sparse_dict[subdict["id"]] = subdict
    else:
        sparse_dict = None

    # <----------------- Load the sub-clip (segment) cache for multi-granularity ----------------->
    # Only loaded when --lambda_seg > 0; lambda_seg == 0 keeps segment_cache=None
    # so the whole-video path is an EXACT no-op (identical to baseline).
    segment_cache = None
    if args.lambda_seg > 0:
        if args.subclip_cache == "auto":
            subclip_path = os.path.join(
                args.path, "CLIP_Embedding", args.dataset,
                "train_subclipK{}_{}.pt".format(args.num_subclips, args.model))
        else:
            subclip_path = args.subclip_cache
        print("Loading sub-clip (segment) cache: {}".format(subclip_path))
        sc = torch.load(subclip_path, map_location="cpu")
        # Map each parent video id -> its row in the whole-video TRAIN cache.
        parent_id_to_row = {vid: r for r, vid in enumerate(train_set.ids)}
        segment_cache = {
            "subclip_img_feats": sc["subclip_img_feats"].float(),
            "subclip_parent": sc["subclip_parent"].long(),
            "labels": sc["labels"].long(),
            "parent_id_to_row": parent_id_to_row,
            "video_text_feats": train_set.text_feats.float(),
        }
        print("Segment cache loaded: TotalSub={}, K={}, parents={}".format(
            segment_cache["subclip_img_feats"].shape[0],
            sc.get("num_subclips", args.num_subclips),
            len(parent_id_to_row)))

    # <----------------- Construct the model ----------------->

    #list(enumerate(train_dl))
    image_feat_dim = list(enumerate(train_dl))[0][1]["image_feats"].shape[1]
    text_feat_dim = list(enumerate(train_dl))[0][1]["text_feats"].shape[1]
    print("Image feature dimension: ", image_feat_dim)
    print("Text feature dimension: ", text_feat_dim)

    model = classifier_hateClipper(
        image_feat_dim, text_feat_dim, args.num_layers, args.proj_dim,
        args.map_dim, args.fusion_mode, dropout=args.dropout, batch_norm=args.batch_norm, args=args)
    model.to(args.device)
    print(model)
    # evaluate_split(train, dev, "dev")

    # <----------------- Train the model ----------------->
    model, best_epoch_path = model_pass(
        train_dl,
        dev_dl,
        test_seen_dl,
        model,
        epochs=args.epochs,
        log_interval=args.log_interval,
        args=args,
        artifacts=None,
        train_set=train_set,
        sparse_dict=sparse_dict,
        segment_cache=segment_cache,
    )

if __name__ == "__main__":
    args = parse_args()

    # set the seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
