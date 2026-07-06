import torch.nn as nn
import torch
from model.evaluate_rac import retrieve_evaluate_RAC_, final_evaluation
from model.classifier import classifier_hateClipper
from model.loss import compute_loss
import argparse
import wandb
from data_loader.rac_dataloader import CLIP2Dataloader
from data_loader.dataset import (
    load_feats_from_CLIP,
    load_archive_feats_split,
    resolve_archive_path,
)


class classifier_hateClipperArchive(nn.Module):
    """(b) 'stream' variant of classifier_hateClipper: a third feature stream
    for the MLLM structured-archive embedding.

    The archive embedding travels concatenated at the END of text_feats
    (text_feats = [original_text | archive]); the model splits it internally,
    so the frozen loss.py / consensus.py call signature
    model(img_feats, text_feats, return_embed=True) is untouched.

    Architecture mirrors classifier_hateClipper exactly, plus an archive_proj
    stream whose L2-normalised map_dim output is concatenated onto the fused
    (img x text) representation before the MLP: fusion input dim = base + map_dim.
    """

    def __init__(self, image_dim, text_dim, archive_dim, num_layers, proj_dim,
                 map_dim, fusion_mode, dropout=None, batch_norm=False, args=None):
        super(classifier_hateClipperArchive, self).__init__()
        self.fusion_mode = fusion_mode
        self.text_dim = text_dim
        self.archive_dim = archive_dim

        self.img_proj = nn.Sequential(
            nn.Linear(image_dim, map_dim), nn.Dropout(dropout[0]))
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, map_dim), nn.Dropout(dropout[0]))
        self.archive_proj = nn.Sequential(
            nn.Linear(archive_dim, map_dim), nn.Dropout(dropout[0]))

        if fusion_mode == 'concat':
            input_shape = map_dim * 2
        elif fusion_mode == 'align':
            input_shape = map_dim
        elif fusion_mode == 'cross':
            input_shape = map_dim ** 2
        # third (archive) stream is concatenated after the img/text fusion
        input_shape = input_shape + map_dim

        layers = list()
        layers.append(nn.Dropout(dropout[1]))
        for _ in range(num_layers):
            layers.append(nn.Linear(input_shape, proj_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(proj_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout[2]))
            input_shape = proj_dim
        self.mlp = nn.Sequential(*layers)
        # video datasets are binary (single logit), same as the parent default
        self.output_layer = nn.Linear(proj_dim, 1)

    def forward(self, img_feats, text_feats, return_embed=False):
        text_part = text_feats[:, : self.text_dim]
        archive_part = text_feats[:, self.text_dim:]

        img_feats = self.img_proj(img_feats)
        text_part = self.text_proj(text_part)
        archive_part = self.archive_proj(archive_part)

        img_feats = nn.functional.normalize(img_feats, p=2, dim=1)
        text_part = nn.functional.normalize(text_part, p=2, dim=1)
        archive_part = nn.functional.normalize(archive_part, p=2, dim=1)

        if self.fusion_mode == 'concat':
            x = torch.cat((img_feats, text_part), dim=1)
        elif self.fusion_mode == 'align':
            x = torch.mul(img_feats, text_part)
        elif self.fusion_mode == 'cross':
            x = torch.bmm(img_feats.unsqueeze(2),
                          text_part.unsqueeze(1)).flatten(1, 2)
        x = torch.cat((x, archive_part), dim=1)

        # Same convention as the parent: embed = pre-activation of last Linear
        embed = self.mlp[:-2](x)
        output = self.output_layer(self.mlp(x))
        if return_embed:
            return output, embed
        return output

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
        choices=["full", "driftneg", "milmax", "consensus", "selfscore"],
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
             "(max hate logit) and contrast at that representative only.\n"
             "  consensus = retrieval-consensus segment denoising "
             "(DESIGN_iter3 SS2): pseudo-label each sub-clip by the agreement "
             "between its video label and a similarity-weighted kNN vote of "
             "VIDEO-level labels from the labelled whole-video train memory; "
             "only confident (margin>=tau) segments enter the contrastive "
             "term; the (Y_v=hate, vote=benign) cell is never a positive and "
             "supplies the within-video drifting hard negative. Trained with "
             "an EM outer loop (--em_rounds).\n"
             "  selfscore = MIST/C2FPL-style control: identical pipeline but "
             "pseudo-labels come from the model's OWN sub-clip hate score "
             "instead of retrieval neighbours (warm-started from inherited "
             "labels in round 0).")

    # <----------------- MLLM structured-archive (E0b) Configs ----------------->
    arg_parser.add_argument(
        "--archive_feats", type=str, default=None,
        help="Enable archive integration. 'auto' -> "
             "<path>/CLIP_Embedding/<dataset>/{split}_archive_openai_clip-vit-"
             "large-patch14-336_HF.pt; or a template containing '{split}'; or "
             "a directory holding the per-split archive .pt files. "
             "None (default) = archive fully OFF, bit-for-bit baseline.")
    arg_parser.add_argument(
        "--archive_mode", type=str, default="knn",
        choices=["knn", "stream", "replace", "both"],
        help="How to inject the archive embedding (only when --archive_feats "
             "is set):\n"
             "  knn     = (a) kNN memory-key augmentation: eval-time retrieval "
             "keys become [normalize(fused) | alpha*normalize(archive)], so the "
             "kNN-vote similarity is a weighted combination "
             "(cos_fused + alpha^2*cos_archive)/(1+alpha^2). Training is "
             "IDENTICAL to baseline; only the retrieval classifier (and hence "
             "val selection) changes.\n"
             "  stream  = (b) third feature stream: archive embedding gets its "
             "own projection and is concatenated into the classifier-head "
             "fusion input (transported inside text_feats; the model splits "
             "internally so the frozen loss.py call signature is unchanged).\n"
             "  replace = (c) control: archive embedding REPLACES text_feats.\n"
             "  both    = (a)+(b) combined: third training stream AND eval-time "
             "kNN memory-key augmentation (--archive_alpha applies to the kNN "
             "part exactly as in knn).")
    arg_parser.add_argument(
        "--archive_alpha", type=float, default=1.0,
        help="Weight of the archive channel in archive_mode=knn. Effective "
             "similarity weight is alpha^2/(1+alpha^2) (alpha=1 -> equal).")

    # <----------------- P4: archive-field auxiliary distillation Configs ----------------->
    arg_parser.add_argument(
        "--lambda_aux", type=float, default=0.0,
        help="Weight of the P4 archive-field auxiliary distillation loss "
             "(small linear heads on the fused embedding predict the MLLM "
             "archive schema fields for each TRAIN video). 0 == OFF, exact "
             "no-op (bit-for-bit baseline). Pre-registered value 0.1.")
    arg_parser.add_argument(
        "--aux_fields", type=str,
        default="explicitness,modality,mechanism,target_group",
        help="Comma-separated archive schema fields to distil (only when "
             "--lambda_aux > 0).")
    arg_parser.add_argument(
        "--aux_archive_version", type=str, default="v2",
        help="Archive version for the P4 aux targets (train split only).")

    # <----------------- P5: counterfactual hard-negative twin Configs ----------------->
    arg_parser.add_argument(
        "--cf_negs", type=lambda x: (str(x).lower() == "true"), default=False,
        help="Add the MLLM sanitized-counterfactual twin of each TRAIN positive video as "
             "ONE extra per-anchor hard negative in the contrastive loss (weight 1.0, no "
             "new hyperparam). False == OFF, exact no-op (bit-for-bit baseline).")
    arg_parser.add_argument(
        "--cf_negs_random", type=lambda x: (str(x).lower() == "true"), default=False,
        help="Control: give each anchor a RANDOMLY chosen OTHER anchor's sanitized twin "
             "text as the extra negative (tests whether the per-anchor pairing matters). "
             "Only meaningful with --cf_negs True.")
    arg_parser.add_argument(
        "--cf_twin_cache", type=str, default="auto",
        help="Path to the twin text cache .pt (p5_generate_twins.py), or 'auto' -> "
             "<path>/CLIP_Embedding/<dataset>/train_cftwin_<model>.pt. Only loaded when "
             "--cf_negs True.")

    # <----------------- Consensus / selfscore (EM) Configs ----------------->
    arg_parser.add_argument(
        "--consensus_topk", type=int, default=10,
        help="k for the whole-video kNN vote in seg_mode=consensus.")
    arg_parser.add_argument(
        "--consensus_margin", type=float, default=0.2,
        help="Margin threshold tau on |2*vote-1| (consensus) or |2*p-1| "
             "(selfscore): only sub-clips with margin >= tau get a confident "
             "pseudo-label; the rest only join the whole-video term.")
    arg_parser.add_argument(
        "--em_rounds", type=int, default=2,
        help="Number of EM rounds (M-steps) for seg_mode consensus/selfscore. "
             "Each round retrains the head from the SAME seeded init with the "
             "current pseudo-labels, then re-encodes + re-votes (E-step). "
             "Pseudo-label flip rates are reported per round.")
    arg_parser.add_argument(
        "--consensus_use_drift", type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Keep the drifting hard-negative: push ROLE_POS anchors away from "
             "their same-video (Y_v=hate, vote=benign) sub-clip.")
    arg_parser.add_argument(
        "--consensus_conflict", type=str, default="ignore",
        choices=["ignore", "hardneg"],
        help="Handling of the (Y_v=benign, vote=hate) cell: ignore (default) "
             "or add those sub-clips to the mining corpus as label-0 "
             "confusable hard negatives.")
    arg_parser.add_argument(
        "--consensus_space", type=str, default="clip",
        choices=["clip", "archive", "blend", "mm"],
        help="Embedding space for the consensus E-step kNN vote (seg_mode="
             "consensus only). W5 prescription for the MHC-EN failure:\n"
             "  clip    = current behaviour (default): round-0 raw frozen-CLIP "
             "concat space, later EM rounds the trained head's fused space. "
             "Bit-for-bit identical to pre-W5 code.\n"
             "  archive = vote in the MLLM structured-archive CLIP-text space: "
             "each TRAIN video contributes its archive embedding as the memory "
             "key, and every sub-clip queries with its PARENT video's archive "
             "embedding (the vote was already video-level de facto -- this "
             "makes it explicit and gives it eyes on speech/on-screen-text "
             "evidence). Round-invariant across EM rounds.\n"
             "  blend   = concatenated key [l2n(base) | a*l2n(archive)] "
             "(base = clip-space key of the current round), so the vote "
             "similarity is (cos_base + a^2*cos_archive)/(1+a^2) with "
             "a = --consensus_space_alpha.\n"
             "  mm      = EXP_mm_segment_keys: multimodal SEGMENT-level keys "
             "[sqrt(1-w)*l2n(frame CLIP) | sqrt(w)*l2n(window-ASR CLIP text)] "
             "so the vote similarity is (1-w)*cos_img + w*cos_segtext, "
             "w = --mm_text_weight; memory = the same two-channel form of "
             "the whole-video keys. Requires the *_subclipK<K>_mm_* cache "
             "(--mm_subclip_cache). Round-invariant across EM rounds.")
    arg_parser.add_argument(
        "--consensus_space_alpha", type=float, default=1.0,
        help="Archive-channel weight a for --consensus_space blend "
             "(effective similarity weight a^2/(1+a^2); a=1 -> equal).")
    arg_parser.add_argument(
        "--mm_text_weight", type=float, default=0.5,
        help="Text-channel weight w in the mm vote similarity "
             "(1-w)*cos_img + w*cos_segtext (consensus_space=mm only).")
    arg_parser.add_argument(
        "--mm_empty_text", type=str, default="parent",
        choices=["parent", "zero"],
        help="Text channel of a sub-clip window with NO ASR text "
             "(consensus_space=mm only): 'parent' (default) = fall back to "
             "the parent video's whole-video CLIP text embedding; 'zero' = "
             "zero text channel (the key renormalises to visual-only).")
    arg_parser.add_argument(
        "--mm_subclip_cache", type=str, default="auto",
        help="Path to the TRAIN multimodal sub-clip cache .pt "
             "(generate_subclip_mm_embedding_HF.py), or 'auto' to derive "
             "<path>/CLIP_Embedding/<dataset>/train_subclipK<K>_mm_<model>.pt. "
             "Only loaded when --consensus_space mm.")

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
    archive_bank=None,
    aux_pack=None,
    cf_pack=None,
):
    # P4: the aux linear heads are optimised jointly with the model. When aux_pack
    # is None (lambda_aux == 0) the optimizer is built over model.parameters() only,
    # i.e. byte-identical to the baseline.
    if aux_pack is not None:
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(aux_pack["module"].parameters()),
            lr=args.lr)
    else:
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
                aux_pack=aux_pack,
                cf_pack=cf_pack,
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
                archive_bank=archive_bank,
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
                archive_bank=archive_bank,
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
    if args.archive_feats is not None:
        exp_name += "_arc-{}{}".format(
            args.archive_mode,
            "-a{}".format(args.archive_alpha)
            if args.archive_mode in ("knn", "both") else "")
    # P4: distinguish aux runs from the lambda_aux=0 floor within the same group.
    if float(getattr(args, "lambda_aux", 0.0)) > 0:
        exp_name += "_p4aux-l{}".format(args.lambda_aux)
    # P5: distinguish cf-negative runs from the floor within the same group.
    if getattr(args, "cf_negs", False):
        exp_name += "_p5cf{}".format("rand" if getattr(args, "cf_negs_random", False) else "")
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

    # <----------------- MLLM structured-archive (E0b) injection ----------------->
    # Fully inert when --archive_feats is None (bit-for-bit baseline).
    archive_bank = None
    archive_dim = 0
    if args.archive_feats is not None:
        if args.dataset == "FB":
            raise NotImplementedError(
                "--archive_feats is only wired for the 3-split video datasets")
        arc_by_split = {}
        for split, tup in (("train", train), ("dev_seen", dev),
                           ("test_seen", test_seen)):
            arc_path = resolve_archive_path(
                args.archive_feats, args.path, args.dataset, split)
            # tup = [ids, img_feats, text_feats, labels]; re-order archive rows
            # to the main cache id order with STRICT id lookup (raises on any
            # missing id -- no silent zero-fill).
            arc_by_split[split] = load_archive_feats_split(arc_path, list(tup[0]))
            print("[archive] {} <- {} ({} rows, dim {})".format(
                split, arc_path, arc_by_split[split].shape[0],
                arc_by_split[split].shape[1]))
        archive_dim = arc_by_split["train"].shape[1]

        if args.archive_mode == "replace":
            # (c) control: archive embedding REPLACES the text stream.
            for split, tup in (("train", train), ("dev_seen", dev),
                               ("test_seen", test_seen)):
                tup[2] = arc_by_split[split]
        if args.archive_mode in ("stream", "both"):
            # (b) third stream: transported inside text_feats; model splits.
            for split, tup in (("train", train), ("dev_seen", dev),
                               ("test_seen", test_seen)):
                tup[2] = torch.cat(
                    (tup[2].float(), arc_by_split[split]), dim=1)
        if args.archive_mode in ("knn", "both"):
            # (a) kNN memory-key augmentation: build an id -> archive-embedding
            # bank over all splits; used ONLY inside eval-time retrieval
            # (evaluate_rac.retrieve_evaluate_RAC_). Training is untouched.
            all_ids = list(train[0]) + list(dev[0]) + list(test_seen[0])
            all_feats = torch.cat(
                (arc_by_split["train"], arc_by_split["dev_seen"],
                 arc_by_split["test_seen"]), dim=0)
            archive_bank = {
                "row": {vid: r for r, vid in enumerate(all_ids)},
                "feats": all_feats,
                "alpha": args.archive_alpha,
            }
            # leak canary: video ids must be unique across splits
            assert len(archive_bank["row"]) == len(all_ids), \
                "Duplicate video ids across splits -- investigate before training"
            print("[archive] kNN bank built: {} ids, alpha={}".format(
                len(all_ids), args.archive_alpha))

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
        # W5: archive-space consensus voting. Load the per-video MLLM
        # structured-archive embeddings aligned to the TRAIN cache order and
        # stash them for the E-step. Only touched when the non-default voting
        # space is requested; --consensus_space clip (default) leaves this
        # block dead and the run bit-for-bit identical to pre-W5 code.
        if (args.seg_mode == "consensus"
                and getattr(args, "consensus_space", "clip")
                in ("archive", "blend")):
            arc_src = (args.archive_feats
                       if args.archive_feats is not None else "auto")
            arc_train_path = resolve_archive_path(
                arc_src, args.path, args.dataset, "train")
            segment_cache["archive_feats"] = load_archive_feats_split(
                arc_train_path, list(train_set.ids))
            print("[consensus] E-step voting space '{}' (alpha={}): train "
                  "archive <- {} ({} rows, dim {})".format(
                      args.consensus_space, args.consensus_space_alpha,
                      arc_train_path,
                      segment_cache["archive_feats"].shape[0],
                      segment_cache["archive_feats"].shape[1]))
        # EXP_mm_segment_keys: multimodal (frame + window-ASR text) sub-clip
        # keys for the E-step vote. Only touched when --consensus_space mm;
        # every other configuration leaves this block dead (bit-for-bit).
        if (args.seg_mode == "consensus"
                and getattr(args, "consensus_space", "clip") == "mm"):
            if args.mm_subclip_cache == "auto":
                mm_path = os.path.join(
                    args.path, "CLIP_Embedding", args.dataset,
                    "train_subclipK{}_mm_{}.pt".format(
                        args.num_subclips, args.model))
            else:
                mm_path = args.mm_subclip_cache
            print("Loading multimodal sub-clip cache: {}".format(mm_path))
            mmc = torch.load(mm_path, map_location="cpu")
            # Hard alignment guards: the mm cache must be the SAME sub-clip
            # corpus (row-for-row) as the visual cache loaded above.
            assert torch.equal(mmc["subclip_parent"].long(),
                               segment_cache["subclip_parent"]), \
                "mm cache subclip_parent mismatch with the visual cache"
            assert torch.equal(mmc["subclip_img_feats"].float(),
                               segment_cache["subclip_img_feats"]), \
                "mm cache visual feats differ from the visual cache"
            segment_cache["subclip_txt_feats"] = \
                mmc["subclip_txt_feats"].float()
            segment_cache["subclip_txt_has_text"] = \
                mmc["subclip_txt_has_text"].bool()
            n_txt = int(segment_cache["subclip_txt_has_text"].sum())
            total = segment_cache["subclip_txt_has_text"].numel()
            print("[consensus] E-step voting space 'mm' (w={}, empty={}): "
                  "{}/{} sub-clip windows have ASR text ({:.1f}%) <- {}".format(
                      args.mm_text_weight, args.mm_empty_text, n_txt, total,
                      100.0 * n_txt / max(total, 1), mm_path))

    # <----------------- Construct the model ----------------->

    #list(enumerate(train_dl))
    image_feat_dim = list(enumerate(train_dl))[0][1]["image_feats"].shape[1]
    text_feat_dim = list(enumerate(train_dl))[0][1]["text_feats"].shape[1]
    print("Image feature dimension: ", image_feat_dim)
    print("Text feature dimension: ", text_feat_dim)

    def build_model():
        # archive_mode 'stream'/'both' uses the third-stream variant; every
        # other configuration (archive off / knn / replace) constructs the
        # ORIGINAL classifier_hateClipper with identical arguments (same RNG
        # draws).
        if args.archive_feats is not None and args.archive_mode in ("stream", "both"):
            return classifier_hateClipperArchive(
                image_feat_dim, text_feat_dim - archive_dim, archive_dim,
                args.num_layers, args.proj_dim, args.map_dim, args.fusion_mode,
                dropout=args.dropout, batch_norm=args.batch_norm, args=args)
        return classifier_hateClipper(
            image_feat_dim, text_feat_dim, args.num_layers, args.proj_dim,
            args.map_dim, args.fusion_mode, dropout=args.dropout,
            batch_norm=args.batch_norm, args=args)

    model = build_model()
    model.to(args.device)
    print(model)
    # evaluate_split(train, dev, "dev")

    # <----------------- P4: archive-field auxiliary distillation heads ----------------->
    # Fully inert when --lambda_aux == 0 (no heads built, no RNG drawn) -> the
    # lambda_aux=0 run is byte-identical to the baseline. When lambda_aux > 0, the
    # aux heads are built with the global CPU-RNG state SAVED and RESTORED, so the
    # DataLoader shuffle order (RandomSampler uses the global CPU generator) is
    # unchanged and the ONLY difference from the floor is the aux gradient.
    aux_pack = None
    if float(getattr(args, "lambda_aux", 0.0)) > 0:
        from utils.p4_archive_fields import (
            load_archive_records, derive_vocab, encode_record)
        fields = [f.strip() for f in args.aux_fields.split(",") if f.strip()]
        arc_recs = load_archive_records(
            args.dataset, "train", args.aux_archive_version)
        vocab = derive_vocab(arc_recs)
        train_ids = list(train_set.ids)
        id_to_row = {vid: r for r, vid in enumerate(train_ids)}
        enc = [encode_record(arc_recs.get(vid), vocab) for vid in train_ids]
        targets, valids, specs, dims = {}, {}, {}, {}
        for field in fields:
            specs[field] = {"type": vocab[field]["type"]}
            dims[field] = len(vocab[field]["classes"])
            dtype = (torch.long if vocab[field]["type"] == "single"
                     else torch.float)
            targets[field] = torch.tensor(
                [enc_i[field][0] for enc_i in enc], dtype=dtype,
                device=args.device)
            valids[field] = torch.tensor(
                [bool(enc_i[field][1]) for enc_i in enc], dtype=torch.bool,
                device=args.device)
        rng_cpu = torch.get_rng_state()
        aux_module = nn.ModuleDict(
            {field: nn.Linear(args.proj_dim, dims[field]) for field in fields})
        torch.set_rng_state(rng_cpu)
        aux_module.to(args.device)
        aux_pack = {"module": aux_module, "id_to_row": id_to_row,
                    "specs": specs, "targets": targets, "valids": valids,
                    "vocab": vocab, "fields": fields}
        print("[p4aux] lambda_aux={} fields={} dims={} coverage(mech={}, tg={})".format(
            args.lambda_aux, fields, dims,
            vocab["mechanism"].get("coverage"),
            vocab["target_group"].get("coverage")))
        print("[p4aux] valid train targets per field: {} / {}".format(
            {f: int(valids[f].sum().item()) for f in fields}, len(train_ids)))

    # <----------------- P5: counterfactual twin negative bank ----------------->
    # Inert when --cf_negs False (no cache load) -> byte-identical baseline. No new params
    # (the twin reuses the SAME head), so the optimizer/model init are untouched; only the
    # per-step loss gets one extra negative. Loading the cache draws no torch RNG.
    cf_pack = None
    if getattr(args, "cf_negs", False):
        if args.cf_twin_cache == "auto":
            cf_path = os.path.join(
                args.path, "CLIP_Embedding", args.dataset,
                "train_cftwin_{}.pt".format(args.model))
        else:
            cf_path = args.cf_twin_cache
        tw = torch.load(cf_path, map_location="cpu")
        tw_ids = [i for sub in tw["ids"] for i in sub]
        twin_text = tw["text_feats"].float()
        flipped = tw["flipped"].bool()
        assert twin_text.shape[1] == text_feat_dim, \
            "twin text dim {} != model text dim {}".format(
                twin_text.shape[1], text_feat_dim)
        if getattr(args, "cf_negs_random", False):
            # each valid anchor gets ANOTHER valid anchor's twin text (seeded derangement)
            valid_rows = [r for r in range(len(tw_ids)) if bool(flipped[r])]
            assert len(valid_rows) >= 2, \
                "cf_negs_random needs >=2 verified twins to derange"
            rng = np.random.default_rng(args.seed)
            perm = list(valid_rows)
            deranged = False
            for _ in range(20):
                rng.shuffle(perm)
                if all(a != b for a, b in zip(valid_rows, perm)):
                    deranged = True
                    break
            if not deranged:
                # guaranteed fixed-point-free fallback: cyclic shift by 1
                perm = valid_rows[1:] + valid_rows[:1]
            remapped = twin_text.clone()
            for a, b in zip(valid_rows, perm):
                remapped[a] = twin_text[b]
            twin_text = remapped
        cf_pack = {
            "id_to_row": {vid: r for r, vid in enumerate(tw_ids)},
            "twin_text": twin_text.to(args.device),
            "valid": flipped.to(args.device),
        }
        print("[p5cf] cf_negs=True random={} twins={} valid(flipped)={} <- {}".format(
            getattr(args, "cf_negs_random", False), len(tw_ids),
            int(flipped.sum().item()), cf_path))

    # <----------------- Train the model ----------------->
    if segment_cache is not None and args.seg_mode in ("consensus", "selfscore"):
        # EM driver (DESIGN_iter3 SS2). Each round: M-step = retrain the head
        # from the SAME seeded init with the current sub-clip pseudo-labels;
        # E-step = re-encode memory/sub-clips with the round's best (val-
        # selected) head, rebuild the FAISS index, re-assign pseudo-labels and
        # report the flip rate. Round-0 labels: consensus -> kNN vote in the
        # raw frozen-CLIP space; selfscore -> inherited labels (warm start).
        # Only active when lambda_seg > 0; the lambda_seg == 0 baseline path
        # below is untouched (exact no-op guarantee).
        from utils.consensus import (
            consensus_estep, selfscore_init, selfscore_estep, flip_rate)

        base_output = args.output_path
        em_rounds = max(1, int(args.em_rounds))
        if args.seg_mode == "consensus":
            roles, margins = consensus_estep(segment_cache, train_set, None, args)
        else:
            roles, margins = selfscore_init(segment_cache)
        prev_roles = roles

        for em_round in range(em_rounds):
            segment_cache["pseudo_role"] = roles
            segment_cache["pseudo_margin"] = margins
            args.output_path = os.path.join(
                base_output, "em_round{}".format(em_round), "")
            os.makedirs(args.output_path + "/ckpt/", exist_ok=True)
            # Same seeded init every round: isolates the pseudo-label effect.
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            model = build_model()
            model.to(args.device)
            print("[EM] ===== round {}/{} (seg_mode={}) =====".format(
                em_round + 1, em_rounds, args.seg_mode))
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
                archive_bank=archive_bank,
            )
            # E-step with the round's best (val-selected) head. After the LAST
            # round this is only used to report the final flip rate.
            if best_epoch_path is not None and os.path.exists(best_epoch_path):
                model.load_state_dict(torch.load(
                    best_epoch_path, map_location=args.device))
            if args.seg_mode == "consensus":
                roles, margins = consensus_estep(
                    segment_cache, train_set, model, args)
            else:
                roles, margins = selfscore_estep(
                    segment_cache, train_set, model, args)
            fr = flip_rate(prev_roles, roles)
            print("[EM] pseudo-label flip rate after round {}: {:.4f}".format(
                em_round + 1, fr))
            prev_roles = roles
        args.output_path = base_output
        print("[EM] finished {} rounds; final round best ckpt: {}".format(
            em_rounds, best_epoch_path))
    else:
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
            archive_bank=archive_bank,
            aux_pack=aux_pack,
            cf_pack=cf_pack,
        )

if __name__ == "__main__":
    args = parse_args()

    # set the seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
