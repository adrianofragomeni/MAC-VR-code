from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import json
import os
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
import datetime
from os.path import join, exists

import torch
import torch.nn.functional as F
from tvr.models.tokenization_clip import SimpleTokenizer as ClipTokenizer
from tvr.dataloaders.data_dataloaders import DATALOADER_DICT
from tvr.models.modeling import DiCoSA, AllGather
from tvr.models.optimization import BertAdam
from tvr.utils.metrics import compute_metrics

from tvr.utils.comm import is_main_process, synchronize
from tvr.utils.logger import setup_logger
from tvr.utils.metric_logger import MetricLogger

allgather = AllGather.apply

global logger
#print(torch.cuda.current_device())
#os.environ['LOCAL_RANK']


def get_args(description='Disentangled Representation Learning for Text-Video Retrieval'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_train", type=int, default=0, help="Whether to run training.")
    parser.add_argument("--do_eval", type=int, default=0, help="Whether to run evaluation.")

    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")
    parser.add_argument('--anno_path', type=str, default='data/MSR-VTT/anns', help='annotation path')
    parser.add_argument('--video_path', type=str, default='data/MSR-VTT/videos', help='video path')

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--coef_lr', type=float, default=1e-3, help='coefficient for bert branch.')
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.")
    parser.add_argument('--weight_decay', type=float, default=0.2, help='weight decay')
    parser.add_argument('--epochs', type=int, default=5, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=128, help='batch size eval')

    parser.add_argument('--max_words', type=int, default=32, help='max text token number')
    parser.add_argument('--max_frames', type=int, default=12, help='max key frames')
    parser.add_argument('--video_framerate', type=int, default=1, help='framerate to sample video frame')

    parser.add_argument("--device", default='cpu', type=str, help="cpu/cuda")
    parser.add_argument("--world_size", default=1, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument("--distributed", default=0, type=int, help="multi machine DDP")

    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--base_encoder", default="ViT-B/32", type=str, help="Choose a CLIP version")
    parser.add_argument('--agg_module', type=str, default="seqTransf", choices=["None", "seqLSTM", "seqTransf"],
                        help="choice a feature aggregation module for video.")
    parser.add_argument('--interaction', type=str, default='wti', help="interaction type for retrieval.")
    parser.add_argument('--num_hidden_layers', type=int, default=4)
    
    parser.add_argument('--temp', type=float, default=5)

    parser.add_argument('--center', type=int, default=8)
    
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--beta', type=float, default=0.005)
    
    parser.add_argument("--t2v_k", default=1, type=int)
    parser.add_argument("--t2v_beta", default=70, type=float)
    parser.add_argument("--t2v_theta", default=3, type=float)
    parser.add_argument("--v2t_k", default=1, type=int)
    parser.add_argument("--v2t_beta", default=70, type=float)
    parser.add_argument("--v2t_theta", default=5, type=float)
    parser.add_argument('--t2v_temp', type=float, default=0.01)
    parser.add_argument('--v2t_temp', type=float, default=0.01)

    parser.add_argument("--number_visual_tags_train", default=8, type=int)
    parser.add_argument("--number_textual_tags_test", default=8, type=int)
    
    parser.add_argument("--number_textual_tags_train", default=10, type=int)
    parser.add_argument("--number_visual_tags_test", default=10, type=int)
    
    
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--output_path_concepts", default=None, type=str, required=False, help="Output path concepts.")
    args = parser.parse_args()

    return args


def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available():
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        args.world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    if torch.cuda.is_available():
        torch.distributed.barrier()
    logger.info("local_rank: {} world_size: {}".format(args.local_rank, args.world_size))

    if args.batch_size % args.world_size != 0 or args.batch_size_val % args.world_size != 0:
        raise ValueError(
            "Invalid batch_size/batch_size_val and world_size parameter: {}%{} and {}%{}, should be == 0".format(
                args.batch_size, args.world_size, args.batch_size_val, args.world_size))

    logger.info("Effective parameters:")
    for key in sorted(args.__dict__):
        logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args


def build_model(args):
    model = DiCoSA(args)
    if args.init_model:
        if not exists(args.init_model):
            raise FileNotFoundError
        model_state_dict = torch.load(args.init_model, map_location='cpu')
        model.load_state_dict(model_state_dict, strict=False)

    model.to(args.device)
    return model


def build_dataloader(args):
    ## ####################################
    # dataloader loading
    ## ####################################
    tokenizer = ClipTokenizer()
    assert args.datatype in DATALOADER_DICT

    assert DATALOADER_DICT[args.datatype]["test"] is not None or DATALOADER_DICT[args.datatype]["val"] is not None

    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)

    if DATALOADER_DICT[args.datatype]["val"] is not None:
        val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, subset="val")
    else:
        val_dataloader, val_length = test_dataloader, test_length

    ## report validation results if the ["test"] is None
    if test_dataloader is None:
        test_dataloader, test_length = val_dataloader, val_length

    if isinstance(test_length, int):
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %d", val_length)
    elif len(test_length) == 2:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %dt %dv", test_length[0], test_length[1])
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d %d", len(test_dataloader[0]), len(test_dataloader[1]))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %dt %dv", val_length[0], val_length[1])

    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_length)
        logger.info("  Batch size = %d", args.batch_size)
        logger.info("  Num steps = %d", len(train_dataloader) * args.epochs)
    else:
        train_dataloader, train_sampler = None, None

    train_dataloader0, train_length, train_sampler0 = DATALOADER_DICT[args.datatype]["train_test"](args, tokenizer)

    return test_dataloader, val_dataloader, train_dataloader, train_sampler, train_dataloader0, train_sampler0


def prep_optimizer(args, model, num_train_optimization_steps, local_rank):
    if hasattr(model, 'module'):
        model = model.module
    lr = args.lr  # 0.0001
    coef_lr = args.coef_lr  # 0.001
    weight_decay = args.weight_decay  # 0.2
    warmup_proportion = args.warmup_proportion
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n]
    decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n]

    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n]
    no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." not in n]

    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': lr * coef_lr},
        {'params': [p for n, p in decay_noclip_param_tp], 'weight_decay': weight_decay},
        {'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': lr * coef_lr},
        {'params': [p for n, p in no_decay_noclip_param_tp], 'weight_decay': 0.0}
    ]

    scheduler = None
    
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=warmup_proportion,
                         schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                         t_total=num_train_optimization_steps, weight_decay=weight_decay,
                         max_grad_norm=1.0)

    if torch.cuda.is_available():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                          find_unused_parameters=True)

    return optimizer, scheduler, model


def save_model(epoch, args, model, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name == "" else type_name + ".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)
    return output_model_file


def reduce_loss(loss, args):
    world_size = args.world_size
    if world_size < 2:
        return loss
    with torch.no_grad():
        torch.distributed.reduce(loss, dst=0)
        if torch.distributed.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            loss /= world_size
    return loss


def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                scheduler, global_step, max_steps, val_dataloader):
    global logger
    global best_score
    global meters

    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    total_loss = 0
    
    end = time.time()
    logit_scale = 0
    for step, batch in enumerate(train_dataloader, start=1):
        global_step += 1
        data_time = time.time() - end

        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        text_ids, text_mask,visual_c_ids, visual_c_mask, textual_c_ids, textual_c_mask, video, video_mask, inds, idx = batch
        loss, nce_loss, inter_loss, intra_t_loss, intra_v_loss = model(text_ids, text_mask,visual_c_ids, visual_c_mask, textual_c_ids, textual_c_mask, video, video_mask, idx, global_step)

        if n_gpu > 1:
            # print(loss.shape)
            loss = loss.mean()  # mean() to average on multi-gpu.
            nce_loss = nce_loss.mean()
            inter_loss = inter_loss.mean()
            intra_t_loss = intra_t_loss.mean()
            intra_v_loss = intra_v_loss.mean()

        with torch.autograd.detect_anomaly():
            loss.backward()
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()  # Update learning rate schedule
        
        optimizer.zero_grad()

        # https://github.com/openai/CLIP/issues/46
        if hasattr(model, 'module'):
            torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
            logit_scale = model.module.clip.logit_scale.exp().item()
        else:
            torch.clamp_(model.clip.logit_scale.data, max=np.log(100))
            logit_scale = model.clip.logit_scale.exp().item()

        batch_time = time.time() - end
        end = time.time()

        reduced_l = reduce_loss(loss, args)
        reduced_nce_loss = reduce_loss(nce_loss, args)
        reduced_inter_loss = reduce_loss(inter_loss, args)
        reduced_intra_t_loss = reduce_loss(intra_t_loss, args)
        reduced_intra_v_loss= reduce_loss(intra_v_loss, args)
        meters.update(time=batch_time, data=data_time, loss=float(reduced_l),
                      loss_NCE=float(reduced_nce_loss), loss_InterO=float(reduced_inter_loss), loss_IntraT=float(reduced_intra_t_loss), loss_IntraV=float(reduced_intra_v_loss))

        eta_seconds = meters.time.global_avg * (max_steps - global_step)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if (global_step % log_step == 0 or global_step == 1) and is_main_process():
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "epoch: {epoch}/{max_epoch}",
                        "iteration: {iteration}/{max_iteration}",
                        "{meters}",
                        "lr: {lr}",
                        "logit_scale: {logit_scale:.2f}"
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    epoch=epoch,
                    max_epoch=args.epochs,
                    iteration=global_step,
                    max_iteration=max_steps,
                    meters=str(meters),
                    lr="/".join([str('%.9f' % itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                    logit_scale=logit_scale,
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if global_step % (log_step * 3) == 0 or global_step == 1:
            R1 = eval_epoch(args, model, val_dataloader, args.device)
            if args.local_rank == 0:
                output_model_file = save_model(epoch, args, model, type_name="step{}".format(global_step))
                if best_score <= R1:
                    best_score = R1
                    output_model_file = save_model(epoch, args, model, type_name="best")
            model.train()

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step


def _run_on_single_gpu(args, model, t_mask_list, v_mask_list, t_feat_list, v_feat_list, cls_list,batch_visual_c_mask_list, batch_textual_c_mask_list, batch_cls_textual_c_list, batch_cls_visual_c_list, train_t_mask_list, train_v_mask_list, train_t_feat_list, train_v_feat_list, train_cls_list,train_cls_visual_c_list,train_cls_textual_c_list, mini_batch=16):
    # Returns list of retrieved top k videos based on the sims matrix
    def get_retrieved_videos(sims, k, theta):
        argm = np.argsort(-sims, axis=1)
        topk = argm[:,:k].reshape(-1)
        retrieved_videos, occurrence_count = np.unique(topk, return_counts=True)
        return retrieved_videos[occurrence_count>=theta]

    # Returns list of indices to normalize from sims based on videos
    def get_index_to_normalize(sims, videos):
        argm = np.argsort(-sims, axis=1)[:,0]
        result = np.array(list(map(lambda x: x in videos, argm)))
        result = np.nonzero(result)
        return result
    
    def qb_norm(train_test, test_test, k, beta, theta):
        retrieved_videos = get_retrieved_videos(train_test, k, theta)
        test_test_normalized = test_test
        train_test = np.exp(train_test*beta)
        test_test = np.exp(test_test*beta)

        normalizing_sum = np.sum(train_test, axis=0)
        index_for_normalizing = get_index_to_normalize(test_test, retrieved_videos)
        test_test_normalized[index_for_normalizing, :] = \
            np.divide(test_test[index_for_normalizing, :], normalizing_sum)
        return test_test_normalized
    
    tsne_save=False
    sim_matrix = []
    visual_concepts=[]
    text_concepts=[]
    text=[]
    video=[]

    logger.info('[finish] map to main gpu')

    batch_t_mask = torch.split(t_mask_list, mini_batch)
    batch_v_mask = torch.split(v_mask_list, mini_batch)
    batch_t_feat = torch.split(t_feat_list, mini_batch)
    batch_v_feat = torch.split(v_feat_list, mini_batch)
    batch_cls_feat = torch.split(cls_list, mini_batch)
    
    batch_cls_visual_c_feat = torch.split(batch_cls_visual_c_list, mini_batch)
    batch_cls_textual_c_feat = torch.split(batch_cls_textual_c_list, mini_batch)
    
    train_batch_t_mask = torch.split(train_t_mask_list, mini_batch)
    train_batch_v_mask = torch.split(train_v_mask_list, mini_batch)
    train_batch_t_feat = torch.split(train_t_feat_list, mini_batch)
    train_batch_v_feat = torch.split(train_v_feat_list, mini_batch)
    train_batch_cls_feat = torch.split(train_cls_list, mini_batch)

    train_batch_cls_visual_c_feat = torch.split(train_cls_visual_c_list, mini_batch)
    train_batch_cls_textual_c_feat = torch.split(train_cls_textual_c_list, mini_batch)
    
    logger.info('[finish] map to main gpu')
    
    with torch.no_grad():        
        for idx1, (t_mask, t_feat, cls,cls_textual_c) in enumerate(zip(batch_t_mask, batch_t_feat, batch_cls_feat,batch_cls_textual_c_feat)):
            each_row = []
            if tsne_save:
                each_row_visual_concepts = []
                each_row_text_concepts = []
                each_row_text = []
                each_row_video = []
            
            for idx2, (v_mask, v_feat, cls_visual_c) in enumerate(zip(batch_v_mask, batch_v_feat,batch_cls_visual_c_feat)):
                if tsne_save:
                    logits, z_v_norm, z_t_norm, z_a_norm, z_b_norm, *_tmp= model.get_similarity_logits(t_feat, cls,cls_visual_c,cls_textual_c, v_feat, t_mask, v_mask)
                    each_row.append(logits.cpu().detach().numpy())
                    each_row_visual_concepts.append(z_v_norm.cpu().detach().numpy().tolist())
                    each_row_text_concepts.append(z_t_norm.cpu().detach().numpy().tolist())
                    each_row_text.append(z_a_norm.cpu().detach().numpy().tolist())
                    each_row_video.append(z_b_norm.cpu().detach().numpy().tolist())
                else:
                    logits,*_tmp= model.get_similarity_logits(t_feat, cls,cls_visual_c,cls_textual_c, v_feat, t_mask, v_mask)
                    each_row.append(logits.cpu().detach().numpy())
                
            each_row = np.concatenate(tuple(each_row), axis=-1)
          
            sim_matrix.append(each_row)
            if tsne_save:
                visual_concepts.append(each_row_visual_concepts)
                text_concepts.append(each_row_text_concepts)
                text.append(each_row_text)
                video.append(each_row_video)


    sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
    
    if tsne_save:
        print("Creation Visual Aux Concepts")
        visual_concepts=visual_concepts[0]
        visual_concepts=np.concatenate(tuple(visual_concepts), axis=0)
        
        print("Creation Textual Aux Concepts")
        t_concepts=[]
        for lst in text_concepts:
            t_concepts.append(lst[0])
        t_concepts=np.concatenate(tuple(t_concepts), axis=0)
        
        print("Creation Text Concepts")
        t=[]
        for lst in text:
            t.append(lst[0])
        t=np.concatenate(tuple(t), axis=0)
        
        print("Creation Video Concepts")
        v= []
        for i in video:
            v.append(np.concatenate(tuple(i), axis=1))
        
        v = np.concatenate(tuple(v), axis=0)
        
        concepts={"visual_auxiliary_concepts":visual_concepts.tolist(), "textual_auxiliary_concepts":t_concepts.tolist(),"text":t.tolist(),"video":v.tolist()}   
        print("Saving concepts...")
        with open(args.output_path_concepts, 'w') as fp:
            json.dump(concepts, fp)  

    train_test_t2v, train_test_v2t = [], []

    with torch.no_grad():
        for idx1, (t_mask, t_feat, cls, cls_textual_c) in enumerate(zip(train_batch_t_mask, train_batch_t_feat, train_batch_cls_feat, train_batch_cls_textual_c_feat)):
            each_row = []
            for idx2, (v_mask, v_feat,cls_visual_c) in enumerate(zip(batch_v_mask, batch_v_feat, batch_cls_visual_c_feat)):
                logits, *_tmp= model.get_similarity_logits(t_feat, cls,cls_visual_c,cls_textual_c, v_feat, t_mask, v_mask)
                logits = logits.cpu().detach().numpy()
                each_row.append(logits)
            each_row = np.concatenate(tuple(each_row), axis=-1)
            train_test_t2v.append(each_row)
    train_test_t2v = np.concatenate(tuple(train_test_t2v), axis=0)

    with torch.no_grad():
        for idx1, (t_mask, t_feat, cls, cls_textual_c) in enumerate(zip(batch_t_mask, batch_t_feat, batch_cls_feat, batch_cls_textual_c_feat)):
            each_row = []
            for idx2, (v_mask, v_feat, cls_visual_c) in enumerate(zip(train_batch_v_mask, train_batch_v_feat, train_batch_cls_visual_c_feat)):
                logits, *_tmp = model.get_similarity_logits(t_feat, cls,cls_visual_c,cls_textual_c, v_feat, t_mask, v_mask)
                logits = logits.cpu().detach().numpy()
                each_row.append(logits)
            each_row = np.concatenate(tuple(each_row), axis=-1)
            train_test_v2t.append(each_row)
    train_test_v2t = np.concatenate(tuple(train_test_v2t), axis=0)

    t2v_normalized = qb_norm(train_test_t2v.copy(), sim_matrix.copy(), args.t2v_k, args.t2v_beta, args.t2v_theta)
    v2t_normalized = qb_norm(train_test_v2t.T.copy(), sim_matrix.T.copy(), args.v2t_k, args.v2t_beta, args.v2t_theta)


    return sim_matrix, t2v_normalized, v2t_normalized    
    return sim_matrix, t2v_normalized, v2t_normalized


def eval_epoch(args, model, test_dataloader, device):
    
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    model.eval()
    # ----------------------------
    # 1. cache the features
    # ----------------------------
    batch_mask_t, batch_mask_v, batch_feat_t, batch_feat_v, ids_t = [], [], [], [], []
    batch_cls = []
    batch_visual_c_mask, batch_textual_c_mask, batch_cls_textual_c, batch_cls_visual_c= [], [], [], []
    

    batch_mask_train_t, batch_mask_train_v, batch_feat_train_t, batch_feat_train_v, ids_train_t = [], [], [], [], []
    batch_train_cls = []
    batch_train_cls_textual, batch_train_cls_visual= [], []
    
    
    with torch.no_grad():
    
        logger.info('[start] extract train feature')
        for bid, batch in enumerate(train_dataloader0):
            batch = tuple(t.to(device) for t in batch)
            text_ids, text_mask, visual_c_ids, visual_c_mask, textual_c_ids, textual_c_mask, video, video_mask, inds, _ = batch
            
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            
            text_feat, video_feat, cls, cls_visual_c,cls_textual_c = model.get_text_video_feat(text_ids, text_mask, video, video_mask, visual_c_ids, visual_c_mask,textual_c_ids, textual_c_mask)

            ids_train_t.append(inds)
            batch_mask_train_t.append(text_mask)
            batch_mask_train_v.append(video_mask)
            batch_feat_train_t.append(text_feat)
            batch_feat_train_v.append(video_feat)
            batch_train_cls.append(cls)
            batch_train_cls_textual.append(cls_textual_c)
            batch_train_cls_visual.append(cls_visual_c)
            
            print("{}/{}\r".format(bid, len(train_dataloader0)), end="")

        ids_train_t = allgather(torch.cat(ids_train_t, dim=0), args).squeeze()
        batch_mask_train_t = allgather(torch.cat(batch_mask_train_t, dim=0), args)
        batch_mask_train_v = allgather(torch.cat(batch_mask_train_v, dim=0), args)
        batch_feat_train_t = allgather(torch.cat(batch_feat_train_t, dim=0), args)
        batch_feat_train_v = allgather(torch.cat(batch_feat_train_v, dim=0), args)
        batch_train_cls = allgather(torch.cat(batch_train_cls, dim=0), args)
        batch_train_cls_textual = allgather(torch.cat(batch_train_cls_textual, dim=0), args)
        batch_train_cls_visual = allgather(torch.cat(batch_train_cls_visual, dim=0), args)
        logger.info('[finish] extract train feature')
        
        tic = time.time()
        
        logger.info('[start] extract text+video feature')
        for batch in tqdm(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            text_ids, text_mask, visual_c_ids, visual_c_mask, textual_c_ids, textual_c_mask, video, video_mask, inds, _ = batch
            
            text_feat, video_feat, cls, cls_visual_c,cls_textual_c = model.get_text_video_feat(text_ids, text_mask, video, video_mask, visual_c_ids, visual_c_mask,textual_c_ids, textual_c_mask)
            ids_t.append(inds)
            batch_mask_t.append(text_mask)
            batch_mask_v.append(video_mask)
            batch_visual_c_mask.append(visual_c_mask)
            batch_textual_c_mask.append(textual_c_mask)
            batch_feat_t.append(text_feat)
            batch_feat_v.append(video_feat)
            batch_cls.append(cls)
            batch_cls_textual_c.append(cls_textual_c)
            batch_cls_visual_c.append(cls_visual_c)
        ids_t = allgather(torch.cat(ids_t, dim=0), args).squeeze()
        batch_mask_t = allgather(torch.cat(batch_mask_t, dim=0), args)
        batch_mask_v = allgather(torch.cat(batch_mask_v, dim=0), args)
        batch_feat_t = allgather(torch.cat(batch_feat_t, dim=0), args)
        batch_feat_v = allgather(torch.cat(batch_feat_v, dim=0), args)
        batch_cls = allgather(torch.cat(batch_cls, dim=0), args)
        
        batch_cls_visual_c = allgather(torch.cat(batch_cls_visual_c, dim=0), args)
        batch_cls_textual_c = allgather(torch.cat(batch_cls_textual_c, dim=0), args)
        batch_visual_c_mask = allgather(torch.cat(batch_visual_c_mask, dim=0), args)
        batch_textual_c_mask = allgather(torch.cat(batch_textual_c_mask, dim=0), args)
        
        batch_mask_t[ids_t] = batch_mask_t.clone()
        batch_mask_v[ids_t] = batch_mask_v.clone()
        batch_feat_t[ids_t] = batch_feat_t.clone()
        batch_feat_v[ids_t] = batch_feat_v.clone()
        batch_cls[ids_t] = batch_cls.clone()
        
        batch_cls_visual_c[ids_t] = batch_cls_visual_c.clone()
        batch_cls_textual_c[ids_t] = batch_cls_textual_c.clone()
        batch_visual_c_mask[ids_t] = batch_visual_c_mask.clone()
        batch_textual_c_mask[ids_t] = batch_textual_c_mask.clone()
        
        batch_mask_t = batch_mask_t[:ids_t.max() + 1, ...]
        batch_mask_v = batch_mask_v[:ids_t.max() + 1, ...]
        batch_feat_t = batch_feat_t[:ids_t.max() + 1, ...]
        batch_feat_v = batch_feat_v[:ids_t.max() + 1, ...]
        batch_cls = batch_cls[:ids_t.max() + 1, ...]
        
        batch_cls_visual_c = batch_cls_visual_c[:ids_t.max() + 1, ...]
        batch_cls_textual_c = batch_cls_textual_c[:ids_t.max() + 1, ...]
        batch_visual_c_mask = batch_visual_c_mask[:ids_t.max() + 1, ...]
        batch_textual_c_mask = batch_textual_c_mask[:ids_t.max() + 1, ...]
        logger.info('[finish] extract text+video feature')

    toc1 = time.time()

    logger.info('{} {} {} {}'.format(len(batch_mask_t), len(batch_mask_v), len(batch_feat_t), len(batch_feat_v)))
    # ----------------------------------
    # 2. calculate the similarity
    # ----------------------------------
    logger.info('[start] calculate the similarity')
    with torch.no_grad():
        sim_matrix0, t2v_sim_matrix, v2t_sim_matrix = _run_on_single_gpu(args, model, batch_mask_t, batch_mask_v, batch_feat_t, batch_feat_v, batch_cls,batch_visual_c_mask, batch_textual_c_mask, batch_cls_textual_c, batch_cls_visual_c, batch_mask_train_t, batch_mask_train_v, batch_feat_train_t, batch_feat_train_v,batch_train_cls,batch_train_cls_visual,batch_train_cls_textual)
    logger.info('[end] calculate the similarity')

    toc2 = time.time()
    logger.info('[start] compute_metrics')
    
    logger.info("sim matrix size: {}, {}".format(sim_matrix0.shape[0], sim_matrix0.shape[1]))
    
    #with open('./qualitative/sim.json', 'w') as fp:
    #    json.dump(t2v_sim_matrix.tolist(), fp)  
    # no inference strategy
    tv_metric_basic=compute_metrics(sim_matrix0)
    vt_metric_basic=compute_metrics(sim_matrix0.T)

    # +DSL
    tv_matrix_dsl = torch.from_numpy(sim_matrix0).cuda()
    tv_matrix_dsl = tv_matrix_dsl * F.softmax(tv_matrix_dsl / args.t2v_temp, dim=0) * len(tv_matrix_dsl)
    tv_matrix_dsl = tv_matrix_dsl.cpu().numpy()
    tv_metrics_dsl = compute_metrics(tv_matrix_dsl)

    vt_matrix_dsl = torch.from_numpy(sim_matrix0).cuda()
    vt_matrix_dsl = vt_matrix_dsl * F.softmax(vt_matrix_dsl / args.v2t_temp, dim=1) * len(vt_matrix_dsl)
    vt_matrix_dsl = vt_matrix_dsl.cpu().numpy()
    vt_metrics_dsl = compute_metrics(vt_matrix_dsl.T)

    # +QB
    tv_metrics = compute_metrics(t2v_sim_matrix)
    vt_metrics = compute_metrics(v2t_sim_matrix)

    logger.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix0), len(sim_matrix0[0])))

    logger.info('[end] compute_metrics')

    toc3 = time.time()
    logger.info("time profile: feat {:.1f}s match {:.5f}s metrics {:.5f}s".format(toc1 - tic, toc2 - toc1, toc3 - toc2))

    logger.info("Text-to-Video (Basic): R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@50: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}".
                format(tv_metric_basic['R1'], tv_metric_basic['R5'], tv_metric_basic['R10'], tv_metric_basic['R50'], tv_metric_basic['MR'], tv_metric_basic['MeanR']))
    logger.info("Video-to-Text (Basic): R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@50: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}".
                format(vt_metric_basic['R1'], vt_metric_basic['R5'], vt_metric_basic['R10'], vt_metric_basic['R50'], vt_metric_basic['MR'], vt_metric_basic['MeanR']))

    logger.info("Text-to-Video (DSL): R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@50: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}".
                format(tv_metrics_dsl['R1'], tv_metrics_dsl['R5'], tv_metrics_dsl['R10'], tv_metrics_dsl['R50'], tv_metrics_dsl['MR'], tv_metrics_dsl['MeanR']))
    logger.info("Video-to-Text (DSL): R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@50: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}".
                format(vt_metrics_dsl['R1'], vt_metrics_dsl['R5'], vt_metrics_dsl['R10'], vt_metrics_dsl['R50'], vt_metrics_dsl['MR'], vt_metrics_dsl['MeanR']))

    logger.info("Text-to-Video (QB): R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@50: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}".
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['R50'], tv_metrics['MR'], tv_metrics['MeanR']))
    logger.info("Video-to-Text (QB): R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@50: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}".
                format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['R50'], vt_metrics['MR'], vt_metrics['MeanR']))

    return tv_metrics['R1']


def main():
    global logger
    global best_score
    global meters
    global train_dataloader0
    global train_sampler0

    meters = MetricLogger(delimiter="  ")
    args = get_args()
    if not exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger('tvr', args.output_dir, args.local_rank)

    args = set_seed_logger(args)

    model = build_model(args)
    model=model.cuda()

    test_dataloader, val_dataloader, train_dataloader, train_sampler, train_dataloader0, train_sampler0 = build_dataloader(args)
    ## ####################################
    # train and eval
    ## ####################################
    if args.do_train:
        tic = time.time()
        max_steps = len(train_dataloader) * args.epochs
        _max_steps = len(train_dataloader) * 5
        optimizer, scheduler, model = prep_optimizer(args, model, _max_steps, args.local_rank)

        best_score = 0.00001
        best_output_model_file = "None"
        global_step = 0
        for epoch in range(args.epochs):
            if train_sampler is not None: train_sampler.set_epoch(epoch)
            synchronize()
            torch.cuda.empty_cache()
            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader,
                                               args.device, args.world_size, optimizer,
                                               scheduler, global_step, max_steps, val_dataloader)
            torch.cuda.empty_cache()
            R1 = eval_epoch(args, model, val_dataloader, args.device)
            torch.cuda.empty_cache()
            synchronize()

            if args.local_rank == 0:
                output_model_file = save_model(epoch, args, model, type_name="")

                if best_score <= R1:
                    best_score = R1
                    best_output_model_file = output_model_file
                    torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                               'best.pth')
                logger.info("The best model is: {}, the R1 is: {:.4f}".format(best_output_model_file, best_score))
            synchronize()
        toc = time.time() - tic
        training_time = time.strftime("%Hh %Mmin %Ss", time.gmtime(toc))
        logger.info("*" * 20 + '\n' + f'training finished with {training_time}' + "*" * 20 + '\n')

        # test on the best checkpoint
        model = model.module
        if args.local_rank == 0:
            model.load_state_dict(torch.load('best.pth', map_location='cpu'), strict=False)
        if torch.cuda.is_available():
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                              find_unused_parameters=True)

        torch.cuda.empty_cache()
        eval_epoch(args, model, test_dataloader, args.device)
        synchronize()

    elif args.do_eval:
        eval_epoch(args, model, test_dataloader, args.device)


if __name__ == "__main__":
    main()
