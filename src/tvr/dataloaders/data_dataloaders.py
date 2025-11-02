import torch
from torch.utils.data import DataLoader
from .dataloader_msrvtt_retrieval import MSRVTTDataset
from .dataloader_activitynet_retrieval import ActivityNetDataset
from .dataloader_didemo_retrieval import DiDeMoDataset
from .dataloader_lsmdc_retrieval import LsmdcDataset
from .dataloader_yc2_retrieval import YouCookDataset
from .dataloader_charades_retrieval import CharadesDataset
from .dataloader_tgif_retrieval import TGIFDataset



def dataloader_tgif_train(args, tokenizer):
    tgif_dataset = TGIFDataset(
        subset='train',
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )
    try:
        train_sampler = torch.utils.data.distributed.DistributedSampler(tgif_dataset)
    except:
        train_sampler = None  # cpu
    dataloader = DataLoader(
        tgif_dataset,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(tgif_dataset), train_sampler

def dataloader_tgif_test(args, tokenizer, subset="test"):
    tgif_testset = TGIFDataset(
        subset=subset,
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )

    try:
        test_sampler = torch.utils.data.distributed.DistributedSampler(tgif_testset)
    except:
        test_sampler = None  # cpu
    dataloader_tgif = DataLoader(
        tgif_testset,
        batch_size=args.batch_size_val // args.world_size,
        num_workers=args.workers,
        shuffle=False,
        sampler=test_sampler,
        drop_last=False,
    )
    return dataloader_tgif, len(tgif_testset)

def dataloader_tgif_train_test(args, tokenizer):
    tgif_dataset = TGIFDataset(
        subset='train_test',
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )
    try:
        train_sampler = torch.utils.data.distributed.DistributedSampler(tgif_dataset)
    except:
        train_sampler = None  # cpu
    dataloader = DataLoader(
        tgif_dataset,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(tgif_dataset), train_sampler


def dataloader_charades_train(args, tokenizer):
    charades_dataset = CharadesDataset(
        subset='train',
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )
    try:
        train_sampler = torch.utils.data.distributed.DistributedSampler(charades_dataset)
    except:
        train_sampler = None  # cpu
    dataloader = DataLoader(
        charades_dataset,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(charades_dataset), train_sampler

def dataloader_charades_test(args, tokenizer, subset="test"):
    charades_testset = CharadesDataset(
        subset=subset,
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )

    try:
        test_sampler = torch.utils.data.distributed.DistributedSampler(charades_testset)
    except:
        test_sampler = None  # cpu
    dataloader_charades = DataLoader(
        charades_testset,
        batch_size=args.batch_size_val // args.world_size,
        num_workers=args.workers,
        shuffle=False,
        sampler=test_sampler,
        drop_last=False,
    )
    return dataloader_charades, len(charades_testset)

def dataloader_charades_train_test(args, tokenizer):
    charades_dataset = CharadesDataset(
        subset='train_test',
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )
    try:
        train_sampler = torch.utils.data.distributed.DistributedSampler(charades_dataset)
    except:
        train_sampler = None  # cpu
    dataloader = DataLoader(
        charades_dataset,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(charades_dataset), train_sampler

def dataloader_yc2_train(args, tokenizer):
    yc_dataset = YouCookDataset(
        subset='train',
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )
    try:
        train_sampler = torch.utils.data.distributed.DistributedSampler(yc_dataset)
    except:
        train_sampler = None  # cpu
    dataloader = DataLoader(
        yc_dataset,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(yc_dataset), train_sampler

def dataloader_yc2_test(args, tokenizer, subset="test"):
    yc_testset = YouCookDataset(
        subset=subset,
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )

    try:
        test_sampler = torch.utils.data.distributed.DistributedSampler(yc_testset)
    except:
        test_sampler = None  # cpu
    dataloader_yc = DataLoader(
        yc_testset,
        batch_size=args.batch_size_val // args.world_size,
        num_workers=args.workers,
        shuffle=False,
        sampler=test_sampler,
        drop_last=False,
    )
    return dataloader_yc, len(yc_testset)

def dataloader_yc2_train_test(args, tokenizer):
    yc_dataset = YouCookDataset(
        subset='train_test',
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )
    try:
        train_sampler = torch.utils.data.distributed.DistributedSampler(yc_dataset)
    except:
        train_sampler = None  # cpu
    dataloader = DataLoader(
        yc_dataset,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(yc_dataset), train_sampler




def dataloader_msrvtt_train(args, tokenizer):
    msrvtt_dataset = MSRVTTDataset(
        subset='train',
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )
    try:
        train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    except:
        train_sampler = None  # cpu
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msrvtt_dataset), train_sampler


def dataloader_msrvtt_test(args, tokenizer, subset="test"):
    msrvtt_testset = MSRVTTDataset(
        subset=subset,
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )

    try:
        test_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_testset)
    except:
        test_sampler = None  # cpu
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        batch_size=args.batch_size_val // args.world_size,
        num_workers=args.workers,
        shuffle=False,
        sampler=test_sampler,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_testset)


def dataloader_msrvtt_train_test(args, tokenizer):
    msrvtt_dataset = MSRVTTDataset(
        subset='train_test',
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )
    try:
        train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    except:
        train_sampler = None  # cpu
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msrvtt_dataset), train_sampler


def dataloader_lsmdc_train(args, tokenizer):
    lsmdc_dataset = LsmdcDataset(
        subset='train',
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(lsmdc_dataset)
    dataloader = DataLoader(
        lsmdc_dataset,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(lsmdc_dataset), train_sampler


def dataloader_lsmdc_train_test(args, tokenizer):
    lsmdc_dataset = LsmdcDataset(
        subset='train_test',
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(lsmdc_dataset)
    dataloader = DataLoader(
        lsmdc_dataset,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(lsmdc_dataset), train_sampler


def dataloader_lsmdc_test(args, tokenizer, subset="test"):
    lsmdc_testset = LsmdcDataset(
        subset=subset,
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )
    try:
        test_sampler = torch.utils.data.distributed.DistributedSampler(lsmdc_testset)
    except:
        test_sampler = None  # cpu
    dataloader_lsmdc = DataLoader(
        lsmdc_testset,
        batch_size=args.batch_size_val // args.world_size,
        num_workers=args.workers,
        shuffle=False,
        sampler=test_sampler,
        drop_last=False,
    )
    return dataloader_lsmdc, len(lsmdc_testset)


def dataloader_activity_train(args, tokenizer):
    activity_dataset = ActivityNetDataset(
        subset="train",
        data_path=args.anno_path,
        features_path=args.video_path,
        max_words=args.max_words,
        feature_framerate=args.video_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(activity_dataset)
    dataloader = DataLoader(
        activity_dataset,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(activity_dataset), train_sampler


def dataloader_activity_train_test(args, tokenizer):
    activity_dataset = ActivityNetDataset(
        subset="train_test",
        data_path=args.anno_path,
        features_path=args.video_path,
        max_words=args.max_words,
        feature_framerate=args.video_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(activity_dataset)
    dataloader = DataLoader(
        activity_dataset,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(activity_dataset), train_sampler


def dataloader_activity_test(args, tokenizer, subset="test"):
    activity_testset = ActivityNetDataset(
        subset=subset,
        data_path=args.anno_path,
        features_path=args.video_path,
        max_words=args.max_words,
        feature_framerate=args.video_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames
    )
    try:
        test_sampler = torch.utils.data.distributed.DistributedSampler(activity_testset)
    except:
        test_sampler = None  # cpu
    dataloader_activity = DataLoader(
        activity_testset,
        batch_size=args.batch_size_val // args.world_size,
        num_workers=args.workers,
        shuffle=False,
        sampler=test_sampler,
        drop_last=False,
    )
    return dataloader_activity, len(activity_testset)


def dataloader_msvd_train(args, tokenizer):
    msvd_dataset = MsvdDataset(
        subset="train",
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msvd_dataset)
    dataloader = DataLoader(
        msvd_dataset,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msvd_dataset), train_sampler


def dataloader_msvd_train_test(args, tokenizer):
    msvd_dataset = MsvdDataset(
        subset="train_test",
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msvd_dataset)
    dataloader = DataLoader(
        msvd_dataset,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msvd_dataset), train_sampler


def dataloader_msvd_test(args, tokenizer, subset="test"):
    msvd_testset = MsvdDataset(
        subset=subset,
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )
    try:
        test_sampler = torch.utils.data.distributed.DistributedSampler(msvd_testset)
    except:
        test_sampler = None  # cpu
    dataloader_msvd = DataLoader(
        msvd_testset,
        batch_size=args.batch_size_val // args.world_size,
        num_workers=args.workers,
        shuffle=False,
        sampler=test_sampler,
        drop_last=False,
    )
    return dataloader_msvd, len(msvd_testset)


def dataloader_didemo_train(args, tokenizer):
    didemo_dataset = DiDeMoDataset(
        subset="train",
        data_path=args.anno_path,
        features_path=args.video_path,
        max_words=args.max_words,
        feature_framerate=args.video_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(didemo_dataset)
    dataloader = DataLoader(
        didemo_dataset,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(didemo_dataset), train_sampler


def dataloader_didemo_train_test(args, tokenizer):
    didemo_dataset = DiDeMoDataset(
        subset="train_test",
        data_path=args.anno_path,
        features_path=args.video_path,
        max_words=args.max_words,
        feature_framerate=args.video_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(didemo_dataset)
    dataloader = DataLoader(
        didemo_dataset,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(didemo_dataset), train_sampler


def dataloader_didemo_test(args, tokenizer, subset="test"):
    didemo_testset = DiDeMoDataset(
        subset=subset,
        data_path=args.anno_path,
        features_path=args.video_path,
        max_words=args.max_words,
        feature_framerate=args.video_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames
    )
    try:
        test_sampler = torch.utils.data.distributed.DistributedSampler(didemo_testset)
    except:
        test_sampler = None  # cpu
    dataloader_didemo = DataLoader(
        didemo_testset,
        batch_size=args.batch_size_val // args.world_size,
        num_workers=args.workers,
        shuffle=False,
        sampler=test_sampler,
        drop_last=False,
    )
    return dataloader_didemo, len(didemo_testset)


DATALOADER_DICT = {}
DATALOADER_DICT["msrvtt"] = {"train": dataloader_msrvtt_train,
                             "val": dataloader_msrvtt_test,
                             "test": None,
                             "train_test": dataloader_msrvtt_train_test}
DATALOADER_DICT["lsmdc"] = {"train": dataloader_lsmdc_train,
                            "val": dataloader_lsmdc_test,
                            "test": dataloader_lsmdc_test,
                            "train_test": dataloader_lsmdc_train_test}
DATALOADER_DICT["activity"] = {"train":dataloader_activity_train,
                               "val":dataloader_activity_test,
                               "test":None,
                               "train_test": dataloader_activity_train_test}
DATALOADER_DICT["didemo"] = {"train":dataloader_didemo_train,
                             "val":None,
                             "test":dataloader_didemo_test,
                             "train_test":dataloader_didemo_train_test}
DATALOADER_DICT["yc2"] = {"train": dataloader_yc2_train,
                             "val": dataloader_yc2_test,
                             "test": None,
                             "train_test": dataloader_yc2_train_test}
DATALOADER_DICT["charades"] = {"train": dataloader_charades_train,
                             "train_test":dataloader_charades_train_test,
                             "val": dataloader_charades_test,
                             "test": None}
DATALOADER_DICT["tgif"] = {"train": dataloader_tgif_train,
                             "train_test":dataloader_tgif_train_test,
                             "val": dataloader_tgif_test,
                             "test": dataloader_tgif_test}

