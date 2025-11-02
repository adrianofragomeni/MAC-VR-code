from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import json
import tempfile
import pandas as pd
from os.path import join, splitext, exists
from collections import OrderedDict
from .dataloader_retrieval_gif import RetrievalDataset
import numpy as np
import os
from PIL import Image


class TGIFDataset(RetrievalDataset):
    """MSRVTT dataset."""

    def __init__(self, subset, anno_path, video_path, tokenizer, max_words=32,
                 max_frames=12, video_framerate=1, image_resolution=224, mode='all', config=None):
        super(TGIFDataset, self).__init__(subset, anno_path, video_path, tokenizer, max_words,
                                            max_frames, video_framerate, image_resolution, mode, config=config)
        pass

    def _get_anns(self, subset='train'):
        """
        video_dict: dict: video_id -> video_path
        sentences_dict: list: [(video_id, caption)] , caption (list: [text:, start, end])
        """

        csv_path = {'train': join(self.anno_path, 'train.txt'),
                    'val': join(self.anno_path, 'test.txt'),
                    'test': join(self.anno_path, 'test.txt'),
                    'train_test': join(self.anno_path, 'val.txt')}[subset]
        if exists(csv_path):
            csv = pd.read_csv(csv_path,header=None)
            annotations_csv=pd.read_csv(join(self.anno_path, 'annotations.csv'))
            vid_ids=[i.split("/")[-1] for i in annotations_csv.video_id]
            annotations_dict=dict(zip(vid_ids,annotations_csv.caption))
            annotations_visual_dict=dict(zip(vid_ids,annotations_csv.visual_concepts))
            annotations_textual_dict=dict(zip(vid_ids,annotations_csv.textual_concepts))
        else:
            raise FileNotFoundError

        video_id_list = os.listdir(self.video_path)

        video_dict = OrderedDict()
        sentences_dict = OrderedDict()
        visual_concept_dict = OrderedDict()
        textual_concept_dict = OrderedDict()

        if subset == 'train':
            for row in csv.values:
                v=row[0].split("/")[-1]
                if v in video_id_list:
                    sentences_dict[len(sentences_dict)] = (v, (annotations_dict[v], None, None))               
                    visual_concept_dict[len(visual_concept_dict)] = (v, [annotations_visual_dict[v], None, None])
                    textual_concept_dict[len(textual_concept_dict)] = (v, [annotations_textual_dict[v], None, None])
                    video_dict[v] = join(self.video_path, "{}".format(v))
        elif subset == 'train_test':
            used = []
            for row in csv.values:
                v=row[0].split("/")[-1]
                if v in video_id_list and v not in used:

                    used.append(v)
                    sentences_dict[len(sentences_dict)] = (v, (annotations_dict[v], None, None))
                    visual_concept_dict[len(visual_concept_dict)] = (v, [annotations_visual_dict[v], None, None])
                    textual_concept_dict[len(textual_concept_dict)] = (v, [annotations_textual_dict[v], None, None])
                    video_dict[v] = join(self.video_path, "{}".format(v))
        else:
            for row in csv.values:
                v=row[0].split("/")[-1]

                if v in video_id_list:

                    sentences_dict[len(sentences_dict)] = (v, (annotations_dict[v], None, None))
                    visual_concept_dict[len(visual_concept_dict)] = (v, [annotations_visual_dict[v], None, None])
                    textual_concept_dict[len(textual_concept_dict)] = (v, [annotations_textual_dict[v], None, None])
                    video_dict[v] = join(self.video_path, "{}".format(v))

        unique_sentence = set([v[1][0] for v in sentences_dict.values()])
        print('[{}] Unique sentence is {} , all num is {}'.format(subset, len(unique_sentence), len(sentences_dict)))

        return video_dict, sentences_dict, textual_concept_dict, visual_concept_dict
