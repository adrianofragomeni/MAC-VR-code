# MAC-VR
This repo contains code of MAC-VR model used in the BMVC paper Leveraging Modality Tags for Enhanced Cross-Modal Video Retrieval.

## Quick Start Guide
Our code is based on the original code [DiCoSA](https://github.com/jpthu17/DiCoSA). Follow the ```Setup code enviroment```, ```Download CLIP Model``` and ```Compress Video``` in the DiCoSA repository. All the dependencies can be found in `MacVR_env.yml`

### Tag Extraction
To extract tags from a video, we used the original [VideoLLama2](https://github.com/DAMO-NLP-SG/VideoLLaMA2) with the following prompt:
```
A general tag of an action is a fundamental and overarching idea that encapsulates the essential principles, commonalities, or recurrent patterns within a specific behaviour or activity, providing a higher-level understanding of the underlying themes and purpose associated with that action. What are the top 10 general tags that capture the fundamental idea of this action? Give me a bullet list as output where each point is a general tag, and use one or two significant words per tag and do not give any explanation.
```
To extract tags from a caption, we used the original [Llama3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) with the following prompt:
```
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
USER:
You are a conversational AI agent. You typically extract the general tags of an action.

A general tag of an action is a fundamental and overarching idea that encapsulates the essential principles, commonalities, or recurrent patterns within a specific behaviour or activity, providing a higher-level understanding of the underlying themes and purpose associated with that action.

Given the following action: 
1) {}

What are the top 10 general tags of the above action? Use one or two significant words per tag and do not give any explanation.

ASSISTANT: 
```
where {} is the corresponding caption.

We use different temperature values (i.e., 0.7, 0.8, 0.9, 1.0) to extract tags from a video and its corresponding caption. We use the whole video and the corresponding paragraph for DiDeMo dataset.
After extraction we clean the tags using the code in ```cleaning_tags.py```.

### Data

Add the videos of a dataset in the corresponding folder ```./data/name_dataset/videos```.
You can find the updated annotations with the extracted tags in ```./data/name_dataset/anns```.

### Train

To train the model you can find the commnad line in the corresponding sh file: ```train_multigpu_name_dataset.sh```.

### Test

To test the model you can find the commnad line in the corresponding sh file: ```test_multigpu_name_dataset.sh```.
The checkpoints are stored in the folder ```weights```.


## Citation
```
@inproceedings{fragomeni2025BMVC,
  author       = {Fragomeni, Adriano and Damen, Dima and Wray, Michael},
  title        = {Leveraging Modality Tags for Enhanced Cross-Modal Video Retrieval},
  booktitle    = {British Machine Vision Conference (BMVC)},
  year         = {2025}
}

@inproceedings{ijcai2023p0104,
  title     = {Text-Video Retrieval with Disentangled Conceptualization and Set-to-Set Alignment},
  author    = {Jin, Peng and Li, Hao and Cheng, Zesen and Huang, Jinfa and Wang, Zhennan and Yuan, Li and Liu, Chang and Chen, Jie},
  booktitle = {Proceedings of the Thirty-Second International Joint Conference on
               Artificial Intelligence, {IJCAI-23}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Edith Elkind},
  pages     = {938--946},
  year      = {2023},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2023/104},
  url       = {https://doi.org/10.24963/ijcai.2023/104},
}
```
