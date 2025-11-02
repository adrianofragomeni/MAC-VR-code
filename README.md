# MAC-VR
This repo contains code of MAC-VR model used in the BMVC paper Leveraging Modality Tags for Enhanced Cross-Modal Video Retrieval.

## Tag Extraction
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

We use different temperature values (i.e., 0.7, 0.8, 0.9, 1.0) to extract tags from a video and its corresponding caption.
After extraction we clean the tags using the code in ```cleaning_tags.py```

## Citation
```
@inproceedings{fragomeni2025BMVC,
  author       = {Fragomeni, Adriano and Damen, Dima and Wray, Michael},
  title        = {Leveraging Modality Tags for Enhanced Cross-Modal Video Retrieval},
  booktitle    = {British Machine Vision Conference (BMVC)},
  year         = {2025}
}
```
