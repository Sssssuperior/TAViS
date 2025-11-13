<div align="center">

# [ICCV 2025] TAViS: Text-bridged Audio-Visual Segmentation with Foundation Models

**Approach**: [[Conference Paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Luo_TAViS_Text-bridged_Audio-Visual_Segmentation_with_Foundation_Models_ICCV_2025_paper.pdf) |  [[arxiv Paper]](https://arxiv.org/pdf/2506.11436)

</div>

## Overview
Audio-Visual Segmentation (AVS) faces a fundamental challenge of effectively aligning audio and visual modalities.While recent approaches leverage foundation models to address data scarcity, they often rely on single-modality knowledge or combine foundation models in an off-the-shelf manner, failing to address the cross-modal alignment challenge. In this paper, we present TAViS, a novel framework that couples the knowledge of multimodal foundation models (ImageBind) for cross-modal alignment and a segmentation foundation model (SAM2) for precise segmentation. However, effectively combining these models poses two key challenges: the difficulty in transferring the knowledge between SAM2 and ImageBind due to their different feature spaces, and the insufficiency of using only segmentation loss for supervision. To address these challenges, we introduce a text-bridged design with two key components: (1) a text bridged hybrid prompting mechanism where pseudo text pro vides class prototype information while retaining modality specific details from both audio and visual inputs, and (2) an alignment supervision strategy that leverages text as a bridge to align shared semantic concepts within audio-visual modalities. Our approach achieves superior performance on single-source, multi-source, semantic datasets, and excels in zero-shot settings. 

<img src="https://github.com/Sssssuperior/TAViS/blob/main/backbone_00.png">

## Environmental Setups
```
pip install -r requirements.txt
```

## Start Training
For training, please run the following scripts and change the visible device according to yourself.
The subset can change to [s4, ms3, v2] for different settings.
```
CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_port=25638 train_avs.py --subset "s4"  --name "s4" --config configs/sam_avs_adapter.yaml --pretrained_weights "./model_epoch_best.pth"
```
For testing, 
```
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_port=25628 test_avs.py --subset "ms3"  --name "ms3" --eval "./ckpts/model_epoch_last.pth"
```

## Citation
If you use VSCode in your research or wish to refer to the baseline results published in the Model Zoo, please use the following BibTeX entry.
If you have any questions, please contact me: ziayangluo1110@gmail.com
```
@InProceedings{Luo_2025_ICCV,
    author    = {Luo, Ziyang and Liu, Nian and Yang, Xuguang and Khan, Salman and Anwer, Rao Muhammad and Cholakkal, Hisham and Khan, Fahad Shahbaz and Han, Junwei},
    title     = {TAViS: Text-bridged Audio-Visual Segmentation with Foundation Models},
    booktitle = {ICCV},
    month     = {October},
    year      = {2025},
    pages     = {24014-24023}
}
```
