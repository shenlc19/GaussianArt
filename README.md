<div align="center">

# **GaussianArt**: Unified Modeling of Geometry and Motion for Articulated Objects (3DV 2026)


<div align="center" margin-bottom="6em">
    <span class="author-block">
        <a href="https://shenlc19.github.io/" target="_blank">Licheng Shen✶</a><sup>1</sup>,</span>
    <span class="author-block">
        <a href="https://sainingzhang.github.io/" target="_blank">Saining Zhang ✶</a><sup>1,2</sup>,</span>
        <span class="author-block">
        Honghan Li ✶<sup>1,2</sup>,</span>
        <span class="author-block">
        Peilin Yang<sup>3</sup>,</span>
        <span class="author-block">
        <a href="https://zihao-inorganic.github.io/" target="_blank">Zihao Huang</a><sup>4</sup>,</span>
        <span class="author-block">
        Zongzheng Zhang<sup>1</sup>,</span>
        <span class="author-block">
        <a href="https://sites.google.com/view/fromandto" target="_blank">Hao Zhao†</a><sup>1,5</sup></span>
    <br>
    <p style="font-size: 0.9em; padding: 0.5em 0;">✶ indicates equal contribution 
    † corresponding author</p>
    <span class="author-block">
        <sup>1</sup>Institute for AI Industry Research(AIR), Tsinghua University &nbsp&nbsp 
        <sup>2</sup>Nanyang Technological University
        <sup>3</sup>Beijing Institute of Technology
        <sup>4</sup>Huazhong University of Science and Technology
        <sup>5</sup>Beijing Academy of Artificial Intelligence
    </span>

[Website](https://sainingzhang.github.io/project/gaussianart/) | [Arxiv](https://arxiv.org/abs/2508.14891) | [Data]()
</div>
</div>

![Alt text](assets/teaser.png)

## Environment Setup

Please follow these steps to setup the environment:

```bash
git clone https://github.com/shenlc19/GaussianArt --recursive
cd GaussianArt

# create and initialize conda environment

conda create -n gaussianart python=3.10
conda activate gaussianart

# install pytorch
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

# install dependencies: submodules
pip install -r requirements.txt
pip install submodules/simple-knn 
pip install submodules/art-diff-gaussian-rasterization

```

[Pytorch3d(v0.7.5)](https://github.com/facebookresearch/pytorch3d) is also required. Please clone the repository, checkout v0.7.5 and follow the [instructions](https://github.com/facebookresearch/pytorch3d?tab=readme-ov-file#installation) to install.

## Dataset Preparation

Download the dataset from [link](https://huggingface.co/datasets/LiCheng23/MPArt-90), save the compressed files to ```./data``` and decompress them. Each instance follows the structure below:

```bash
 ./data
    ├── model_id
    │   ├── start
    │   ├── end
    │   ├── gt
    │   ├── transforms_test_end.json
    │   ├── transforms_test_start.json
    │   ├── transforms_test.json
    │   ├── transforms_train_end.json
    │   ├── transforms_train_start.json
    │   ├── transforms_train.json
    └── ...

```

## Training

Use ```model_id``` as a parameter to run the following command:

```
python run.py --model_id {model_id}
```

The training process includes depth-semantic initialization and motion-appearance joint optimization. The results will be saved to ```output/MPArt-90/{model_id}```.

## Evaluation

Evaluate the motion axis prediction by running the following command: 

```
python eval_axis.py -m output/MPArt-90/{model_id}
```

## Render

Run the following command to render the reconstructed articulated object and visualize the part-level motion:

```bash
python render_video.py -m output/MPArt-90/{model_id}
```

## Citation

If you find our paper and/or code helpful, please consider citing:

```
@misc{shen2025gaussianartunifiedmodelinggeometry,
      title={GaussianArt: Unified Modeling of Geometry and Motion for Articulated Objects}, 
      author={Licheng Shen and Saining Zhang and Honghan Li and Peilin Yang and Zihao Huang and Zongzheng Zhang and Hao Zhao},
      year={2025},
      eprint={2508.14891},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.14891}, 
}
```

## Acknowledgement

We acknowledge the authors of [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [ArtGS](https://github.com/YuLiu-LY/ArtGS), [DigitalTwinArt](https://github.com/NVlabs/DigitalTwinArt)  for making their outstanding projects publicly available.