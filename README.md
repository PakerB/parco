# PARCO

[![arXiv](https://img.shields.io/badge/arXiv-2409.03811-b31b1b.svg)](https://arxiv.org/abs/2409.03811) [![Slack](https://img.shields.io/badge/slack-chat-611f69.svg?logo=slack)](https://join.slack.com/t/ai4co-community/shared_invite/zt-3jsdjs3ec-3KHdV3HwanL884mq_9tyYw)
[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace Dataset](https://img.shields.io/badge/%F0%9F%A4%97-Dataset-yellow)](https://huggingface.co/ai4co/parco)
[![HuggingFace Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow)](https://huggingface.co/datasets/ai4co/parco)
[![Slideslive](https://img.shields.io/badge/SlidesLive-Video-da0f30.svg)](https://recorder-v3.slideslive.com/?share=106087&s=34a9a21f-3f2a-4a4d-9979-98fe9e9d7f33)
[![Google Slides](https://img.shields.io/badge/Google-Slides-f3b421.svg)](https://docs.google.com/presentation/d/18cM_0-PNgTRatMlrFF9AM9ngMohtqNo85BmN5BoLn14/edit?usp=sharing)


<div align="center">
<i> PARCO has been accepted at NeurIPS 2025! 🥳</i>
</div>

<br>

Code repository for "PARCO: Parallel AutoRegressive Models for Multi-Agent Combinatorial Optimization"

<div align="center">
    <img src="assets/ar-vs-par.png" style="width: 100%; height: auto;">
    <i> Autoregressive policy (AR) and Parallel Autoregressive (PAR) decoding </i>
</div>

<br>

<div align="center">
    <img src="assets/parco-model.png" style="width: 100%; height: auto;">
    <i> PARCO Model</i>
</div>

## 🚀 Usage

### Installation

We use [uv](https://docs.astral.sh/uv/getting-started/installation/) for fast installation and dependency management:

```bash
uv venv
source .venv/bin/activate
uv sync --all-extras
```

To download the data and checkpoints from HuggingFace automatically, you can use:

```bash
python scripts/download_hf.py
```

### Quickstart Notebooks

We made examples for each problem that can be trained under two minutes on consumer hardware. You can find them in the `examples/` folder:

- [1.quickstart-hcvrp.ipynb](examples/1.quickstart-hcvrp.ipynb): HCVRP (Heterogeneous Capacitated Vehicle Routing Problem)
- [2.quickstart-omdcpdp.ipynb](examples/2.quickstart-omdcpdp.ipynb): OMDCPDP (Open Multi-Depot Capacitated Pickup and Delivery Problem)
- [3.quickstart-ffsp.ipynb](examples/3.quickstart-ffsp.ipynb): FFSP (Flexible Flow Shop Scheduling Problem)

### Train your own model

You can train your own model using the `train.py` script. For example, to train a model for the HCVRP problem, you can run:

```bash
python train.py experiment=hcvrp
```

you can change the `experiment` parameter to `omdcpdp` or `ffsp` to train the model for the OMDCPDP or FFSP problem, respectively.

Note on legacy FFSP code: the initial version we made was not yet integrated in RL4CO, so we left it the [`parco/tasks/ffsp_old`](parco/tasks/ffsp_old/README.md) folder, so you can still use it.

### Testing

You may run the `test.py` script to evaluate the model, e.g. with greedy decoding:

```bash
python test.py --problem hcvrp --decode_type greedy --batch_size 128
```

(note: we measure time with single instance -- batch size 1, but larger makes the overall evaluation faster), or with sampling:

```bash
python test.py --problem hcvrp --decode_type sampling --batch_size 1 --sample_size 1280
```

### Other scripts

- Data generation: We also include scripts to re-generate data manually (reproducible via random seeds) with `python scripts/generate_data.py`.
- OR-Tools: We additionally include a script to solve the problem using OR-Tools with `python scripts/run_ortools.py`.

## 🤩 Citation

If you find PARCO valuable for your research or applied projects:

```bibtex
@inproceedings{berto2025parco,
    title={{PARCO: Parallel AutoRegressive Models for Multi-Agent Combinatorial Optimization}}, 
    author={Federico Berto and Chuanbo Hua and Laurin Luttmann and Jiwoo Son and Junyoung Park and Kyuree Ahn and Changhyun Kwon and Lin Xie and Jinkyoo Park},
    booktitle={Advances in Neural Information Processing Systems},
    year={2025},
    url={https://github.com/ai4co/parco}
}
```

We will also be happy if you cite the RL4CO framework that we used to create PARCO:

```bibtex
@inproceedings{berto2025rl4co,
    title={{RL4CO: an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark}},
    author={Federico Berto and Chuanbo Hua and Junyoung Park and Laurin Luttmann and Yining Ma and Fanchen Bu and Jiarui Wang and Haoran Ye and Minsu Kim and Sanghyeok Choi and Nayeli Gast Zepeda and Andr\'e Hottung and Jianan Zhou and Jieyi Bi and Yu Hu and Fei Liu and Hyeonah Kim and Jiwoo Son and Haeyeon Kim and Davide Angioni and Wouter Kool and Zhiguang Cao and Jie Zhang and Kijung Shin and Cathy Wu and Sungsoo Ahn and Guojie Song and Changhyun Kwon and Lin Xie and Jinkyoo Park},
    booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
    year={2025},
    url={https://github.com/ai4co/rl4co}
}
```

---

<div align="center">
    <a href="https://github.com/ai4co">
        <img src="https://raw.githubusercontent.com/ai4co/assets/main/svg/ai4co_animated_full.svg" alt="AI4CO Logo" style="width: 30%; height: auto;">
    </a>
</div>
