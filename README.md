<h1 align="center">
   KV-Ground: Small GUI Grounding Models for High-Resolution Images
</h1>


<p align="center">
  <a href="https://huggingface.co/vocaela/KV-Ground-8B-BaseGuiOwl1.5"><img src="https://img.shields.io/badge/Hugging%20Face-8B--BaseGuiOwl1.5-orange?logo=huggingface" alt="KV-Ground-8B-BaseGuiOwl1.5"></a>
  <a href="https://huggingface.co/vocaela/KV-Ground-4B-BaseGuiOwl1.5-0228"><img src="https://img.shields.io/badge/Hugging%20Face-4B--BaseGuiOwl1.5--0228-orange?logo=huggingface" alt="KV-Ground-4B-BaseGuiOwl1.5-0228"></a>
  <a href="https://huggingface.co/vocaela/KV-Ground-4B-BaseQw3vl"><img src="https://img.shields.io/badge/Hugging%20Face-4B--BaseQw3vl-orange?logo=huggingface" alt="KV-Ground-4B-BaseQw3vl"></a>
  <a href="https://github.com/vocaela/kv-ground"><img src="https://img.shields.io/badge/GitHub-Repository-green?logo=github" alt="GitHub"></a>
</p>
<p align="center">
  <em>
    By
    <a href="https://www.kingsware.cn/">Kingsware</a>
    &
    <a href="https://vocaela.ai/">Vocaela AI</a>
  </em>
</p>


High-resolution GUI grounding is an extreme important task for professional workforce scenarios like desktop RPA, where the massive usage also implies the favoring of compact models. High-resolution GUI grounding remains challenging to VLMs. The recent progress on ScreenSpot-Pro are mostly from reasoning CoT and agentic framework on top of models, which trade-in significant extra computing time / integration complexity for the quality boost. For pure instruct model, the improvement highly correlates to model size scaling up. However, we argue that grounding as a very fundamental atomic capability needs intrinsic improvement. High-resolution grounding is challenging to VLMs largely due to two reasons: (1) Vision-encoders were pre-trained on images of very low resolution. (2) Lack of high-quality high-resolution annotated data. We belive that even the top-tier VLMs are under-trained on high-resolution grounding data. To address this problem, we synthesize high-quality high-resolution GUI grounding data, and continue post-training SOTA VLMs with SFT followed by RFT (GRPO). The approach is verified on several models with promising results. Without reasoning CoT, on high-resolution grounding benchmark [ScreenSpot-Pro](https://gui-agent.github.io/grounding-leaderboard/):
- [KV-Ground-8B-BaseGuiOwl1.5](https://huggingface.co/vocaela/KV-Ground-8B-BaseGuiOwl1.5) achieves 72.0, No.1 under 8B, No.2 across all the models.
- [KV-Ground-4B-BaseGuiOwl1.5-0228](https://huggingface.co/vocaela/KV-Ground-4B-BaseGuiOwl1.5-0228) achieves 67.0, No.1 under 4B, No.4 under 8B, No.8 across all models.
- [KV-Ground-4B-BaseQw3vl](https://huggingface.co/vocaela/KV-Ground-4B-BaseQw3vl) achieves 63.2, No.3 under 4B.
While optimized for high-resolution images, they maintains excellent performance on regular-resolution tasks as well, with 93.8 / 94.1 / 94.6 on ScreenSpot-V2.

---


## Benchmark Results

- Impact of continue post-training on base models

  For the purpose of controlled comparison, all these numbers are re-/produced by us, using the exact evaluation code in this repo. The baseline numbers may be different from the sources. Please see below `note` section for the controlled setup.

  - Model only

    | Models                               | ScreenSpot-Pro | ScreenSpot-V2 | OSWorld-G    | OSWorld-G-refined | UI-Vision   |
    |--------------------------------------|:--------------:|:-------------:|:------------:|:-----------------:|:-----------:|
    | Base: Qwen3-VL-4B-Instruct*          | 59.5           | 93.1          | 63.3         | 71.1              | 30.4        |
    | **KV-Ground-4B-BaseQw3vl***          | 63.2 (+3.7)    | 94.6 (+1.5)   | 64.0 (0.7)   | 71.2 (+0.1)       | 32.6 (+2.2) |
    | Base: GUI-Owl-1.5-4B-Instruct*       | 65.3           | 92.8          | 61.7         | 66.8              | 30.4        |
    | **KV-Ground-4B-BaseGuiOwl1.5-0228*** | 67.0 (+1.7)    | 94.1 (+1.3)   | 64.2 (+2.5)  | 69.5 (+2.7)       | 33.3 (+2.9) |
    | Base: GUI-Owl-1.5-8B-Instruct*       | 70.5           | 93.5          | 64.7         | 67.9              | 38.3        |
    | **KV-Ground-8B-BaseGuiOwl1.5***      | 72.0 (+1.5)    | 93.8 (+0.3)   | 67.7 (+3.0)  | 71.1 (+3.2)       | 39.0 (+0.7) |

    The results tell us:
    - Our continual post-training method brings in consistent improvements, even on the latest strong GUI-Owl-1.5-4B-Instruct model.
    - The high-resolution optimized training doesn't harm regular resolution tasks
  
    Especially, it is beyond our expectation on the improvement over GUI-Owl-1.5-4B-Instruct because the technical report of GUI-Owl-1.5-4B-Instruct discloses that they have already applied extensive data synthesize / augementation targeted for high-resolution images.

  - With zoom-in strategy on ScreenSpot-Pro
    
    We use the same Zoom-In strategy described in the technical reports of MAI-UI and GUI-Owl-1.5-4B-Instruct/GUI-Owl-1.5-8B-Instruct. The overhead is to do two rounds of inference.
  
    | Model + Zoom-In                                | ScreenSpot-Pro | 
    |------------------------------------------------|:--------------:|
    | Base: Qwen3-VL-4B-Instruct + Zoom-In*          | 69.4           |
    | **KV-Ground-4B-BaseQw3vl** + Zoom-In*          | 70.3 (+0.9)    |
    | Base: GUI-Owl-1.5-4B-Instruct + Zoom-In*       | 76.1           |
    | **KV-Ground-4B-BaseGuiOwl1.5-0228** + Zoom-In* | 76.4 (+0.3)    |
    | Base: GUI-Owl-1.5-8B-Instruct + Zoom-In*       | 78.0           |
    | **KV-Ground-8B-BaseGuiOwl1.5** + Zoom-In*      | 78.6 (+0.6)    |
    
    When applied with the Zoom-In, the relative gain is still visible although marginal. It is expected as zoom-in can be considered a rough upper-bound for a given model. Fine-grained modeling improvement optimizing high-resolution is trying to approach to the upper-bound by one-time inference.

- Comparision with top models (ranked by ScreenSpot-Pro)
  
  We consider all top models from [ScreenSpot-Pro leaderboard](https://gui-agent.github.io/grounding-leaderboard/) and related most recent technical reports. We only compare pure model capabillity and hence exclude those multi-step methods such as using zoom-in, MVP or agentic frameworks.

  On ScreenSpot-Pro:
  - KV-Ground-8B-BaseGuiOwl1.5: No.1 under 8B, No.2 among all the models, and
  - KV-Ground-4B-BaseGuiOwl1.5-0228: No.1 under 4B, No.4 under 8B, and No.8 among all the models
  - KV-Ground-4B-BaseQw3vl: No.3 under 4B

  | Models                               | ScreenSpot-Pro | ScreenSpot-V2 | OSWorld-G | OSWorld-G-refined | UI-Vision  |
  |--------------------------------------|:--------------:|:-------------:|:---------:|:-----------------:|:----------:|
  | *Specialized GUI Models*
  | GUI-Owl-1.5-32B-Instruct             | 72.9      | 95.3  | 66.8   | 69.7  | -     |
  | **KV-Ground-8B-BaseGuiOwl1.5***      | 72.0      | 93.8  | 67.7   | 71.1  | 39.0  |
  | Holo2-235B-A22B                      | 70.6      | 95.9  | 79.0   | -     | -     |
  | GUI-Owl-1.5-8B-Instruct*             | 70.5      | 93.5  | 64.7   | 67.9  | 38.3  |
  | UI-Venus-1.5-30B-A3B                 | 69.6      | 96.2  | 70.6   | 76.4  | 54.7  |
  | UI-Venus-1.5-8B                      | 68.4      | 95.9  | 69.7   | 74.1  | 46.5  |
  | MAI-UI-32B                           | 67.9      | 96.5  | 67.6   | 73.9  | 47.1  |
  | **KV-Ground-4B-BaseGuiOwl1.5-0228*** | 67.0      | 94.1  | 64.2   | 69.5  | 33.3  |
  | Holo2-30B-A3B                        | 66.1      | 94.9  | 76.1   | -     | -     |
  | MAI-UI-8B                            | 65.8      | 95.2  | 60.1   | 68.6  | 40.7  |
  | GUI-Owl-1.5-4B-Instruct*             | 65.3      | 92.8  | 61.7   | 66.8  | 30.4  |
  | **KV-Ground-4B-BaseQw3vl***          | 63.2      | 94.6  | 64.0   | 71.2  | 32.6  |
  | Step-GUI-8B                          | 62.6      | 95.1  | 70.0   | -     | -     |
  | Step-GUI-4B                          | 60.0      | 93.6  | 66.9   | -     | -     |
  | Holo2-8B                             | 58.9      | 93.2  | 70.1   | -     | -     |
  | Holo2-4B                             | 57.2      | 93.2  | 69.4   | -     | -     |
  | GUI-Owl-7B                           | 54.9      | 92.8  | 55.9   | -     | -     |
  | OpenCUA-7B                           | 50.0      | 92.3  | 55.3   | -     | 29.7  |
  | UI-Venus-1.0-7B                      | 50.8      | 94.1  | 54.6   | 61.7  | 36.8  |
  | GTA1-7B                              | 50.1      | 92.4  | 60.1   | 67.7  | -     |
  | UI-TARS-1.5-7B                       | 35.7      | 91.6  | 52.8   | 64.2  | -     |
  | *General VLMs*
  | Qwen3-VL-4B*                         | 59.5      | 93.1  | 63.3   | 71.1  | 30.4  |
  | Qwen3-VL-8B                          | 54.6      | -     | 58.2   | -     | -     | 
  

- Comparision with agentic approaches on ScreenSpot-Pro

  We list out top 10 players reported on ScreenSpot-Pro leaderboard and related technical reports:
  - KV-Ground-8B-BaseGuiOwl1.5 + Zoom-In: No.1 in systems w/ model under 8B, No.2 across all players
  - KV-Ground-4B-BaseGuiOwl1.5-0228 + Zoom-In: No.1 in systems w/ model under 4B, No.6 across all players

    | Model / Agentic                                | ScreenSpot-Pro | 
    |------------------------------------------------|:--------------:|
    | GUI-Owl-1.5-32B-Instruct + Zoom-In             | 80.3           |
    | **KV-Ground-8B-BaseGuiOwl1.5** + Zoom-In*      | 78.6           |
    | Holo2-235B-A22B (Agentic)                      | 78.5           |
    | GUI-Owl-1.5-8B-Instruct + Zoom-In*             | 78.0           |
    | MAI-UI-32B (MVP)                               | 77.5           |
    | **KV-Ground-4B-BaseGuiOwl1.5-0228** + Zoom-In* | 76.4           |
    | GUI-Owl-1.5-4B-Instruct + Zoom-In*             | 76.1           |
    | Holo2-30B-A3B (Agentic)                        | 75.2           |
    | MVP_Qwen3VL-32B                                | 74.1           |
    | MAI-UI-32B (Zoom In)                           | 73.5           |
    
---

> Notes:
>  - By default numbers are from each source. Those numbers produced by us are indicated by `*`, which may be different from the numbers reported from the sources.
>  - It is known that models are usually sensitive to certain pre-designed prompt. As all the runs produced by us are based on Qwen3-VL backbone, for fair and simple comparision, the same prompt structure of `system -> user-image -> user-instruct` is used. And the same [system message](./utils.py) is used, which is the default Qwen3-VL computer-use format prompt adopted by the [ScreenSpot-Pro leaderboard](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding/blob/211ff96b54849f80cca20e02ef03233d3f0e6fdf/models/qwen3vl.py#L20). For OSWorld-G and OSWorld-G-refined, minor modification is made to instruct the refusal setting.


## Run Evaluation

The scripts are tested on cuda12.8.

```bash
pip install -r requirements.txt
```

Edit `run_eval.sh` to provide the paths for model, data, and output, and run

```bash
bash run_eval.sh
```

---

# License

This code repo is under MIT license.

The model on Huggingface is under `CC-BY-NC-SA 4.0`.