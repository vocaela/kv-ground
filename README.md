<h1 align="center">
   KV-Ground-4B: Small GUI Grounding Models for High-Resolution Images
</h1>


<p align="center">
  <a href="https://github.com/vocaela/kv-ground"><img src="https://img.shields.io/badge/GitHub-Repository-green?logo=github" alt="GitHub"></a>
  <a href="https://huggingface.co/vocaela/KV-Ground-4B-BaseGuiOwl1.5"><img src="https://img.shields.io/badge/Hugging%20Face-4B--BaseGuiOwl1.5-orange?logo=huggingface" alt="KV-Ground-4B-BaseGuiOwl1.5"></a>
  <a href="https://huggingface.co/vocaela/KV-Ground-4B-BaseQw3vl"><img src="https://img.shields.io/badge/Hugging%20Face-4B--BaseQw3vl-orange?logo=huggingface" alt="KV-Ground-4B-BaseQw3vl"></a>
</p>
<p align="center">
  <em>
    By
    <a href="https://www.kingsware.cn/">Kingsware</a>
    &
    <a href="https://vocaela.ai/">Vocaela AI</a>
  </em>
</p>


High-resolution GUI grounding is an extreme important task for professional workforce scenarios like desktop RPA, where the massive usage also implies the favoring of compact models. High-resolution GUI grounding remains challenging to VLMs. The recent progress on ScreenSpot-Pro are mostly from reasoning CoT and agentic framework on top of models, which trade-in significant extra computing time / integration complexity for the quality boost. For pure instruct model, the improvement highly correlates to model size scaling up. However, we argue that grounding as a very fundamental atomic capability needs intrinsic improvement. High-resolution grounding is challenging to VLMs largely due to two reasons: (1) Vision-encoders were pre-trained on images of very low resolution. (2) Lack of high-quality high-resolution annotated data. We belive that even the top-tier VLMs are under-trained on high-resolution grounding data. To address this problem, we synthesize high-quality high-resolution GUI grounding data, and continue post-training Qwen3-VL-4B-Instruct and GUI-Owl-1.5-4B-Instruct with SFT followed by RFT (GRPO). Without reasoning CoT, [KV-Ground-4B-BaseQw3vl](https://huggingface.co/vocaela/KV-Ground-4B-BaseQw3vl) and [KV-Ground-4B-BaseGuiOwl1.5](https://huggingface.co/vocaela/KV-Ground-4B-BaseGuiOwl1.5) achieve 63.2 / 66.5 on ScreenSpot-Pro, ranked into the best models at 4B range. Meanwhile, they maintains excellent performance on regular-resolution tasks with 94.6 / 94.3  on ScreenSpot-V2.

---


## Benchmark Results

- Impact of continue post-training on base models

  For the purpose of controlled comparison, all these numbers are re-/produced by us, using the exact evaluation code in this repo. The baseline numbers may be different from the sources. Please see below `note` section for the controlled setup.

  | Models                          | ScreenSpot-Pro | ScreenSpot-V2 | OSWorld-G    | OSWorld-G-refined | UI-Vision   |
  |---------------------------------|:--------------:|:-------------:|:------------:|:-----------------:|:-----------:|
  | Base: Qwen3-VL-4B-Instruct*     | 59.5           | 93.1          | 63.3         | 71.1              | 30.4        |
  | **KV-Ground-4B-BaseQw3vl***     | 63.2 (+3.7)    | 94.6 (+1.5)   | 64.0 (0.7)   | 71.2 (+0.1)       | 32.6 (+2.2) |
  | Base: GUI-Owl-1.5-4B-Instruct*  | 65.3           | 92.8          | 61.7         | 66.8              | 30.4        |
  | **KV-Ground-4B-BaseGuiOwl1.5*** | 66.5 (+1.2)    | 94.3 (+1.5)   | 62.8 (+1.1)  | 69.1 (+2.3)       | 32.2 (+1.8) |
  
  The results tell us:
  - Our continuous post-training method brings in consistent improvements, even on the latest strong GUI-Owl-1.5-4B-Instruct model.
  - The high-resolution optimized training doesn't harm regular resolution tasks
  
  Especially, it is beyond our expectation on the improvement over GUI-Owl-1.5-4B-Instruct because the technical report of GUI-Owl-1.5-4B-Instruct discloses that they have already applied extensive data synthesize / augementation targeted for high-resolution images.

- Comparision with top models under 8B (ranked by ScreenSpot-Pro)
  
  We consider the top models under 8B from [ScreenSpot-Pro leaderboard](https://gui-agent.github.io/grounding-leaderboard/) and related most recent technical reports.

  | Models                          | ScreenSpot-Pro | ScreenSpot-V2 | OSWorld-G | OSWorld-G-refined | UI-Vision  |
  |---------------------------------|:--------------:|:-------------:|:---------:|:-----------------:|:----------:|
  | *Specialized GUI Models*
  | UI-Venus-1.5-8B                 | 68.4  | 93.2  | 69.4   | -     | -     |
  | **KV-Ground-4B-BaseGuiOwl1.5*** | 66.5  | 94.3  | 62.8   | 69.1  | 32.2  |
  | MAI-UI-8B                       | 65.8  | 95.2  | 60.1   | 68.6  | 40.7  |
  | GUI-Owl-1.5-4B-Instruct*        | 65.3  | 92.8  | 55.3   | 60.4  | 30.4  |
  | **KV-Ground-4B-BaseQw3vl***     | 63.2  | 94.6  | 64.0   | 71.2  | 32.6  |
  | Step-GUI-8B                     | 62.6  | 95.1  | 70.0   | -     | -     |
  | Step-GUI-4B                     | 60.0  | 93.6  | 66.9   | -     | -     |
  | Holo2-8B                        | 58.9  | 93.2  | 70.1   | -     | -     |
  | Holo2-4B                        | 57.2  | 93.2  | 69.4   | -     | -     |
  | GUI-Owl-7B                      | 54.9  | 92.8  | 55.9   | -     | -     |
  | OpenCUA-7B                      | 50.0  | 92.3  | 55.3   | -     | 29.7  |
  | UI-Venus-1.0-7B                 | 50.8  | 94.1  | 54.6   | 61.7  | 36.8  |
  | GTA1-7B                         | 50.1  | 92.4  | 60.1   | 67.7  | -     |
  | UI-TARS-1.5-7B                  | 35.7  | 91.6  | 52.8   | 64.2  | -     |
  | *General VLMs*
  | Qwen3-VL-4B*                    | 59.5  | 93.1  | 63.3   | 71.1  | 30.4  |
  | Qwen3-VL-8B                     | 54.6  | -     | 58.2   | -     | -     | 
  
> Notes:
>  - We only compare pure model capabillity and hence exclude those multi-step methods such as using zoom-in, MVP or agentic frameworks.
>  - By default numbers are from each source. Those numbers produced by us are indicated by `*`, which may be different from the numbers reported from the sources.
>  - It is known that models are usually sensitive to certain pre-designed prompt. As all the runs produced by us are based on Qwen3-VL backbone, for fair and simple comparision, the same prompt structure of `system -> user-image -> user-instruct` is used. And the same [system message](./utils.py) is used, which is the default Qwen3-VL computer-use format prompt adopted by the [ScreenSpot-Pro leaderboard](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding/blob/211ff96b54849f80cca20e02ef03233d3f0e6fdf/models/qwen3vl.py#L20). For OSWorld-G and OSWorld-G-refined, minor modification is made to instruct the refusal setting.

---


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