<h1 align="center">
   KV-Ground-4B: Small GUI Grounding Model for High-Resolution Images
</h1>


<p align="center">
  <a href="https://github.com/vocaela/kv-ground"><img src="https://img.shields.io/badge/GitHub-Repository-green?logo=github" alt="GitHub"></a>
  <a href="https://huggingface.co/vocaela/KV-Ground-4B-BaseQw3vl"><img src="https://img.shields.io/badge/Hugging%20Face-Model-orange?logo=huggingface" alt="Hugging Face"></a>
</p>
<p align="center">
  <em>
    By
    <a href="https://www.kingsware.cn/">Kingsware</a>
    &
    <a href="https://vocaela.ai/">Vocaela AI</a>
  </em>
</p>


High-resolution GUI grounding is an extreme important task for professional workforce scenarios like desktop RPA, where the massive usage also implies the favoring of compact models. High-resolution GUI grounding remains challenging to VLMs. The recent progress on ScreenSpot-Pro are mostly from reasoning CoT and agentic framework on top of models, which trade-in significant extra computing time / integration complexity for the quality boost. For pure instruct model, the improvement highly correlates to model size scaling up. However, we argue that grounding as a very fundamental atomic capability needs intrinsic improvement. High-resolution grounding is challenging to VLMs largely due to two reasons: (1) Vision-encoders were pre-trained on images of very low resolution. (2) Lack of high-quality high-resolution annotated data. We belive that even the top-tier VLMs are under-trained on high-resolution grounding data. To address this problem, we synthesize high-quality high-resolution GUI grounding data, and continue post-training Qwen3-VL-4B-Instruct with SFT followed by RFT (GRPO). Without reasoning CoT, KV-Ground-4B achieves 63.1 on ScreenSpot-Pro, becoming one of the best model at 4B range. Meanwhile, it maintains excellent performance on regular-resolution tasks with 94.8 on ScreenSpot-V2.

---

## Benchmark Results

| Models                      | ScreenSpot-Pro | ScreenSpot-V2 | OSWorld-G | OSWorld-G-refined | UI-Vision  |
|-----------------------------|:--------------:|:-------------:|:---------:|:-----------------:|:----------:|
| *General VLMs*
| Qwen3-VL-4B*                | 59.5  | 93.1  |  55.3  | 60.4  | 30.4  |
| Qwen3-VL-8B                 | 54.6  | -     | 58.2   | -     | -     | 
| *Specialized GUI Models*
| Holo2-4B                    | 57.2  | 93.2  | 69.4   | -     | -     |
| Step-GUI-4B                 | 60.0  | 93.6  | 66.9   | -     | -     |
| OpenCUA-7B                  | 50.0  | 92.3  | 55.3   | -     | 29.7  |
| GTA1-7B                     | 50.1  | 92.4  | 60.1   | 67.7  | -     |
| GUI-Owl-7B                  | 54.9  | 92.8  | 55.9   | -     | -     |
| UI-TARS-1.5-7B              | 35.7  | 91.6  | 52.8   | 64.2  | -     |
| UI-Venus-1.0-7B             | 50.8  | 94.1  | 54.6   | 61.7  | 36.8  |
| Holo2-8B                    | 58.9  | 93.2  | 70.1   | -     | -     |
| Step-GUI-8B                 | 62.6  | 95.1  | 70.0   | -     | -     |
| MAI-UI-8B                   | 65.8  | 95.2  | 60.1   | 68.6  | 40.7  |
| **KV-Ground-4B-BaseQw3vl*** | 63.2  | 94.8  | 58.2   | 64.5  | 32.6  |

> Notes:
>  - By default numbers are copied from each source
>  - `*` indicates the results produced by us
>  - For all the runs produced by us, for fair comparison, the same prompt structure of `system -> user-image -> user-instruct` is used. Similarly, the same [system message](./utils.py) is used, which is the default Qwen3-VL computer-use format prompt and also adopted by the [ScreenSpot-Pro leaderboard](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding/blob/211ff96b54849f80cca20e02ef03233d3f0e6fdf/models/qwen3vl.py#L20). For OSWorld-G and OSWorld-G-refined, minor modification is made to instruct the refusal setting.



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