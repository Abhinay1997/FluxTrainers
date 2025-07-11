# FluxTrainers
A repo for training Flux with different techniques. Empty for now but will add code for them soon. Planning to do long term support for these.

Changelog:

**11th July**
* Added Batch Overfit test for flux_draft.py. WIP, will not work without dataset NagaSaiAbhinay/MJ-Prompts table.parquet already downloaded. Will be fixed soon.

1. Flux DRaFt & DRaFT-LV
   Note: Its in a standalone repo right now, will move it here. https://arxiv.org/abs/2309.17400
2. Flux DPO
   Note: Pending https://arxiv.org/abs/2311.12908
3. Flux LoRA
   Note: Pending but the plan is to support speedups and optimizers from heavyball. 
4. Flux SingLoRA
   Note: SingLoRA is cheaper to train. Priority. https://www.arxiv.org/abs/2507.05566
5. T-LoRA
   Note: Skeptical if a single image can be used to train a lora. https://arxiv.org/abs/2507.05964
7. Tokenverse
   Note: Entirely new branch of the model to train. Pretty exciting. https://arxiv.org/abs/2501.12224
