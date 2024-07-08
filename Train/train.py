import argparse
import torch
import os
from datasets import load_dataset, load_from_disk, DatasetDict
from datetime import timedelta
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, set_seed
from tqdm import tqdm
from transformers import set_seed, default_data_collator
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from flash_attn.losses.cross_entropy import CrossEntropyLoss
import math
from accelerate.utils import (
    InitProcessGroupKwargs,
    set_seed,
    DummyOptim,
    DummyScheduler,
)
from easy_context import (
    prepare_seq_parallel_inputs,
    apply_seq_parallel_monkey_patch,
    prepare_dataloader,
    apply_unsloth_offloaded_gradient_checkpoint_monkey_patch
)

import logging
def setup_logging(save_dir):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(os.path.join(save_dir, 'training_log.txt'))
    file_flag = logging.StreamHandler()

    for handler in [file_handler, file_flag]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
# apply_unsloth_offloaded_gradient_checkpoint_monkey_patch()

def main(args):
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    if args.wandb:
        import wandb
        wandb.init(mode="offline")
        wandb.login()
    set_seed(args.seed)

    timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=1_000_000))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulate_every,
        mixed_precision="bf16",
        log_with="wandb" if args.wandb else None,
        kwargs_handlers=[timeout],
        # fsdp_plugin=fsdp_plugin,
    )
    accelerator.init_trackers(project_name=args.wandb, init_kwargs={"wandb":{"name":args.output_dir.split("/")[-1]}})
    accelerator.print(f"Total GPUS: {accelerator.num_processes}")

    train_dataset = load_dataset('parquet', data_files=args.dataset)

    train_dataset = train_dataset["train"]
    

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=accelerator.device,
        torch_dtype=torch.bfloat16,
        rope_theta=args.rope_theta,
        _attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    logger = setup_logging(args.output_dir) 
    assert isinstance(
        model, (transformers.LlamaForCausalLM, transformers.MistralForCausalLM)
    ), "Only support llama and mistral model"
    model_type = (
        "llama" if isinstance(model, transformers.LlamaForCausalLM) else "mistral"
    )
    apply_seq_parallel_monkey_patch(args.parallel_mode, model_type)

    if "input_ids" not in train_dataset.column_names:
        raise RuntimeError("Dataset must include an `input_ids` feature")
    # remove everything that is not input_ids
    train_dataset = train_dataset.shuffle(seed=args.seed)
    print("Dataset Size:", len(train_dataset))
    train_loader = DataLoader(
        train_dataset,
        collate_fn=default_data_collator,
        shuffle=True,
        batch_size=args.batch_size,
    )
    if args.learning_rate != 2e-5:
        accelerator.print(f"Warning: You also need to modify accelerate_configs/zero3_offload.json to change the learning rate")
    optim = DummyOptim(model.parameters(), lr=args.learning_rate)
    scheduler = DummyScheduler(
        optim,
        num_training_steps=args.max_train_steps,
        total_num_steps=args.max_train_steps,
    )
    model, optim, scheduler = accelerator.prepare(model, optim, scheduler)
    train_loader = prepare_dataloader(args.parallel_mode, train_loader, accelerator)
    model.gradient_checkpointing_enable()

    accelerator.register_for_checkpointing(scheduler)

    accelerator.print(f"Max train steps: {args.max_train_steps}")
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0

    model.train()
    args.seq_length =32767
    loss_func = CrossEntropyLoss(inplace_backward=True)
    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"][..., : args.seq_length+1][..., :-1]
        target_ids = batch["labels"][..., : args.seq_length+1][..., 1:]
        # target_ids = batch["input_ids"][..., : args.seq_length+1][..., 1:]
        # print(input_ids)
        # print(target_ids)
        position_ids = (
            torch.arange(args.seq_length).unsqueeze(0).expand(input_ids.shape[0], -1)
        )

        # shard the input_ids according to the world size and rank according to zig zag attention

        prepared = prepare_seq_parallel_inputs(
            args.parallel_mode,
            input_ids,
            position_ids,
            target_ids,
            accelerator.process_index,
            accelerator.num_processes,
            accelerator.device,
        )
        local_input_ids = prepared["local_input_ids"]
        local_position_ids = prepared["local_position_ids"]
        local_target_ids = prepared["local_target_ids"]
     
        
        loss_log = None
        with accelerator.accumulate(model):
            logits = model(
                local_input_ids,
                position_ids=local_position_ids,
            ).logits
            loss = loss_func(
                logits.reshape(-1, logits.shape[-1]), local_target_ids.reshape(-1)
            )
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                gathered_loss = accelerator.reduce(loss.clone().detach(), "mean")
                loss_log = {
                    "loss": gathered_loss.item(),
                    "ppl": math.exp(gathered_loss.item()),
                }
                accelerator.log(loss_log, step=completed_steps)
                logging_loss = f"Step {completed_steps}: Loss = {gathered_loss.item():.4f}, PPL = {math.exp(gathered_loss.item()):.4f}"
                
                if accelerator.is_main_process:
                    logger.info(logging_loss)

            optim.step()
            scheduler.step()
            optim.zero_grad()

        
        if (completed_steps+1) % args.save_interval == 0 and completed_steps+1 != args.max_train_steps:
            if args.output_dir is not None:
                accelerator.print(f"Saving model to {args.output_dir}/checkpoint-{completed_steps}")

                # Ensure all processes are synced before saving
                accelerator.wait_for_everyone()

                # Retrieve the state dictionary in a way that's compatible with distributed training
                state_dict = accelerator.get_state_dict(model)

                # Determine the checkpoint path and ensure it's only created by the main process
                checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{completed_steps}")
                if accelerator.is_main_process:
                    os.makedirs(checkpoint_path, exist_ok=True)

                # Save model using the Accelerator's recommended method for distributed setups
                accelerator.unwrap_model(model).save_pretrained(
                    checkpoint_path,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                    state_dict=state_dict
                )
                tokenizer.save_pretrained(checkpoint_path)

                accelerator.print(f"Checkpoint saved at step {completed_steps}")


        if accelerator.sync_gradients:
            progress_bar.update(1)
            if loss_log is not None:
                progress_bar.set_postfix(loss_log)
            completed_steps += 1

        if completed_steps >= args.max_train_steps:
            break

    accelerator.print(f"Training Finished")
    accelerator.end_training()

    if args.output_dir is not None:
        accelerator.print(f"Saving model to {args.output_dir}")

        accelerator.wait_for_everyone()

        state_dict = accelerator.get_state_dict(model)

        accelerator.unwrap_model(model).save_pretrained(
            f"{args.output_dir}",
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=state_dict,
        )
        tokenizer.save_pretrained(args.output_dir)
        accelerator.print(f"Saving Finished")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--batch-size", type=int, default=1)
    args.add_argument("--gradient-accumulate-every", type=int, default=8)
    args.add_argument("--output-dir", type=str, required=True)
    args.add_argument("--wandb", type=str)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--max-train-steps", type=int, default=400)
    args.add_argument("--learning-rate", type=float, default=2e-5)
    args.add_argument("--rope-theta", type=float, default=1000000)
    args.add_argument("--save-interval",type=int, default=50)
    # args.add_argument("--rotary_emb_base", type=float, default=1000000)
  
    args.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    args.add_argument(
        "--dataset",
        type=str,
        default="emozilla/pg_books-tokenized-bos-eos-chunked-65536",
    )
    args.add_argument("--seq-length", type=int, default=16384)
    args.add_argument(
        "--parallel_mode",
        type=str,
        choices=["zigzag_ring_attn", "dist_flash_attn", "ulysses_attn", "data_parallel"],
    )
    main(args.parse_args())
