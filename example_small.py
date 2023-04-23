# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
#import torch_directml
import fire
import time
import json

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA

driver = "cuda"
#driver = "cpu"
#driver = "dml"
#dml = torch_directml.device()



def setup_model_parallel() -> Tuple[int, int]:
    global driver
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    #if driver=="cudax":
    #    torch.distributed.init_process_group("nccl")
    torch.distributed.init_process_group("gloo")
    initialize_model_parallel(world_size)
    if driver=="cuda":
        torch.cuda.set_device(local_rank)
    if driver=="dml":
        pass

    # seed must be the same in all processes
    torch.manual_seed(123)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    global driver
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading..")
    with torch.no_grad():
        #checkpoint = torch.load(ckpt_path, map_location="cuda:0")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=1+0*max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    if driver=="cuda":
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
    if driver=="cpu" or driver=="dml":
        torch.set_default_tensor_type(torch.HalfTensor)
    model = Transformer(model_args)
    #if driver=="cuda":
    torch.set_default_tensor_type(torch.FloatTensor)
    #if driver=="cpu":
    #torch.set_default_tensor_type(torch.HalfTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
):
    #print("****"+str(dml))
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, 1+0*max_batch_size
    )

    prompt = "Once upon a time, there were three bears. They"

    while prompt != "":
        results = generator.generate(
            [prompt], max_gen_len=256, temperature=temperature, top_p=top_p
        )

        for result in results:
            print(result)
            print("\n==================================\n")

        prompt = str(input())


if __name__ == "__main__":
    fire.Fire(main)


'''
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        "Building a website can be done in 10 simple steps:\n",
        # Few shot prompts: https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api
        """Tweet: "I hate it when my phone battery dies."
Sentiment: Negative
###
Tweet: "My day has been 👍"
Sentiment: Positive
###
Tweet: "This is the link to the article"
Sentiment: Neutral
###
Tweet: "This new music video was incredibile"
Sentiment:""",
        """Translate English to French:

sea otter => loutre de mer

peppermint => menthe poivrée

plush girafe => girafe peluche

cheese =>""",
    ]
'''
