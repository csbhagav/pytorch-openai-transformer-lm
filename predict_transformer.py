import argparse
from typing import List, Callable, Any

from model_pytorch import TransformerModel, DEFAULT_CONFIG
from model_pytorch import load_openai_pretrained_model
from text_utils import TextEncoder
import torch
import numpy as np
import pickle


def get_range_vector(size: int, device: int) -> torch.Tensor:
    """
    Returns a range vector with the desired size, starting at 0. The CUDA implementation
    is meant to avoid copy data from CPU to GPU.
    """
    if device > -1:
        return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
    else:
        return torch.arange(0, size, dtype=torch.long)


def pad_sequence_to_length(sequence: List,
                           desired_length: int,
                           default_value: Callable[[], Any] = lambda: 0,
                           padding_on_right: bool = True) -> List:
    """
    Take a list of objects and pads it to the desired length, returning the padded list.  The
    original list is not modified.

    Parameters
    ----------
    sequence : List
        A list of objects to be padded.

    desired_length : int
        Maximum length of each sequence. Longer sequences are truncated to this length, and
        shorter ones are padded to it.

    default_value: Callable, default=lambda: 0
        Callable that outputs a default value (of any type) to use as padding values.  This is
        a lambda to avoid using the same object when the default value is more complex, like a
        list.

    padding_on_right : bool, default=True
        When we add padding tokens (or truncate the sequence), should we do it on the right or
        the left?

    Returns
    -------
    padded_sequence : List
    """
    # Truncates the sequence to the desired length.
    if padding_on_right:
        padded_sequence = sequence[:desired_length]
    else:
        padded_sequence = sequence[-desired_length:]
    # Continues to pad with default_value() until we reach the desired length.
    for _ in range(desired_length - len(padded_sequence)):
        if padding_on_right:
            padded_sequence.append(default_value())
        else:
            padded_sequence.insert(0, default_value())
    return padded_sequence


def transformer_predict(input_file: str, output_file: str, text_encoder: TextEncoder, device: int):
    print(input_file)
    n_ctx = 512

    transformer = TransformerModel(DEFAULT_CONFIG, vocab=40993, n_ctx=n_ctx)
    load_openai_pretrained_model(transformer, n_ctx=n_ctx, n_special=3)

    with open(input_file) as f:
        sentences = f.readlines()

    encoded_sentences = text_encoder.encode(sentences)

    input_tensor = torch.Tensor(
        [pad_sequence_to_length(s, desired_length=512) for s in encoded_sentences]
        , device="cuda").cuda().long()
    masks = [np.concatenate(
        (np.ones(len(s)) + np.zeros(n_ctx - len(s))))
        for s in encoded_sentences]

    batch_size, num_timesteps = input_tensor.size()

    positional_encodings = get_range_vector(num_timesteps, device) + n_ctx

    batch_tensor = torch.stack(
        [input_tensor,
         positional_encodings.expand(batch_size, num_timesteps)],
        dim=-1)

    transformer = transformer.cuda()
    transformer_embeddings = transformer(batch_tensor)
    transformer_embeddings_numpy = transformer_embeddings.data.cpu().numpy()
    dict = {
        "mask": masks,
        "embedding": transformer_embeddings_numpy
    }
    pickle.dump(dict, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--encoder_path', type=str, default='model/encoder_bpe_40000.json')
    parser.add_argument('--bpe_path', type=str, default='model/vocab_40000.bpe')
    parser.add_argument('--device', type=int, default='-1')

    args = parser.parse_args()
    print(args)

    input_file = args.input_file
    output_file = args.output_file
    encoder_path = args.encoder_path
    bpe_path = args.bpe_path
    device = args.device

    text_encoder = TextEncoder(encoder_path, bpe_path)

    transformer_predict(
        input_file=input_file,
        output_file=output_file,
        text_encoder=text_encoder,
        device=device)
