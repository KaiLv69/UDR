from sentence_transformers import SentenceTransformer
from typing import Dict
import torch


class IndexEmbedder(torch.nn.Module):
    def __init__(self, model_name) -> None:
        super().__init__()
        self.embedder = SentenceTransformer(model_name)

    def forward(self, instruction, enc_text, **kwargs) -> Dict[str, torch.Tensor]:
        input = [[i, e, 0] for i, e in zip(instruction, enc_text)]
        enc_emb = self.embedder.encode(input, show_progress_bar=False)
        return enc_emb