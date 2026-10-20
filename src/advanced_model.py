import torch
import torch.nn as nn
from typing import Optional, Tuple


class CrossAttentionBlock(nn.Module):
    """Allows Modality A to attend to Modality B.
    For example, Text query attending to Audio key/value to resolve sarcastic tones,
    or Text query attending to Facial expression key/value.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4):
        super(CrossAttentionBlock, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Args:
            query, key, value shape: (batch, embed_dim) or (batch, seq_len, embed_dim)
        """
        needs_squeeze = False
        if query.dim() == 2:
            q = query.unsqueeze(1)
            k = key.unsqueeze(1)
            v = value.unsqueeze(1)
            needs_squeeze = True
        else:
            q, k, v = query, key, value

        attn_output, _ = self.multihead_attn(q, k, v)

        # Add & Norm
        out1 = self.layer_norm(q + attn_output)

        # Feed Forward
        ffn_output = self.ffn(out1)

        # Add & Norm
        out2 = self.layer_norm2(out1 + ffn_output)

        if needs_squeeze:
            return out2.squeeze(1)
        return out2


class SOTAFusionModel(nn.Module):
    """State-of-the-art Multimodal Fusion Model with Cross-Attention and Conversational Context.

    Stages:
    1. Modality projection: Text (768), Audio (768), Face (2048) -> hidden_dim (256)
    2. Cross-modal attention: Text attending to Audio, Text attending to Face
    3. Conversational context: Recurrent GRU tracking dialogue state across utterances
    4. Optional emotional vector injection (e.g., VAD scores or LLM summary embeddings)
    5. Final classification head (7 emotion classes)
    """

    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 768,
        face_dim: int = 2048,
        hidden_dim: int = 256,
        emotion_dim: int = 300,
        num_classes: int = 7,
    ):
        super(SOTAFusionModel, self).__init__()

        # 1. Project all modalities to common hidden dimension
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.face_proj = nn.Linear(face_dim, hidden_dim)

        # 2. Cross-Modal Attention Layers
        self.text_audio_attn = CrossAttentionBlock(hidden_dim)
        self.text_face_attn = CrossAttentionBlock(hidden_dim)

        # 3. Conversational Context Memory (GRU)
        self.context_gru = nn.GRU(
            input_size=hidden_dim * 3,
            hidden_size=hidden_dim * 2,
            batch_first=True,
        )

        # 4. Emotional Vector Projection (e.g. VAD scores / LLM embeddings)
        self.emotion_vector_proj = nn.Linear(emotion_dim, hidden_dim)

        # 5. Final Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(
        self,
        text_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        face_feat: torch.Tensor,
        emotion_vector: Optional[torch.Tensor] = None,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:
            text_feat: (batch, text_dim)
            audio_feat: (batch, audio_dim)
            face_feat: (batch, face_dim)
            emotion_vector: (batch, emotion_dim) optional
            hidden_state: (1, batch, hidden_dim * 2) previous utterance context

        Returns:
            logits: (batch, num_classes)
            new_hidden_state: (1, batch, hidden_dim * 2)
        """
        # Project to common dimension
        t = torch.relu(self.text_proj(text_feat))
        a = torch.relu(self.audio_proj(audio_feat))
        f = torch.relu(self.face_proj(face_feat))

        # Cross Attention
        t_a = self.text_audio_attn(query=t, key=a, value=a)
        t_f = self.text_face_attn(query=t, key=f, value=f)

        # Concatenate attended features
        fused = torch.cat([t, t_a, t_f], dim=1)  # (batch, hidden_dim * 3)

        # Pass through Conversational GRU
        fused_seq = fused.unsqueeze(1)  # (batch, 1, hidden_dim * 3)
        context_out, new_hidden_state = self.context_gru(fused_seq, hidden_state)
        context_feat = context_out.squeeze(1)  # (batch, hidden_dim * 2)

        # Combine with emotional vector (or zero tensor if None)
        if emotion_vector is not None:
            e_feat = torch.relu(self.emotion_vector_proj(emotion_vector))
        else:
            e_feat = torch.zeros(
                (text_feat.size(0), self.text_proj.out_features),
                device=text_feat.device,
            )

        final_rep = torch.cat([context_feat, e_feat], dim=1)  # (batch, hidden_dim * 3)
        logits = self.classifier(final_rep)

        return logits, new_hidden_state


if __name__ == "__main__":
    batch = 4
    t = torch.randn(batch, 768)
    a = torch.randn(batch, 768)
    f = torch.randn(batch, 2048)
    e = torch.randn(batch, 300)

    model = SOTAFusionModel()
    logits, h = model(t, a, f, e)
    print("SOTAFusionModel forward pass successful!")
    print(f"Logits shape: {logits.shape}, Hidden shape: {h.shape}")
