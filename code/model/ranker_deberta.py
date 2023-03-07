from transformers import DebertaV2Model, DebertaV2PreTrainedModel
from transformers.models.deberta.modeling_deberta import ContextPooler


import torch
from torch import nn
import torch.nn.functional as F



class Ranker(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.deberta = DebertaV2Model(config)
        self.pooler  = ContextPooler(config)

        self.scorer  = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, 1)
        )

    def forward(self, **inputs):
        labels   = inputs.pop('labels')
        neg_labels = inputs.pop('neg_labels')
        examples = inputs.pop('examples')

        inputs1 = inputs.pop('inputs1')
        inputs2 = inputs.pop('inputs2')

        outputs1 = self.deberta(**inputs1)
        outputs2 = self.deberta(**inputs2)
        pooled_output1 = self.pooler(outputs1[0]) # [B, d]
        pooled_output2 = self.pooler(outputs2[0]) # [B, d]

        score1 = self.scorer(pooled_output1)
        score2 = self.scorer(pooled_output2)

        scores = torch.cat((score1, score2), dim=1) # [B, 2]
        loss = F.cross_entropy(scores, labels)

        loss1 = F.binary_cross_entropy_with_logits(score1, torch.ones_like(score1))
        loss2 = F.binary_cross_entropy_with_logits(score2.squeeze(), neg_labels.float())

        if self.training is False:
            import random
            for example, s1, s2 in zip(examples, score1, score2):
                if random.random() < 0.01:
                    print()
                    # print(f'[{example.label} | {s1.item():.2f}, {s2.item():.2f}]', example.question, example.answer1, example.answer2)
                    print(f'[{example.label} {s1.item()>s2.item()}]', example.question)
                    print(example.answer, f'{s1.item():.2f}')
                    print(example.neg_answer, f'{s2.item():.2f}')
                    print()

        return {
            'loss': loss + loss1 + loss2,
            'predictions': scores.argmax(dim=1).detach().cpu().numpy(),
            'ground-truth': labels.cpu().numpy(),
        }

    def predict(self, **inputs):
        outputs = self.deberta(**inputs)
        pooled_output = self.pooler(outputs[0])
        
        score = self.scorer(pooled_output)[:, 0]

        score = score.sigmoid()

        return score.detach().cpu().numpy()
