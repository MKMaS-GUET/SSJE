import torch

class SSJELoss():
    def __init__(self, senti_criterion, entity_criterion, model, optimizer, scheduler, max_grad_norm):
        self._senti_criterion = senti_criterion
        self._entity_criterion = entity_criterion
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._max_grad_norm = max_grad_norm

    def compute(self, entity_logits, senti_logits, entity_types, senti_types, entity_sample_masks, senti_sample_masks):
        # entity loss
        entity_logits = entity_logits.view(-1, entity_logits.shape[-1])
        entity_types = entity_types.view(-1)
        entity_sample_masks = entity_sample_masks.view(-1).float()

        entity_loss = self._entity_criterion(entity_logits, entity_types)
        entity_loss = (entity_loss * entity_sample_masks).sum() / entity_sample_masks.sum()

        # sentiment loss
        senti_sample_masks = senti_sample_masks.view(-1).float()
        senti_count = senti_sample_masks.sum()

        if senti_count.item() != 0:
            senti_logits = senti_logits.view(-1, senti_logits.shape[-1])
            senti_types = senti_types.view(-1, senti_types.shape[-1])

            senti_loss = self._senti_criterion(senti_logits, senti_types)
            senti_loss = senti_loss.sum(-1) / senti_loss.shape[-1]
            senti_loss = (senti_loss * senti_sample_masks).sum() / senti_count

            # joint loss
            train_loss = entity_loss + senti_loss
        else:
            # corner case: no positive/negative sentiment samples
            train_loss = entity_loss

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step()
        self._scheduler.step()
        self._model.zero_grad()
        return train_loss.item()