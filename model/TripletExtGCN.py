# GCN
from transformers import BertModel
from transformers import BertPreTrainedModel
from transformers import BertConfig
from torch import nn as nn
import torch
from trainer import util, sampling
import os
from layer.GCN import GCN
def get_token(h: torch.tensor, x: torch.tensor, token: int):
    """ Get specific token embedding (e.g. [CLS]) """
    emb_size = h.shape[-1]

    token_h = h.view(-1, emb_size)
    flat = x.contiguous().view(-1)

    # get contextualized embedding of given token
    token_h = token_h[flat == token, :]

    return token_h

class TripletExtGCN(BertPreTrainedModel):
    VERSION = '1.1'
    def __init__(self, config: BertConfig, cls_token: int, sentiment_types: int, entity_types: int,
                 size_embedding: int, prop_drop: float, freeze_transformer: bool, max_pairs: int = 100):
        super(TripletExtGCN, self).__init__(config)
        # BERT model
        self.bert = BertModel(config)
        self.gcn = GCN()
        self.senti_classifier = nn.Linear(config.hidden_size * 3 + size_embedding * 2, sentiment_types)
        self.entity_classifier = nn.Linear(config.hidden_size * 2 + size_embedding, entity_types)
        self.size_embeddings = nn.Embedding(100, size_embedding)
        self.dropout = nn.Dropout(prop_drop)
        self._cls_token = cls_token
        self._sentiment_types = sentiment_types
        self._entity_types = entity_types
        self._max_pairs = max_pairs

        self.neg_span_all = 0
        self.neg_span = 0

        # weight initialization
        self.init_weights()
        if freeze_transformer:
            print("Freeze transformer weights")

            # freeze all transformer weights
            for param in self.bert.parameters():
                param.requires_grad = False

    def _forward_train(self, encodings: torch.tensor, context_masks: torch.tensor, entity_masks: torch.tensor,
                       entity_sizes: torch.tensor, sentiments: torch.tensor, senti_masks: torch.tensor, adj):
        context_masks = context_masks.float()
        h = self.bert(input_ids=encodings, attention_mask=context_masks)[0]
        # 图卷积计算
        h_gcn, pool_mask = self.gcn(adj, h)
        h = h + h_gcn
        batch_size = encodings.shape[0]

        # print("encodings:", encodings, encodings.size())
        # print("entity_masks", entity_masks, entity_masks.size())

        # entity_classify
        size_embeddings = self.size_embeddings(entity_sizes)
        entity_clf, entity_spans_pool = self._classify_entities(encodings, h, entity_masks, size_embeddings)

        # relation_classify
        h_large = h.unsqueeze(1).repeat(1, max(min(sentiments.shape[1], self._max_pairs), 1), 1, 1)
        senti_clf = torch.zeros([batch_size, sentiments.shape[1], self._sentiment_types]).to(self.senti_classifier.weight.device)

        # obtain sentiment logits
        # chunk processing to reduce memory usage
        for i in range(0, sentiments.shape[1], self._max_pairs):
            # classify sentiment candidates
            chunk_senti_logits = self._classify_sentiments(entity_spans_pool, size_embeddings,
                                                        sentiments, senti_masks, h_large, i)
            senti_clf[:, i:i + self._max_pairs, :] = chunk_senti_logits

        return entity_clf, senti_clf

    def _forward_eval(self, encodings: torch.tensor, context_masks: torch.tensor, entity_masks: torch.tensor,
                      entity_sizes: torch.tensor, entity_spans: torch.tensor, entity_sample_masks: torch.tensor, adj):
        context_masks = context_masks.float()

        h = self.bert(input_ids=encodings, attention_mask=context_masks)[0]
        # calculate gcn
        h_gcn, pool_mask = self.gcn(adj, h)
        h = h + h_gcn
        batch_size = encodings.shape[0]
        ctx_size = context_masks.shape[-1]

        # entity_classify
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes
        entity_clf, entity_spans_pool = self._classify_entities(encodings, h, entity_masks, size_embeddings)

        # ignore entity candidates that do not constitute an actual entity for sentiments (based on classifier)
        sentiments, senti_masks, senti_sample_masks = self._filter_spans(entity_clf, entity_spans,
                                                                    entity_sample_masks, ctx_size)
        senti_sample_masks = senti_sample_masks.float().unsqueeze(-1)
        h_large = h.unsqueeze(1).repeat(1, max(min(sentiments.shape[1], self._max_pairs), 1), 1, 1)
        senti_clf = torch.zeros([batch_size, sentiments.shape[1], self._sentiment_types]).to(
            self.senti_classifier.weight.device)

        # obtain sentiment logits
        # chunk processing to reduce memory usage
        for i in range(0, sentiments.shape[1], self._max_pairs):
            # classify sentiment candidates
            chunk_senti_logits = self._classify_sentiments(entity_spans_pool, size_embeddings,
                                                        sentiments, senti_masks, h_large, i)
            # apply sigmoid
            chunk_senti_clf = torch.sigmoid(chunk_senti_logits)
            senti_clf[:, i:i + self._max_pairs, :] = chunk_senti_clf

        senti_clf = senti_clf * senti_sample_masks  # mask

        # apply softmax
        entity_clf = torch.softmax(entity_clf, dim=2)

        return entity_clf, senti_clf, sentiments

    def _classify_entities(self, encodings, h, entity_masks, size_embeddings):
        m = (entity_masks.unsqueeze(-1) == 0).float() * (-1e30)
        entity_spans_pool = m + h.unsqueeze(1).repeat(1, entity_masks.shape[1], 1, 1)
        entity_spans_pool = entity_spans_pool.max(dim=2)[0]
        # get cls token as candidate context representation
        entity_ctx = get_token(h, encodings, self._cls_token)

        # create candidate representations including context, max pooled span and size embedding
        entity_repr = torch.cat([entity_ctx.unsqueeze(1).repeat(1, entity_spans_pool.shape[1], 1),
                                 entity_spans_pool, size_embeddings], dim=2)
        entity_repr = self.dropout(entity_repr)

        # classify entity candidates
        entity_clf = self.entity_classifier(entity_repr)

        return entity_clf, entity_spans_pool

    def _classify_sentiments(self, entity_spans, size_embeddings, sentiments, senti_masks, h, chunk_start):
        batch_size = sentiments.shape[0]

        # create chunks if necessary
        if sentiments.shape[1] > self._max_pairs:
            sentiments = sentiments[:, chunk_start:chunk_start + self._max_pairs]
            senti_masks = senti_masks[:, chunk_start:chunk_start + self._max_pairs]
            h = h[:, :sentiments.shape[1], :]

        # get pairs of entity candidate representations
        entity_pairs = util.batch_index(entity_spans, sentiments)
        entity_pairs = entity_pairs.view(batch_size, entity_pairs.shape[1], -1)

        # get corresponding size embeddings
        size_pair_embeddings = util.batch_index(size_embeddings, sentiments)
        size_pair_embeddings = size_pair_embeddings.view(batch_size, size_pair_embeddings.shape[1], -1)

        # sentiment context (context between entity candidate pair)
        # mask non entity candidate tokens
        m = ((senti_masks == 0).float() * (-1e30)).unsqueeze(-1)
        senti_ctx = m + h
        # max pooling
        senti_ctx = senti_ctx.max(dim=2)[0]
        # set the context vector of neighboring or adjacent entity candidates to zero
        senti_ctx[senti_masks.to(torch.uint8).any(-1) == 0] = 0

        # create sentiment candidate representations including context, max pooled entity candidate pairs
        # and corresponding size embeddings
        senti_repr = torch.cat([senti_ctx, entity_pairs, size_pair_embeddings], dim=2)
        senti_repr = self.dropout(senti_repr)

        # classify sentiment candidates
        chunk_senti_logits = self.senti_classifier(senti_repr)
        return chunk_senti_logits

    def log_sample_total(self,neg_entity_count_all):
        log_path = os.path.join('./log/', 'countSample.txt')
        with open(log_path, mode='a', encoding='utf-8') as f:
            f.write('neg_entity_count_all: \n')
            self.neg_span_all += len(neg_entity_count_all)
            f.write(str(self.neg_span_all))
            f.write('\nneg_entity_count: \n')
            self.neg_span += len((neg_entity_count_all !=0).nonzero())
            f.write(str(self.neg_span))
            f.write('\n')
        f.close()

    def _filter_spans(self, entity_clf, entity_spans, entity_sample_masks, ctx_size):
        batch_size = entity_clf.shape[0]
        entity_logits_max = entity_clf.argmax(dim=-1) * entity_sample_masks.long()  # get entity type (including none)
        batch_sentiments = []
        batch_senti_masks = []
        batch_senti_sample_masks = []

        for i in range(batch_size):
            rels = []
            senti_masks = []
            sample_masks = []

            # get spans classified as entities
            self.log_sample_total(entity_logits_max[i])
            non_zero_indices = (entity_logits_max[i] != 0).nonzero().view(-1)
            non_zero_spans = entity_spans[i][non_zero_indices].tolist()
            non_zero_indices = non_zero_indices.tolist()


            # create sentiments and masks
            for i1, s1 in zip(non_zero_indices, non_zero_spans):
                for i2, s2 in zip(non_zero_indices, non_zero_spans):
                    if i1 != i2:
                        rels.append((i1, i2))
                        senti_masks.append(sampling.create_senti_mask(s1, s2, ctx_size))
                        sample_masks.append(1)

            if not rels:
                # case: no more than two spans classified as entities
                batch_sentiments.append(torch.tensor([[0, 0]], dtype=torch.long))
                batch_senti_masks.append(torch.tensor([[0] * ctx_size], dtype=torch.bool))
                batch_senti_sample_masks.append(torch.tensor([0], dtype=torch.bool))
            else:
                # case: more than two spans classified as entities
                batch_sentiments.append(torch.tensor(rels, dtype=torch.long))
                batch_senti_masks.append(torch.stack(senti_masks))
                batch_senti_sample_masks.append(torch.tensor(sample_masks, dtype=torch.bool))

        # stack
        device = self.senti_classifier.weight.device
        batch_sentiments = util.padded_stack(batch_sentiments).to(device)
        batch_senti_masks = util.padded_stack(batch_senti_masks).to(device)
        batch_senti_sample_masks = util.padded_stack(batch_senti_sample_masks).to(device)

        return batch_sentiments, batch_senti_masks, batch_senti_sample_masks

    def forward(self, *args, evaluate=False, **kwargs):
        if not evaluate:
            return self._forward_train(*args, **kwargs)
        else:
            return self._forward_eval(*args, **kwargs)

_MODELS = {
    'TripletExtGCN': TripletExtGCN,
}

def get_model(name):
    return _MODELS[name]


