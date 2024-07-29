import torch
import torch.nn as nn
import pytorch_lightning as pl
import vilt.modules.vision_transformer as vit

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from vilt.modules import heads, objectives, vilt_utils
from collections import OrderedDict


class ViLTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )

        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        if self.hparams.config["load_path"] == "":
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config
            )
        else:
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False, config=self.hparams.config
            )

        self.pooler = heads.Pooler(config["hidden_size"])
        self.pooler.apply(objectives.init_weights)

        if config["loss_names"]["mlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(config["hidden_size"])
            self.itm_score.apply(objectives.init_weights)

        if config["loss_names"]["mpp"] > 0:
            self.mpp_score = heads.MPPHead(bert_config)
            self.mpp_score.apply(objectives.init_weights)

        # ===================== Downstream ===================== #
        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
        ):
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            new_state_dict = OrderedDict()
            
            for n in state_dict:
                if 'qkv' in n:
                    q, k, v = state_dict[n].chunk(3, dim=0)
                    new_state_dict[n.replace('qkv', 'q')] = q
                    new_state_dict[n.replace('qkv', 'k')] = k
                    new_state_dict[n.replace('qkv', 'v')] = v

                    new_state_dict[n.replace('qkv','q_init')] = q.clone()
                    new_state_dict[n.replace('qkv','k_init')] = k.clone()
                    new_state_dict[n.replace('qkv','v_init')] = v.clone()

                    new_state_dict[n.replace('qkv','q_ema')] = q.clone()
                    new_state_dict[n.replace('qkv','k_ema')] = k.clone()
                    new_state_dict[n.replace('qkv','v_ema')] = v.clone()

            new_state_dict.update(state_dict)
            missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
            print('missing keys:', missing_keys)
            print()
            print('unexpected keys:', unexpected_keys)
            print()

        hs = self.hparams.config["hidden_size"]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(hs * 2, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            emb_data = self.token_type_embeddings.weight.data
            self.token_type_embeddings = nn.Embedding(3, hs)
            self.token_type_embeddings.apply(objectives.init_weights)
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

        if self.hparams.config["loss_names"]["irtr"] > 0:
            self.rank_output = nn.Linear(hs, 1)
            self.rank_output.weight.data = self.itm_score.fc.weight.data[1:, :]
            self.rank_output.bias.data = self.itm_score.fc.bias.data[1:]
            self.margin = 0.2
            for p in self.itm_score.parameters():
                p.requires_grad = False

        vilt_utils.set_metrics(self)
        self.current_tasks = list()

        self.moil_step_size = 1

        for p in self.text_embeddings.parameters():
            p.requires_grad = False
        for p in self.token_type_embeddings.parameters():
            p.requires_grad = False
        for p in self.transformer.parameters():
            p.requires_grad = False
        for p in self.pooler.parameters():
            p.requires_grad = False
        
        for layer in range(config["num_layers"]):
            for p in self.transformer.blocks[layer].attn.q.parameters():
                p.requires_grad = True
            for p in self.transformer.blocks[layer].attn.q_lora.parameters():
                p.requires_grad = True
            
            for p in self.transformer.blocks[layer].attn.k.parameters():
                p.requires_grad = True
            for p in self.transformer.blocks[layer].attn.k_lora.parameters():
                p.requires_grad = True
            
            for p in self.transformer.blocks[layer].attn.v.parameters():
                p.requires_grad = True
            for p in self.transformer.blocks[layer].attn.v_lora.parameters():
                p.requires_grad = True
            
            for p in self.transformer.blocks[layer].attn.adapter.parameters():
                p.requires_grad = True
        
        self.I = nn.Parameter(torch.eye(config["hidden_size"], config["hidden_size"]))
        self.I.requires_grad = False

        # Double check
        enabled = OrderedDict()
        for name, param in self.named_parameters():
            if param.requires_grad:
                enabled[name] = None
        print(f"Parameters to be updated: {enabled.keys()}")
        print()
        total_num = sum(p.numel() for p in self.parameters())
        model_num = sum(p.numel() for n, p in self.named_parameters() 
        if '_init' not in n and '_ema' not in n and '_val' not in n and 'lora' not in n and 'adapter' not in n)
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        ada_num = sum(p.numel() for n, p in self.named_parameters() if ('adapter' in n or 'lora' in n ) and p.requires_grad)
        heads_num = sum(p.numel() for n, p in self.named_parameters() if ('pooler' in n or 'rank_output' in n or '_classifier' in n) and p.requires_grad)

        params = {'Total': total_num, 'Model': model_num, 'Trainable': trainable_num, 'Adapter': ada_num, 'Head': heads_num}

        print(params)
        print('Head/Model:', round(heads_num/model_num*100, 2), '%')
        print('Adapter/Model:', round(ada_num/model_num*100, 2), '%')

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
        val_mode=None
    ):
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        text_embeds = self.text_embeddings(text_ids)

        if image_embeds is None and image_masks is None:
            img = batch[imgkey][0]
            (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.hparams.config["max_image_len"],
                mask_it=mask_image,
            )
        else:
            patch_index, image_labels = (
                None,
                None,
            )

        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )

        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        co_masks = torch.cat([text_masks, image_masks], dim=1)

        x = co_embeds

        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=co_masks, val_mode=val_mode, I=self.I)

        x = self.transformer.norm(x)
        text_feats, image_feats = (
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1] :],
        )
        cls_feats = self.pooler(x)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "image_labels": image_labels,
            "image_masks": image_masks,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "patch_index": patch_index,
        }

        return ret

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Masked Patch Prediction
        if "mpp" in self.current_tasks:
            ret.update(objectives.compute_mpp(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm_wpa(self, batch))

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))

        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_nlvr2(self, batch))

        # Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        def compute_qkv_weight(m, m_adapter, m_init, m_lora):
            _weight = (m_init.weight + m_lora.B @ m_lora.A) @ (self.I + m_adapter.B.weight @ m_adapter.A.weight)
            _bias = (m_init.weight + m_lora.B @ m_lora.A) @ (m_adapter.B.weight @ m_adapter.A.bias + m_adapter.B.bias) + m_init.bias
            return compute_linear_loss(m, _weight, _bias)

        def compute_linear_loss(m, _weight, _bias):
            loss_m_weight = ((m.weight - _weight)**2).mean(dim=0).mean(dim=0)
            loss_m_bias = ((m.bias - _bias)**2).mean()
            return loss_m_weight + loss_m_bias

        def compute_ema(m, m_ema):
            m_ema.weight.data = ema_rate*m_ema.weight.data + (1-ema_rate)*m.weight.data
            m_ema.bias.data = ema_rate*m_ema.bias.data + (1-ema_rate)*m.bias.data
        
        ema_rate = self.hparams.config["ema"]
        loss_q = loss_k = loss_v = 0

        if self.trainer.global_step%self.moil_step_size==0 and self.trainer.global_step!=0:
            for layer in range(len(self.transformer.blocks)):
                a = self.transformer.blocks[layer].attn
                compute_ema(a.q, a.q_ema)
                compute_ema(a.k, a.k_ema)
                compute_ema(a.v, a.v_ema)

                loss_q += compute_qkv_weight(a.q_ema, a.adapter, a.q_init, a.q_lora)
                loss_k += compute_qkv_weight(a.k_ema, a.adapter, a.k_init, a.k_lora)
                loss_v += compute_qkv_weight(a.v_ema, a.adapter, a.v_init, a.v_lora)
        
        return total_loss + loss_q + loss_k + loss_v

    def training_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return vilt_utils.set_schedule(self)
