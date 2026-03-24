import torch
import torch.nn as nn

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from copy import deepcopy
import torch.nn.functional as F

_tokenizer = _Tokenizer()

__all__ = ['dpcoop', 'DPCoop']

numb_1 = [6, 13, 7, 6, 6, 18, 3, 11, 17, 12, 8, 7, 11, 1, 11, 11, 9, 8, 5, 6]
numb_tensor1 = torch.tensor(numb_1)

numb_2 = [2, 6, 11, 18, 3, 6, 5, 2, 6, 11, 6, 12, 9, 6, 2, 7, 11, 6, 6, 2]
numb_tensor2 = torch.tensor(numb_2)

numb_tensor = [1, 2, 3, 5, 6, 7, 8, 9, 11, 13, 17, 18]

# 创建初始化张量
tensor_0 = torch.tensor([0.6, 0.4]).repeat(20, 1)
tensor_1 = torch.tensor([0.8, 0.2]).repeat(20, 1)
tensor_w = torch.stack([tensor_0, tensor_1], dim=1)


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model_conv_proj(state_dict or model.state_dict(), cfg)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class MLCPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx_pos = cfg.TRAINER.COOP_MLC.N_CTX_POS
        n_ctx_neg = cfg.TRAINER.COOP_MLC.N_CTX_NEG
        n_ctx_evi = cfg.TRAINER.COOP_MLC.N_CTX_EVI

        ctx_init_pos = cfg.TRAINER.COOP_MLC.POSITIVE_PROMPT_INIT.strip()
        ctx_init_neg = cfg.TRAINER.COOP_MLC.NEGATIVE_PROMPT_INIT.strip()
        ctx_init_evi = cfg.TRAINER.COOP_MLC.EVI_PROMPT_INIT.strip()

        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if ctx_init_pos and ctx_init_neg:
            # use given words to initialize context vectors
            ctx_init_pos = ctx_init_pos.replace("_", " ")
            ctx_init_neg = ctx_init_neg.replace("_", " ")
            n_ctx_pos = len(ctx_init_pos.split(" "))
            n_ctx_neg = len(ctx_init_neg.split(" "))
            prompt_pos = clip.tokenize(ctx_init_pos)
            prompt_neg = clip.tokenize(ctx_init_neg)
            with torch.no_grad():
                embedding_pos = clip_model.token_embedding(prompt_pos).type(dtype)
                embedding_neg = clip_model.token_embedding(prompt_neg).type(dtype)
            ctx_vectors_pos = embedding_pos[0, 1: 1 + n_ctx_pos, :]
            ctx_vectors_neg = embedding_neg[0, 1: 1 + n_ctx_neg, :]
            prompt_prefix_pos = ctx_init_pos
            prompt_prefix_neg = ctx_init_neg
            if cfg.TRAINER.COOP_MLC.CSC:
                ctx_vectors_pos_ = []
                ctx_vectors_neg_ = []
                for _ in range(n_cls):
                    ctx_vectors_pos_.append(deepcopy(ctx_vectors_pos))
                    ctx_vectors_neg_.append(deepcopy(ctx_vectors_neg))
                ctx_vectors_pos = torch.stack(ctx_vectors_pos_, dim=0)
                ctx_vectors_neg = torch.stack(ctx_vectors_neg_, dim=0)

        else:
            # Random Initialization
            if cfg.TRAINER.COOP_MLC.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors_pos = torch.empty(n_cls, n_ctx_pos, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(n_cls, n_ctx_neg, ctx_dim, dtype=dtype)
                ctx_vectors_evi = torch.empty(n_cls, n_ctx_evi, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors_pos = torch.empty(n_ctx_pos, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(n_ctx_neg, ctx_dim, dtype=dtype)
                ctx_vectors_evi = torch.empty(n_ctx_evi, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors_pos, std=0.02)
            nn.init.normal_(ctx_vectors_neg, std=0.02)
            nn.init.normal_(ctx_vectors_evi, std=0.02)
            prompt_prefix_pos = " ".join(["X"] * n_ctx_pos)
            prompt_prefix_neg = " ".join(["X"] * n_ctx_neg)
            prompt_prefix_evi = " ".join(["X"] * n_ctx_evi)

        print(f'Initial positive context: "{prompt_prefix_pos}"')
        print(f'Initial negative  context: "{prompt_prefix_neg}"')
        print(f'Initial evi  context: "{prompt_prefix_evi}"')
        print(f"Number of positive context words (tokens): {n_ctx_pos}")
        print(f"Number of negative context words (tokens): {n_ctx_neg}")
        print(f"Number of evi context words (tokens): {n_ctx_evi}")

        self.ctx_pos = nn.Parameter(ctx_vectors_pos)  # to be optimized
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)  # to be optimized
        self.ctx_evi = nn.Parameter(ctx_vectors_evi)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]

        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts_pos = [prompt_prefix_pos + " " + name + "." for name in classnames]
        prompts_neg = [prompt_prefix_neg + " " + name + "." for name in classnames]
        prompts_evi = [prompt_prefix_evi + " " + name + "." for name in classnames]

        tokenized_prompts_pos = []
        tokenized_prompts_neg = []
        tokenized_prompts_evi = []

        for p_pos, p_neg, p_evi in zip(prompts_pos, prompts_neg, prompts_evi):
            tokenized_prompts_pos.append(clip.tokenize(p_pos))
            tokenized_prompts_neg.append(clip.tokenize(p_neg))
            tokenized_prompts_evi.append(clip.tokenize(p_evi))


        tokenized_prompts_pos = torch.cat(tokenized_prompts_pos)
        tokenized_prompts_neg = torch.cat(tokenized_prompts_neg)
        tokenized_prompts_evi = torch.cat(tokenized_prompts_evi)

        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(dtype)
            embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(dtype)
            embedding_evi = clip_model.token_embedding(tokenized_prompts_evi).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix_pos", embedding_pos[:, :1, :])
        self.register_buffer("token_suffix_pos", embedding_pos[:, 1 + n_ctx_pos:, :])
        self.register_buffer("token_prefix_neg", embedding_neg[:, :1, :])
        self.register_buffer("token_suffix_neg", embedding_neg[:, 1 + n_ctx_neg:, :])
        self.register_buffer("token_prefix_evi", embedding_evi[:, :1, :])
        self.register_buffer("token_suffix_evi", embedding_evi[:, 1 + n_ctx_evi:, :])

        self.n_cls = n_cls
        self.n_ctx_pos = n_ctx_pos
        self.n_ctx_neg = n_ctx_neg
        self.n_ctx_evi = n_ctx_evi

        tokenized_prompts = torch.cat(
            [tokenized_prompts_neg, tokenized_prompts_pos, tokenized_prompts_evi],
            dim=0)  # torch.Tensor
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.name_lens = name_lens

    def forward(self, cls_id=None):
        ctx_pos = self.ctx_pos
        ctx_neg = self.ctx_neg
        ctx_evi = self.ctx_evi

        if ctx_pos.dim() == 2:
            if cls_id is None:
                ctx_pos = ctx_pos.unsqueeze(0).expand(self.n_cls, -1, -1)
            else:
                ctx_pos = ctx_pos.unsqueeze(0).expand(len(cls_id), -1, -1)
        else:
            if cls_id is not None:
                ctx_pos = ctx_pos[cls_id]

        if ctx_neg.dim() == 2:
            if cls_id is None:
                ctx_neg = ctx_neg.unsqueeze(0).expand(self.n_cls, -1, -1)
            else:
                ctx_neg = ctx_neg.unsqueeze(0).expand(len(cls_id), -1, -1)
        else:
            if cls_id is not None:
                ctx_neg = ctx_neg[cls_id]

        if ctx_evi.dim() == 2:
            if cls_id is None:
                ctx_evi = ctx_evi.unsqueeze(0).expand(self.n_cls, -1, -1)
            else:
                ctx_evi = ctx_evi.unsqueeze(0).expand(len(cls_id), -1, -1)
        else:
            if cls_id is not None:
                ctx_evi = ctx_evi[cls_id]

        if cls_id is None:
            prefix_pos = self.token_prefix_pos
            prefix_neg = self.token_prefix_neg
            prefix_evi = self.token_prefix_evi
            suffix_pos = self.token_suffix_pos
            suffix_neg = self.token_suffix_neg
            suffix_evi = self.token_suffix_evi
        else:
            prefix_pos = self.token_prefix_pos[cls_id]
            prefix_neg = self.token_prefix_neg[cls_id]
            prefix_evi = self.token_prefix_evi[cls_id]
            suffix_pos = self.token_suffix_pos[cls_id]
            suffix_neg = self.token_suffix_neg[cls_id]
            suffix_evi = self.token_suffix_evi[cls_id]

        prompts_pos = torch.cat(
            [
                prefix_pos,  # (n_cls, 1, dim)
                ctx_pos,  # (n_cls, n_ctx, dim)
                suffix_pos,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        prompts_neg = torch.cat(
            [
                prefix_neg,  # (n_cls, 1, dim)   20 1 512
                ctx_neg,  # (n_cls, n_ctx, dim)  20 12 512
                suffix_neg,  # (n_cls, *, dim)   20 64 512
            ],
            dim=1,
        )

        prompts_evi = torch.cat(
            [
                prefix_evi,  # (n_cls, 1, dim)
                ctx_evi,  # (n_cls, n_ctx, dim)
                suffix_evi,  # (n_cls, *, dim)
            ],
            dim=1,
        )


        prompts = torch.cat([prompts_neg, prompts_pos, prompts_evi], dim=0)

        if cls_id is not None:
            tokenized_prompts_pos = self.tokenized_prompts[self.n_cls:self.n_cls * 2][cls_id]
            tokenized_prompts_neg = self.tokenized_prompts[:self.n_cls][cls_id]
            tokenized_prompts_evi = self.tokenized_prompts[self.n_cls * 2: self.n_cls * 3][cls_id]
            tokenized_prompts = torch.cat(
                [tokenized_prompts_neg, tokenized_prompts_pos, tokenized_prompts_evi], dim=0)
        else:
            tokenized_prompts = self.tokenized_prompts

        return prompts, tokenized_prompts


class DPCoop(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.visual_encoder_type = cfg.MODEL.BACKBONE.NAME
        self.prompt_learner = MLCPromptLearner(cfg, classnames, clip_model)

        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = cfg.TRAINER.COOP_MLC.LS
        self.dtype = clip_model.dtype
        self.cfg = cfg
        # self.w = nn.Parameter(torch.tensor(0.8))

    def forward(self, image, cls_id=None):
        # get image and text features
        image_features, coarse_img_fea, attn_weights, _ = self.image_encoder(image.type(self.dtype))
        prompts, tokenized_prompts = self.prompt_learner(cls_id)
        text_features = self.text_encoder(prompts, tokenized_prompts)

        # normalize features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features_norm = image_features / image_features.norm(dim=1, keepdim=True)
        image_features_norm_coarse = coarse_img_fea / coarse_img_fea.norm(dim=1, keepdim=True)

        # Class-Specific Region Feature Aggregation
        output = 20 * F.conv1d(image_features_norm, text_features[:, :, None])
        output_coarse = 20 * F.conv1d(image_features_norm_coarse, text_features[:, :, None])
        b, c, _ = output.shape
        output_evi = output[:, (c * 2) // 3:]  # evi
        # output_evi = self.generate_evi_mask(output_evi, 5)
        pos = output[:, c // 3: (c * 2) // 3]
        neg_all = output[:, :c // 3]

        output_evi_coarse = output_coarse[:, (c * 2) // 3:]  # evi
        # output_evi = self.generate_evi_mask(output_evi, 5)
        pos_coarse = output_coarse[:, c // 3: (c * 2) // 3]
        neg_all_coarse = output_coarse[:, :c // 3]

        # neg = 0.5 * pos + 0.5 * neg_all
        # ood = output[:, c:]  # (8 1 197)
        # ood_ex = ood.expand(-1, 20, -1)
        # neg_ood = torch.cat((neg_all, ood), dim=1)

        # topk_values, topk_indices = torch.topk(neg_all, k=2, dim=1)
        # mask = (topk_indices[:, 0, :].unsqueeze(1) == torch.arange(neg_all.size(1), device=neg_all.device).view(1, -1,
        #                                                                                                         1))
        # neg = torch.where(mask, topk_values[:, 1, :].unsqueeze(1),
        #                   topk_values[:, 0, :].unsqueeze(1))
        all_my = 0.2 * pos + 0.8 * neg_all
        all_my_coarse = 0.2 * pos_coarse + 0.8 * neg_all_coarse

        neg = self.compute_max(all_my)
        neg_coarse = self.compute_max(all_my_coarse)

        pos = 0.8 * pos + 0.2 * neg_all
        pos_coarse = 0.8 * pos_coarse + 0.2 * neg_all_coarse

        w_half = F.softmax(output_evi, dim=-1)
        w_half_coarse = F.softmax(output_evi_coarse, dim=-1)
        w = torch.cat([w_half, w_half], dim=1)
        # p = pos * mask
        w_coarse = torch.cat([w_half_coarse, w_half_coarse], dim=1)

        output = torch.cat([neg, pos], dim=1)
        output_coarse = torch.cat([neg_coarse, pos_coarse], dim=1)
        output = 1.1 * (output * w).sum(-1)
        output_coarse = 1.1 * (output_coarse * w_coarse).sum(-1)

        b, c = output.shape

        # convert the shape of logits to [b, 2, num_class]
        logits = output.resize(b, 2, c // 2)
        logits_coarse = output_coarse.resize(b, 2, c // 2)

        return logits,  logits_coarse

    @property
    def network_name(self):
        name = ''
        name += 'DualCoop-{}'.format(self.visual_encoder_type)
        return name

    def backbone_params(self):
        params = []
        for name, param in self.named_parameters():
            if "image_encoder" in name and "prompt_learner" not in name and 'attnpool' not in name:
                params.append(param)
        return params

    def compute_modified_max1(self, neg_all):
        """
        计算修正后的最大值，使用8/2准则：
        1. 取每个位置的前三大值(A,B,C)
        2. 如果当前元素在前三大中，则找到另外两个元素
        3. 对另外两个元素排序，将较大的乘以0.8，较小的乘以0.2，然后求和
        4. 如果当前元素不在前三大中，则取前两大值，将较大的乘以0.8，较小的乘以0.2，然后求和

        Args:
            neg_all: 输入张量，形状为(batch_size, seq_len, feature_dim)
        Returns:
            修正后的张量，形状与输入相同
        """
        batch_size, seq_len, feature_dim = neg_all.shape

        # 1. 获取前三大值和索引 [形状(batch_size, 3, feature_dim)]
        top3_values, top3_indices = torch.topk(neg_all, k=3, dim=1)
        A = top3_values[:, 0]  # 最大值
        B = top3_values[:, 1]  # 第二值
        C = top3_values[:, 2]  # 第三值

        # 2. 准备当前索引张量 [形状(batch_size, seq_len, feature_dim, 1)]
        current_indices = torch.arange(seq_len, device=neg_all.device
                                       ).view(1, seq_len, 1, 1).expand(batch_size, seq_len, feature_dim, 1)

        # 3. 扩展top3索引用于比较 [形状(batch_size, seq_len, feature_dim, 3)]
        top3_indices_expanded = top3_indices.unsqueeze(1)  # 添加seq_len维度 [batch,1,3,feat]
        top3_indices_expanded = top3_indices_expanded.expand(-1, seq_len, -1, -1)  # 扩展seq_len
        top3_indices_expanded = top3_indices_expanded.permute(0, 1, 3, 2)  # 调整为[batch,seq,feat,3]

        # 4. 判断当前元素在前三大中的具体位置 [形状(batch_size, seq_len, feature_dim, 3)]
        position_in_top3 = (top3_indices_expanded == current_indices)

        # 5. 获取每个位置是否是A、B或C
        is_A = position_in_top3[:, :, :, 0]  # 当前元素是最大值A
        is_B = position_in_top3[:, :, :, 1]  # 当前元素是第二值B
        is_C = position_in_top3[:, :, :, 2]  # 当前元素是第三值C
        is_in_top3 = is_A | is_B | is_C  # 当前元素是否在前三大中

        # 6. 准备A、B、C的扩展版本用于计算
        A_expanded = A.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq, feat]
        B_expanded = B.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq, feat]
        C_expanded = C.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq, feat]

        # 7. 应用8/2准则
        # Case 1: 当前元素是A (最大值) - 使用B和C，较大的乘以0.8，较小的乘以0.2
        # 比较B和C的大小
        B_gt_C = B_expanded > C_expanded
        case_A = torch.where(B_gt_C,
                            0.8 * B_expanded + 0.2 * C_expanded,
                             0.8 * C_expanded + 0.2 * B_expanded)

        # Case 2: 当前元素是B (第二值) - 使用A和C，较大的乘以0.8，较小的乘以0.2
        # A总是大于C，所以直接使用A和C
        case_B = 0.8 * A_expanded + 0.2 * C_expanded

        # Case 3: 当前元素是C (第三值) - 使用A和B，较大的乘以0.8，较小的乘以0.2
        # A总是大于B，所以直接使用A和B
        case_C = 0.8 * A_expanded + 0.2 * B_expanded

        # Case 4: 当前元素不在前三大中 - 使用A和B，较大的乘以0.8，较小的乘以0.2
        # A总是大于B，所以直接使用A和B
        case_not_top3 =  0.8 * A_expanded + 0.2 * B_expanded

        # 8. 组合结果
        result = torch.zeros_like(neg_all)
        result = torch.where(is_A, case_A, result)
        result = torch.where(is_B, case_B, result)
        result = torch.where(is_C, case_C, result)
        result = torch.where(~is_in_top3, case_not_top3, result)

        return result
    def attn_params(self):
        params = []
        for name, param in self.named_parameters():
            if 'attnpool' in name and 'image_encoder' in name:
                params.append(param)
                print(name)
        return params

    def prompt_params(self):
        params = []
        for name, param in self.named_parameters():
            if "prompt_learner" in name:
                params.append(param)
        return params

    def visualize(self, image, cls_id=None):
        image_features, attn_weights = self.image_encoder(image.type(self.dtype))
        prompts, tokenized_prompts = self.prompt_learner(cls_id)
        text_features = self.text_encoder(prompts, tokenized_prompts)

        # normalize features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features_norm = image_features / image_features.norm(dim=1, keepdim=True)

        # Class-Specific Region Feature Aggregation
        output = 20 * F.conv1d(image_features_norm, text_features[:, :, None])
        return output

    def keep_topk_values_vectorized(self, pos, k=20):
        """
        高效保留每个特征图中前k个最大值，其余置0
        参数:
            pos: 形状为 (batch, seq_len, num_maps) 的张量
            k: 要保留的最大值数量
        返回:
            处理后的张量
        """
        # 获取前k个最大值及其索引 (形状都是 [batch, k, num_maps])
        topk_values, topk_indices = torch.topk(pos, k=k, dim=1)

        # 创建全零张量并填充topk值
        result = torch.zeros_like(pos)
        result.scatter_(dim=1, index=topk_indices, src=topk_values)

        return result

    def compute_max(self, neg_all):
        topk_values, topk_indices = torch.topk(neg_all, k=2, dim=1)
        mask = (topk_indices[:, 0, :].unsqueeze(1) == torch.arange(neg_all.size(1), device=neg_all.device).view(1, -1,
                                                                                                                1))
        neg = torch.where(mask, topk_values[:, 1, :].unsqueeze(1),
                          topk_values[:, 0, :].unsqueeze(1))
        return neg

    def compute_modified_max(self, neg_all):
        """
        计算修正后的最大值：
        1. 取每个位置的前三大值(A,B,C)
        2. 如果当前元素在前三大中，则取另外两个的平均
        3. 否则取A和B的平均

        Args:
            neg_all: 输入张量，形状为(batch_size, seq_len, feature_dim)
        Returns:
            修正后的张量，形状与输入相同
        """
        batch_size, seq_len, feature_dim = neg_all.shape

        # 1. 获取前三大值和索引 [形状(batch_size, 3, feature_dim)]
        top3_values, top3_indices = torch.topk(neg_all, k=3, dim=1)

        # 2. 准备当前索引张量 [形状(batch_size, seq_len, feature_dim, 1)]
        current_indices = torch.arange(seq_len, device=neg_all.device
                                       ).view(1, seq_len, 1, 1).expand(batch_size, seq_len, feature_dim, 1)

        # 3. 扩展top3索引用于比较 [形状(batch_size, seq_len, feature_dim, 3)]
        top3_indices_expanded = top3_indices.unsqueeze(1)  # 添加seq_len维度 [batch,1,3,feat]
        top3_indices_expanded = top3_indices_expanded.expand(-1, seq_len, -1, -1)  # 扩展seq_len
        top3_indices_expanded = top3_indices_expanded.permute(0, 1, 3, 2)  # 调整为[batch,seq,feat,3]

        # 4. 判断当前元素是否在前三大中 [形状(batch_size, seq_len, feature_dim)]
        is_in_top3 = (top3_indices_expanded == current_indices).any(dim=-1)

        # 5. 准备计算case1: 当前在top3中，取另外两个的平均
        # 创建mask标记非当前元素 [形状(batch_size, seq_len, feature_dim, 3)]
        case1_mask = (top3_indices_expanded != current_indices)

        # 复制top3_values并应用mask [形状(batch_size, seq_len, feature_dim, 3)]
        case1_values = top3_values.unsqueeze(1).expand(-1, seq_len, -1, -1)  # [batch,seq,3,feat]
        case1_values = case1_values.permute(0, 1, 3, 2)  # [batch,seq,feat,3]
        case1_values = torch.where(case1_mask, case1_values, torch.nan)

        # 计算非nan值的平均 [形状(batch_size, seq_len, feature_dim)]
        case1_avg = torch.nanmean(case1_values, dim=-1)

        # 6. 准备计算case2: 当前不在top3中，取A和B的平均
        case2_avg = top3_values[:, :2].mean(dim=1, keepdim=True).expand(-1, seq_len, -1)

        # 7. 组合结果
        result = torch.where(is_in_top3, case1_avg, case2_avg)

        return result


def dpcoop(cfg, classnames, **kwargs):
    print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
    clip_model = load_clip_to_cpu(cfg)

    clip_model.float()

    print("Building dualcoop")
    model = DPCoop(cfg, classnames, clip_model)

    if not cfg.TRAINER.FINETUNE_BACKBONE:
        print('Freeze the backbone weights')
        backbone_params = model.backbone_params()
        for param in backbone_params:
            param.requires_grad_(False)

    if not cfg.TRAINER.FINETUNE_ATTN:
        print('Freeze the attn weights')
        attn_params = model.attn_params()
        for param in attn_params:
            param.requires_grad_(False)

    for param in model.text_encoder.parameters():
        param.requires_grad_(False)

    # 检查
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"参数名称: {name}, 形状: {param.shape}, 参数量: {param.numel()}")

    if torch.cuda.is_available() and cfg.USE_CUDA:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    # Note that multi-gpu training could be slow because CLIP's size is
    # big, which slows down the copy operation in DataParallel
    device_count = torch.cuda.device_count()
    if device_count > 1:
        print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        model = nn.DataParallel(model)
    return model
