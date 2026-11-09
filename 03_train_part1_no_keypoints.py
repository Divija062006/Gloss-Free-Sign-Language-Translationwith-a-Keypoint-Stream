# =============================================================================
# Step 3 of 5 - Training part 1: models without keypoints (E1, E3)
# Project : Gloss-free sign language translation with a keypoint stream
# Team    : Claude5
#
# Reimplementation of GFSLT-VLP on pre-extracted ResNet18 features:
#   E1  GFSLT      : translation only
#   E3  GFSLT-VLP  : visual-language pretraining (contrastive + masked sentence
#                    restoration, lambda = 0.1), then translation
# Glosses are never used. Each model is evaluated on dev and test with beam
# search (beam 5); BLEU-1..4 and ROUGE-L are reported.
#
# Differences from the paper (single free GPU): frozen ResNet18, feature-level
# augmentation, word-level decoder trained from scratch (no mBART), XLM-R text
# encoder for VLP, width 512, AdamW, 100 translation / 60 VLP epochs.
#
# Kaggle  : Accelerator = GPU T4, Internet = On
# Input   : output of step 1 (01_resnet_features)
# Output  : /kaggle/working/runs/ - results.json, checkpoints, curves,
#           translations for dev and test
# =============================================================================

# ======================= settings =======================
QUICK_TEST = False       # True: short debugging run on a small subset

EXPERIMENTS = [   # models trained in this script
    dict(name="E1_rgb",        use_rgb=True,  use_kp=False, vlp=False),
    dict(name="E3_rgb_vlp",    use_rgb=True,  use_kp=False, vlp=True),
]
SEEDS = [0]              # random seeds; e.g. [0, 1, 2] to report mean ± std

CFG = dict(
    d_model=512, n_heads=8, enc_layers=3, dec_layers=3, ff=2048, dropout=0.3, kp_emb=256,
    batch_size=32, lr=3e-4, text_lr=2e-5, weight_decay=1e-3, warmup_frac=0.05, grad_clip=1.0,
    slt_epochs=100, vlp_epochs=60, eval_every=5,
    label_smoothing=0.2,                   # paper value
    mask_rate=0.15, vlp_lambda=0.1,        # paper values (Appendix B.2, B.4)
    aug_vlp=True, aug_slt=True,            # paper found augmentation in both stages works best
    beam=5, length_penalty=1.0, max_dec_len=60,   # paper decoding settings
    text_encoder="xlm-roberta-base",       # pretrained multilingual text encoder for VLP; "scratch" = train one from zero
    num_workers=2,
)
OUT_DIR = "/kaggle/working/runs"
if QUICK_TEST:
    CFG.update(slt_epochs=2, vlp_epochs=1, eval_every=1, beam=2)

import os, math, json, time, random, pickle, collections
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "| torch", torch.__version__)
os.makedirs(OUT_DIR, exist_ok=True)
SPLITS = ["train", "dev", "test"]

def find_file(fname, roots=("/kaggle/input", "/kaggle/working")):
    for root in roots:
        for dp, dn, fn in os.walk(root):
            dn[:] = [d for d in dn if d not in ("train", "dev", "test") and not d.startswith("fullFrame")]
            if fname in fn:
                return os.path.join(dp, fname)
    return None

NEED_KP = any(e["use_kp"] for e in EXPERIMENTS)
NEED_RGB = any(e["use_rgb"] for e in EXPERIMENTS)
paths = {"csv_" + s: find_file(f"PHOENIX-2014-T.{s}.corpus.csv") for s in SPLITS}
if NEED_RGB: paths.update({"rgb_" + s: find_file(f"resnet18_{s}.pkl") for s in SPLITS})
if NEED_KP:  paths.update({"kp_" + s: find_file(f"kp_{s}.pkl") for s in SPLITS})
for k, v in paths.items(): print(f"{k:10s} {v}")
missing = [k for k, v in paths.items() if v is None]
assert not missing, f"Missing files {missing}: attach the output datasets of notebooks 1 and 2."

# Layout written by 02_keypoints_mediapipe.py: 0-10 body, 11-31 first hand, 32-52 second hand, 53-74 face.
N_KP = 75
L_SHOULDER, R_SHOULDER = 5, 6
KP_DIM = N_KP * 3        # x, y, detected-flag per point

def preprocess_kp(raw):
    # raw: T x 75 x 3 (pixel x, pixel y, detected). Returns xy (T x 75 x 2) normalised, valid (T x 75).
    raw = raw.astype(np.float32)
    T = raw.shape[0]
    xy = raw[..., :2].copy()
    valid = raw[..., 2] > 0.5
    t = np.arange(T)
    for j in range(raw.shape[1]):              # fill missed detections by linear interpolation in time
        v = valid[:, j]
        if v.all():
            continue
        if not v.any():
            xy[:, j] = np.nan
            continue
        for c in range(2):
            xy[:, j, c] = np.interp(t, t[v], xy[v, j, c])
    sh = valid[:, L_SHOULDER] & valid[:, R_SHOULDER]
    if sh.any():                               # centre on the shoulders, scale by shoulder width
        center = np.median((xy[sh, L_SHOULDER] + xy[sh, R_SHOULDER]) / 2, axis=0)
        scale = float(np.median(np.linalg.norm(xy[sh, L_SHOULDER] - xy[sh, R_SHOULDER], axis=1)))
    else:
        fin = np.isfinite(xy).all(-1)
        center = np.median(xy[fin], axis=0) if fin.any() else np.zeros(2, np.float32)
        scale = 100.0
    xy = (xy - center) / max(scale, 1e-3)
    xy = np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)
    return xy.astype(np.float32), valid.astype(np.float32)

def augment_kp(xy, rng):
    a = np.deg2rad(rng.uniform(-10, 10))
    s = rng.uniform(0.9, 1.1)
    M = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32) * np.array([s * rng.uniform(0.95, 1.05), s], np.float32)
    xy = xy @ M.T + rng.uniform(-0.1, 0.1, size=2).astype(np.float32)
    return xy + rng.normal(0, 0.01, size=xy.shape).astype(np.float32)

def temporal_resample(T, rng, lo=0.8, hi=1.2):
    # random speed change (0.8x-1.2x) with small jitter: the feature-level stand-in for video augmentation
    newT = max(8, int(round(T * rng.uniform(lo, hi))))
    idx = np.linspace(0, T - 1, newT) + rng.uniform(-0.5, 0.5, newT)
    return np.sort(np.clip(np.round(idx), 0, T - 1).astype(int))

def _ngrams(toks, n):
    return collections.Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))

def corpus_bleu(refs, hyps, max_order=4):
    # refs, hyps: lists of token lists. Corpus-level BLEU, uniform weights, no smoothing, x100.
    match, possible = [0] * max_order, [0] * max_order
    ref_len = hyp_len = 0
    for r, h in zip(refs, hyps):
        ref_len += len(r); hyp_len += len(h)
        for n in range(1, max_order + 1):
            match[n - 1] += sum((_ngrams(r, n) & _ngrams(h, n)).values())
            possible[n - 1] += max(len(h) - n + 1, 0)
    if hyp_len == 0 or min(match) == 0:
        return 0.0
    log_p = sum(math.log(m / p) for m, p in zip(match, possible)) / max_order
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    return 100.0 * bp * math.exp(log_p)

def _lcs(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]

def rouge_l(refs, hyps, beta=1.2):
    scores = []
    for r, h in zip(refs, hyps):
        l = _lcs(r, h) if r and h else 0
        if l == 0:
            scores.append(0.0); continue
        p, rc = l / len(h), l / len(r)
        scores.append((1 + beta ** 2) * p * rc / (rc + beta ** 2 * p))
    return 100.0 * sum(scores) / max(len(scores), 1)

def all_metrics(refs, hyps):
    m = {f"B{n}": round(corpus_bleu(refs, hyps, n), 2) for n in range(1, 5)}
    m["ROUGE"] = round(rouge_l(refs, hyps), 2)
    return m

PAD, BOS, EOS, UNK, MASK = 0, 1, 2, 3, 4
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>", "<mask>"]

def clean(sentence):
    return [w for w in str(sentence).strip().split() if w != "."]

class Vocab:
    def __init__(self, sentences):
        cnt = collections.Counter(w for s in sentences for w in s)
        self.itos = SPECIALS + sorted(cnt)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
    def encode(self, words): return [self.stoi.get(w, UNK) for w in words]
    def decode(self, ids):   return [self.itos[i] for i in ids if i > MASK]
    def __len__(self): return len(self.itos)

def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)

DATA, NAMES = {}, {}
t0 = time.time()
for s in SPLITS:
    corpus = pd.read_csv(paths["csv_" + s], sep="|")
    rgb = load_pkl(paths["rgb_" + s]) if NEED_RGB else {}
    kp = load_pkl(paths["kp_" + s]) if NEED_KP else {}
    names = []
    for name, sent in zip(corpus["name"], corpus["translation"]):
        if (NEED_RGB and name not in rgb) or (NEED_KP and name not in kp):
            continue     # every experiment uses exactly the same videos
        d = {"words": clean(sent)}
        T = []
        if NEED_RGB: d["rgb"] = rgb[name]; T.append(len(d["rgb"]))
        if NEED_KP:
            d["kpxy"], d["kpv"] = preprocess_kp(kp[name]); T.append(len(d["kpxy"]))
            d["kpxy"] = d["kpxy"].astype(np.float16); d["kpv"] = d["kpv"].astype(np.uint8)
        T = min(T)
        for k in ("rgb", "kpxy", "kpv"):
            if k in d: d[k] = d[k][:T]
        DATA[name] = d
        names.append(name)
    NAMES[s] = names
    print(f"{s:5s}: {len(names)} of {len(corpus)} videos usable")
    del rgb, kp

if QUICK_TEST:
    NAMES = {"train": NAMES["train"][:256], "dev": NAMES["dev"][:48], "test": NAMES["test"][:48]}

VOCAB = Vocab([DATA[n]["words"] for n in NAMES["train"]])
for n in DATA: DATA[n]["ids"] = VOCAB.encode(DATA[n]["words"])
print(f"vocabulary: {len(VOCAB)} words | loaded in {time.time() - t0:.0f}s")
print("example:", NAMES["dev"][0], "->", " ".join(DATA[NAMES["dev"][0]]["words"]))

class SignDataset(Dataset):
    def __init__(self, names, use_rgb, use_kp, augment):
        self.names, self.use_rgb, self.use_kp, self.augment = names, use_rgb, use_kp, augment
    def __len__(self): return len(self.names)
    def __getitem__(self, i):
        name = self.names[i]
        d = DATA[name]
        T = len(d["rgb"]) if "rgb" in d else len(d["kpxy"])
        rng = np.random.default_rng([torch.initial_seed() % (2 ** 32), i])
        idx = temporal_resample(T, rng) if self.augment else np.arange(T)
        out = {"name": name, "words": d["words"], "ids": d["ids"], "rgb": None, "kp": None}
        if self.use_rgb:
            x = d["rgb"][idx].astype(np.float32)
            if self.augment:                       # feature dropout
                x = x * (rng.random(x.shape) > 0.1) / 0.9
            out["rgb"] = x
        if self.use_kp:
            xy = d["kpxy"][idx].astype(np.float32)
            if self.augment:
                xy = augment_kp(xy, rng)
            out["kp"] = np.concatenate([xy.reshape(len(idx), -1), d["kpv"][idx].astype(np.float32)], 1)
        return out

def collate(batch):
    B = len(batch)
    lens = [len(b["rgb"] if b["rgb"] is not None else b["kp"]) for b in batch]
    Tm = int(math.ceil(max(lens) / 4) * 4)           # two /2 poolings later
    mask = torch.zeros(B, Tm, dtype=torch.bool)
    rgb = torch.zeros(B, Tm, batch[0]["rgb"].shape[1]) if batch[0]["rgb"] is not None else None
    kp = torch.zeros(B, Tm, KP_DIM) if batch[0]["kp"] is not None else None
    U = max(len(b["ids"]) for b in batch) + 2
    tokens = torch.full((B, U), PAD, dtype=torch.long)
    for i, (b, L) in enumerate(zip(batch, lens)):
        mask[i, :L] = True
        if rgb is not None: rgb[i, :L] = torch.from_numpy(b["rgb"])
        if kp is not None:  kp[i, :L] = torch.from_numpy(b["kp"])
        ids = [BOS] + b["ids"] + [EOS]
        tokens[i, :len(ids)] = torch.tensor(ids)
    return {"rgb": rgb, "kp": kp, "mask": mask, "tokens": tokens,
            "words": [b["words"] for b in batch], "names": [b["name"] for b in batch]}

def make_loader(split, exp, augment, shuffle):
    ds = SignDataset(NAMES[split], exp["use_rgb"], exp["use_kp"], augment)
    return DataLoader(ds, batch_size=CFG["batch_size"], shuffle=shuffle, collate_fn=collate,
                      num_workers=CFG["num_workers"], drop_last=shuffle, pin_memory=(device == "cuda"))

def to_dev(batch):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=4000):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe = torch.zeros(max_len, d)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)
    def forward(self, x):
        return x + self.pe[: x.size(1)].unsqueeze(0).to(x.dtype)

def conv_block(i, o):
    return nn.Sequential(nn.Conv1d(i, o, 5, padding=2), nn.BatchNorm1d(o), nn.ReLU(inplace=True), nn.MaxPool1d(2, 2))

class VisualEncoder(nn.Module):
    def __init__(self, cfg, use_rgb, use_kp, rgb_dim=512):
        super().__init__()
        d = cfg["d_model"]
        self.use_rgb, self.use_kp = use_rgb, use_kp
        in_dim = 0
        if use_rgb:
            in_dim += rgb_dim
        if use_kp:
            e = cfg["kp_emb"]
            self.kp_mlp = nn.Sequential(nn.Linear(KP_DIM, e), nn.LayerNorm(e), nn.ReLU(inplace=True),
                                        nn.Dropout(0.1), nn.Linear(e, e), nn.ReLU(inplace=True))
            in_dim += e
        self.in_drop = nn.Dropout(0.1)
        self.temporal = nn.Sequential(conv_block(in_dim, d), conv_block(d, d))
        self.proj, self.proj_bn = nn.Linear(d, d), nn.BatchNorm1d(d)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pe, self.drop = PositionalEncoding(d), nn.Dropout(cfg["dropout"])
        layer = nn.TransformerEncoderLayer(d, cfg["n_heads"], cfg["ff"], cfg["dropout"],
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg["enc_layers"], enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, rgb, kp, mask):
        parts = []
        if self.use_rgb: parts.append(rgb)
        if self.use_kp:  parts.append(self.kp_mlp(kp))
        x = self.in_drop(torch.cat(parts, -1)) * mask.unsqueeze(-1).to(rgb.dtype if rgb is not None else kp.dtype)
        x = self.temporal(x.transpose(1, 2))                                   # B x d x T/4
        x = F.relu(self.proj_bn(self.proj(x.transpose(1, 2)).transpose(1, 2))).transpose(1, 2)   # B x T/4 x d
        B, M, _ = x.shape
        lens = (mask.sum(1) // 4).clamp(min=1)
        valid = torch.arange(M, device=x.device).unsqueeze(0) < lens.unsqueeze(1)
        x = torch.cat([self.cls.expand(B, -1, -1).to(x.dtype), self.pe(x)], 1)
        valid = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=x.device), valid], 1)
        h = self.norm(self.encoder(self.drop(x), src_key_padding_mask=~valid))
        return h, valid, h[:, 0]                 # memory, memory mask (True = real), [CLS]

class TextDecoder(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        d = cfg["d_model"]
        self.emb = nn.Embedding(vocab_size, d, padding_idx=PAD)
        nn.init.normal_(self.emb.weight, 0, d ** -0.5)
        with torch.no_grad(): self.emb.weight[PAD].zero_()
        self.scale = math.sqrt(d)
        self.pe, self.drop = PositionalEncoding(d), nn.Dropout(cfg["dropout"])
        layer = nn.TransformerDecoderLayer(d, cfg["n_heads"], cfg["ff"], cfg["dropout"],
                                           activation="gelu", batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, cfg["dec_layers"])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, vocab_size, bias=False)
        self.out.weight = self.emb.weight        # tied input/output embeddings

    def forward(self, tgt, memory, mem_valid):
        U = tgt.size(1)
        causal = torch.triu(torch.ones(U, U, dtype=torch.bool, device=tgt.device), 1)
        x = self.drop(self.pe(self.emb(tgt) * self.scale))
        h = self.dec(x, memory, tgt_mask=causal, tgt_key_padding_mask=(tgt == PAD),
                     memory_key_padding_mask=~mem_valid)
        return self.out(self.norm(h))

class SLTModel(nn.Module):
    def __init__(self, vocab_size, cfg, use_rgb, use_kp):
        super().__init__()
        self.visual = VisualEncoder(cfg, use_rgb, use_kp)
        self.decoder = TextDecoder(vocab_size, cfg)
    def forward(self, b):
        mem, valid, _ = self.visual(b["rgb"], b["kp"], b["mask"])
        return self.decoder(b["tokens"][:, :-1], mem, valid)

class ScratchTextEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["d_model"]
        self.emb = nn.Embedding(len(VOCAB), d, padding_idx=PAD)
        self.pe = PositionalEncoding(d)
        layer = nn.TransformerEncoderLayer(d, cfg["n_heads"], cfg["ff"], cfg["dropout"],
                                           activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 3, enable_nested_tensor=False)
        self.norm, self.out_dim = nn.LayerNorm(d), d
    def forward(self, word_lists):
        ids = [VOCAB.encode(ws) or [UNK] for ws in word_lists]
        x = torch.full((len(ids), max(map(len, ids))), PAD, dtype=torch.long, device=device)
        for i, s in enumerate(ids): x[i, :len(s)] = torch.tensor(s, device=device)
        valid = x != PAD
        h = self.norm(self.enc(self.pe(self.emb(x)), src_key_padding_mask=~valid))
        pooled = (h * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp(min=1)
        return h, valid, pooled

class HFTextEncoder(nn.Module):
    def __init__(self, name):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        m = AutoModel.from_pretrained(name)
        self.enc = m.get_encoder() if getattr(m.config, "is_encoder_decoder", False) else m
        self.out_dim = m.config.hidden_size
        self.mask_token = self.tok.mask_token or self.tok.unk_token
    def forward(self, word_lists):
        texts = [" ".join(self.mask_token if w == "<mask>" else w for w in ws) for ws in word_lists]
        t = self.tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        h = self.enc(input_ids=t["input_ids"], attention_mask=t["attention_mask"]).last_hidden_state
        valid = t["attention_mask"].bool()
        pooled = (h * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp(min=1)
        return h, valid, pooled

def build_text_encoder():
    name = CFG["text_encoder"]
    if name and name != "scratch":
        try:
            enc = HFTextEncoder(name)
            print("text encoder:", name)
            return enc
        except Exception as e:
            print(f"Could not load {name} ({e}); using a text encoder trained from scratch instead.")
    print("text encoder: scratch transformer")
    return ScratchTextEncoder(CFG)

def mask_words(words, rate, rng):
    # BERT-style: of the chosen words, 80% -> <mask>, 10% -> random word, 10% unchanged
    out = list(words)
    for i in range(len(out)):
        if rng.random() < rate:
            r = rng.random()
            if r < 0.8:   out[i] = "<mask>"
            elif r < 0.9: out[i] = VOCAB.itos[rng.randrange(len(SPECIALS), len(VOCAB))]
    return out

class VLPModel(nn.Module):
    def __init__(self, cfg, use_rgb, use_kp, text_encoder):
        super().__init__()
        d = cfg["d_model"]
        self.visual = VisualEncoder(cfg, use_rgb, use_kp)
        self.decoder = TextDecoder(len(VOCAB), cfg)
        self.text = text_encoder
        self.v_head, self.t_head = nn.Linear(d, d), nn.Linear(text_encoder.out_dim, d)
        self.mem_proj = nn.Linear(text_encoder.out_dim, d)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.rng = random.Random(0)

    def forward(self, b):
        _, _, cls = self.visual(b["rgb"], b["kp"], b["mask"])
        _, _, pooled = self.text(b["words"])
        v = F.normalize(self.v_head(cls).float(), dim=-1)
        t = F.normalize(self.t_head(pooled).float(), dim=-1)
        logits = self.logit_scale.exp().clamp(max=100) * v @ t.T
        keys = [tuple(w) for w in b["words"]]
        same = torch.tensor([[a == c for c in keys] for a in keys], dtype=torch.float32, device=logits.device)
        target = same / same.sum(1, keepdim=True)
        loss_s = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
        acc = (logits.argmax(1) == torch.arange(len(keys), device=logits.device)).float().mean()

        masked = [mask_words(w, CFG["mask_rate"], self.rng) for w in b["words"]]
        h, valid, _ = self.text(masked)
        out = self.decoder(b["tokens"][:, :-1], self.mem_proj(h.detach()), valid)   # paper Alg.1: L_c updates the decoder
        loss_c = F.cross_entropy(out.reshape(-1, out.size(-1)).float(), b["tokens"][:, 1:].reshape(-1), ignore_index=PAD)
        return loss_s, loss_c, acc

NEG = -1e9

@torch.no_grad()
def greedy(decoder, mem, valid, max_len):
    B = mem.size(0)
    ys = torch.full((B, 1), BOS, dtype=torch.long, device=mem.device)
    done = torch.zeros(B, dtype=torch.bool, device=mem.device)
    for _ in range(max_len):
        logits = decoder(ys, mem, valid)[:, -1].float()
        logits[:, PAD] = NEG; logits[:, BOS] = NEG
        nxt = logits.argmax(-1).masked_fill(done, PAD)
        ys = torch.cat([ys, nxt.unsqueeze(1)], 1)
        done |= nxt == EOS
        if done.all(): break
    out = []
    for row in ys[:, 1:].tolist():
        seq = []
        for w in row:
            if w in (EOS, PAD): break
            seq.append(w)
        out.append(seq)
    return out

@torch.no_grad()
def beam_search(decoder, mem, valid, beam, max_len, alpha):
    lp = lambda n: ((5 + n) / 6) ** alpha            # GNMT length penalty (paper ref. [35])
    alive, finished = [([BOS], 0.0)], []
    for _ in range(max_len):
        k = len(alive)
        seqs = torch.tensor([s for s, _ in alive], device=mem.device)
        logp = F.log_softmax(decoder(seqs, mem.expand(k, -1, -1), valid.expand(k, -1))[:, -1].float(), -1)
        logp[:, PAD] = NEG; logp[:, BOS] = NEG
        scores = torch.tensor([sc for _, sc in alive], device=mem.device).unsqueeze(1) + logp
        V = logp.size(1)
        top_s, top_i = scores.view(-1).topk(min(2 * beam, scores.numel()))
        new_alive = []
        for s, i in zip(top_s.tolist(), top_i.tolist()):
            b, w = divmod(i, V)
            seq = alive[b][0] + [w]
            if w == EOS:
                finished.append((seq, s / lp(len(seq) - 1)))
            else:
                new_alive.append((seq, s))
            if len(new_alive) == beam: break
        alive = new_alive
        if len(finished) >= beam or not alive: break
    if not finished:
        finished = [(s, sc / lp(len(s) - 1)) for s, sc in alive]
    best = max(finished, key=lambda x: x[1])[0]
    return [w for w in best[1:] if w not in (EOS, PAD)]

@torch.no_grad()
def translate(model, loader, beam):
    model.eval()
    names, refs, hyps = [], [], []
    for b in loader:
        b = to_dev(b)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            mem, valid, _ = model.visual(b["rgb"], b["kp"], b["mask"])
            if beam <= 1:
                out = greedy(model.decoder, mem, valid, CFG["max_dec_len"])
            else:
                out = [beam_search(model.decoder, mem[i:i + 1], valid[i:i + 1], beam,
                                   CFG["max_dec_len"], CFG["length_penalty"]) for i in range(mem.size(0))]
        names += b["names"]; refs += b["words"]; hyps += [VOCAB.decode(o) for o in out]
    return names, refs, hyps

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def make_optim(model, epochs, steps_per_epoch, text_params=()):
    text_ids = {id(p) for p in text_params}
    groups = [{"params": [p for p in model.parameters() if id(p) not in text_ids and p.requires_grad], "lr": CFG["lr"]}]
    if text_ids:
        groups.append({"params": [p for p in text_params if p.requires_grad], "lr": CFG["text_lr"]})
    opt = torch.optim.AdamW(groups, weight_decay=CFG["weight_decay"])
    total = max(1, epochs * steps_per_epoch)
    warm = max(1, int(CFG["warmup_frac"] * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: (st + 1) / warm if st < warm else 0.5 * (1 + math.cos(math.pi * (st - warm) / max(1, total - warm))))
    return opt, sched

def make_scaler():
    try:    return torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    except Exception: return torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

def step(loss, model, opt, sched, scaler):
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
    scaler.step(opt); scaler.update(); sched.step()

def run_vlp(exp, seed):
    set_seed(seed)
    path = os.path.join(OUT_DIR, f"{exp['name']}_s{seed}_vlp.pth")
    model = VLPModel(CFG, exp["use_rgb"], exp["use_kp"], build_text_encoder()).to(device)
    loader = make_loader("train", exp, CFG["aug_vlp"], shuffle=True)
    text_params = list(model.text.parameters()) if isinstance(model.text, HFTextEncoder) else []
    opt, sched = make_optim(model, CFG["vlp_epochs"], len(loader), text_params)
    scaler, hist, t0 = make_scaler(), [], time.time()
    for ep in range(1, CFG["vlp_epochs"] + 1):
        model.train(); ls = lc = ac = 0.0
        for b in loader:
            b = to_dev(b)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                loss_s, loss_c, acc = model(b)
                loss = loss_s + CFG["vlp_lambda"] * loss_c
            step(loss, model, opt, sched, scaler)
            ls += loss_s.item(); lc += loss_c.item(); ac += acc.item()
        n = len(loader)
        hist.append({"epoch": ep, "L_s": ls / n, "L_c": lc / n, "match_acc": ac / n})
        print(f"  [VLP {exp['name']}] ep {ep:3d}  L_s {ls/n:.3f}  L_c {lc/n:.3f}  "
              f"video-text match acc {ac/n:.1%}  ({(time.time()-t0)/60:.1f} min)", flush=True)
    torch.save({"visual": model.visual.state_dict(), "decoder": model.decoder.state_dict()}, path)
    del model, opt; torch.cuda.empty_cache()
    return path, hist

def run_slt(exp, seed, init_path=None):
    set_seed(seed)
    tag = f"{exp['name']}_s{seed}"
    model = SLTModel(len(VOCAB), CFG, exp["use_rgb"], exp["use_kp"]).to(device)
    if init_path:   # paper stage 2: inherit the pretrained visual encoder and text decoder
        ck = torch.load(init_path, map_location=device)
        model.visual.load_state_dict(ck["visual"]); model.decoder.load_state_dict(ck["decoder"])
        print("  loaded VLP weights from", os.path.basename(init_path))
    train_l = make_loader("train", exp, CFG["aug_slt"], shuffle=True)
    dev_l = make_loader("dev", exp, False, shuffle=False)
    test_l = make_loader("test", exp, False, shuffle=False)
    opt, sched = make_optim(model, CFG["slt_epochs"], len(train_l))
    scaler, hist, best, t0 = make_scaler(), [], -1.0, time.time()
    best_path = os.path.join(OUT_DIR, f"{tag}_best.pth")
    for ep in range(1, CFG["slt_epochs"] + 1):
        model.train(); tot = 0.0
        for b in train_l:
            b = to_dev(b)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                logits = model(b)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), b["tokens"][:, 1:].reshape(-1),
                                       ignore_index=PAD, label_smoothing=CFG["label_smoothing"])
            step(loss, model, opt, sched, scaler)
            tot += loss.item()
        rec = {"epoch": ep, "train_loss": tot / len(train_l)}
        if ep % CFG["eval_every"] == 0 or ep == CFG["slt_epochs"]:
            _, refs, hyps = translate(model, dev_l, beam=1)
            rec["dev_B4_greedy"] = corpus_bleu(refs, hyps, 4)
            if rec["dev_B4_greedy"] > best:
                best = rec["dev_B4_greedy"]; torch.save(model.state_dict(), best_path)
        hist.append(rec)
        msg = f"  [SLT {tag}] ep {ep:3d}  loss {rec['train_loss']:.3f}"
        if "dev_B4_greedy" in rec: msg += f"  dev BLEU-4 (greedy) {rec['dev_B4_greedy']:.2f}  best {best:.2f}"
        print(msg + f"  ({(time.time()-t0)/60:.1f} min)", flush=True)

    model.load_state_dict(torch.load(best_path, map_location=device))
    result = {"name": exp["name"], "seed": seed, "use_rgb": exp["use_rgb"], "use_kp": exp["use_kp"], "vlp": exp["vlp"],
              "best_dev_greedy_B4": best, "train_minutes": round((time.time() - t0) / 60, 1)}
    for split, loader in (("dev", dev_l), ("test", test_l)):
        names, refs, hyps = translate(model, loader, beam=CFG["beam"])
        result[split] = all_metrics(refs, hyps)
        with open(os.path.join(OUT_DIR, f"{tag}_{split}_translations.tsv"), "w") as f:
            f.write("name\treference\thypothesis\n")
            for n, r, h in zip(names, refs, hyps):
                f.write(f"{n}\t{' '.join(r)}\t{' '.join(h)}\n")
    print(f"  => {tag}  dev {result['dev']}  test {result['test']}")
    del model, opt; torch.cuda.empty_cache()
    return result, hist

# ---- smoke test: run every stage once on a small subset before the full experiments ----
_FULL_NAMES, _FULL_CFG = NAMES, dict(CFG)
NAMES = {"train": _FULL_NAMES["train"][:128], "dev": _FULL_NAMES["dev"][:16], "test": _FULL_NAMES["test"][:16]}
CFG.update(slt_epochs=1, vlp_epochs=1, eval_every=1, beam=2)
for _exp in EXPERIMENTS:
    print("smoke test:", _exp["name"])
    _init = run_vlp(_exp, 0)[0] if _exp["vlp"] else None
    run_slt(_exp, 0, _init)
NAMES, CFG = _FULL_NAMES, _FULL_CFG
print("\nSMOKE TEST PASSED - starting the real training\n")


RESULTS_FILE = os.path.join(OUT_DIR, "results_quicktest.json" if QUICK_TEST else "results.json")
results = json.load(open(RESULTS_FILE)) if os.path.exists(RESULTS_FILE) and not QUICK_TEST else []
done = {(r["name"], r["seed"]) for r in results}
histories = {}
t_all = time.time()
for exp in EXPERIMENTS:
    for seed in SEEDS:
        if (exp["name"], seed) in done:
            print("skip (already done):", exp["name"], seed); continue
        print(f"\n===== {exp['name']}  seed {seed} =====")
        init, vlp_hist = None, None
        if exp["vlp"]:
            init, vlp_hist = run_vlp(exp, seed)
        res, hist = run_slt(exp, seed, init)
        res["vlp_history"], res["slt_history"] = vlp_hist, hist
        results.append(res)
        json.dump(results, open(RESULTS_FILE, "w"), indent=1)    # saved after every run
print(f"\nall runs finished in {(time.time() - t_all)/60:.0f} min")

rows = []
for r in results:
    for split in ("dev", "test"):
        rows.append({"experiment": r["name"], "seed": r["seed"], "split": split, **r[split]})
df = pd.DataFrame(rows)
cols = ["B1", "B2", "B3", "B4", "ROUGE"]
table = df.groupby(["experiment", "split"])[cols].mean().unstack("split").swaplevel(axis=1).sort_index(axis=1, level=0)
display(table.round(2))
if len(SEEDS) > 1:
    print("mean and standard deviation over seeds:")
    display(df.groupby(["experiment", "split"])[cols].agg(["mean", "std"]).round(2))

paper = pd.DataFrame({"dev B4": [19.84, 22.12], "test B4": [19.66, 21.44], "test ROUGE": [40.93, 42.49]},
                     index=["paper GFSLT (fine-tuned ResNet, 200 ep)", "paper GFSLT-VLP"])
print("Paper numbers for reference (not directly comparable: they fine-tune the CNN on raw video):")
display(paper)
df.to_csv(os.path.join(OUT_DIR, "results_table.csv"), index=False)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for r in results:
    h = r["slt_history"]
    lbl = f"{r['name']} s{r['seed']}"
    ax[0].plot([x["epoch"] for x in h], [x["train_loss"] for x in h], label=lbl)
    ev = [x for x in h if "dev_B4_greedy" in x]
    ax[1].plot([x["epoch"] for x in ev], [x["dev_B4_greedy"] for x in ev], marker="o", ms=3, label=lbl)
ax[0].set_title("SLT training loss"); ax[0].set_xlabel("epoch")
ax[1].set_title("dev BLEU-4 (greedy decoding)"); ax[1].set_xlabel("epoch")
for a in ax: a.legend(fontsize=8); a.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "curves.png"), dpi=110); plt.show()

vlp_runs = [r for r in results if r.get("vlp_history")]
if vlp_runs:
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.5))
    for r in vlp_runs:
        h = r["vlp_history"]
        ax[0].plot([x["L_s"] for x in h], label=f"{r['name']} contrastive L_s")
        ax[0].plot([x["L_c"] for x in h], "--", label=f"{r['name']} masked L_c")
        ax[1].plot([x["match_acc"] for x in h], label=r["name"])
    ax[0].set_title("VLP losses"); ax[1].set_title("VLP: video picks its own sentence in the batch")
    for a in ax: a.legend(fontsize=8); a.grid(alpha=0.3); a.set_xlabel("epoch")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "vlp_curves.png"), dpi=110); plt.show()

def load_tsv(name, seed=SEEDS[0], split="test"):
    p = os.path.join(OUT_DIR, f"{name}_s{seed}_{split}_translations.tsv")
    return pd.read_csv(p, sep="\t", keep_default_na=False).set_index("name") if os.path.exists(p) else None

base, full = load_tsv(EXPERIMENTS[0]["name"]), load_tsv(EXPERIMENTS[-1]["name"])
if base is not None and full is not None:
    j = base.join(full["hypothesis"], rsuffix="_full")
    j["b4_base"] = [corpus_bleu([r.split()], [h.split()], 4) for r, h in zip(j.reference, j.hypothesis)]
    j["b4_full"] = [corpus_bleu([r.split()], [h.split()], 4) for r, h in zip(j.reference, j.hypothesis_full)]
    better = j[j.b4_full > j.b4_base].sort_values("b4_full", ascending=False).head(4)
    worse = j[j.b4_full < j.b4_base].head(2)
    for title, part in (("Full model better than baseline", better), ("Full model worse than baseline", worse)):
        print(f"\n### {title}")
        for name, row in part.iterrows():
            print(f"REF : {row.reference}\n{EXPERIMENTS[0]['name']:14s}: {row.hypothesis}\n"
                  f"{EXPERIMENTS[-1]['name']:14s}: {row.hypothesis_full}\n")
print("Everything (results.json, results_table.csv, curves, translations) is in", OUT_DIR)