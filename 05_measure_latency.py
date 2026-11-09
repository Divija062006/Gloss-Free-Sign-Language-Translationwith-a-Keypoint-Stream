# =============================================================================
# Step 5 of 5 - Parameters, latency and a re-check of test BLEU-4
# Project : Gloss-free sign language translation with a keypoint stream
# Team    : Claude5
#
# Loads the best checkpoint of each model (E1-E4), counts parameters, and times
# the visual encoder + beam-5 decoding per test video on the GPU (batch size 1,
# first 10 videos excluded as warm-up). Test BLEU-4 is recomputed as a check.
# Also times the frozen ResNet18 for one average-length test video.
#
# Kaggle  : Accelerator = GPU T4, Internet = On
# Input   : outputs of steps 1, 2, 3 and 4
# Output  : printed summary table
# =============================================================================

# ======================= settings =======================
QUICK_TEST = False       # True: short debugging run on a small subset

EXPERIMENTS = [   # defines which inputs are loaded (keypoints included)
    dict(name="E2_rgb_kp",     use_rgb=True,  use_kp=True,  vlp=False),
    dict(name="E4_rgb_kp_vlp", use_rgb=True,  use_kp=True,  vlp=True),
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


# ======================= parameters, latency and test BLEU-4 re-check =======================
import time
def n_params(m): return sum(p.numel() for p in m.parameters())
rows = []
for name, use_kp in [("E1_rgb", False), ("E2_rgb_kp", True), ("E3_rgb_vlp", False), ("E4_rgb_kp_vlp", True)]:
    ck = find_file(f"{name}_s0_best.pth", roots=("/kaggle/input",))
    assert ck, f"checkpoint for {name} not found: attach the outputs of steps 3 and 4"
    model = SLTModel(len(VOCAB), CFG, True, use_kp).to(device)
    model.load_state_dict(torch.load(ck, map_location=device)); model.eval()
    ds = SignDataset(NAMES["test"], True, use_kp, augment=False)
    times, refs, hyps = [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            b = to_dev(collate([ds[i]]))
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                mem, valid, _ = model.visual(b["rgb"], b["kp"], b["mask"])
                out = beam_search(model.decoder, mem, valid, CFG["beam"], CFG["max_dec_len"], CFG["length_penalty"])
            torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
            refs.append(b["words"][0]); hyps.append(VOCAB.decode(out))
    t = np.array(times[10:]) * 1000   # skip 10 warm-up videos
    rows.append({"model": name, "params_M": round(n_params(model) / 1e6, 2), "latency_ms_mean": round(float(t.mean()), 1),
                 "latency_ms_median": round(float(np.median(t)), 1), "test_B4_recheck": round(corpus_bleu(refs, hyps, 4), 2)})
    print(rows[-1], flush=True)
    del model; torch.cuda.empty_cache()

# frozen ResNet18 cost per test video (average test length), for reference
import torchvision
rn = torchvision.models.resnet18(weights=None); rn.fc = nn.Identity(); rn = rn.eval().to(device)
avg_T = int(np.mean([len(DATA[n]["rgb"]) for n in NAMES["test"]]))
x = torch.randn(avg_T, 3, 224, 224, device=device)
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
    for _ in range(3): rn(x)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(10): rn(x)
    torch.cuda.synchronize()
print(f"\nResNet18 feature extraction: {(time.perf_counter() - t0) / 10 * 1000:.1f} ms per video ({avg_T} frames), params {n_params(rn)/1e6:.2f} M")
print("\n=== Summary ===")
print(pd.DataFrame(rows).to_string(index=False))
