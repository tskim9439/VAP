"""VapAsrForStreamingASR — HF PreTrainedModel 판 mono 스트리밍 ASR (vapasr/uslm/mono_model.MonoInterleavedASR 와 수치 동일).

구성: encoder(Nemotron [56,0], 온라인 특징; 기본 동결) → adapter(LN·MLP) → thinker(Qwen3-ASR thinker, 오디오 타워 제거, full FT).
V2-a 두 스트림 입력(config.enc_tap_layer ≥ 0, add_enc_tap): 인코더 아래층 출력을 adapter_tap(같은 구조)으로 바꿔 adapter 출력에 더한다 — 마지막 층에서 줄어든 화자 정보(V2 M0)를
  LLM 입력에 다시 넣는다(VibeVoice-ASR 의 acoustic + semantic 합과 같은 방식, 화자 과제·외부 모듈 없음). 특징 텐서는 (B, S, K, Din): S = 1(V1) 또는 2(마지막 층, 탭 층).
시퀀스·손실 규약은 mono_model 과 같다: 라벨 위치에서만 lm_head 를 계산하고 <NEXT_AUDIO> 는 next_weight 로 가중.
저장: adapter + thinker 만 safetensors 에 쓴다(인코더는 동결이면 .nemo 에서 다시 만들고, 학습했으면 함께 저장). lm_head 는 임베딩과 tied.
생성: from_qwen(Qwen3-ASR 디렉토리) 로 새 모델, from_legacy(ckpt-last.pt) 로 기존 산출물 변환, from_pretrained(dir) 로 재로드."""
import os, time, math
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.utils import ModelOutput
from .configuration_vapasr import VapAsrConfig
from .p2_losses import soft_ce, activity_bce, gather_audio_targets
from .varlen_attention import is_varlen, packed_kwargs

SLOT_TOKENS = ("<SPK_A>", "<SPK_B>", "<SPK_3>", "<SPK_4>")      # 화자 슬롯 A·B·C·D(V2: C·D = Phase 2 예약 토큰) — enable_slots

class Adapter(nn.Module):                                   # uslm/model.Adapter 와 동일 구조(키 이름 net.* 유지 → 기존 ckpt 호환)
    def __init__(self, d_in=1024, d_out=1024, h=2048):
        super().__init__(); self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, h), nn.GELU(), nn.Linear(h, d_out))
    def forward(self, x): return self.net(x)

@dataclass
class VapAsrOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    loss_next: Optional[float] = None       # <NEXT_AUDIO> 위치 평균 CE (가중 전)
    loss_text: Optional[float] = None       # 텍스트 위치 평균 CE
    top1_text: Optional[float] = None       # 텍스트 위치 top-1 정확도
    n_labels: Optional[int] = None
    loss_eot: Optional[float] = None        # Phase 2: EOT 위치 평균 CE(soft 혼합 후)
    loss_act: Optional[float] = None        # Phase 2: lane 활동 BCE
    act_acc: Optional[float] = None
    n_soft: Optional[int] = None
    loss_sem: Optional[float] = None        # semantic commit: <SEM_END> 타깃 위치 평균 CE (배치에 SEM 이 없으면 None)
    loss_turn: Optional[float] = None       # 턴 종료(mono <EOT>, v0.2 <TURN_END>) 타깃 위치 평균 CE (없으면 None)
    top1_sem: Optional[float] = None        # <SEM_END> 타깃 위치 top-1 정확도 (없으면 None)
    n_sem: Optional[int] = None
    sem_fp: Optional[float] = None          # SEM 이 타깃이 아닌 라벨 위치 중 argmax=SEM 비율 (조기·오확정 경향)
    n_turn: Optional[int] = None
    top1_switch: Optional[float] = None     # V3-p1 C1(dec_kind 를 줬을 때만): 교대 태그 위치 top-1 정확도
    switch_miss: Optional[float] = None     # 교대 위치 중 argmax 가 글자(특수 토큰 아님)인 비율 — 태그 없이 이어 쓰는 놓친 교대
    switch_wrong_slot: Optional[float] = None   # 교대 위치 중 argmax 가 다른 슬롯 태그인 비율(2 슬롯이면 지금 슬롯 재선언)
    loss_switch: Optional[float] = None     # 교대 위치 평균 CE(가중 전)
    n_switch: Optional[int] = None
    top1_hold: Optional[float] = None       # <EOT> 뒤 머무름 결정 위치 top-1 정확도
    n_hold: Optional[int] = None

class VapAsrForStreamingASR(PreTrainedModel):
    config_class = VapAsrConfig
    base_model_prefix = "vapasr"
    supports_gradient_checkpointing = True
    _tied_weights_keys = ["thinker.lm_head.weight"]
    _keys_to_ignore_on_load_unexpected = [r"^encoder\.", r"^thinker\.audio_tower\."]
    _keys_to_ignore_on_load_missing = [r"^encoder\."]

    def __init__(self, config: VapAsrConfig, thinker: Optional[nn.Module] = None):
        super().__init__(config)
        self.adapter = Adapter(config.adapter_d_in, config.adapter_d_out, config.adapter_hidden)
        self.adapter_tap = Adapter(config.adapter_d_in, config.adapter_d_out, config.enc_tap_hidden) if getattr(config, "enc_tap_layer", -1) >= 0 else None   # V2-a(이름이 adapter 로 시작 → adapter lr 그룹)
        h = int(getattr(config, "spk_inject_hidden", 0) or 0)                                    # E2 대상 화자 활동 주입(이름이 adapter 로 시작 → adapter lr 그룹)
        self.adapter_inject = nn.Sequential(nn.Linear(config.adapter_d_in, h), nn.GELU(), nn.Linear(h, config.adapter_d_in)) if h > 0 else None
        n = int(getattr(config, "slot_input_slots", 0) or 0)                                      # E5 슬롯 활동 입력(오디오 토큰 += Σ a_s·e_s, 이름이 adapter 로 시작 → adapter lr 그룹)
        self.adapter_slot = nn.Linear(n, config.adapter_d_out, bias=False) if n > 0 else None
        if thinker is None:
            from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRThinkerConfig
            from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRThinkerForConditionalGeneration
            thinker = Qwen3ASRThinkerForConditionalGeneration(Qwen3ASRThinkerConfig.from_dict(dict(config.thinker)))
        if hasattr(thinker, "audio_tower"): del thinker.audio_tower               # 오디오 타워는 쓰지 않는다(Nemotron 이 대신)
        self.thinker = thinker
        self.encoder: Optional[nn.Module] = None                                  # attach_encoder 로 붙인다(무거운 NeMo 복원은 __init__ 밖에서)
        self.sp_ids = dict(config.sp_ids); self.next_audio = self.sp_ids.get("<NEXT_AUDIO>"); self.empty_audio = self.sp_ids.get("<EMPTY_AUDIO>")
        self.register_buffer("blocked", torch.tensor(sorted(set(config.blocked_ids)), dtype=torch.long), persistent=False)
        self.act_head = nn.Sequential(nn.Linear(config.hidden_size, config.act_hidden), nn.GELU(), nn.Linear(config.act_hidden, config.lanes)) if config.lanes > 0 else None   # Phase 2 lane 활동 헤드(audio 위치)
        self.spk_head = self._spk_head(config.spk_slots) if getattr(config, "spk_slots", 0) > 0 and config.lanes == 0 else None   # Stage 3 화자 슬롯 활동 헤드(audio 위치, 학습 보조)
        self.post_init()

    # ── HF 규약(tie·임베딩)
    def get_input_embeddings(self): return self.thinker.get_input_embeddings()
    def set_input_embeddings(self, v): self.thinker.set_input_embeddings(v)
    def get_output_embeddings(self): return self.thinker.lm_head
    def _init_weights(self, module):                                              # thinker 는 자체 초기화, adapter 는 기본(nn) 초기화 → 아무것도 덮지 않는다
        return

    # ── 인코더
    def attach_encoder(self, path: Optional[str] = None, trainable: Optional[bool] = None, state_dict: Optional[dict] = None):
        from ..features.online import NemotronOnline
        enc = NemotronOnline(right_context=self.config.encoder_right_context, path=path)
        if state_dict: enc.enc.load_state_dict(state_dict)
        tr = self.config.encoder_trainable if trainable is None else trainable; enc.set_trainable(tr); self.config.encoder_trainable = tr
        self.encoder = enc.to(next(self.adapter.parameters()).device); return self
    def encode(self, wav, wav_len, K, act=None):                                  # (B,L) → (B,S,Kmax,Din) — S = 1, 탭이 있으면 2(마지막 층, 탭 층)
        """act (B, ≥Kmax) 대상 화자 활동(0–1, E2) — 있으면 adapter_inject 로 인코더 pre-encode 출력에 주입한다(없으면 예전 그대로)."""
        assert self.encoder is not None, "encoder 없음 — attach_encoder() 먼저"
        assert act is None or self.adapter_inject is not None, "대상 화자 활동을 주었는데 adapter_inject 가 없다(add_spk_inject)"
        tap = int(getattr(self.config, "enc_tap_layer", -1)); kw = {"inject": (self.adapter_inject, act)} if act is not None else {}
        return self.encoder(wav, wav_len, K, taps=(tap,), **kw) if tap >= 0 else self.encoder(wav, wav_len, K, **kw)[:, None]
    def add_spk_inject(self, hidden: int = 512) -> None:
        """E2 대상 화자 활동 주입을 켠다(SSA, arXiv:2506.22646): adapter_inject = Linear(D, h)·GELU·Linear(h, D), 마지막 Linear 0 초기화 → 붙인 직후 출력은 그대로.
        이미 있으면 그대로(재개 — 은닉 차원이 다르면 멈춘다)."""
        if self.adapter_inject is not None:
            assert int(self.config.spk_inject_hidden) == int(hidden), f"이미 은닉 {self.config.spk_inject_hidden} — {hidden} 으로 바꾸지 않는다"; return
        d = self.config.adapter_d_in; dev = next(self.adapter.parameters()).device
        self.adapter_inject = nn.Sequential(nn.Linear(d, int(hidden)), nn.GELU(), nn.Linear(int(hidden), d)).to(dev)
        with torch.no_grad(): self.adapter_inject[-1].weight.zero_(); self.adapter_inject[-1].bias.zero_()
        self.config.spk_inject_hidden = int(hidden)
    def chunk_embed(self, feats, act=None):                                       # (B,S,K,Din) → (B,K,D)
        """act (B, K', ≤n) 슬롯 활동(0–1, E5 — 슬롯 순서 = 창에서 먼저 말한 순서) — 있으면 오디오 토큰에 adapter_slot(act) = Σ a_s·e_s 를 더한다(K' ≠ K 면 0 채움·자름)."""
        ce = self.adapter(feats[:, 0].float())
        if self.adapter_tap is not None:
            assert feats.shape[1] >= 2, f"두 스트림 모델(enc_tap_layer {self.config.enc_tap_layer})인데 특징 스트림이 {feats.shape[1]} 개 — model.encode 로 만든 특징을 쓴다"
            ce = ce + self.adapter_tap(feats[:, 1].float())
        if act is not None:
            assert self.adapter_slot is not None, "슬롯 활동을 주었는데 adapter_slot 이 없다(add_slot_input)"
            a = act.float().to(ce.device); n, K = self.adapter_slot.in_features, ce.shape[1]
            a = F.pad(a, (0, max(0, n - a.shape[-1]), 0, max(0, K - a.shape[1])))[:, :K, :n]
            ce = ce + self.adapter_slot(a)
        return ce

    def add_slot_input(self, n: int = 4) -> None:
        """E5 슬롯 활동 입력을 켠다(2026-10-26 — output-diarization-integration-20261025 6 절): adapter_slot = Linear(n, D, bias 없음), 0 초기화 → 붙인 직후 출력은 그대로.
        이미 있으면 그대로(재개 — 슬롯 수가 다르면 멈춘다)."""
        if self.adapter_slot is not None:
            assert int(self.config.slot_input_slots) == int(n), f"이미 슬롯 {self.config.slot_input_slots} — {n} 으로 바꾸지 않는다"; return
        dev = next(self.adapter.parameters()).device
        self.adapter_slot = nn.Linear(int(n), self.config.adapter_d_out, bias=False).to(dev)
        with torch.no_grad(): self.adapter_slot.weight.zero_()
        self.config.slot_input_slots = int(n)

    def add_enc_tap(self, layer: int, hidden: Optional[int] = None) -> None:
        """V2-a 두 스트림 입력을 켠다: 인코더 layer 층(0 부터) 출력 → adapter_tap(adapter 와 같은 LN·MLP) → adapter 출력에 더한다.
        adapter_tap 의 마지막 Linear 를 0 으로 초기화 → 붙인 직후 모델 출력은 V1 과 같다(학습이 탭을 쓰는 만큼만 바뀐다). 이미 같은 층이면 그대로(재개)."""
        n = len(self.encoder.enc.layers) if self.encoder is not None else None
        assert n is None or 0 <= int(layer) < n, f"탭 층 {layer}: 인코더 층 0–{n - 1}"
        if self.adapter_tap is not None:
            assert int(self.config.enc_tap_layer) == int(layer), f"이미 탭 층 {self.config.enc_tap_layer} — 다른 층({layer})으로 바꾸지 않는다"; return
        h = int(hidden or self.config.adapter_hidden); dev = next(self.adapter.parameters()).device
        self.adapter_tap = Adapter(self.config.adapter_d_in, self.config.adapter_d_out, h).to(dev)
        with torch.no_grad(): self.adapter_tap.net[-1].weight.zero_(); self.adapter_tap.net[-1].bias.zero_()
        self.config.enc_tap_layer, self.config.enc_tap_hidden = int(layer), h

    # ── 학습 forward
    def build(self, feats, ids, is_audio, chunk_of, packed: bool = False, act_in=None):
        emb = self.get_input_embeddings(); E = emb(ids); ce = self.chunk_embed(feats, act_in).to(E.dtype); D = E.shape[-1]
        if packed: ce = ce.reshape(1, -1, D)                                          # (B,Kmax,D) → (1,B·Kmax,D); chunk_of 는 packing.pack_batch 의 전역 인덱스
        g = torch.gather(ce, 1, chunk_of.clamp(min=0)[..., None].expand(-1, -1, D))
        return torch.where(is_audio[..., None], g, E)

    def forward(self, ids=None, is_audio=None, chunk_of=None, labels=None, mask=None, wav=None, wav_len=None, K=None, feats=None, next_weight: Optional[float] = None, return_dict: bool = True,
                labels_alt=None, soft_w=None, activity=None, activity_mask=None, act_weight: Optional[float] = None, pos_weight=None, position_ids=None, sem_mask=None,
                sem_only_mask=None, dec_kind=None, target_act=None, act_in=None, **_):
        """wav(+wav_len, K) 또는 feats 중 하나. labels 는 -100 이 손실 제외. next_weight 기본 config.next_weight.
        Phase 2(config.lanes>0): labels_alt/soft_w 가 있으면 EOT 후보 위치의 label 을 (labels: w, labels_alt: 1−w) 두 점 분포로 학습(정본 §5.2),
        activity (B,K,R)·activity_mask (B,K) 가 있으면 audio 위치 hidden 으로 lane 활동 BCE 를 더한다(정본 §6). EOT 위치 가중 config.eot_weight.
        semantic commit(config.sem_registry 가 있으면, lanes 와 무관): <SEM_END> 타깃 가중 config.sem_weight, mono 턴 종료(<EOT>; v0.2 는 <TURN_END>)를 함께 학습하면 turn_weight,
        pos_weight (B,L) 은 labels 와 같은 인덱싱의 위치별 가중 덮어쓰기(0 = 기본, >0 = 그 값; hard negative 결정 위치). 덮어쓰기는 SEM/TURN 가중보다 우선하므로
        결정 위치가 이벤트 타깃에 떨어지지 않게 하는 것은 데이터셋 책임이다.
        sem_mask (B,L) bool(labels 와 같은 인덱싱; packing 이면 (1,N)): True 인 위치는 softmax 에서 <SEM_END>·턴 종료 logit 을 −∞ 로 뺀다 — SEM 라벨이 없는 ASR 행
        (SemCommitDataset sem_free)이 텍스트·NEXT 는 그대로 배우면서 'SEM 아님' 음성 신호는 주지 않게. 그 위치에 이벤트 타깃이 있으면 AssertionError. sem_fp 통계는 그 위치를 분모에 넣는다.
        sem_only_mask (B,L) bool(같은 인덱싱): True 인 위치는 <SEM_END> logit 만 뺀다 — 턴 종료(<EOT>)는 학습하는 Stage 3 대화 행(vapasr/data/cst_dialogue.collate_cst).
        labels_alt/soft_w(두 점 soft target)는 lanes 와 무관하게 받는다(Stage 3 mono <EOT> soft label).
        position_ids (1, N) 가 있으면 packing 경로(vapasr/hf/packing.py): thinker 입력은 여러 샘플을 이은 한 줄, 샘플마다 0 부터 다시 세는 위치,
        attention_mask 없음 → transformers 가 블록 대각 causal mask 를 만든다. 인코더 출력은 (1, B·Kmax) 로 펴고 activity 도 같이 편다.
        thinker 가 varlen attention(vapasr/hf/varlen_attention.py, --attn-impl varlen)이면: packing 은 position_ids 로 cu_seqlens 를 만들어 넘기고(Σnᵢ² attention),
        패딩 배치는 모델 안에서 한 줄로 unpad 해 같은 커널로 돌린 뒤 (B, L) 로 되돌린다(_thinker_unpadded — pad 토큰은 어느 층에서도 계산하지 않는다).
        dec_kind (B,L)(labels 와 같은 인덱싱, 0 · 1 교대 · 2 머무름 — cst_dialogue.collate_cst, V3-p1 C1)가 있으면 결정 통계(top1_switch·switch_miss·switch_wrong_slot·
        loss_switch·top1_hold)만 계산한다 — 손실은 그대로(가중은 데이터셋의 pos_weight 가 맡는다)."""
        if wav is not None: feats = self.encode(wav, wav_len, K, act=target_act)          # target_act (B, Kmax): E2 대상 화자 활동(없으면 예전 그대로)
        nw = self.config.next_weight if next_weight is None else float(next_weight)
        packed = position_ids is not None; varlen = is_varlen(self.thinker)
        if packed:
            fa = packed_kwargs(position_ids=position_ids) if varlen else {}
            h = self.thinker.model(inputs_embeds=self.build(feats, ids, is_audio, chunk_of, packed=True, act_in=act_in), attention_mask=None, position_ids=position_ids,
                                   use_cache=False, **fa).last_hidden_state
            if activity is not None:
                activity = activity.reshape(1, -1, activity.shape[-1])
                activity_mask = activity_mask.reshape(1, -1) if activity_mask is not None else None
        elif varlen and mask is not None:
            h = self._thinker_unpadded(self.build(feats, ids, is_audio, chunk_of, act_in=act_in), mask.bool())
        else:
            h = self.thinker.model(inputs_embeds=self.build(feats, ids, is_audio, chunk_of, act_in=act_in), attention_mask=mask).last_hidden_state
        tgt = labels[:, 1:]; sel = tgt != -100; t = tgt[sel]
        logits = self.thinker.lm_head(h[:, :-1][sel]).float()                       # 라벨 위치만 (전 위치 fp32 logits 는 OOM)
        alt = labels_alt[:, 1:][sel] if labels_alt is not None else None; w = soft_w[:, 1:][sel] if soft_w is not None else None
        eot_id = self.config.phase2_registry.get("<EOT>", -1) if self.config.lanes > 0 else -1
        tag_ids = torch.tensor([v for k, v in self.config.phase2_registry.items() if k != "<EOT>"], device=logits.device, dtype=torch.long) if self.config.lanes > 0 and getattr(self.config, "tag_weight", 1.0) != 1.0 else None
        sem = getattr(self.config, "sem_registry", None) or {}
        if sem:
            from ..data.semcommit_tokens import sem_ids_of
            sem_id, turn_id = sem_ids_of(sem); turn_id = -1 if turn_id is None else turn_id
        else:                                                                         # semcommit 데이터셋(토큰을 스스로 추가)만 쓰고 add_semantic_tokens 를 빠뜨리면 SEM 이 평문 text(가중 1, 초기화 없음)로 조용히 학습된다
            from ..data.semcommit_tokens import SEM_REGISTRY, LEGACY_TURN_ID
            sem_id = turn_id = -1
            assert not torch.isin(t, torch.tensor(list(SEM_REGISTRY.values()) + [LEGACY_TURN_ID], device=t.device)).any(), "labels 에 <SEM_END> 가 있는데 config.sem_registry 가 비어 있음 — model.add_semantic_tokens(tok) 먼저"
        if sem_mask is not None and sem_id >= 0:                                     # SEM 중립(라벨 없는 ASR 행): 그 위치 softmax 에서 <SEM_END>·턴 종료를 뺀다 → 'SEM 아님' 신호가 없다
            sm = sem_mask[:, 1:][sel].bool()
            ev_ids = [i for i in (sem_id, turn_id) if i >= 0]
            assert not (sm & torch.isin(t, torch.tensor(ev_ids, device=t.device))).any(), "sem_mask 위치에 <SEM_END>/턴 종료 타깃 — SEM 중립 행에 이벤트 라벨이 있으면 안 된다"
            col = torch.zeros(logits.shape[1], dtype=torch.bool, device=logits.device); col[ev_ids] = True
            logits = logits.masked_fill(sm[:, None] & col[None, :], float("-inf"))
        if sem_only_mask is not None and sem_id >= 0:                                # Stage 3 대화 행: <SEM_END> 만 중립, <EOT> 는 학습
            so = sem_only_mask[:, 1:][sel].bool()
            assert not (so & (t == sem_id)).any(), "sem_only_mask 위치에 <SEM_END> 타깃 — 그 행은 SEM 을 쓰지 않아야 한다"
            col = torch.zeros(logits.shape[1], dtype=torch.bool, device=logits.device); col[sem_id] = True
            logits = logits.masked_fill(so[:, None] & col[None, :], float("-inf"))
        evt = {i: float(getattr(self.config, k, 1.0)) for i, k in ((sem_id, "sem_weight"), (turn_id, "turn_weight")) if i >= 0} or None
        pw = pos_weight[:, 1:][sel] if pos_weight is not None else None
        loss, st = soft_ce(logits, t, alt, w, self.next_audio, nw, eot_id=eot_id, eot_weight=self.config.eot_weight if eot_id >= 0 else 1.0, tag_ids=tag_ids, tag_weight=getattr(self.config, "tag_weight", 1.0),
                           evt_weights=evt, pos_override=pw, sem_id=sem_id, turn_id=turn_id)
        loss_act, act_st = None, {}
        head = self.act_head if self.act_head is not None else self.spk_head                   # Phase 2 lane 활동 | Stage 3 화자 슬롯 활동(같은 audio 위치 BCE)
        if head is not None and activity is not None:
            ha, tg = gather_audio_targets(h, is_audio, chunk_of, activity, activity_mask if activity_mask is not None else torch.ones(activity.shape[:2], device=h.device))
            aw = float(act_weight) if act_weight is not None else (self.config.act_weight if self.act_head is not None else self.config.spk_weight)
            loss_act, act_st = activity_bce(head(ha).float(), tg); loss = loss + aw * loss_act
        elif self.spk_head is not None and self.training:                                     # 활동 라벨 없는 행(replay): 헤드를 0 가중으로 그래프에 묶어 DDP(find_unused_parameters=False)가 기다리지 않게
            loss = loss + 0.0 * sum(p.sum() for p in self.spk_head.parameters())
        dec = self._decision_stats(logits, t, dec_kind[:, 1:][sel]) if dec_kind is not None else {}
        out = VapAsrOutput(loss=loss, loss_next=st["loss_next"], loss_text=st["loss_text"], top1_text=st["top1_text"], n_labels=int(t.numel()),
                           loss_eot=st["loss_eot"], loss_act=(loss_act.item() if loss_act is not None else None), act_acc=act_st.get("act_acc"), n_soft=st["n_soft"],
                           loss_sem=st["loss_sem"] if st.get("n_sem") else None, top1_sem=st["top1_sem"] if st.get("n_sem") else None, n_sem=st.get("n_sem"), sem_fp=st.get("sem_fp"),
                           loss_turn=st["loss_turn"] if st.get("n_turn") else None, n_turn=st.get("n_turn"), **dec)
        return out if return_dict else (loss,)

    @torch.no_grad()
    def _decision_stats(self, logits, t, dk) -> dict:
        """교대(dk 1)·머무름(dk 2) 위치 통계(손실과 무관) — 위치가 없으면 그 쪽 키를 내지 않는다. 글자 = 특수 토큰(config.sp_ids·차단 목록)이 아닌 id."""
        dk = dk.to(t.device); pred = logits.argmax(-1); out = {}
        special = torch.tensor(sorted(set(self.sp_ids.values()) | set(self.config.blocked_ids)), device=t.device, dtype=torch.long)
        slots = torch.tensor([self.sp_ids[n] for n in (getattr(self.config, "slot_tokens", None) or SLOT_TOKENS[:2]) if n in self.sp_ids], device=t.device, dtype=torch.long)
        sw, ho = dk == 1, dk == 2
        if sw.any():
            ps, ts = pred[sw], t[sw]
            out.update(top1_switch=(ps == ts).float().mean().item(), switch_miss=(~torch.isin(ps, special)).float().mean().item(),
                       switch_wrong_slot=(torch.isin(ps, slots) & (ps != ts)).float().mean().item(), loss_switch=F.cross_entropy(logits[sw], ts).item(), n_switch=int(sw.sum()))
        if ho.any(): out.update(top1_hold=(pred[ho] == t[ho]).float().mean().item(), n_hold=int(ho.sum()))
        return out

    def _thinker_unpadded(self, E, sel):
        """varlen: 패딩 배치 E (B,L,D) 의 유효 토큰(sel (B,L) bool)만 한 줄 packed 로 thinker 에 넣고(cu_seqlens = 행별 유효 토큰 수)
        결과를 (B,L,D) 로 되돌린다 — packing.pack_batch 와 같은 계산. pad 위치 hidden 은 0 (손실·통계는 라벨/오디오 위치, 즉 유효 토큰만 읽는다).
        위치는 원래 열 인덱스: 패딩 sdpa 경로(가려진 토큰은 보이지 않지만 위치는 유지)와 같다 — 오른쪽 패딩은 0 부터 세는 packing 과 같고,
        왼쪽 패딩·중간이 빈 마스크도 RoPE 상대 거리가 sdpa 와 같다(0 부터 다시 세면 구멍 양쪽 거리가 줄어든다)."""
        pos = torch.arange(sel.shape[1], device=sel.device).expand_as(sel)[sel][None]
        hp = self.thinker.model(inputs_embeds=E[sel][None], attention_mask=None, position_ids=pos, use_cache=False,
                                **packed_kwargs(lengths=sel.sum(1))).last_hidden_state[0]
        return hp.new_zeros(*sel.shape, hp.shape[-1]).index_put((sel,), hp)

    def _spk_head(self, slots: int) -> nn.Module:
        return nn.Sequential(nn.Linear(self.config.hidden_size, self.config.act_hidden), nn.GELU(), nn.Linear(self.config.act_hidden, slots))

    def add_spk_head(self, slots: int = 2, weight: float = 0.5) -> None:
        """Stage 3(mono) 화자 슬롯 활동 헤드를 붙인다 — 청크마다 '슬롯 s 화자가 말하는 중' BCE(LLM 의 audio 위치 hidden, 슬롯 순서 = 창에서 먼저 말한 사람 A).
        텍스트 화자 태그는 δ 청크 늦게 나오므로 지금 청크의 슬롯을 맞히려면 목소리로 정체를 추적해야 한다. 디코드는 쓰지 않는다. 이미 같은 슬롯 수면 가중만 바꾼다.
        이미 있는 헤드보다 슬롯이 많으면(V2: 2 → 4) 늘린다 — 앞 슬롯 행·은닉층은 그대로, 새 출력 행은 기존 행 평균(학습한 A·B 를 잃지 않게). 줄이지는 않는다."""
        assert self.config.lanes == 0, "Phase 2(lanes>0) 는 act_head 가 lane 활동을 맡는다"
        W = self.get_input_embeddings().weight; slots = int(slots)
        if self.spk_head is None:
            self.spk_head = self._spk_head(slots).to(device=W.device, dtype=W.dtype)
        elif int(self.config.spk_slots) < slots:
            old = self.spk_head[-1]; new = nn.Linear(old.in_features, slots).to(device=old.weight.device, dtype=old.weight.dtype); n0 = old.out_features
            with torch.no_grad():
                new.weight[:n0] = old.weight; new.bias[:n0] = old.bias
                new.weight[n0:] = old.weight.mean(0, keepdim=True); new.bias[n0:] = old.bias.mean()
            self.spk_head[-1] = new
        else: assert int(self.config.spk_slots) == slots, f"화자 헤드 슬롯 {self.config.spk_slots} → {slots}: 줄이지 않는다"
        self.config.spk_slots = slots; self.config.spk_weight = float(weight)

    def enable_slots(self, tok, n: int = 4, seed: int = 0) -> List[int]:
        """화자 슬롯을 n 개(최대 4: A·B·C·D)로 켠다 — C·D 는 Phase 2 예약 토큰 <SPK_3>·<SPK_4>(동결 registry 151717·151718, 새 어휘 없음).
        mono 체크포인트에서 예약만 된 행(config.sem_reserved)은 tied lm_head 의 full-vocab softmax 음성으로만 학습돼 logit 이 눌려 있으므로
        평균 + 잡음으로 다시 초기화하고(add_semantic_tokens reinit_turn 과 같은 이유), decode 차단과 예약 표시에서 뺀다. config.slot_tokens 에 슬롯 순서를 적는다.
        반복 호출은 아무것도 바꾸지 않는다. → 슬롯 토큰 id 목록."""
        from ..data.semcommit_tokens import init_rows_mean_noise, BASE_VOCAB
        from ..data.dialogue_tokens import FROZEN_REGISTRY
        assert 2 <= int(n) <= len(SLOT_TOKENS), f"슬롯 수 2–{len(SLOT_TOKENS)}: {n}"
        names = list(SLOT_TOKENS[: int(n)]); ids = [int(tok.convert_tokens_to_ids(t)) for t in names]
        assert ids == [FROZEN_REGISTRY[t] for t in names], f"슬롯 토큰 id 가 동결 registry 와 다르다: {dict(zip(names, ids))}"
        reserved = set(getattr(self.config, "sem_reserved", None) or [])
        W = self.get_input_embeddings().weight; assert max(ids) < W.shape[0], "임베딩 여유 행 없음"
        init_rows_mean_noise(W, [i for t, i in zip(names, ids) if t in reserved], upto=BASE_VOCAB, seed=seed)
        self.config.sem_reserved = [t for t in (self.config.sem_reserved or []) if t not in names]
        self.config.sp_ids.update(dict(zip(names, ids))); self.sp_ids = dict(self.config.sp_ids)
        self.config.blocked_ids = sorted(set(self.config.blocked_ids) - set(ids)); self.blocked = torch.tensor(self.config.blocked_ids, dtype=torch.long, device=W.device)
        self.config.slot_tokens = names
        return ids

    def activity_logits(self, h_audio: torch.Tensor) -> torch.Tensor:
        """Phase 2 lane 활동 logits (…, R) — 추론 parser(lane_state) 가 닫힘 판정에 쓴다."""
        assert self.act_head is not None, "Phase 2 활동 헤드 없음(config.lanes=0)"; return self.act_head(h_audio)

    def add_phase2_tokens(self, tok, R: int = 6, seed: int = 0) -> Dict[str, int]:
        """Phase 1 체크포인트(E2)에 Phase 2 registry 를 더한다: <SPK_3..R>, <ONSET>, <EOT> 를 tokenizer·config.sp_ids 에 추가하고 임베딩 행을 평균+잡음으로 초기화(기존 A/B 행은 그대로),
        활동 헤드(R 행)를 만들고, decode 차단 목록에서 <SPK_A>/<SPK_B> 를 뺀다(정본 §1·§11). 동결 registry(vapasr/data/schemas/phase2-registry.json)와 id 가 같아야 한다.
        add_semantic_tokens(mono) 가 예약·차단해 둔 Phase 2 id(config.sem_reserved)도 새 행으로 보고 평균+잡음으로 다시 초기화·차단 해제한다 — 예약 행은 tied lm_head 의
        full-vocab softmax 음성으로 학습돼 '신선'하지 않으므로 E2 → Phase 2 경로와 같게 맞춘다(일반 Phase 1 모델에서는 해당 없음). 끝나면 sem_reserved 를 비운다."""
        from ..data.dialogue_tokens import add_phase2_specials, PHASE2_SPECIALS, load_frozen_registry, LANE_TOKENS
        ids = add_phase2_specials(tok); frozen = load_frozen_registry()
        for name in PHASE2_SPECIALS: assert ids[name] == frozen[name], f"registry 불일치 {name}: tokenizer {ids[name]} vs 동결 {frozen[name]}"
        reserved = set(getattr(self.config, "sem_reserved", None) or [])
        new = [ids[n] for n in PHASE2_SPECIALS if n not in self.config.sp_ids or n in reserved]
        W = self.get_input_embeddings().weight; assert max(ids.values()) < W.shape[0], "임베딩 여유 행 없음"
        if new:
            g = torch.Generator().manual_seed(seed)
            with torch.no_grad():
                mu = W[: min(ids.values())].float().mean(0)
                for r in new: W[r] = (mu + 0.02 * torch.randn(mu.shape, generator=g)).to(W.dtype)
        self.config.sp_ids.update({n: ids[n] for n in PHASE2_SPECIALS}); self.config.special_tokens = list(dict.fromkeys(self.config.special_tokens + PHASE2_SPECIALS)); self.sp_ids = dict(self.config.sp_ids)
        self.config.phase2_registry = {n: ids[n] for n in LANE_TOKENS[:R] + ["<ONSET>", "<EOT>"]}; self.config.lanes = R; self.config.sem_reserved = []
        self.config.blocked_ids = sorted(set(self.config.blocked_ids) - {ids["<SPK_A>"], ids["<SPK_B>"]} - {ids[n] for n in PHASE2_SPECIALS}); self.blocked = torch.tensor(self.config.blocked_ids, dtype=torch.long, device=W.device)
        if self.act_head is None: self.act_head = nn.Sequential(nn.Linear(self.config.hidden_size, self.config.act_hidden), nn.GELU(), nn.Linear(self.config.act_hidden, R)).to(W.device)
        return ids

    def add_semantic_tokens(self, tok, seed: int = 0, block_reserved_phase2: bool = True, check_frozen: bool = True, turn: bool = False,
                            reinit_turn: bool = False) -> Dict[str, int]:
        """semantic commit 토큰 <SEM_END> 를 더한다(add_phase2_tokens 와 같은 방식, vapasr/data/semcommit_tokens.py).
        tokenizer 에 Phase 1 + Phase 2 + SEM 을 이 순서로 추가 → SEM=151723 (check_frozen: 동결 id 대조 — 실제 Qwen tokenizer 에서만 의미).
        config.sp_ids 에 없던 행만 mean(W[:기본 어휘]) + 0.02·randn(seed) 으로 초기화(기존 Phase 1/2 행은 그대로). mono(lanes=0) 에서는 Phase 2 id 가 예약만 되므로
        block_reserved_phase2 면 decode 차단 목록에 넣는다(학습되지 않은 행 방출 방지). SEM 은 차단하지 않는다. 반복 호출해도 결과가 같다(idempotent).
        turn(mono 전용): 턴 종료를 <EOT>(Phase 2 와 같은 토큰)로 함께 학습 → config.sem_registry 에 <EOT> 를 넣고 차단·예약에서 뺀다(<TURN_END> 는 폐기).
        mono 에서 이번에 예약만 한 Phase 2 이름은 config.sem_reserved 에 적는다 → 나중에 add_phase2_tokens 가 그 행을 다시 초기화한다.
        reinit_turn(turn 과 함께): <EOT> 가 예약(sem_reserved)만 돼 있던 행이면 mean+noise 로 다시 초기화한다 — 예약 행은 tied lm_head 의 full-vocab
        softmax 음성으로만 학습돼 logit 이 깊이 눌려 있다(Stage 1·2 체크포인트에서 Stage 3 <EOT> 를 켤 때). 한 번 켠 뒤에는 예약이 아니므로 다시 하지 않는다."""
        from ..data.semcommit_tokens import add_semantic_specials, assert_frozen_semantic, init_rows_mean_noise, semantic_blocked_ids, new_special_rows, PHASE2_SPECIALS, SEM_SPECIALS, BASE_VOCAB, TURN_TOKEN
        assert not (turn and self.config.lanes > 0), "turn 은 mono 전용 — Phase 2 모델은 <EOT> 를 phase2_registry 로 이미 학습한다"
        ids = add_semantic_specials(tok)
        if check_frozen: assert_frozen_semantic(ids)
        W = self.get_input_embeddings().weight; assert max(ids.values()) < W.shape[0], f"임베딩 여유 행 없음: max id {max(ids.values())} ≥ rows {W.shape[0]}"
        old_res = set(getattr(self.config, "sem_reserved", None) or [])
        stale = [ids[TURN_TOKEN]] if (turn and reinit_turn and self.config.lanes == 0 and TURN_TOKEN in old_res) else []
        init_rows_mean_noise(W, sorted(set(new_special_rows(ids, self.config.sp_ids)) | set(stale)), upto=min(BASE_VOCAB, min(ids.values())), seed=seed)
        self.config.sem_reserved = [n for n in PHASE2_SPECIALS if (n not in self.config.sp_ids or n in old_res) and not (turn and n == TURN_TOKEN)] if self.config.lanes == 0 else []
        self.config.sp_ids.update(ids); self.config.special_tokens = list(dict.fromkeys(self.config.special_tokens + PHASE2_SPECIALS + SEM_SPECIALS)); self.sp_ids = dict(self.config.sp_ids)
        self.config.sem_registry = {n: ids[n] for n in SEM_SPECIALS + ([TURN_TOKEN] if turn else [])}
        self.config.blocked_ids = semantic_blocked_ids(self.config.blocked_ids, ids, self.config.lanes, block_reserved_phase2, turn=turn); self.blocked = torch.tensor(self.config.blocked_ids, dtype=torch.long, device=W.device)
        return ids

    # ── 스트리밍 디코드 (mono_model.stream_decode 와 동일)
    @torch.inference_mode()
    def stream_decode(self, feats, prefix_ids: List[int], max_per_chunk: int = 0, next_bias: float = 0.0, max_flush_rounds: Optional[int] = None, runaway_cap: Optional[int] = None, max_total_per_chunk: float = 6.0):
        """feats (K,Din) 또는 wav (T,) → ([(chunk k, token id)], forced_next 횟수, tick_ms 목록, flush 라운드 수).
        폭주 방지: 청크당 runaway_cap(config, 기본 8) 토큰 · 스트림 전체 max_total_per_chunk×K 토큰(부분 학습 모델의 free-running 디코드가 수만 step 으로 늘어지던 문제)."""
        from transformers import DynamicCache
        max_flush_rounds = self.config.max_flush_rounds if max_flush_rounds is None else max_flush_rounds; runaway_cap = self.config.runaway_cap if runaway_cap is None else runaway_cap
        if feats.dim() == 1:
            K = torch.tensor([int(round(feats.shape[0] / 16000 * self.config.frame_hz))], device=feats.device)
            feats = self.encode(feats[None], torch.tensor([feats.shape[0]], device=feats.device), K)[0]
        emb = self.get_input_embeddings(); dev = feats.device; cache = DynamicCache(); ce = self.chunk_embed(feats[None])[0].to(emb.weight.dtype)
        def step(e):
            out = self.thinker(inputs_embeds=e.view(1, -1, e.shape[-1]), past_key_values=cache, use_cache=True); return out.logits[0, -1].float()
        step(emb(torch.tensor(prefix_ids, device=dev))); e_next = emb.weight[self.next_audio]; e_empty = emb.weight[self.empty_audio]
        out, forced, ticks = [], 0, []; K = ce.shape[0]; max_total = int(max_total_per_chunk * K) + 64; cap = min(max_per_chunk, runaway_cap) if max_per_chunk else runaway_cap
        def emit_round(k, logits):
            nonlocal forced; n = 0
            while True:
                logits[self.blocked] = float("-inf"); logits[self.next_audio] -= next_bias
                tid = int(logits.argmax())
                if tid == self.next_audio or n >= cap or len(out) >= max_total:
                    forced += int(tid != self.next_audio); step(e_next); return n
                out.append((k, tid)); n += 1; logits = step(emb.weight[tid])
        for k in range(K):
            t = time.time(); emit_round(k, step(ce[k])); torch.cuda.synchronize(); ticks.append((time.time() - t) * 1000)
        rounds = 0
        for r in range(max_flush_rounds):
            rounds += 1
            if emit_round(K + r, step(e_empty)) == 0: break
        return out, forced, ticks, rounds

    # ── 저장/로드
    def save_pretrained(self, save_directory, *args, state_dict=None, **kw):
        """동결 인코더는 저장하지 않는다(.nemo 에서 재구성). 학습한 인코더(encoder_trainable) 또는 encoder_saved(초기화 체크포인트에서 가져온, .nemo 와 다른 인코더) 는 함께 저장.
        (2026-09-17 D1 사고: E2 의 학습된 인코더를 동결해 쓰면서 encoder_trainable=False 로 저장 → 재로드 시 .nemo 원본 인코더가 붙어 text 손실 0.08→1.5. encoder_saved 로 막는다.)"""
        sd = state_dict if state_dict is not None else self.state_dict()
        if not (self.config.encoder_trainable or getattr(self.config, "encoder_saved", False)): sd = {k: v for k, v in sd.items() if not k.startswith("encoder.")}
        return super().save_pretrained(save_directory, *args, state_dict=sd, **kw)

    @classmethod
    def from_pretrained(cls, path, *args, encoder_path: Optional[str] = None, load_encoder: bool = True, **kw):
        m = super().from_pretrained(path, *args, **kw)
        if load_encoder:
            enc_sd = None
            if m.config.encoder_trainable or getattr(m.config, "encoder_saved", False):     # 저장된 인코더 가중치를 safetensors 에서 읽어 붙인다
                from safetensors.torch import load_file
                import glob
                enc_sd = {}
                for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
                    for k, v in load_file(f).items():
                        if k.startswith("encoder.enc."): enc_sd[k[len("encoder.enc."):]] = v
            m.attach_encoder(encoder_path, state_dict=enc_sd or None)
        return m

    @classmethod
    def from_qwen(cls, qwen_dir: str, tokenizer=None, encoder_path: Optional[str] = None, load_encoder: bool = True, seed: int = 0, **cfg_kw):
        """Qwen3-ASR 디렉토리(0.6B·1.7B)에서 새 모델. 특수 토큰을 tokenizer 에 추가하고 그 임베딩 행을 평균+잡음으로 초기화(mono_model 과 동일). adapter 는 random,
        출력 차원은 thinker hidden(1.7B = 2048)."""
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
        from transformers import AutoTokenizer
        from ..uslm.interleave_data import add_specials, SPECIAL_TOKENS
        tok = tokenizer or AutoTokenizer.from_pretrained(qwen_dir)
        sp_ids = add_specials(tok)
        full = Qwen3ASRForConditionalGeneration.from_pretrained(qwen_dir, dtype=torch.float32); thinker = full.thinker; del full
        c = thinker.config; pre = tok("<|im_start|>system\n<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
        blocked = [c.audio_token_id, c.audio_start_token_id, c.audio_end_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|im_start|>"),
                   tok.convert_tokens_to_ids("<asr_text>"), sp_ids["<EMPTY_AUDIO>"], sp_ids["<SPK_A>"], sp_ids["<SPK_B>"]] + [v for k, v in sp_ids.items() if k.startswith("<DELAY_")]
        cfg_kw.setdefault("adapter_d_out", int(c.to_dict().get("text_config", {}).get("hidden_size", 1024)))   # 0.6B 1024, 1.7B 2048 — adapter 출력 = thinker 임베딩 차원
        config = VapAsrConfig(thinker=c.to_dict(), thinker_name_or_path=qwen_dir, sp_ids=sp_ids, special_tokens=list(SPECIAL_TOKENS), blocked_ids=sorted(set(blocked)),
                              audio_pad_id=c.audio_token_id, prefix_ids=pre, **cfg_kw)
        assert config.adapter_d_out == config.hidden_size, f"adapter_d_out {config.adapter_d_out} ≠ thinker hidden {config.hidden_size}"
        m = cls(config, thinker=thinker)
        W = m.get_input_embeddings().weight; rows = sorted(sp_ids.values()); assert max(rows) < W.shape[0], "임베딩에 특수 토큰 여유 행 없음"
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            mu = W[: len(tok) - len(sp_ids)].float().mean(0)
            for r in rows: W[r] = (mu + 0.02 * torch.randn(mu.shape, generator=g)).to(W.dtype)
        for p in m.thinker.parameters(): p.requires_grad_(True)
        if load_encoder: m.attach_encoder(encoder_path)
        return m, tok

    @classmethod
    def from_legacy(cls, ckpt_path: str, qwen_dir: str, tokenizer=None, encoder_path: Optional[str] = None, load_encoder: bool = True, **cfg_kw):
        """experiments/s1_train_mono.py 산출물(ckpt-last.pt / ckpt-<step>.pt, full FT) → HF 모델."""
        st = torch.load(ckpt_path, map_location="cpu"); st = st["model"] if "model" in st and "adapter" not in st else st
        assert st.get("full_ft") and st.get("thinker"), "full FT ckpt 만 변환(LoRA 는 미지원)"
        m, tok = cls.from_qwen(qwen_dir, tokenizer, encoder_path, load_encoder=False, **cfg_kw)
        assert dict(st["sp_ids"]) == m.config.sp_ids, f"특수 토큰 id 불일치: {st['sp_ids']} vs {m.config.sp_ids}"
        m.adapter.load_state_dict(st["adapter"])
        missing, unexpected = m.thinker.load_state_dict({k: v.float() for k, v in st["thinker"].items()}, strict=False)
        missing = [k for k in missing if not k.startswith("audio_tower")]; assert not missing and not unexpected, (missing[:5], unexpected[:5])
        if st.get("encoder"): m.config.encoder_trainable = True
        if load_encoder: m.attach_encoder(encoder_path, state_dict=st.get("encoder"))
        return m, tok

    # ── 기존 학습 코드와의 호환(load_trainable_state 형식)
    def trainable_state(self):
        enc = {"encoder": {k: v.detach().cpu() for k, v in self.encoder.enc.state_dict().items()}} if (self.encoder is not None and self.config.encoder_trainable) else {}
        if self.adapter_tap is not None: enc["adapter_tap"] = self.adapter_tap.state_dict()
        if self.adapter_inject is not None: enc["adapter_inject"] = self.adapter_inject.state_dict()
        if self.adapter_slot is not None: enc["adapter_slot"] = self.adapter_slot.state_dict()
        return dict(adapter=self.adapter.state_dict(), thinker={k: v.detach().to(torch.bfloat16).cpu() for k, v in self.thinker.state_dict().items()}, sp_ids=self.sp_ids, mono=True, full_ft=True, **enc)
