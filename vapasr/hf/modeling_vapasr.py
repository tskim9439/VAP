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
BEGIN_TOKEN = "<ONSET>"                                         # E9 발화 시작 표지 <BEGIN>(동결 registry 151721) — add_slot_blocks
X_TOKEN = "<SPK_5>"                                             # E14 비귀속 칸 태그(2026-11-03) — 4 슬롯 모델이 쓰지 않는 예약 토큰, add_x_slot
TAG_STRIDE = 1 << 20                                            # E9 packing 의 tag_chunk = 표본 · TAG_STRIDE + 청크(cst_dialogue.TAG_STRIDE 와 같다)

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
    loss_diar: Optional[float] = None       # E6: 인과 화자 분리 헤드 Sort·PIT 손실(가중 전)
    diar_acc: Optional[float] = None        # E6: 헤드 프레임 정확도(임계 config.act_threshold, 도착 순서)
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
    _keys_to_ignore_on_load_unexpected = [r"^encoder\.", r"^thinker\.audio_tower\.", r"^n3d_net\."]      # n3d_net: 흡수 N3D 가지(from_pretrained 가 따로 읽는다)
    _keys_to_ignore_on_load_missing = [r"^encoder\.", r"^n3d_net\."]

    def __init__(self, config: VapAsrConfig, thinker: Optional[nn.Module] = None):
        super().__init__(config)
        self.adapter = Adapter(config.adapter_d_in, config.adapter_d_out, config.adapter_hidden)
        self.adapter_tap = Adapter(config.adapter_d_in, config.adapter_d_out, config.enc_tap_hidden) if getattr(config, "enc_tap_layer", -1) >= 0 else None   # V2-a(이름이 adapter 로 시작 → adapter lr 그룹)
        h = int(getattr(config, "spk_inject_hidden", 0) or 0)                                    # E2 대상 화자 활동 주입(이름이 adapter 로 시작 → adapter lr 그룹)
        self.adapter_inject = nn.Sequential(nn.Linear(config.adapter_d_in, h), nn.GELU(), nn.Linear(h, config.adapter_d_in)) if h > 0 else None
        n = int(getattr(config, "slot_input_slots", 0) or 0)                                      # E5 슬롯 활동 입력(오디오 토큰 += Σ a_s·e_s, 이름이 adapter 로 시작 → adapter lr 그룹)
        self.adapter_slot = nn.Linear(n, config.adapter_d_out, bias=False) if n > 0 else None
        n = int(getattr(config, "tag_act_lags", 0) or 0)                                         # E9 태그 활동 입력(태그 임베딩 += W·[a_s(k) … a_s(k − n + 1)], 이름이 adapter 로 시작 → adapter lr 그룹)
        self.adapter_tagact = nn.Linear(n, config.adapter_d_out, bias=False) if n > 0 else None
        sa = getattr(config, "slot_audio_cfg", None)                                               # E10 슬롯 추출기(vapasr/hf/slot_audio — 슬롯별 소리 토큰을 태그 자리에, adapter lr 그룹)
        if sa:
            from .slot_audio import SlotAudio
            self.adapter_slotx = SlotAudio(**sa); self.adapter_mixnull = nn.Parameter(torch.zeros(config.adapter_d_out))   # 섞인 소리 토큰을 지울 때 넣는 빈 오디오 임베딩
        else: self.adapter_slotx = None; self.adapter_mixnull = None
        dh = getattr(config, "diar_head_cfg", None)                                              # E6 인과 화자 분리 헤드(인코더 탭 위, 같은 그래프 — 이름이 adapter 가 아니라 thinker lr 그룹)
        if dh:
            from .diar import DiarBranch
            self.diar_head = DiarBranch(**dh)
        else: self.diar_head = None
        nc = getattr(config, "n3d_cfg", None)                                                    # E11 고정 화자 분리 전문가(vapasr/hf/n3d — nn.Module 이 아니라 state_dict·DDP 밖, 처음 쓸 때 읽는다)
        if nc:
            from .n3d import N3DExpert
            self.n3d = N3DExpert(nc)
        else: self.n3d = None
        self.n3d_net = None                                                                       # 흡수(absorb_n3d)한 N3D 가지 — 하위 모듈이라 체크포인트에 함께(from_pretrained 가 다시 세운다)
        if thinker is None:
            from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRThinkerConfig
            from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRThinkerForConditionalGeneration
            thinker = Qwen3ASRThinkerForConditionalGeneration(Qwen3ASRThinkerConfig.from_dict(dict(config.thinker)))
        if hasattr(thinker, "audio_tower"): del thinker.audio_tower               # 오디오 타워는 쓰지 않는다(Nemotron 이 대신)
        self.thinker = thinker
        self.encoder: Optional[nn.Module] = None                                  # attach_encoder 로 붙인다(무거운 NeMo 복원은 __init__ 밖에서)
        self.sp_ids = dict(config.sp_ids); self.next_audio = self.sp_ids.get("<NEXT_AUDIO>"); self.empty_audio = self.sp_ids.get("<EMPTY_AUDIO>")
        if getattr(config, "slot_blocks", False):                                            # E9 <BEGIN>(<ONSET>)은 디코드에서 막지 않는다 — 막힌 채 저장된 체크포인트
            bid = int(config.sp_ids.get(getattr(config, "begin_token", None) or BEGIN_TOKEN, -1))   # (E10 final·E11 — add_slot_blocks 수정 전 학습)도 읽을 때 푼다(2026-10-30)
            if bid in set(config.blocked_ids): config.blocked_ids = sorted(set(config.blocked_ids) - {bid})
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
        self.encoder = enc.to(next(self.adapter.parameters()).device)
        if getattr(self, "n3d_net", None) is not None and self.config.n3d_cfg.get("shared_mel"): self.n3d.pre = self._shared_mel(force=True)   # 흡수 N3D 가 우리 mel 을 쓰면 새 인코더 것으로
        return self
    def encode(self, wav, wav_len, K, act=None, n3d: bool = True):               # (B,L) → (B,S,Kmax,Din) — S = 1, 탭이 있으면 2(마지막 층, 탭 층)
        """act (B, ≥Kmax) 대상 화자 활동(0–1, E2) — 있으면 adapter_inject 로 인코더 pre-encode 출력에 주입한다(없으면 예전 그대로).
        E6(화자 분리 헤드가 있으면): (B, 2, Kmax, Din) — 스트림 1 = 헤드 예측 슬롯 활동(앞 N 차원, 나머지 0). chunk_embed 가 그 스트림을 슬롯 활동 입력으로 쓴다
        → 디코드 경로(decode_batch·stream_decode·realtime·회의 통째)가 그대로 헤드 예측을 쓴다.
        E10(슬롯 추출기가 있으면): (B, 3, Kmax, Din) — 스트림 2 = 추출기의 화자 단서 임베딩(adapter_slotx.embed, 앞 d_emb 차원) — slot_audio_of 가 쓴다.
        E11(고정 화자 분리 전문가 n3d 가 있고 n3d=True 면): 스트림 1 을 전문가 활동(이 오디오로 스트리밍, 앞 N 채널 0/1)으로 바꾼다(헤드가 없으면 스트림 1 을 더한다).
        회의 통째(long_session.encode_long)는 조각마다 n3d=False 로 부르고 회의 전체로 한 번 바꾼다(화자 캐시가 회의 내내 이어지게)."""
        assert self.encoder is not None, "encoder 없음 — attach_encoder() 먼저"
        if self.n3d is not None and act is None and n3d:
            f = self.encode(wav, wav_len, K, n3d=False); a = self.n3d_act(wav, wav_len, K).to(f.dtype)
            return self.with_act_stream(f, a)
        if self.diar_head is not None and act is None:
            if self.adapter_slotx is not None:
                f, dlog, e = self.encode_diar(wav, wav_len, K, with_emb=True); a = self.predict_act(dlog).to(f.dtype)
                return torch.cat([f, F.pad(a, (0, f.shape[-1] - a.shape[-1]))[:, None], F.pad(e.to(f.dtype), (0, f.shape[-1] - e.shape[-1]))[:, None]], 1)
            f, dlog = self.encode_diar(wav, wav_len, K); a = self.predict_act(dlog).to(f.dtype)
            return torch.cat([f, F.pad(a, (0, f.shape[-1] - a.shape[-1]))[:, None]], 1)
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
    def chunk_embed(self, feats, act=None, mix_mask=None):                       # (B,S,K,Din) → (B,K,D)
        """act (B, K', ≤n) 슬롯 활동(0–1, E5 — 슬롯 순서 = 창에서 먼저 말한 순서) — 있으면 오디오 토큰에 adapter_slot(act) = Σ a_s·e_s 를 더한다(K' ≠ K 면 0 채움·자름).
        mix_mask (B,) bool(E10): True 인 행은 섞인 소리(어댑터 출력)를 빈 오디오 임베딩 adapter_mixnull 로 바꾼다(활동 몫 adapter_slot 은 그대로)."""
        ce = self.adapter(feats[:, 0].float())
        if self.adapter_tap is not None:
            assert feats.shape[1] >= 2, f"두 스트림 모델(enc_tap_layer {self.config.enc_tap_layer})인데 특징 스트림이 {feats.shape[1]} 개 — model.encode 로 만든 특징을 쓴다"
            ce = ce + self.adapter_tap(feats[:, 1].float())
        if mix_mask is not None and self.adapter_mixnull is not None:
            ce = torch.where(mix_mask.to(ce.device)[:, None, None], self.adapter_mixnull.to(ce.dtype)[None, None].expand_as(ce), ce)
        if act is None and self.diar_head is not None and feats.shape[1] >= 2: act = feats[:, 1, :, : self.adapter_slot.in_features]   # E6 encode 의 예측 활동 스트림
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

    def add_slot_blocks(self, tok, lags: int = 12, seed: int = 0) -> int:
        """E9 슬롯 블록 형식을 켠다(2026-10-28 — wiki output-diarization-integration-20261025 9 절, cst_dialogue 모듈 설명 E9).
        <BEGIN> = 동결 registry 의 <ONSET>(151721, 새 어휘 없음): mono 체크포인트에서 예약만 된 행(config.sem_reserved)은 평균 + 잡음으로 다시 초기화하고
        decode 차단·예약 표시에서 뺀다(enable_slots 와 같은 이유). adapter_tagact = Linear(lags, D, bias 없음) 0 초기화 — 태그 위치 임베딩에 그 슬롯의 최근 lags 청크
        활동 [a_s(k), a_s(k − 1), …] 을 더한다(붙인 직후 태그 임베딩은 그대로). 반복 호출은 아무것도 바꾸지 않는다(lags 가 다르면 멈춘다). → <BEGIN> id."""
        from ..data.semcommit_tokens import init_rows_mean_noise, BASE_VOCAB
        from ..data.dialogue_tokens import FROZEN_REGISTRY
        bid = int(tok.convert_tokens_to_ids(BEGIN_TOKEN)); assert bid == FROZEN_REGISTRY[BEGIN_TOKEN], f"{BEGIN_TOKEN} id {bid} ≠ 동결 registry {FROZEN_REGISTRY[BEGIN_TOKEN]}"
        if self.adapter_tagact is not None:
            assert int(self.config.tag_act_lags) == int(lags), f"이미 태그 활동 {self.config.tag_act_lags} 청크 — {lags} 로 바꾸지 않는다"
            if bid in set(self.config.blocked_ids):                                             # 2026-10-30 E10 사고: 슬롯 블록 init(E9 final)에서 stage3_train 의
                self.config.blocked_ids = sorted(set(self.config.blocked_ids) - {bid})         # add_semantic_tokens 가 <ONSET> 을 다시 막았는데 여기서 일찍 돌아가 풀지 못했다
                self.blocked = torch.tensor(self.config.blocked_ids, dtype=torch.long, device=self.blocked.device)   # → 디코드가 <BEGIN> 을 못 내 새 턴을 시작하지 못했다
            return bid
        W = self.get_input_embeddings().weight; assert bid < W.shape[0], "임베딩 여유 행 없음"
        if BEGIN_TOKEN in set(getattr(self.config, "sem_reserved", None) or []): init_rows_mean_noise(W, [bid], upto=BASE_VOCAB, seed=seed)
        self.config.sem_reserved = [t for t in (self.config.sem_reserved or []) if t != BEGIN_TOKEN]
        self.config.sp_ids[BEGIN_TOKEN] = bid; self.sp_ids = dict(self.config.sp_ids)
        self.config.blocked_ids = sorted(set(self.config.blocked_ids) - {bid}); self.blocked = torch.tensor(self.config.blocked_ids, dtype=torch.long, device=W.device)
        self.adapter_tagact = nn.Linear(int(lags), self.config.adapter_d_out, bias=False).to(next(self.adapter.parameters()).device)
        with torch.no_grad(): self.adapter_tagact.weight.zero_()
        self.config.slot_blocks, self.config.begin_token, self.config.tag_act_lags = True, BEGIN_TOKEN, int(lags)
        return bid

    def add_x_slot(self, tok) -> int:
        """E14 비귀속 칸을 켠다(2026-11-03 — cst_dialogue 모듈 설명 x_slot): X 태그 = 예약 토큰 <SPK_5>(새 어휘 없음). 처음 켤 때 그 입력 임베딩 행을 슬롯 태그 넷의
        평균으로 둔다(태그 꼴 입력에서 시작). 출력 차단은 그대로다(X 태그는 입력만). 활동 게이트 슬롯 블록(E13) 위에만. 반복 호출은 아무것도 바꾸지 않는다. → X 태그 id."""
        xid = int(tok.convert_tokens_to_ids(X_TOKEN)); assert xid == int(self.sp_ids.get(X_TOKEN, xid)), f"{X_TOKEN} id {xid} ≠ config {self.sp_ids.get(X_TOKEN)}"
        assert getattr(self.config, "slot_blocks", False) and int(getattr(self.config, "slot_gate", 0) or 0) > 0, "E14 비귀속 칸은 활동 게이트 슬롯 블록(config.slot_gate) 위에"
        if getattr(self.config, "x_slot", False): return xid
        W = self.get_input_embeddings().weight; assert xid < W.shape[0], "임베딩 여유 행 없음"
        with torch.no_grad(): W[xid] = W[torch.tensor(self.slot_ids(), device=W.device)].mean(0)
        self.config.sp_ids[X_TOKEN] = xid; self.sp_ids = dict(self.config.sp_ids)
        self.config.sem_reserved = [t for t in (self.config.sem_reserved or []) if t != X_TOKEN]; self.config.x_slot = True
        return xid

    @property
    def x_id(self) -> int:
        """E14 X 태그 id(config.x_slot), 없으면 −1."""
        return int(self.sp_ids[X_TOKEN]) if getattr(self.config, "x_slot", False) else -1

    def slot_ids(self, n: Optional[int] = None) -> List[int]:
        """슬롯 태그 id(A, B, C, D 순) — config.slot_tokens(enable_slots) 또는 A·B."""
        names = list(getattr(self.config, "slot_tokens", None) or SLOT_TOKENS[:2]); return [int(self.sp_ids[t]) for t in names[: (n or len(names))]]

    def tag_act_embed(self, act, b, k, s):
        """E9: 태그 (표본 b, 청크 k, 슬롯 s)(각 (n,) long) → adapter_tagact([a_s(k), a_s(k − 1), …, a_s(k − lags + 1)]) (n, D). act (B, K', N) 0–1 — 범위 밖 청크·슬롯은 0
        (flush 라운드 k ≥ K' 는 앞 청크 활동만 남는다)."""
        Wn = self.adapter_tagact.in_features; Kp, N = act.shape[1], act.shape[-1]
        j = k[:, None] - torch.arange(Wn, device=k.device)[None]
        ok = (j >= 0) & (j < Kp) & (s < N)[:, None]
        v = act[b[:, None], j.clamp(0, Kp - 1), s.clamp(max=N - 1)[:, None]].float() * ok
        return self.adapter_tagact(v.to(self.adapter_tagact.weight.dtype))

    def tag_embeds(self, k: int, act, n: int, X=None):
        """E9 디코드: 청크(라운드) k 의 태그 n 개 입력 임베딩 (n, D) — 태그 행 + 그 슬롯 활동(act (K, N) 0/1, None 이면 태그 행만)
        + E10 슬롯 소리 토큰 X (K, N, D)[k](k ≥ K — flush — 는 더하지 않는다)."""
        emb = self.get_input_embeddings().weight; E = emb[torch.tensor(self.slot_ids(n), device=emb.device)]
        if self.adapter_tagact is not None and act is not None:
            s = torch.arange(n, device=emb.device)
            E = E + self.tag_act_embed(act[None].to(emb.device), torch.zeros_like(s), torch.full_like(s, int(k)), s).to(E.dtype)
        if X is not None and int(k) < X.shape[0]: E = E + X[int(k), :n].to(device=E.device, dtype=E.dtype)
        return E

    def add_slot_audio(self, d: int = 512, blocks: int = 4, kernel: int = 15, ffn: int = 2048, mix_drop: float = 0.5) -> None:
        """E10 슬롯 추출기를 켠다(2026-10-29 — wiki output-diarization-integration-20261025 10 절, vapasr/hf/slot_audio): 인코더 마지막 층 + 화자 단서 임베딩 + 슬롯 활동 →
        슬롯별 소리 토큰(태그 자리에 더한다), 출력 Linear 0 초기화 → 붙인 직후 출력은 그대로. adapter_mixnull = 섞인 소리 토큰을 지울 때의 빈 임베딩(0 초기화).
        화자 단서 임베딩(추출기 embed — 탭 가중합·LN·투영)은 지금 헤드의 embed 값으로 시작한다(헤드를 나중에 바꿔 끼워도 추출기 입력은 그대로).
        mix_drop: 학습 표본마다 이 확률로 섞인 소리 토큰을 지운다(config.slot_mix_drop). 슬롯 블록(E9)·화자 분리 헤드(E6)가 있어야 한다. 이미 있으면 같은 설정인지만 본다."""
        assert getattr(self.config, "slot_blocks", False) and self.diar_head is not None, "E10 은 슬롯 블록(add_slot_blocks)과 화자 분리 헤드(add_diar_head) 뒤에"
        from .slot_audio import SlotAudio
        hc = self.diar_head.cfg
        cfg = dict(n_streams=int(hc["n_streams"]), d_enc=int(hc["d_in"]), d_emb=int(hc["d"]), d=int(d), blocks=int(blocks), kernel=int(kernel), ffn=int(ffn), d_out=int(self.config.adapter_d_out))
        assert cfg["d_emb"] <= int(self.config.adapter_d_in) == cfg["d_enc"], f"화자 단서 임베딩 {cfg['d_emb']} 차원은 인코더 특징 {self.config.adapter_d_in} 차원 안에 담겨야 한다(encode 셋째 스트림)"
        self.config.slot_mix_drop = float(mix_drop)
        if self.adapter_slotx is not None:
            assert dict(self.config.slot_audio_cfg) == cfg, f"이미 다른 슬롯 추출기 {self.config.slot_audio_cfg} — {cfg}"; return
        dev = next(self.adapter.parameters()).device
        self.adapter_slotx = SlotAudio(**cfg).to(dev); self.adapter_mixnull = nn.Parameter(torch.zeros(self.config.adapter_d_out, device=dev))
        with torch.no_grad():                                                                     # 화자 단서 임베딩 = 지금 헤드의 embed 에서 시작
            self.adapter_slotx.mix.copy_(self.diar_head.mix); self.adapter_slotx.ln_t.load_state_dict(self.diar_head.ln.state_dict())
            self.adapter_slotx.proj_t.load_state_dict(self.diar_head.proj.state_dict())
        self.config.slot_audio_cfg = cfg

    def slot_audio_of(self, feats, act, seg: int = 4096):
        """E10 디코드: feats (S ≥ 3, K, D)(model.encode·encode_long — 스트림 0 내용·2 헤드 임베딩), act (K, N) → 슬롯 소리 토큰 (K, N, D) 또는 None(E10 아님)."""
        if self.adapter_slotx is None or act is None: return None
        assert feats.dim() == 3 and feats.shape[0] >= 3, f"E10 특징은 (3, K, D)(내용·활동·헤드 임베딩) — {tuple(feats.shape)}"
        de = int(self.config.slot_audio_cfg["d_emb"])
        return self.adapter_slotx(feats[0][None], feats[2][None, :, :de], act[None].to(feats.device), seg=seg)[0]

    def block_act(self, feats, act=None):
        """E9 디코드의 태그 활동 (K, N): act 를 주면 그것, 화자 분리 헤드(E6) 특징이면 예측 활동 스트림(chunk_embed 와 같은 것), 아니면 None. feats (S, K, D)."""
        if act is not None: return act
        if self.diar_head is not None and self.adapter_slot is not None and feats.shape[0] >= 2: return feats[1, :, : self.adapter_slot.in_features]
        return None

    def _tag_add(self, E, ids, tag_chunk, act, packed: bool, slot_x=None):
        """build: 태그 위치(tag_chunk ≥ 0) 임베딩에 tag_act_embed 를 더한다(act 없으면 그대로). packing 이면 tag_chunk = 표본 · TAG_STRIDE + 청크.
        slot_x (B, Kx, N, D)(E10): 태그 (b, k, s) 에 slot_x[b, k, s] 를 더한다(k ≥ Kx 는 0)."""
        sel = tag_chunk >= 0
        if (act is None and slot_x is None) or not bool(sel.any()): return E
        tc = tag_chunk[sel]
        b = tc // TAG_STRIDE if packed else torch.arange(tag_chunk.shape[0], device=tc.device)[:, None].expand_as(tag_chunk)[sel]
        k = tc % TAG_STRIDE if packed else tc
        s = (ids[sel][:, None] == torch.tensor(self.slot_ids(), device=ids.device)[None]).long().argmax(-1)
        add = 0
        if act is not None: self._tagact_used = True; add = self.tag_act_embed(act.to(E.device), b, k, s).to(E.dtype)
        if slot_x is not None:
            Kx, Nx = slot_x.shape[1], slot_x.shape[2]; ok = (k < Kx) & (s < Nx)
            add = add + slot_x[b, k.clamp(max=Kx - 1), s.clamp(max=Nx - 1)].to(E.dtype) * ok[:, None].to(E.dtype); self._slotx_used = True
        return E.index_put((sel,), E[sel] + add)

    def add_diar_head(self, cfg: dict, taps, state: Optional[dict] = None, weight: float = 0.5, alpha: float = 0.5, threshold: float = 0.45, ramp=(0.3, 0.9)) -> None:
        """E6(2026-10-26 — output-diarization-integration-20261025 6 절): 인과 화자 분리 헤드(vapasr.hf.diar.DiarBranch, cfg 의 lookahead)를 인코더 탭(taps — 층 번호,
        스트림 0 = 마지막 층) 위에 붙인다. state(E1c diar.pt 의 state)가 있으면 불러온다. 이미 있으면 같은 cfg·탭인지만 본다(재개)."""
        assert int(getattr(self.config, "enc_tap_layer", -1)) < 0, "E6 은 두 스트림 입력(enc_tap_layer)과 함께 쓰지 않는다"
        if self.diar_head is not None:
            assert dict(self.config.diar_head_cfg) == dict(cfg) and list(self.config.diar_taps) == list(taps), "이미 다른 화자 분리 헤드가 있다"; return
        from .diar import DiarBranch
        self.diar_head = DiarBranch(**cfg).to(next(self.adapter.parameters()).device)
        if state is not None: self.diar_head.load_state_dict(state)
        self.config.diar_head_cfg, self.config.diar_taps = dict(cfg), [int(x) for x in taps]
        self.config.diar_weight, self.config.diar_alpha, self.config.act_threshold, self.config.act_pred_ramp = float(weight), float(alpha), float(threshold), [float(x) for x in ramp]

    def encode_diar(self, wav, wav_len, K, with_emb: bool = False):
        """E6: 인코더를 한 번 돌려 (ASR 특징 (B, 1, Kmax, D), 화자 분리 logits (B, Kmax, N)) — 헤드는 탭 스트림 전부(0 = 마지막 층)를 본다.
        with_emb(E10): 슬롯 추출기의 화자 단서 임베딩 (B, Kmax, d_emb)(adapter_slotx.embed — 같은 탭 스트림에서)도 함께 → 셋."""
        assert self.encoder is not None and self.diar_head is not None
        allf = self.encoder(wav, wav_len, K, taps=tuple(self.config.diar_taps))
        pad = torch.arange(allf.shape[2], device=allf.device)[None] >= K.to(allf.device)[:, None]
        if with_emb: return allf[:, :1], self.diar_head(allf, pad), self.adapter_slotx.embed(allf)
        return allf[:, :1], self.diar_head(allf, pad)

    def predict_act(self, logits):
        """화자 분리 logits → 0/1 슬롯 활동(임계 config.act_threshold)."""
        return (torch.sigmoid(logits.float()) > float(self.config.act_threshold)).float()

    def add_n3d(self, path: Optional[str] = None, **cfg) -> None:
        """E11(2026-10-29 — wiki output-n3d-branch-20261029): 고정 화자 분리 전문가 Nemotron 3 Diarization 을 붙인다(vapasr/hf/n3d — 가중치는 체크포인트 밖, config 의 경로).
        활동(도착 순서 앞 N 채널, N = 슬롯 활동 입력 수)이 헤드 예측 대신 슬롯 활동 입력이 된다 — 학습은 표본마다 확률 ρ 로 전문가 활동(정답 슬롯 자리로 맞춤),
        아니면 흔든 정답. cfg: n3d.DEFAULTS 키(chunk·rc 추론, train_chunk·train_rc 학습, fifo·cache·period, threshold, dtype). 다시 부르면 설정을 바꾼다(평가에서 경로·청크)."""
        assert int(getattr(self.config, "slot_input_slots", 0) or 0) > 0, "고정 화자 분리 전문가는 슬롯 활동 입력(add_slot_input)과 함께"
        assert self.adapter_tap is None and self.adapter_inject is None, "E11 은 두 스트림 입력(enc_tap)·대상 화자 주입(E2)과 함께 쓰지 않는다"
        assert not int(getattr(self.config, "act_shift", 0) or 0), "E11 은 E8 act_shift 와 함께 쓰지 않는다"
        from .n3d import DEFAULTS, N3DExpert
        old = dict(getattr(self.config, "n3d_cfg", None) or {})
        new = {**DEFAULTS, **old, **{k: v for k, v in cfg.items() if v is not None}, **({"path": os.path.abspath(path)} if path else {})}
        assert set(new) <= set(DEFAULTS) | {"path", "embedded", "train", "shared_mel"}, f"모르는 n3d 설정: {sorted(set(new) - set(DEFAULTS) - {'path'})}"
        assert not (old.get("embedded") and new.get("path") != old.get("path")), "흡수한 N3D(n3d_net)는 경로를 바꾸지 않는다 — 가지가 체크포인트 안에 있다"
        self.config.n3d_cfg = new
        if self.n3d is None or self.n3d.cfg.get("path") != new.get("path"): self.n3d = N3DExpert(new)
        else: self.n3d.cfg = {**self.n3d.cfg, **new}

    def absorb_n3d(self, train: Optional[dict] = None, state: Optional[dict] = None, share_mel: Optional[bool] = None) -> None:
        """N3D 흡수(2026-10-31, 사용자 "N3D 흡수하는 방향도 좋아" — wiki output-diarizer-replacement-20261031 길 3): add_n3d 의 고정 전문가를 모델 안 가지로 옮긴다.
        - 가중치는 하위 모듈 n3d_net(state_dict 의 n3d_net.*)이라 체크포인트에 함께 저장되고, NeMo 설정은 save_pretrained 가 n3d_config.yaml 로 남긴다 — .nemo 가 필요 없다.
        - mel(share_mel): None 이면 우리 인코더 전처리기가 N3D 것과 같은 값을 낼 때만 그것을 쓴다(_shared_mel). 설정은 같지만 N3D 체크포인트가 filterbank·window 를
          bf16 으로 담아 값이 조금 달라(2026-10-31 확인 — 이때는 N3D 자신의 mel 모듈을 가지 안에 둔다) 출력이 고정 전문가와 같다. True 면 우리 것을 쓴다(실험용 — 출력이 조금 바뀐다).
        - train·state(n3d_graft.n3d_trainable 설정·학습값 — G2·흡수 미세조정 graft.pt 의 n3d_train·n3d_state)를 주면 그 꼴로 감싸고 값을 넣는다.
        가중치는 고정(requires_grad False — 옵티마이저·DDP 밖)이고 모듈은 늘 eval 이다(N3DExpert.load)."""
        assert self.n3d is not None, "absorb_n3d 는 add_n3d 다음에"
        if self.n3d_net is None:
            from .n3d import load_n3d
            net, _ = load_n3d(self.n3d.path(), "cpu")
            if train:
                from .n3d_graft import load_n3d_trainable, n3d_trainable
                if state is not None: load_n3d_trainable(net, train, state)
                else: n3d_trainable(net, train)
            for p in net.parameters(): p.requires_grad_(False)
            self.n3d_net = net.eval().to(next(self.adapter.parameters()).device)
        else: assert not train or self.config.n3d_cfg.get("train") == dict(train), "이미 흡수한 N3D 와 다른 train 설정"
        pre = self._shared_mel(force=bool(share_mel)) if share_mel is not False else None; self.n3d.m = self.n3d_net; self.n3d.pre = pre
        self.config.n3d_cfg = {**self.config.n3d_cfg, "embedded": True, "shared_mel": pre is not None, **({"train": dict(train)} if train else {})}
        self.n3d.cfg = {**self.n3d.cfg, **self.config.n3d_cfg}

    def _shared_mel(self, force: bool = False):
        """흡수: N3D mel 을 만들 우리 인코더 전처리기 — force 면 있으면 그것, 아니면 같은 신호에서 두 전처리기(eval)의 값이 같을 때만. 아니면 None(N3D 자신의 것)."""
        pre = getattr(self.encoder, "pre", None) if self.encoder is not None else None
        if pre is None or self.n3d_net is None: return None
        if force: return pre
        own = self.n3d_net.preprocessor; dev = next(self.n3d_net.parameters()).device
        x = (torch.randn(2, 16000, generator=torch.Generator().manual_seed(0)) * 0.1).to(dev); n = torch.tensor([16000, 11000], device=dev)
        outs = []
        for p in (pre, own):
            was = p.training; p.eval()
            with torch.no_grad(), torch.autocast(dev.type, enabled=False): outs.append(p(input_signal=x, length=n))
            p.train(was)
        (fa, la), (fb, lb) = outs; T = int(la.max())                                              # 유효 길이까지만(N3D 는 pad_to 16 으로 뒤를 채운다 — stream_probs 가 길이로 자른다)
        ok = torch.equal(la, lb) and fa.shape[:2] == fb.shape[:2] and min(fa.shape[2], fb.shape[2]) >= T
        return pre if ok and torch.allclose(fa[..., :T], fb[..., :T], atol=1e-5) else None

    def _load_embedded_n3d(self, path: str) -> None:
        """from_pretrained(흡수 체크포인트): n3d_config.yaml 로 N3D 틀을 세우고(train 이면 같은 꼴로 감싼다) safetensors 의 n3d_net.* 를 엄격히 넣는다."""
        import glob
        from omegaconf import OmegaConf
        from safetensors.torch import load_file
        from .n3d import build_n3d, use_sdpa
        c = self.config.n3d_cfg; net, _ = build_n3d(OmegaConf.load(os.path.join(path, "n3d_config.yaml")))
        if c.get("train"):
            from .n3d_graft import n3d_trainable
            n3d_trainable(net, c["train"])
        sd = {k[len("n3d_net."):]: v for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))) for k, v in load_file(f).items() if k.startswith("n3d_net.")}
        net.load_state_dict(sd, strict=True); use_sdpa(net)
        for p in net.parameters(): p.requires_grad_(False)
        self.n3d_net = net.eval().to(next(self.adapter.parameters()).device)
        self.n3d.m = self.n3d_net; self.n3d.pre = self._shared_mel(force=True) if c.get("shared_mel") else None   # 저장할 때 정한 mel 그대로

    def n3d_act(self, wav, wav_len, K):
        """고정 전문가 활동 (B, max K, N) 0/1 — 학습 중(self.training)이면 학습 청크(train_chunk·train_rc), 아니면 추론 청크(chunk·rc).
        keep_n3d_probs(2026-11-03 E13 게이트 진단 — 평가 전용): 같은 활동을 내면서 확률 (B, max K, 8)을 last_n3d_probs 에 남긴다(게이트만 낮은 임계로 열 때).
        config.n3d_soft(E14, 2026-11-03): 0/1 대신 확률 그대로(학습·추론 모두 — 게이트는 디코더가 > 0.5 로 연다)."""
        soft = bool(getattr(self.config, "n3d_soft", False)); keep = getattr(self, "keep_n3d_probs", False) and not self.training
        if soft or keep:
            p = self.n3d.probs(wav, wav_len, K, train=True) if self.training else self.n3d.probs(wav, wav_len, K)
            if keep: self.last_n3d_probs = p.detach()
            th = float(self.n3d.cfg["threshold"]) if hasattr(self.n3d, "cfg") else float(self.n3d.threshold)   # N3DExpert · GraftExpert(G3) 의 act 와 같은 임계
            n = int(self.config.slot_input_slots); return p[..., :n] if soft else (p[..., :n] > th).float()
        return self.n3d.act(wav, wav_len, K, int(self.config.slot_input_slots), train=self.training)

    @staticmethod
    def with_act_stream(f, a):
        """특징 스트림 f (B, S, K, D) 의 스트림 1 을 슬롯 활동 a (B, K', N)(앞 N 차원, 나머지 0)로 바꾼다 — 스트림이 하나면 더한다. K' ≠ K 면 0 채움·자름."""
        K = f.shape[2]; a = F.pad(a.to(f.dtype), (0, f.shape[-1] - a.shape[-1], 0, max(0, K - a.shape[1])))[:, :K][:, None]
        return torch.cat([f[:, :1], a, f[:, 2:]], 1) if f.shape[1] >= 2 else torch.cat([f, a], 1)

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
    def build(self, feats, ids, is_audio, chunk_of, packed: bool = False, act_in=None, tag_chunk=None, slot_x=None, mix_mask=None):
        """tag_chunk(E9 — (B, L) 또는 packing (1, N)): 태그 위치 임베딩에 그 슬롯 최근 활동(act_in, 없으면 E6 예측 활동 스트림)을 더한다(_tag_add).
        slot_x (B, Kmax, N, D)(E10 슬롯 소리 토큰)가 있으면 태그 위치에 그 슬롯·청크의 x 도 더한다(청크 ≥ Kmax — flush — 는 0). mix_mask: chunk_embed."""
        emb = self.get_input_embeddings(); E = emb(ids); D = E.shape[-1]
        ce = (self.chunk_embed(feats, act_in) if mix_mask is None else self.chunk_embed(feats, act_in, mix_mask=mix_mask)).to(E.dtype)   # 지우지 않으면 예전 호출 그대로
        if tag_chunk is not None and self.adapter_tagact is not None:
            a = act_in if act_in is not None else (feats[:, 1, :, : self.adapter_slot.in_features] if self.diar_head is not None and self.adapter_slot is not None and feats.shape[1] >= 2 else None)
            E = self._tag_add(E, ids, tag_chunk, a, packed, slot_x=slot_x)
        if packed: ce = ce.reshape(1, -1, D)                                          # (B,Kmax,D) → (1,B·Kmax,D); chunk_of 는 packing.pack_batch 의 전역 인덱스
        g = torch.gather(ce, 1, chunk_of.clamp(min=0)[..., None].expand(-1, -1, D))
        return torch.where(is_audio[..., None], g, E)

    def forward(self, ids=None, is_audio=None, chunk_of=None, labels=None, mask=None, wav=None, wav_len=None, K=None, feats=None, next_weight: Optional[float] = None, return_dict: bool = True,
                labels_alt=None, soft_w=None, activity=None, activity_mask=None, act_weight: Optional[float] = None, pos_weight=None, position_ids=None, sem_mask=None,
                sem_only_mask=None, dec_kind=None, target_act=None, act_in=None, tag_chunk=None, remap_k=None, remap_perm=None, act_miss=None, **_):
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
        loss_switch·top1_hold)만 계산한다 — 손실은 그대로(가중은 데이터셋의 pos_weight 가 맡는다).
        tag_chunk (E9 슬롯 블록, labels 와 같은 인덱싱): 태그 위치 임베딩에 그 슬롯 최근 활동(act_in)을 더한다(build). 멈춤 label 은 다음 입력 위치의 <NEXT_AUDIO> 라
        손실 규칙은 그대로다."""
        loss_diar_t = diar_st = None; self._tagact_used = False; self._slotx_used = False; demb = None
        n3 = self.n3d is not None and wav is not None
        if wav is not None and (self.diar_head is not None or n3):                                 # E6: 인코더 한 번 → ASR 특징 + 화자 분리 logits(E11: 활동은 고정 전문가)
            assert target_act is None, "E6 는 대상 화자 모드(target_act)와 함께 쓰지 않는다"
            if self.diar_head is not None:
                if self.adapter_slotx is not None: feats, dlog, demb = self.encode_diar(wav, wav_len, K, with_emb=True)   # E10: 헤드 임베딩도(슬롯 추출기 입력)
                else: feats, dlog = self.encode_diar(wav, wav_len, K); demb = None
                pred = self.predict_act(dlog)
            else:
                assert self.adapter_slotx is None, "E10 슬롯 추출기는 화자 분리 헤드(탭 임베딩)와 함께"
                feats, dlog = self.encode(wav, wav_len, K, n3d=False), None; pred = None
            hpred = pred                                                                            # 헤드 통계(diar_st)는 헤드 예측으로
            if n3: pred = self.n3d_act(wav, wav_len, K).to(feats.device)                             # E11: 슬롯 활동 입력 = 고정 전문가 활동(헤드 손실은 그대로)
            from .diar import shift_frames
            R = int(getattr(self.config, "act_shift", 0) or 0)                                     # E8: 헤드 위치 p = 프레임 p − R 의 활동(정답·흔든 정답 입력을 R 만큼 민다)
            if activity is not None and dlog is not None:                                          # 손실은 packing 으로 펴기 전에(활동 (B, Kmax, S))
                from .diar import sort_pit_loss
                Ka, N = min(dlog.shape[1], activity.shape[1]), dlog.shape[-1]
                m = (activity_mask[:, :Ka] if activity_mask is not None else torch.ones(activity.shape[:2], device=dlog.device)[:, :Ka]).bool()
                valid = torch.arange(Ka, device=dlog.device)[None] < K.to(dlog.device)[:, None]; m = m & valid
                tgt = F.pad(activity[:, :Ka].float(), (0, max(0, N - activity.shape[-1])))[..., :N]
                if remap_k is not None:                                                             # slot_remap: 헤드 목표는 창 내내 같은 화자 정체(바꾸기 전 순서)로 되돌린다
                    from .diar import remap_cols
                    tgt = remap_cols(tgt, remap_k, remap_perm, inverse=True)
                if getattr(self.config, "diar_align", False):                                       # E7: 대화 전체 순서 슬롯 창 — 헤드 손실은 창 도착 순서로
                    from .diar import arrival_sort
                    tgt = arrival_sort(tgt, m)
                if R: tgt, m = shift_frames(tgt, R), shift_frames(m, R, False) & valid                 # E8: 끝 패딩 위치(p ≥ K)는 그대로 뺀다
                loss_diar_t, _, _ = sort_pit_loss(dlog[:, :Ka].float(), tgt, m, float(self.config.diar_alpha))
                with torch.no_grad(): mf = m.float()[..., None]; diar_st = float(((hpred[:, :Ka] == tgt).float() * mf).sum() / (mf.sum() * N).clamp(min=1))
            if act_in is not None and self.training:                                               # 표본마다 확률 ρ 로 헤드 예측, 아니면 흔든 정답(E5 와 같은 입력)
                rho = float(getattr(self, "act_pred_rho", self.config.act_pred_ramp[-1]))
                ai = act_in.to(pred.device).float(); ai = F.pad(ai, (0, max(0, pred.shape[-1] - ai.shape[-1]), 0, max(0, pred.shape[1] - ai.shape[1])))[:, : pred.shape[1], : pred.shape[-1]]
                ai = shift_frames(ai, R)                                                            # E8: 흔든 정답도 헤드 예측과 같은 시점(p − R)으로
                pa = pred
                if (getattr(self.config, "diar_align", False) or n3) and activity is not None:     # E7: 예측(창 도착 순서)을 정답(대화 전체 순서일 수 있다) 자리로 — AOSC 흉내
                    # E11: 전문가 채널 순서(전문가가 정한 도착 순서)도 정답 자리로 — 순열이 어긋난 입력이 라벨과 부딪히지 않게(추론은 채널 k = 슬롯 k, cpWER 는 순열 불변)
                    from .diar import align_slots
                    ref = F.pad(activity.float(), (0, max(0, pred.shape[-1] - activity.shape[-1]), 0, max(0, pred.shape[1] - activity.shape[1])))[:, : pred.shape[1], : pred.shape[-1]]
                    if remap_k is not None:                                                         # slot_remap: 예측(화자 정체 고정)은 바꾸기 전 순서로 맞추고, 바꾼 프레임부터 새 슬롯으로 옮긴다
                        from .diar import remap_cols
                        ref = remap_cols(ref, remap_k, remap_perm, inverse=True)
                    pa = align_slots(pred, shift_frames(ref, R).to(pred.device))
                    if remap_k is not None: pa = remap_cols(pa, remap_k, remap_perm)
                if act_miss is not None:                                                            # E14: 흉내 낸 놓침 프레임은 전문가 활동 입력에서도 지운다(추론의 N3D 놓침처럼)
                    mi = act_miss.to(pa.device).float()
                    mi = F.pad(mi, (0, max(0, pa.shape[-1] - mi.shape[-1]), 0, max(0, pa.shape[1] - mi.shape[1])))[:, : pa.shape[1], : pa.shape[-1]]; pa = pa * (1 - mi)
                act_in = torch.where((torch.rand(pred.shape[0], device=pred.device) < rho)[:, None, None], pa, ai)
            else: act_in = pred
        elif wav is not None: feats = self.encode(wav, wav_len, K, act=target_act)          # target_act (B, Kmax): E2 대상 화자 활동(없으면 예전 그대로)
        slot_x = mix_mask = None
        if self.adapter_slotx is not None and tag_chunk is not None and demb is not None and act_in is not None:   # E10: 슬롯 소리 토큰(같은 활동 입력으로)
            slot_x = self.adapter_slotx(feats[:, 0], demb, act_in)
            p_ = float(getattr(self.config, "slot_mix_drop", 0.0) or 0.0)
            if self.training and p_ > 0: mix_mask = torch.rand(feats.shape[0], device=feats.device) < p_
        nw = self.config.next_weight if next_weight is None else float(next_weight)
        packed = position_ids is not None; varlen = is_varlen(self.thinker)
        if packed:
            fa = packed_kwargs(position_ids=position_ids) if varlen else {}
            h = self.thinker.model(inputs_embeds=self.build(feats, ids, is_audio, chunk_of, packed=True, act_in=act_in, tag_chunk=tag_chunk, slot_x=slot_x, mix_mask=mix_mask), attention_mask=None, position_ids=position_ids,
                                   use_cache=False, **fa).last_hidden_state
            if activity is not None:
                activity = activity.reshape(1, -1, activity.shape[-1])
                activity_mask = activity_mask.reshape(1, -1) if activity_mask is not None else None
        elif varlen and mask is not None:
            h = self._thinker_unpadded(self.build(feats, ids, is_audio, chunk_of, act_in=act_in, tag_chunk=tag_chunk, slot_x=slot_x, mix_mask=mix_mask), mask.bool())
        else:
            h = self.thinker.model(inputs_embeds=self.build(feats, ids, is_audio, chunk_of, act_in=act_in, tag_chunk=tag_chunk, slot_x=slot_x, mix_mask=mix_mask), attention_mask=mask).last_hidden_state
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
        if loss_diar_t is not None: loss = loss + float(self.config.diar_weight) * loss_diar_t
        elif self.diar_head is not None and self.training: loss = loss + 0.0 * sum(p.sum() for p in self.diar_head.parameters())   # 활동 라벨 없는 행(replay): DDP 가 기다리지 않게
        if self.adapter_slot is not None and act_in is None and self.training: loss = loss + 0.0 * self.adapter_slot.weight.sum()   # E5 활동 입력 없는 행(replay): 같은 이유(E6 은 헤드 예측이 늘 들어간다)
        if self.adapter_tagact is not None and not self._tagact_used and self.training: loss = loss + 0.0 * self.adapter_tagact.weight.sum()   # E9 태그 없는 행(replay): 같은 이유
        if self.adapter_slotx is not None and self.training:                                    # E10: 쓰지 않은 행(replay)·지우지 않은 배치도 그래프에 묶는다(DDP)
            if not self._slotx_used: loss = loss + 0.0 * sum(p.sum() for p in self.adapter_slotx.parameters())
            loss = loss + 0.0 * self.adapter_mixnull.sum()
        dec = self._decision_stats(logits, t, dec_kind[:, 1:][sel]) if dec_kind is not None else {}
        out = VapAsrOutput(loss=loss, loss_next=st["loss_next"], loss_text=st["loss_text"], top1_text=st["top1_text"], n_labels=int(t.numel()),
                           loss_eot=st["loss_eot"], loss_act=(loss_act.item() if loss_act is not None else None), act_acc=act_st.get("act_acc"), n_soft=st["n_soft"],
                           loss_sem=st["loss_sem"] if st.get("n_sem") else None, top1_sem=st["top1_sem"] if st.get("n_sem") else None, n_sem=st.get("n_sem"), sem_fp=st.get("sem_fp"),
                           loss_turn=st["loss_turn"] if st.get("n_turn") else None, n_turn=st.get("n_turn"),
                           loss_diar=(loss_diar_t.item() if loss_diar_t is not None else None), diar_acc=diar_st, **dec)
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
        out = super().save_pretrained(save_directory, *args, state_dict=sd, **kw)
        if getattr(self, "n3d_net", None) is not None:                                            # 흡수 N3D: 틀(NeMo 설정) — 가중치는 safetensors 의 n3d_net.*
            from omegaconf import OmegaConf
            OmegaConf.save(self.n3d_net.cfg, os.path.join(save_directory, "n3d_config.yaml"))
        return out

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
        if (getattr(m.config, "n3d_cfg", None) or {}).get("embedded"): m._load_embedded_n3d(path)   # 흡수 N3D 가지(체크포인트 안)
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
        if self.adapter_tagact is not None: enc["adapter_tagact"] = self.adapter_tagact.state_dict()
        if self.adapter_slotx is not None: enc["adapter_slotx"] = self.adapter_slotx.state_dict(); enc["adapter_mixnull"] = self.adapter_mixnull.detach().cpu()
        if self.diar_head is not None: enc["diar_head"] = self.diar_head.state_dict()
        return dict(adapter=self.adapter.state_dict(), thinker={k: v.detach().to(torch.bfloat16).cpu() for k, v in self.thinker.state_dict().items()}, sp_ids=self.sp_ids, mono=True, full_ft=True, **enc)
