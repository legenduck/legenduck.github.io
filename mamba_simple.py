# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class Mamba(nn.Module):
    def __init__(
        self,
        d_model,  # 입력 차원
        d_state=16,  # 상태 차원
        d_conv=4,  # 컨볼루션 차원
        expand=2,  # 확장 계수
        dt_rank="auto",  # 시간 단계 순위
        dt_min=0.001,  # 최소 시간 간격
        dt_max=0.1,  # 최대 시간 간격
        dt_init="random",  # 시간 간격 초기화 방법
        dt_scale=1.0,  # 시간 간격 스케일
        dt_init_floor=1e-4,  # 초기화 바닥값
        conv_bias=True,  # 컨볼루션 편향 사용 여부
        bias=False,  # 편향 사용 여부
        use_fast_path=True,  # 빠른 경로 옵션 사용 여부
        layer_idx=None,  # 레이어 인덱스
        device=None,  # 디바이스
        dtype=None,  # 데이터 타입
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model  # 입력 차원
        self.d_state = d_state  # 상태 차원
        self.d_conv = d_conv  # 컨볼루션 차원
        self.expand = expand  # 확장 계수
        self.d_inner = int(self.expand * self.d_model)  # 내부 차원
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank  # 시간 단계 순위
        self.use_fast_path = use_fast_path  # 빠른 경로 옵션 사용 여부
        self.layer_idx = layer_idx  # 레이어 인덱스

        # 입력 프로젝션 레이어
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # 1D 컨볼루션 레이어
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"  # 활성화 함수
        self.act = nn.SiLU()  # SiLU 활성화 함수

        # X 프로젝션 레이어
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # 초기 특별한 dt 투영을 설정하여 초기화에서 분산을 보존합니다.
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # dt_bias를 초기화하여 F.softplus(dt_bias)가 dt_min과 dt_max 사이에 있도록 합니다.
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # softplus의 역함수
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # S4D 실제 초기화
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" 매개변수
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        # 출력 프로젝션 레이어
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        # 입력 데이터의 배치 크기, 시퀀스 길이, 차원을 가져옵니다.
        batch, seqlen, dim = hidden_states.shape

        # 추론 파라미터로부터 상태를 가져옵니다. 없으면 None으로 설정합니다.
        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            # 시퀀스 길이 오프셋이 있는 경우, 단계 메서드를 사용하여 출력을 계산합니다.
            if inference_params.seqlen_offset > 0:
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # 입력 데이터를 모델에 맞는 형태로 변환합니다.
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        # 실제 계산을 위해 연산 경로를 선택합니다.
        A = -torch.exp(self.A_log.float())
        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:
            # 빠른 경로를 사용하여 계산합니다.
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,
                None,
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            # 일반적인 경로를 따라 계산합니다.
            x, z = xz.chunk(2, dim=1)
            if conv_state is not None:
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            # 선택적으로 마스킹된 스캔을 사용하여 출력을 계산합니다.
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        # 입력 데이터의 dtype을 가져옵니다.
        dtype = hidden_states.dtype
        # 하나의 토큰만 있는 디코딩을 지원합니다.
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))

        x, z = xz.chunk(2, dim=-1)

        if causal_conv1d_update is None:
            # 컨볼루션 상태를 업데이트합니다.
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            # 원래 컨볼루션 방식으로 업데이트합니다.
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.linear(dt, self.dt_proj.weight)
        A = -torch.exp(self.A_log.float())

        if selective_state_update is None:
            # 선택적으로 상태를 업데이트하고 출력을 계산합니다.
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)
        else:
            # 선택적으로 상태를 업데이트하고 출력을 계산합니다.
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        # 추론 캐시를 할당합니다.
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        # 캐시로부터 상태를 가져옵니다.
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32  # FP32에서 잔차 연산 수행 여부
        self.fused_add_norm = fused_add_norm  # 결합된 Add-Norm 연산 여부
        self.mixer = mixer_cls(dim)  # Mixer 레이어 초기화
        self.norm = norm_cls(dim)  # 정규화 레이어 초기화
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"  # RMSNorm이 None이 아닌지 확인
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"  # 결합된 Add-Norm 연산에 LayerNorm 또는 RMSNorm만 지원되는지 확인

    # 순방향 전파 함수 정의
    def forward(
        self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None, inference_params=None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.fused_add_norm:
            # 결합된 Add-Norm 연산이 아닌 경우
            residual = (hidden_states + residual) if residual is not None else hidden_states  # 잔차 계산
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))  # 정규화 수행
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)  # 잔차를 FP32로 변환
        else:
            # 결합된 Add-Norm 연산인 경우
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn  # RMSNorm 또는 LayerNorm 함수 선택
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )  # 결합된 Add-Norm 연산 수행
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)  # Mixer 레이어에 hidden_states 전달
        return hidden_states, residual

    # 추론 캐시 할당 함수 정의
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
