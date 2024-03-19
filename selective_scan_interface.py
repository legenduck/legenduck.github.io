# Copyright (c) 2023, Tri Dao, Albert Gu.

import torch
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn
    import causal_conv1d_cuda
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_cuda = None

import selective_scan_cuda


class SelectiveScanFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                return_last_state=False):
        # 입력 텐서들이 메모리에 연속적인지 확인하고, 필요한 경우 연속적으로 만듭니다.
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()
        # B와 C의 차원이 3인 경우에는 차원을 재배치합니다.
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
            ctx.squeeze_C = True
        # selective_scan_cuda.fwd 함수를 사용하여 forward pass를 수행합니다.
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
        # context에 softplus 여부와 z 존재 여부를 저장합니다.
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        # 마지막 상태를 추출합니다.
        last_state = x[:, :, -1, 1::2]  # (배치, 차원, d상태)
        if not ctx.has_z:
            # z가 없는 경우, backward pass를 위해 필요한 텐서들을 context에 저장하고 결과를 반환합니다.
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out if not return_last_state else (out, last_state)
        else:
            # z가 있는 경우, backward pass를 위해 필요한 텐서들을 context에 저장하고 결과를 반환합니다.
            ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)
            out_z = rest[0]
            return out_z if not return_last_state else (out_z, last_state)

    @staticmethod
    def backward(ctx, dout, *args):
        if not ctx.has_z:
            # z가 없는 경우, 저장된 텐서들을 불러옵니다.
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            z = None
            out = None
        else:
            # z가 있는 경우, 저장된 텐서들을 불러옵니다.
            u, delta, A, B, C, D, z, delta_bias, x, out = ctx.saved_tensors
        # dout 텐서가 메모리에 연속적인지 확인하고 필요한 경우 연속적으로 만듭니다.
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        # selective_scan_cuda.bwd 함수를 사용하여 backward pass를 수행합니다.
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, z, delta_bias, dout, x, out, None, ctx.delta_softplus,
            False  # out_z를 다시 계산할 필요가 없습니다.
        )
        # ctx에 저장된 z가 있을 경우 dz를 불러옵니다.
        dz = rest[0] if ctx.has_z else None
        # dB와 dC 텐서의 차원이 1인 경우에는 차원을 줄입니다.
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        # du, ddelta, dA, dB, dC, dD, ddelta_bias, dz를 반환합니다.
        return (du, ddelta, dA, dB, dC,
                dD if D is not None else None,
                dz,
                ddelta_bias if delta_bias is not None else None,
                None,
                None)


def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                     return_last_state=False):
    """if return_last_state is True, returns (out, last_state)
    last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
    not considered in the backward pass.
    """
    return SelectiveScanFn.apply(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                      return_last_state=False):
    """
    Selective Scan 연산을 수행하는 함수입니다.

    Args:
        u (torch.Tensor): 입력 Tensor u. 크기는 (batch, dim, L).
        delta (torch.Tensor): 입력 Tensor delta. 크기는 (batch, dim, L).
        A (torch.Tensor): 입력 Tensor A. 크기는 (dim, dstate).
        B (torch.Tensor): 입력 Tensor B. 크기는 (dim, N, L) 또는 (dim, 1, dstate, L)입니다.
                          B.dim() == 3이면 (dim, 1, dstate, L)로 변환됩니다.
        C (torch.Tensor): 입력 Tensor C. 크기는 (dim, N, L) 또는 (dim, 1, dstate, L)입니다.
                          C.dim() == 3이면 (dim, 1, dstate, L)로 변환됩니다.
        D (torch.Tensor, optional): 입력 Tensor D. 크기는 (dim,)입니다.
        z (torch.Tensor, optional): 입력 Tensor z. 크기는 (batch, dim, L)입니다.
        delta_bias (torch.Tensor, optional): delta에 더해질 편향 값입니다. 크기는 (dim,)입니다.
        delta_softplus (bool, optional): True이면 delta에 softplus 함수를 적용합니다.
        return_last_state (bool, optional): True이면 마지막 상태 값을 반환합니다.

    Returns:
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: Selective Scan 연산의 결과입니다.
            결과 Tensor의 크기는 (batch, dim, L)입니다.
            return_last_state가 True인 경우 마지막 상태 Tensor도 함께 반환됩니다.
            마지막 상태 Tensor의 크기는 (batch, dim, dstate)입니다.
    """
    # 입력 텐서의 dtype을 저장합니다.
    dtype_in = u.dtype
    # 입력 텐서를 float 타입으로 변환합니다.
    u = u.float()
    delta = delta.float()
    # delta_bias가 주어진 경우 delta에 더해줍니다.
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    # delta_softplus가 True인 경우 delta에 softplus 함수를 적용합니다.
    if delta_softplus:
        delta = F.softplus(delta)
    # 입력 텐서의 크기 정보를 저장합니다.
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    # B와 C가 변수인지 여부를 확인합니다.
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    # A가 복소수인 경우 B와 C도 복소수로 변환합니다.
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
        B = B.float()
        C = C.float()
    # 출력 Tensor 초기화
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    # deltaA 계산
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    # deltaB_u 계산
    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    # C가 변수이고 4차원인 경우 반복된 C를 생성합니다.
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    # 마지막 상태 초기화
    last_state = None
    # 각 타임 스텝에 대해 selective scan 연산 수행
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        # 마지막 타임 스텝인 경우 마지막 상태를 저장합니다.
        if i == u.shape[2] - 1:
            last_state = x
        # 복소수인 경우 실수부를 사용합니다.
        if y.is_complex():
            y = y.real * 2
        ys.append(y)
    # 결과 Tensor 생성
    y = torch.stack(ys, dim=2) # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    # z가 주어진 경우 silu 함수를 적용합니다.
    if z is not None:
        out = out * F.silu(z)
    # 결과 Tensor를 입력 텐서의 dtype으로 변환합니다.
    out = out.to(dtype=dtype_in)
    # return_last_state가 True인 경우 마지막 상태를 반환합니다.
    return out if not return_last_state else (out, last_state)


class MambaInnerFn(torch.autograd.Function):
    """
    MambaInnerFn 클래스는 Mamba 모델의 내부 함수를 정의합니다.
    """
    
    @staticmethod
    @custom_fwd
    def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                out_proj_weight, out_proj_bias,
                A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
                C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1):
        """
        forward 메서드는 Mamba 모델의 forward 연산을 수행합니다.

        Args:
            ctx (torch.autograd.function.Context): Autograd context 객체입니다.
            xz (torch.Tensor): 입력 Tensor xz. 크기는 (batch, dim, seqlen)입니다.
            conv1d_weight (torch.Tensor): Conv1d 가중치 Tensor. 크기는 (dim, kernel_size)입니다.
            conv1d_bias (torch.Tensor): Conv1d 편향 Tensor. 크기는 (dim,)입니다.
            x_proj_weight (torch.Tensor): 입력 프로젝션 가중치 Tensor. 크기는 (output_dim, dim)입니다.
            delta_proj_weight (torch.Tensor): Delta 프로젝션 가중치 Tensor. 크기는 (output_dim, dim)입니다.
            out_proj_weight (torch.Tensor): 출력 프로젝션 가중치 Tensor. 크기는 (dim, output_dim)입니다.
            out_proj_bias (torch.Tensor): 출력 프로젝션 편향 Tensor. 크기는 (dim,)입니다.
            A (torch.Tensor): 입력 Tensor A. 크기는 (dim, dstate)입니다.
            B (torch.Tensor, optional): 입력 Tensor B. 크기는 (dim, dstate)입니다.
            C (torch.Tensor, optional): 입력 Tensor C. 크기는 (dim, dstate)입니다.
            D (torch.Tensor, optional): 입력 Tensor D. 크기는 (dim,)입니다.
            delta_bias (torch.Tensor, optional): Delta에 더해질 편향 값입니다. 크기는 (dim,)입니다.
            B_proj_bias (torch.Tensor, optional): B에 더해질 프로젝션 편향 값입니다. 크기는 (dstate,)입니다.
            C_proj_bias (torch.Tensor, optional): C에 더해질 프로젝션 편향 값입니다. 크기는 (dstate,)입니다.
            delta_softplus (bool, optional): True이면 delta에 softplus 함수를 적용합니다.
            checkpoint_lvl (int, optional): Checkpoint 레벨입니다. 0 또는 1이어야 합니다.

        Returns:
            torch.Tensor: Mamba 모델의 forward 연산 결과입니다.
        """
        
        assert causal_conv1d_cuda is not None, "causal_conv1d_cuda is not available. Please install causal-conv1d."
        assert checkpoint_lvl in [0, 1]
        
        # 입력 텐서의 정보 저장
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        
        # Autocast 활성화 시 dtype 변환
        if torch.is_autocast_enabled():
            x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            delta_proj_weight = delta_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_weight = out_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_bias = (out_proj_bias.to(dtype=torch.get_autocast_gpu_dtype())
                             if out_proj_bias is not None else None)
        
        # 입력 Tensor xz가 contiguous하지 않은 경우 contiguous하게 만듦
        if xz.stride(-1) != 1:
            xz = xz.contiguous()
        
        # conv1d_weight의 shape 재배열
        conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
        
        # 입력 Tensor xz를 x와 z로 분리
        x, z = xz.chunk(2, dim=1)
        
        # conv1d_out 계산
        conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
        conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
            x, conv1d_weight, conv1d_bias, None, None, None, True
        )
        
        # x_dbl 계산
        x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
        
        # delta 계산
        delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L)
        
        # B와 C가 변수인지 여부 저장
        ctx.is_variable_B = B is None
        ctx.is_variable_C = C is None
        ctx.B_proj_bias_is_None = B_proj_bias is None
        ctx.C_proj_bias_is_None = C_proj_bias is None
        
        # B가 변수인 경우 초기화 및 조정
        if B is None:
            B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl dstate)
            if B_proj_bias is not None:
                B = B + B_proj_bias.to(dtype=B.dtype)
            if not A.is_complex():
                B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if B.stride(-1) != 1:
                B = B.contiguous()
        
        # C가 변수인 경우 초기화 및 조정
        if C is None:
            C = x_dbl[:, -d_state:]  # (bl dstate)
            if C_proj_bias is not None:
                C = C + C_proj_bias.to(dtype=C.dtype)
            if not A.is_complex():
                C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if C.stride(-1) != 1:
                C = C.contiguous()
        
        # D가 주어진 경우 contiguous하게 만듦
        if D is not None:
            D = D.contiguous()
        
        # selective_scan_cuda.fwd 메서드 호출
        out, scan_intermediates, out_z = selective_scan_cuda.fwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
        )
        
        # ctx에 필요한 정보 저장
        ctx.delta_softplus = delta_softplus
        ctx.out_proj_bias_is_None = out_proj_bias is None
        ctx.checkpoint_lvl = checkpoint_lvl
        
        # checkpoint_lvl에 따라 conv1d_out과 delta 재계산
        if checkpoint_lvl >= 1:
            conv1d_out, delta = None, None
        
        # 역전파를 위한 중간값 저장
        ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
                              delta_proj_weight, out_proj_weight, conv1d_out, delta,
                              A, B, C, D, delta_bias, scan_intermediates, out)
        
        # 출력 계산 및 반환
        return F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)

    @staticmethod
    @custom_bwd
    def backward(ctx, dout):
        """
        backward 메서드는 Mamba 모델의 backward 연산을 수행합니다.

        Args:
            ctx (torch.autograd.function.Context): Autograd context 객체입니다.
            dout (torch.Tensor): 출력에 대한 손실의 기울기 Tensor입니다. 크기는 (batch, seqlen, dim)입니다.

        Returns:
            Tuple[torch.Tensor]: 입력 Tensor들에 대한 손실의 기울기 Tensor들의 튜플입니다.
        """
        
        # 필요한 정보 로드
        (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, out_proj_weight,
         conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, out) = ctx.saved_tensors
        
        # 입력 Tensor 정보 저장
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        x, z = xz.chunk(2, dim=1)
        
        # dout이 contiguous하지 않은 경우 contiguous하게 만듦
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        
        # checkpoint_lvl에 따라 conv1d_out과 delta 재계산
        if ctx.checkpoint_lvl == 1:
            conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
                x, conv1d_weight, conv1d_bias, None, None, None, True
            )
            delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
                              "d (b l) -> b d l", l = L)
        
        # selective_scan_cuda.bwd 메서드 호출
        dxz = torch.empty_like(xz)  # (batch, dim, seqlen)
        dx, dz = dxz.chunk(2, dim=1)
        dout = rearrange(dout, "b l e -> e (b l)")
        dout_y = rearrange(out_proj_weight.t() @ dout, "d (b l) -> b d l", l=L)
        dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z = selective_scan_cuda.bwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, dout_y, scan_intermediates, out, dz,
            ctx.delta_softplus,
            True  # option to recompute out_z
        )
        
        # 출력에 대한 손실의 기울기를 계산
        dout_proj_weight = torch.einsum("eB,dB->ed", dout, rearrange(out_z, "b d l -> d (b l)"))
        dout_proj_bias = dout.sum(dim=(0, 1)) if not ctx.out_proj_bias_is_None else None
        dD = dD if D is not None else None
        
        # dx_dbl 초기화
        dx_dbl = torch.empty_like(x_dbl)
        
        # B가 변수인 경우 역전파
        dB_proj_bias = None
        if ctx.is_variable_B:
            if not A.is_complex():
                dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
            dx_dbl[:, delta_rank:delta_rank + d_state] = dB  # (bl d)
            dB = None
        
        # C가 변수인 경우 역전파
        dC_proj_bias = None
        if ctx.is_variable_C:
            if not A.is_complex():
                dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
            dx_dbl[:, -d_state:] = dC  # (bl d)
            dC = None
        
        # ddelta_proj_weight 계산
        ddelta = rearrange(ddelta, "b d l -> d (b l)")
        ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
        
        # dx_dbl 초기화
        dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
        
        # dconv1d_out 계산
        dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
        dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
        dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
        dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        
        # conv1d 역전파
        dx, dconv1d_weight, dconv1d_bias, *_ = causal_conv1d_cuda.causal_conv1d_bwd(
            x, conv1d_weight, conv1d_bias, dconv1d_out, None, None, None, dx, False, True
        )
        dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
        dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        
        # 반환
        return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
                dout_proj_weight, dout_proj_bias,
                dA, dB, dC, dD,
                ddelta_bias if delta_bias is not None else None,
                dB_proj_bias, dC_proj_bias, None)



def mamba_inner_fn(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
    out_proj_weight, out_proj_bias,
    A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, delta_softplus=True
):
    """
    MambaInnerFn을 호출하여 주어진 입력에 대한 forward pass를 수행합니다.
    
    Parameters:
        xz (torch.Tensor): 입력 데이터 xz (batch, dim, seqlen).
        conv1d_weight (torch.Tensor): 1D 컨볼루션 가중치.
        conv1d_bias (torch.Tensor): 1D 컨볼루션 편향.
        x_proj_weight (torch.Tensor): x_proj 가중치.
        delta_proj_weight (torch.Tensor): delta_proj 가중치.
        out_proj_weight (torch.Tensor): out_proj 가중치.
        out_proj_bias (torch.Tensor): out_proj 편향.
        A (torch.Tensor): A 행렬.
        B (torch.Tensor, optional): B 행렬. 기본값은 None.
        C (torch.Tensor, optional): C 행렬. 기본값은 None.
        D (torch.Tensor, optional): D 행렬. 기본값은 None.
        delta_bias (torch.Tensor, optional): delta 편향. 기본값은 None.
        B_proj_bias (torch.Tensor, optional): B_proj 편향. 기본값은 None.
        C_proj_bias (torch.Tensor, optional): C_proj 편향. 기본값은 None.
        delta_softplus (bool, optional): delta_softplus 사용 여부. 기본값은 True.
    
    Returns:
        torch.Tensor: forward pass의 결과.
    """
    return MambaInnerFn.apply(xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                              out_proj_weight, out_proj_bias,
                              A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, delta_softplus)


def mamba_inner_ref(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
    out_proj_weight, out_proj_bias,
    A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, delta_softplus=True
):
    """
    Reference 버전의 selective_scan_fn을 호출하여 주어진 입력에 대한 forward pass를 수행합니다.
    
    Parameters:
        xz (torch.Tensor): 입력 데이터 xz (batch, dim, seqlen).
        conv1d_weight (torch.Tensor): 1D 컨볼루션 가중치.
        conv1d_bias (torch.Tensor): 1D 컨볼루션 편향.
        x_proj_weight (torch.Tensor): x_proj 가중치.
        delta_proj_weight (torch.Tensor): delta_proj 가중치.
        out_proj_weight (torch.Tensor): out_proj 가중치.
        out_proj_bias (torch.Tensor): out_proj 편향.
        A (torch.Tensor): A 행렬.
        B (torch.Tensor, optional): B 행렬. 기본값은 None.
        C (torch.Tensor, optional): C 행렬. 기본값은 None.
        D (torch.Tensor, optional): D 행렬. 기본값은 None.
        delta_bias (torch.Tensor, optional): delta 편향. 기본값은 None.
        B_proj_bias (torch.Tensor, optional): B_proj 편향. 기본값은 None.
        C_proj_bias (torch.Tensor, optional): C_proj 편향. 기본값은 None.
        delta_softplus (bool, optional): delta_softplus 사용 여부. 기본값은 True.
    
    Returns:
        torch.Tensor: forward pass의 결과.
    """
    assert causal_conv1d_fn is not None, "causal_conv1d_fn is not available. Please install causal-conv1d."
    L = xz.shape[-1]
    delta_rank = delta_proj_weight.shape[1]
    d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
    x, z = xz.chunk(2, dim=1)
    x = causal_conv1d_fn(x, rearrange(conv1d_weight, "d 1 w -> d w"), conv1d_bias, activation="silu")
    # We're being very careful here about the layout, to avoid extra transposes.
    # We want delta to have d as the slowest moving dimension
    # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
    x_dbl = F.linear(rearrange(x, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
    delta = delta_proj_weight @ x_dbl[:, :delta_rank].t()
    delta = rearrange(delta, "d (b l) -> b d l", l=L)
    if B is None:  # variable B
        B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl d)
        if B_proj_bias is not None:
            B = B + B_proj_bias.to(dtype=B.dtype)
        if not A.is_complex():
            B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            B = rearrange(B, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2).contiguous()
    if C is None:  # variable B
        C = x_dbl[:, -d_state:]  # (bl d)
        if C_proj_bias is not None:
            C = C + C_proj_bias.to(dtype=C.dtype)
        if not A.is_complex():
            C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            C = rearrange(C, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2).contiguous()
    y = selective_scan_fn(x, delta, A, B, C, D, z=z, delta_bias=delta_bias, delta_softplus=True)
    return F.linear(rearrange(y, "b d l -> b l d"), out_proj_weight, out_proj_bias)
