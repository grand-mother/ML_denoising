"""
LEGACY loss functions — reproduction of archived checkpoints ONLY.

These are the multi-domain losses exactly as they were in
``raytune_lib_sept25/raytune_training_function.py`` when the archived checkpoints
under ``/sps/grand/macias/Sam_Result/`` were trained (production ``multi_v3`` run:
2025-10-28). They are kept so that a run configuration read straight from an
archived ``best_trial_config.json`` still resolves — those files carry
``"criterion": "multi_v3"`` / ``"multi_v2"``, names that no longer exist in the
current API.

**Do not use these for new training.**

Known defect (referee revision, reply2): the phase term applies L1/MSE directly
to ``torch.angle`` of the FFT. ``torch.angle`` returns the principal value in
(-pi, pi], so it is discontinuous at the wrap-around: a true phase error of
epsilon across the +/-pi boundary is scored as ~2*pi instead of ~epsilon. The
corrected versions in ``utils/paper_losses_metrics`` use the wrapped residual
``atan2(sin d_phi, cos d_phi)`` and are exposed as the current ``multi_l1`` /
``multi_mse`` criteria. Anything trained from now on should use those.

The mapping between the old and current names is:

===============  ================  =================================================
archived name    current name      difference
===============  ================  =================================================
``multi_v2``     ``multi_mse``     MSE-based; old = direct phase, new = wrapped
``multi_v3``     ``multi_l1``      L1-based;  old = direct phase, new = wrapped
===============  ================  =================================================

``multi_v4`` (an STFT-weighted variant) is not reproduced here: no archived
checkpoint used it.
"""

import torch
import torch.nn.functional as F


def multi_domain_loss_v2(clean, pred, mag_weight, phase_weight):
    """LEGACY (archived ``multi_v2``). MSE time + magnitude + DIRECT phase.

    Phase term is discontinuous at +/-pi — see the module docstring. Use
    ``utils.paper_losses_metrics.multi_domain_mse_loss`` for new work.
    """
    time_loss = F.mse_loss(pred, clean)

    clean_fft = torch.fft.rfft(clean, dim=-1)
    pred_fft = torch.fft.rfft(pred, dim=-1)

    mag_loss = F.mse_loss(torch.abs(pred_fft), torch.abs(clean_fft))
    phase_loss = F.mse_loss(torch.angle(pred_fft), torch.angle(clean_fft))

    return time_loss + mag_weight * mag_loss + phase_weight * phase_loss


def multi_domain_loss_v3(clean, pred, mag_weight, phase_weight):
    """LEGACY (archived ``multi_v3`` — the production checkpoint). L1 time +
    magnitude + DIRECT phase.

    This is the loss that trained
    ``/sps/grand/macias/Sam_Result/multi_v3_CNN_100epochs_36samples``. Phase term
    is discontinuous at +/-pi — see the module docstring. Use
    ``utils.paper_losses_metrics.multi_domain_l1_loss`` for new work.
    """
    time_loss = F.l1_loss(pred, clean)

    clean_fft = torch.fft.rfft(clean, dim=-1)
    pred_fft = torch.fft.rfft(pred, dim=-1)

    mag_loss = F.l1_loss(torch.abs(pred_fft), torch.abs(clean_fft))
    phase_loss = F.l1_loss(torch.angle(pred_fft), torch.angle(clean_fft))

    return time_loss + mag_weight * mag_loss + phase_weight * phase_loss
