"""Knowledge distillation loss. Pre-registration §5. Arms B and D only."""

import torch.nn.functional as F

T_DEFAULT = 4.0
LAMBDA = 0.9


def kd_loss(student_logits, teacher_logits, targets, T=T_DEFAULT, lam=LAMBDA):
    """(1 - lam) * CE(z_s, y) + lam * T^2 * KL(teacher || student), both softened by T.

    Two things here are easy to get wrong and both change the loss scale:

    * T**2 sits INSIDE the soft term, before lam, so lam keeps its meaning at any T. The
      effective soft:hard gradient ratio is lam/(1-lam) = 9:1 at lam=0.9.
    * reduction='batchmean', not 'mean'. 'mean' also divides by the class count, silently
      scaling the soft term down by 10x on CIFAR-10.

    F.kl_div(input, target) computes KL(target || input) with input as log-probabilities,
    so passing the student as input gives KL(teacher || student), which is what §5 asks for.
    """
    hard = F.cross_entropy(student_logits, targets)
    soft = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction="batchmean",
    )
    return (1 - lam) * hard + lam * (T ** 2) * soft
