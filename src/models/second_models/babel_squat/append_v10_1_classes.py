# -*- coding: utf-8 -*-
"""
vidio-health-trainer
Script: append_v10_1_classes.py
Purpose: joint_meta_16way_moe_net.py에 v10.1 Structured16Head 및 MoENetV10_1 클래스 안전 추가
"""

v10_1_code = '''

# ==============================================================================
# [v10.1 신규] Exact 16-State Structured Energy Head & MoE Network
# ==============================================================================

class Structured16Head(nn.Module):
    """
    vidio-health-trainer v10.1
    Exact 16-State Energy-Based Structured Head
    - Unary Energy (4D): c1, c2, c3, c4 개별 결함 log-odds
    - Clean Energy (1D): 완전 정상 (State 0) baseline 에너지
    - Pairwise Interaction (6D): 6개 결함 쌍 상호작용 (v_kl)
    - Triple Kinetic Chain (1D): c1(척추) x c3(무릎) x c4(발) 운동사슬 상호작용 (r)
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.clean_energy = nn.Linear(in_dim, 1)
        self.unary = nn.Linear(in_dim, 4)
        self.pair  = nn.Linear(in_dim, 6)
        self.triple = nn.Linear(in_dim, 1)
        self.register_buffer('states', torch.from_numpy(STATE_TO_DEFECTS_MAP))
        self.pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u = self.unary(h)    # (B, 4)
        v = self.pair(h)     # (B, 6)
        r = self.triple(h)   # (B, 1)
        c0 = self.clean_energy(h) # (B, 1)

        # 1. Unary energy: u @ S^T -> (B, 16)
        energy = torch.matmul(u, self.states.t()).clone()

        # State 0 (Clean) gets clean_energy
        energy[:, 0:1] = energy[:, 0:1] + c0

        # 2. Pairwise interactions
        for e, (i, j) in enumerate(self.pairs):
            pair_mask = self.states[:, i] * self.states[:, j]
            energy = energy + v[:, e:e+1] * pair_mask.unsqueeze(0)

        # 3. Triple kinetic chain: c1(0) * c3(2) * c4(3)
        triple_mask = self.states[:, 0] * self.states[:, 2] * self.states[:, 3]
        energy = energy + r * triple_mask.unsqueeze(0)

        return energy, u


class StructuredExpertSubNet(nn.Module):
    """v10.1 구조화 에너지 헤드 장착 전문가 서브 네트워크"""
    def __init__(self, in_features: int, hidden_dim: int = 48, dropout_rate: float = 0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU()
        )
        self.energy_head = Structured16Head(in_dim=hidden_dim // 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.energy_head(h)


class JointMeta16WayMoENetV10_1(nn.Module):
    """
    vidio-health-trainer v10.1
    View-Conditioned Mixture-of-Experts (MoE) + Exact 16-State Structured Energy Head
    - 3대 전문 서브 네트워크 (Front, Rear, Kinetic Chain) 모두 Structured16Head 탑재
    - 게이팅 네트워크가 에너지 로짓을 동적 가중합하여 최종 물리적 결합 에너지 생성
    """
    def __init__(self, in_features: int = 39, hidden_dim: int = 48, dropout_rate: float = 0.20):
        super().__init__()
        self.in_features = in_features

        self.expert_front = StructuredExpertSubNet(in_features, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.expert_rear = StructuredExpertSubNet(in_features, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.expert_kinetic = StructuredExpertSubNet(in_features, hidden_dim=hidden_dim, dropout_rate=dropout_rate)

        self.gate_net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(32, 3)
        )

        self.register_buffer('M_matrix', torch.from_numpy(STATE_TO_DEFECTS_MAP))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l16_front, l4_front = self.expert_front(x)
        l16_rear, l4_rear = self.expert_rear(x)
        l16_kin, l4_kin = self.expert_kinetic(x)

        gate_logits = self.gate_net(x)
        gate_weights = torch.softmax(gate_logits, dim=-1)

        g1 = gate_weights[:, 0:1]
        g2 = gate_weights[:, 1:2]
        g3 = gate_weights[:, 2:3]

        fused_l16 = g1 * l16_front + g2 * l16_rear + g3 * l16_kin
        fused_l4 = g1 * l4_front + g2 * l4_rear + g3 * l4_kin

        return fused_l16, fused_l4, gate_weights


class MoEConsistencyLossV10_1(nn.Module):
    """
    vidio-health-trainer v10.1 학습 손실
    - Softmax 기반 Reverse-ASL (Softmax Clean Probability P(Clean) 사용)
    - 16-State Weighted CE Loss
    - 4-Way Marginal BCE Loss
    - Consistency Loss
    - Gate Balance Regularizer
    """
    def __init__(
        self,
        weight_ce: float = 0.50,
        weight_clean: float = 0.20,
        weight_bce: float = 0.20,
        weight_consistency: float = 0.05,
        weight_gate_balance: float = 0.05,
        lambda_fc: float = 3.0,
        gamma_pos: float = 1.0,
        gamma_neg: float = 2.0,
        class_weights_16: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.w_ce = weight_ce
        self.w_clean = weight_clean
        self.w_bce = weight_bce
        self.w_cons = weight_consistency
        self.w_gate = weight_gate_balance
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights_16)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.lambda_fc = lambda_fc
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.register_buffer('M_matrix', torch.from_numpy(STATE_TO_DEFECTS_MAP))

    def forward(
        self,
        logits_16: torch.Tensor,
        logits_4: torch.Tensor,
        targets_16: torch.Tensor,
        targets_4: torch.Tensor,
        gate_weights: torch.Tensor,
        y_clean: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 1. 16-state CE Loss
        loss_ce = self.ce_loss(logits_16, targets_16)

        # 2. Softmax 기반 Reverse-ASL
        p16 = torch.softmax(logits_16, dim=-1)
        p_clean = p16[:, 0].clamp(1e-7, 1.0 - 1e-7)
        y = y_clean.float()
        loss_pos = -y * (1.0 - p_clean).pow(self.gamma_pos) * torch.log(p_clean)
        loss_neg = -self.lambda_fc * (1.0 - y) * p_clean.pow(self.gamma_neg) * torch.log(1.0 - p_clean)
        loss_fc = (loss_pos + loss_neg).mean()

        # 3. 4-Way Marginal BCE
        loss_bce = self.bce_loss(logits_4, targets_4)

        # 4. Consistency Loss
        p4_from_16 = torch.matmul(p16, self.M_matrix)
        p4_from_bce = torch.sigmoid(logits_4)
        loss_cons = F.mse_loss(p4_from_16, p4_from_bce)

        # 5. Gate Balance Regularizer
        mean_gates = torch.mean(gate_weights, dim=0)
        uniform = torch.full_like(mean_gates, 1.0 / 3.0)
        loss_gate = F.kl_div(mean_gates.log(), uniform, reduction='sum')

        total = (
            self.w_ce   * loss_ce
            + self.w_clean * loss_fc
            + self.w_bce  * loss_bce
            + self.w_cons * loss_cons
            + self.w_gate * loss_gate
        )

        return total, {
            'loss_total':   float(total.item()),
            'loss_ce':      float(loss_ce.item()),
            'loss_fc':      float(loss_fc.item()),
            'loss_bce':     float(loss_bce.item()),
            'loss_cons':    float(loss_cons.item()),
            'loss_gate':    float(loss_gate.item())
        }
'''

target_file = r'D:\src\models\second_models\babel_squat\joint_meta_16way_moe_net.py'
with open(target_file, 'r', encoding='utf-8') as f:
    orig = f.read()

if 'JointMeta16WayMoENetV10_1' not in orig:
    with open(target_file, 'a', encoding='utf-8') as f:
        f.write(v10_1_code)
    print('Appended v10.1 classes successfully!')
else:
    print('v10.1 classes already present!')
