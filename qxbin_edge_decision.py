import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

class QxBinEdgeDecision:
    """
    QxBin Edge Decision Maker - Tier 1: Embedded / IoT Real-time Uncertainty
    Builds on core QxBin: Binary Probability Matrix + fractional (bias^n, (1-bias)^m)
    Use case: Probabilistic decision under noisy inputs (e.g., drone obstacle avoidance, sensor fusion)
    Designed for resource-constrained devices. Room temperature classical hardware.
    """
    def __init__(self, grid_size: int = 5, num_actions: int = 4):
        self.grid_size = grid_size
        self.num_actions = num_actions  # e.g., Up, Down, Left, Right or custom actions
        self.state = np.random.rand(grid_size, grid_size).astype(np.float64)
        self._normalize()
        self.history = []  # Track collapses for feedback learning

    def _normalize(self):
        s = self.state.sum()
        if s > 0:
            self.state /= s

    def apply_superposition(self, bias: float = 0.75, n: int = 2, m: int = 1, sensor_feedback: float = 0.0):
        """Evolve with sensor-influenced bias adjustment for real-world feedback loops."""
        adjusted_bias = np.clip(bias + sensor_feedback * 0.2, 0.4, 0.95)
        frac = adjusted_bias ** n
        tail = (1.0 - adjusted_bias) ** m
        vec = np.linspace(frac, tail, self.grid_size)
        new_matrix = np.outer(vec, vec)
        self.state = (self.state + new_matrix) * 0.5
        self._normalize()
        return self.state

    def decide_action(self) -> Tuple[int, np.ndarray]:
        """Collapse matrix to select action probabilistically - like measuring a spinning coin."""
        flat = self.state.flatten()
        idx = np.random.choice(len(flat), p=flat)
        collapsed = np.zeros_like(flat)
        collapsed[idx] = 1.0
        collapsed = collapsed.reshape(self.state.shape)
        action = idx % self.num_actions
        self.history.append(action)
        return action, collapsed

    def visualize(self, title: str = "QxBin Edge Decision Matrix"):
        plt.figure(figsize=(6, 6))
        plt.imshow(self.state, cmap='viridis', interpolation='nearest')
        plt.title(title)
        plt.colorbar(label='Probability')
        plt.show()

if __name__ == "__main__":
    print("🚀 QxBin Edge Decision v2 - Real-time Uncertain Navigation / Decision Making")
    qx_dec = QxBinEdgeDecision(grid_size=6, num_actions=4)
    print("Initial state shape:", qx_dec.state.shape)
    
    # Simulate 5 decision steps with varying "sensor" feedback (noisy real-world input)
    for step in range(5):
        feedback = 0.1 if step % 2 == 0 else -0.05  # Example noisy sensor input
        qx_dec.apply_superposition(bias=0.8, n=3, m=1, sensor_feedback=feedback)
        action, collapsed = qx_dec.decide_action()
        print(f"Step {step+1}: Selected Action={action} | Sensor Feedback={feedback:.2f}")
    
    qx_dec.visualize("Edge QxBin Decision - Probability Matrix After Evolution")
    print("Decision history (action sequence):", qx_dec.history)
    print("Ready for embedding in drones, IoT, edge AI agents.")