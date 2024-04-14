from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import mpmath
import numpy as np
from tqdm import tqdm


@dataclass
class SamplerResult:
    px: np.ndarray
    py: np.ndarray


class GibbsSampler:
    def __init__(
        self,
        z: np.ndarray,
        nx: np.ndarray,
        ny: np.ndarray,
        px_param: Tuple[float, float],
        py_param: Tuple[float, float],
    ) -> None:
        """
        z: 陽性と判定された患者数
        nx: 真の患者数
        ny: 真の非患者数
        px_param: pxのBeta事前分布のパラメータ
        py_param: pyのBeta事前分布のパラメータ
        """
        assert len(z) == len(nx) == len(ny)

        self.z = z
        self.nx = nx
        self.ny = ny
        self.px_param = px_param
        self.py_param = py_param
        self.sample_size = len(z)

    def _sample_px(self, x: np.ndarray) -> float:
        """
        pxの完全条件付き分布からのサンプリング
        """
        a0, b0 = self.px_param
        a = a0 + x.sum()
        b = b0 + self.nx.sum() - x.sum()
        return np.random.beta(a, b)

    def _sample_py(self, x: np.ndarray, z: np.ndarray) -> float:
        """
        pyの完全条件付き分布からのサンプリング
        """
        y = z - x
        a0, b0 = self.py_param
        a = a0 + y.sum()
        b = b0 + self.ny.sum() - y.sum()
        return np.random.beta(a, b)

    def _sample_x(self, z: np.ndarray, px: float, py: float) -> np.ndarray:
        """
        xの完全条件付き分布からのサンプリング
        """
        x = np.zeros(self.sample_size)

        for i in range(self.sample_size):
            conditional_probs = self._conditional_probs_of_x(
                z=z[i], nx=self.nx[i], ny=self.ny[i], px=px, py=py
            )
            values = list(conditional_probs.keys())
            probs = list(conditional_probs.values())
            x[i] = np.random.choice(values, p=probs)

        return x

    def run(
        self, draws: int = 2000, burn_in: int = 1000, chains: int = 2, seed: int = 0
    ) -> SamplerResult:
        """
        MCMCを実行する
        """
        assert draws > burn_in
        np.random.seed(seed)

        fig, axes = plt.subplots(2, tight_layout=True)
        axes[0].set_title("px")
        axes[1].set_title("py")

        px_chains, py_chains = [], []

        for _ in range(chains):
            px_chain, py_chain = [], []
            px, py = np.random.beta(*self.px_param), np.random.beta(*self.py_param)

            for _ in tqdm(range(draws)):
                x = self._sample_x(z=self.z, px=px, py=py)
                px = self._sample_px(x=x)
                py = self._sample_py(x=x, z=self.z)

                px_chain.append(px)
                py_chain.append(py)

            px_chain, py_chain = px_chain[burn_in:], py_chain[burn_in:]
            self._plot_samples(px_chain=px_chain, py_chain=py_chain, axes=axes)

        px_chains += px_chain
        py_chains += py_chain

        fig.show()

        return SamplerResult(px=np.array(px_chains), py=np.array(py_chains))

    @staticmethod
    def _plot_samples(
        px_chain: List[float], py_chain: List[float], axes: List[plt.Axes]
    ) -> None:
        axes[0].hist(px_chain, density=True)
        axes[1].hist(py_chain, density=True)

    @staticmethod
    def _conditional_probs_of_x(
        z: int, nx: int, ny: int, px: float, py: float, dps: int = 10
    ) -> Dict[int, float]:
        """
        xの完全条件付き分布
        xの値から対応する確率値へのマッピングを返す

        Note: 桁落ちを考慮してmpmathによる任意精度の浮動小数点で計算
        """
        mpmath.mp.dps = dps  # 小数の桁数

        probs = {}
        total_prob = mpmath.mpf("0")

        x_min, x_max = max(0, z - ny), min(nx, z)
        for x in range(x_min, x_max + 1):
            y = z - x
            prob = (
                mpmath.binomial(nx, x)
                * mpmath.power(px, x)
                * mpmath.power(1 - px, nx - x)
                * mpmath.binomial(ny, y)
                * mpmath.power(py, y)
                * mpmath.power(1 - py, ny - y)
            )
            probs[x] = prob
            total_prob += prob

        conditional_probs = {x: float(prob / total_prob) for x, prob in probs.items()}

        return conditional_probs
