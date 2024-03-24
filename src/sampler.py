import math
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
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
            x_min, x_max = max(0, z[i] - self.ny[i]), min(self.nx[i], z[i])
            values = list(range(x_min, x_max + 1))
            weights = [
                self._conditional_dist_of_x(
                    x=x, z=z[i], nx=self.nx[i], ny=self.ny[i], px=px, py=py
                )
                for x in values
            ]
            x[i] = np.random.choice(values, p=weights)

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
    def _conditional_dist_of_x(
        x: int, z: int, nx: int, ny: int, px: float, py: float
    ) -> float:
        """
        xの完全条件付き分布
        Note: 正規化定数は無視
        """
        y = z - x

        if not (0 <= x <= nx and 0 <= y <= ny):
            return 0

        return (
            math.comb(nx, x)
            * (px**x)
            * ((1 - px) ** (nx - x))
            * math.comb(ny, y)
            * (py**y)
            * ((1 - py) ** (ny - y))
        )
