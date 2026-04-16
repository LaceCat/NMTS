from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from env.physics.thickener import DEFAULT_INITIAL_CONCENTRATION_PROFILE, ThickenerModel


def configure_matplotlib():
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def dry_mass_by_profile(profile: np.ndarray, model: ThickenerModel):
    rho_mix = model.c2d(profile)
    layer_volume = model.area * model.detaz
    layer_mass = profile * rho_mix * layer_volume
    return layer_mass, rho_mix, layer_volume


def draw_profile(ax, profile: np.ndarray, title: str, subtitle: str):
    n_layers = len(profile)
    cmap = plt.get_cmap("YlOrBr")
    norm = plt.Normalize(0.0, 0.75)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_layers)
    ax.set_xticks([])
    ax.set_yticks(np.arange(0.5, n_layers + 0.5, 1.0))
    ax.set_yticklabels([f"L{i}" for i in range(1, n_layers + 1)])
    ax.set_title(f"{title}\n{subtitle}", fontsize=13, pad=12)

    for i, conc in enumerate(profile):
        y = n_layers - 1 - i
        rect = Rectangle((0.15, y), 0.7, 1.0, facecolor=cmap(norm(conc)), edgecolor="black", linewidth=1.0)
        ax.add_patch(rect)
        ax.text(0.5, y + 0.5, f"{conc:.3f}", ha="center", va="center", fontsize=10, color="black")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)


def build_formula_text(
    uniform_total: float,
    custom_total: float,
    difference: float,
    layer_volume: float,
    total_volume: float,
    custom_mean: float,
):
    return "\n".join(
        [
            "初始干矿量计算公式",
            "",
            r"$V_{layer}=\pi R^2H/N$",
            rf"$= \pi \times 15^2 \times 5.79 / 10 = {layer_volume:.2f}\ \mathrm{{m^3}}$",
            "",
            r"$\rho(c)=\rho_s / [\rho_s-(\rho_s-\rho_w)c]$",
            r"$\rho_s=4.27,\ \rho_w=1.00\ \mathrm{t/m^3}$",
            "",
            r"$M_0=\sum_{i=1}^{10} c_i \cdot \rho(c_i) \cdot V_{layer}$",
            "",
            rf"全 0.66 初始化: $M_0={uniform_total:.2f}\ \mathrm{{t}}$",
            rf"设定剖面初始化: $M_0={custom_total:.2f}\ \mathrm{{t}}$",
            rf"差值: $\Delta M={difference:.2f}\ \mathrm{{t}}$",
            "",
            "为什么会差这么多",
            rf"1. 全 0.66 等于把 {total_volume:.2f} m^3 整个浓密机都放在高浓区。",
            rf"2. 我们的设定剖面平均浓度只有 {custom_mean:.4f}，上部 6 层明显更稀。",
            "3. 干矿量不是只和 c 成正比，而是和 c·rho(c) 成正比。",
            "4. 高浓度层不仅 c 更大，混合密度 rho(c) 也更大，所以总质量差距被进一步放大。",
        ]
    )


def main():
    configure_matplotlib()

    model = ThickenerModel()
    uniform_profile = np.full(model.N_LAYERS, 0.66, dtype=np.float64)
    custom_profile = DEFAULT_INITIAL_CONCENTRATION_PROFILE.astype(np.float64)

    uniform_layer_mass, _, layer_volume = dry_mass_by_profile(uniform_profile, model)
    custom_layer_mass, _, _ = dry_mass_by_profile(custom_profile, model)

    uniform_total = float(uniform_layer_mass.sum())
    custom_total = float(custom_layer_mass.sum())
    difference = uniform_total - custom_total
    total_volume = layer_volume * model.N_LAYERS
    custom_mean = float(custom_profile.mean())

    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.8])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    draw_profile(ax1, uniform_profile, "对照初始化", "10 层全部设为 0.66")
    draw_profile(ax2, custom_profile, "当前初始化", "按 AI 版 reset 平均剖面设定")

    ax3.axis("off")
    ax3.text(
        0.0,
        1.0,
        build_formula_text(
            uniform_total=uniform_total,
            custom_total=custom_total,
            difference=difference,
            layer_volume=layer_volume,
            total_volume=total_volume,
            custom_mean=custom_mean,
        ),
        ha="left",
        va="top",
        fontsize=12,
        linespacing=1.5,
    )

    fig.suptitle("浓密机两种初始化浓度剖面与初始干矿量差异", fontsize=18, y=1.02)

    output_dir = Path(__file__).resolve().parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / "initial_mass_profile_comparison.png"
    svg_path = output_dir / "initial_mass_profile_comparison.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")


if __name__ == "__main__":
    main()
