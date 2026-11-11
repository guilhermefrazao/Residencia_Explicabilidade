import os
import numpy as np
import matplotlib

# Usa janela interativa quando existe display (Tk); senão cai para Agg (só salva PNGs)
try:
    if not os.environ.get("DISPLAY"):
        raise RuntimeError("sem display")
    import tkinter  # noqa: F401
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure


plt.style.use("Solarize_Light2")

OUT = "images"
GRAY, LINE, BLUE, RED, CREAM = "#eee8d5", "#93a1a1", "#268bd2", "#dc322f", "#fdf6e3"
DIVERGING = LinearSegmentedColormap.from_list("solarized_div", [RED, CREAM, BLUE]) 
POSITIVE = LinearSegmentedColormap.from_list("solarized_pos", [GRAY, BLUE])  


def save_figure(fig: Figure, name: str, out: str):
    os.makedirs(out, exist_ok=True)

    path = os.path.join(out, name)

    fig.savefig(path, dpi=130)
    plt.close(fig)

    print(f"figura salva em {path}")


def add_colorbar(fig: Figure, axes: np.ndarray, cmap: Colormap, lo: float, hi: float, label: str):
    fig.colorbar(ScalarMappable(Normalize(lo, hi), cmap), ax=np.ravel(axes).tolist(), shrink=0.5, label=label)


def draw_network(ax: Axes, sizes: list, values: dict = None, cmap: Colormap = DIVERGING, vmin: float = None, vmax: float = 1.0, image: np.ndarray = None, highlight: int = None, title: str = None):
    values = values or {}

    n_layers = len(sizes)
    xs = 3.0 + 2.5 * np.arange(n_layers)
    ys = [np.linspace(5, -5, n) if n > 1 else np.zeros(1) for n in sizes]

    norm = Normalize(-vmax if vmin is None else vmin, vmax)
    x_in, width = 1.5, xs[-1] + 1.0

    # conexões (estrutura da rede): entrada -> camada 1 -> ... -> saída
    segments = [[(x_in, 0.0), (xs[0], y)] for y in ys[0]]

    for a in range(n_layers - 1):
        segments += [[(xs[a], y0), (xs[a + 1], y1)] for y0 in ys[a] for y1 in ys[a + 1]]

    alpha = float(np.clip(300 / len(segments), 0.03, 0.5))

    ax.add_collection(LineCollection(segments, colors=LINE, linewidths=0.4, alpha=alpha, zorder=1))

    for a, (x, y) in enumerate(zip(xs, ys)):
        n, v = len(y), values.get(a)

        colors = [GRAY] * n if v is None else cmap(norm(np.asarray(v)))

        ax.scatter(np.full(n, x), y, s=float(np.clip(2500 / n, 20, 500)), c=colors, edgecolors="k", linewidths=0.8, zorder=3)

        if v is not None and n <= 12:
            for yy, vv, col in zip(y, v, colors):
                luminance = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
                text_color = "white" if luminance < 0.5 else "black"

                ax.text(x, yy, f"{vv:.2f}", ha="center", va="center", fontsize=6.5, color=text_color, zorder=4)

        name = "saída\n(10 dígitos)" if a == n_layers - 1 else f"camada {a + 1}\n({n} neurônios)"

        ax.text(x, -5.6, name, ha="center", va="top", fontsize=8)

    for digit, yy in enumerate(ys[-1]):
        ax.text(xs[-1] + 0.45, yy, str(digit), va="center", fontsize=8)

    if highlight is not None:
        ax.scatter([xs[-1]], [ys[-1][highlight]], s=800, facecolors="none", edgecolors="#f5a623", linewidths=2.5, zorder=4)

    if image is not None:
        inset = ax.inset_axes([0.0, 0.3, (x_in - 0.1) / width, 0.4])
        inset.imshow(image, cmap="gray")
        inset.axis("off")
    else:
        ax.text(x_in - 0.6, 0, "entrada\n784 px", ha="center", va="center", fontsize=8)

    ax.set_xlim(0, width)
    ax.set_ylim(-7.2, 5.8)
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=10)


def plot_neuron_contributions(contribs: list, X: np.ndarray, y: np.ndarray, preds: np.ndarray, sizes: list, n_images: int = 3, out: str = OUT):
    signed = [c[:, : c.shape[1] // 2] + c[:, c.shape[1] // 2:] for c in contribs] 
    vmax = max(np.abs(s[:n_images]).max() for s in signed)

    fig, axes = plt.subplots(n_images, 1, figsize=(9, 4.6 * n_images), squeeze=False, layout="constrained")

    for i, ax in enumerate(axes[:, 0]):
        values = {a: s[i] for a, s in enumerate(signed)}
        title = f"imagem {i} | rótulo {y[i]} | predito {preds[i]} (círculo laranja)"

        draw_network(ax, sizes, values=values, cmap=DIVERGING, vmax=vmax, image=X[i], highlight=preds[i], title=title)

    add_colorbar(fig, axes, DIVERGING, -vmax, vmax, "contribuição do neurônio (vermelho < 0 < azul)")

    save_figure(fig, "1_contribuicao_por_neuronio.png", out)


def plot_positive_data(datas: list, X: np.ndarray, y: np.ndarray, preds: np.ndarray, sizes: list, n_images: int = 3, out: str = OUT):
    vmax = max(d[:n_images].max() for d in datas)

    fig, axes = plt.subplots(n_images, 1, figsize=(9, 4.6 * n_images), squeeze=False, layout="constrained")

    for i, ax in enumerate(axes[:, 0]):
        values = {a: d[i] for a, d in enumerate(datas)}
        title = f"T da imagem {i} | rótulo {y[i]} | predito {preds[i]} (negativas = 0, cinza)"

        draw_network(ax, sizes, values=values, cmap=POSITIVE, vmin=0, vmax=vmax, image=X[i], highlight=preds[i], title=title)

    add_colorbar(fig, axes, POSITIVE, 0, vmax, "T (contribuição positiva normalizada)")

    save_figure(fig, "2_T_contribuicoes_positivas.png", out)


def draw_bars(ax: Axes, v: np.ndarray, lim: float):
    ax.bar(np.arange(len(v)), v, color=np.where(v >= 0, BLUE, RED), width=0.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylim(-lim, lim)
    ax.tick_params(labelsize=6)


def plot_modes(T: np.ndarray, loadings: np.ndarray, dictionary: np.ndarray, X: np.ndarray, y: np.ndarray, layer: int, n_images: int = 5, n_modes: int = 5, out: str = OUT):
    modes = [int(k) for k in np.argsort(-loadings[:n_images].sum(0))[:n_modes]]

    rec = loadings @ dictionary
    n_cols = 3 + len(modes)

    fig, axes = plt.subplots(n_images, n_cols, figsize=(2.1 * n_cols, 1.7 * n_images + 1.2), squeeze=False, layout="constrained")

    for i in range(n_images):
        terms = [loadings[i, k] * dictionary[k] for k in modes]
        lim = 1.1 * max(np.abs(T[i]).max(), np.abs(rec[i]).max(), *(np.abs(t).max() for t in terms), 1e-9)

        axes[i, 0].imshow(X[i], cmap="gray")
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        axes[i, 0].set_ylabel(f"rótulo {y[i]}")

        draw_bars(axes[i, 1], T[i], lim)

        for j, k in enumerate(modes):
            ax = axes[i, 2 + j]

            draw_bars(ax, terms[j], lim)
            ax.text(0.5, 0.97, f"presença {loadings[i, k]:.2f}", transform=ax.transAxes, ha="center", va="top", fontsize=7)

        draw_bars(axes[i, -1], rec[i], lim)
        axes[i, -1].plot(T[i], "k.", ms=4)

    titles = ["imagem", "T[i] (real)"] + [f"modo {k}\n(loadings × átomo)" for k in modes] + ["reconstrução\n(• = T real)"]

    for ax, title in zip(axes[0], titles):
        ax.set_title(title, fontsize=8)

    for ax in axes[-1, 1:]:
        ax.set_xlabel("neurônio", fontsize=7)

    fig.suptitle(f"Camada {layer + 1}: T[i] ≈ Σ loadings[i,k] · modo_k  (barras = neurônios; azul > 0, vermelho < 0)", fontsize=10)

    save_figure(fig, f"3_modos_camada{layer + 1}.png", out)


def plot_mode_weights(dictionary: np.ndarray, sizes: list, layer: int, corr: np.ndarray = None, n_modes: int = 5, out: str = OUT):
    n_atoms = dictionary.shape[0]

    if corr is not None and corr.shape[1] == n_atoms:
        modes = np.argsort(-corr.max(0))[:n_modes] 
    else:
        modes = np.arange(min(n_modes, n_atoms))

    vmax = np.abs(dictionary[modes]).max()
    n_rows = int(np.ceil(len(modes) / 2))

    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 4.4 * n_rows), squeeze=False, layout="constrained")

    for ax, k in zip(axes.ravel(), modes):
        extra = f" | melhor dígito {corr[:, k].argmax()} (corr {corr[:, k].max():.2f})" if corr is not None else ""

        draw_network(ax, sizes, values={layer: dictionary[k]}, cmap=DIVERGING, vmax=vmax, title=f"modo {k}{extra}")

    for ax in axes.ravel()[len(modes):]:
        ax.axis("off")

    add_colorbar(fig, axes, DIVERGING, -vmax, vmax, "peso do neurônio no modo (vermelho < 0 < azul)")

    fig.suptitle(f"Pesos do dicionário na camada {layer + 1} (cinza = camadas não analisadas)", fontsize=11)

    save_figure(fig, f"4_pesos_dos_modos_camada{layer + 1}.png", out)


def visualize_all(sizes: list, X: np.ndarray, y: np.ndarray, preds: np.ndarray, contribs: list, datas: list, loadings: list, dictionaries: list, corrs: list, out: str = OUT, n_images: int = 3):
    plot_neuron_contributions(contribs, X, y, preds, sizes, n_images=n_images, out=out)
    plot_positive_data(datas, X, y, preds, sizes, n_images=n_images, out=out)

    for l, (T, loading, dictionary, corr) in enumerate(zip(datas, loadings, dictionaries, corrs)):
        if loading.shape[1] == 0:
            print(f"camada {l + 1}: sem modos vivos, figuras 3 e 4 puladas")
            continue

        plot_modes(T, loading, dictionary, X, y, l, out=out)
        plot_mode_weights(dictionary, sizes, l, corr=corr, out=out)


def plot_pixel_maps(X: np.ndarray, maps: np.ndarray, y: np.ndarray, layer: int, out: str = OUT):
    n_images, n_modes = maps.shape[:2]
    vmax = np.abs(maps).max()

    fig, axes = plt.subplots(n_modes + 1, n_images, figsize=(2.2 * n_images, 2.2 * (n_modes + 1)), squeeze=False, layout="constrained")

    for i in range(n_images):
        axes[0, i].imshow(X[i], cmap="gray")
        axes[0, i].set_title(f"rótulo {y[i]}", fontsize=9)

        for k in range(n_modes):
            axes[k + 1, i].imshow(maps[i, k], cmap=DIVERGING, vmin=-vmax, vmax=vmax)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    for k in range(n_modes):
        axes[k + 1, 0].set_ylabel(f"modo {k}")

    add_colorbar(fig, axes, DIVERGING, -vmax, vmax, "contribuição do pixel (vermelho < 0 < azul)")

    save_figure(fig, f"5_mapas_de_pixels_camada{layer + 1}.png", out)


def visualize_neural_contributions(log_contribution: np.ndarray, layer: int = 0, out: str = OUT):
    """log_contribution: scope.log_contributions[layer], shape [N, 2*H] = [parte negativa | parte positiva] (ei_split)."""
    H = log_contribution.shape[1] // 2
    neg, pos = log_contribution[:, :H], log_contribution[:, H:]

    total_pos = pos.sum(axis=0)           # [H] soma sobre as N amostras
    total_neg = np.abs(neg).sum(axis=0)   # [H]
    total_abs = total_pos + total_neg

    print(f"log_contribution: {log_contribution.shape} -> [N amostras, 2*H] com H={H} neurônios")
    print(f"  negativa {neg.shape} | positiva {pos.shape} | soma por neurônio {total_abs.shape}")
    print(f"  soma positiva : {np.round(total_pos, 2)}")
    print(f"  soma |negativa|: {np.round(total_neg, 2)}")

    fig = plt.figure(figsize=(10, 4))
    idx = np.arange(H)
    plt.bar(idx, total_pos, color=BLUE, label="contribuição positiva")
    plt.bar(idx, total_neg, bottom=total_pos, color=RED, label="|contribuição negativa|")
    plt.xlabel("Índice do neurônio")
    plt.ylabel("Contribuição absoluta total (soma sobre as amostras)")
    plt.title(f"Contribuição por neurônio na camada {layer + 1} ({log_contribution.shape[0]} amostras)")
    plt.legend()
    plt.tight_layout()

    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, f"contributions_layer{layer + 1}_1.png"), dpi=130)
    plt.show()
