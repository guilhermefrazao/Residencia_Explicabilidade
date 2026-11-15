import os
import numpy as np
import matplotlib

matplotlib.use("Agg")  # só salva PNGs; o Tk só é carregado em show_image, quando uma figura precisa ser mostrada

import matplotlib.pyplot as plt

from datetime import datetime
from zoneinfo import ZoneInfo
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.figure import Figure


plt.style.use("Solarize_Light2")

OUT = "images"
GRAY, LINE, BLUE, RED, CREAM = "#eee8d5", "#93a1a1", "#268bd2", "#dc322f", "#fdf6e3"
DIVERGING = LinearSegmentedColormap.from_list("solarized_div", [RED, CREAM, BLUE]) 
POSITIVE = LinearSegmentedColormap.from_list("solarized_pos", [GRAY, BLUE])
NEGATIVE = LinearSegmentedColormap.from_list("solarized_neg", [GRAY, RED])  


def show_image(path: str):
    try:
        plt.switch_backend("TkAgg")
    except Exception:
        return  # sem display ou sem Tk: fica só o PNG salvo

    image = plt.imread(path)

    plt.figure(figsize=(12, 12 * image.shape[0] / image.shape[1]), layout="constrained")
    plt.imshow(image)
    plt.axis("off")
    plt.show()

    plt.close("all")
    plt.switch_backend("Agg")


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


def draw_bars(ax: Axes, v: np.ndarray, lim: float):
    ax.bar(np.arange(len(v)), v, color=np.where(v >= 0, BLUE, RED), width=0.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylim(-lim, lim)
    ax.tick_params(labelsize=6)


MAX_MODES, MAX_LABELS = 40, 10


def select_modes(positive: np.ndarray, negative: np.ndarray):
    n_modes = positive.shape[1]

    if n_modes <= MAX_MODES:
        return np.arange(n_modes)

    # mais de MAX_MODES modos: mantém os de maior contribuição total (positiva + |negativa|), na ordem original; negative vem com sinal (<= 0)
    total = (positive - negative).sum(axis=(0, 2, 3))

    return np.sort(np.argsort(-total)[:MAX_MODES])


def show_modes(maps: np.ndarray, modes: np.ndarray):
    maps = np.concatenate([maps[:MAX_LABELS], maps[:MAX_LABELS].sum(1, keepdims=True)], axis=1)  # no máximo MAX_LABELS linhas; última coluna = soma de TODOS os modos (antes do corte)

    return maps[:, [*modes, maps.shape[1] - 1]]


def class_scales(*maps: np.ndarray):
    # escala por dígito: as colunas dos modos dividem o máximo da linha (entre todos os mapas dados); a coluna "soma" tem escala própria
    peak = np.max([m.max(axis=(2, 3)) for m in maps], axis=0)
    peak[:, :-1] = peak[:, :-1].max(1, keepdims=True)

    return np.maximum(peak, 1e-9)[..., None, None]


class Visualization:
    def __init__(self, dataset: str, network: str, out: str = OUT):
        date = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%d-%m-%Y - %H:%M")

        self.out = os.path.join(out, f"{dataset}_{network}_{date}")

    def save_figure(self, fig: Figure, name: str, show: bool = False):
        os.makedirs(self.out, exist_ok=True)

        path = os.path.join(self.out, name)

        fig.savefig(path, dpi=130)
        plt.close(fig)

        print(f"figura salva em {path}")

        if show:
            show_image(path)


    def plot_neuron_contributions(self, contribs: list, X: np.ndarray, y: np.ndarray, preds: np.ndarray, sizes: list, n_images: int = 3):
        # contribs = scope.log_contributions (compute_contributions): por camada, [N, 2H] = [parte negativa | parte positiva] (ei_split)
        # negativa + positiva = contribuição com sinal de cada neurônio, por imagem
        signed = [c[:, : c.shape[1] // 2] + c[:, c.shape[1] // 2:] for c in contribs]
        vmax = max(np.abs(s[:n_images]).max() for s in signed)

        fig, axes = plt.subplots(n_images, 1, figsize=(9, 4.6 * n_images), squeeze=False, layout="constrained")

        for i, ax in enumerate(axes[:, 0]):
            # contribuição da imagem i em cada camada | X[i]: imagem de teste, y[i]: rótulo do dataset, preds[i]: argmax da saída da rede
            values = {a: s[i] for a, s in enumerate(signed)}
            title = f"imagem {i} | rótulo {y[i]} | predito {preds[i]} (círculo laranja)"

            draw_network(ax, sizes, values=values, cmap=DIVERGING, vmax=vmax, image=X[i], highlight=preds[i], title=title)

        add_colorbar(fig, axes, DIVERGING, -vmax, vmax, "contribuição do neurônio (vermelho < 0 < azul)")

        self.save_figure(fig, "1_contribuicao_por_neuronio.png")


    def plot_positive_data(self, datas: list, X: np.ndarray, y: np.ndarray, preds: np.ndarray, sizes: list, n_images: int = 3):
        # datas = retorno de compute_neural_correlations: metade positiva de log_contributions[l] dividida pelo desvio padrão global (o T que entra no SAE)
        vmax = max(d[:n_images].max() for d in datas)

        fig, axes = plt.subplots(n_images, 1, figsize=(9, 4.6 * n_images), squeeze=False, layout="constrained")

        for i, ax in enumerate(axes[:, 0]):
            # T da imagem i em cada camada (negativas já zeradas)
            values = {a: d[i] for a, d in enumerate(datas)}
            title = f"T da imagem {i} | rótulo {y[i]} | predito {preds[i]} (negativas = 0, cinza)"

            draw_network(ax, sizes, values=values, cmap=POSITIVE, vmin=0, vmax=vmax, image=X[i], highlight=preds[i], title=title)

        add_colorbar(fig, axes, POSITIVE, 0, vmax, "T (contribuição positiva normalizada)")

        self.save_figure(fig, "2_T_contribuicoes_positivas.png")


    def plot_mode_weights(self, dictionary: np.ndarray, corr: np.ndarray, layer: int, label: int):
        # corr = bscope.mtx_corr(digits, loadings) (correlation_results), [dígitos, modos vivos]; argmax(0) = dígito mais correlacionado de cada modo
        modes = np.where(corr.argmax(0) == label)[0]
        modes = modes[np.argsort(-corr[label, modes])]

        if len(modes) == 0:
            print(f"camada {layer + 1}: nenhum modo tem o dígito {label} como maior correlação")
            return

        # dictionary = sae.dictionary (load_sae), só modos vivos: linha k = pesos do átomo k sobre os neurônios da camada
        weights = dictionary[modes]
        n_neurons = weights.shape[1]
        vmax = np.abs(weights).max()

        fig, (ax_weights, ax_total) = plt.subplots(2, 1, figsize=(0.6 * n_neurons + 3, 0.45 * len(modes) + 4), height_ratios=[len(modes), 3], sharex=True, layout="constrained")

        image = ax_weights.imshow(weights, cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
        ax_weights.grid(False)
        ax_weights.set_yticks(np.arange(len(modes)), [f"modo {k} (corr {corr[label, k]:.2f})" for k in modes])
        fig.colorbar(image, ax=ax_weights, label="peso no modo")

        ax_total.bar(np.arange(n_neurons), np.abs(weights).sum(0), color=BLUE)
        ax_total.set_xticks(np.arange(n_neurons))
        ax_total.set_xlabel("neurônio")
        ax_total.set_ylabel("Σ |peso| dos modos")

        fig.suptitle(f"Camada {layer + 1}: {len(modes)} modos mais correlacionados com o dígito {label}", fontsize=11)

        self.save_figure(fig, f"3_pesos_dos_modos_digito{label}_camada{layer + 1}.png")


    def visualize_all(self, sizes: list, X: np.ndarray, y: np.ndarray, preds: np.ndarray, contribs: list, dictionaries: list, corrs: list, label: int, n_images: int = 3):
        self.plot_neuron_contributions(contribs, X, y, preds, sizes, n_images=n_images)

        for l, (dictionary, corr) in enumerate(zip(dictionaries, corrs)):
            if corr is None:
                print(f"camada {l + 1}: sem modos vivos, pesos dos modos pulados")
                continue

            self.plot_mode_weights(dictionary, corr, l, label)


    def plot_class_grid(self, panels: np.ndarray, layer: int, title: str, filename: str, modes: np.ndarray):
        n_digits, n_cols = panels.shape[:2]

        fig, axes = plt.subplots(n_digits, n_cols, figsize=(0.9 * n_cols, 0.9 * n_digits + 1), squeeze=False, layout="constrained")

        for digit, row in enumerate(axes):
            for ax, panel in zip(row, panels[digit]):
                ax.imshow(panel)
                ax.set_xticks([])
                ax.set_yticks([])

            row[0].set_ylabel(f"{digit}")

        for k, ax in enumerate(axes[0]):
            ax.set_title("soma" if k == n_cols - 1 else f"{modes[k]}", fontsize=7)

        fig.suptitle(f"Camada {layer + 1}: {title}", fontsize=10)

        self.save_figure(fig, filename)

    def plot_class_mode_maps(self, maps: np.ndarray, layer: int, name: str, cmap: Colormap, modes: np.ndarray):
        # maps = expanded_contributions.class_mode_maps: [dígitos, modos, H, W], média das imagens acertadas de cada dígito (gradiente × entrada ponderado pelos átomos)
        maps = show_modes(maps, modes)
        panels = cmap(maps / class_scales(maps))[..., :3]

        self.plot_class_grid(panels, layer, f"contribuições {name} (magnitude) médias por dígito (linhas) e modo (colunas) | escala por dígito (soma tem escala própria)", f"5_mapas_{name}_por_digito_camada{layer + 1}.png", modes)

    def plot_class_mode_maps_joint(self, positive: np.ndarray, negative: np.ndarray, layer: int, modes: np.ndarray):
        # positive e negative = expanded_contributions.class_mode_maps, [dígitos, modos, H, W]; negative passado em magnitude (-negative)
        positive, negative = show_modes(positive, modes), show_modes(negative, modes)
        scales = class_scales(positive, negative)

        # pintura subtrativa sobre o fundo: positivo tinge de azul, negativo de vermelho; pixels com os dois ficam escuros (sem cancelamento)
        base, blue, red = (np.array(to_rgb(c)) for c in (CREAM, BLUE, RED))
        panels = base - (positive / scales)[..., None] * (base - blue) - (negative / scales)[..., None] * (base - red)

        self.plot_class_grid(np.clip(panels, 0, 1), layer, "contribuições positivas (azul) e negativas (vermelho) médias por dígito (linhas) e modo (colunas) | mesma escala para as duas (soma tem escala própria)", f"5_mapas_conjuntos_por_digito_camada{layer + 1}.png", modes)

    def plot_label_images(self, images: np.ndarray, modes: np.ndarray, label: int, layer: int, n_cols: int = 10):
        # images = imagens de teste do dígito (dataset.data); modes = modo de maior loading de cada imagem (loadings_layer_{l}.npy, argmax)
        n_rows = int(np.ceil(len(images) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.3 * n_cols, 1.4 * n_rows + 0.8), squeeze=False, layout="constrained")

        for ax in axes.ravel():
            ax.axis("off")

        for ax, image, mode in zip(axes.ravel(), images, modes):
            ax.imshow(image, cmap="gray")
            ax.set_title(f"modo {mode}", fontsize=7)

        fig.suptitle(f"Dígito {label}: {len(images)} imagens do teste, ordenadas pelo modo dominante da camada {layer + 1}", fontsize=10)

        self.save_figure(fig, f"5_imagens_do_digito_{label}_camada{layer + 1}.png")

    def plot_ablation(self, results: dict, label: int, fractions: np.ndarray):
        # results = patching.neural_ablation.sweep: {método: métricas por fração de neurônios removidos em cada camada}; "random" = média dos sorteios de neurônios mantidos
        curves = {
            "acc_label": (f"acurácia só do dígito {label}", BLUE, "-"),
            "acc_others": ("acurácia dos outros dígitos", RED, "-"),
            "acc_all": ("acurácia de todas as imagens", LINE, "-"),
            "frac_pred_label": (f"fração prevista como {label}", "#b58900", "--"),
        }
        titles = {"sae": "mantém os neurônios de maior score do SAE", "ig": "baseline sem SAE: maior contribuição média do IG", "random": "controle: mantém neurônios ao acaso"}

        fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4.5), sharey=True, squeeze=False, layout="constrained")

        for ax, (name, result) in zip(axes[0], results.items()):
            for metric, (text, color, style) in curves.items():
                ax.plot(fractions * 100, result[metric] * 100, style, color=color, marker="o", ms=3, label=text)

            ax.set_title(titles.get(name, name), fontsize=10)
            ax.set_xlabel("% de neurônios removidos em cada camada")

        axes[0, 0].set_ylabel("%")
        axes[0, 0].legend(fontsize=8)

        fig.suptitle(f"Ablação por zero (activation patching) para o dígito {label}", fontsize=11)

        self.save_figure(fig, f"8_ablacao_digito{label}.png")
