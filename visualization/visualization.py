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
from scipy.ndimage import median_filter


plt.style.use("Solarize_Light2")


def class_scales(*maps: np.ndarray):
    # escala por dígito: as colunas dos modos dividem o máximo da linha (entre todos os mapas dados); a coluna "soma" tem escala própria
    peak = np.max([m.max(axis=(2, 3)) for m in maps], axis=0)
    peak[:, :-1] = peak[:, :-1].max(1, keepdims=True)

    return np.maximum(peak, 1e-9)[..., None, None]


class Visualization:
    def __init__(self, dataset: str, network: str, out: str = "images", class_names: list = None):
        self.class_names = class_names

        date = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%d-%m-%Y - %H:%M")

        self.out = os.path.join(out, f"{dataset}_{network}_{date}")

        self.GRAY, self.LINE, self.BLUE, self.RED, self.CREAM = "#eee8d5", "#93a1a1", "#268bd2", "#dc322f", "#fdf6e3"
        self.DIVERGING = LinearSegmentedColormap.from_list("solarized_div", [self.RED, self.CREAM, self.BLUE])
        self.POSITIVE = LinearSegmentedColormap.from_list("solarized_pos", [self.GRAY, self.BLUE])
        self.NEGATIVE = LinearSegmentedColormap.from_list("solarized_neg", [self.GRAY, self.RED])
        self.MAX_MODES, self.MAX_LABELS, self.MAX_CONNECTIONS, self.MAX_NEURONS = 10, 10, 100_000, 64
        self.MEDIAN_SIZE, self.CONTRAST = 4, 7  # filtro de mediana (1 = desligado) e fator de contraste (1 = desligado) dos mapas de contribuição


    def draw_network(self, ax: Axes, sizes: list, values: dict = None, cmap: Colormap = None, vmin: float = None, vmax: float = 1.0, image: np.ndarray = None, highlight: int = None, title: str = None, lines=True):
        values = values or {}
        cmap = self.DIVERGING if cmap is None else cmap  # o padrão é um atributo da instância: não pode ser o default do parâmetro

        n_layers = len(sizes)
        xs = 3.0 + 2.5 * np.arange(n_layers)
        ys = [np.linspace(5, -5, n) if n > 1 else np.zeros(1) for n in sizes]

        norm = Normalize(-vmax if vmin is None else vmin, vmax)
        x_in, width = 1.5, xs[-1] + 1.0

        # Se a rede for menor desenha linhas entre câmadas
        if lines:
            segments = [[(x_in, 0.0), (xs[0], y)] for y in ys[0]]

            for a in range(n_layers - 1):
                segments += [[(xs[a], y0), (xs[a + 1], y1)] for y0 in ys[a] for y1 in ys[a + 1]]

            alpha = float(np.clip(300 / len(segments), 0.03, 0.5))

            ax.add_collection(LineCollection(segments, colors=self.LINE, linewidths=0.4, alpha=alpha, zorder=1))

            # Adiciona o nome do label da última câmada dos neurônios 
            for digit, yy in enumerate(ys[-1]):
                ax.text(xs[-1] + 0.45, yy, str(digit), va="center", fontsize=8)

        # Cria a visualização dos neurônios por câmada 
        for a, (x, y) in enumerate(zip(xs, ys)):
            n, v = len(y), values.get(a)

            colors = [self.GRAY] * n if v is None else cmap(norm(np.asarray(v)))

            ax.scatter(np.full(n, x), y, s=float(np.clip(2500 / n, 20, 500)), c=colors, edgecolors="k", linewidths=0.8, zorder=3)

            if v is not None and n <= 12:
                for yy, vv, col in zip(y, v, colors):
                    luminance = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
                    text_color = "white" if luminance < 0.5 else "black"

                    ax.text(x, yy, f"{vv:.2f}", ha="center", va="center", fontsize=6.5, color=text_color, zorder=4)

            name = f"saída\n{sizes[-1]}" if a == n_layers - 1 else f"camada {a + 1}\n({n} neurônios)"

            ax.text(x, -5.6, name, ha="center", va="top", fontsize=8)

        # Cria o highlight laranja da última câmada
        if highlight is not None:
            ax.scatter([xs[-1]], [ys[-1][highlight]], s=800, facecolors="none", edgecolors="#f5a623", linewidths=2.5, zorder=4)

        # Adiciona a imagem classifica a entrada
        if image is not None:
            inset = ax.inset_axes([0.0, 0.3, (x_in - 0.1) / width, 0.4])
            inset.imshow(image, cmap="gray")
            inset.axis("off")
        else:
            ax.text(x_in - 0.6, 0, "entrada\npx", ha="center", va="center", fontsize=8)

        ax.set_xlim(0, width)
        ax.set_ylim(-7.2, 5.8)
        ax.axis("off")

        # Título
        if title:
            ax.set_title(title, fontsize=10)


    def draw_bars(self, ax: Axes, v: np.ndarray, lim: float):
        ax.bar(np.arange(len(v)), v, color=np.where(v >= 0, self.BLUE, self.RED), width=0.8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylim(-lim, lim)
        ax.tick_params(labelsize=6)


    def show_modes(self, maps: np.ndarray):
        maps = maps[:self.MAX_LABELS]  # no máximo MAX_LABELS linhas

        return np.concatenate([maps, maps.sum(1, keepdims=True)], axis=1)  # última coluna = soma dos modos mostrados


    def name(self, label: int):
        return str(label) if self.class_names is None else self.class_names[label]


    def save_figure(self, fig: Figure, name: str):
        os.makedirs(self.out, exist_ok=True)

        path = os.path.join(self.out, name)

        fig.savefig(path, dpi=130)
        plt.close(fig)

        print(f"figura salva em {path}")


    @staticmethod
    def add_colorbar(fig: Figure, axes: np.ndarray, cmap: Colormap, lo: float, hi: float, label: str):
        fig.colorbar(ScalarMappable(Normalize(lo, hi), cmap), ax=np.ravel(axes).tolist(), shrink=0.5, label=label)


    def plot_neuron_contributions(self, contribs: list, X: np.ndarray, y: np.ndarray, preds: np.ndarray, sizes: list, n_images: int = 3, lines=True):
        # contribs = scope.log_contributions: por camada e por imagem = [parte negativa | parte positiva] (ei_split)
        signed = [c[:, : c.shape[1] // 2] + c[:, c.shape[1] // 2:] for c in contribs]
        # Valor absoluto para definir a escolha na direita.
        vmax = max(np.abs(s[:n_images]).max() for s in signed)

        fig, axes = plt.subplots(n_images, 1, figsize=(9, 4.6 * n_images), squeeze=False, layout="constrained")

        # Cria a png por imagem e rede.
        for i, ax in enumerate(axes[:, 0]):
            # contribuição da imagem i em cada camada | X[i]: imagem de teste, y[i]: rótulo do dataset, preds[i]: argmax da saída da rede
            values = {a: s[i] for a, s in enumerate(signed)}
            title = f"imagem {i} | rótulo {y[i]} | predito {preds[i]} (círculo laranja)"

            self.draw_network(ax, sizes, values=values, cmap=self.DIVERGING, vmax=vmax, image=X[i], highlight=preds[i], title=title, lines=lines)

        #Desenha a escala dos neurônios como uma mapa separando por cor.
        self.add_colorbar(fig, axes, self.DIVERGING, -vmax, vmax, "contribuição do neurônio (vermelho < 0 < azul)")

        self.save_figure(fig, "1_contribuicao_por_neuronio.png")


    def plot_positive_data(self, datas: list, X: np.ndarray, y: np.ndarray, preds: np.ndarray, sizes: list, n_images: int = 3):
        # datas = retorno de compute_neural_correlations: metade positiva de log_contributions[l] dividida pelo desvio padrão global (o T que entra no SAE)
        vmax = max(d[:n_images].max() for d in datas)

        fig, axes = plt.subplots(n_images, 1, figsize=(9, 4.6 * n_images), squeeze=False, layout="constrained")

        for i, ax in enumerate(axes[:, 0]):
            # T da imagem i em cada camada (negativas já zeradas)
            values = {a: d[i] for a, d in enumerate(datas)}
            title = f"T da imagem {i} | rótulo {y[i]} | predito {preds[i]} (negativas = 0, cinza)"

            self.draw_network(ax, sizes, values=values, cmap=self.POSITIVE, vmin=0, vmax=vmax, image=X[i], highlight=preds[i], title=title)

        self.add_colorbar(fig, axes, self.POSITIVE, 0, vmax, "T (contribuição positiva normalizada)")

        self.save_figure(fig, "2_T_contribuicoes_positivas.png")


    def plot_mode_weights(self, dictionary: np.ndarray, corr: np.ndarray, layer: int, label: int):
        max_corr = corr.argmax(0)
        modes = np.where(max_corr == label)[0]
        modes = modes[np.argsort(-corr[label, modes])][:self.MAX_MODES]

        if len(modes) == 0 or len(modes) <= 3:
            print(f"camada {layer + 1}: nenhum modo tem a classe {self.name(label)} como maior correlação")
            return

        weights = dictionary[modes]

        neurons = np.argsort(-np.abs(weights).sum(0), kind="stable")[:self.MAX_NEURONS]
        weights = weights[:, neurons]
        n_neurons = weights.shape[1]
        vmax = np.abs(weights).max()

        fig, (ax_weights, ax_total) = plt.subplots(2, 1, figsize=(min(0.6 * n_neurons + 3, 14), 0.45 * len(modes) + 4), height_ratios=[len(modes), 3], sharex=True, layout="constrained")

        # Imagem de cima
        image = ax_weights.imshow(weights, cmap=self.DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
        ax_weights.grid(False)
        ax_weights.set_yticks(np.arange(len(modes)), [f"modo {k} (corr {corr[label, k]:.2f})" for k in modes])
        fig.colorbar(image, ax=ax_weights, label="contribuição ao modo")

        # Imagem de baixo
        ax_total.bar(np.arange(n_neurons), np.abs(weights).sum(0), color=self.BLUE)
        ax_total.set_xticks(np.arange(n_neurons), neurons, rotation=90)
        ax_total.set_xlabel("número do neurônio")
        ax_total.set_ylabel("Contribuição do neurônio ao modo.")

        fig.suptitle(f"Layer {layer + 1}: {len(modes)} modos mais correlacionados com a classe {self.name(label)}", fontsize=11)

        self.save_figure(fig, f"3_neurônios_por_modos_na_classe:{label}_camada{layer + 1}.png")


    def visualize_all(self, sizes: list, X: np.ndarray, y: np.ndarray, preds: np.ndarray, contribs: list, dictionaries: list, corrs: list, labels: list, n_images: int = 2):
        if sum(a * b for a, b in zip(sizes, sizes[1:])) <= self.MAX_CONNECTIONS:
            self.plot_neuron_contributions(contribs, X, y, preds, sizes, n_images=n_images)
        else:
            self.plot_neuron_contributions(contribs, X, y, preds, sizes, n_images=n_images, lines=False)

        for l, (dictionary, corr) in enumerate(zip(dictionaries, corrs)):
            if corr is None:
                print(f"camada {l + 1}: sem modos vivos, pesos dos modos pulados")
                continue

            for label in labels:
                self.plot_mode_weights(dictionary, corr, l, label)


    def enhance(self, *maps: np.ndarray):
        # maps: [classes, modos(+soma), H, W], já médios sobre os canais de cor (mode_maps); devolve os mesmos mapas em [0, 1], prontos para virar máscara
        # 1) filtro de mediana (kernel MEDIAN_SIZE) só nos eixos H e W: reduz ruído
        filtered = [median_filter(m, size=(1, 1, self.MEDIAN_SIZE, self.MEDIAN_SIZE)) for m in maps]

        # 2) normalização para [0, 1]: máximo da linha da classe (compartilhado entre os mapas dados; a coluna "soma" tem escala própria)
        scales = class_scales(*filtered)

        # 3) fator de contraste e 4) clip para [0, 1]
        return [np.clip(self.CONTRAST * m / scales, 0, 1) for m in filtered]


    def plot_class_grid(self, panels: np.ndarray, layer: int, title: str, filename: str, modes: np.ndarray, originals: np.ndarray = None):
        n_digits, n_cols = panels.shape[:2]
        per_class = np.ndim(modes) == 2  # modes 1D (iguais para todas as classes) ou [classes, modos] (cada classe com os seus)
        modes = np.broadcast_to(modes, (n_digits, n_cols - 1))
        offset = int(originals is not None)  # com `originals`, a 1ª coluna é a imagem original da classe

        fig, axes = plt.subplots(n_digits, n_cols + offset, figsize=(0.9 * (n_cols + offset), 0.9 * n_digits + 1), squeeze=False, layout="constrained")

        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])

        for digit, row in enumerate(axes):
            if offset:
                row[0].imshow(originals[digit])

            for ax, panel in zip(row[offset:], panels[digit]):
                ax.imshow(panel)

            row[0].set_ylabel(self.name(digit), rotation=0, ha="right", va="center", fontsize=7)

        if offset:
            axes[0, 0].set_title("original", fontsize=7)

        for digit, row in enumerate(axes):
            for k, ax in enumerate(row[offset:]):
                if digit == 0 or per_class:  # modes 1D: títulos só na primeira linha
                    ax.set_title("soma" if k == n_cols - 1 else f"{modes[digit, k]}", fontsize=6 if per_class else 7)

        fig.suptitle(f"Camada {layer + 1}: {title}", fontsize=10)

        self.save_figure(fig, filename)


    def plot_class_mode_maps(self, maps: np.ndarray, layer: int, name: str, cmap: Colormap, modes: np.ndarray, images: np.ndarray = None):
        maps = self.show_modes(maps)
        enhanced, = self.enhance(maps)

        # 5) máscara multiplicativa: com imagem, o mapa escurece os pixels sem contribuição; sem imagem, vira mapa de calor
        panels = cmap(enhanced)[..., :3] if images is None else images[:, None] * enhanced[..., None]

        self.plot_class_grid(panels, layer, f"contribuições {name} (magnitude) por classe e modo (colunas)", f"5_mapas_{name}_por_digito_camada{layer + 1}.png", modes, images)


    def plot_label_images(self, images: np.ndarray, modes: np.ndarray, label: int, layer: int, n_cols: int = 10):
        # images = imagens de teste do dígito (dataset.data); modes = modo de maior loading de cada imagem (loadings_layer_{l}.npy, argmax)
        n_rows = int(np.ceil(len(images) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.3 * n_cols, 1.4 * n_rows + 0.8), squeeze=False, layout="constrained")

        for ax in axes.ravel():
            ax.axis("off")

        for ax, image, mode in zip(axes.ravel(), images, modes):
            ax.imshow(image, cmap="gray")
            ax.set_title(f"modo {mode}", fontsize=7)

        fig.suptitle(f"Classe {self.name(label)}: {len(images)} imagens do teste, ordenadas pelo modo dominante da camada {layer + 1}", fontsize=10)

        self.save_figure(fig, f"6_imagens_do_digito_{label}_camada{layer + 1}.png")


    def plot_class_accuracy(self, results: dict, label: int, layer: int, texts: dict, title: str, prefix: str):
        # results = {situação: métricas} (patching.neural_ablation / neural_preservation); um painel por situação, todos com o mesmo eixo x (classes do teste) e y (acurácia da classe)
        colors = {"intact": self.LINE, "sae": self.BLUE, "random": self.RED}

        fig, axes = plt.subplots(len(results), 1, figsize=(10, 2.2 * len(results) + 1.2), sharex=True, sharey=True, squeeze=False, layout="constrained")

        for ax, (name, result) in zip(axes[:, 0], results.items()):
            accuracy = result["acc_per_class"]

            ax.plot(np.arange(len(accuracy)), accuracy, color=colors[name], lw=0.8)
            ax.axvline(label, color="#b58900", ls="--", lw=1)  # a classe analisada
            ax.set_ylim(-0.02, 1.05)
            ax.set_ylabel("acurácia")
            ax.set_title(f"{texts[name]} | acc total {result['acc_all']:.2f}, prevista como {self.name(label)} {result['frac_pred_label']:.2f}", fontsize=9, loc="left")

        axes[-1, 0].set_xlabel("classes do dataset de teste")

        fig.suptitle(title, fontsize=10)

        self.save_figure(fig, f"{prefix}_camada{layer + 1}_{self.name(label).replace('/', '_')}.png")


    def plot_ablation(self, results: dict, label: int, layer: int, removed_fraction: float):
        texts = {"intact": "modelo intacto", "sae": "remove os neurônios dos modos da classe (SAE)", "random": "remove neurônios ao acaso (mesma quantidade)"}

        self.plot_class_accuracy(results, label, layer, texts, f"Ablação por zero na camada {layer + 1}: {100 * removed_fraction:.1f}% dos neurônios, escolhidos pelos modos da classe {self.name(label)} (linha tracejada)", "7_ablation")


    def plot_preservation(self, results: dict, label: int, layer: int, kept_fraction: float):
        texts = {"intact": "modelo intacto", "sae": "mantém só os neurônios dos modos da classe (SAE)", "random": "mantém só neurônios ao acaso (mesma quantidade)"}

        self.plot_class_accuracy(results, label, layer, texts, f"Preservação na camada {layer + 1}: só {100 * kept_fraction:.1f}% dos neurônios (modos da classe {self.name(label)}, linha tracejada) ficam, o resto é zerado", "7_preservation")