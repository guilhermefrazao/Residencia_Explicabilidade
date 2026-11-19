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


def class_scales(*maps: np.ndarray, percentile: float = 100):
    # escala por dígito: as colunas dos modos dividem o pico da linha (entre todos os mapas dados); a coluna "soma" tem escala própria
    # pico = percentil dos valores do mapa (100 = máximo): um percentil menor evita que alguns pixels extremos escureçam o resto
    spatial = tuple(range(2, maps[0].ndim))  # H e W nas imagens; as variáveis nos vetores

    peak = np.max([np.percentile(m, percentile, axis=spatial) for m in maps], axis=0)
    peak[:, :-1] = peak[:, :-1].max(1, keepdims=True)

    return np.maximum(peak, 1e-9)[(...,) + (None,) * len(spatial)]


class Visualization:
    def __init__(self, dataset: str, network: str, out: str = "images", class_names: list = None, experiment_name: str = ""):
        self.class_names = class_names
        self.observation_names = None  # nomes das variáveis da observação (RL); None mostra o índice

        date = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%d-%m-%Y - %H:%M")

        name = f"{dataset}_{network}_{date}"

        self.out = os.path.join(out, f"{experiment_name}_{name}" if experiment_name else name)  # pasta das figuras: prefixada pelo experimento

        self.GRAY, self.LINE, self.BLUE, self.RED, self.CREAM = "#eee8d5", "#93a1a1", "#268bd2", "#dc322f", "#fdf6e3"
        self.DIVERGING = LinearSegmentedColormap.from_list("solarized_div", [self.RED, self.CREAM, self.BLUE])
        self.POSITIVE = LinearSegmentedColormap.from_list("solarized_pos", [self.GRAY, self.BLUE])
        self.NEGATIVE = LinearSegmentedColormap.from_list("solarized_neg", [self.GRAY, self.RED])
        self.MAX_MODES, self.MAX_CONNECTIONS, self.MAX_NEURONS = 10, 100_000, 64
        self.MEDIAN_SIZE, self.CONTRAST, self.PEAK_PERCENTILE = 4, 2, 99  # filtro de mediana (1 = desligado), fator de contraste (1 = desligado) e percentil que define o pico (100 = máximo) dos mapas de contribuição


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
        if image is not None and np.ndim(image) >= 2:  # 2D (tons de cinza) ou 3D (HxWxC) = imagem
            inset = ax.inset_axes([0.0, 0.3, (x_in - 0.1) / width, 0.4])
            inset.imshow(image, cmap="gray")
            inset.axis("off")
        elif image is not None:  # vetor (ex.: observação de RL): escreve os valores no lugar da imagem
            names = self.observation_names or []
            values_text = "\n".join(f"{names[i] if i < len(names) else i}: {v:.3f}" for i, v in enumerate(np.ravel(image)))
            ax.text(x_in - 0.6, 0, f"entrada\n{values_text}", ha="center", va="center", fontsize=8)
        else:
            ax.text(x_in - 0.6, 0, "entrada\npx", ha="center", va="center", fontsize=8)

        ax.set_xlim(0, width)
        ax.set_ylim(-7.2, 5.8)
        ax.axis("off")

        # Título
        if title:
            ax.set_title(title, fontsize=10)


    def show_modes(self, maps: np.ndarray):
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

        # Cria a png por matriz e rede.
        for i, ax in enumerate(axes[:, 0]):
            # contribuição da matriz i em cada camada | X[i]: matriz de teste, y[i]: Esperado do dataset, preds[i]: argmax da saída da rede
            values = {a: s[i] for a, s in enumerate(signed)}
            title = f"matriz {i} | Esperado {y[i]} | predito {preds[i]} (círculo laranja)"

            self.draw_network(ax, sizes, values=values, cmap=self.DIVERGING, vmax=vmax, image=X[i], highlight=preds[i], title=title, lines=lines)

        #Desenha a escala dos neurônios como uma mapa separando por cor.
        self.add_colorbar(fig, axes, self.DIVERGING, -vmax, vmax, "contribuição do neurônio (vermelho < 0 < azul)")

        self.save_figure(fig, "1_contribuicao_por_neuronio.png")


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

        self.save_figure(fig, f"2_neurônios_por_modos_na_classe:{label}_camada{layer + 1}.png")


    def visualize_all(self, sizes: list, X: np.ndarray, y: np.ndarray, preds: np.ndarray, contributions: list, dictionaries: list, corrs: list, labels: list, n_images: int = 2):
        if sum(a * b for a, b in zip(sizes, sizes[1:])) <= self.MAX_CONNECTIONS:
            self.plot_neuron_contributions(contributions, X, y, preds, sizes, n_images=n_images)
        else:
            self.plot_neuron_contributions(contributions, X, y, preds, sizes, n_images=n_images, lines=False)

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
        scales = class_scales(*filtered, percentile=self.PEAK_PERCENTILE)

        # 3) fator de contraste e 4) clip para [0, 1]
        return [np.clip(self.CONTRAST * m / scales, 0, 1) for m in filtered]


    def plot_class_grid(self, panels: np.ndarray, layer: int, title: str, filename: str, modes: np.ndarray, originals: np.ndarray = None, labels: list = None):
        # labels[i] = id (da label) da linha i, para o nome da classe; sem labels, a posição da linha é usada como id
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

            row[0].set_ylabel(self.name(digit if labels is None else labels[digit]), rotation=0, ha="right", va="center", fontsize=7)

        if offset:
            axes[0, 0].set_title("original", fontsize=7)

        for digit, row in enumerate(axes):
            for k, ax in enumerate(row[offset:]):
                if digit == 0 or per_class:  # modes 1D: títulos só na primeira linha
                    ax.set_title("soma" if k == n_cols - 1 else f"{modes[digit, k]}", fontsize=6 if per_class else 7)

        fig.suptitle(f"Camada {layer + 1}: {title}", fontsize=10)

        self.save_figure(fig, filename)


    def plot_class_mode_maps_images(self, maps_single: np.ndarray, maps_mean: np.ndarray, layer: int, modes: np.ndarray, labels: list, input_data: np.ndarray = None):
        # maps_single: [classes, modos, H, W] da 1ª imagem de cada classe (a mesma de input_data); maps_mean: o mesmo, médio sobre as imagens da classe
        # máscara multiplicativa: o mapa da imagem mostrada escurece os pixels sem contribuição (só faz sentido com a imagem original)
        if input_data is not None:
            enhanced, = self.enhance(self.show_modes(maps_single))

            self.plot_class_grid(input_data[:, None] * enhanced[..., None], layer, "contribuições Positivas (máscara na imagem) por classe e modo", f"3_mapas_Positivaos_mascara_imagem_camada{layer + 1}.png", modes, input_data, labels)

        # heatmap da média da classe: não corresponde a uma imagem só, então sem a coluna "original"
        enhanced, = self.enhance(self.show_modes(maps_mean))

        self.plot_class_grid(self.POSITIVE(enhanced)[..., :3], layer, "contribuições Positivas (heatmap da média da classe) por classe e modo", f"3_mapas_Positivos_heatmap_medio_camada{layer + 1}.png", modes, None, labels)


    def plot_class_mode_maps_vector(self, maps: np.ndarray, layer: int, modes: np.ndarray, labels: list = None):
        # maps: [classes, modos, variáveis] (contribuição de cada variável da observação para cada modo); um mapa de calor variável x modo por classe
        n_classes, n_cols, n_vars = maps.shape
        normalized = maps / np.maximum(maps.max(axis=(1, 2), keepdims=True), 1e-9)  # máximo da classe: os modos de uma classe dividem a mesma escala

        names = self.observation_names or []
        var_labels = [names[i] if i < len(names) else str(i) for i in range(n_vars)]
        modes = np.broadcast_to(modes, (n_classes, n_cols))

        fig, axes = plt.subplots(n_classes, 1, figsize=(0.8 * n_cols + 2, 1.0 * n_vars * n_classes + 1.2), squeeze=False, layout="constrained")

        for digit, ax in enumerate(axes[:, 0]):
            ax.imshow(normalized[digit].T, cmap=self.POSITIVE, vmin=0, vmax=1, aspect="auto")
            ax.set_yticks(range(n_vars), var_labels, fontsize=7)
            ax.set_xticks(range(n_cols), [f"{m}" for m in modes[digit]], fontsize=7)
            ax.grid(False)
            ax.set_ylabel(self.name(digit if labels is None else labels[digit]), rotation=0, ha="right", va="center", fontsize=8)

        fig.suptitle(f"Camada {layer + 1}: contribuições Positivas (magnitude) por variável, classe e modo", fontsize=10)

        self.save_figure(fig, f"3_mapas_Positivas_por_variavel_camada{layer + 1}.png")


    def ablation_texts(self):
        return {"intact": "modelo intacto", "sae": "remove os neurônios dos modos da classe (SAE)", "random": "remove neurônios ao acaso (mesma quantidade)"}


    def preservation_texts(self):
        return {"intact": "modelo intacto", "sae": "mantém só os neurônios dos modos da classe (SAE)", "random": "mantém só neurônios ao acaso (mesma quantidade)"}


    def plot_ablation_images(self, results: dict, label: int, layer: int, removed_fraction: float):
        self.plot_class_accuracy(results, label, layer, self.ablation_texts(), f"Remoção de - {100 * removed_fraction:.1f}% dos neurônios camada {layer + 1}, Relacionados com a classe - {self.name(label)}", "4_ablation")


    def plot_ablation_vector(self, results: dict, label: int, layer: int, removed_fraction: float):
        self.plot_class_bars(results, label, layer, self.ablation_texts(), f"Remoção de - {100 * removed_fraction:.1f}% dos neurônios camada {layer + 1}, Relacionados com a classe - {self.name(label)}", "4_ablation_vector")


    def plot_preservation_vector(self, results: dict, label: int, layer: int, kept_fraction: float):
        self.plot_class_bars(results, label, layer, self.preservation_texts(), f"Preservação de - {100 * kept_fraction:.1f}% dos neurônios camada {layer + 1}, Relacionados com a classe - {self.name(label)}", "4_preservation_vector")


    def plot_preservation_images(self, results: dict, label: int, layer: int, kept_fraction: float):
        texts = self.preservation_texts()

        self.plot_class_accuracy(results, label, layer, texts, f"Preservação de - {100 * kept_fraction:.1f}% dos neurônios camada {layer + 1}, Relacionados com a classe - {self.name(label)}", "4_preservation")

    

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

        self.save_figure(fig, f"5_imagens_do_digito_{label}_camada{layer + 1}.png")


    def plot_class_bars(self, results: dict, label: int, layer: int, texts: dict, title: str, prefix: str):
        # para poucas classes (ex.: ações de RL), a curva de acurácia por classe não diz nada: um gráfico de barras por situação, com as 3 métricas que importam
        colors = {"intact": self.LINE, "sae": self.BLUE, "random": self.RED}
        metrics = ["acc_label", "frac_pred_label", "acc_all"]

        fig, axes = plt.subplots(len(results), 1, figsize=(6, 2.4 * len(results) + 1.2), sharey=True, squeeze=False, layout="constrained")

        for ax, (name, result) in zip(axes[:, 0], results.items()):
            bars = ax.bar(metrics, [float(result[m]) for m in metrics], color=colors[name])

            ax.bar_label(bars, fmt="%.3f", fontsize=8)
            ax.set_ylim(0, 1.12)
            ax.set_ylabel("valor")
            ax.set_title(texts[name], fontsize=9, loc="left")

        fig.suptitle(title, fontsize=10)

        self.save_figure(fig, f"{prefix}_camada{layer + 1}_{self.name(label).replace('/', '_')}.png")


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