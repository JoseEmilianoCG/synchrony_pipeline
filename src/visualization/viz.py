from pathlib import Path
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 16
})

def load_pickle(path: str):
    """Carga un archivo pickle.

    Args:
        path: Ruta al archivo .pkl.

    Returns:
        Objeto cargado desde pickle.

    Raises:
        FileNotFoundError: Si el archivo no existe.
    """
    file_path = Path(path)
    assert file_path.exists(), f"No existe el archivo: {path}"

    with open(file_path, "rb") as f:
        return pickle.load(f)


def ensure_directories():
    """Crea las carpetas necesarias para el reporte."""
    base = Path("reports")
    figures = base / "figures"
    eda_dir = figures / "eda"
    models_dir = figures / "models"
    optimization_dir = figures / "optimization"

    eda_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    optimization_dir.mkdir(parents=True, exist_ok=True)

    return {
        "base": base,
        "figures": figures,
        "eda": eda_dir,
        "models": models_dir,
        "optimization": optimization_dir,
    }


def validate_metrics(metrics: dict):
    """Valida que las métricas estén en el rango correcto."""
    assert isinstance(metrics, dict), "all_metrics.pkl debe ser un diccionario"
    assert len(metrics) > 0, "No hay modelos en all_metrics.pkl"

    for model_name, model_data in metrics.items():
        assert "accuracy" in model_data, f"Falta accuracy en {model_name}"
        assert "precision" in model_data, f"Falta precision en {model_name}"
        assert "report" in model_data, f"Falta report en {model_name}"

        accuracy = model_data["accuracy"]
        precision = model_data["precision"]

        assert 0.0 <= accuracy <= 1.0, f"Accuracy fuera de rango en {model_name}"
        assert 0.0 <= precision <= 1.0, f"Precision fuera de rango en {model_name}"

        report = model_data["report"]
        assert "macro avg" in report, f"Falta macro avg en report de {model_name}"
        assert "f1-score" in report["macro avg"], f"Falta f1-score macro avg en {model_name}"


def plot_significant_features(eda_df: pd.DataFrame, output_path: Path):
    """Grafica el número de características significativas por sesión."""
    assert not eda_df.empty, "eda_report está vacío"
    assert "Session" in eda_df.columns, "Falta columna Session"
    assert "Significant" in eda_df.columns, "Falta columna Significant"

    summary = (
        eda_df.groupby("Session")["Significant"]
        .sum()
        .reset_index(name="Num_Significant_Features")
    )

    plt.figure(figsize=(9, 5))

    bars = plt.bar(
        summary["Session"].astype(str),
        summary["Num_Significant_Features"],
        color="#64B5CD",
        alpha=0.9
    )

    plt.xlabel("Sesión")
    plt.ylabel("Número de características significativas")
    plt.title("Características significativas de sincronía EEG por sesión", pad=12)

    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    # valores arriba de las barras
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.2,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_top_effect_sizes(eda_df: pd.DataFrame, output_path: Path, top_n: int = 15):
    """Grafica las características con mayor tamaño de efecto absoluto."""
    assert "Feature" in eda_df.columns, "Falta columna Feature"
    assert "Cliffs_delta" in eda_df.columns, "Falta columna Cliffs_delta"

    temp = eda_df.copy()
    temp["abs_effect"] = temp["Cliffs_delta"].abs()
    top_features = temp.sort_values("abs_effect", ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))

    bars = plt.barh(
        top_features["Feature"].str.replace("_", " "),
        top_features["abs_effect"],
        color="#64B5CD",
        alpha=0.9
    )

    plt.xlabel("|Delta de Cliff|")
    plt.ylabel("Característica")
    plt.title("Principales características EEG por tamaño de efecto", pad=12)

    plt.gca().invert_yaxis()
    plt.grid(axis="x", linestyle="--", alpha=0.4)

    # valores al final de cada barra
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.01,
            bar.get_y() + bar.get_height()/2,
            f"{width:.2f}",
            va="center",
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_condition_means(dfs_by_stage: dict, output_path: Path):
    """Grafica el promedio global de sincronía por condición y métrica."""
    assert isinstance(dfs_by_stage, dict), "dfs_by_stage debe ser dict"
    assert all(k in dfs_by_stage for k in ["baseline", "shared", "individual"]), \
        "Faltan condiciones en dfs_by_stage"

    rows = []

    for condition, df in dfs_by_stage.items():
        assert isinstance(df, pd.DataFrame), f"{condition} no es DataFrame"
        assert "Metric" in df.columns, f"Falta columna Metric en {condition}"

        non_feature_cols = ["Session", "Metric", "Epoch"]
        feature_cols = [c for c in df.columns if c not in non_feature_cols]

        for metric_name in df["Metric"].unique():
            sub = df[df["Metric"] == metric_name]
            global_mean = sub[feature_cols].mean().mean()

            rows.append({
                "Condition": condition,
                "Metric": metric_name.upper(),
                "GlobalMeanSynchrony": global_mean
            })

    summary_df = pd.DataFrame(rows)

    # Orden y nombres
    condition_order = ["baseline", "shared", "individual"]
    condition_labels = {
        "baseline": "Baseline",
        "shared": "Shared",
        "individual": "Individual"
    }

    summary_df["Condition"] = pd.Categorical(
        summary_df["Condition"],
        categories=condition_order,
        ordered=True
    )
    summary_df = summary_df.sort_values("Condition")
    summary_df["ConditionLabel"] = summary_df["Condition"].map(condition_labels)

    pivot_df = summary_df.pivot(
        index="ConditionLabel",
        columns="Metric",
        values="GlobalMeanSynchrony"
    )

    ax = pivot_df.plot(
        kind="bar",
        figsize=(9, 5),
        width=0.75
    )

    colors = ["#4C72B0", "#52DDAC"]
    for bars, color in zip(ax.containers, colors):
        for bar in bars:
            bar.set_color(color)
            bar.set_alpha(0.9)

    ax.set_title("Sincronía promedio por condición y métrica", pad=12)
    ax.set_xlabel("Condición")
    ax.set_ylabel("Sincronía promedio (u.a.)")
    ax.legend(title="Métrica", frameon=True)

    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    # Etiquetas numéricas sobre las barras
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def infer_target_mapping(feat_df: pd.DataFrame, eda_df: pd.DataFrame):
    """Infiere qué valor de Target corresponde a Shared e Individual."""
    assert "Target" in feat_df.columns, "Falta columna Target en feat_mat"
    assert "Feature" in eda_df.columns, "Falta columna Feature en eda_report"
    assert "Mean_Shared" in eda_df.columns, "Falta Mean_Shared en eda_report"
    assert "Mean_Individual" in eda_df.columns, "Falta Mean_Individual en eda_report"

    sig_df = eda_df[eda_df["Significant"] == True].copy()
    assert not sig_df.empty, "No hay features significativas para inferir el mapeo"

    feature = sig_df.iloc[0]["Feature"]
    assert feature in feat_df.columns, f"La feature {feature} no existe en feat_mat"

    group_means = feat_df.groupby("Target")[feature].mean()

    if len(group_means.index) != 2:
        raise ValueError("Target no tiene exactamente 2 clases")

    t0, t1 = group_means.index.tolist()
    mean0 = group_means.loc[t0]
    mean1 = group_means.loc[t1]

    shared_ref = sig_df.iloc[0]["Mean_Shared"]
    individual_ref = sig_df.iloc[0]["Mean_Individual"]

    dist_option_1 = abs(mean0 - shared_ref) + abs(mean1 - individual_ref)
    dist_option_2 = abs(mean0 - individual_ref) + abs(mean1 - shared_ref)

    if dist_option_1 <= dist_option_2:
        return {t0: "Shared", t1: "Individual"}
    return {t0: "Individual", t1: "Shared"}


def plot_boxplot_significant_features(
    feat_df: pd.DataFrame,
    eda_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 3
):
    """Genera boxplots para las features significativas más importantes."""
    assert isinstance(feat_df, pd.DataFrame), "feat_mat debe ser DataFrame"
    assert isinstance(eda_df, pd.DataFrame), "eda_report debe ser DataFrame"
    assert "Target" in feat_df.columns, "Falta columna Target en feat_mat"
    assert "Feature" in eda_df.columns, "Falta columna Feature en eda_report"
    assert "Significant" in eda_df.columns, "Falta columna Significant en eda_report"
    assert "p_value_fdr" in eda_df.columns, "Falta p_value_fdr en eda_report"
    assert "Cliffs_delta" in eda_df.columns, "Falta Cliffs_delta en eda_report"

    sig_df = eda_df[eda_df["Significant"] == True].copy()
    assert not sig_df.empty, "No hay features significativas para boxplot"

    sig_df["abs_effect"] = sig_df["Cliffs_delta"].abs()

    top_features = (
        sig_df.sort_values(["p_value_fdr", "abs_effect"], ascending=[True, False])
        .head(top_n)["Feature"]
        .tolist()
    )

    label_map = infer_target_mapping(feat_df, eda_df)
    target_values = sorted(feat_df["Target"].unique().tolist())

    fig, axes = plt.subplots(1, len(top_features), figsize=(5 * len(top_features), 5), squeeze=False)

    for ax, feature in zip(axes[0], top_features):
        assert feature in feat_df.columns, f"La feature {feature} no existe en feat_mat"

        grouped_data = []
        grouped_labels = []

        for target_value in target_values:
            vals = feat_df.loc[feat_df["Target"] == target_value, feature].dropna().values
            grouped_data.append(vals)
            grouped_labels.append(label_map.get(target_value, f"Target {target_value}"))

        box = ax.boxplot(
            grouped_data,
            tick_labels=grouped_labels,
            patch_artist=True,
            flierprops=dict(
            marker='o',
            markersize=2,      # 👈 tamaño más pequeño
            markerfacecolor='gray',
            alpha=0.5
    )
)

        colors = ["#B0BDD3", "#D1EEE4"]
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)

        for median in box["medians"]:
            median.set_color("black")
            median.set_linewidth(2)

        ax.set_title(feature.replace("_", " "), fontsize=13)
        ax.set_xlabel("Condición")
        ax.set_ylabel("Valor de sincronía (u.a.)")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.suptitle("Características significativas de sincronía EEG", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_feature_correlation_heatmap(feat_df: pd.DataFrame, output_path: Path, top_n: int = 20):
    """Genera heatmap de correlación de las features con mayor varianza."""
    assert isinstance(feat_df, pd.DataFrame), "feat_mat debe ser DataFrame"
    assert not feat_df.empty, "feat_mat está vacío"

    excluded = ["Session", "Target"]
    feature_cols = [c for c in feat_df.columns if c not in excluded]
    assert len(feature_cols) > 0, "No se encontraron columnas de features"

    variances = feat_df[feature_cols].var().sort_values(ascending=False)
    selected_features = variances.head(top_n).index.tolist()

    corr = feat_df[selected_features].corr()

    plt.figure(figsize=(10, 8))
    im = plt.imshow(corr, aspect="auto")
    plt.colorbar(im, label="Correlation")
    plt.xticks(range(len(selected_features)), selected_features, rotation=90)
    plt.yticks(range(len(selected_features)), selected_features)
    plt.title("Feature correlation heatmap (top variance features)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_importance(best_model, feat_df: pd.DataFrame, output_path: Path, top_n: int = 15):
    """Grafica la importancia de características del mejor modelo."""
    assert hasattr(best_model, "feature_importances_"), \
        "El modelo no tiene atributo feature_importances_"

    importances = best_model.feature_importances_

    if hasattr(best_model, "feature_names_in_"):
        feature_cols = list(best_model.feature_names_in_)
    else:
        excluded = ["Target"]
        feature_cols = [c for c in feat_df.columns if c not in excluded]

        min_len = min(len(importances), len(feature_cols))
        feature_cols = feature_cols[:min_len]
        importances = importances[:min_len]

    assert len(importances) == len(feature_cols), \
        f"No coinciden importances ({len(importances)}) y feature_cols ({len(feature_cols)})"

    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importances
    }).sort_values("Importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))

    bars = plt.barh(
        importance_df["Feature"].str.replace("_", " "),
        importance_df["Importance"],
        color="#64B5CD", 
        alpha=0.9
    )

    plt.xlabel("Importancia")
    plt.ylabel("Característica")
    plt.title("Importancia de variables del mejor modelo", pad=12)

    plt.gca().invert_yaxis()
    plt.grid(axis="x", linestyle="--", alpha=0.4)

    # ✨ valores al final de cada barra
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.002,
            bar.get_y() + bar.get_height()/2,
            f"{width:.3f}",
            va="center",
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def build_metrics_table(metrics: dict) -> pd.DataFrame:
    """Construye una tabla comparativa de métricas."""
    rows = []

    for model_name, model_data in metrics.items():
        row = {
            "Modelo": model_name,
            "Exactitud": model_data["accuracy"],
            "Precisión": model_data["precision"],
            "F1 (macro)": model_data["report"]["macro avg"]["f1-score"],
            "F1 (ponderado)": model_data["report"]["weighted avg"]["f1-score"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # 🔥 redondeo (clave para que no se vea feo en LaTeX)
    df = df.round(3)

    return df.sort_values("Exactitud", ascending=False).reset_index(drop=True)


def plot_model_comparison(metrics_df: pd.DataFrame, output_path: Path):
    """Grafica comparación de desempeño de modelos."""
    plt.figure(figsize=(9, 5))

    x = range(len(metrics_df))
    width = 0.35

    bars1 = plt.bar(
        [i - width / 2 for i in x],
        metrics_df["Exactitud"],
        width=width,
        label="Exactitud",
        color="#4C72B0",
        alpha=0.9
    )

    bars2 = plt.bar(
        [i + width / 2 for i in x],
        metrics_df["F1 (macro)"],
        width=width,
        label="F1 (macro)",
        color="#52DDAC",
        alpha=0.9
    )

    plt.xticks(list(x), metrics_df["Modelo"], rotation=20)
    plt.ylim(0, 1.0)

    plt.xlabel("Modelo")
    plt.ylabel("Puntaje")
    plt.title("Comparación de desempeño de modelos", pad=12)

    plt.legend(frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    # ✨ valores arriba de las barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2,
                height + 0.01,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_metrics_table(metrics_df: pd.DataFrame, output_csv: Path):
    """Guarda la tabla comparativa en CSV."""
    metrics_df.to_csv(output_csv, index=False)


def write_latex_report(paths: dict, metrics_df: pd.DataFrame):
    """Escribe un reporte LaTeX base"""
    tex_path = paths["base"] / "final_report.tex"

    latex_table = metrics_df.to_latex(index=False, float_format="%.3f", escape=True)

    content = f"""
\\documentclass[12pt]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage[spanish]{{babel}}
\\usepackage{{graphicx}}
\\usepackage{{float}}
\\usepackage{{booktabs}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{array}}
\\setlength{{\\parskip}}{{0.6em}}
\\setlength{{\\parindent}}{{0pt}}


\\begin{{document}}

\\begin{{titlepage}}
    \\centering

    \\vspace*{{1.5cm}}

    \\vspace{{1.3cm}}

    {{\\Large Tecnológico de Monterrey\\par}}
    \\vspace{{0.6cm}}

    {{\\Huge \\textbf{{Proyecto Final Integrador}}\\par}}
    \\vspace{{0.4cm}}

    {{\\Large Dominio: Neuroingeniería\\par}}

    \\vspace{{2cm}}

    {{\\large
    José Emiliano Calderón Gurubel\\par
    Dana Paola Rosete Gómez\\par
    José Ricardo Cañedo Verdugo\\par
    Daisy Karina Núñez Flores\\par
    }}

    \\vspace{{1.5cm}}

    {{\\large \\textit{{Applied Computing}}\\par}}

    \\vfill

    {{\\large 10 de abril de 2026\\par}}

\\end{{titlepage}}

\\section{{Resumen}}
\\addcontentsline{{toc}}{{section}}{{Resumen}}

\\vspace{{0.3cm}}

Este reporte presenta un análisis integrado que combina exploración de datos, pruebas estadísticas, modelos de aprendizaje automático y optimización, aplicado a características de sincronía EEG obtenidas en condiciones de lectura compartida e individual. El objetivo es evaluar si estas características, derivadas de díadas familiares, permiten discriminar entre distintos estados cognitivos asociados a cada condición. Los resultados indican que la sincronía cerebral constituye un indicador informativo para diferenciar entre lectura individual y compartida, destacando el potencial de estas métricas en el análisis de interacción cognitiva.

\\vspace{{0.8cm}}

\\section{{Metodología}}

El pipeline desarrollado se estructuró en módulos secuenciales que permiten transformar señales EEG crudas en resultados interpretables mediante análisis estadístico, aprendizaje automático y visualización.

En primer lugar (C1), se realizó la carga, limpieza y preprocesamiento de las señales EEG obtenidas en distintas condiciones experimentales: línea base, lectura compartida y lectura individual. A partir de estas señales, se extrajeron características de sincronía cerebral entre pares de electrodos utilizando métricas como PLI y wPLI en distintas bandas de frecuencia. Estas características se organizaron en matrices de datos para su posterior análisis.

Posteriormente, se llevó a cabo un análisis exploratorio de datos (EDA), en el cual se evaluaron diferencias entre condiciones mediante pruebas estadísticas no paramétricas. En particular, se utilizaron comparaciones entre grupos (Paired) acompañadas de corrección por múltiples pruebas (FDP), así como el cálculo del tamaño de efecto mediante la medición Cliff’s d y Wilcoxon. 

En la etapa de aprendizaje automático (C4), se entrenaron distintos modelos de clasificación, incluyendo Random Forest, Support Vector Machine (SVM) y Gradient Boosting, con el objetivo de distinguir entre condiciones experimentales a partir de las características EEG. El desempeño de los modelos se evaluó mediante métricas como exactitud, precisión y F1-score.

Adicionalmente, se consideró una etapa de optimización (C3), en la cual se exploran estrategias para mejorar el desempeño de los modelos y la selección de características, contribuyendo a una mejor generalización del sistema.

Finalmente, en el módulo de visualización e integración (C6), se generaron figuras y reportes que permiten interpretar los resultados obtenidos, incluyendo análisis de características significativas, comparaciones entre condiciones, matrices de correlación y desempeño de modelos.

En este proyecto, la comparación final se realizó entre los modelos de aprendizaje automático implementados en C4. Los módulos C2 y C5 no fueron incluidos en la tabla comparativa final debido a que, de acuerdo con la naturaleza del problema y las reglas de composición del framework para equipos de este tamaño, C2 no resultó aplicable y C5 fue omitido justificadamente.

\\section{{Resultados del análisis exploratorio (C1)}}
\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.8\\textwidth]{{figures/eda/significant_features_by_session.png}}
    \\caption{{Número de características de sincronía EEG estadísticamente significativas por sesión.}}
\\end{{figure}}

\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.9\\textwidth]{{figures/eda/top_effect_sizes.png}}
    \\caption{{Principales características EEG ordenadas por magnitud del tamaño de efecto absoluto (Delta de Cliff).}}
\\end{{figure}}

\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.8\\textwidth]{{figures/eda/condition_means.png}}
    \\caption{{Sincronía promedio EEG por condición experimental (línea base, lectura compartida y lectura individual) y por métrica.}}
\\end{{figure}}

\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.95\\textwidth]{{figures/eda/boxplot_significant_features.png}}
    \\caption{{Diagramas de caja de las características de sincronía EEG más significativas, comparando lectura compartida e individual.}}
\\end{{figure}}

\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.9\\textwidth]{{figures/eda/feature_correlation_heatmap.png}}
    \\caption{{Mapa de correlación de las características EEG con mayor varianza.}}
\\end{{figure}}

\\section{{Resultados de aprendizaje automático (C4)}}
\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.8\\textwidth]{{figures/models/model_comparison.png}}
    \\caption{{Comparación del desempeño de los modelos utilizando exactitud y F1 macro.}}
\\end{{figure}}

\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.9\\textwidth]{{figures/models/feature_importance.png}}
    \\caption{{Importancia de variables en el modelo de mejor desempeño.}}
\\end{{figure}}

\\section{{Resultados de optimización (C3)}}

\\section{{Tabla comparativa de modelos}}
\\begin{{table}}[H]
\\centering
\\caption{{Comparación de desempeño entre modelos de aprendizaje automático.}}
{latex_table}
\\end{{table}}

\\section{{Conclusiones}}

Texto

\\end{{document}}
"""

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(content)

def generate_report():
    """Genera figuras, tabla comparativa y reporte LaTeX."""
    paths = ensure_directories()

    eda_df = load_pickle("src/C1/outputs/eda_report.pkl")
    dfs_by_stage = load_pickle("src/C1/outputs/dfs_by_stage.pkl")
    feat_df = load_pickle("src/C1/outputs/feat_mat.pkl")

    metrics = load_pickle("src/C4/outputs/all_metrics.pkl")
    best_model = load_pickle("src/C4/outputs/best_model.pkl")

    assert isinstance(eda_df, pd.DataFrame), "eda_report.pkl debe ser DataFrame"
    assert isinstance(feat_df, pd.DataFrame), "feat_mat.pkl debe ser DataFrame"
    validate_metrics(metrics)

    plot_significant_features(
        eda_df,
        paths["eda"] / "significant_features_by_session.png"
    )

    plot_top_effect_sizes(
        eda_df,
        paths["eda"] / "top_effect_sizes.png"
    )

    plot_condition_means(
        dfs_by_stage,
        paths["eda"] / "condition_means.png"
    )

    plot_boxplot_significant_features(
        feat_df,
        eda_df,
        paths["eda"] / "boxplot_significant_features.png"
    )

    plot_feature_correlation_heatmap(
        feat_df,
        paths["eda"] / "feature_correlation_heatmap.png"
    )

    metrics_df = build_metrics_table(metrics)

    plot_model_comparison(
        metrics_df,
        paths["models"] / "model_comparison.png"
    )

    plot_feature_importance(
        best_model,
        feat_df,
        paths["models"] / "feature_importance.png"
    )

    save_metrics_table(
        metrics_df,
        paths["base"] / "model_comparison.csv"
    )

    write_latex_report(paths, metrics_df)

    print("C6 ejecutado correctamente.")
    print(f"Reporte LaTeX: {paths['base'] / 'final_report.tex'}")
    print(f"Tabla comparativa: {paths['base'] / 'model_comparison.csv'}")
    print(f"Figuras generadas en: {paths['figures']}")


if __name__ == "__main__":
    generate_report()