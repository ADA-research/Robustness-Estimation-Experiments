"""Shared loading, metric and plotting helpers for the analysis notebooks.

All result files live in ../results (see the README for a description of the columns).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = ROOT / "analysis"
DATA_DIR = ROOT / "results"
PLOT_DIR = ANALYSIS_DIR / "plots"
TABLE_DIR = ANALYSIS_DIR / "tables"
PLOT_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)

# alpha,beta-CROWN on CIFAR-10: the JAIR run (3 600 s timeout per query) on a class-balanced sample of 100 test images.
# image_id is the CIFAR-10 test index (converted from the position in the JAIR sample, see the README).
CIFAR_ABCROWN_RESULTS = DATA_DIR / "cifar10" / "abcrown_3600s.csv"
CIFAR_SAMPLE = DATA_DIR / "cifar10" / "balanced_sample_test_indices.csv"

# Discretisation of the search spaces used in experiments/cifar10/ and experiments/imagenet/
CIFAR_STEP = 0.0039      # np.arange(0.00, 0.4, 0.0039)
IMAGENET_STEP = 0.0001   # np.arange(0.00, 0.4, 0.0001)
CIFAR_CAP = 0.3978       # largest value in the CIFAR search space -> assigned when no adversarial example is found
IMAGENET_CAP = 0.3999    # largest value in the ImageNet search space
MNIST_CAP = 101 / 255    # largest value of the MNIST search space reached by binary search (FAB failures: 0.4)

# Instances at the cap are failed attacks: no adversarial example was found, so they are not an observed value.
# With EXCLUDE_FAILURES = True, averages, standard deviations and statistical tests use successful instances only
# and the tables report failure rates. Rank-based results (RMC, VBA choice, f_opt) are unaffected either way.
# Set to False to include the cap values, as the submitted manuscript did. This does not reproduce all of its tables,
# since other corrections apply as well.
EXCLUDE_FAILURES = True

IMAGENET_NAMES = {
    "eva_giant_patch14_224.clip_ft_in1k": "eva",
    "beitv2_large_patch16_224.in1k_ft_in22k_in1k": "beitv2",
    "vit_large_patch14_clip_224.openai_ft_in12k_in1k": "openai_1",
    "vit_large_patch14_clip_224.openai_ft_in1k": "openai_2",
    "vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k": "laion_2b",
}
IMAGENET_ORDER = ["eva", "beitv2", "openai_2", "laion_2b", "openai_1"]
CIFAR_ORDER = ["conv_big", "cifar_7_1024", "resnet_4b"]
MNIST_ORDER = ["mnist_relu_3_50", "mnist_relu_9_100", "mnist_relu_4_1024", "mnist-net_256x2"]
ALGORITHM_ORDER = ["ABcrown", "AutoAttack", "PGD", "PGD random", "FGSM", "targeted FAB", "untargeted FAB"]
CIFAR_ALGORITHMS = ["ABcrown", "AutoAttack", "PGD", "FGSM"]
MNIST_NAMES = {
    "abcrown": "ABcrown",
    "autoattack": "AutoAttack",
    "pgd_40": "PGD",
    "pgd_40_random": "PGD random",
    "fgsm": "FGSM",
    "targeted_fab_attack": "targeted FAB",
    "untargeted_fab_attack": "untargeted FAB",
}
MNIST_STEP = 1 / 255

# Figure conventions of the paper
FULL_XLIM = (0, 0.39)    # MNIST/ImageNet CDFs end just before the search cap
PAPER_LEGEND_LABELS = {"PGD random": "PGD_random", "targeted FAB": "targeted_FAB", "untargeted FAB": "untargeted_FAB"}
# Colours follow the position in the method list of the dataset (MNIST: ALGORITHM_ORDER, CIFAR-10/ImageNet:
# CIFAR_ALGORITHMS), so a method keeps its colour within a dataset even if alpha,beta-CROWN is not plotted.
MNIST_COLORS = dict(zip(ALGORITHM_ORDER, sns.color_palette("Paired")))
CIFAR_IMAGENET_COLORS = dict(zip(CIFAR_ALGORITHMS, sns.color_palette("Paired")))
# Figures D1/D2 use the raw verifier names in this order
RAW_NAMES = {short: raw for raw, short in MNIST_NAMES.items()}
RAW_ORDER = ["abcrown", "autoattack", "untargeted_fab_attack", "fgsm", "targeted_fab_attack", "pgd_40", "pgd_40_random"]
SANS = {"font.family": "sans-serif"}   # Figure 5 and the heatmaps in Appendix D use the default sans-serif font


def set_style(figsize=(11.7, 8.27)):
    sns.set_style("darkgrid")
    sns.set_theme(rc={"figure.figsize": figsize})
    sns.set_palette(sns.color_palette("Paired"))
    sns.set_style({"font.family": "serif", "font.serif": "Times New Roman"})


def map_attack(item):
    if "PGD" in item:
        return "PGD"
    elif "AutoAttack" in item:
        return "AutoAttack"
    else:
        return "FGSM"


def map_top_k(item):
    return "top-1" if "top-1" in item else "top-5"


def load_attack_results(file_name, network_names=None, cap=None):
    """Load a compiled VERONA result file, map verifier strings to short names and drop duplicate runs."""
    df = pd.read_csv(DATA_DIR / file_name)
    df["top_k"] = df.verifier.apply(map_top_k)
    df["verifier"] = df.verifier.apply(map_attack)
    df["smallest_sat_value"] = pd.to_numeric(df.smallest_sat_value, errors="coerce")
    df["network"] = df.network.str.replace("_standard", "", regex=False)
    if network_names is not None:
        df["network"] = df.network.map(lambda x: network_names.get(x, x))
    # Some images appear more than once. Keep the last row; the files are not guaranteed to be in chronological order.
    df = df.drop_duplicates(subset=["network", "verifier", "top_k", "image_id"], keep="last")
    if cap is not None:
        df["failed"] = df.smallest_sat_value >= cap - 1e-9
    return df.reset_index(drop=True)


def load_cifar():
    return load_attack_results("cifar10/attacks.csv", cap=CIFAR_CAP)


def load_cifar_monotonicity():
    return load_attack_results("cifar10/monotonicity.csv")


def load_imagenet():
    return load_attack_results("imagenet/attacks_top5.csv", IMAGENET_NAMES, cap=IMAGENET_CAP)


def load_imagenet_top1():
    return load_attack_results("imagenet/attacks_top1.csv", IMAGENET_NAMES, cap=IMAGENET_CAP)


def load_imagenet_monotonicity():
    return load_attack_results("imagenet/monotonicity.csv", IMAGENET_NAMES)


def load_mnist(all_verifiers=False):
    """MNIST results (first 100 images for abcrown, complete test set for the attacks).

    FAB returns a continuous perturbation size; as in the original analysis it is projected onto the 1/255 grid.
    Instances where FAB found no adversarial example have p* = 0.4.
    """
    df = pd.read_csv(DATA_DIR / "mnist" / "results.csv", low_memory=False)
    is_fab = df.verifier.str.contains("fab_attack")
    df.loc[is_fab, "smallest_sat_value"] = np.ceil(df.loc[is_fab, "smallest_sat_value"] * 255) / 255
    if not all_verifiers:
        df = df[df.verifier.isin(MNIST_NAMES.keys())].copy()
        df["verifier"] = df.verifier.map(MNIST_NAMES)
    df["failed"] = df.smallest_sat_value >= MNIST_CAP - 1e-9
    return df.reset_index(drop=True)


def load_mnist_monotonicity():
    """Iterative search over all epsilons, first 100 images (Table 2); same columns as the CIFAR-10/ImageNet files.

    `smallest_sat_value` is "undefined" when no epsilon gave an adversarial example. AutoAttack on mnist_relu_4_1024
    appears twice; as for the other result files, the last row is kept.
    """
    df = pd.read_csv(DATA_DIR / "mnist" / "monotonicity.csv")
    df["verifier"] = df.verifier.map(MNIST_NAMES)
    df["smallest_sat_value"] = pd.to_numeric(df.smallest_sat_value, errors="coerce")
    df = df.drop_duplicates(subset=["network", "verifier", "image_id"], keep="last")
    return df.reset_index(drop=True)


def load_cifar_abcrown(path=CIFAR_ABCROWN_RESULTS):
    df = pd.read_csv(path)
    df = df[df.network.str.endswith("_standard")].copy()
    df["network"] = df.network.str.replace("_standard", "", regex=False)
    df["verifier"] = "ABcrown"
    df["top_k"] = "top-1"
    df["failed"] = df.smallest_sat_value >= CIFAR_CAP - 1e-9
    return df[["network", "image_id", "original_label", "epsilon_value", "smallest_sat_value", "total_time", "verifier", "top_k", "failed"]]


def load_cifar_sample_ids():
    """CIFAR-10 test indices of the class-balanced 100-image sample used for alpha,beta-CROWN."""
    return pd.read_csv(CIFAR_SAMPLE).image_id


def snap_cifar_values(df):
    """Round attack and abcrown values onto the search space 0.0039*k.

    Both use the same search space, but the stored floats can differ in the last digits. Without rounding,
    equal results could be counted as different (e.g. for the VBA, RMC and f_opt).
    """
    df = df.copy()
    for column in ["epsilon_value", "smallest_sat_value"]:
        df[column] = np.round(df[column] / CIFAR_STEP) * CIFAR_STEP
    return df


def failure_mask(df, cap):
    """Instances for which the attack never found an adversarial example (value set to the search cap)."""
    return df.smallest_sat_value >= cap - 1e-9


def successful(df):
    """Rows used for averages and statistical tests (all rows if EXCLUDE_FAILURES is False)."""
    if EXCLUDE_FAILURES and "failed" in df.columns:
        return df[~df.failed]
    return df


def failure_rate(df):
    return df.failed.mean() * 100 if "failed" in df.columns else 0.0


# --------------------------------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------------------------------

def restrict_to_common_instances(df):
    """Keep only (network, image) pairs for which every method has a result."""
    n_methods = df.verifier.nunique()
    counts = df.groupby(["network", "image_id"]).verifier.nunique()
    keep = counts[counts == n_methods].index
    return df.set_index(["network", "image_id"]).loc[keep].reset_index()


def virtual_best(df):
    """VBA per instance: method with the smallest p*; ties are broken by the shortest running time."""
    vba = df.sort_values(["smallest_sat_value", "total_time"]).groupby(["network", "image_id"]).head(1).copy()
    vba["verifier"] = "VBA"
    return vba


def relative_marginal_contribution(df):
    """Fraction of instances on which a method found a strictly smaller p* than all other methods."""
    best = df.groupby(["network", "image_id"]).smallest_sat_value.transform("min")
    winners = df[np.isclose(df.smallest_sat_value, best)]
    unique_winners = winners[winners.groupby(["network", "image_id"]).verifier.transform("size") == 1]
    n_instances = df.groupby(["network", "image_id"]).ngroups
    return unique_winners.verifier.value_counts() / n_instances


def ratio_to_vba(method_df, vba_df):
    """Avg(p*) of a method divided by Avg(p*) of the VBA, both over the instances on which the method succeeded.

    Using the same instances keeps the ratio >= 1; otherwise a method that fails on hard instances could appear
    better than the VBA.
    """
    method_df = successful(method_df)
    vba_values = vba_df.set_index(["network", "image_id"]).smallest_sat_value
    keys = method_df.set_index(["network", "image_id"]).index
    return method_df.smallest_sat_value.mean() / vba_values.loc[keys].mean()


def performance_table(df):
    """Time [s], Avg(p*), RMC and RatioAvg(p*) per method (Table 3)."""
    df = restrict_to_common_instances(df)
    vba = virtual_best(df)
    rmc = relative_marginal_contribution(df)
    vba_mean = successful(vba).smallest_sat_value.mean()
    rows = [["VBA", vba.total_time.mean(), vba_mean, np.nan, 1.0, failure_rate(vba)]]
    for verifier in [v for v in ALGORITHM_ORDER if v in df.verifier.unique()]:
        t_df = df[df.verifier == verifier]
        mean = successful(t_df).smallest_sat_value.mean()
        rows.append([verifier, t_df.total_time.mean(), mean, rmc.get(verifier, 0.0), ratio_to_vba(t_df, vba), failure_rate(t_df)])
    columns = ["algorithm", "Time [s]", "Avg(p*)", "RMC", "RatioAvg(p*)", "Failed [%]"]
    return pd.DataFrame(rows, columns=columns).set_index("algorithm")


def statistics_table(df):
    """Running time (mean/median/min/max) and p* (mean/min/max/std) per method, including the VBA (Tables C2-C7).

    The running times are very skewed (a few hard instances dominate the verifier's mean), so the median is reported
    instead of the standard deviation.
    """
    df = restrict_to_common_instances(df)
    df = pd.concat([virtual_best(df), df])
    order = ["VBA"] + [v for v in ALGORITHM_ORDER if v in df.verifier.unique()]
    time = df.groupby("verifier").total_time.agg(["mean", "median", "min", "max"])
    p_star = successful(df).groupby("verifier").smallest_sat_value.agg(["mean", "min", "max", "std"])
    failed = df.groupby("verifier").apply(failure_rate).rename("%")
    result = pd.concat({"Time": time, "p*": p_star, "Failed": failed.to_frame()}, axis=1)
    return result.loc[order]


def tight_instances_table(df, step):
    """Statistics on the instances where abcrown is as tight as possible, i.e. p* - eps* = one step (Table 4)."""
    abcrown = df[df.verifier == "ABcrown"]
    tight = abcrown[np.isclose(abcrown.smallest_sat_value - abcrown.epsilon_value, step)][["network", "image_id"]]
    t_df = df.merge(tight, on=["network", "image_id"])
    t_df = restrict_to_common_instances(t_df)
    optimal = t_df[t_df.verifier == "ABcrown"].set_index(["network", "image_id"]).smallest_sat_value
    vba = virtual_best(t_df)
    rows = []
    for verifier, v_df in [("VBA", vba)] + [(v, t_df[t_df.verifier == v]) for v in ALGORITHM_ORDER if v in t_df.verifier.unique()]:
        v_df = v_df.set_index(["network", "image_id"])
        f_opt = np.isclose(v_df.smallest_sat_value, optimal.loc[v_df.index]).mean()
        mean = successful(v_df).smallest_sat_value.mean()
        rows.append([verifier, v_df.total_time.mean(), mean, f_opt, ratio_to_vba(v_df.reset_index(), vba)])
    return pd.DataFrame(rows, columns=["algorithm", "time [s]", "Avg(p*)", "f_opt", "RatioAvg(p*)"]).set_index("algorithm"), len(tight)


NETWORK_ORDER = MNIST_ORDER + CIFAR_ORDER + IMAGENET_ORDER


def ordered_networks(networks, order=None):
    order = order or NETWORK_ORDER
    return [n for n in order if n in set(networks)] + [n for n in networks if n not in order]


def lognormal_ks_table(df, measure="smallest_sat_value", network_order=None):
    """K-S test of each distribution against a log-normal fitted on the same data."""
    df = successful(df)
    rows = []
    for network in ordered_networks(df.network.unique(), network_order):
        for verifier in [v for v in ALGORITHM_ORDER if v in df.verifier.unique()]:
            data = df[(df.network == network) & (df.verifier == verifier)][measure]
            if len(data) == 0:
                continue
            sigma, loc, scale = st.lognorm.fit(data, loc=0)
            statistic, p_value = st.kstest(data, st.lognorm.cdf, args=(sigma, loc, scale), alternative="two-sided")
            rows.append([network, verifier, statistic, p_value])
    result = pd.DataFrame(rows, columns=["network", "algorithm", "statistic", "p-value"])
    result["passed"] = result["p-value"] > 0.05
    return result


def subset_vs_full_ks_table(subset_df, full_df, measure="smallest_sat_value", exclude_subset=True, algorithms=None,
                            network_order=None):
    """Two-sample K-S test: distribution on the subset vs. on the complete test set (Tables D14, D17).

    With exclude_subset=True the subset images are removed from the test set, so both samples are disjoint
    as required by the two-sample K-S test (reviewer comment B.13).
    """
    subset_df, full_df = successful(subset_df), successful(full_df)
    if exclude_subset:
        subset_keys = subset_df.set_index(["network", "image_id"]).index
        full_df = full_df[~full_df.set_index(["network", "image_id"]).index.isin(subset_keys)]
    if algorithms is None:
        algorithms = [v for v in ALGORITHM_ORDER if v in full_df.verifier.unique()]
    rows = []
    for verifier in algorithms:
        for network in ordered_networks(full_df.network.unique(), network_order):
            data_1 = np.sort(subset_df[(subset_df.network == network) & (subset_df.verifier == verifier)][measure].values)
            data_2 = np.sort(full_df[(full_df.network == network) & (full_df.verifier == verifier)][measure].values)
            if len(data_1) == 0 or len(data_2) == 0:
                continue
            statistic, p_value = st.ks_2samp(data_1, data_2, alternative="two-sided")
            rows.append([verifier, network, np.mean(data_1), np.mean(data_2), statistic, p_value])
    result = pd.DataFrame(rows, columns=["algorithm", "network", "subset_mean", "full_mean", "statistic", "p-value"])
    result["passed"] = result["p-value"] > 0.05
    return result


def compute_pairwise_ks(df, verifier_names, measure_name="smallest_sat_value"):
    df = successful(df)
    statistic_df = pd.DataFrame(index=verifier_names, columns=verifier_names, dtype=float)
    mask_df = pd.DataFrame(index=verifier_names, columns=verifier_names, dtype=float)
    for a in verifier_names:
        for b in verifier_names:
            statistic, p_value = st.ks_2samp(df[df.verifier == a][measure_name].values,
                                             df[df.verifier == b][measure_name].values, alternative="two-sided")
            statistic_df.loc[a, b] = statistic
            mask_df.loc[a, b] = 1 if p_value >= 0.05 else 0
    return statistic_df, mask_df


def to_latex(df, float_format="%.3f", **kwargs):
    t_df = df.copy()
    escape = lambda x: x.replace("_", r"\_").replace("%", r"\%") if isinstance(x, str) else x
    if isinstance(t_df.index, pd.MultiIndex):
        t_df.index = pd.MultiIndex.from_tuples([tuple(escape(v) for v in x) for x in t_df.index], names=t_df.index.names)
    else:
        t_df.index = [escape(x) for x in t_df.index]
    for column in t_df.columns:
        if not pd.api.types.is_numeric_dtype(t_df[column]) and not pd.api.types.is_bool_dtype(t_df[column]):
            t_df[column] = t_df[column].map(escape)
    if isinstance(t_df.columns, pd.MultiIndex):
        t_df.columns = pd.MultiIndex.from_tuples([tuple(escape(v) for v in x) for x in t_df.columns])
    else:
        t_df.columns = [escape(x) for x in t_df.columns]
    return t_df.to_latex(float_format=float_format, **kwargs)


# Labels as in the paper (applied to the LaTeX output, after escaping)
PAPER_TABLE_LABELS = [
    ("RatioAvg(p*)", r"RatioAvg($\overline{\varepsilon}^*$)"),
    ("Avg(p*)", r"Avg($\overline{\varepsilon}^*$)"),
    ("{p*}", r"{$\overline{\varepsilon}^*$}"),
    ("f\_opt", r"$f_{opt}$"),
    ("ABcrown", r"$\alpha,\beta$-CROWN"),
]


def show_first_of_group(df, columns):
    """Show the value of each grouping column only in the first row of its group, as in the paper."""
    df = df.copy()
    for column in columns:
        repeated = df[column].eq(df[column].shift())
        df[column] = df[column].astype(object).where(~repeated, "")
    return df


def save_table(df, name, group=None, formatters=None, **kwargs):
    if group:
        df = show_first_of_group(df, group)
    if formatters:   # one format per column, applied here because to_latex ignores them next to float_format
        df = df.copy()
        for column, formatter in zip(df.columns, formatters):
            df[column] = df[column].map(formatter)
    latex = to_latex(df, **kwargs)
    for old, new in PAPER_TABLE_LABELS:
        latex = latex.replace(old, new)
    (TABLE_DIR / f"{name}.tex").write_text(latex)
    return latex


# --------------------------------------------------------------------------------------------------------------
# LaTeX in the format of the paper (main_springer.tex)
#
# The Springer template does not allow \input, so each file contains only the tabular block of one table. It replaces
# the old \begin{tabular}...\end{tabular} in the paper; caption and \label stay in the paper.
# --------------------------------------------------------------------------------------------------------------

PAPER_METHOD_NAMES = {"ABcrown": r"\abcrown", "PGD random": r"PGD\_random", "targeted FAB": r"targeted\_FAB",
                      "untargeted FAB": r"untargeted\_FAB"}
EPS_BAR = r"$\overline{\varepsilon}^*$"
AVG = r"$Avg(\overline{\varepsilon}^*)$"


def tex_method(name):
    return PAPER_METHOD_NAMES.get(name, name)


def tex_network(name):
    return r"\texttt{" + name.replace("_", r"\_") + "}"


def tex_number(value, decimals, thousands=False):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    return f"{value:,.{decimals}f}" if thousands else f"{value:.{decimals}f}"


def write_paper_table(name, label, lines):
    text = f"% Replaces the tabular of \\label{{{label}}} in main_springer.tex\n" + "\n".join(lines) + "\n"
    (TABLE_DIR / f"{name}.tex").write_text(text)
    return text


def tex_performance_table(blocks):
    """Table 3: blocks = [(title, subset label, subset table, full table)] from performance_table()."""
    columns = [r"\makecell{Time\\{[}s{]}}", AVG, "RMC", r"\makecell{Ratio\\" + AVG + "}", r"\makecell{Failed\\{[}\%{]}}"]
    header = "algorithm & " + " & ".join(columns + columns) + r" \\"
    # The Failed columns make the table wider than before; the smaller spacing keeps it at the width of the submitted one
    lines = [r"\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{lrrrrrrrrrr}"]
    for i, (title, subset_label, subset, full) in enumerate(blocks):
        if i > 0:
            lines += [r"\bottomrule", r"\\"]
        lines += [r"\toprule", rf" & \multicolumn{{10}}{{c}}{{{title}}} \\",
                  rf" & \multicolumn{{5}}{{c}}{{{subset_label}}} & \multicolumn{{5}}{{c}}{{Complete test set}} \\",
                  r" \cmidrule(lr){2-6}\cmidrule(lr){7-11}", header, r"\midrule"]
        for algorithm in subset.index:
            cells = []
            for table in (subset, full):
                row = table.loc[algorithm] if algorithm in table.index else None
                if row is None:
                    cells += ["-"] * 5
                    continue
                cells += [tex_number(row["Time [s]"], 3), tex_number(row["Avg(p*)"], 3),
                          "-" if algorithm == "VBA" else tex_number(row["RMC"], 3),
                          tex_number(row["RatioAvg(p*)"], 3), tex_number(row["Failed [%]"], 1)]
            lines.append(f"{tex_method(algorithm)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return lines


def tex_tight_table(blocks):
    """Table 4: blocks = [(title, table)] from tight_instances_table()."""
    lines = [r"\begin{tabularx}{.9\textwidth}{Xrrrr}"]
    for i, (title, table) in enumerate(blocks):
        if i > 0:
            lines += [r"\bottomrule", r"\\"]
        lines += [r"\toprule", rf"\multicolumn{{5}}{{c}}{{{title}}} \\",
                  rf"algorithm & time [s] & {AVG} & $f_{{\text{{opt}}}}$ & Ratio {AVG} \\", r"\midrule"]
        for algorithm, row in table.iterrows():
            values = [row["time [s]"], row["Avg(p*)"], row["f_opt"], row["RatioAvg(p*)"]]
            lines.append(f"{tex_method(algorithm)} & " + " & ".join(tex_number(v, 3) for v in values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabularx}"]
    return lines


def tex_statistics_table(table, time_decimals, value_decimals, thousands=False):
    """Tables C2-C7 from statistics_table()."""
    lines = [r"\small", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
             rf" & \multicolumn{{4}}{{c}}{{Time}} & \multicolumn{{4}}{{c}}{{{EPS_BAR}}} & Failed \\",
             r"  \cmidrule(lr){2-5}\cmidrule(lr){6-9}",
             r"algorithm & mean & median & min & max & mean & min & max & std & [\%] \\", r"\midrule"]
    for algorithm, row in table.iterrows():
        cells = [tex_number(row[("Time", c)], time_decimals, thousands) for c in ["mean", "median", "min", "max"]]
        cells += [tex_number(row[("p*", c)], value_decimals) for c in ["mean", "min", "max", "std"]]
        cells.append(tex_number(row[("Failed", "%")], 1))
        lines.append(f"{tex_method(algorithm)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return lines


def tex_abcrown_lognormal_table(upper, lower):
    """Table D8: K-S tests of the alpha,beta-CROWN upper (p*) and lower (eps*) bounds per network."""
    lines = [r"\begin{tabular}{lllrrr}", r"\toprule",
             r"network & algorithm & measure & statistic & p-value & passed \\", r"\midrule"]
    networks = ordered_networks(upper.network.unique())
    for i, network in enumerate(networks):
        for j, (measure, table) in enumerate([(EPS_BAR, upper), (r"$\underline{\varepsilon}^*$", lower)]):
            row = table[table.network == network].iloc[0]
            first = (tex_network(network) + " & abcrown") if j == 0 else " & "
            end = r" \\ \hline" if j == 1 and i < len(networks) - 1 else r" \\"
            lines.append(f"{first} & {measure} & {tex_number(row.statistic, 2)} & {tex_number(row['p-value'], 2)} & "
                         f"{row.passed}{end}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return lines


def tex_grouped_test_table(table, group, columns, headers):
    """Tables D9-D17: `group` (network or algorithm) is shown only on the first row of its block."""
    spec = "ll" + "r" * len(columns)
    lines = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    other = "algorithm" if group == "network" else "network"
    for value, block in table.groupby(group, sort=False):
        for j, (_, row) in enumerate(block.iterrows()):
            first = (tex_network(value) if group == "network" else tex_method(value)) if j == 0 else ""
            second = tex_method(row[other]) if other == "algorithm" else tex_network(row[other])
            cells = [str(row[c]) if c == "passed" else tex_number(row[c], 2) for c in columns]
            lines.append(f"{first} & {second} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return lines


# --------------------------------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------------------------------

def plot_cdfs(df, hue, ax, title, xlim, hue_order=None, palette=None, xtick=None, labels=None):
    """CDF per hue level; `xtick` sets the tick spacing, `labels` renames hue levels in the legend."""
    df = df.rename(columns={"verifier": "algorithm"})
    if hue == "algorithm" and palette is None and hue_order is not None:
        colors = CIFAR_IMAGENET_COLORS if set(hue_order) <= set(CIFAR_ALGORITHMS) else MNIST_COLORS
        palette = {name: colors[name] for name in hue_order}
    if labels:
        df = df.assign(**{hue: df[hue].replace(labels)})
        if hue_order is not None:
            hue_order = [labels.get(name, name) for name in hue_order]
        if isinstance(palette, dict):
            palette = {labels.get(name, name): color for name, color in palette.items()}
    sns.ecdfplot(df, x="smallest_sat_value", hue=hue, ax=ax, hue_order=hue_order, palette=palette)
    ax.set_xlim(*xlim)
    if xtick:
        ax.xaxis.set_major_locator(plt.MultipleLocator(xtick))
    ax.set_title(title)
    ax.set_xlabel("Epsilon value")
    ax.set_ylabel("Fraction $p*$ values found")


def plot_verification_gap(df, ax, color=None):
    """Lighter area between the eps* and p* CDFs of the complete verifier."""
    df = df.sort_values(by="smallest_sat_value")
    cdf_x = np.linspace(0, 1, len(df))
    # zorder 1.8: above the grid (0.5 or 1.5, depending on axes.axisbelow) and below the CDF lines (2)
    ax.fill_betweenx(cdf_x, df.smallest_sat_value, np.sort(df.epsilon_value.values), color=color, zorder=1.8)


def scatter_with_spearman(df, verifier_a, verifier_b, ax, lim, xtick=None):
    """Per-instance comparison of two methods, overlapping points are shown via the count (Figure 5)."""
    a_df = df[df.verifier == verifier_a][["network", "image_id", "smallest_sat_value"]].rename(columns={"smallest_sat_value": verifier_a})
    b_df = df[df.verifier == verifier_b][["network", "image_id", "smallest_sat_value"]].rename(columns={"smallest_sat_value": verifier_b})
    merged = pd.merge(a_df, b_df, on=["network", "image_id"])
    counts = merged.groupby([verifier_a, verifier_b]).size().reset_index(name="count")
    sns.scatterplot(counts, x=verifier_a, y=verifier_b, size="count", ax=ax, color=sns.color_palette("deep")[0])
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    if xtick:
        ax.xaxis.set_major_locator(plt.MultipleLocator(xtick))
    ax.set_axisbelow(True)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color="red", linestyle="dashed", linewidth=1, zorder=-2)
    res = st.spearmanr(merged[verifier_a], merged[verifier_b])
    ax.text(.4, .1, "Spearman's r = {:.2f},".format(res.statistic) + " p-value: {:.2f}".format(res.pvalue), transform=ax.transAxes)
    return res


def to_raw_names(matrix_df):
    """Rename and reorder a method x method matrix to the raw verifier names used in Figures D1/D2."""
    matrix_df = matrix_df.rename(index=RAW_NAMES, columns=RAW_NAMES)
    order = [name for name in RAW_ORDER if name in matrix_df.index]
    return matrix_df.loc[order, order]


def plot_pairwise_ks(df, verifier_names, ax, title):
    statistic_df, mask_df = compute_pairwise_ks(df, verifier_names)
    statistic_df, mask_df = to_raw_names(statistic_df), to_raw_names(mask_df)
    cmap = LinearSegmentedColormap.from_list("", [(0, sns.color_palette("Paired", 8)[0]),
                                                  (.05, sns.color_palette("Paired", 8)[6]),
                                                  (1, sns.color_palette("Paired", 8)[6])])
    sns.heatmap(mask_df, annot=statistic_df, fmt=".2f", cmap=cmap, vmin=0.0, vmax=1.0, cbar=False, ax=ax)
    ax.set_title(title)


def remove_legends_except(axes, keep):
    for ax in np.ravel(axes):
        if ax is not keep and ax.get_legend() is not None:
            ax.get_legend().remove()
