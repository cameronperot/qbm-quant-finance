import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qbm.utils import get_project_dir, load_artifact, save_artifact
from qbm.plotting import plot_autocorrelation_grid, plot_tail_concentrations_grid, plot_qq

# pd.set_option("display.max_columns", None)

project_dir = get_project_dir()
results_dir = project_dir / "results/data/rbm"
plots_dir = project_dir / "results/plots/rbm"
tables_dir = project_dir / "latex/tables/rbm"
if not tables_dir.exists():
    tables_dir.mkdir(parents=True)
if not plots_dir.exists():
    plots_dir.mkdir(parents=True)
artifacts_dir = project_dir / "artifacts"

models = load_artifact(project_dir / "scripts/rbm/models.json")


def str_map(x, digits=2, factor=1):
    return f"{x * factor:0.{digits}f}"


def save_table(table, file_name):
    with open(tables_dir / file_name, "w", encoding="utf-8") as f:
        f.write(table + "\n")


def textbf(x):
    return "\\textbf{%s}" % x


log_returns = pd.read_csv(
    artifacts_dir / f"{models['baseline']['id']}/log_returns.csv",
    index_col="date",
    parse_dates=["date"],
)

qq_rmses = {}
ccs = {}
cc_rmses = {}
volatilities = {}
volatilities_low = {}
volatilities_high = {}
tails = {}
ac_times = {}
# combine the data into merged dataframes
for model_name, model_info in models.items():
    prefix = model_info["prefix"]
    model_id = model_info["id"]
    model_results_dir = artifacts_dir / f"{model_id}/results"

    config = (load_artifact(artifacts_dir / f"{model_id}/config.json"),)
    save_artifact(config, results_dir / f"configs/{model_name}.json")

    # QQ RMSE
    qq_rmse = pd.read_csv(model_results_dir / "data/qq_rmse.csv", index_col="currency_pair")
    column_map = {column: f"{prefix}_{column}" for column in qq_rmse.columns}
    qq_rmse.loc["Mean"] = qq_rmse.mean()
    qq_rmse.loc["Mean", "std"] = np.sqrt(np.sum(qq_rmse["std"] ** 2))
    qq_rmse.rename(columns=column_map, inplace=True)
    qq_rmses[prefix] = qq_rmse

    # correlation coefficients data
    cc_data = pd.read_csv(
        model_results_dir / "data/correlation_coefficients_data.csv",
        index_col="currency_pairs",
    )
    column_map = {column: f"{prefix}_{column}_data" for column in cc_data.columns}
    cc_data.rename(columns=column_map, inplace=True)

    # correlation coefficients sample means
    cc_sample_means = pd.read_csv(
        model_results_dir / "data/correlation_coefficients_sample_means.csv",
        index_col="currency_pairs",
    )
    column_map = {column: f"{prefix}_{column}_mean" for column in cc_sample_means.columns}
    cc_sample_means.rename(columns=column_map, inplace=True)

    # correlation coefficients sample stds
    cc_sample_stds = pd.read_csv(
        model_results_dir / "data/correlation_coefficients_sample_stds.csv",
        index_col="currency_pairs",
    )
    column_map = {column: f"{prefix}_{column}_std" for column in cc_sample_stds.columns}
    cc_sample_stds.rename(columns=column_map, inplace=True)

    # correlation coefficients combine
    cc_df = pd.concat([cc_data, cc_sample_means, cc_sample_stds], axis=1)
    ccs[prefix] = cc_df

    # volatility
    volatility = pd.read_csv(
        model_results_dir / "data/volatilities.csv", index_col="currency_pair"
    )
    column_map = {
        column: f"{prefix}_{column.lower().replace(' ', '_')}"
        for column in volatility.columns
    }
    volatility.rename(columns=column_map, inplace=True)
    volatilities[prefix] = volatility

    # conditional volatilities
    if "V" in prefix:
        volatility_low = pd.read_csv(
            model_results_dir / "data/volatilities_low.csv", index_col="currency_pair",
        )
        column_map = {
            column: f"{prefix}_{column.lower().replace(' ', '_')}"
            for column in volatility_low.columns
        }
        volatility_low.rename(columns=column_map, inplace=True)
        volatilities_low[prefix] = volatility_low

        volatility_high = pd.read_csv(
            model_results_dir / "data/volatilities_high.csv", index_col="currency_pair",
        )
        column_map = {
            column: f"{prefix}_{column.lower().replace(' ', '_')}"
            for column in volatility_high.columns
        }
        volatility_high.rename(columns=column_map, inplace=True)
        volatilities_high[prefix] = volatility_high

    # tails data
    tails_data = pd.read_csv(
        model_results_dir / "data/tails_data.csv", index_col="currency_pair"
    )
    column_map = {column: f"{prefix}_{column}_data" for column in tails_data.columns}
    tails_data.rename(columns=column_map, inplace=True)

    # tails sample means
    tails_sample_means = pd.read_csv(
        model_results_dir / "data/tails_sample_means.csv", index_col="currency_pair"
    )
    column_map = {
        column: f"{prefix}_{column}_mean" for column in tails_sample_means.columns
    }
    tails_sample_means.rename(columns=column_map, inplace=True)

    # tails sample stds
    tails_sample_stds = pd.read_csv(
        model_results_dir / "data/tails_sample_stds.csv", index_col="currency_pair",
    )
    column_map = {column: f"{prefix}_{column}_std" for column in tails_sample_stds.columns}
    tails_sample_stds.rename(columns=column_map, inplace=True)

    # tails combine
    tails_df = pd.concat([tails_data, tails_sample_means, tails_sample_stds], axis=1)
    tails[prefix] = tails_df

    # autocorrelation times
    ac_times[prefix] = pd.read_csv(
        model_results_dir / "data/autocorrelation_times.csv", index_col="currency_pair",
    )
    column_map = {column: f"{prefix}_ac_time" for column in ac_times[prefix].columns}
    ac_times[prefix].rename(columns=column_map, inplace=True)

# combined QQ plot
qq_plot_params = {
    "title": "test",
    "xlims": (-0.045, 0.045),
    "ylims": (-0.045, 0.045),
    "xticks": np.linspace(-0.04, 0.04, 9),
    "yticks": np.linspace(-0.04, 0.04, 9),
}
fig, axs = plt.subplots(4, 4, figsize=(12, 12), dpi=300)
for j, (model_name, model_info) in enumerate(models.items()):
    model_id = model_info["id"]
    prefix = model_info["prefix"]
    qq_extrema = load_artifact(artifacts_dir / f"{model_id}/results/data/qq_extrema.json")
    samples = load_artifact(
        artifacts_dir / f"{model_id}/samples_ensemble/{qq_extrema['min'] + 1:03}.pkl"
    )
    for i, column in enumerate(log_returns.columns):
        if i == 0:
            title = f"RBM ({prefix})\n{column}"
        else:
            title = column
        plot_qq(axs[i, j], log_returns[column], samples[column], title, qq_plot_params)
        axs[i, j].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        axs[i, j].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

plt.tight_layout()
plt.savefig(plots_dir / "qq.png")

"""
Plotting
"""
colors = {
    "Data": "tab:cyan",
    "RBM (B)": "tab:blue",
    "RBM (V)": "tab:red",
    "RBM (X)": "tab:orange",
    "RBM (XV)": "tab:green",
}

# plot autocorrelations
acfs = {}
for model_name, model_info in models.items():
    model_id = model_info["id"]
    acfs[f"RBM ({model_info['prefix']})"] = pd.read_csv(
        project_dir / f"artifacts/{model_id}/results/data/autocorrelation_functions.csv",
        index_col="lag",
    )
fig, axs = plot_autocorrelation_grid(acfs, colors)
plt.savefig(plots_dir / "autocorrelation_functions.png")
plt.close(fig)

# plot tail concentrations
combinations = list(itertools.combinations(log_returns.columns, 2))
tail_concentration_dfs = {}
for model_name, model_info in models.items():
    if model_name == "baseline":
        tail_concentration_dfs["Data"] = pd.read_csv(
            project_dir / f"artifacts/{model_id}/log_returns.csv",
            index_col="date",
            parse_dates=["date"],
        )

    model_id = model_info["id"]
    qq_extrema = load_artifact(
        project_dir / f"artifacts/{model_id}/results/data/qq_extrema.json"
    )
    tail_concentration_dfs[f"RBM ({model_info['prefix']})"] = pd.read_pickle(
        project_dir / f"artifacts/{model_id}/samples_ensemble/{qq_extrema['min']+1:03}.pkl"
    )

fig, axs = plot_tail_concentrations_grid(tail_concentration_dfs, combinations, colors)
plt.savefig(plots_dir / "tail_concentrations.png")
plt.close(fig)

"""
LaTeX Tables
"""
# QQ RMSEs table
qq_rmses = pd.concat(qq_rmses.values(), axis=1).applymap(str_map, digits=3, factor=100)
print("QQ RMSEs")
print(qq_rmses)

prefixes = ("B", "V", "X", "XV")
table = [
    r"\begin{tabular}{l r r r r}",
    r"\multicolumn{5}{c}{\textbf{QQ RMSEs}} \\",
    r"\toprule",
    r"Currency Pair & \textbf{RBM (%s)} & \textbf{RBM (%s)} & \textbf{RBM (%s)} & \textbf{RBM (%s)} \\"
    % prefixes,
    r"\midrule",
]
for i, pair in enumerate(qq_rmses.index):
    data = qq_rmses.loc[pair]
    row = [pair]

    for prefix in prefixes:
        μ = data[f"{prefix}_mean"]
        σ = data[f"{prefix}_std"]
        row.append(fr"{μ}\% $\pm$ {σ}\%")

    row = " & ".join(row)
    row += r" \\"
    if i == len(qq_rmses) - 1:
        table.append(r"\midrule")

    table.append(row)

table.append(r"\bottomrule")
table.append(r"\end{tabular}")
table = "\n".join(table)
save_table(table, "qq_rmses.tbl")

# correlation coefficients table
ccs = pd.concat(ccs.values(), axis=1).applymap(str_map, digits=2)
print("Correlation Coefficients")
print(ccs)

prefixes = ("B",)
cc_names = ("Pearson", "Spearman", "Kendall")
table = [
    r"\begin{tabular}{l r r r r r r}"
    r"\multicolumn{7}{c}{\textbf{Correlation Coefficients}} \\",
    r"\toprule",
    r"& \multicolumn{3}{c}{\textbf{Data}} & \multicolumn{3}{c}{\textbf{RBM (%s)}} \\"
    % prefixes,
    r"\cmidrule(lr){2-4}",
    r"\cmidrule(lr){5-7}",
    r"Currency Pairs & %s & %s & %s & %s & %s & %s \\" % (cc_names + cc_names),
    r"\midrule",
]
data_columns = [f"B_{cc_name}_data" for cc_name in cc_names]
for pair in ccs.index:
    data = ccs.loc[pair]
    row = [pair]
    row += [data[column] for column in data_columns]

    for prefix in prefixes:
        for cc_name in cc_names:
            μ = data[f"{prefix}_{cc_name}_mean"]
            σ = data[f"{prefix}_{cc_name}_std"]
            row.append(fr"{μ} $\pm$ {σ}")

    row = " & ".join(row)
    row += r" \\"
    table.append(row)

prefixes = ("V", "X")
cc_names = ("Pearson", "Spearman", "Kendall")
table += [
    r"\midrule",
    r"& \multicolumn{3}{c}{\textbf{RBM (%s)}} & \multicolumn{3}{c}{\textbf{RBM (%s)}} \\"
    % prefixes,
    r"\cmidrule(lr){2-4}",
    r"\cmidrule(lr){5-7}",
    r"Currency Pairs & %s & %s & %s & %s & %s & %s \\" % (cc_names + cc_names),
    r"\midrule",
]
for pair in ccs.index:
    data = ccs.loc[pair]
    row = [pair]

    for prefix in prefixes:
        for cc_name in cc_names:
            μ = data[f"{prefix}_{cc_name}_mean"]
            σ = data[f"{prefix}_{cc_name}_std"]
            row.append(fr"{μ} $\pm$ {σ}")

    row = " & ".join(row)
    row += r" \\"
    table.append(row)

prefixes = ("XV",)
cc_names = ("Pearson", "Spearman", "Kendall")
table += [
    r"\midrule",
    r"& \multicolumn{3}{c}{\textbf{RBM (%s)}} & \\" % prefixes,
    r"\cmidrule(lr){2-4}",
    r"Currency Pairs & %s & %s & %s & & & \\" % cc_names,
    r"\midrule",
]
for pair in ccs.index:
    data = ccs.loc[pair]
    row = [pair]

    for prefix in prefixes:
        for cc_name in cc_names:
            μ = data[f"{prefix}_{cc_name}_mean"]
            σ = data[f"{prefix}_{cc_name}_std"]
            row.append(fr"{μ} $\pm$ {σ}")

    for cc_name in cc_names:
        row.append("")

    row = " & ".join(row)
    row += r" \\"
    table.append(row)

table.append(r"\bottomrule")
table.append(r"\end{tabular}")
table = "\n".join(table)
save_table(table, "correlation_coefficients.tbl")

# volatility table
volatilities = pd.concat(volatilities.values(), axis=1).applymap(
    str_map, digits=2, factor=100
)
print("Volatilities")
print(volatilities)

prefixes = ("B", "V", "X", "XV")
table = [
    r"\begin{tabular}{l r r r r r}",
    r"\multicolumn{6}{c}{\textbf{Historical Volatilities}} \\",
    r"\toprule",
    r"Currency Pair & \textbf{Data} & \textbf{RBM (%s)} & \textbf{RBM (%s)} & \textbf{RBM (%s)} & \textbf{RBM (%s)} \\"
    % prefixes,
    r"\midrule",
]
for pair in volatilities.index:
    data = volatilities.loc[pair]
    row = [pair, fr"{data['B_data']}\%"]

    for prefix in prefixes:
        μ = data[f"{prefix}_sample_mean"]
        σ = data[f"{prefix}_sample_std"]
        row.append(fr"{μ}\% $\pm$ {σ}\%")

    row = " & ".join(row)
    row += r" \\"
    table.append(row)

table.append(r"\bottomrule")
table.append(r"\end{tabular}")
table = "\n".join(table)
save_table(table, "volatilities.tbl")

# low conditional volatilities table
volatilities_low = pd.concat(volatilities_low.values(), axis=1).applymap(
    str_map, digits=2, factor=100
)
volatilities_high = pd.concat(volatilities_high.values(), axis=1).applymap(
    str_map, digits=2, factor=100
)
print("Volatilities (Low)")
print(volatilities_low)
print("Volatilities High")
print(volatilities_high)

prefixes = ("V", "XV")
table = [
    r"\begin{tabular}{l r r r r r r}",
    r"\multicolumn{7}{c}{\textbf{Conditional Volatilities}} \\",
    r"\toprule",
    r"& \multicolumn{3}{c}{\textbf{Low Regime}} & \multicolumn{3}{c}{\textbf{High Regime}} \\",
    r"\cmidrule(lr){2-4}",
    r"\cmidrule(lr){5-7}",
    r"Currency Pair & \textbf{Data} & \textbf{RBM (%s)} & \textbf{RBM (%s)} & \textbf{Data} & \textbf{RBM (%s)} & \textbf{RBM (%s)} \\"
    % (prefixes + prefixes),
    r"\midrule",
]
for pair in volatilities_low.index:
    data_low = volatilities_low.loc[pair]
    data_high = volatilities_high.loc[pair]
    row = [pair, fr"{data_low['V_data']}\%"]

    for prefix in prefixes:
        μ = data_low[f"{prefix}_sample_mean"]
        σ = data_low[f"{prefix}_sample_std"]
        row.append(fr"{μ}\% $\pm$ {σ}\%")

    row.append(fr"{data_high['V_data']}\%")

    for prefix in prefixes:
        μ = data_high[f"{prefix}_sample_mean"]
        σ = data_high[f"{prefix}_sample_std"]
        row.append(fr"{μ}\% $\pm$ {σ}\%")

    row = " & ".join(row)
    row += r" \\"
    table.append(row)

table.append(r"\bottomrule")
table.append(r"\end{tabular}")
table = "\n".join(table)
save_table(table, "conditional_volatilities.tbl")

# tails table
tails = pd.concat(tails.values(), axis=1).applymap(str_map, digits=2, factor=100)
print("Tails")
print(tails)

prefixes = ("B", "V", "X", "XV")
table = [
    r"\begin{tabular}{l r r r r r}",
    r"\multicolumn{6}{c}{\textbf{Lower Tails (1st Percentile)}} \\",
    r"\toprule",
    r"Currency Pair & \textbf{Data} & \textbf{RBM (%s)} & \textbf{RBM (%s)} & \textbf{RBM (%s)} & \textbf{RBM (%s)} \\"
    % prefixes,
    r"\midrule",
]

columns_low = [column for column in tails.columns if "quantile_01" in column]
for pair in tails.index:
    data = tails.loc[pair]
    row = [pair, fr"{data['B_quantile_01_data']}\%"]

    for prefix in prefixes:
        μ = data[f"{prefix}_quantile_01_mean"]
        σ = data[f"{prefix}_quantile_01_std"]
        row.append(fr"{μ}\% $\pm$ {σ}\%")

    row = " & ".join(row)
    row += r" \\"
    table.append(row)

table += [
    r"\bottomrule \\",
    r"\multicolumn{6}{c}{\textbf{Upper Tails (99th Percentile)}} \\",
    r"\toprule",
    r"Currency Pair & \textbf{Data} & \textbf{RBM (%s)} & \textbf{RBM (%s)} & \textbf{RBM (%s)} & \textbf{RBM (%s)} \\"
    % prefixes,
    r"\midrule",
]

columns_high = [column for column in tails.columns if "quantile_99" in column]
for pair in tails.index:
    data = tails.loc[pair]
    row = [pair, fr"{data['B_quantile_99_data']}\%"]

    for prefix in prefixes:
        μ = data[f"{prefix}_quantile_99_mean"]
        σ = data[f"{prefix}_quantile_99_std"]
        row.append(fr"{μ}\% $\pm$ {σ}\%")

    row = " & ".join(row)
    row += r" \\"
    table.append(row)

table.append(r"\bottomrule")
table.append(r"\end{tabular}")
table = "\n".join(table)
save_table(table, "tails.tbl")

# autocorrelation times table
ac_times = pd.concat(ac_times.values(), axis=1).applymap(str_map, digits=1)
print("Autocorrelation Times")
print(ac_times)

prefixes = ("B", "V", "X", "XV")
table = [
    r"\begin{tabular}{l r r r r}",
    r"\multicolumn{5}{c}{\textbf{Integrated Autocorrelation Times}} \\",
    r"\toprule",
    r"Currency Pair & \textbf{RBM (%s)} & \textbf{RBM (%s)} & \textbf{RBM (%s)} & \textbf{RBM (%s)} \\"
    % prefixes,
    r"\midrule",
]
for pair in ac_times.index:
    data = ac_times.loc[pair]
    row = [pair]

    for prefix in prefixes:
        τ = data[f"{prefix}_ac_time"]
        row.append(fr"{τ}")

    row = " & ".join(row)
    row += r" \\"
    table.append(row)

table.append(r"\bottomrule")
table.append(r"\end{tabular}")
table = "\n".join(table)
save_table(table, "autocorrelation_times.tbl")
