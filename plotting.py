import seaborn as sns
import pandas as pd
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt

def visualize_logit_tree(logits_per_step, tokenizer, chosen_tokens, top_k=5):
    """
    Visualize candidate token branches as a tree.

    Args:
        logits_per_step: list of dicts
            Each dict maps token_id -> log_likelihood at a decoding step.
        tokenizer_name: str
            HuggingFace tokenizer to use for decoding token IDs.
        top_k: int
            Maximum number of candidates per step to visualize.

    Returns:
        None (plots tree with matplotlib)
    """
    G = nx.DiGraph()
    root = "ROOT"
    G.add_node(root, label="START")
    chosen_this_step = "ROOT"
    last_chosen = []
    node_colours = ["lightblue"]
    output_tokens = chosen_tokens[-len(logits_per_step):]
    #print(f"num nodes {len(G.nodes())} num colours = {len(node_colours)}")
    # Build the graph
    for step, cand_dict in enumerate(logits_per_step):
        #print(f"step {step} num nodes {len(G.nodes())} num colours = {len(node_colours)}")
        last_chosen.append(chosen_this_step)
        candidates = sorted(cand_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
        # Convert to probabilities assuming all mass is within top_k
        logliks = [ll for _, ll in candidates]
        max_ll = max(logliks)
        exp_shifted = [math.exp(ll - max_ll) for ll in logliks]
        total = sum(exp_shifted)
        probs = [v / total for v in exp_shifted]
        for (token_id, loglik), prob in zip(candidates, probs):
            if prob < 0.01:
              continue
            else:
              token_text = tokenizer.decode([token_id])
              node_name = f"{step}-{token_id}"
              label = f"{token_text}\n{prob:.2f}"

              if token_id == output_tokens[step]:
                G.add_node(node_name, label=label, step=step)
                node_colours.append("orange")
                chosen_this_step = node_name
              else:
                G.add_node(node_name, label=label, step=step)
                node_colours.append("lightblue")

              if step == 0:
                  G.add_edge(root, node_name)
              else:
                  G.add_edge(last_chosen[-1], node_name)


    # Layout: manually place nodes by step
    pos = {}
    step_groups = {}
    for node, data in G.nodes(data=True):
        step = data.get("step", -1)
        step_groups.setdefault(step, []).append(node)

    for step, nodes in step_groups.items():
        # spread nodes vertically
        for i, node in enumerate(nodes):
            pos[node] = (step, -i)
    pos[root] = (-1, 0)

    # Draw
    plt.figure(figsize=(18, 4))
    nx.draw(G, pos, with_labels=False, node_size=2000, node_color=node_colours, arrows=True)
    node_labels = nx.get_node_attributes(G, "label")
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=7)

    plt.title("Logit Branching Tree")
    plt.axis("off")
    plt.show()

def calibration_plots3(df, correct_col, columns, bins= 10, fixed_lim = True, plot_title = " ", header_override = ['Max Probability', 'Other']):
    conf_column = columns[0]
    i = bins+1
    if fixed_lim:
        bmin = 0
        bmax = 1
    else:
        bmin = df[conf_column].min()
        bmax = df[conf_column].max()
    bins = np.linspace(bmin -.001, bmax +.001, i)
    df['pred_bin'] = pd.cut(df[conf_column], bins, labels=range(0,i-1, 1))
    df_calib = df[[correct_col, conf_column, 'pred_bin']].groupby('pred_bin').agg({correct_col: ['mean', 'count'], conf_column: ['mean']})
    df_calib.columns = ["_".join(a) for a in df_calib.columns.to_flat_index()]
    df_calib.reset_index(inplace=True)
    df_calib['proportion'] = df_calib[correct_col + '_count'] / df_calib[correct_col + '_count'].sum()
    df_calib["pred_bin"] = df_calib["pred_bin"].astype(int)
    df_calib.sort_values("pred_bin", inplace=True, ascending=True)
    df_calib.reset_index(drop=True, inplace=True)
    df_calib.fillna(value = 0, inplace = True)
    f, axes = plt.subplots(2,2, figsize=(10,10))
    ax = axes[0,0]
    sns.barplot(data=df_calib.reset_index(drop=True), x='pred_bin', y= 'proportion', ax=ax)
    ax.set_ylabel("Samples (%)")
    ax.set_xlabel("Confidence")
    ax.set_title(header_override[0])
    ax.set_xticks(ticks =  range(len(df_calib[conf_column+'_mean'])),labels = [f"{100*b:2.0f}%" for b in bins[:-1]], rotation=90)
    ax.set_ylim((0,.5))
    ax= axes[1,0]
    sns.barplot(data=df_calib.reset_index(drop=True), x='pred_bin', y=correct_col + '_mean', ax=ax, hue=2, legend=None)
    ax.plot([0,i-2],[0, 1], c='teal',linestyle='dashed' )
    plt.title(plot_title)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(ticks =  range(len(df_calib[conf_column+'_mean'])),labels = [f"{100*b:2.0f}%" for b in bins[:-1]], rotation=90)

    conf_column = columns[1]
    df['pred_bin'] = pd.cut(df[conf_column], bins, labels=range(0,i-1, 1))
    df_calib = df[[correct_col, conf_column, 'pred_bin']].groupby('pred_bin').agg({correct_col: ['mean', 'count'], conf_column: ['mean']})
    df_calib.columns = ["_".join(a) for a in df_calib.columns.to_flat_index()]
    df_calib.reset_index(inplace=True)
    df_calib['proportion'] = df_calib[correct_col + '_count'] / df_calib[correct_col + '_count'].sum()
    df_calib["pred_bin"] = df_calib["pred_bin"].astype(int)
    df_calib.sort_values("pred_bin", inplace=True, ascending=True)
    df_calib.reset_index(drop=True, inplace=True)
    df_calib.fillna(value = 0, inplace = True)
    ax = axes[0,1]
    sns.barplot(data=df_calib.reset_index(drop=True), x='pred_bin', y= 'proportion', ax=ax)
    ax.set_ylabel(None)
    ax.set_title(header_override[1])
    ax.set_xlabel("Confidence")
    ax.set_xticks(ticks =  range(len(df_calib[conf_column+'_mean'])),labels = [f"{100*b:2.0f}%" for b in bins[:-1]], rotation=90)
    ax.set_ylim((0,.5))
    ax= axes[1,1]
    sns.barplot(data=df_calib.reset_index(drop=True), x='pred_bin', y=correct_col + '_mean', ax=ax, hue=2, legend=None)
    ax.plot([0,i-2],[0, 1], c='teal',linestyle='dashed' )
    plt.title(plot_title)
    ax.set_xlabel("Confidence")
    ax.set_ylabel(None)
    ax.set_xticks(ticks =  range(len(df_calib[conf_column+'_mean'])),labels = [f"{100*b:2.0f}%" for b in bins[:-1]], rotation=90)


def calibration_plot(df, correct_col, conf_column, bins= 10, fixed_lim = True, plot_title = " ", header_override = ['Max Probability', 'Other']):
    i = bins+1
    if fixed_lim:
        bmin = 0
        bmax = 1
    else:
        bmin = df[conf_column].min()
        bmax = df[conf_column].max()
    bins = np.linspace(bmin -.001, bmax +.001, i)
    df['pred_bin'] = pd.cut(df[conf_column], bins, labels=range(0,i-1, 1))
    df_calib = df[[correct_col, conf_column, 'pred_bin']].groupby('pred_bin').agg({correct_col: ['mean', 'count'], conf_column: ['mean']})
    df_calib.columns = ["_".join(a) for a in df_calib.columns.to_flat_index()]
    df_calib.reset_index(inplace=True)
    df_calib['proportion'] = df_calib[correct_col + '_count'] / df_calib[correct_col + '_count'].sum()
    df_calib["pred_bin"] = df_calib["pred_bin"].astype(int)
    df_calib.sort_values("pred_bin", inplace=True, ascending=True)
    df_calib.reset_index(drop=True, inplace=True)
    df_calib.fillna(value = 0, inplace = True)
    f, axes = plt.subplots(2,1, figsize=(10,5))
    ax = axes[0,0]
    sns.barplot(data=df_calib.reset_index(drop=True), x='pred_bin', y= 'proportion', ax=ax)
    ax.set_ylabel("Samples (%)")
    ax.set_xlabel("Confidence")
    ax.set_title(header_override[0])
    ax.set_xticks(ticks =  range(len(df_calib[conf_column+'_mean'])),labels = [f"{100*b:2.0f}%" for b in bins[:-1]], rotation=90)
    ax.set_ylim((0,.5))
    ax= axes[1,0]
    sns.barplot(data=df_calib.reset_index(drop=True), x='pred_bin', y=correct_col + '_mean', ax=ax, hue=2, legend=None)
    ax.plot([0,i-2],[0, 1], c='teal',linestyle='dashed' )
    plt.title(plot_title)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(ticks =  range(len(df_calib[conf_column+'_mean'])),labels = [f"{100*b:2.0f}%" for b in bins[:-1]], rotation=90)
    
