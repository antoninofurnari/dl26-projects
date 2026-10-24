#!/usr/bin/env python3
"""
analyze_participants.py
-----------------------
Analizza le annotazioni del Multi-Instance Retrieval di EPIC-KITCHENS-100
per decidere QUALI partecipanti scaricare prima di lanciare gli script di download.

Per ogni participant_id riporta:
  - quanti segmenti (narration_id) ha in train e in test
  - quante sessioni video uniche (video_id) servirebbe scaricare
  - se compare in train, in test, o in entrambi
  - quante classi verbo/nome distinte copre (utile per la varieta' semantica)

Uso:
    python analyze_participants.py --ann-dir /path/to/epic-kitchens-100-annotations
    python analyze_participants.py --ann-dir ... --top 10 --budget-clips 8000

Output:
    - tabella riassuntiva a schermo
    - participants_summary.csv con tutte le statistiche
    - una proposta di subset e la stringa --participants pronta da copiare
"""

import argparse
import ast
import os
import sys

import pandas as pd


# --------------------------------------------------------------------------- #
# Caricamento
# --------------------------------------------------------------------------- #

def load_split(ann_dir, split):
    """Carica EPIC_100_retrieval_{split}.csv (o il .pkl se il csv manca)."""
    base = os.path.join(ann_dir, "retrieval_annotations", f"EPIC_100_retrieval_{split}")
    csv_path, pkl_path = base + ".csv", base + ".pkl"

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    elif os.path.exists(pkl_path):
        df = pd.read_pickle(pkl_path)
    else:
        sys.exit(
            f"[ERRORE] Non trovo ne' {csv_path} ne' {pkl_path}.\n"
            f"         Controlla --ann-dir: deve puntare alla root del repo "
            f"epic-kitchens-100-annotations."
        )

    # In alcune versioni il narration_id e' l'indice invece che una colonna.
    if "narration_id" not in df.columns:
        df = df.reset_index()

    # Fallback: ricava participant_id / video_id dal narration_id (es. P01_01_0).
    if "participant_id" not in df.columns:
        df["participant_id"] = df["narration_id"].str.split("_").str[0]
    if "video_id" not in df.columns:
        df["video_id"] = df["narration_id"].str.rsplit("_", n=1).str[0]

    df["split"] = split
    return df


def parse_noun_classes(value):
    """all_noun_classes puo' essere una lista vera o la sua stringa '[3, 7]'."""
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return list(parsed) if isinstance(parsed, (list, tuple, set)) else [parsed]
        except (ValueError, SyntaxError):
            return []
    if pd.isna(value):
        return []
    return [value]


# --------------------------------------------------------------------------- #
# Statistiche
# --------------------------------------------------------------------------- #

def per_participant_stats(df):
    """Costruisce una riga di statistiche per ogni participant_id."""
    rows = []
    for pid, g in df.groupby("participant_id"):
        train, test = g[g.split == "train"], g[g.split == "test"]

        nouns = set()
        if "all_noun_classes" in g.columns:
            for v in g["all_noun_classes"]:
                nouns.update(parse_noun_classes(v))
        elif "noun_class" in g.columns:
            nouns.update(g["noun_class"].dropna().tolist())

        verbs = set(g["verb_class"].dropna().tolist()) if "verb_class" in g.columns else set()

        if len(train) and len(test):
            presence = "both"
        elif len(train):
            presence = "train-only"
        else:
            presence = "test-only"

        rows.append({
            "participant_id": pid,
            "presence": presence,
            "n_train_clips": len(train),
            "n_test_clips": len(test),
            "n_total_clips": len(g),
            "n_videos_to_download": g["video_id"].nunique(),
            "n_verb_classes": len(verbs),
            "n_noun_classes": len(nouns),
        })

    out = pd.DataFrame(rows).sort_values("n_total_clips", ascending=False)
    # Quante clip "guadagni" per ogni sessione video scaricata: piu' alto = piu' efficiente.
    out["clips_per_video"] = (out.n_total_clips / out.n_videos_to_download).round(1)
    return out.reset_index(drop=True)


def propose_subset(stats, budget_clips):
    """
    Proposta greedy: prima i partecipanti presenti in ENTRAMBI gli split
    (cosi' hai train e test), ordinati per efficienza clip/video,
    finche' non si raggiunge il budget di clip richiesto.
    """
    pool = stats[stats.presence == "both"].sort_values("clips_per_video", ascending=False)
    if pool.empty:
        # Nessun partecipante in entrambi: prendo i migliori di ciascuno split.
        pool = pd.concat([
            stats[stats.presence == "train-only"].nlargest(3, "clips_per_video"),
            stats[stats.presence == "test-only"].nlargest(2, "clips_per_video"),
        ])
        return pool

    chosen, total = [], 0
    for _, row in pool.iterrows():
        if total >= budget_clips:
            break
        chosen.append(row)
        total += row.n_total_clips
    return pd.DataFrame(chosen)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ann-dir", required=True,
                    help="Root del repo epic-kitchens-100-annotations")
    ap.add_argument("--top", type=int, default=15,
                    help="Quante righe mostrare a schermo (default 15)")
    ap.add_argument("--budget-clips", type=int, default=8000,
                    help="Quante clip totali vuoi circa nel subset (default 8000)")
    ap.add_argument("--out", default="participants_summary.csv",
                    help="File CSV di output")
    args = ap.parse_args()

    train = load_split(args.ann_dir, "train")
    test = load_split(args.ann_dir, "test")
    df = pd.concat([train, test], ignore_index=True)

    print("\n=== PANORAMICA GENERALE ===")
    print(f"Segmenti train : {len(train):,}")
    print(f"Segmenti test  : {len(test):,}")
    print(f"Sessioni video totali referenziate : {df.video_id.nunique()}")
    print(f"Partecipanti totali                : {df.participant_id.nunique()}")

    stats = per_participant_stats(df)

    both = stats[stats.presence == "both"]
    print("\n=== COME SI DIVIDONO GLI SPLIT ===")
    print(f"In entrambi gli split : {len(both)}  -> {sorted(both.participant_id.tolist())}")
    print(f"Solo train            : {(stats.presence == 'train-only').sum()}")
    print(f"Solo test             : {(stats.presence == 'test-only').sum()}")
    if both.empty:
        print("\n[ATTENZIONE] Nessun partecipante compare in entrambi gli split: "
              "dovrai scaricare partecipanti diversi per train e per test.")

    print(f"\n=== TOP {args.top} PARTECIPANTI (per numero di clip) ===")
    cols = ["participant_id", "presence", "n_train_clips", "n_test_clips",
            "n_videos_to_download", "clips_per_video", "n_verb_classes", "n_noun_classes"]
    print(stats[cols].head(args.top).to_string(index=False))

    subset = propose_subset(stats, args.budget_clips)
    print(f"\n=== PROPOSTA DI SUBSET (~{args.budget_clips} clip) ===")
    print(subset[cols].to_string(index=False))
    print(f"\nClip totali nel subset : {int(subset.n_total_clips.sum()):,} "
          f"(train {int(subset.n_train_clips.sum()):,} / test {int(subset.n_test_clips.sum()):,})")
    print(f"Sessioni video da scaricare : {int(subset.n_videos_to_download.sum())}")
    print("\nComando pronto da usare:")
    print(f"  python epic_downloader.py --videos --action-retrieval "
          f"--participants {','.join(sorted(subset.participant_id.tolist()))}")

    stats.to_csv(args.out, index=False)
    print(f"\nStatistiche complete salvate in: {args.out}\n")


if __name__ == "__main__":
    main()