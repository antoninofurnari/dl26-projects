"""
label_analysis.py — Semantic Class Overlap Analysis
═══════════════════════════════════════════════════════
Analizza la sovrapposizione semantica tra le classi di:
  - HMDB-51   (Source 1)
  - UCF-101   (Source 2)
  - Kinetics  (Target)

Output:
  - overlap_hmdb_kinetics.csv   matrice similarità HMDB vs Kinetics
  - overlap_ucf_kinetics.csv    matrice similarità UCF  vs Kinetics
  - overlap_report.txt          top match per ogni classe
  - heatmap_hmdb_kinetics.html  heatmap interattiva HMDB vs Kinetics
  - heatmap_ucf_kinetics.html   heatmap interattiva UCF  vs Kinetics

Uso:
  python3 label_analysis.py ../../data
  python3 label_analysis.py ../../data --output risultati/
"""

import argparse
import csv
import difflib
import json
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# CARICAMENTO CLASSI
# ══════════════════════════════════════════════════════════════════════════════

def load_hmdb_classes(hmdb_root: Path):
    return sorted(d.name for d in hmdb_root.iterdir() if d.is_dir())


def load_ucf_classes(ucf_csv: Path):
    classes = set()
    with open(ucf_csv, newline="") as f:
        for row in csv.DictReader(f):
            classes.add(row["tag"])
    return sorted(classes)


def load_kinetics_classes(kin_root: Path):
    train_dir = kin_root / "train"
    return sorted(d.name for d in train_dir.iterdir() if d.is_dir())


# ══════════════════════════════════════════════════════════════════════════════
# SIMILARITÀ SEMANTICA
# ══════════════════════════════════════════════════════════════════════════════

def normalize(name: str) -> set:
    """
    Normalizza il nome di una classe in un insieme di parole.
    'shoot_ball' → {'shoot', 'ball'}
    'PlayingCello' → {'playing', 'cello'}
    'high jump'   → {'high', 'jump'}
    """
    import re
    # Inserisce spazio prima di maiuscole (CamelCase → Camel Case)
    name = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', name)
    # Sostituisce separatori con spazi
    name = name.replace('_', ' ').replace('-', ' ')
    return set(name.lower().split())


def jaccard_similarity(a: str, b: str) -> float:
    """Similarità Jaccard sulle parole normalizzate."""
    wa, wb = normalize(a), normalize(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def sequence_similarity(a: str, b: str) -> float:
    """Similarità carattere per carattere (difflib)."""
    a_norm = ' '.join(sorted(normalize(a)))
    b_norm = ' '.join(sorted(normalize(b)))
    return difflib.SequenceMatcher(None, a_norm, b_norm).ratio()


def semantic_similarity(a: str, b: str) -> float:
    """
    Score combinato: media di Jaccard e sequence similarity.
    Range: 0.0 (nessuna somiglianza) → 1.0 (identici)
    """
    j = jaccard_similarity(a, b)
    s = sequence_similarity(a, b)
    return round((j + s) / 2, 4)


# ══════════════════════════════════════════════════════════════════════════════
# MATRICE DI OVERLAP
# ══════════════════════════════════════════════════════════════════════════════

def compute_overlap_matrix(source_classes, target_classes):
    """
    Calcola la matrice (n_source x n_target) di similarità semantica.
    Restituisce una lista di liste di float.
    """
    matrix = []
    for sc in source_classes:
        row = [semantic_similarity(sc, tc) for tc in target_classes]
        matrix.append(row)
    return matrix


def top_matches(source_classes, target_classes, matrix, top_k=3):
    """
    Per ogni classe sorgente, restituisce i top_k match nel target.
    """
    results = {}
    for i, sc in enumerate(source_classes):
        row = matrix[i]
        ranked = sorted(enumerate(row), key=lambda x: -x[1])[:top_k]
        results[sc] = [(target_classes[j], score) for j, score in ranked]
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SALVATAGGIO CSV
# ══════════════════════════════════════════════════════════════════════════════

def save_csv(source_classes, target_classes, matrix, path: Path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['source\\target'] + target_classes)
        for i, sc in enumerate(source_classes):
            writer.writerow([sc] + matrix[i])
    print(f"  Salvato: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# REPORT TESTUALE
# ══════════════════════════════════════════════════════════════════════════════

def save_report(hmdb_top, ucf_top, path: Path):
    lines = []
    lines.append("=" * 60)
    lines.append("SEMANTIC CLASS OVERLAP REPORT")
    lines.append("=" * 60)

    lines.append("\n── HMDB-51 → Kinetics (Top 3 match per classe) ──\n")
    for cls, matches in hmdb_top.items():
        lines.append(f"  {cls:<25} →")
        for target, score in matches:
            bar = '█' * int(score * 20)
            lines.append(f"      {score:.3f} {bar:<20} {target}")

    lines.append("\n── UCF-101 → Kinetics (Top 3 match per classe) ──\n")
    for cls, matches in ucf_top.items():
        lines.append(f"  {cls:<25} →")
        for target, score in matches:
            bar = '█' * int(score * 20)
            lines.append(f"      {score:.3f} {bar:<20} {target}")

    # Statistiche globali
    lines.append("\n── Statistiche ──\n")

    hmdb_best = {cls: matches[0][1] for cls, matches in hmdb_top.items()}
    ucf_best  = {cls: matches[0][1] for cls, matches in ucf_top.items()}

    hmdb_avg = sum(hmdb_best.values()) / len(hmdb_best)
    ucf_avg  = sum(ucf_best.values())  / len(ucf_best)

    lines.append(f"  HMDB-51 → Kinetics  similarità media best-match: {hmdb_avg:.3f}")
    lines.append(f"  UCF-101 → Kinetics  similarità media best-match: {ucf_avg:.3f}")

    hmdb_high = [(c, s) for c, s in hmdb_best.items() if s >= 0.5]
    ucf_high  = [(c, s) for c, s in ucf_best.items()  if s >= 0.5]

    lines.append(f"\n  Classi HMDB con score ≥ 0.5 verso Kinetics: {len(hmdb_high)}/{len(hmdb_best)}")
    for c, s in sorted(hmdb_high, key=lambda x: -x[1]):
        lines.append(f"    {s:.3f}  {c}")

    lines.append(f"\n  Classi UCF con score ≥ 0.5 verso Kinetics: {len(ucf_high)}/{len(ucf_best)}")
    for c, s in sorted(ucf_high, key=lambda x: -x[1]):
        lines.append(f"    {s:.3f}  {c}")

    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Salvato: {path}")
    print('\n'.join(lines))


# ══════════════════════════════════════════════════════════════════════════════
# HEATMAP HTML (senza matplotlib)
# ══════════════════════════════════════════════════════════════════════════════

def score_to_color(score: float) -> str:
    """Converte score 0-1 in colore da bianco a rosso scuro."""
    r = int(255)
    g = int(255 * (1 - score ** 0.5))
    b = int(255 * (1 - score ** 0.5))
    return f"rgb({r},{g},{b})"


def save_heatmap_html(source_classes, target_classes, matrix,
                      title: str, path: Path):
    """
    Genera una heatmap interattiva come file HTML.
    Ogni cella mostra il valore al passaggio del mouse.
    """

    # Costruisce le righe della tabella
    rows_html = []
    for i, sc in enumerate(source_classes):
        cells = []
        for j, score in enumerate(matrix[i]):
            color = score_to_color(score)
            tc = target_classes[j]
            cell = (
                f'<td style="background:{color};width:8px;height:14px;" '
                f'title="{sc} → {tc}: {score:.3f}"></td>'
            )
            cells.append(cell)
        rows_html.append(
            f'<tr><td style="font-size:10px;white-space:nowrap;'
            f'padding-right:4px;">{sc}</td>' + ''.join(cells) + '</tr>'
        )

    # Header colonne (ruotato)
    header_cells = ''.join(
        f'<th style="writing-mode:vertical-rl;font-size:9px;'
        f'transform:rotate(180deg);height:80px;">{tc}</th>'
        for tc in target_classes
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: monospace; padding: 20px; background: #f5f5f5; }}
  h2 {{ color: #333; }}
  table {{ border-collapse: collapse; }}
  td, th {{ border: none; }}
  .legend {{ display:flex; align-items:center; gap:8px; margin-top:12px; }}
  .grad {{ width:200px; height:16px;
           background: linear-gradient(to right, white, rgb(255,0,0)); 
           border:1px solid #ccc; }}
</style>
</head>
<body>
<h2>{title}</h2>
<p>Ogni cella = similarità semantica tra classe sorgente (riga) e classe Kinetics (colonna).<br>
   <b>Rosso scuro</b> = alta similarità &nbsp;|&nbsp; <b>Bianco</b> = nessuna similarità.<br>
   Passa il mouse su una cella per vedere il valore esatto.</p>
<table>
  <thead>
    <tr><th></th>{header_cells}</tr>
  </thead>
  <tbody>
    {''.join(rows_html)}
  </tbody>
</table>
<div class="legend">
  <span>0.0</span>
  <div class="grad"></div>
  <span>1.0</span>
</div>
<p style="color:#888;font-size:11px;margin-top:20px;">
  Score = media(Jaccard similarity, SequenceMatcher) sui nomi normalizzati delle classi.
</p>
</body>
</html>"""

    with open(path, 'w') as f:
        f.write(html)
    print(f"  Salvato: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", help="Path alla cartella data/")
    parser.add_argument("--output", default="label_analysis_output",
                        help="Cartella di output")
    args = parser.parse_args()

    root    = Path(args.data_root)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Carica classi ─────────────────────────────────────────────────────────
    print("\nCaricamento classi…")
    hmdb_classes = load_hmdb_classes(root / "HMDB51")
    ucf_classes  = load_ucf_classes(root / "train.csv")
    kin_classes  = load_kinetics_classes(
        root / "kinetics400_5per" / "kinetics400_5per"
    )

    print(f"  HMDB-51 : {len(hmdb_classes)} classi")
    print(f"  UCF-101 : {len(ucf_classes)} classi")
    print(f"  Kinetics: {len(kin_classes)} classi")

    # ── Matrici di similarità ─────────────────────────────────────────────────
    print("\nCalcolo matrice HMDB-51 vs Kinetics…")
    hmdb_matrix = compute_overlap_matrix(hmdb_classes, kin_classes)

    print("Calcolo matrice UCF-101 vs Kinetics…")
    ucf_matrix  = compute_overlap_matrix(ucf_classes,  kin_classes)

    # ── Top match ─────────────────────────────────────────────────────────────
    hmdb_top = top_matches(hmdb_classes, kin_classes, hmdb_matrix, top_k=3)
    ucf_top  = top_matches(ucf_classes,  kin_classes, ucf_matrix,  top_k=3)

    # ── Salvataggio ───────────────────────────────────────────────────────────
    print("\nSalvataggio output…")
    save_csv(hmdb_classes, kin_classes, hmdb_matrix,
             out_dir / "overlap_hmdb_kinetics.csv")
    save_csv(ucf_classes,  kin_classes, ucf_matrix,
             out_dir / "overlap_ucf_kinetics.csv")
    save_report(hmdb_top, ucf_top,
                out_dir / "overlap_report.txt")
    save_heatmap_html(hmdb_classes, kin_classes, hmdb_matrix,
                      "HMDB-51 → Kinetics Semantic Overlap",
                      out_dir / "heatmap_hmdb_kinetics.html")
    save_heatmap_html(ucf_classes, kin_classes, ucf_matrix,
                      "UCF-101 → Kinetics Semantic Overlap",
                      out_dir / "heatmap_ucf_kinetics.html")

    print(f"\n✅ Analisi completata. Output in: {out_dir}/")


if __name__ == "__main__":
    main()