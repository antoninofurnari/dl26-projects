import sys
import os
import yaml
import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib as mpl
import matplotlib.pyplot as plt

# Aggiungiamo la root del progetto al PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.multisource_da import MultiSourceDA
from src.data.datasets import build_loaders

def main():
    # 1. Setup dell'ambiente e dei plot
    os.makedirs(os.path.join(project_root, 'figures'), exist_ok=True)
    mpl.rcParams.update({
        'figure.facecolor': '#FAFAEF',
        'axes.facecolor':   '#FAFAEF',
        'grid.color':       '#E0E0E0',
        'grid.alpha':        0.5,
        'font.size':         10,
    })
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device utilizzato: {device}")

    # 2. Caricamento Configurazione
    config_path = os.path.join(project_root, "experiments", "configs", "model_v1_in.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"Configurazione caricata: {config_path}")

    # 3. Caricamento Modello
    model = MultiSourceDA(
        in_dim=2048,
        num_classes=11,
        source_domains=['hmdb51', 'ucf101'],
        embed_dim=config['embed_dim'],
        num_domains=3
    ).to(device)

    ckpt_path = os.path.join(project_root, "experiments", "checkpoints", "msda_model_v1_in.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"Pesi del modello caricati da {ckpt_path}")

    # 4. Caricamento Dati
    print("Inizializzazione dataloaders...")
    source_loaders, target_loader, _, _ = build_loaders(config)
    num_batches = 3
    
    real_pca_features, real_pca_labels = [], []
    
    with torch.no_grad():
        print("Estrazione feature da HMDB-51...")
        it_s1 = iter(source_loaders['hmdb51'])
        for _ in range(num_batches):
            x, _, _ = next(it_s1)
            real_pca_features.append(model.encode(x.to(device)).cpu().numpy())
            real_pca_labels.extend(['HMDB51'] * len(x))
            
        print("Estrazione feature da UCF-101...")
        it_s2 = iter(source_loaders['ucf101'])
        for _ in range(num_batches):
            x, _, _ = next(it_s2)
            real_pca_features.append(model.encode(x.to(device)).cpu().numpy())
            real_pca_labels.extend(['UCF101'] * len(x))
            
        print("Estrazione feature dal Target (Kinetics)...")
        it_tgt = iter(target_loader)
        for _ in range(num_batches):
            x, _, _ = next(it_tgt)
            real_pca_features.append(model.encode(x.to(device)).cpu().numpy())
            real_pca_labels.extend(['Kinetics'] * len(x))
            
    all_features = np.vstack(real_pca_features)
    all_labels   = np.array(real_pca_labels)
    
    # 5. Calcolo PCA
    print(f"Calcolo PCA su {len(all_features)} vettori a {config['embed_dim']} dimensioni...")
    pca = PCA(n_components=2, random_state=42)
    features_2d = pca.fit_transform(all_features)
    
    # 6. Generazione del Grafico
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('#FAFAEF')
    ax.set_facecolor('#FAFAEF')
    
    palette = {'HMDB51': '#A8B8FF', 'UCF101': '#C8A800', 'Kinetics': '#EBC4C8'}
    markers = {'HMDB51': 'o',       'UCF101': 's',       'Kinetics': '^'}
    
    for domain in ['HMDB51', 'UCF101', 'Kinetics']:
        idx = all_labels == domain
        ax.scatter(features_2d[idx, 0], features_2d[idx, 1],
                   label=domain, color=palette[domain], marker=markers[domain],
                   alpha=0.75, edgecolors='white', linewidths=0.5, s=110)
                   
    ax.set_title("PCA 2D — Feature Reali dell'Encoder (Modello Addestrato)", fontsize=13, fontweight='bold')
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza spiegata)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza spiegata)")
    ax.legend(title="Dominio", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    # 7. Salvataggio
    save_path = os.path.join(project_root, "figures", "pca_alignment_real.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Grafico generato e salvato con successo in: {save_path}")

if __name__ == "__main__":
    main()
