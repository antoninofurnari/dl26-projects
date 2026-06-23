<<<<<<< HEAD
> **NOTE:** This is the template for your report. Delete this note block before submission.

# [Project Title]
- **Group ID**: [E.g., G07]
- **Project ID**: [E.g., 1]
=======
# Multi-source Domain Adaptation per il Riconoscimento di Azioni

- **Group ID**: DataLost
- **Project ID**: Track 9
>>>>>>> origin/risultati-finali

---

## 1. Introduction and Objective
<<<<<<< HEAD
*Describe the objective of the project, why it is relevant, and what specific problem you are trying to solve. What is your main goal or initial hypothesis?*

## 2. Contribution and Added Value
*Summarize what was done concisely. ("We built a model based on X for task Y"). Highlight your added value compared to merely running existing code (e.g., new loss, different architecture, specific data augmentation, etc.).*

## 3. Data Used
Describe formally the used data:
- Where do they come from?
- What are the main statistics (number of samples, train/test split)?
- What kind of preprocessing or augmentation did you apply to prepare them for training?*

## 4. Methodology and Architecture
*Detail the experiments and how your system is built. What architecture did you use as a baseline? How did you modify it? Describe the network topology, key layers, the loss function used, and the training logic.*

## 5. Results and Discussion
Insert here the quantitative tables with the achieved results and compare your solution with the baseline. **Do not limit yourself to pasting numbers**, but comment on them:
- Why does model A perform better than model B?
- Are there classes where the model is particularly weak?
- Show qualitative examples (e.g., inserting correctly vs. incorrectly predicted images).*

Example:

**Table 1**: Quantitative results

| Model | Metric 1 | Metric 2 |
| :--- | :---: | :---: |
| Baseline | 65.4% | 45.1 |
| Our Final Model | 72.1% | 50.4 |

## 6. Conclusion and Limitations
*Summarize the project's outcome. What are the current limitations (e.g., requires too much memory, fails in low-light conditions)? If you had more time, what future experiments would you run?*

## 7. Additional Information

### 7.1 Contribution Breakdown
*Detail clearly who did what within the group.*
- **Person 1**: ...
- **Person 2**: ...
- **Person 3**: ...

### 7.2 Use of Artificial Intelligence
*Declare here the possible use of tools like Copilot or ChatGPT, specifying in which phases they helped you (e.g., writing boilerplate, debugging, documentation), keeping in mind that the architectural design and the responsibility for the result are yours.*
=======

Il progetto affronta una sfida complessa nel campo del Deep Learning: la **Multi-source Domain Adaptation (MSDA)** applicata al riconoscimento di azioni nei video. A nostra disposizione abbiamo due dataset sorgente completamente etichettati, **HMDB-51** e **UCF-101**, e un dataset target privo di etichette, costituito da un sottoinsieme di **Kinetics-400**. 
L'obiettivo principale è progettare e addestrare un modello in grado di combinare la conoscenza estratta dalle due sorgenti per classificare le azioni nel dominio target. La sfida metodologica consiste nel dotare il modello della capacità di valutare dinamicamente l'affidabilità di ciascuna sorgente a seconda del campione analizzato.

Affrontare questo problema ha un'importante rilevanza pratica: annotare manualmente i video è un'operazione costosa e lenta. Sfruttare dataset già etichettati per operare su nuovi domini sconosciuti rappresenta un'esigenza concreta. Nel nostro scenario *multi-source*, la difficoltà aumenta rispetto all'adattamento a singola sorgente, poiché le distribuzioni dei dati non sono equivalenti: una sorgente potrebbe rivelarsi più informativa dell'altra rispetto al target, e il modello deve imparare a gestire questa variabilità in autonomia.

Siamo partiti da due ipotesi:
(i) L'allineamento avversariale delle distribuzioni riduce il *domain shift*, portando a un miglioramento dell'accuratezza sul target.
(ii) Una pesatura dinamica delle sorgenti permette di sfruttare meglio i due dataset rispetto a un loro utilizzo statico. 

Il percorso per verificare queste ipotesi ha richiesto una revisione metodologica. In una prima fase, utilizzando un backbone pre-addestrato sullo stesso dataset del target, si verificava un *information leakage*. Dopo aver eliminato questa perdita di informazione, adottando un backbone pre-addestrato esclusivamente su immagini 2D (ImageNet), è stato possibile valutare oggettivamente l'efficacia della domain adaptation.

## 2. Contribution and Added Value

Abbiamo sviluppato un sistema MSDA composto da: un encoder condiviso, classificatori specifici per ciascuna sorgente, un discriminatore di dominio addestrato tramite *Gradient Reversal Layer* (GRL), e un meccanismo di **ensemble pesato dinamicamente** che unisce le predizioni in base alla loro confidenza sul target.

Rispetto all'esecuzione di codice preesistente, il progetto presenta i seguenti contributi:

- **Meccanismo di pesatura dinamica per confidenza:** Abbiamo implementato un sistema che combina i classificatori calcolando pesi dinamici per ogni batch target, basandosi sull'entropia delle predizioni. Questo meccanismo ha permesso di quantificare esplicitamente il **rapporto di influenza** tra le sorgenti (Obiettivo 4).
- **Mappatura semantica:** Abbiamo affrontato il setting a tre dataset costruendo uno spazio di etichette condiviso, allineando tassonomie eterogenee.
- **Simulazione del *DataLost*:** Abbiamo implementato la simulazione di sorgenti mancanti (source drop) per testare la resilienza del sistema in caso di assenza improvvisa di dati.
- **Studi di impatto e scalabilità:** Abbiamo valutato l'effetto del peso avversariale e la scalabilità del sistema al variare del numero di sorgenti.
- **Diagnosi e risoluzione del *Leakage*:** Abbiamo identificato che l'uso di un backbone pre-addestrato su Kinetics introduceva un bias metodologico nei risultati. Sostituendolo con un backbone ImageNet (I3D), abbiamo definito un setting di valutazione privo di leakage.
- **Identificazione e correzione di due difetti metodologici:** il collasso dei pesi per similarità coseno (causato dall'attivazione ReLU) e l'overconfidence del meccanismo di pesatura in presenza di sbilanciamento delle classi (§4.5, §5).

Per ragioni computazionali, il progetto opera su **feature pre-estratte** da un backbone congelato, permettendo di concentrare le risorse sull'allineamento delle distribuzioni.

## 3. Data Used

**Provenienza dei Dati**
- **HMDB-51** (Sorgente 1): Un dataset costituito da clip organizzate in cartelle per classe, fornite come sequenze di frame JPG.
- **UCF-101** (Sorgente 2): Comprende 101 classi con clip in formato video `.avi`, suddivise per categoria.
- **Kinetics-400** (Target): Abbiamo utilizzato un sottoinsieme (`kinetics400_5per`) con clip video `.mp4`. Durante l'addestramento queste clip sono state utilizzate rigorosamente senza etichette.

**Spazio di Etichette Condiviso**
Essendo le tassonomie differenti, la Domain Adaptation in contesto *closed-set* richiede uno spazio comune. Abbiamo costruito l'intersezione delle tre tassonomie, basandoci sul benchmark UCF-HMDB (Chen et al., 2019), ottenendo **11 classi** condivise: 
`climb, golf, kick_ball, pullup, punch, pushup, ride_bike, ride_horse, shoot_ball, shoot_bow, walk`.

La classe `fencing`, pur essendo presente nel benchmark originale UCF-HMDB, è stata **esclusa** poiché assente in Kinetics-400 (l'unica etichetta affine, `sword fighting`, descriveva di fatto un'azione diversa). Abbiamo scelto di prediligere 11 classi concettualmente "pulite" piuttosto che forzare mappature semantiche discutibili. Per ogni classe canonica, abbiamo mappato con cura i nomi originali (ad esempio: `shoot_ball` si traduce in `Basketball` per UCF e `shooting basketball` per Kinetics). Alcuni accoppiamenti hanno richiesto un'attenta analisi per essere giustificati, come `kick_ball` (`SoccerPenalty` / `kicking soccer ball`) o `walk` (`WalkingWithDog` / `walking the dog`). La coerenza di questa mappatura viene verificata automaticamente dal nostro codice prima di procedere all'estrazione.

**Statistiche Post-Filtraggio**

| Dataset | Clip Totali | Note sul Bilanciamento |
| :--- | :---: | :--- |
| **HMDB-51** | 1684 | Principalmente bilanciato (~100-130 per classe), **ad eccezione di `walk` (548 clip, pari al **32.5%** dell'intero split filtrato)**. |
| **UCF-101** | 1457 | Ottimo bilanciamento (100-164 clip per classe). |
| **Kinetics** | 303 | Molto piccolo e sbilanciato (12-45 clip per classe). |


| Classe canonica | HMDB-51 | UCF-101 | Kinetics-400 |
|---|---|---|---|
| `climb` | climb | RockClimbingIndoor | rock climbing |
| `golf` | golf | GolfSwing | golf driving range |
| `kick_ball` | kick_ball | SoccerPenalty | kicking soccer ball |
| `pullup` | pullup | PullUps | pull ups |
| `punch` | punch | Punch | punching person (boxing) |
| `pushup` | pushup | PushUps | push up |
| `ride_bike` | ride_bike | Biking | riding a bike |
| `ride_horse` | ride_horse | HorseRiding | riding or walking with horse |
| `shoot_ball` | shoot_ball | Basketball | shooting basketball |
| `shoot_bow` | shoot_bow | Archery | archery |
| `walk` | walk | WalkingWithDog | walking the dog |

La predominanza della classe `walk` in HMDB e le dimensioni ridotte di Kinetics si sono rivelati fattori determinanti per l'interpretazione dei risultati.

**Preprocessing e Scelta del Backbone**
Per limitare l'uso di memoria, usiamo un backbone 3D congelato che estrae le feature salvandole su disco in vettori monodimensionali. Per la lettura dei video usiamo PyAV in streaming, estraendo i 16 frame necessari per le clip.

La scelta del backbone ha rappresentato un punto di svolta. Il nostro primo esperimento si basava su **R3D-18 pre-addestrata su Kinetics-400**. Questo approccio, tuttavia, generava un inevitabile **information leakage**: essendo il target un sottoinsieme di Kinetics, la rete aveva già imparato a riconoscerne le distribuzioni, gonfiando la Baseline (Zero-Shot) a 0.756 e mascherando il reale domain shift. 
Per riportare onestà scientifica all'esperimento, siamo passati a una **ResNet-50 pre-addestrata esclusivamente su ImageNet** (immagini 2D), poi "gonfiata" a 3D tramite la tecnica I3D (Carreira & Zisserman). Questo backbone produce vettori da 2048 dimensioni e, non avendo mai processato alcun video di Kinetics in vita sua, ci garantisce un setting in cui il domain shift è genuino. **Tutti i risultati riportati nella Sezione 5 utilizzano questo backbone "onesto"**.

## 4. Methodology and Architecture

**4.1 Panoramica dell'Architettura**
Il sistema si basa su quattro componenti:
1. Un **Encoder Condiviso** (`E`), che elabora i dati provenienti da tutti i domini.
2. Un **Classificatore specifico** per ciascuna sorgente (HMDB e UCF), focalizzato sulle 11 classi.
3. Un **Discriminatore di Dominio**, addestrato in modo avversariale per riconoscere la provenienza delle feature.
4. Un **Ensemble Pesato Dinamicamente**, che in fase di inferenza sul target combina le predizioni affidandosi alla confidenza.

```text
                          +-> Classificatore_HMDB -> logit
feature --> Encoder E ----+-> Classificatore_UCF  -> logit
            (condiviso)   +-> GRL -> Discriminatore di dominio
                               |
                               v
                    Ensemble pesato per confidenza
                               |
                               v
                    Predizione finale sul Target
```

**4.2 Encoder Condiviso**
L'encoder proietta le feature da 2048D in uno spazio di embedding a 256D tramite una rete densa (MLP) con Batch Normalization, ReLU e Dropout. La rete è addestrata per produrre rappresentazioni discriminative per la classificazione e simultaneamente "domain-invariant" per il discriminatore.

**4.3 Classificatori per Sorgente**
Ogni sorgente utilizza una testa lineare dedicata. L'utilizzo di teste separate permette di modellare le specificità di ciascun dataset per poi pesarle selettivamente.

**4.4 Allineamento Avversariale (GRL)**
Il discriminatore di dominio stima la provenienza delle feature, mentre l'encoder viene ottimizzato per massimizzarne l'errore. Questo è ottenuto tramite il **Gradient Reversal Layer (GRL)**: durante il *forward pass* funge da identità, mentre nel *backward pass* inverte il segno del gradiente moltiplicandolo per $-\lambda$. 

Il fattore $\lambda$ viene schedulato partendo da 0 e crescendo gradualmente, in modo da stabilizzare inizialmente la classificazione delle azioni prima di forzare gradualmente l'invarianza di dominio.

**4.5 Ensemble Pesato per Confidenza**
Durante l'inferenza, le previsioni dei due classificatori vengono pesate. Inizialmente, la metrica scelta era la **similarità del coseno** tra i centroidi. Tuttavia, a causa dell'attivazione ReLU, le feature si concentravano nel quadrante positivo, portando la similarità a saturare costantemente su valori prossimi a ~0.999 e causando il collasso dei pesi su 0.5/0.5.

La soluzione adottata è la **pesatura basata sull'entropia**. Valutando l'entropia media delle predizioni su un batch target, il classificatore con minore incertezza riceve un peso proporzionalmente maggiore tramite una funzione Softmax con temperatura:

$$w_i = \frac{\exp(-H_i / T)}{\sum_j \exp(-H_j / T)}, \quad H_i = -\sum_c p_{ic} \log p_{ic}$$

dove $H_i$ è l'entropia media del classificatore $i$ sul batch target, $T$ è la temperatura e $p_{ic}$ è la probabilità predetta per la classe $c$. Negli esperimenti è stato utilizzato un valore di $T=0.1$. Questo approccio ha permesso inoltre di misurare il **Rapporto di Influenza** tra le sorgenti.

**4.6 Logica di Training e Funzione di Perdita**
Ad ogni step viene estratto un batch per ciascun dominio (Target, S1, S2). La Loss complessiva è la somma della Cross-Entropy di classificazione (sulle sorgenti) e la Loss Avversariale del discriminatore (su tutti i domini):

$$\mathcal{L}_\text{tot} = \mathcal{L}_\text{cls}^{S_1} + \mathcal{L}_\text{cls}^{S_2} + \lambda_\text{adv} \cdot \mathcal{L}_\text{adv}^{\text{TGT}}$$

con $\mathcal{L}_\text{cls}$ = Cross-Entropy sulle sorgenti etichettate e $\mathcal{L}_\text{adv}$ = Cross-Entropy del discriminatore di dominio (a 3 classi) applicata a tutti i domini. L'ottimizzatore utilizzato è Adam.

**4.7 Valutazione Baseline**
Per quantificare il Domain Shift, testiamo una **Baseline Source-only**: un modello addestrato sulle sorgenti senza allineamento GRL, testato zero-shot sul target. La differenza tra questa Baseline e il modello MSDA rappresenta il contributo netto dell'adattamento di dominio.

**4.8 Dinamiche Aggiuntive Esplorate**
- **Simulazione Source Drop (DataLost)**: spegnimento stocastico di una sorgente a livello di batch per simulare scenari di reti instabili.
- **Peso Avversariale Modulabile**: test con allineamento nullo (0.0), parziale (0.3) e totale (1.0).
- **Bilanciamento Dati**: sottocampionamento della classe maggioritaria `walk` in HMDB per ridurre i bias predittivi.

## 5. Results and Discussion

I risultati che seguono sono basati sul backbone ImageNet per evitare il data leak.

**Tabella 1**: Ablation study sul peso avversariale e sul bilanciamento

| Modello | Accuracy Target | Macro-Accuracy | Influenza HMDB/UCF |
| :--- | :---: | :---: | :---: |
| Baseline (No DA) | 0.667 | n.d. (modello non addestr.) | - |
| MSDA (adv=0.0) | 0.683 | 0.679 | 0.54 / 0.46 |
| MSDA (adv=0.3) | 0.680 | 0.685 | 0.63 / 0.37 |
| MSDA (adv=1.0) | 0.713 | 0.718 | 0.64 / 0.36 |
| **MSDA (adv=1.0) + `walk` bilanciata** | **0.759** | **0.748** | **0.50 / 0.50** |

**Tabella 2**: Effetto del Leakage sulle prestazioni

| Configurazione | Backbone Kinetics (Leakage) | Backbone ImageNet (Onesto) |
| :--- | :---: | :---: |
| Baseline (No DA) | 0.756 | 0.667 |
| MSDA (adv=1.0) | 0.700 | 0.713 |

**Tabella 3**: Studio di Scalabilità

| Sorgenti Attive | Accuracy Target | Δ vs combinato |
| :--- | :---: | :---: |
| Solo HMDB-51 | 0.419 | −0.294 |
| Solo UCF-101 | 0.690 | −0.023 |
| Media pesata (teorica) | ~0.555 | −0.158 |
| **HMDB-51 + UCF-101** | **0.713** | — |

**Discussione Critica:**

- **Analisi del Leakage:** La Tabella 2 evidenzia che il backbone Kinetics causava una Baseline artificialmente elevata (0.756). L'applicazione della MSDA in questo scenario riduceva le prestazioni (0.700), andando ad alterare uno spazio latente pre-allineato al target. Utilizzando il backbone ImageNet, la Baseline iniziale scende a 0.667, ma l'applicazione della DA garantisce un incremento fino a 0.713, confermando l'efficacia del metodo in assenza di bias.

- **Impatto del Bilanciamento:** L'equilibratura della classe `walk` (sottocampionata da 548 a 150 clip in HMDB) ha portato l'accuratezza target a **0.759** (+9.2 pp rispetto alla baseline). Contestualmente, l'influenza delle sorgenti è passata da uno sbilanciamento verso HMDB (0.64) a una distribuzione equa (0.50/0.50), mitigando l'effetto di *overconfidence* causato dalla sproporzione dei dati.

- **Sinergia Multi-Source:** La Tabella 3 mostra che l'addestramento separato su HMDB e UCF porta ad accuratezze inferiori (0.419 e 0.690). L'uso congiunto delle sorgenti raggiunge 0.713, indicando che il modello sfrutta le caratteristiche complementari dei due dataset.

## 6. Conclusion and Limitations

Il progetto implementa un sistema per la Multi-Source Domain Adaptation tramite encoder condiviso, GRL e un meccanismo di ensemble dinamico per confidenza. 

L'analisi ha evidenziato l'impatto metodologico del *data leak* derivante dall'utilizzo di pesi pre-addestrati sullo stesso dominio del target, motivando il passaggio a una valutazione basata su ImageNet. In questo contesto, l'allineamento avversariale e l'uso congiunto di più sorgenti hanno ridotto efficacemente il Domain Shift, con risultati ulteriormente migliorati dal bilanciamento delle classi.

Limitazioni attuali:
- **Overconfidence del Softmax:** La pesatura per confidenza favorisce eccessivamente le sorgenti sbilanciate, richiedendo un attento preprocessing o l'introduzione di tecniche compensative come il *temperature scaling*.
- **Backbone Spaziale:** Un modello inizializzato su ImageNet è privo di modellazione temporale profonda. Un backbone addestrato su dataset video estranei al target, dove il reasoning temporale è strettamente necessario (es. *Something-Something*, le cui azioni dipendono dalla direzione del movimento e non dagli oggetti), migliorerebbe la stima delle dinamiche temporali preservando l'assenza di leakage. Tale dataset non è stato tuttavia utilizzato poiché non è disponibile gratuitamente senza approvazione accademica, e un eventuale fine-tuning from scratch avrebbe reintrodotto gli onerosi costi computazionali già riscontrati con la rete R(2+1)D.
- **Dimensione del Target:** La varianza nelle metriche valutative è parzialmente causata dal ridotto numero di clip disponibili nel dataset Target (303 in totale).
- **Feature Congelate:** L'addestramento offline su feature pre-estratte impedisce all'encoder profondo di adattarsi end-to-end.

## 7. Additional Information

**7.1 Breakdown dei Contributi**
- **Jhoannis Caccamo:** Ha implementato il codice base, strutturando la pipeline di preprocessing per l'estrazione e l'armonizzazione delle clip dai dataset.
- **Alessia Maccarrone:** Ha gestito la fase di addestramento e valutazione, individuando il problema del *data leak* ed elaborando l'analisi interpretativa delle metriche.
- **Matteo Vullo:** Ha curato la logica di bilanciamento delle classi e implementato le visualizzazioni grafiche.

**7.2 Utilizzo di Strumenti di Assistenza**
Durante lo sviluppo sono stati utilizzati strumenti basati su Intelligenza Artificiale per accelerare la scrittura del codice di utilità, assistere nelle fasi di debugging e supportare la formattazione di grafici e reportistica. Le scelte architetturali, l'analisi delle metriche e l'interpretazione dei risultati sono rimaste responsabilità esclusiva del team.


**7.3 Architettura Software e Strumenti**
L'implementazione tecnica del progetto è modulare e organizzata nella cartella `src/`. Di seguito i principali file creati e le librerie sfruttate:

- **Librerie Core:**
  - `torchvision` (in particolare `torchvision.models.video` e `torchvision.transforms`): essenziale per caricare il backbone `r50_i3d` inflato da ImageNet e per applicare le pipeline di data augmentation e standardizzazione (crop, resize, normalizzazione) alle clip.
  - `av` (PyAV): utilizzato intensivamente in fase di preprocessing per la decodifica dei container video (`.mp4`, `.avi`). Questa libreria ha permesso uno streaming altamente efficiente, estraendo solo i 16 frame necessari saltando l'oneroso caricamento in RAM di interi filmati.
  - `PyTorch` (framework principale): per la gestione dei Tensori, l'AutoGrad e l'ottimizzazione tramite Adam.

- **Moduli Sviluppati (`src/`):**
  - `src/extract_features.py`: Script pipeline che utilizza `PyAV` e `torchvision` per processare tutti i dataset video grezzi, estrarre gli embedding 2048-D e salvarli offline su disco in tensori leggeri, risolvendo le limitazioni di RAM e tempo macchina.
  - `src/data/feature_dataset.py`: Contiene i Custom PyTorch Dataset ottimizzati per il caricamento in memoria vettoriale. Sostituisce l'I/O video in fase di training, accelerando vertiginosamente le epoche (da ore a minuti).
  - `src/models/multisource_da.py` e `src/models/grl.py`: Contengono le definizioni architetturali, inclusi l'Encoder Condiviso, le Teste di Classificazione, il Discriminatore e l'implementazione personalizzata del *Gradient Reversal Layer* mediante `torch.autograd.Function`.
  - `src/training/train.py`: Il cuore operativo. Definisce il training loop UDA (Unsupervised Domain Adaptation), calcolando simultaneamente la classificazione supervisionata sulle sorgenti e la Loss Avversariale per l'allineamento dei tre domini in un unico step di retropropagazione.

---

### Riferimenti
- Y. Ganin, V. Lempitsky, *Unsupervised Domain Adaptation by Backpropagation*, ICML 2015.
- M.-H. Chen et al., *Temporal Attentive Alignment for Large-Scale Video Domain Adaptation*, ICCV 2019.
- D. Tran et al., *A Closer Look at Spatiotemporal Convolutions for Action Recognition*, CVPR 2018.
>>>>>>> origin/risultati-finali
