# doomLaya contro doomGemma — risultati per l'articolo

*Misure del 25 settembre 2026 su soft-2 (laboratorio di dexmac). Documento di lavoro: tutti i numeri vengono dai file in `results/`, prodotti dagli script in `src/`. English version: [RESULTS.md](RESULTS.md).*

## In una riga

Con gli stessi 979 esempi, una LoRA di 27 minuti porta Gemma 4 E2B al livello di Laya v3 in partita (6/6 contro 6/6 su MAP01) e sopra nelle decisioni singole. A parità di taglia un decoder è veloce quanto Laya (22 contro 23 ms); a parità di risultato Laya resta circa 3 volte più veloce. Su una mappa nuova non esce nessuno, neanche l'insegnante a regole: su questo banco la generalizzazione non si può misurare.

## 1. Domanda

I "System One models" (Jev di TypeSafe, chiuso; Laya di Convai, aperto, 421M, encoder non autoregressivo) promettono decisioni tipizzate con probabilità calibrate in un solo passaggio, al posto degli LLM. Il progetto doomLaya ha mostrato Laya, adattato a Doom, finire FreeDoom MAP01 più in fretta di Jev.

Le domande:
1. il vantaggio viene dall'**architettura** "System One" o dal fatto di avere un **modello piccolo addestrato sul compito**?
2. un LLM generico addestrato sugli **stessi dati** fa altrettanto?
3. qualcuno dei due ha imparato a **giocare**, oppure ha imparato **quel livello**?
4. quanto sono vicini all'**insegnante a regole** da cui vengono le etichette?

## 2. Lavori precedenti (verificati il 25 set 2026)

| lavoro | cosa fa | cosa manca rispetto a questo |
|---|---|---|
| [azalio/doomLaya](https://github.com/azalio/doomLaya) | Laya v3 contro Jev su MAP01, un seme (48) | nessun LLM addestrato; scrivono che "la geometria di un livello nuovo non è stata provata" |
| [mithalouni/system-one-open](https://github.com/mithalouni/system-one-open) | replica aperta di Jev su Gemma 4 E2B con LoRA; Doom giocato a freddo | nessun numero su Doom, nessun addestramento su Doom, nessun confronto con Laya |
| [SauerkrautLM-Doom, arXiv 2604.07385](https://arxiv.org/html/2604.07385) | modello da 1,3M contro LLM a freddo su `defend_the_center` | LLM non addestrati: confronto squilibrato |
| [anth.us, "Jev vs Laya: same labels"](https://anth.us/blog/jev-vs-laya/) | Laya addestrato contro Jev sulle stesse etichette | classificazione del sentimento; niente LLM generativi, niente Doom |
| [jwalin-shah/tensor-logic #92](https://github.com/jwalin-shah/tensor-logic/issues/92) | propone "Laya contro LLM addestrati" | solo proposta, non eseguita |

Non risulta pubblicato un confronto con **gli stessi dati di Laya v3, una LoRA su un LLM generativo, più semi, una mappa mai vista, la calibrazione e la latenza a parità di hardware**.

## 3. Banco di prova

- **Hardware**: Ryzen 5 5600X, RTX 4070 12 GB + RTX 2070 SUPER 8 GB, driver 595.91, Ubuntu 24.04. Latenze confrontabili misurate tutte **sulla 4070**.
- **Software**:
  - doomLaya `b25edd3`, pacchetto `laya` `42626c3`;
  - checkpoint `azalio/laya-doom-v3`, sha256 di ogni file verificato con il loro `SHA256SUMS`;
  - ViZDoom 1.3.1 con `freedoom2.wad`, torch 2.14 (CUDA 13.0), transformers 4.57 per Laya e 5.17 con peft 0.21 per la LoRA;
  - llama.cpp `7077abbe1`.
- **Gioco**: codice di doomLaya invariato. 35 tic/s, richieste **asincrone** al massimo ogni 0,5 s, il comando precedente resta valido 2 s. Il gioco **non aspetta** il modello: un modello lento reagisce in ritardo.
- **Domande**: l'agente trasforma lo stato in testo (salute, armi, nemici visibili con id e distanza, oggetti, porte) e pone due domande a scelta:
  - `command`: spara al nemico #N, raccogli l'oggetto #N, apri la porta, uscita, ritirata, esplora, aspetta;
  - `weapon`.
  L'esecutore mira, calcola il percorso e preme i tasti. **Le scelte sono tutte del modello.**
- **Ponte per gli LLM** (`serve_llm.py`): stessa API `/predict` del server di Laya. Ogni opzione diventa una lettera e una **grammatica** permette un solo token fra le lettere valide. Le probabilità vengono dai `top_logprobs`, rinormalizzate sulle opzioni. È **punteggio in un solo passaggio** (lettura del prompt più un token), non generazione libera: niente errori di interpretazione. In modalità `--separate` le due domande partono in parallelo, con lo **stesso prompt usato in addestramento**.

## 4. Addestramento

- **Dati**: `training/v3/train.json` (979 domande) e `validation.json` (130), **gli stessi di Laya v3**. Seed di gioco 42 e 43 per l'addestramento, 44 per la validazione, **tutto su MAP01**. Etichette assegnate offline dalle regole del gioco (vedi TRAINING.md di doomLaya).
- **LoRA** (`lora_train.py`):
  - rango 16, alfa 32, dropout 0,05, su `q,k,v,o,gate,up,down` del modello di linguaggio;
  - AdamW, lr 1e-4, weight decay 0,01, accumulo 8, clip 1, seme 771;
  - **perdita solo sulla lettera della risposta**;
  - si tiene l'epoca migliore per comando + arma in validazione, come fa doomLaya.

| modello | parametri addestrati | tempo per epoca (4070) | epoca migliore | adattatore |
|---|---|---|---|---|
| Gemma 4 E2B (5,1 miliardi totali, circa 2 "effettivi") | 24,2M | 550 s | 2 di 3 | 97 MB |
| Gemma 3 270M (268M) | 3,8M | 101 s | 5 di 5 | 15 MB |

- **Messa in uso**: fusione della LoRA, conversione in GGUF, Q8_0. doomGemma E2B pesa 4,9 GB e doom-270M 292 MB.
- **Laya v3** è quello pubblicato da azalio: 3 fasi, addestrando la testa e poi gli ultimi 3 strati dell'encoder. Non l'abbiamo riaddestrato.
- **Due intoppi tecnici**:
  - con Gemma 4 il gradient checkpointing produce un `CheckpointError`, quindi è disattivato;
  - la tabella `embed_tokens_per_layer` di E2B pesa 4,4 GiB, più degli strati. È tenuta in RAM, e il calcolo resta sulla sola 4070.

## 5. Risultati

### 5.1 Decisioni singole: 130 domande di validazione

| modello | comando (80) | arma (50) | totale |
|---|---|---|---|
| Gemma 3 270M a freddo | 0,16 | 0,02 | 0,11 |
| Laya base inglese (`convaiinnovations/laya`, radice, @55cf4c4) | 0,41 | 0,58 | 0,48 |
| Laya typed-decisions | 0,40 | 0,60 | 0,48 |
| Gemma 4 E2B a freddo | 0,51 | 0,78 | 0,62 |
| **Laya v3** | 0,725 | 0,98 | 0,82 |
| **doom-270M** (Gemma 3 270M + LoRA) | 0,89 | 1,00 | 0,93 |
| **doomGemma** (E2B + LoRA) | **0,96** | **1,00** | **0,98** |

Gemma E2B **senza addestramento batte già Laya senza addestramento**, e vale per **entrambi** i checkpoint di Laya. La model card di Convai avverte che typed-decisions è specializzato su quattro flussi di lavoro sintetici; per questo abbiamo provato anche la **base inglese** (ModernBERT-large, 421M, sha256 `891102d3…` verificato): stesso risultato (0,48). L'errore tipico dei modelli a freddo è "raccogli" quando la risposta giusta era "uscita".

### 5.2 Calibrazione (130 domande)

ECE a 15 intervalli sulla confidenza massima e Brier multi-classe. La temperatura è stimata con **validazione incrociata a 2 parti sulla validazione**, quindi nessun numero è misurato sui dati con cui è stato tarato (`calibra.py`).

| modello | ECE grezzo | ECE tarato | Brier grezzo | Brier tarato | T stimata |
|---|---|---|---|---|---|
| Gemma 3 270M a freddo | 0,65 | 0,25 | 1,23 | 0,81 | 5–9 |
| Gemma 4 E2B a freddo | **0,38** | 0,22 | 0,77 | 0,60 | **4,3–4,6** |
| Laya typed-decisions | 0,24 | 0,22 | 0,68 | 0,65 | 0,45–0,8 |
| Laya base inglese | **0,085** | 0,12 | 0,65 | 0,66 | 1,05–1,5 |
| **Laya v3** | 0,06 | 0,11 ⚠️ | 0,26 | 0,25 | 1,5 |
| doom-270M | 0,05 | 0,07 | 0,11 | 0,11 | 1,05–1,5 |
| **doomGemma** | **0,04** | **0,03** | **0,05** | **0,04** | 0,55–0,6 |

- **Laya base è ben calibrato ma poco discriminante.** L'ECE basso (0,085) viene da confidenze basse e uniformi, non da una buona distinzione fra risposte giuste e sbagliate: l'accuratezza è 0,48 e il Brier (0,65) è lo stesso di typed-decisions. In partita sceglie "aspetta" 350 volte su 350, un comportamento degenere. **Nota**: la model card di Convai dice il contrario, cioè che la base "esce sovra-confidente" (ECE medio 0,466 prima della taratura sul loro banco); che qui non lo sia è una differenza da segnalare, non una conferma.
- **Gli LLM a freddo sono molto presuntuosi.** Gemma E2B risponde con probabilità 1,00 anche quando sbaglia, e la sua confidenza va "raffreddata" di un fattore circa 4,4.
- **Dopo la LoRA doomGemma è il modello meglio calibrato**, anche senza taratura.
- ⚠️ **Su Laya v3 la taratura migliora la NLL ma peggiora l'ECE**. Con 65 domande per metà è rumore statistico: lo riportiamo così.

### 5.3 Latenza a parità di hardware (RTX 4070)

Gli stessi **1.416 pacchetti di gioco** (stato più le due domande vere) sono rigiocati in sequenza. Si misura il tempo end-to-end, con 10 richieste di riscaldamento (`latenza.py`, `latenza_diretta.py`).

| modello e modo di servirlo | p50 | p90 | p99 | decisioni/s |
|---|---|---|---|---|
| **Laya v3**, suo server (FastAPI + PyTorch) | **23 ms** | 24 ms | 25 ms | 41,8 |
| **doom-270M, calcolo diretto** PyTorch (un passaggio, 2 domande in batch, logit delle lettere) | **22 ms** | 23 ms | 24 ms | **44,7** |
| doom-270M via ponte + llama.cpp Q8_0 | 71 ms | 88 ms | 113 ms | 13,2 |
| doomGemma E2B via ponte + llama.cpp Q8_0 | 59 ms | 109 ms | 118 ms | 14,1 |
| Gemma E2B a freddo, stesso ponte | 59 ms | 108 ms | 117 ms | 14,1 |
| doomGemma E2B, calcolo diretto bf16 | 84 ms | 87 ms | 100 ms | 12,0 |

- **A parità di taglia e di modo di calcolo, il decoder è veloce quanto l'encoder "System One"** (22 contro 23 ms): l'architettura non dà un vantaggio di velocità **intrinseco**.
- **A parità di risultato in partita il quadro è diverso.** Il decoder da 270M veloce quanto Laya esce solo 2 volte su 6; quello che pareggia Laya in partita è doomGemma, che fa 59 ms via llama.cpp e 84 ms in calcolo diretto, cioè **da 2,5 a 3,7 volte più lento**, e pesa 4,9 GB contro meno di 1 GB. **Con questi dati non esiste ancora un decoder che sia insieme veloce quanto Laya e bravo quanto Laya.** Resta aperto se il vantaggio venga dall'encoder bidirezionale, che su questo compito renderebbe di più a parità di parametri, o solo dal fatto che nessuno ha ancora provato a stringere il decoder fra 270M e 2B.
- **La cadenza del banco non mette alla prova la latenza.** Il gioco chiede una decisione al massimo ogni 0,5 s: fra 23 e 109 ms, in partita non cambia quasi niente. Il pareggio non dimostra che la latenza non conti; dice che **qui non viene misurata**. Conterebbe in un ciclo di controllo più stretto, per esempio su un robot.
- **Con llama.cpp il 270M costa come E2B** (71 contro 59 ms). Oltre una certa soglia il tempo lo spende la **catena di servizio** (HTTP, grammatica, logprob su un vocabolario da 262.000 parole), non il modello.
- **Via llama.cpp la latenza di Gemma ha due gobbe** (circa 59 e circa 108 ms): è il riuso della cache del prompt fra stati simili. In partita questo effetto ha fatto sembrare doomGemma più veloce su MAP02 (42–62 ms): **era la cache, non il modello**.
- Nota di contesto: nella prima serie Laya girava sulla 2070S (38 ms) e sembrava più lento di quanto sia.

### 5.4 In partita: MAP01, la mappa di addestramento

Difficoltà 3, massimo 180 s. **Il seme 48 non è nell'addestramento; i semi 49–53 non sono né nell'addestramento né nella validazione.**

**Seme 48, tutti i giocatori:**

| giocatore | esce | uccisioni | morti | latenza in partita (p50) |
|---|---|---|---|---|
| Laya base inglese | ❌ (sta fermo: "aspetta" 350 volte su 350) | 0 | 0 | 26 ms |
| Laya typed-decisions | ❌ | 8 | 1 | 39 ms (2070S) |
| Gemma 4 26B-A4B a freddo | ❌ | 13 | 1 | 355 ms |
| Bonsai 27B ternario a freddo | ❌ | 6 | 2 | 979 ms |
| Gemma 4 E2B a freddo | ❌ | 10 | 0 | 93 ms |
| doom-270M | ❌ | 5 | 0 | 77 ms |
| **Laya v3** | ✅ 71,0 s | 15 | 0 | 38 ms (2070S) |
| **doomGemma** | ✅ **56,6 s** | 13 | 0 | 109 ms |
| *Jev 1.13 (dato doomLaya, non riprodotto)* | *✅ 69,0 s* | *13* | *0* | *357 ms* |
| **oracolo** (le regole di etichettatura come giocatore) | ✅ 65,5 s | 15 | 0 | < 1 ms |

I modelli a freddo hanno una **fissa per raccogliere oggetti**: Gemma E2B ha scelto "raccogli" 271 volte su 350, Gemma 26B 274 su 349. Quando vedono nemici sparano meno della metà delle volte (Gemma 26B 41 su 89, Bonsai 20 su 102).

**Semi 48–53 (6 partite ciascuno):**

| giocatore | esce | tempo medio | intervallo | uccisioni medie | morti |
|---|---|---|---|---|---|
| **Laya v3** | 6/6 | 59,5 s | 42,9–71,0 s | 13,5 | 0 |
| **doomGemma** | 6/6 | 60,0 s | 56,6–64,2 s | 13,0 | 0 |
| doom-270M | **2/6** | 68,5 s (quando esce) | 62,8–74,1 s | 9,7 | 0 |
| **oracolo** | 6/6 | **81,6 s** | 64,5–124,9 s | 16,0 | 1 |

**Pareggio in partita fra Laya v3 e doomGemma.** Laya è più variabile, doomGemma più costante. Il 56,6 contro 71,0 del seme 48 era in parte fortuna.

**L'insegnante a regole come riferimento.** L'oracolo è la funzione `gold` di `training/build_dataset.py`, copiata identica, che riceve lo stesso pacchetto e lo stesso stato grezzo da cui sono nate le etichette e passa dallo stesso esecutore con la stessa cadenza (`oracolo.py`). Esce 6 volte su 6 in 81,6 s di media. L'oracolo **non ottimizza il tempo di uscita**: le sue regole includono raccogliere gli oggetti utili, e infatti raccoglie di più (17,7 in media, fino a 31) e uccide di più (16,0 contro circa 13). **I modelli escono prima dell'insegnante (circa 60 s contro 82) perché lo imitano in modo imperfetto e raccolgono meno.** È una conseguenza dell'imitazione imperfetta, non un merito dei modelli. Sul solo tempo di uscita, quindi, 60 s non è il tetto del banco.

### 5.5 In partita: MAP02, mappa mai vista (semi 48–50, massimo 300 s)

| giocatore | esce | uccisioni (per seme) | morti (per seme) |
|---|---|---|---|
| Laya typed-decisions | 0/3 | 57 · 46 · 38 | 11 · 7 · 6 |
| Gemma E2B a freddo | 0/3 | 46 · 42 · 49 | 3 · 3 · 3 |
| Laya v3 | 0/3 | 27 · 22 · 36 | 2 · 1 · 3 |
| doomGemma | 0/3 | 11 · 21 · 26 | 0 · 1 · 2 |
| doom-270M | 0/3 | 47 · 18 · 15 | 4 · 1 · 1 |
| **oracolo** | **0/3** | 26 · 13 · 20 | 2 · 0 · 1 |

**Nessuno esce, 0 su 18, oracolo compreso.** L'opzione "uscita" era **sempre** fra le scelte (il navigatore conosce la posizione dell'uscita dalla mappa) e i modelli l'hanno anche scelta (doomGemma 69 volte, Laya v3 38 nel seme 48). Il problema è a valle: **l'oracolo sceglie "apri la porta" circa 500 volte su 580** e l'esecutore passa 7.600–9.500 tic davanti alla porta del **settore 37** (coordinate 508, −764). **Verificato leggendo `freedoom2.wad`:** tutte e 6 le linedef di quella porta hanno lo special **27, "DR, chiave gialla"**: si apre con USE solo se il giocatore ha la chiave gialla. La chiave gialla c'è su MAP02, alle coordinate (−256, 800), lontana dalla porta. La catena si blocca per tre ragioni combinate:
  - il sensore di doomLaya considera "porta apribile" anche le porte a chiave (special 26, 27, 28 e 32–34) e le offre come opzione `open_door`;
  - la regola dell'insegnante apre qualsiasi porta chiusa entro 4 m;
  - le chiavi entrano fra le opzioni solo quando sono viste entro 12 m, e questa è lontana e fuori vista.

Su MAP02 **fallisce l'intera catena** (regole, esecutore, rappresentazione dello stato), non la capacità dei modelli di generalizzare: **qui la generalizzazione non si può misurare.** *(Le uccisioni si sommano anche dopo le morti: chi muore spesso riparte e ne accumula di più.)*

### 5.6 Quanto l'esecutore rifiuta i comandi ("binari")

È la quota di tempo in cui l'esecutore **non può** eseguire il comando del modello: strada bloccata o bersaglio irraggiungibile (`binari.py`, dalla telemetria).

| giocatore | MAP01 | MAP02 |
|---|---|---|
| Bonsai a freddo | 72% | — |
| **doom-270M** | **60%** (53% bloccato) | **76%** |
| Laya typed-decisions | 43% | 25% |
| Gemma 26B a freddo | 39% | — |
| Gemma E2B a freddo | 37% | 26% |
| doomGemma | 22% | 8,5% |
| Laya v3 | 13% | 9,7% |
| oracolo | 20,5% | 3,3% |

Questo spiega doom-270M: **ha la validazione alta (0,93) ma continua a scegliere azioni impossibili in quel momento**, e in partita esce solo 2 volte su 6. Le domande prese una per una non misurano la coerenza da un istante all'altro.

## 6. Cosa si può affermare

1. **L'addestramento sul compito conta più dell'architettura.** A freddo perdono tutti, e Gemma E2B è anzi migliore di Laya. Con gli stessi 979 esempi, una LoRA di 27 minuti su una scheda da consumatore porta un LLM generico al livello del modello specializzato in partita, e sopra nelle decisioni singole e nella calibrazione.
2. **Latenza: a parità di taglia nessun vantaggio intrinseco dell'architettura** (decoder da 270M a 22 ms, Laya a 23). **A parità di risultato in partita, però, Laya è circa 3 volte più veloce** del decoder che lo pareggia, perché ci arriva con un modello più piccolo. Su questo banco la differenza non pesa, perché le decisioni sono chieste ogni 0,5 s.
3. **La precisione sulle domande singole non predice la partita.** doom-270M ha 0,93 in validazione e finisce 2 partite su 6; Laya v3 ha 0,82 e le finisce tutte.
4. **La calibrazione degli LLM a freddo non è affidabile** (ECE 0,38, confidenza 1,00 anche quando sbagliano). Dopo l'addestramento sul compito diventa la migliore del gruppo.
5. **I modelli escono prima dell'insegnante a regole** (circa 60 s contro 82) **perché lo imitano in modo imperfetto e raccolgono meno**. L'insegnante non ottimizza il tempo di uscita: sul solo tempo, 60 s non è il tetto del banco.
6. **La generalizzazione a mappe nuove non è misurabile su questo banco.** Su MAP02 non esce nessuno, **neanche l'insegnante a regole**: la catena regole + esecutore + stato si blocca davanti a una **porta a chiave gialla** (verificato nel WAD), perché la chiave è lontana e le porte a chiave vengono offerte come apribili. Per misurare davvero la generalizzazione servirebbero un insieme di azioni che sappia cercare le chiavi e un sensore che non offra porte che non si possono aprire; solo dopo ha senso chiedersi se i modelli imitano l'insegnante anche lì.

## 7. Limiti (da dire nell'articolo)

- Addestramento e validazione solo su **MAP01**. MAP02 (3 semi) si è rivelata irrisolvibile dall'intera catena, oracolo compreso; altre mappe non sono state provate.
- Le etichette sono di un **insegnante a regole**, non di partite umane: i modelli imparano a imitare quelle regole, e il confronto misura quanto bene lo fanno.
- **130 domande** di validazione: sulla calibrazione ogni metà ne ha 65, e i decimali vanno presi con cautela.
- **Laya v3** è stato addestrato da altri, in 3 fasi e con un metodo diverso dal nostro. Non abbiamo cercato di migliorarlo.
- I numeri di **Jev** vengono da doomLaya e non sono riprodotti (API a pagamento).
- La **latenza "diretta"** del 270M è misurata con un nostro script PyTorch, non con un server di produzione. Quella di Laya passa dal suo server HTTP: il confronto è prudente a favore di Laya.
- Il ponte legge solo i **primi 20 logprob**. Le opzioni fuori da quella lista ricevono una probabilità minima (1e-4). Non ha avuto effetti su accuratezza e calibrazione, ma va detto.
- La **prima partita di E2B a freddo** su MAP01 usava il prompt con le due domande insieme. Le misure successive usano domande separate, come in addestramento.
- **Gemma 3 270M** viene dalla copia di unsloth (`model.safetensors` sha256 `700b710a…`). L'originale di Google richiede un account e non abbiamo potuto confrontare le impronte.
- Tutti i tempi di partita sono **tempo di gioco**; la partita scorre in tempo reale.

## 8. Materiale disponibile

- `media/doomLaya_vs_doomGemma.gif`: 5 s affiancati, MAP01 seme 48, secondi 4,5–9,5, con il pannello delle probabilità.
- `results/`: risultati grezzi (calibrazione, latenza, partite MAP01/MAP02, oracolo, Laya base, storia dell'addestramento).
- `src/`: tutti gli script; `patches/doomLaya-agent.patch`: le 6 righe cambiate in `agent.py` di doomLaya (5 aggiunte, 1 modificata) (`--model llm` e `--model oracolo`).
- Adattatori LoRA su Hugging Face: [dexmac/doomgemma-e2b-lora](https://huggingface.co/dexmac/doomgemma-e2b-lora) e [dexmac/doom-gemma3-270m-lora](https://huggingface.co/dexmac/doom-gemma3-270m-lora).
