# Dwukierunkowa uwaga skrośna w fuzji obrazu histopatologicznego z danymi klinicznymi

Kod źródłowy do manuskryptu *„Dwukierunkowa uwaga skrośna nie realizuje interakcji* *międzymodalnej: analiza degeneracji architektonicznej i dekompozycja wariancji* *w fuzji obrazu histopatologicznego z danymi klinicznymi"*.

Autor: Tomasz Pruś (badacz niezależny) Licencja kodu: MIT · Licencja manuskryptu i rycin: CC BY 4.0

## Czego dotyczy praca

Trzy główne wyniki, w kolejności siły dowodu:

1. **Degeneracja architektoniczna.** Kierunek uwagi skrośnej, w którym obraz pełni rolę zapytania, a wektor kliniczny klucza i wartości, jest matematycznie zdegenerowany: przy jednoelementowym kluczu softmax przyjmuje tożsamościowo wartość 1, więc wyjście modułu jest **identyczne co do bitu dla dowolnie różnych** **obrazów**. Dowód analityczny, potwierdzony numerycznie.

2. **Fuzja późna przewyższa fuzję wczesną.** Uśrednienie rangowe predykcji dwóch modeli jednomodalnych (AUC 0,772) jest jedynym wariantem bijącym wszystkie pozostałe po korekcie Holma.

3. **Dekompozycja wariancji.** Na 450 wytrenowanych modelach: podział danych odpowiada za 86,2% zmienności AUC, inicjalizacja wag za 11,5%, architektura za 2,3%. Wariancja inicjalizacji przewyższa architektoniczną 5,1-krotnie.

## Wymagania

```
pip install -r requirements.txt
```

**Uwaga dla Windows:** `openslide-python` wymaga dodatkowo binariów OpenSlide. Pobierz z [https://openslide.org/download/](https://openslide.org/download/), rozpakuj i dodaj katalog `bin` do zmiennej `PATH`. Bez tego krok 0 nie zadziała (pozostałe kroki tak).

Obliczenia wykonano na CPU. Buforowanie reprezentacji obrazu (krok 2) sprawia, że kroki 3–4 trwają minuty, nie godziny, także bez akceleratora.

## Dane

Nie są dołączone — pobierz samodzielnie z Genomic Data Commons:

- **Dane kliniczne:** `nationwidechildrens.org\\\_clinical\\\_patient\\\_brca.txt` → umieść w katalogu `CLINICAL\\\_DIR`

- **Preparaty:** pliki `.svs` projektu TCGA-BRCA → umieść w `WSI\\\_DIR`

Ścieżki ustawia się w jednym miejscu: zmienna `TCGA\\\_ROOT` w `config.py`.

## Kolejność uruchamiania

Każdy krok wypisuje na końcu nazwę następnego.

```
\\\# KROK 0 — kafelkowanie preparatów (wymaga OpenSlide; kilka godzin)    
python step0\\\_preprocess\\\_wsi.py --method patches    
    
\\\# KROK 0b — uzgodnienie schematu tabeli mapowania    
python step0b\\\_normalize\\\_mapping.py    
    
\\\# KROK 1b — filtr materiału wg barcode'u TCGA (odrzuca tkankę prawidłową)    
python step1b\\\_filter\\\_slides.py    
    
\\\# KROK 1 — budowa zbioru danych klinicznych    
python step1\\\_build\\\_dataset.py --mapping "\\\<OUTPUT\\\_DIR\\\>\\\\wsi\\\_processed\\\_mapping\\\_FILTERED.csv"    
    
\\\# KROK 2 — przeliczenie reprezentacji obrazu (zamrożony EfficientNetB0; kilka godzin)    
python step2\\\_precompute\\\_embeddings.py --mapping "\\\<OUTPUT\\\_DIR\\\>\\\\wsi\\\_processed\\\_mapping\\\_FILTERED.csv"    
    
\\\# KROK 3 — trening i ewaluacja: 450 modeli (~1 h)    
python step3\\\_train\\\_eval\\\_cv.py --target er --deterministic --seeds 3 ^    
       --arms tab img concat concat\\\_wide cross cross\\\_a2b --tag final --save-preds    
    
\\\# KROK 4 — fuzja późna i zgodność modalności    
python step4\\\_late\\\_fusion.py --preds "\\\<OUTPUT\\\_DIR\\\>\\\\cv\\\_results\\\\cv\\\_preds\\\_er\\\_final.csv" --out figures    
    
\\\# RYCINY I TABELE    
python make\\\_fig1\\\_fig2.py --out figures --dpi 600    
python make\\\_figures.py --results cv\\\_results --out figures --dpi 600
```

Kroki 0 i 2 obsługują `--resume`. Kroki 3 i 4 przyjmują `--lang en` dla rycin z podpisami angielskimi.

## Kolejność uruchamiania — wersja skrócona

| Krok | Skrypt | Wyjście |
| - | - | - |
| 0 | `step0\\\_preprocess\\\_wsi.py` | kafelki PNG + `image\\\_mapping.csv` |
| 0b | `step0b\\\_normalize\\\_mapping.py` | `wsi\\\_processed\\\_mapping.csv` |
| 1b | `step1b\\\_filter\\\_slides.py` | `wsi\\\_processed\\\_mapping\\\_FILTERED.csv` |
| 1 | `step1\\\_build\\\_dataset.py` | `dataset\\\_FIXED.csv`, `feature\\\_columns.csv` |
| 2 | `step2\\\_precompute\\\_embeddings.py` | `cache/embeddings.npy` (~1,3 GB) |
| 3 | `step3\\\_train\\\_eval\\\_cv.py` | `cv\\\_folds\\\_\\\*.csv`, `cv\\\_preds\\\_\\\*.csv` |
| 4 | `step4\\\_late\\\_fusion.py` | ryciny 7–9, tabele S4–S6 |
| — | `make\\\_fig1\\\_fig2.py`, `make\\\_figures.py` | ryciny 1–6, tabele S1–S3 |


## Odtwarzalność

Flaga `--deterministic` w kroku 3 włącza `tf.config.experimental.enable\\\_op\\\_determinism()`, ustawia `TF\\\_DETERMINISTIC\\\_OPS` i `TF\\\_CUDNN\\\_DETERMINISTIC` **przed importem TensorFlow** oraz ogranicza wykonanie do jednego wątku. Zweryfikowano: dwa niezależne uruchomienia pełnego eksperymentu dały wyniki identyczne co do bitu dla wszystkich 450 modeli.

Bez tej flagi powtórzenia różnią się na poziomie pojedynczych foldów nawet o 0,20 AUC — co samo w sobie jest jednym z wyników opisanych w pracy (rozdz. 3.6).

**Odtwarzalność potwierdzono dla wersji pakietów przypiętych w `requirements.txt`.** Inne wersje TensorFlow mogą dawać inne wartości.

**`cv\\\_preds\\\_er\\\_final.csv`** — predykcje na poziomie pacjenta dla wszystkich 450 modeli. Pozwala odtworzyć analizę fuzji późnej (`step4\\\_late\\\_fusion.py`) bez ponownego uruchamiania kroku 3.

## Uwagi metodyczne wbudowane w kod

Kod zawiera zabezpieczenia wynikające z błędów wykrytych w trakcie prac. Warto o nich wiedzieć przed modyfikacją:

- **Test wycieku etykiety** (krok 1): drzewo decyzyjne trenowane na samych cechach musi wypaść *poniżej* poziomu klasy większościowej. Wynik powyżej 0,95 oznacza wyciek i zatrzymuje pipeline.

- **Kontrola przypisania pacjent ↔ preparat** (krok 1b): identyfikator odczytany z nazwy pliku `.svs` porównywany z tabelą mapowania; jakakolwiek niezgodność przerywa wykonanie.

- **Wykluczenie tkanki prawidłowej** (krok 1b): bez tego filtra model uczy się przewidywać status receptorów guza z obrazów zdrowej tkanki (12,5% kafelków w surowej kohorcie TCGA-BRCA).

- **Rozstrzyganie HER2 wg ASCO/CAP** (krok 1): wynik FISH ma pierwszeństwo przed IHC, przypadki niejednoznaczne są wykluczane, a nie przypisywane arbitralnie.

- **Ewaluacja na poziomie pacjenta** (krok 3): agregacja przed obliczeniem metryk. Ewaluacja per kafelek zawyża wyniki i zaniża wariancję.

- **Zabezpieczenie plików wynikowych** (krok 3): plik o niezgodnym schemacie kolumn jest archiwizowany, a nie nadpisywany. Nie chroni to jednak przed sklejeniem dwóch przebiegów o **tym samym** schemacie — używaj różnych wartości `--tag`.

## Struktura katalogu wyjściowego

```
Pipeline\\\_Output/    
├── processed\\\_images/            kafelki PNG (krok 0)    
├── cache/    
│   ├── embeddings.npy           bufor reprezentacji, float16    
│   └── embeddings\\\_index.csv    
├── cv\\\_results/    
│   ├── cv\\\_folds\\\_er\\\_final.csv    metryki, 450 wierszy    
│   └── cv\\\_preds\\\_er\\\_final.csv    predykcje na poziomie pacjenta    
├── figures/                     ryciny 1–9, tabele S1–S6    
├── wsi\\\_processed\\\_mapping.csv    
├── wsi\\\_processed\\\_mapping\\\_FILTERED.csv    
├── dataset\\\_FIXED.csv    
├── feature\\\_columns.csv    
└── table\\\_material\\\_composition.csv
```

## Cytowanie

```
Software:

Pruś T. Dwukierunkowa uwaga skrośna nie realizuje interakcji międzymodalnej:    
analiza degeneracji architektonicznej i dekompozycja wariancji w fuzji obrazu    
histopatologicznego z danymi klinicznymi. 2026. https://doi.org/10.5281/zenodo.22129327
```

## Kontakt

[prus.tomasz.1972@gmail.com](mailto:prus.tomasz.1972@gmail.com)

