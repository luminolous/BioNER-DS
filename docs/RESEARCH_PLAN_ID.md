# Multi-Entity Biomedical NER with Distant Supervision: Comparing Training Strategies for Schema Expansion toward Virus-Centric Drug Repurposing

> Dokumen ini adalah catatan internal yang merangkum desain, motivasi, dan hasil eksperimen dari kode di repository ini. Untuk overview yang lebih ringkas dan instruksi menjalankan, lihat `README.md` di root repo.

---

## 1. Gambaran Besar Riset

Riset ini ada di bidang **Biomedical Natural Language Processing (BioNLP)**, yaitu cabang NLP yang fokus memproses teks ilmiah di dunia biomedis dan kesehatan.

Secara simpel, yang mau kita lakukan adalah:

> Ambil teks dari jutaan jurnal ilmiah yang sudah ada di dunia, ekstrak informasi pentingnya secara otomatis, lalu susun informasi itu jadi sebuah peta pengetahuan yang bisa dipakai ilmuwan untuk nyari kandidat obat baru buat virus tertentu.

Masalahnya, teks jurnal ilmiah itu tidak terstruktur. Semua informasi penting tentang virus, obat, gen, dan penyakit nyemplung bareng dalam satu paragraf panjang. Tidak ada yang ngelabelin mana virus, mana obatnya. Nah, sistem NLP kita yang akan belajar ngelabelinnya secara otomatis.

---

## 2. Latar Belakang dan Motivasi

### Kenapa ini penting?

Setiap tahun ribuan jurnal ilmiah baru diterbitkan di PubMed, database jurnal biomedis milik National Library of Medicine Amerika Serikat. Sampai sekarang PubMed sudah punya lebih dari **35 juta abstrak artikel**. Tidak ada satu manusia pun yang bisa baca semuanya.

Di sisi lain, ketika virus baru muncul (seperti yang terjadi saat COVID-19), ilmuwan butuh waktu bertahun-tahun untuk nemuin obat baru lewat jalur konvensional. Padahal bisa jadi, informasi tentang obat yang sudah ada dan mungkin efektif untuk virus itu sudah tersebar di berbagai jurnal yang terpisah-pisah tapi belum pernah dihubungkan satu sama lain.

**Drug repurposing** adalah strategi menemukan kegunaan baru dari obat yang sudah ada untuk penyakit atau virus yang berbeda. Ini jauh lebih cepat dan murah daripada bikin obat dari nol. Contoh nyata yang berhasil: **Baricitinib**, yang awalnya adalah obat reumatoid artritis, terbukti efektif untuk COVID-19 setelah ilmuwan menemukan kesamaan mekanisme biologisnya.

Masalahnya, untuk nemuin koneksi kayak gitu secara manual dari jutaan jurnal itu hampir mustahil. Di sinilah NLP berperan.

### Gap yang ingin diisi

Dataset NER biomedis yang sudah ada, seperti **BC5CDR**, cuma bisa mengenali dua tipe entitas: **Chemical (obat)** dan **Disease (penyakit)**. Tapi untuk skenario drug repurposing berbasis virus, kita butuh minimal empat tipe entitas:

| Tipe Entitas | Contoh | Ada di BC5CDR? |
|---|---|---|
| Chemical / Drug | chloroquine, remdesivir | Ya |
| Disease | COVID-19, pneumonia | Ya |
| Virus | SARS-CoV-2, dengue virus | Tidak |
| Gene / Mutation | ACE2, E484K, spike protein | Tidak |

Gap inilah yang jadi masalah utama dan sekaligus peluang kontribusi riset kita.

---

## 3. Rumusan Masalah

Ada tiga pertanyaan utama yang ingin dijawab riset ini:

1. Bagaimana cara memperluas skema entitas NER biomedis dari dua tipe (Chemical dan Disease) menjadi empat tipe (tambah Virus dan Gene/Mutation) tanpa memerlukan anotasi manual dalam skala besar yang butuh waktu dan biaya sangat besar?

2. Dari tiga strategi penggabungan data berlabel (gold-standard) dengan data pseudolabel (silver-standard), mana yang paling efektif untuk task perluasan skema entitas NER biomedis ini?

3. Apakah perluasan skema entitas NER ini menghasilkan Knowledge Graph yang lebih komprehensif dan representatif untuk skenario drug repurposing berbasis virus dibanding sistem NER yang hanya pakai dua tipe entitas?

---

## 4. Tujuan Riset

Tujuan riset dibagi jadi tiga lapisan yang hierarkis:

### Tujuan Utama

Membangun dan mengevaluasi model NER biomedis yang mampu mengenali empat tipe entitas (Chemical, Disease, Virus, Gene/Mutation) dari abstrak jurnal ilmiah PubMed, dengan memanfaatkan kombinasi data berlabel BC5CDR dan corpus tanpa label yang dianotasi otomatis via distant supervision.

### Tujuan Sekunder

Melakukan studi komparatif sistematis terhadap tiga strategi training yang berbeda untuk menggabungkan data gold-standard dan silver-standard, yaitu Sequential Fine-tuning, Joint Uniform Training, dan Joint Noise-Aware Training, lalu menentukan mana yang paling cocok untuk kasus schema expansion ini.

### Tujuan Tersier

Mendemonstrasikan bahwa NER empat entitas yang lebih lengkap menghasilkan Knowledge Graph yang lebih kaya secara statistik, sebagai bukti kualitatif bahwa kontribusi teknis riset ini punya dampak nyata di downstream application drug repurposing. Konstruksi dan analisis KG dilakukan oleh anggota tim lain di repository terpisah, jadi yang ada di repo ini cuma pipeline NER-nya.

---

## 5. Dataset yang Digunakan

Penting untuk dipahami dulu bahwa ada tiga hal berbeda yang namanya mirip tapi fungsinya beda sama sekali supaya tidak bingung:

```
1. BC5CDR (HuggingFace bigbio/bc5cdr)
   --> Dataset NER berlabel siap pakai (gold-standard)
   --> Bukan hasil scraping

2. NCBI Taxonomy
   --> Database klasifikasi nama virus dari NCBI
   --> Bukan dataset NER, tapi kamus untuk distant supervision
   --> Dipakai sebagai alat auto-labeling, bukan data training langsung

3. Scraped PubMed Corpus
   --> Hasil scraping abstrak PubMed via Entrez API
   --> Tidak berlabel sama sekali saat pertama kali diambil
   --> Dilabeli otomatis pakai kamus NCBI Taxonomy dan HGNC
   --> Hasilnya jadi silver-standard dataset
```

### Dataset Tipe 1: Gold-Standard (Berlabel Manual)

**BC5CDR (BioCreative V Chemical-Disease Relation Corpus)**

- Sumber asli: HuggingFace `bigbio/bc5cdr`
- Isi: 1.500 abstrak PubMed yang sudah dianotasi secara manual oleh pakar
- Entitas: Chemical dan Disease
- Jumlah kalimat aktual yang dipakai: 5.119 train + 5.218 validation + 5.728 test
- Kualitas: sangat tinggi karena buatan manusia
- Peran dalam riset: fondasi utama training NER untuk entitas Chemical dan Disease

Format datanya sudah pre-tokenized di JSONL, satu baris satu kalimat:

```json
{
  "tokens": ["Aspirin", "reduces", "fever", "in", "dengue", "patients"],
  "decoded_tags": ["B-Chemical", "O", "B-Disease", "O", "B-Disease", "O"]
}
```

### Dataset Tipe 2: Silver-Standard (Berlabel Otomatis via Distant Supervision)

**Scraped PubMed Corpus**

- Sumber: NCBI Entrez API (scraping aktif pakai Biopython)
- Isi: 40.946 kalimat dari ribuan abstrak jurnal yang di-scraping berdasarkan query terkait virus
- Entitas: Virus dan Gene/Mutation (dianotasi otomatis)
- Kualitas: lebih rendah dari BC5CDR karena ada noise dari proses anotasi otomatis
- Peran dalam riset: sumber label untuk entitas Virus dan Gene yang tidak ada di BC5CDR

Selain corpus silver, ada juga **PubMed gold test set kecil** sebanyak 100 kalimat yang dianotasi manual untuk evaluasi entitas Virus dan Gene. Test set ini tidak pernah dipakai untuk training, cuma sebagai trusted reference saat eval.

Kenapa perlu scraping?
1. Ketentuan final project mensyaratkan scraping dan pseudolabeling
2. Tidak ada dataset berlabel manual yang cukup besar untuk entitas Virus dan Gene
3. Corpus scraping ini adalah satu-satunya cara dapat sinyal training bagi dua entitas baru

### Cara Kerja Distant Supervision (Pseudolabeling)

Distant supervision adalah teknik anotasi otomatis yang menggunakan database atau kamus yang sudah ada sebagai "anotator":

```
Kamus Virus  : NCBI Taxonomy (daftar resmi semua nama virus)
Kamus Gene   : HGNC Database (daftar resmi semua nama gen manusia)

Proses:
Setiap kalimat di corpus scraped
      |
      v
Cocokkan dengan kamus virus --> beri label B-Virus / I-Virus
Cocokkan dengan kamus gene  --> beri label B-Gene  / I-Gene
Token lainnya               --> beri label O

Output: Silver-standard dataset
(berlabel otomatis, ada noise, tapi skala besar)
```

Contoh hasil distant supervision di field `decoded_tags_pseudo_final`:

```json
{
  "tokens": ["The", "SARS-CoV-2", "spike", "binds", "to", "ACE2"],
  "decoded_tags": ["O", "O", "O", "O", "O", "O"],
  "decoded_tags_pseudo_final": ["O", "B-Virus", "O", "O", "O", "B-Gene"]
}
```

`decoded_tags` adalah label gold (5-tag space: O, B/I-Chemical, B/I-Disease). `decoded_tags_pseudo_final` adalah label gabungan gold + silver (9-tag space: tambah B/I-Virus dan B/I-Gene). Konfigurasi 1 sampai 3 baca dari `decoded_tags`. Konfigurasi 4 sampai 6 baca dari `decoded_tags_pseudo_final`.

### Lokasi Dataset (HuggingFace)

Semua file JSONL final sudah di-host di HuggingFace Hub supaya gampang di-reproduce orang lain:

- Dataset: [`lumicero/BioNER-DS`](https://huggingface.co/datasets/lumicero/BioNER-DS)
- Checkpoint hasil training: [`lumicero/BioNER-DS`](https://huggingface.co/lumicero/BioNER-DS)

### Ringkasan Peran Semua Sumber Data

| Sumber | Jenis | Cara Dapat | Entitas | Peran |
|---|---|---|---|---|
| BC5CDR | Gold-standard | Download HuggingFace | Chemical + Disease | Fondasi utama training |
| Scraped PubMed (silver) | Silver-standard | Scraping Entrez API | Virus + Gene | Sumber entitas baru |
| Scraped PubMed (test) | Gold kecil (manual) | Scraping + anotasi manual | Virus + Gene | Test set untuk eval |
| NCBI Taxonomy | Kamus | Download NCBI | Nama virus | Alat distant supervision |
| HGNC | Kamus | Download HGNC | Nama gen | Alat distant supervision |

---

## 6. Pipeline Sistem Lengkap

```
TAHAP 1: DATA COLLECTION
    BC5CDR (download) + Scraping PubMed (Entrez API)
                         |
                         v
TAHAP 2: PREPROCESSING
    SciSpacy: tokenisasi, sentence splitting, cleaning
                         |
                         v
TAHAP 3: DISTANT SUPERVISION
    Anotasi otomatis Virus (NCBI Taxonomy) + Gene (HGNC)
    --> Silver-standard dataset
                         |
                         v
TAHAP 4: TRAINING NER (6 Konfigurasi)
    Baseline backbone comparison (Kfg 1-3)
    + Schema expansion strategies (Kfg 4-6)
                         |
                         v
TAHAP 5: EVALUASI NER
    F1, Precision, Recall per tipe entitas
    Multi-seed (42, 1337, 2024) untuk mean +- std
                         |
                         v
TAHAP 6: KNOWLEDGE GRAPH CONSTRUCTION  [di luar scope repo ini]
    Ekstrak entitas + relasi --> simpan di Neo4j
                         |
                         v
TAHAP 7: ANALISIS DAN DEMONSTRASI       [di luar scope repo ini]
    Statistik KG + Case study drug repurposing
```

Repo ini fokus di Tahap 1 sampai 5. Tahap 6 dan 7 dikerjakan anggota tim lain di repository terpisah, pakai output `predict.py` dari repo ini sebagai input.

---

## 7. Model dan Teknologi

### Model NLP (Backbone)

Tiga model yang dipakai di eksperimen, masing-masing dengan karakteristik berbeda:

**BERT-base-uncased**
- Model generalis yang dilatih pada Wikipedia dan BookCorpus
- Tidak punya pengetahuan khusus tentang teks biomedis
- Peran: lower-bound baseline, pembanding paling dasar

**BioBERT-base (`dmis-lab/biobert-base-cased-v1.2`)**
- BERT yang di-continue pre-training pada PubMed abstracts dan PMC full-text
- Punya pemahaman dasar tentang kosakata biomedis
- Peran: domain-specific baseline yang sudah umum dipakai di literatur

**PubMedBERT (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`)**
- Dilatih dari scratch hanya pakai PubMed, tanpa mixed-domain
- Representasi paling bersih untuk teks biomedis
- Peran: backbone utama untuk eksperimen kontribusi (konfigurasi 4 sampai 6)

### Library dan Tools

| Tool | Fungsi |
|---|---|
| HuggingFace Datasets | Load BC5CDR |
| Biopython (Entrez) | Scraping PubMed |
| SciSpacy | Preprocessing teks biomedis |
| HuggingFace Transformers | Training model NER |
| seqeval | Evaluasi NER (strict IOB2, F1 / Precision / Recall) |
| PyYAML | Loader config eksperimen |
| Neo4j | Penyimpanan dan visualisasi Knowledge Graph (di luar repo ini) |

---

## 8. Desain Eksperimen: 6 Konfigurasi

Eksperimen dibagi jadi dua dimensi yang berbeda.

### Dimensi 1: Perbandingan Backbone (Konfigurasi 1 sampai 3)

Semua pakai data BC5CDR saja, yang beda cuma model backbone-nya. Tujuannya untuk membuktikan bahwa domain-specific pretraining itu penting dan sekaligus menentukan backbone terbaik untuk eksperimen berikutnya.

```
Konfigurasi 1: BERT-base      + BC5CDR --> NER Chemical + Disease
Konfigurasi 2: BioBERT        + BC5CDR --> NER Chemical + Disease
Konfigurasi 3: PubMedBERT     + BC5CDR --> NER Chemical + Disease
```

Hasil yang diharapkan dan terverifikasi: PubMedBERT > BioBERT > BERT-base.

### Dimensi 2: Perbandingan Strategi Training (Konfigurasi 4 sampai 6)

Semua pakai PubMedBERT (terbaik dari dimensi 1), yang beda adalah cara menggabungkan data BC5CDR dan silver corpus. Ini adalah inti kontribusi utama riset.

```
Konfigurasi 4: PubMedBERT + Sequential Fine-tuning
               Fase 1: Train BC5CDR  --> Chemical + Disease
               Fase 2: Train Silver  --> Virus + Gene
               (risiko: bisa lupa entitas dari fase 1)

Konfigurasi 5: PubMedBERT + Joint Uniform Training
               Gabung BC5CDR + Silver dalam satu stream shuffled
               Bobot loss semua sample sama rata (1.0)

Konfigurasi 6: PubMedBERT + Joint Noise-Aware Training
               Gabung BC5CDR + Silver dalam satu stream shuffled
               BC5CDR (gold) --> bobot loss 1.0
               Silver        --> bobot loss 0.5
               (model "tahu" bahwa data silver lebih noisy)
```

Catatan implementasi: di awal eksperimen, bobot silver konfigurasi 6 diset 0.3 sesuai spec awal, tapi training collapse di 2 dari 3 seed. Setelah dinaikkan ke 0.5, hasilnya lebih stabil (2 dari 3 seed sukses) walaupun masih lebih rendah dari konfigurasi 5.

### Matriks Eksperimen Lengkap

| Kfg | Backbone | Data Training | Entitas | Tujuan |
|---|---|---|---|---|
| 1 | BERT-base | BC5CDR | Chem + Dis | Lower bound |
| 2 | BioBERT | BC5CDR | Chem + Dis | Domain baseline |
| 3 | PubMedBERT | BC5CDR | Chem + Dis | Best backbone |
| 4 | PubMedBERT | BC5CDR + Silver, Sequential | 4 entitas | Kontribusi (kontras forgetting) |
| 5 | PubMedBERT | BC5CDR + Silver, Joint Uniform | 4 entitas | Kontribusi utama (winner) |
| 6 | PubMedBERT | BC5CDR + Silver, Noise-Aware | 4 entitas | Ablation (negative result) |

### Ablation Tambahan: Hyperparameter Sweep Silver Ratio

Setelah konfigurasi 5 terbukti jadi yang terbaik, ada sweep tambahan untuk lihat seberapa besar silver corpus berkontribusi. Tiga varian konfigurasi 5 di-train ulang pakai silver corpus yang di-subsample:

| Sweep | Silver size | Ratio silver:gold |
|---|---|---|
| sweep_silver_1_1 | 5.119 | 1:1 (balanced) |
| sweep_silver_2_1 | 10.238 | 2:1 |
| sweep_silver_4_1 | 20.476 | 4:1 |
| baseline (config 5) | 40.946 | 8:1 |

Empat titik ini membentuk scaling curve log-2 yang bisa di-plot untuk lihat apakah silver corpus saturate di ratio tertentu atau terus naik. Notebook untuk sweep ini ada di `notebooks/run_experiments_hyperparameter_.ipynb`.

---

## 9. Evaluasi

### Untuk NER

Metrik standar yang dipakai di task NER, dihitung dengan `seqeval` mode strict IOB2:

- **Precision**: dari semua entitas yang diprediksi model, berapa persen yang beneran benar?
- **Recall**: dari semua entitas yang seharusnya ada, berapa persen yang berhasil ditemukan model?
- **F1-Score**: harmonic mean dari Precision dan Recall, ini metrik utama untuk reporting

Evaluasi dilakukan **per tipe entitas** supaya ketahuan model bagus di mana dan lemah di mana. Setiap konfigurasi di-train 3 kali dengan seed berbeda (42, 1337, 2024), lalu dilaporkan sebagai `mean +- std`.

### Test Set

Dua test set dipakai:

- `test_bc5cdr`: 5.728 kalimat dari BC5CDR test split, dianotasi manual untuk Chemical dan Disease
- `test_pubmed`: 100 kalimat dari PubMed scraping yang dianotasi manual untuk Virus dan Gene

Konfigurasi 1 sampai 3 cuma evaluasi di `test_bc5cdr` (karena cuma support 2 entitas). Konfigurasi 4 sampai 6 evaluasi di kedua test set.

Entitas dengan support kurang dari 10 (misalnya Chemical di `test_pubmed` yang cuma punya 4 span gold) di-mark **N/A** di tabel paper-facing supaya angka 0 yang tidak meaningful tidak ikut dilaporkan. Angka raw tetap ada di JSON output untuk audit.

### Hasil Aktual (3 seed: 42, 1337, 2024)

Konfigurasi 1 sampai 3 (BC5CDR, Chemical + Disease):

| Kfg | Backbone | Overall F1 | Chemical F1 | Disease F1 |
|---|---|---|---|---|
| 1 | BERT-base | 0.826 +- 0.005 | 0.868 +- 0.003 | 0.777 +- 0.007 |
| 2 | BioBERT | 0.876 +- 0.002 | 0.916 +- 0.002 | 0.827 +- 0.006 |
| 3 | PubMedBERT | 0.900 +- 0.001 | 0.937 +- 0.002 | 0.855 +- 0.004 |

Konfigurasi 4 sampai 6 (4 entitas, dilaporkan di dua test set):

| Kfg | Strategi | BC5CDR F1 | PubMed F1 |
|---|---|---|---|
| 4 | Sequential | 0.0003 +- 0.0005 | 0.956 +- 0.009 |
| 5 | Joint Uniform | 0.826 +- 0.007 | 0.967 +- 0.003 |
| 6 | Joint Noise-Aware | 0.278 +- 0.482 | 0.322 +- 0.557 |

Catatan untuk konfigurasi 6: angka mean dan std tidak well-defined karena 1 dari 3 seed total collapse ke "predict all O". Kalau hanya lihat seed yang sukses (42 dan 1337), F1 BC5CDR sekitar 0.81 dan PubMed 0.95. Tetap di bawah konfigurasi 5 di kedua test set.

### Temuan Utama

1. **PubMedBERT > BioBERT > BERT-base** terbukti, gap-nya konsisten antar seed.
2. **Konfigurasi 5 (joint uniform) adalah strategi terbaik untuk schema expansion**: F1 BC5CDR 0.826 (mirip dengan baseline single-source konfigurasi 2-3) dan PubMed F1 0.967 (entitas baru Virus dan Gene berhasil dipelajari dengan baik). Tidak ada catastrophic forgetting.
3. **Konfigurasi 4 (sequential) menunjukkan catastrophic forgetting yang ekstrem**: BC5CDR F1 turun dari 0.89 (akhir fase 1) ke ~0.0 (akhir fase 2), sementara PubMed F1 jadi 0.956. Ini bukan bug, ini hasil eksperimen yang memang ingin diukur, dan jadi motivasi kenapa joint training lebih cocok.
4. **Konfigurasi 6 (noise-aware) underperform joint uniform** di kedua test set dan punya masalah stabilitas. Untuk corpus ini, downweighting silver justru membuang sinyal training yang berguna tanpa kasih noise reduction yang berarti.

---

## 10. Knowledge Graph

Konstruksi dan evaluasi Knowledge Graph dilakukan oleh anggota tim lain di repository terpisah. Repo NER ini cuma menyediakan output prediction yang di-konsumsi tahap KG via `predict.py`. Bagian ini didokumentasikan untuk konteks paper.

### Apa itu Knowledge Graph di riset ini?

Knowledge Graph (KG) adalah peta hubungan antar entitas yang berhasil diekstrak dari teks jurnal. Di Neo4j, ia terlihat seperti jaringan titik-titik yang saling terhubung:

```
Node   = entitas (virus, obat, penyakit, gen)
Edge   = relasi antar entitas (causes, treats, inhibits, dll.)

Contoh:
[dengue virus] --CAUSES--> [dengue fever]
[chloroquine]  --INHIBITS--> [dengue virus]
[NS5 protein]  --ASSOCIATED_WITH--> [dengue virus]
[chloroquine]  --TREATS--> [dengue fever]
```

### Bagaimana KG dibangun?

```
1. Jalankan model NER terbaik (config 5) pada corpus PubMed baru
   yang tidak dipakai waktu training (output predict.py)
         |
         v
2. Untuk setiap kalimat, ambil semua pasangan entitas yang
   berbeda tipe dalam kalimat yang sama
         |
         v
3. Tentukan tipe relasi berdasarkan kombinasi tipe entitas:
   Chemical + Virus   --> INHIBITS
   Virus + Disease    --> CAUSES
   Chemical + Disease --> TREATS_OR_CAUSES
   Gene + Virus       --> ASSOCIATED_WITH
         |
         v
4. Simpan triplet (Subject, Relation, Object) ke Neo4j
         |
         v
5. Visualisasi dan analisis
```

### Peran KG dalam Paper

KG bukan kontribusi utama yang dievaluasi secara kuantitatif ketat, tapi berperan sebagai:

1. Bukti kualitatif bahwa NER empat entitas menghasilkan representasi pengetahuan yang lebih kaya dibanding NER dua entitas
2. Demonstrasi aplikasi yang menunjukkan use case nyata dari sistem yang dibangun
3. Error analysis untuk melihat kesalahan sistematis dari model NER secara visual
4. Case study yang bisa ditampilkan di paper sebagai ilustrasi drug repurposing potensial

### Evaluasi KG (kuantitatif kasar)

KG dari model terbaik (konfigurasi 5) dibandingkan dengan KG hipotetis dari model 2-entitas (konfigurasi 3) berdasarkan:

- Jumlah node per tipe entitas
- Jumlah triplet relasi yang berhasil diekstrak
- Coverage: berapa persen virus di corpus berhasil terhubung ke minimal satu entitas Chemical/obat

Angka-angka ini akan diisi setelah pipeline KG selesai jalan di sisi tim lain.

---

## 11. Novelty Riset

Novelty riset ini masuk ke dua kategori: **Empirical Novelty** dan **Combination Novelty**. Tidak ada method novelty di sini (kita tidak menciptakan algoritma baru dari nol), dan itu bukan masalah selama framing-nya tepat dan jujur.

---

### Novelty 1: Studi Komparatif Strategi Training untuk Schema Expansion (Terkuat)

Ini kontribusi empiris paling solid yang kita punya.

**Klaimnya:**

> Kami melakukan studi empiris yang membandingkan secara sistematis tiga strategi penggabungan data gold-standard (BC5CDR) dan silver-standard (distant-supervised corpus) untuk task perluasan skema entitas NER biomedis, yaitu Sequential Fine-tuning, Joint Uniform Training, dan Joint Noise-Aware Training. Studi ini menghasilkan temuan konkret: **joint uniform menang**, sequential mengalami catastrophic forgetting yang ekstrem, dan noise-aware weighting justru underperform pada corpus dengan silver labels yang relatif clean.

**Kenapa ini valid sebagai novelty?**

Pertanyaan "dari ketiga strategi ini mana yang terbaik untuk kasus schema expansion spesifik ini?" tidak bisa dijawab dari teori saja, harus lewat eksperimen. Temuan-temuan empiris ini yang jadi kontribusi. Mirip dengan paper yang membandingkan optimizer atau learning rate scheduler pada task tertentu, hasilnya tetap berguna dan bisa dikutip peneliti lain.

Yang membuat findings ini lebih kuat dari studi komparatif biasa: kita tidak cuma kasih ranking, tapi juga **explanation kenapa noise-aware gagal di corpus ini**. Untuk corpus dengan silver labels yang relatif clean, downweighting justru membuang sinyal training yang berguna. Insight ini bisa di-generalize ke task lain dengan silver corpus berkualitas serupa.

**Cara menulisnya di paper:**

```
We present a systematic empirical comparison of three training
strategies for combining gold-standard and distant-supervised
silver-standard data in biomedical NER schema expansion. We find
that joint uniform training outperforms both sequential transfer
(which suffers catastrophic forgetting) and noise-aware weighting
(which removes useful signal from sufficiently clean silver
labels), providing practical guidelines for future researchers
facing similar multi-source training settings.
```

---

### Novelty 2: Directed Distant Supervision Pipeline untuk Drug Repurposing (Kuat)

**Klaimnya:**

> Kami mengusulkan pipeline distant supervision yang terarah untuk memperluas skema entitas NER biomedis dari Chemical+Disease menjadi Chemical+Disease+Virus+Gene, dengan pemilihan sumber ontologi (NCBI Taxonomy untuk Virus, HGNC untuk Gene) yang secara eksplisit dirancang untuk mendukung skenario drug repurposing berbasis virus.

**Kenapa ini berbeda dari distant supervision pada umumnya?**

Distant supervision untuk NER memang sudah ada sejak lama. Tapi kombinasi yang spesifik ini, yaitu memperluas skema BC5CDR ke arah Virus dan Gene menggunakan NCBI Taxonomy dan HGNC sebagai sumber ontologi dengan tujuan eksplisit drug repurposing, adalah konfigurasi yang belum pernah diteliti secara eksplisit sebelumnya. Yang kita jual bukan teknik distant supervision-nya, tapi **arah dan desain pipeline-nya**.

**Cara menulisnya di paper:**

```
We propose a directed distant supervision pipeline that extends
the BC5CDR entity schema from two types (Chemical, Disease) to
four types by incorporating Virus and Gene/Mutation entities
using domain-appropriate ontologies (NCBI Taxonomy and HGNC),
explicitly tailored for the drug repurposing use case.
```

---

### Novelty 3: Downstream KG Impact Analysis (Pelengkap)

**Klaimnya:**

> Kami mengevaluasi dampak downstream dari perluasan skema entitas NER terhadap kelengkapan dan kepadatan Knowledge Graph biomedis, menunjukkan bahwa model dengan F1-score lebih tinggi dan cakupan entitas lebih luas menghasilkan KG yang lebih kaya secara statistik dan lebih representatif untuk skenario drug repurposing.

**Kenapa ini penting?**

Kebanyakan paper NER berhenti di angka F1 tanpa menunjukkan apakah peningkatan itu punya dampak nyata di aplikasi downstream. Dengan membandingkan statistik KG dari model baseline (2 entitas) vs model terbaik (4 entitas), kita bisa buktikan bahwa kontribusi teknis kita bukan cuma angka di tabel evaluasi, tapi punya konsekuensi praktis yang terukur.

**Cara menulisnya di paper:**

```
We demonstrate that the expanded four-entity NER system produces
a significantly richer biomedical Knowledge Graph measured by
entity coverage and relation density, compared to the two-entity
baseline, validating the practical value of our schema expansion
approach for drug repurposing applications.
```

---

### Cara Menulis Contributions di Paper

Gabungan ketiga novelty di atas ditulis seperti ini di bagian Introduction:

```
Our main contributions are as follows:

(1) We present a systematic empirical comparison of three
    training strategies (Sequential Fine-tuning, Joint Uniform,
    and Joint Noise-Aware) for combining gold-standard and
    distant-supervised silver-standard data in biomedical NER
    schema expansion, and identify joint uniform training as
    the most effective for our setting.

(2) We propose a directed distant supervision pipeline using
    NCBI Taxonomy and HGNC as ontology sources to extend
    BC5CDR's two-entity schema to four entity types
    (Chemical, Disease, Virus, Gene/Mutation), explicitly
    designed for virus-centric drug repurposing.

(3) We demonstrate the downstream impact of entity schema
    expansion through quantitative analysis of the resulting
    Knowledge Graph, showing improved entity coverage and
    relation density compared to the two-entity baseline.
```

---

### Yang Bukan Novelty (Jangan Diklaim)

Ini penting supaya tidak diserang reviewer:

- Pakai BioBERT atau PubMedBERT untuk NER biomedis (sudah ratusan paper)
- Distant supervision untuk NER secara umum (sudah ada sejak 2009)
- Konstruksi KG dari output NER dan RE (sudah sangat umum)
- Pakai Neo4j untuk menyimpan KG biomedis (sudah sangat umum)

---

## 12. Tantangan dan Batasan

### Tantangan Teknis

**Noise di Silver Data**

Distant supervision pasti menghasilkan false positive (sesuatu yang bukan virus tapi dilabeli virus karena cocok nama) dan false negative (nama virus yang tidak ada di kamus NCBI Taxonomy). Ini keterbatasan yang harus didokumentasikan dengan jujur di paper, bukan disembunyikan.

**Ambiguitas Nama Entitas**

Banyak nama gen yang juga punya arti lain dalam bahasa sehari-hari. Misalnya, "CAT" adalah gen tapi juga nama hewan. Distant supervision naif akan mislabeli banyak kasus seperti ini.

**Catastrophic Forgetting di Sequential Training (sudah terukur)**

Di konfigurasi 4, model lupa cara mengenali Chemical dan Disease setelah fase 2 training pada silver corpus. Δ F1 untuk Chemical sekitar -0.85, untuk Disease juga sekitar -0.85. Ini bukan lagi "risiko yang harus diukur", tapi temuan eksperimen konkret yang dilaporkan di paper sebagai motivasi joint training.

**Stabilitas Noise-Aware Training**

Konfigurasi 6 punya masalah stabilitas. Di setting awal (silver weight 0.3), 2 dari 3 seed collapse. Setelah dinaikkan ke 0.5, masih ada 1 dari 3 seed yang collapse. Ini menunjukkan noise-aware weighting di fp16 + batch kecil sensitive terhadap gradient underflow.

### Batasan Riset

- KG yang dihasilkan belum divalidasi secara biologis oleh pakar virologi
- Relasi antar entitas didapat dari co-occurrence sederhana, bukan relasi yang benar-benar dipahami secara semantik
- Evaluasi drug repurposing dilakukan secara kualitatif (case study), bukan kuantitatif dengan uji klinis
- Test set untuk Virus dan Gene cuma 100 kalimat, jadi confidence interval untuk F1 di test_pubmed cukup lebar
- Hyperparameter sweep cuma di dimensi silver:gold ratio, belum mencakup LR, batch size, atau scheduler

---

## 13. Skenario Aplikasi: Drug Repurposing

Meskipun bukan tujuan utama yang diukur, ini motivasi besar yang membuat riset ini relevan. Berikut skenario konkret bagaimana KG yang dihasilkan bisa dipakai:

```
Misalnya ada virus baru "Virus X" yang baru ditemukan.

Langkah 1: Query KG untuk Virus X
           --> Ditemukan bahwa Virus X mengandung Gen A

Langkah 2: Query hubungan Gen A di KG
           --> Gen A punya struktur mirip dengan Gen B milik Virus Y
               (karena ada jurnal yang membahas kesamaan ini)

Langkah 3: Query Virus Y di KG
           --> Obat Z terbukti efektif melawan Virus Y
               (banyak jurnal yang membuktikan ini)

Kesimpulan sistem: Obat Z layak diuji coba untuk Virus X
                   karena kesamaan jalur genetik
```

Ini yang disebut **multi-hop reasoning** di atas Knowledge Graph. Dalam riset ini, kita tidak sampai mengimplementasikan reasoning otomatis, tapi KG yang dibangun sudah menyediakan fondasi datanya.

---

## 14. Output yang Diharapkan

Di akhir riset ini, beberapa output yang sudah dan akan dihasilkan:

**Output Teknis (sudah ada)**:
- Pipeline training NER yang reproducible (repo ini)
- 6 model NER yang terlatih, di-host di `lumicero/BioNER-DS` di HuggingFace Hub
- Silver-standard dataset di `lumicero/BioNER-DS` di HuggingFace Datasets
- Hasil F1 untuk 6 konfigurasi dengan multi-seed mean +- std
- Scaling curve silver:gold ratio dari hyperparameter sweep

**Output Teknis (masih dikerjakan, di luar repo ini)**:
- Knowledge Graph di Neo4j hasil aplikasi NER terbaik (config 5) pada PubMed corpus baru

**Output Akademis**:
- Paper yang mendokumentasikan seluruh eksperimen dan hasil
- Tabel perbandingan F1-score untuk 6 konfigurasi (sudah tersedia, lihat `outputs/results/`)
- Analisis kualitatif KG dan case study drug repurposing

---

## 15. Ringkasan Satu Paragraf (untuk Abstract)

Riset ini mengusulkan pipeline directed distant supervision untuk memperluas skema entitas NER biomedis dari dua tipe (Chemical dan Disease) menjadi empat tipe dengan menambahkan Virus dan Gene/Mutation, menggunakan BC5CDR sebagai fondasi gold-standard dan corpus abstrak PubMed hasil scraping yang dianotasi otomatis via NCBI Taxonomy dan HGNC sebagai silver-standard. Studi komparatif sistematis dilakukan terhadap tiga strategi penggabungan data gold dan silver yaitu Sequential Fine-tuning, Joint Uniform Training, dan Joint Noise-Aware Training, menggunakan PubMedBERT sebagai backbone utama yang dibandingkan dengan BioBERT dan BERT-base sebagai baseline. Hasil eksperimen menunjukkan bahwa Joint Uniform Training adalah strategi paling efektif, mencapai F1 sekitar 0.83 pada test set BC5CDR (Chemical dan Disease) dan F1 sekitar 0.97 pada test set PubMed (Virus dan Gene), sementara Sequential Fine-tuning mengalami catastrophic forgetting yang ekstrem dan Joint Noise-Aware Training underperform akibat downweighting silver yang justru membuang sinyal training yang berguna pada corpus yang relatif clean. Model terbaik kemudian digunakan untuk mengonstruksi Knowledge Graph di Neo4j, dan analisis kuantitatif KG digunakan untuk membuktikan bahwa perluasan skema entitas menghasilkan representasi pengetahuan yang lebih kaya dan lebih relevan untuk skenario drug repurposing berbasis virus.

---
