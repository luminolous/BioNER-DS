# Multi-Entity Biomedical NER with Distant Supervision:
Comparing Training Strategies for Schema Expansion
toward Virus-Centric Drug Repurposing

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
| Virus | SARS-CoV-2, dengue virus | **Tidak** |
| Gene / Mutation | ACE2, E484K, spike protein | **Tidak** |

Gap inilah yang jadi masalah utama dan sekaligus peluang kontribusi riset kita.

---

## 3. Rumusan Masalah

Ada tiga pertanyaan utama yang ingin dijawab riset ini:

**1.** Bagaimana cara memperluas skema entitas NER biomedis dari dua tipe (Chemical dan Disease) menjadi empat tipe (tambah Virus dan Gene/Mutation) tanpa memerlukan anotasi manual dalam skala besar yang butuh waktu dan biaya sangat besar?

**2.** Dari tiga strategi penggabungan data berlabel (gold-standard) dengan data pseudolabel (silver-standard), mana yang paling efektif untuk task perluasan skema entitas NER biomedis ini?

**3.** Apakah perluasan skema entitas NER ini menghasilkan Knowledge Graph yang lebih komprehensif dan representatif untuk skenario drug repurposing berbasis virus dibanding sistem NER yang hanya pakai dua tipe entitas?

---

## 4. Tujuan Riset

Tujuan riset ini dibagi jadi tiga lapisan yang hierarkis:

### Tujuan Utama
Membangun dan mengevaluasi model NER biomedis yang mampu mengenali empat tipe entitas (Chemical, Disease, Virus, Gene/Mutation) dari abstrak jurnal ilmiah PubMed, dengan memanfaatkan kombinasi data berlabel BC5CDR dan corpus tanpa label yang dianotasi otomatis via distant supervision.

### Tujuan Sekunder
Melakukan studi komparatif sistematis terhadap tiga strategi training yang berbeda untuk menggabungkan data gold-standard dan silver-standard, yaitu Sequential Fine-tuning, Joint Uniform Training, dan Joint Noise-Aware Training.

### Tujuan Tersier
Mendemonstrasikan bahwa NER empat entitas yang lebih lengkap menghasilkan Knowledge Graph yang lebih kaya secara statistik, sebagai bukti kualitatif bahwa kontribusi teknis riset ini punya dampak nyata di downstream application drug repurposing.

---

## 5. Dataset yang Digunakan

Penting untuk dipahami dulu bahwa ada **tiga hal berbeda** yang namanya mirip tapi fungsinya beda sama sekali supaya tidak bingung:

```
1. ncbi/ncbi_disease (HuggingFace)
   --> Dataset NER berlabel siap pakai (gold-standard)
   --> Sama statusnya dengan BC5CDR, tinggal download
   --> Bukan hasil scraping

2. NCBI Taxonomy
   --> Database klasifikasi nama virus dari NCBI
   --> Bukan dataset NER, tapi kamus untuk distant supervision
   --> Dipakai sebagai alat auto-labeling, bukan data training langsung

3. Scraped PubMed Corpus
   --> Ini yang beneran hasil scraping via Entrez API
   --> Tidak berlabel sama sekali saat pertama kali diambil
   --> Dilabeli otomatis pakai kamus NCBI Taxonomy dan HGNC
   --> Hasilnya jadi silver-standard dataset
```

Riset ini menggunakan **dua jenis dataset** dengan peran yang berbeda:

### Dataset Tipe 1: Gold-Standard (Berlabel Manual)

**BC5CDR (BioCreative V Chemical-Disease Relation Corpus)**

- Sumber: HuggingFace `bigbio/bc5cdr`
- Isi: 1.500 abstrak PubMed yang sudah dianotasi secara manual oleh pakar
- Entitas: Chemical dan Disease
- Juga punya anotasi relasi Chemical-Disease
- Jumlah kalimat: sekitar 15.000 kalimat (train/val/test)
- Kualitas: sangat tinggi karena buatan manusia (gold-standard)
- Peran dalam riset: **Fondasi utama training NER** untuk entitas Chemical dan Disease

Format datanya seperti ini:
```
Tokens : ["Aspirin", "reduces", "fever", "in", "dengue", "patients"]
Labels : [B-Chemical, O, B-Disease, O, B-Disease, O]
```

### Dataset Tipe 2: Silver-Standard (Berlabel Otomatis via Distant Supervision)

**Scraped PubMed Corpus**

- Sumber: NCBI Entrez API (scraping aktif menggunakan Biopython)
- Isi: 10.000 sampai 15.000 abstrak jurnal tentang virus yang di-scraping berdasarkan query
- Entitas: Virus dan Gene/Mutation (dianotasi otomatis)
- Kualitas: lebih rendah dari BC5CDR karena ada noise dari proses anotasi otomatis
- Peran dalam riset: **Sumber label untuk entitas Virus dan Gene** yang tidak ada di BC5CDR

Kenapa perlu scraping? Karena:
1. Ketentuan final project mensyaratkan scraping dan pseudolabeling
2. Tidak ada dataset berlabel manual yang cukup besar untuk entitas Virus dan Gene
3. Corpus yang di-scraping ini adalah satu-satunya cara untuk mendapatkan sinyal training bagi dua entitas baru tersebut

### Cara Kerja Distant Supervision (Pseudolabeling)

Distant supervision adalah teknik anotasi otomatis yang menggunakan database / kamus yang sudah ada sebagai "anotator":

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

Contoh hasil distant supervision:
```
Kalimat : "The SARS-CoV-2 spike protein binds to ACE2 receptor"
Labels  : [B-Virus, I-Virus, B-Gene, I-Gene, O, O, B-Gene, O]
```

### Ringkasan Peran Semua Sumber Data

| Sumber | Jenis | Cara Dapat | Entitas | Peran |
|---|---|---|---|---|
| BC5CDR | Gold-standard | Download HuggingFace | Chemical + Disease | Fondasi utama training |
| ncbi/ncbi_disease | Gold-standard | Download HuggingFace | Disease | Penguat Disease (opsional) |
| Scraped PubMed | Silver-standard | Scraping Entrez API | Virus + Gene | Sumber entitas baru |
| NCBI Taxonomy | Kamus (bukan dataset) | Download NCBI | Nama virus | Alat distant supervision |
| HGNC | Kamus (bukan dataset) | Download HGNC | Nama gen | Alat distant supervision |

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
TAHAP 4: TRAINING NER (7 Konfigurasi)
    Baseline (Kfg 1-3) + Kontribusi Utama (Kfg 4-6)
                         |
                         v
TAHAP 5: EVALUASI NER
    F1, Precision, Recall per tipe entitas
                         |
                         v
TAHAP 6: KNOWLEDGE GRAPH CONSTRUCTION
    Ekstrak entitas + relasi --> simpan di Neo4j
                         |
                         v
TAHAP 7: ANALISIS DAN DEMONSTRASI
    Statistik KG + Case study drug repurposing
```

---

## 7. Model dan Teknologi

### Model NLP (Backbone)

Tiga model yang dipakai dalam eksperimen, masing-masing dengan karakteristik berbeda:

**BERT-base-uncased**
- Model generalis yang dilatih pada Wikipedia dan BookCorpus
- Tidak punya pengetahuan khusus tentang teks biomedis
- Peran: lower-bound baseline, pembanding paling dasar

**BioBERT-base**
- BERT yang di-continue pre-training pada PubMed abstracts dan PMC full-text
- Punya pemahaman dasar tentang kosakata biomedis
- Peran: domain-specific baseline yang sudah umum dipakai

**PubMedBERT**
- Dilatih dari scratch HANYA menggunakan PubMed, tanpa mixed-domain
- Representasi paling bersih untuk teks biomedis
- Peran: backbone utama untuk eksperimen kontribusi (konfigurasi 4 sampai 6)

### Library dan Tools

| Tool | Fungsi |
|---|---|
| HuggingFace Datasets | Load BC5CDR |
| Biopython (Entrez) | Scraping PubMed |
| SciSpacy | Preprocessing teks biomedis |
| HuggingFace Transformers | Training model NER |
| seqeval | Evaluasi NER (F1, Precision, Recall) |
| Neo4j | Penyimpanan dan visualisasi Knowledge Graph |

---

## 8. Desain Eksperimen: 7 Konfigurasi

Eksperimen dibagi jadi dua dimensi yang berbeda:

### Dimensi 1: Perbandingan Backbone (Konfigurasi 1 sampai 3)

Semua pakai data BC5CDR saja, yang beda cuma model backbone-nya. Tujuannya untuk membuktikan bahwa domain-specific pretraining itu penting dan sekaligus menentukan backbone terbaik untuk dipakai di eksperimen berikutnya.

```
Konfigurasi 1: BERT-base      + BC5CDR --> NER Chemical + Disease
Konfigurasi 2: BioBERT        + BC5CDR --> NER Chemical + Disease
Konfigurasi 3: PubMedBERT     + BC5CDR --> NER Chemical + Disease
```

Hasil yang diharapkan: PubMedBERT > BioBERT > BERT-base

### Dimensi 2: Perbandingan Strategi Training (Konfigurasi 4 sampai 6)

Semua pakai PubMedBERT (terbaik dari dimensi 1), yang beda adalah cara menggabungkan data BC5CDR dan silver data. Ini adalah **inti kontribusi utama riset**.

```
Konfigurasi 4: PubMedBERT + Sequential Fine-tuning
               Fase 1: Train BC5CDR  --> Chemical + Disease
               Fase 2: Train Silver  --> Virus + Gene
               (risiko: bisa lupa entitas dari fase 1)

Konfigurasi 5: PubMedBERT + Joint Uniform Training
               Gabung BC5CDR + Silver, latih sekaligus
               Bobot loss semua sample sama rata

Konfigurasi 6: PubMedBERT + Joint Noise-Aware Training
               Gabung BC5CDR + Silver, latih sekaligus
               BC5CDR (gold) --> bobot loss 1.0
               Silver        --> bobot loss 0.3
               (model "tahu" bahwa data silver lebih noisy)
```

### Matriks Eksperimen Lengkap

| Kfg | Backbone | Data Training | Entitas | Tujuan |
|---|---|---|---|---|
| 1 | BERT-base | BC5CDR | Chem + Dis | Lower bound |
| 2 | BioBERT | BC5CDR | Chem + Dis | Domain baseline |
| 3 | PubMedBERT | BC5CDR | Chem + Dis | Best backbone |
| 4 | PubMedBERT | BC5CDR + Silver, Sequential | 4 entitas | Kontribusi v1 |
| 5 | PubMedBERT | BC5CDR + Silver, Joint Uniform | 4 entitas | Kontribusi v2 |
| 6 | PubMedBERT | BC5CDR + Silver, Noise-Aware | 4 entitas | Kontribusi v3 |

---

## 9. Evaluasi

### Untuk NER

Metrik standar yang dipakai di task NER:

- **Precision**: dari semua entitas yang diprediksi model, berapa persen yang beneran benar?
- **Recall**: dari semua entitas yang seharusnya ada, berapa persen yang berhasil ditemukan model?
- **F1-Score**: harmonic mean dari Precision dan Recall, ini yang jadi metrik utama

Evaluasi dilakukan **per tipe entitas** supaya ketahuan model bagus di mana dan lemah di mana:

```
Contoh output evaluasi:
Entity      Precision  Recall  F1-Score
Chemical    0.xx       0.xx    0.xx
Disease     0.xx       0.xx    0.xx
Virus       0.xx       0.xx    0.xx   <-- Entitas baru!
Gene        0.xx       0.xx    0.xx   <-- Entitas baru!
```

Test set yang dipakai:
- Untuk Chemical dan Disease: test set BC5CDR yang sudah ada
- Untuk Virus dan Gene: held-out set kecil yang dianotasi manual oleh tim (200 sampai 300 kalimat)

### Untuk Knowledge Graph

Evaluasi KG dilakukan secara kuantitatif dengan membandingkan dua KG: satu dari model baseline (konfigurasi 3) dan satu dari model terbaik (salah satu dari konfigurasi 4 sampai 6):

- Jumlah node per tipe entitas
- Jumlah triplet relasi yang berhasil diekstrak
- Coverage: berapa persen virus di corpus berhasil terhubung ke minimal satu entitas Chemical/obat

---

## 10. Knowledge Graph

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
1. Jalankan model NER terbaik pada corpus PubMed baru
   (yang tidak dipakai waktu training)
         |
         v
2. Untuk setiap kalimat, ambil semua pasangan entitas
   yang berbeda tipe dalam kalimat yang sama
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

1. **Bukti kualitatif** bahwa NER empat entitas menghasilkan representasi pengetahuan yang lebih kaya dibanding NER dua entitas
2. **Demonstrasi aplikasi** yang menunjukkan use case nyata dari sistem yang dibangun
3. **Error analysis** untuk melihat kesalahan sistematis dari model NER secara visual
4. **Case study** yang bisa ditampilkan di paper sebagai ilustrasi drug repurposing potensial

---

## 11. Novelty Riset

Novelty riset ini masuk ke dua kategori: **Empirical Novelty** dan **Combination Novelty**. Tidak ada method novelty di sini (kita tidak menciptakan algoritma baru dari nol), dan itu bukan masalah selama framing-nya tepat dan jujur.

---

### Novelty 1: Studi Komparatif Strategi Training untuk Schema Expansion (Terkuat)

Ini adalah kontribusi empiris paling solid yang kita punya.

**Klaimnya:**
> Kami melakukan studi empiris yang membandingkan secara sistematis tiga strategi penggabungan data gold-standard (BC5CDR) dan silver-standard (distant-supervised corpus) untuk task perluasan skema entitas NER biomedis, yaitu Sequential Fine-tuning, Joint Uniform Training, dan Joint Noise-Aware Training. Studi ini menghasilkan temuan konkret tentang strategi mana yang paling efektif untuk kasus schema expansion dari dua entitas ke empat entitas.

**Kenapa ini valid sebagai novelty?**

Pertanyaan "dari ketiga strategi ini mana yang terbaik untuk kasus schema expansion spesifik ini?" tidak bisa dijawab dari teori saja, harus lewat eksperimen. Temuannya sendiri yang jadi kontribusi. Ini mirip dengan paper yang membandingkan optimizer atau learning rate scheduler pada task tertentu, hasilnya tetap berguna dan bisa dikutip peneliti lain.

**Cara menulisnya di paper:**
```
We present a systematic empirical comparison of three training
strategies for combining gold-standard and distant-supervised
silver-standard data in biomedical NER schema expansion,
providing practical guidelines for future researchers facing
similar multi-source training settings.
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
    schema expansion.

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

Distant supervision pasti menghasilkan false positive (sesuatu yang bukan virus tapi dilabeli virus karena cocok nama) dan false negative (nama virus yang tidak ada di kamus NCBI Taxonomy). Ini adalah keterbatasan yang harus didokumentasikan dengan jujur di paper, bukan disembunyikan.

**Ambiguitas Nama Entitas**

Banyak nama gen yang juga punya arti lain dalam bahasa sehari-hari. Misalnya, "CAT" adalah gen tapi juga nama hewan. Distant supervision naif akan mislabeli banyak kasus seperti ini.

**Catastrophic Forgetting di Sequential Training**

Pada konfigurasi 4 (sequential), ada risiko bahwa model lupa cara mengenali Chemical dan Disease setelah dilatih pada data Virus dan Gene. Ini harus diukur dan dilaporkan.

### Batasan Riset

- KG yang dihasilkan belum divalidasi secara biologis oleh pakar virologi
- Relasi antar entitas didapat dari co-occurrence sederhana, bukan relasi yang benar-benar dipahami secara semantik
- Evaluasi drug repurposing dilakukan secara kualitatif (case study), bukan kuantitatif dengan uji klinis

---

## 13. Skenario Aplikasi: Drug Repurposing

Meskipun bukan tujuan utama yang diukur, ini adalah motivasi besar yang membuat riset ini relevan. Berikut skenario konkret bagaimana KG yang dihasilkan bisa dipakai:

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

Ini yang disebut **multi-hop reasoning** di atas Knowledge Graph. Dalam riset ini, kita tidak sampai mengimplementasikan reasoning otomatis, tapi KG yang kita bangun sudah menyediakan fondasi datanya.

---

## 14. Output yang Diharapkan

Di akhir riset ini, ada beberapa output yang akan dihasilkan:

**Output Teknis:**
- Model NER terlatih untuk 4 tipe entitas biomedis
- Silver-standard dataset (corpus PubMed yang sudah dianotasi otomatis via distant supervision)
- Knowledge Graph yang tersimpan di Neo4j

**Output Akademis:**
- Paper yang mendokumentasikan seluruh eksperimen dan hasil
- Tabel perbandingan F1-score untuk 6 konfigurasi
- Analisis kualitatif KG dan case study drug repurposing

---

---

## 15. Ringkasan Satu Paragraf (untuk Abstract)

Riset ini mengusulkan pipeline directed distant supervision untuk memperluas skema entitas NER biomedis dari dua tipe (Chemical dan Disease) menjadi empat tipe dengan menambahkan Virus dan Gene/Mutation, menggunakan BC5CDR sebagai fondasi gold-standard dan corpus abstrak PubMed hasil scraping yang dianotasi otomatis via NCBI Taxonomy dan HGNC sebagai silver-standard. Studi komparatif sistematis dilakukan terhadap tiga strategi penggabungan data gold dan silver yaitu Sequential Fine-tuning, Joint Uniform Training, dan Joint Noise-Aware Training, menggunakan PubMedBERT sebagai backbone utama yang dibandingkan dengan BioBERT dan BERT-base sebagai baseline. Temuan eksperimen memberikan panduan praktis tentang strategi training mana yang paling efektif untuk task schema expansion dalam domain biomedis. Model terbaik kemudian digunakan untuk mengonstruksi Knowledge Graph di Neo4j, dan analisis kuantitatif KG digunakan untuk membuktikan bahwa perluasan skema entitas menghasilkan representasi pengetahuan yang lebih kaya dan lebih relevan untuk skenario drug repurposing berbasis virus.

---