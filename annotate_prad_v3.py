"""
전립선암 표적치료제 표적 유전자 주석 추가 (v3용)
node_importance_v3.tsv -> node_importance_v3_annotated.tsv

규칙:
  - 전립선암 표적치료제 표적 유전자: [PRAD target] 접두사 + 약물명 포함
  - TMPRSS2-ERG fusion 전립선암(ergPRAD) 특화: (ergPRAD) 접두사
  - drug 노드는 annotation 없음 (gene만 대상)
"""

import pandas as pd
import os

BASE = r"C:/Users/user/Desktop/CNN_CVPR/__DATAAn__/random_work_test"
TSV_IN  = os.path.join(BASE, "node_importance_v3.tsv")
TSV_OUT = os.path.join(BASE, "node_importance_v3_annotated.tsv")

# ──────────────────────────────────────────────────────────────────────────────
# 주석 딕셔너리
# 형식: 'GENE': (is_ergPRAD: bool, 'annotation text with drugs')
# is_ergPRAD=True → "(ergPRAD) " 접두사 자동 부착
# ──────────────────────────────────────────────────────────────────────────────
PRAD_ANNOTATIONS = {

    # ── ERG / TMPRSS2-ERG fusion axis ────────────────────────────────────────
    "ERG": (True,
        "ETS transcription factor; TMPRSS2-ERG fusion driver ~50% PRAD; "
        "transcriptional target in ergPRAD; "
        "Drugs: indirect via BET inhibitors (JQ1, OTX015, ABBV-744), "
        "EZH2 inhibitors (tazemetostat, GSK126)"),
    "TMPRSS2": (True,
        "Androgen-regulated serine protease; 5' fusion partner of ERG in ergPRAD; "
        "AR-driven expression links AR signaling to ERG overexpression; "
        "Drugs: enzalutamide, apalutamide (suppress AR-driven TMPRSS2)"),
    "ETV1": (False,
        "ETS transcription factor; TMPRSS2-ETV1 fusion in ~1% PRAD; "
        "Drugs: MEK inhibitors (trametinib, cobimetinib)"),
    "ETV4": (False,
        "ETS transcription factor; rare TMPRSS2-ETV4 fusion in PRAD; "
        "Drugs: MEK inhibitors"),
    "ETV5": (False,
        "ETS transcription factor; rare ETS fusion in PRAD; "
        "Drugs: MEK inhibitors (investigational)"),
    "ETV6": (False,
        "ETS transcription factor; rare rearrangement in PRAD; "
        "Drugs: investigational ETS inhibitors"),
    "SPDEF": (True,
        "ETS transcription factor; prostate-specific; AR co-activator; "
        "upregulated in ERG-fusion PRAD; "
        "Drugs: AR inhibitors (enzalutamide, apalutamide) indirectly"),

    # ── Androgen Receptor (AR) axis ───────────────────────────────────────────
    "AR": (False,
        "Androgen receptor; primary PRAD driver; "
        "Drugs: enzalutamide (Xtandi), apalutamide (Erleada), "
        "darolutamide (Nubeqa) [ARSi]; abiraterone acetate (Zytiga) [CYP17A1i]; "
        "leuprolide, degarelix [GnRH agonist/antagonist]"),
    "FOXA1": (False,
        "Pioneer transcription factor; AR co-activator; frequently mutated mCRPC; "
        "Drugs: AR inhibitors (enzalutamide) indirectly suppress FOXA1 activity"),
    "HOXB13": (False,
        "AR co-activator; germline G84E mutation increases PRAD risk; "
        "Drugs: AR inhibitors (enzalutamide, apalutamide) indirectly"),
    "NKX3-1": (False,
        "Prostate-specific tumor suppressor; AR target gene; "
        "Drugs: no direct inhibitor; AR-axis therapy relevant"),
    "KLK3": (False,
        "PSA; AR transcriptional target; PRAD biomarker; "
        "Drugs: AR inhibitors suppress KLK3/PSA expression"),
    "KLK2": (False,
        "Kallikrein-2; AR target; PRAD biomarker; "
        "Drugs: AR inhibitors"),
    "SRD5A1": (False,
        "5alpha-reductase type 1; converts testosterone to DHT; "
        "Drugs: dutasteride (Avodart) [dual 5ARi]"),
    "SRD5A2": (False,
        "5alpha-reductase type 2; primary DHT-producing enzyme; "
        "Drugs: finasteride (Proscar), dutasteride (Avodart) [5ARi]"),
    "CYP17A1": (False,
        "Steroidogenesis enzyme (pregnenolone to DHEA); "
        "Drugs: abiraterone acetate (Zytiga) [FDA-approved mCRPC]; "
        "seviteronel, orteronel (investigational)"),
    "CYP19A1": (False,
        "Aromatase; estrogen biosynthesis; involved in CRPC androgen metabolism; "
        "Drugs: exemestane, anastrozole, letrozole [aromatase inhibitors]"),
    "HSD3B1": (False,
        "3beta-HSD; adrenal androgen to DHT conversion in CRPC; "
        "Drugs: abiraterone resistance via A367T mutation; "
        "EPI-7386 (investigational HSD3B1 inhibitor)"),
    "HSD17B3": (False,
        "17beta-HSD3; testosterone biosynthesis; "
        "Drugs: investigational HSD17B inhibitors in CRPC"),
    "NCOR1": (True,
        "AR co-repressor; loss promotes AR activity in ergPRAD; "
        "Drugs: enzalutamide resistance biomarker; "
        "no direct drug; AR inhibitors context"),
    "NCOR2": (False,
        "AR co-repressor (SMRT); loss drives AR constitutive activity; "
        "Drugs: enzalutamide resistance biomarker; next-gen ARSi (darolutamide)"),

    # ── PI3K / AKT / mTOR pathway ─────────────────────────────────────────────
    "PTEN": (True,
        "Tumor suppressor phosphatase; PI3K/AKT negative regulator; "
        "deleted in ~40% PRAD, ~60% mCRPC; "
        "Drugs: ipatasertib (AKTi, PTEN-loss + ARSi context), "
        "capivasertib (AZD5363), GSK2636771 (PI3Kbeta-i in PTEN-null); "
        "synthetic lethality with AR inhibition"),
    "PIK3CA": (False,
        "Catalytic subunit of PI3K-alpha; mutated/amplified in PRAD; "
        "Drugs: alpelisib (Piqray) [PI3Kalpha-i], copanlisib, taselisib"),
    "PIK3CB": (True,
        "PI3Kbeta catalytic subunit; dominant isoform in PTEN-null ergPRAD; "
        "Drugs: GSK2636771, AZD8186 [PI3Kbeta-selective inhibitors]; "
        "clinical trials ongoing in PTEN-null PRAD"),
    "PIK3R1": (False,
        "PI3K regulatory subunit p85alpha; mutations in mCRPC; "
        "Drugs: pan-PI3K inhibitors (buparlisib, copanlisib)"),
    "AKT1": (False,
        "AKT serine/threonine kinase; central PI3K effector; "
        "Drugs: ipatasertib (Roche, FDA-approved context), "
        "capivasertib (AZD5363, AstraZeneca); clinical trials in mCRPC"),
    "AKT2": (False,
        "AKT isoform 2; PI3K/AKT signaling; "
        "Drugs: pan-AKT inhibitors (ipatasertib, capivasertib)"),
    "AKT3": (False,
        "AKT isoform 3; upregulated in neuroendocrine PRAD (NEPC); "
        "Drugs: pan-AKT inhibitors"),
    "MTOR": (False,
        "mTOR kinase; downstream of AKT; "
        "Drugs: everolimus (Afinitor) [mTORC1i], temsirolimus (Torisel); "
        "dual PI3K/mTOR inhibitors: gedatolisib, sapanisertib (INK128)"),
    "TSC1": (False,
        "mTOR negative regulator (hamartin); "
        "Drugs: everolimus, temsirolimus (mTOR inhibitors)"),
    "TSC2": (False,
        "mTOR negative regulator (tuberin); "
        "Drugs: everolimus, temsirolimus (sensitized in TSC2-mutant)"),
    "RICTOR": (False,
        "mTORC2 component; regulates AKT Ser473 phosphorylation; "
        "amplified in PRAD/NEPC; "
        "Drugs: dual mTORC1/2 inhibitors (sapanisertib, vistusertib)"),
    "RPTOR": (False,
        "mTORC1 scaffold; target of rapamycin; "
        "Drugs: everolimus, temsirolimus, sirolimus [rapalogs]"),

    # ── DNA Damage Response / HRR ─────────────────────────────────────────────
    "BRCA2": (False,
        "HRR gene; germline/somatic mutations in ~13% mCRPC; "
        "Drugs: olaparib (Lynparza, FDA-approved mCRPC+BRCA), "
        "rucaparib (Rubraca), niraparib (Zejula), "
        "talazoparib (Talzenna); platinum-based chemotherapy"),
    "BRCA1": (False,
        "HRR gene; mutations in ~3% mCRPC; "
        "Drugs: olaparib (Lynparza), rucaparib (Rubraca) [PARP inhibitors]"),
    "ATM": (False,
        "HRR/DSB repair kinase; mutations in ~7% mCRPC; "
        "Drugs: olaparib (ATM included in FDA label), "
        "AZD0156, AZD1390 [ATM inhibitors, clinical trials]"),
    "CDK12": (False,
        "Transcription-coupled DNA repair kinase; biallelic loss ~7% mCRPC; "
        "Drugs: dinaciclib, THZ531, NVP-2 [CDK12 inhibitors]; "
        "immunotherapy combinations (CDK12-loss = tandem duplicator phenotype)"),
    "PALB2": (False,
        "BRCA2 partner; HRR; mutations sensitize to PARP inhibitors; "
        "Drugs: olaparib, rucaparib [PARP inhibitors per label]"),
    "RAD51": (False,
        "HRR recombinase; core effector; overexpression = PARPi resistance; "
        "Drugs: RAD51 inhibitors (RI-1, B02, CYT-0851 investigational)"),
    "RAD51C": (False,
        "RAD51 paralog; HRR; mutations sensitize to PARP inhibitors; "
        "Drugs: olaparib, niraparib [PARP inhibitors per biomarker panels]"),
    "RAD51D": (False,
        "RAD51 paralog; HRR; "
        "Drugs: olaparib, rucaparib [PARP inhibitor sensitivity]"),
    "BRIP1": (False,
        "BRCA1-interacting helicase; HRR; mutations in mCRPC; "
        "Drugs: olaparib (FDA label includes BRIP1)"),
    "BARD1": (False,
        "BRCA1-binding protein; HRR; "
        "Drugs: PARP inhibitors (BARD1 in emerging biomarker panels)"),
    "NBN": (False,
        "Nibrin; DSB sensing (MRN complex); mutations in mCRPC; "
        "Drugs: PARP inhibitors (NBN in emerging panels)"),
    "CHEK2": (False,
        "DNA damage checkpoint kinase; germline mutations in ~2% PRAD; "
        "Drugs: olaparib (CHEK2 in expanded use); "
        "CHK1/2 inhibitors: prexasertib, MK-1775"),
    "CHEK1": (False,
        "Checkpoint kinase 1; replication stress response; "
        "Drugs: prexasertib (LY2606368) [CHK1i]; "
        "LY2603618; combinations with PARP inhibitors in mCRPC"),
    "ATR": (False,
        "Replication stress kinase; "
        "Drugs: ceralasertib (AZD6738) [ATRi]; "
        "berzosertib (M6620, VX-970); combinations with olaparib in mCRPC"),
    "FANCL": (False,
        "Fanconi anemia E3 ubiquitin ligase; HRR; mutations in mCRPC; "
        "Drugs: PARP inhibitors (Fanconi pathway = HRR biomarker)"),
    "FANCA": (False,
        "Fanconi anemia complementation group A; "
        "Drugs: PARP inhibitors, platinum-based chemotherapy (carboplatin)"),
    "FANCD2": (False,
        "Fanconi anemia core; ICL repair; "
        "Drugs: platinum-based chemotherapy; PARP inhibitors"),
    "FANCG": (False,
        "Fanconi anemia group G; HRR; "
        "Drugs: PARP inhibitors; cisplatin/carboplatin"),
    "RAD54L": (False,
        "HRR chromatin remodeler; RAD51 cofactor; altered in mCRPC; "
        "Drugs: PARP inhibitors (HRR deficiency context)"),
    "PARP1": (False,
        "Poly(ADP-ribose) polymerase 1; BER/SSBR; direct PARP inhibitor target; "
        "Drugs: olaparib (Lynparza), rucaparib (Rubraca), "
        "niraparib (Zejula), talazoparib (Talzenna) [FDA-approved mCRPC+HRR]"),
    "PARP2": (False,
        "PARP family member; also inhibited by clinical PARP inhibitors; "
        "Drugs: olaparib, niraparib, talazoparib [co-inhibited]"),

    # ── Cell Cycle ───────────────────────────────────────────────────────────
    "RB1": (False,
        "Retinoblastoma tumor suppressor; loss in ~10% primary PRAD, ~40% NEPC; "
        "Drugs: CDK4/6 inhibitors (palbociclib, ribociclib, abemaciclib) "
        "less effective with RB1 loss; abemaciclib (Verzenio) clinical trials"),
    "CDKN1B": (False,
        "p27Kip1 CDK inhibitor; frequently lost in PRAD; "
        "Drugs: CDK4/6 inhibitors (palbociclib, ribociclib, abemaciclib); "
        "PI3K/AKT inhibitors restore p27 levels"),
    "CDK4": (False,
        "Cyclin D-CDK4; cell cycle G1/S; "
        "Drugs: palbociclib (Ibrance), ribociclib (Kisqali), "
        "abemaciclib (Verzenio) [CDK4/6 inhibitors]; clinical trials in mCRPC"),
    "CDK6": (False,
        "Cyclin D-CDK6; cell cycle G1/S; "
        "Drugs: palbociclib, ribociclib, abemaciclib [CDK4/6 inhibitors]"),
    "CCND1": (True,
        "Cyclin D1; CDK4/6 activator; amplified/overexpressed in ergPRAD; "
        "Drugs: palbociclib (Ibrance), ribociclib (Kisqali), "
        "abemaciclib (Verzenio) [CDK4/6 inhibitors]"),
    "CCND2": (False,
        "Cyclin D2; CDK4/6 activator; altered in PRAD; "
        "Drugs: CDK4/6 inhibitors (palbociclib, ribociclib, abemaciclib)"),
    "CCND3": (False,
        "Cyclin D3; CDK4/6 activator; "
        "Drugs: CDK4/6 inhibitors"),
    "E2F1": (True,
        "E2F transcription factor; activated by RB1 loss; "
        "upregulated in ergPRAD; drives cell cycle and NEPC genes; "
        "Drugs: CDK4/6 inhibitors reduce E2F1 activity (palbociclib, abemaciclib)"),
    "E2F3": (True,
        "E2F transcription factor; amplified at 6p22 in PRAD; "
        "ERG-cooperating oncogene in ergPRAD; "
        "Drugs: CDK4/6 inhibitors (palbociclib, ribociclib); "
        "BET inhibitors indirectly (JQ1, OTX015)"),
    "PLK1": (False,
        "Polo-like kinase 1; mitosis regulator; "
        "Drugs: volasertib (BI 6727) [PLK1i]; "
        "onvansertib (PCM-075); clinical trials in PRAD"),
    "AURKA": (False,
        "Aurora kinase A; promotes NEPC via MYCN stabilization; "
        "Drugs: alisertib (MLN8237) [AURKAi, clinical trials in PRAD/NEPC]; "
        "danusertib (PHA-739358) [pan-aurora]"),
    "AURKB": (False,
        "Aurora kinase B; mitotic checkpoint; "
        "Drugs: AZD1152 (barasertib) [AURKBi]; danusertib; PRAD trials"),
    "WEE1": (False,
        "G2/M checkpoint kinase; "
        "Drugs: adavosertib (AZD1775) [WEE1i]; "
        "trials in CDK12-mutant and HRR-deficient PRAD"),
    "TOP2A": (False,
        "Topoisomerase IIalpha; amplified with AURKA in aggressive PRAD; "
        "Drugs: doxorubicin, etoposide [Topo II inhibitors]"),
    "TOP2B": (False,
        "Topoisomerase IIbeta; "
        "Drugs: etoposide (used in NEPC regimens: EP regimen)"),

    # ── WNT / beta-catenin pathway ───────────────────────────────────────────
    "CTNNB1": (False,
        "Beta-catenin; WNT effector; activating mutations in ~5% mCRPC; "
        "Drugs: ipafricept (OMP-54F28, WNT ligand trap); "
        "WNT-974, LGK-974 (porcupine inhibitors); PRI-724 (CBP/beta-cat)"),
    "APC": (False,
        "WNT negative regulator; mutations in ~2% mCRPC; "
        "Drugs: porcupine inhibitors (LGK-974, WNT-974) indirectly"),
    "RNF43": (False,
        "WNT negative regulator E3 ubiquitin ligase; "
        "Drugs: LGK-974 (porcupine inhibitor); WNT-974; "
        "mutations activate WNT signaling"),
    "AXIN1": (False,
        "WNT destruction complex scaffold; mutations in mCRPC; "
        "Drugs: Tankyrase inhibitors (XAV939, G007-LK) stabilize AXIN1"),
    "AXIN2": (False,
        "WNT destruction complex; negative regulator; "
        "Drugs: Tankyrase inhibitors (XAV939)"),
    "WNT5A": (False,
        "Non-canonical WNT ligand; promotes PRAD invasion/EMT; "
        "Drugs: anti-WNT5A antibody (investigational); foxy-5 (WNT5A mimetic)"),

    # ── RTK / RAS / RAF / MAPK ───────────────────────────────────────────────
    "ERBB2": (False,
        "HER2 receptor tyrosine kinase; amplified/mutated in ~8% mCRPC; "
        "Drugs: trastuzumab (Herceptin), pertuzumab (Perjeta), "
        "trastuzumab deruxtecan (T-DXd, Enhertu), "
        "lapatinib (Tykerb), tucatinib (Tukysa), neratinib (Nerlynx); "
        "clinical trials in HER2+ mCRPC ongoing"),
    "EGFR": (False,
        "EGF receptor RTK; expressed in PRAD; "
        "Drugs: erlotinib (Tarceva), gefitinib (Iressa), osimertinib (Tagrisso); "
        "cetuximab (Erbitux) [anti-EGFR mAb]; limited single-agent PRAD efficacy"),
    "MET": (False,
        "HGF receptor; amplified in CRPC; drives NEPC; "
        "Drugs: cabozantinib (Cabometyx) [MET/VEGFR2i, FDA-approved mCRPC]; "
        "crizotinib (Xalkori), savolitinib, tepotinib [selective METi]"),
    "FGFR1": (False,
        "FGF receptor 1; amplified in PRAD; "
        "Drugs: erdafitinib (Balversa) [pan-FGFRi], "
        "pemigatinib (Pemazyre), futibatinib [FGFRi]; trials in FGFR-altered PRAD"),
    "FGFR2": (False,
        "FGF receptor 2; altered in mCRPC; "
        "Drugs: erdafitinib, pemigatinib, infigratinib [FGFRi]"),
    "FGFR3": (False,
        "FGF receptor 3; fusions/mutations in PRAD; "
        "Drugs: erdafitinib (Balversa) [FDA-approved FGFR3-mutant urothelial; trials in PRAD]"),
    "IGF1R": (False,
        "IGF-1 receptor; promotes AR-independent CRPC growth; "
        "Drugs: linsitinib (OSI-906) [IGF1Ri]; cixutumumab [anti-IGF1R mAb]; "
        "clinical trials in mCRPC (limited efficacy as monotherapy)"),
    "KRAS": (False,
        "RAS oncogene; mutations rare in PRAD (~3%); "
        "Drugs: sotorasib (Lumakras) [KRAS G12C], adagrasib (Krazati) [KRAS G12C]; "
        "SHP2 inhibitors (TNO155) in combination; tumor-agnostic relevance"),
    "NRAS": (False,
        "RAS oncogene; rare mutations in mCRPC; "
        "Drugs: MEK inhibitors (trametinib, cobimetinib) in combination"),
    "BRAF": (False,
        "RAF kinase; rare mutations in PRAD (~1%); "
        "Drugs: dabrafenib (Tafinlar) + trametinib (Mekinist) [BRAF V600E]; "
        "BRAF fusions: belvarafenib; tumor-agnostic FDA approval"),
    "RAF1": (False,
        "RAF kinase (CRAF); MEK/ERK pathway; "
        "Drugs: sorafenib (Nexavar) [RAF/VEGFRi]; "
        "belvarafenib, LY3009120 [pan-RAF]; MEK inhibitors"),
    "HRAS": (False,
        "RAS oncogene; rare in PRAD; "
        "Drugs: tipifarnib (Zarnestra) [farnesyltransferase inhibitor] for HRAS-mutant"),
    "MAPK1": (False,
        "ERK2; MAPK/ERK pathway effector; "
        "Drugs: ulixertinib (BVD-523) [ERK1/2i]; "
        "MK-8353, SCH900353 [ERKi]; downstream of RAS/RAF/MEK"),
    "MAPK3": (False,
        "ERK1; MAPK/ERK pathway; "
        "Drugs: ERK inhibitors (ulixertinib, MK-8353)"),
    "MAP2K1": (False,
        "MEK1; RAS/RAF effector; "
        "Drugs: trametinib (Mekinist), cobimetinib (Cotellic) [MEKi]; "
        "binimetinib (Mektovi); clinical trials in PRAD"),
    "MAP2K2": (False,
        "MEK2; RAS/RAF effector; "
        "Drugs: trametinib, cobimetinib, binimetinib [MEKi]"),
    "SRC": (False,
        "Non-receptor tyrosine kinase; promotes AR transactivation/invasion; "
        "Drugs: dasatinib (Sprycel) [SRC/BCR-ABL]; bosutinib (Bosulif); "
        "saracatinib (AZD0530); clinical trials in mCRPC"),
    "YES1": (False,
        "SRC family kinase; amplified/activated in PRAD; "
        "Drugs: dasatinib (Sprycel) [SRC/BCR-ABL inhibitor]"),
    "ABL1": (False,
        "ABL tyrosine kinase; activated in CRPC; "
        "Drugs: dasatinib (Sprycel), imatinib (Gleevec), "
        "bosutinib (Bosulif), ponatinib (Iclusig)"),

    # ── JAK / STAT pathway ───────────────────────────────────────────────────
    "STAT3": (True,
        "Signal transducer; activated downstream of IL-6/JAK in ergPRAD; "
        "ERG directly activates STAT3 in ergPRAD; "
        "Drugs: napabucasin (BBI608) [STAT3/cancer stem cell inhibitor]; "
        "silibinin [natural STAT3 inhibitor]; "
        "JAK inhibitors (ruxolitinib) upstream; clinical trials in PRAD"),
    "STAT5A": (False,
        "STAT5; prolactin/GH/IL-2 signaling; promotes AR-mediated PRAD growth; "
        "Drugs: STAT5 inhibitors (SH-4-54, AC-3-019, investigational); "
        "ruxolitinib indirectly [JAK/STAT5i]"),
    "STAT5B": (False,
        "STAT5 isoform; AR cofactor; contributes to CRPC; "
        "Drugs: STAT5 inhibitors (investigational); ruxolitinib [JAKi]"),
    "JAK1": (False,
        "JAK kinase; cytokine signaling; "
        "Drugs: ruxolitinib (Jakafi) [JAK1/2i], baricitinib (Olumiant) [JAK1/2i]; "
        "upadacitinib [JAK1i]; combinations with AR axis agents in PRAD trials"),
    "JAK2": (False,
        "JAK kinase; IL-6/STAT3 pathway in PRAD; "
        "Drugs: ruxolitinib (Jakafi), fedratinib (Inrebic) [JAK2i]; "
        "pacritinib [JAK2/FLT3i]"),

    # ── MYC oncogene ─────────────────────────────────────────────────────────
    "MYC": (True,
        "MYC transcription factor; amplified at 8q24 in ~30% PRAD; "
        "cooperates with ERG fusion in ergPRAD progression; "
        "Drugs: BET inhibitors suppress MYC (JQ1, OTX015, ABBV-744, PLX51107); "
        "Aurora A inhibitors destabilize MYC (alisertib); "
        "CDK9 inhibitors (dinaciclib, AZD4573)"),
    "MYCN": (False,
        "N-MYC; amplified in ~40% NEPC; promotes neuroendocrine transdifferentiation; "
        "Drugs: alisertib (MLN8237) [AURKAi, destabilizes MYCN]; "
        "BET inhibitors (OTX015, JQ1)"),
    "MAX": (False,
        "MYC dimerization partner; MAX inhibitors disrupt MYC-MAX complex; "
        "Drugs: KI-MS2-008, VON-0090 [MAX inhibitors, investigational]"),

    # ── Epigenetics / Chromatin remodeling ───────────────────────────────────
    "EZH2": (True,
        "PRC2 methyltransferase (H3K27me3); overexpressed in mCRPC and ergPRAD; "
        "ERG recruits EZH2 in ergPRAD; "
        "Drugs: tazemetostat (Tazverik) [EZH2i, FDA-approved]; "
        "GSK126, EPZ-6438 (valemetostat); PRAD clinical trials ongoing"),
    "BRD4": (True,
        "BET bromodomain protein; regulates ERG, MYC, AR super-enhancers in ergPRAD; "
        "Drugs: JQ1 (tool compound) [BETi]; OTX015 (birabresib); "
        "ABBV-075 (mivebresib); PLX51107; ABBV-744 [BD2-selective]; "
        "clinical trials in mCRPC/ergPRAD"),
    "BRD2": (True,
        "BET bromodomain; co-regulator of AR and ERG in PRAD; "
        "Drugs: JQ1, OTX015, ABBV-075, ABBV-744 [BET inhibitors]"),
    "BRD3": (True,
        "BET bromodomain; cooperates with ERG in ergPRAD; "
        "Drugs: JQ1, OTX015, birabresib [BET inhibitors]"),
    "BRDT": (False,
        "Testis/PRAD-expressed BET bromodomain; re-expressed in mCRPC; "
        "Drugs: BET inhibitors (JQ1, OTX015, ABBV-744)"),
    "KDM1A": (True,
        "LSD1 histone demethylase (H3K4me1/2); AR coactivator; "
        "upregulated in ergPRAD and CRPC; "
        "Drugs: ORY-1001 (iadademstat) [LSD1i]; GSK-2879552; "
        "CC-90011; INCB059872; PRAD clinical trials"),
    "KDM5C": (False,
        "JARID1C H3K4 demethylase; mutations in mCRPC; "
        "Drugs: KDM5 inhibitors (MRTX-1719 for MTAP-null; CPI-455 investigational)"),
    "KDM6A": (False,
        "UTX H3K27 demethylase; tumor suppressor; mutations in mCRPC; "
        "Drugs: EZH2 inhibitors (tazemetostat) as synthetic lethal in KDM6A-null"),
    "HDAC2": (False,
        "Histone deacetylase 2; AR corepressor regulation; "
        "Drugs: vorinostat (Zolinza) [pan-HDACi]; romidepsin (Istodax); "
        "entinostat [HDAC1/2i]; panobinostat (Farydak); "
        "clinical trials in mCRPC in combination"),
    "CHD1": (True,
        "Chromatin remodeling helicase; deleted in ~27% PRAD (ETS-negative); "
        "CHD1 deletion defines distinct PRAD subtype (non-ergPRAD predominantly); "
        "Drugs: AR inhibitors (enzalutamide sensitization in CHD1-deleted PRAD); "
        "PARP inhibitors (genotoxic stress sensitivity)"),
    "SPOP": (False,
        "E3 ubiquitin ligase adaptor; most frequently mutated in primary PRAD (~15%); "
        "SPOP-mutant PRAD sensitive to BET inhibitors (BRD2/3/4 stabilization); "
        "Drugs: BET inhibitors (JQ1, OTX015, ABBV-744, PLX51107); "
        "CDK4/6 inhibitors (abemaciclib)"),
    "ASXL1": (False,
        "Polycomb-associated chromatin modifier; mutations in mCRPC; "
        "Drugs: EZH2 inhibitors (tazemetostat) in ASXL1-mutant context"),
    "ASXL2": (False,
        "ASXL family; chromatin regulation; altered in PRAD; "
        "Drugs: EZH2 inhibitors (investigational)"),

    # ── TP53 pathway ─────────────────────────────────────────────────────────
    "TP53": (False,
        "Tumor suppressor; mutated/deleted in ~50% mCRPC, ~70% NEPC; "
        "Drugs: APR-246 (eprenetapopt) [p53 reactivator, for TP53 mut]; "
        "idasanutlin (RG7388), navtemadlin (AMG-232), milademetan (DS-3032) "
        "[MDM2i for TP53 WT]; clinical trials in PRAD"),
    "MDM2": (False,
        "p53 negative regulator; amplified in ~5% mCRPC; "
        "Drugs: idasanutlin (RG7388) [MDM2i]; navtemadlin (AMG-232); "
        "milademetan (DS-3032b); siremadlin (HDM201); "
        "clinical trials in MDM2-amplified PRAD"),

    # ── Angiogenesis / VEGF ──────────────────────────────────────────────────
    "KDR": (False,
        "VEGFR2; VEGF receptor; "
        "Drugs: cabozantinib (Cabometyx) [MET/VEGFR2i, FDA-approved mCRPC]; "
        "sunitinib (Sutent), lenvatinib (Lenvima), "
        "axitinib (Inlyta), pazopanib (Votrient) [VEGFRi]"),
    "FLT1": (False,
        "VEGFR1; VEGF receptor; "
        "Drugs: cabozantinib, lenvatinib, sunitinib [multi-VEGFRi]"),
    "PDGFRA": (False,
        "PDGF receptor alpha; tumor microenvironment; "
        "Drugs: imatinib (Gleevec), sunitinib (Sutent) [PDGFRA inhibitors]"),
    "PDGFRB": (False,
        "PDGF receptor beta; stroma/angiogenesis; "
        "Drugs: imatinib, sorafenib (Nexavar), cabozantinib [PDGFRi]"),
    "KIT": (False,
        "c-KIT RTK; mast cell/stroma; "
        "Drugs: imatinib (Gleevec), sunitinib (Sutent), ripretinib [KITi]; "
        "limited single-agent PRAD activity"),
    "FLT3": (False,
        "FLT3 RTK; primarily AML target; expressed in some mCRPC; "
        "Drugs: midostaurin (Rydapt), gilteritinib (Xospata) [FLT3i]"),
    "VEGFA": (False,
        "VEGF-A ligand; angiogenesis driver; "
        "Drugs: bevacizumab (Avastin) [anti-VEGF mAb]; "
        "ramucirumab (Cyramza) [anti-VEGFR2 mAb]; limited PRAD efficacy"),

    # ── EMT / Metastasis ─────────────────────────────────────────────────────
    "CDH1": (False,
        "E-cadherin; epithelial marker; loss promotes EMT/invasion; "
        "Drugs: epigenetic agents restore CDH1 (azacitidine, decitabine) [DNMTi]; "
        "HDAC inhibitors (vorinostat, entinostat)"),
    "VIM": (False,
        "Vimentin; mesenchymal/EMT marker; upregulated in metastatic PRAD; "
        "Drugs: withaferin A (investigational VIM inhibitor)"),
    "SNAI1": (False,
        "Snail transcription factor; EMT driver; represses CDH1; "
        "Drugs: no direct approved drug; GSK3 inhibitors stabilize Snail "
        "(indirectly targeted)"),
    "SNAI2": (False,
        "Slug; EMT transcription factor; promotes PRAD invasion; "
        "Drugs: HDAC inhibitors (vorinostat) downregulate SNAI2"),
    "ZEB1": (False,
        "ZEB1 EMT transcription factor; mesenchymal transition in PRAD; "
        "Drugs: BET inhibitors (JQ1) downregulate ZEB1"),
    "ZEB2": (False,
        "ZEB2 EMT transcription factor; cooperates with ZEB1; "
        "Drugs: HDAC inhibitors; BET inhibitors"),
    "TWIST1": (False,
        "TWIST1 bHLH; EMT driver; promotes PRAD invasion/bone metastasis; "
        "Drugs: no direct approved drug; harmine (investigational)"),

    # ── Neuroendocrine PRAD (NEPC) markers ──────────────────────────────────
    "RET": (False,
        "RET receptor tyrosine kinase; activated/amplified in NEPC; "
        "Drugs: cabozantinib (Cabometyx), vandetanib (Caprelsa) [RET multi-targeted]; "
        "pralsetinib (Gavreto), selpercatinib (Retevmo) [selective RETi]; "
        "tumor-agnostic use for RET fusions/mutations"),
    "SRRM4": (False,
        "Splicing factor; drives neuroendocrine splicing program; key NEPC driver; "
        "Drugs: no approved direct inhibitor; splicing modulator compounds in development"),
    "NCAM1": (False,
        "CD56; neuroendocrine marker; NEPC biomarker; "
        "Drugs: lorvotuzumab mertansine [anti-CD56 ADC, investigational]"),
    "SYP": (False,
        "Synaptophysin; neuroendocrine differentiation marker; clinical NEPC diagnostic"),
    "CHGA": (False,
        "Chromogranin A; neuroendocrine marker; NEPC biomarker; "
        "Drugs: somatostatin analogs (octreotide, lanreotide) in NEPC"),

    # ── Prostate biomarkers ──────────────────────────────────────────────────
    "MSMB": (False,
        "MSMB prostate secretory protein; PRAD risk SNP locus; diagnostic biomarker"),
    "AMACR": (False,
        "Alpha-methylacyl-CoA racemase; diagnostic marker for PRAD; "
        "overexpressed in PRAD vs benign tissue"),

    # ── IDH pathway ──────────────────────────────────────────────────────────
    "IDH1": (False,
        "Isocitrate dehydrogenase 1; rare IDH1 R132 mutations in mCRPC; "
        "Drugs: ivosidenib (Tibsovo) [IDH1i]; olutasidenib (Rezlidhia); "
        "tumor-agnostic context"),
    "IDH2": (False,
        "Isocitrate dehydrogenase 2; rare mutations in mCRPC; "
        "Drugs: enasidenib (Idhifa) [IDH2i]; tumor-agnostic context"),

    # ── HDAC1 (추가) ────────────────────────────────────────────────────────
    "HDAC1": (False,
        "Histone deacetylase 1; AR corepressor complex; "
        "upregulated in CRPC; ERG-HDAC1 interaction in ergPRAD; "
        "Drugs: vorinostat (Zolinza), romidepsin (Istodax) [pan-HDACi]; "
        "entinostat, mocetinostat [HDAC1/2-selective]; "
        "panobinostat (Farydak); clinical trials in mCRPC"),
    "HDAC3": (False,
        "Histone deacetylase 3; AR corepressor (NCoR complex); "
        "Drugs: vorinostat, romidepsin [pan-HDACi]; "
        "RGFP966 [HDAC3-selective, investigational]"),
    "HDAC6": (False,
        "HDAC6; cytoplasmic deacetylase; HSP90/tubulin deacetylation; "
        "promotes AR stability in CRPC; "
        "Drugs: ricolinostat (ACY-1215) [HDAC6-selective]; "
        "citarinostat (ACY-241); combinations with bortezomib in mCRPC"),

    # ── Nuclear receptor coactivators ───────────────────────────────────────
    "SRC1": (False,
        "Steroid receptor coactivator 1 (NCOA1); AR coactivator; "
        "promotes CRPC; upregulated in mCRPC; "
        "Drugs: AR inhibitors (enzalutamide, darolutamide) disrupt SRC1-AR"),
    "SRC3": (False,
        "Steroid receptor coactivator 3 (NCOA3/AIB1); AR coactivator; "
        "amplified/overexpressed in PRAD; "
        "Drugs: gossypol (AT-101) [BCL-2/SRC3 inhibitor]; "
        "SI-2 [SRC3 inhibitor, investigational]"),
}


def build_annotation(gene, annot_dict):
    """유전자명으로 annotation 문자열 반환"""
    if gene not in annot_dict:
        return ""
    is_erg, text = annot_dict[gene]
    if is_erg:
        return f"(ergPRAD) {text}"
    else:
        return f"[PRAD target] {text}"


def main():
    if not os.path.exists(TSV_IN):
        print(f"[ERROR] 입력 파일 없음: {TSV_IN}")
        print("  → node_importance_v3.tsv 생성 후 재실행하세요")
        return

    df = pd.read_csv(TSV_IN, sep="\t")
    print(f"[Load] {TSV_IN}")
    print(f"  총 노드 수: {len(df):,}")

    # annotation 컬럼 추가 (gene 노드만; drug 노드는 빈 string)
    df["prad_annotation"] = df.apply(
        lambda row: build_annotation(row["node"], PRAD_ANNOTATIONS)
                    if row.get("node_type", "gene") == "gene" else "",
        axis=1
    )

    # 저장
    df.to_csv(TSV_OUT, sep="\t", index=False)

    # 통계 출력
    gene_df    = df[df["node_type"] == "gene"]
    annotated  = gene_df[gene_df["prad_annotation"] != ""]
    erg_spec   = gene_df[gene_df["prad_annotation"].str.startswith("(ergPRAD)")]

    print(f"\n[Annotation 결과]")
    print(f"  유전자 노드 수   : {len(gene_df):,}")
    print(f"  주석 추가 유전자 : {len(annotated):,}")
    print(f"    (ergPRAD) 특화 : {len(erg_spec):,}")
    print(f"    [PRAD target]  : {len(annotated) - len(erg_spec):,}")
    print()

    # top-30 annotated 유전자 출력
    print("=" * 100)
    print(f"{'rank':>5}  {'node':<12} {'type':<6} {'visit_prob':>10}  annotation_preview")
    print("=" * 100)
    top_ann = annotated.sort_values("rank").head(30)
    for _, row in top_ann.iterrows():
        tag = "(ergPRAD)" if row["prad_annotation"].startswith("(ergPRAD)") else "[PRAD]  "
        ann_preview = row["prad_annotation"][:70]
        print(f"{int(row['rank']):>5}  {row['node']:<12} {row['node_type']:<6} "
              f"{row['visit_prob']:>10.4%}  {tag}  {ann_preview}")

    print(f"\n[저장 완료] {TSV_OUT}")


if __name__ == "__main__":
    main()
