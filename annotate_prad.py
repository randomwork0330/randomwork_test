"""
전립선암 표적치료제 표적 유전자 주석 추가
node_importance_v2.tsv → node_importance_v2_annotated.tsv

규칙:
  - 전립선암 표적치료제 표적 유전자: annotation 컬럼에 메모
  - TMPRSS2-ERG fusion 전립선암(ergPRAD)에 특화된 경우: "(ergPRAD) ..." 접두사
"""

import pandas as pd

# ──────────────────────────────────────────────
# 1. 주석 딕셔너리
#    형식: 'GENE': (is_ergPRAD, 'annotation text')
#    is_ergPRAD=True → "(ergPRAD) " 접두사 자동 부착
# ──────────────────────────────────────────────
PRAD_ANNOTATIONS = {

    # ── ERG / TMPRSS2-ERG fusion axis ───────────────────────────────
    "ERG": (True,
        "ETS transcription factor; TMPRSS2-ERG fusion driver ~50% PRAD; "
        "transcriptional target in ergPRAD; no approved direct inhibitor, "
        "indirect via BET/EZH2 inhibitors"),
    "TMPRSS2": (True,
        "Androgen-regulated serine protease; 5' fusion partner of ERG in ergPRAD; "
        "AR-driven expression links AR signaling to ERG overexpression"),
    "ETV1": (False,
        "ETS transcription factor; TMPRSS2-ETV1 fusion in ~1% PRAD; "
        "targetable via MEK/ERK pathway inhibition"),
    "ETV4": (False,
        "ETS transcription factor; rare TMPRSS2-ETV4 fusion in PRAD"),
    "ETV5": (False,
        "ETS transcription factor; rare ETS fusion in PRAD"),
    "ETV6": (False,
        "ETS transcription factor; rare rearrangement in PRAD"),
    "SPDEF": (True,
        "ETS transcription factor; prostate-specific; co-activator of AR; "
        "upregulated in ERG-fusion PRAD"),

    # ── Androgen Receptor (AR) axis ─────────────────────────────────
    "AR": (False,
        "Androgen receptor; primary driver of PRAD; "
        "targets: enzalutamide, apalutamide, darolutamide (ARSi); "
        "also targeted by abiraterone (CYP17A1 inhibitor)"),
    "FOXA1": (False,
        "Pioneer transcription factor; AR co-activator in PRAD; "
        "frequently mutated in mCRPC; modulates AR cistrome"),
    "HOXB13": (False,
        "AR co-activator; germline G84E mutation increases PRAD risk; "
        "cooperates with AR in prostate-specific transcription"),
    "NKX3-1": (False,
        "Prostate-specific tumor suppressor; AR target gene; "
        "lost early in PRAD progression"),
    "KLK3": (False,
        "PSA; AR transcriptional target; PRAD biomarker; "
        "used clinically for diagnosis and monitoring"),
    "KLK2": (False,
        "Kallikrein-2; AR target; PRAD biomarker related to PSA"),
    "SRD5A1": (False,
        "5α-reductase type 1; converts testosterone→DHT; "
        "target of dutasteride in PRAD prevention/treatment"),
    "SRD5A2": (False,
        "5α-reductase type 2; primary DHT-producing enzyme in prostate; "
        "target of finasteride, dutasteride (5ARi)"),
    "CYP17A1": (False,
        "Steroidogenesis enzyme; converts pregnenolone→DHEA; "
        "target of abiraterone acetate (FDA-approved mCRPC)"),
    "CYP19A1": (False,
        "Aromatase; estrogen biosynthesis; "
        "involved in CRPC androgen metabolism"),
    "HSD3B1": (False,
        "3β-HSD; adrenal androgen→DHT conversion in CRPC; "
        "A367T mutation confers abiraterone resistance"),
    "HSD17B3": (False,
        "17β-HSD3; testosterone biosynthesis; "
        "expressed in CRPC for intratumoral androgen synthesis"),
    "NCOR1": (True,
        "AR co-repressor; loss promotes AR activity in ergPRAD; "
        "mutations in mCRPC linked to AR antagonist resistance"),
    "NCOR2": (False,
        "AR co-repressor (SMRT); loss drives AR constitutive activity; "
        "mutations associated with enzalutamide resistance"),

    # ── PI3K / AKT / mTOR pathway ────────────────────────────────────
    "PTEN": (True,
        "Tumor suppressor phosphatase; PI3K/AKT negative regulator; "
        "deleted in ~40% PRAD, ~60% mCRPC; "
        "loss activates AKT; synthetic lethality with AR inhibition; "
        "PTEN loss + AR-Si → ipatasertib (AKTi) approval context"),
    "PIK3CA": (False,
        "Catalytic subunit of PI3K; mutated/amplified in PRAD; "
        "targets: alpelisib (PI3Kα inhibitor), clinical trials in PRAD"),
    "PIK3CB": (True,
        "PI3Kβ catalytic subunit; dominant PI3K isoform in PTEN-null ergPRAD; "
        "target: GSK2636771, AZD8186 (PI3Kβ-selective inhibitors)"),
    "PIK3R1": (False,
        "PI3K regulatory subunit p85α; mutations in mCRPC"),
    "AKT1": (False,
        "AKT serine/threonine kinase; central PI3K effector; "
        "targets: ipatasertib, capivasertib (AKTi); approved combinations in PRAD"),
    "AKT2": (False,
        "AKT isoform 2; PI3K/AKT signaling; targeted by pan-AKT inhibitors"),
    "AKT3": (False,
        "AKT isoform 3; upregulated in neuroendocrine PRAD (NEPC)"),
    "MTOR": (False,
        "mTOR kinase; downstream of AKT; "
        "targets: everolimus, temsirolimus (mTORC1i); "
        "dual PI3K/mTOR inhibitors in clinical trials"),
    "TSC1": (False,
        "mTOR negative regulator (tuberin complex); "
        "loss activates mTORC1; mutation sensitizes to mTOR inhibitors"),
    "TSC2": (False,
        "mTOR negative regulator (tuberin); "
        "loss in PRAD activates mTOR signaling"),
    "RICTOR": (False,
        "mTORC2 component; regulates AKT Ser473 phosphorylation; "
        "amplified in PRAD, especially NEPC"),
    "RPTOR": (False,
        "mTORC1 scaffold; target of rapamycin-based inhibitors"),

    # ── DNA Damage Response / HRR ─────────────────────────────────────
    "BRCA2": (False,
        "HRR gene; germline/somatic mutations in ~13% mCRPC; "
        "PARP inhibitor sensitivity; "
        "targets: olaparib (FDA-approved), rucaparib, niraparib, talazoparib"),
    "BRCA1": (False,
        "HRR gene; mutations in ~3% mCRPC; "
        "PARP inhibitor sensitivity (olaparib, rucaparib)"),
    "ATM": (False,
        "HRR/DSB repair kinase; mutations in ~7% mCRPC; "
        "PARP inhibitor sensitivity (olaparib label includes ATM); "
        "ATM inhibitors (AZD0156, AZD1390) in clinical trials"),
    "CDK12": (False,
        "Transcription-coupled DNA repair kinase; "
        "biallelic loss ~7% mCRPC → tandem duplicator phenotype; "
        "immunotherapy sensitivity; "
        "CDK12i (dinaciclib, THZ531) in development"),
    "PALB2": (False,
        "BRCA2 partner; HRR; mutations sensitize to PARP inhibitors; "
        "included in olaparib/rucaparib approved biomarker panels"),
    "RAD51": (False,
        "HRR recombinase; core effector; "
        "RAD51 inhibitors in preclinical PRAD; "
        "overexpression confers PARP inhibitor resistance"),
    "RAD51C": (False,
        "RAD51 paralog; HRR; mutations sensitize to PARP inhibitors; "
        "included in PARP inhibitor biomarker panels"),
    "RAD51D": (False,
        "RAD51 paralog; HRR; PARP inhibitor sensitivity"),
    "BRIP1": (False,
        "BRCA1-interacting helicase; HRR; mutations in mCRPC; "
        "PARP inhibitor sensitivity (olaparib label)"),
    "BARD1": (False,
        "BRCA1-binding protein; HRR; mutations sensitize to PARP inhibitors"),
    "NBN": (False,
        "Nibrin; DSB sensing (MRN complex); "
        "mutations in mCRPC; PARP inhibitor sensitivity context"),
    "CHEK2": (False,
        "DNA damage checkpoint kinase; HRR; "
        "germline mutations in ~2% PRAD; "
        "sensitizes to PARP inhibitors, CHK1/2 inhibitors"),
    "CHEK1": (False,
        "Checkpoint kinase 1; replication stress response; "
        "target: prexasertib (CHK1i), LY2603618; combinations with PARPi"),
    "ATR": (False,
        "Replication stress kinase; "
        "target: ceralasertib (AZD6738), berzosertib (M6620); "
        "combinations with olaparib in mCRPC trials"),
    "FANCL": (False,
        "Fanconi anemia E3 ubiquitin ligase; HRR; "
        "mutations in mCRPC; PARP inhibitor sensitivity"),
    "FANCA": (False,
        "Fanconi anemia complementation group A; "
        "mutations sensitize to platinum/PARP inhibitors"),
    "FANCD2": (False,
        "Fanconi anemia core; ICL repair; "
        "biomarker for platinum/PARP inhibitor sensitivity"),
    "FANCG": (False,
        "Fanconi anemia group G; HRR; PARP inhibitor context"),
    "RAD54L": (False,
        "HRR chromatin remodeler; RAD51 cofactor; "
        "altered in mCRPC; PARP inhibitor sensitivity context"),
    "PARP1": (False,
        "Poly(ADP-ribose) polymerase 1; BER/SSBR; "
        "direct target of PARP inhibitors: olaparib, rucaparib, "
        "niraparib, talazoparib (FDA-approved mCRPC with HRR mutations)"),
    "PARP2": (False,
        "PARP family member; also inhibited by clinical PARP inhibitors; "
        "role in DNA repair in PRAD"),

    # ── Cell Cycle ───────────────────────────────────────────────────
    "RB1": (False,
        "Retinoblastoma tumor suppressor; "
        "loss in ~10% primary PRAD, ~40% mCRPC/NEPC; "
        "loss drives NEPC transition; CDK4/6i less effective with RB1 loss"),
    "CDKN1B": (False,
        "p27Kip1 CDK inhibitor; frequently lost in PRAD; "
        "downstream of PTEN/AKT axis"),
    "CDK4": (False,
        "Cyclin D-CDK4; cell cycle G1/S; "
        "target: palbociclib, ribociclib, abemaciclib (CDK4/6i); "
        "clinical trials ongoing in mCRPC"),
    "CDK6": (False,
        "Cyclin D-CDK6; cell cycle G1/S; "
        "target: CDK4/6 inhibitors (palbociclib, ribociclib, abemaciclib)"),
    "CCND1": (True,
        "Cyclin D1; CDK4/6 activator; amplified/overexpressed in ergPRAD; "
        "target: CDK4/6 inhibitors"),
    "CCND2": (False,
        "Cyclin D2; CDK4/6 activator; altered in PRAD"),
    "CCND3": (False,
        "Cyclin D3; CDK4/6 activator; altered in PRAD"),
    "E2F1": (True,
        "E2F transcription factor; activated by RB1 loss; "
        "upregulated in ergPRAD; drives cell cycle and NEPC genes"),
    "E2F3": (True,
        "E2F transcription factor; amplified at 6p22 in PRAD; "
        "ERG-cooperating oncogene in ergPRAD"),
    "PLK1": (False,
        "Polo-like kinase 1; mitosis regulator; "
        "target: volasertib, onvansertib (PLK1i); clinical trials in PRAD"),
    "AURKA": (False,
        "Aurora kinase A; promotes NEPC via MYCN stabilization; "
        "target: alisertib (AURKAi); clinical trials in PRAD/NEPC"),
    "AURKB": (False,
        "Aurora kinase B; mitotic checkpoint; "
        "target: AZD1152 (barasertib); PRAD trials"),
    "WEE1": (False,
        "G2/M checkpoint kinase; target: adavosertib (AZD1775, WEE1i); "
        "trials in CDK12-mutant and HRR-deficient PRAD"),
    "TOP2A": (False,
        "Topoisomerase IIα; target of anthracyclines; "
        "amplified with AURKA in aggressive PRAD"),
    "TOP2B": (False,
        "Topoisomerase IIβ; target of etoposide; "
        "used in NEPC regimens"),

    # ── WNT / β-catenin pathway ──────────────────────────────────────
    "CTNNB1": (False,
        "β-catenin; WNT effector; activating mutations in ~5% mCRPC; "
        "cooperates with AR; WNT inhibitors (ipafricept) in development"),
    "APC": (False,
        "WNT negative regulator; mutations in ~2% mCRPC; "
        "loss activates β-catenin/WNT signaling"),
    "RNF43": (False,
        "WNT negative regulator; E3 ubiquitin ligase; "
        "mutations activate WNT; target of porcupine inhibitors (LGK-974, WNT-974)"),
    "AXIN1": (False,
        "WNT destruction complex; mutations in mCRPC"),
    "AXIN2": (False,
        "WNT destruction complex; negative regulator"),
    "WNT5A": (False,
        "Non-canonical WNT ligand; promotes PRAD invasion/EMT"),

    # ── RTK / RAS / RAF / MAPK ──────────────────────────────────────
    "ERBB2": (False,
        "HER2 receptor tyrosine kinase; amplified/mutated in ~8% mCRPC; "
        "targets: trastuzumab, pertuzumab, trastuzumab deruxtecan (T-DXd), "
        "lapatinib, tucatinib; clinical trials in HER2+ mCRPC"),
    "EGFR": (False,
        "EGF receptor RTK; expressed in PRAD; "
        "limited single-agent activity; "
        "targets: erlotinib, gefitinib, cetuximab (limited PRAD efficacy)"),
    "MET": (False,
        "HGF receptor; amplified in CRPC; drives NEPC; "
        "targets: cabozantinib (MET/VEGFR2i, FDA-approved mCRPC), crizotinib"),
    "FGFR1": (False,
        "FGF receptor 1; amplified in PRAD; "
        "targets: erdafitinib, pemigatinib (FGFRi); trials in PRAD"),
    "FGFR2": (False,
        "FGF receptor 2; altered in mCRPC; "
        "target: FGFRi (erdafitinib, infigratinib)"),
    "FGFR3": (False,
        "FGF receptor 3; fusions/mutations in PRAD; "
        "target: FGFRi (erdafitinib)"),
    "IGF1R": (False,
        "Insulin-like growth factor 1 receptor; "
        "promotes AR-independent CRPC growth; "
        "targets: linsitinib, cixutumumab; clinical trials in PRAD"),
    "KRAS": (False,
        "RAS oncogene; mutations rare in PRAD (~3%) but present in aggressive cases; "
        "targets: sotorasib/adagrasib (KRAS G12C); "
        "SHP2i combinations; PRAD trials ongoing"),
    "NRAS": (False,
        "RAS oncogene; rare mutations in mCRPC; "
        "targeted by MEK inhibitors (trametinib) in combination"),
    "BRAF": (False,
        "RAF kinase; rare mutations (~1%) in PRAD; "
        "BRAF V600E: dabrafenib+trametinib; "
        "BRAF fusions in mCRPC; tumor-agnostic approval relevant"),
    "RAF1": (False,
        "RAF kinase (CRAF); MEK/ERK pathway; "
        "target: sorafenib, belvarafenib; altered in aggressive PRAD"),
    "HRAS": (False,
        "RAS oncogene; rare in PRAD; "
        "tipifarnib (farnesyltransferase inhibitor) for HRAS-mutant tumors"),
    "MAPK1": (False,
        "ERK2; MAPK/ERK pathway effector; "
        "target: ERK inhibitors (ulixertinib, MK-8353); "
        "downstream of RAS/RAF/MEK in PRAD"),
    "MAPK3": (False,
        "ERK1; MAPK/ERK pathway; co-target with MAPK1"),
    "MAP2K1": (False,
        "MEK1; RAS/RAF effector; "
        "targets: trametinib, cobimetinib (MEKi); clinical trials in PRAD"),
    "MAP2K2": (False,
        "MEK2; RAS/RAF effector; target of MEK inhibitors"),
    "SRC": (False,
        "Non-receptor tyrosine kinase; promotes AR transactivation, invasion; "
        "targets: dasatinib, bosutinib (SRCi); clinical trials in mCRPC"),
    "YES1": (False,
        "SRC family kinase; amplified/activated in PRAD; "
        "target: dasatinib (SRC/BCR-ABL inhibitor)"),
    "ABL1": (False,
        "ABL tyrosine kinase; activated in CRPC; "
        "target: dasatinib, imatinib; clinical trials in mCRPC"),

    # ── JAK / STAT pathway ───────────────────────────────────────────
    "STAT3": (True,
        "Signal transducer; activated downstream of IL-6/JAK in ergPRAD; "
        "ERG directly activates STAT3 in ergPRAD; "
        "targets: STAT3 inhibitors (napabucasin, silibinin); PRAD trials"),
    "STAT5A": (False,
        "STAT5; prolactin/GH/IL-2 signaling; "
        "promotes AR-mediated PRAD growth; "
        "target: STAT5 inhibitors in development"),
    "STAT5B": (False,
        "STAT5 isoform; AR cofactor; contributes to CRPC"),
    "JAK1": (False,
        "JAK kinase; cytokine signaling; "
        "targets: ruxolitinib, baricitinib (JAKi); "
        "trials in combination with AR axis agents in PRAD"),
    "JAK2": (False,
        "JAK kinase; IL-6/STAT3 pathway in PRAD; "
        "target: ruxolitinib, fedratinib (JAK2i)"),

    # ── MYC oncogene ─────────────────────────────────────────────────
    "MYC": (True,
        "MYC transcription factor; amplified at 8q24 in ~30% PRAD; "
        "cooperates with ERG fusion in ergPRAD progression; "
        "indirect targets: BET inhibitors (JQ1, OTX015) suppress MYC; "
        "Aurora A inhibition destabilizes MYC"),
    "MYCN": (False,
        "N-MYC; amplified in ~40% NEPC; promotes neuroendocrine transdifferentiation; "
        "stabilized by AURKA; target: alisertib (AURKA inhibitor) in NEPC"),
    "MAX": (False,
        "MYC dimerization partner; MAX inhibitors disrupt MYC-MAX complex; "
        "target: KI-MS2-008 and other MAX inhibitors in development"),

    # ── Epigenetics / Chromatin remodeling ───────────────────────────
    "EZH2": (True,
        "PRC2 methyltransferase (H3K27me3); "
        "overexpressed in mCRPC and ergPRAD; ERG recruits EZH2 in ergPRAD; "
        "targets: tazemetostat (EZH2i, FDA-approved follicular lymphoma), "
        "GSK126, EPZ-6438; PRAD clinical trials ongoing"),
    "BRD4": (True,
        "BET bromodomain protein; reads acetylated histones; "
        "regulates ERG, MYC, AR super-enhancers in ergPRAD; "
        "targets: JQ1, OTX015, ABBV-075 (BETi); PRAD clinical trials"),
    "BRD2": (True,
        "BET bromodomain; co-regulator of AR and ERG in PRAD; "
        "target: BET inhibitors (JQ1, OTX015, ABBV-744)"),
    "BRD3": (True,
        "BET bromodomain; cooperates with ERG in ergPRAD; "
        "target: BET inhibitors"),
    "BRDT": (False,
        "Testis/PRAD-expressed BET bromodomain; "
        "target: BET inhibitors; re-expressed in mCRPC"),
    "KDM1A": (True,
        "LSD1 histone demethylase (H3K4me1/2); "
        "AR coactivator; upregulated in ergPRAD and CRPC; "
        "targets: ORY-1001, GSK-2879552 (LSD1i); PRAD clinical trials"),
    "KDM5C": (False,
        "JARID1C H3K4 demethylase; "
        "mutations in mCRPC; tumor suppressor role"),
    "KDM6A": (False,
        "UTX H3K27 demethylase; "
        "tumor suppressor; mutations in mCRPC"),
    "HDAC2": (False,
        "Histone deacetylase 2; AR corepressor regulation; "
        "targets: vorinostat, romidepsin, entinostat (HDACi); "
        "clinical trials in mCRPC in combination"),
    "CHD1": (True,
        "Chromatin remodeling helicase; "
        "deleted in ~27% PRAD (ETS-negative); "
        "CHD1 deletion defines distinct PRAD subtype (non-ergPRAD); "
        "CHD1 loss sensitizes to AR inhibition and genotoxic stress"),
    "SPOP": (False,
        "E3 ubiquitin ligase adaptor; "
        "most frequently mutated gene in primary PRAD (~15%); "
        "defines SPOP-mutant PRAD subtype; "
        "mutations cause BET bromodomain protein stabilization; "
        "SPOP-mut PRAD sensitive to BET inhibitors"),
    "ASXL1": (False,
        "Polycomb-associated chromatin modifier; "
        "mutations in mCRPC; EZH2 pathway context"),
    "ASXL2": (False,
        "ASXL family; chromatin regulation; altered in PRAD"),

    # ── TP53 pathway ─────────────────────────────────────────────────
    "TP53": (False,
        "Tumor suppressor; mutated/deleted in ~50% mCRPC, ~70% NEPC; "
        "loss accelerates CRPC/NEPC transition; "
        "targets: APR-246 (eprenetapopt, p53 reactivator); MDM2 inhibitors "
        "(idasanutlin, RG7112) for TP53 WT tumors; clinical trials in PRAD"),
    "MDM2": (False,
        "p53 negative regulator; amplified in ~5% mCRPC; "
        "targets: idasanutlin, navtemadlin, milademetan (MDM2i); "
        "trials in MDM2-amplified PRAD"),

    # ── Angiogenesis / VEGF ──────────────────────────────────────────
    "KDR": (False,
        "VEGFR2; VEGF receptor; "
        "targets: cabozantinib (MET/VEGFR2i, FDA-approved mCRPC), "
        "sunitinib, lenvatinib, axitinib (VEGFRi)"),
    "FLT1": (False,
        "VEGFR1; VEGF receptor; "
        "co-targeted by cabozantinib, lenvatinib"),
    "PDGFRA": (False,
        "PDGF receptor alpha; tumor microenvironment; "
        "targets: imatinib, sunitinib"),
    "PDGFRB": (False,
        "PDGF receptor beta; stroma/angiogenesis; "
        "targets: imatinib, sorafenib, cabozantinib"),
    "KIT": (False,
        "c-KIT RTK; mast cell/stroma; "
        "targets: imatinib, sunitinib; limited PRAD activity"),
    "FLT3": (False,
        "FLT3 RTK; primarily AML target; "
        "expressed in some mCRPC; targets: midostaurin, gilteritinib"),
    "VEGFA": (False,
        "VEGF-A ligand; angiogenesis driver; "
        "target: bevacizumab (anti-VEGF mAb); "
        "limited single-agent PRAD efficacy"),

    # ── EMT / Metastasis ─────────────────────────────────────────────
    "CDH1": (False,
        "E-cadherin; epithelial marker; "
        "loss promotes EMT and PRAD invasion; "
        "epigenetic silencing common in PRAD"),
    "VIM": (False,
        "Vimentin; mesenchymal marker; "
        "upregulated in metastatic PRAD; EMT marker"),
    "SNAI1": (False,
        "Snail transcription factor; EMT driver; "
        "represses CDH1 in PRAD"),
    "SNAI2": (False,
        "Slug; EMT transcription factor; "
        "promotes invasion in PRAD"),
    "ZEB1": (False,
        "ZEB1 EMT transcription factor; "
        "promotes mesenchymal transition in PRAD"),
    "ZEB2": (False,
        "ZEB2 EMT transcription factor; "
        "cooperates with ZEB1 in PRAD EMT"),
    "TWIST1": (False,
        "TWIST1 bHLH; EMT driver; "
        "promotes PRAD invasion and bone metastasis"),

    # ── Neuroendocrine PRAD (NEPC) markers ──────────────────────────
    "RET": (False,
        "RET receptor tyrosine kinase; "
        "activated/amplified in NEPC; "
        "targets: cabozantinib, vandetanib, pralsetinib, selpercatinib (RETi)"),
    "SRRM4": (False,
        "Splicing factor; drives neuroendocrine splicing program; "
        "key driver of NEPC transition; no approved inhibitor yet"),
    "NCAM1": (False,
        "CD56; neuroendocrine marker; "
        "NEPC biomarker; expressed in NEPC"),
    "SYP": (False,
        "Synaptophysin; neuroendocrine differentiation marker; "
        "clinical NEPC diagnostic marker"),
    "CHGA": (False,
        "Chromogranin A; neuroendocrine marker; "
        "NEPC biomarker; serum/tissue diagnostic"),

    # ── Prostate biomarkers ──────────────────────────────────────────
    "MSMB": (False,
        "MSMB prostate secretory protein; PRAD risk SNP locus; biomarker"),
    "AMACR": (False,
        "Alpha-methylacyl-CoA racemase; diagnostic marker for PRAD; "
        "overexpressed in PRAD vs. benign"),

    # ── IDH pathway ──────────────────────────────────────────────────
    "IDH1": (False,
        "Isocitrate dehydrogenase 1; rare IDH1 R132 mutations in mCRPC; "
        "target: ivosidenib (IDH1i); tumor-agnostic context"),
    "IDH2": (False,
        "Isocitrate dehydrogenase 2; rare mutations in mCRPC; "
        "target: enasidenib (IDH2i); tumor-agnostic context"),
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
    tsv_path = r"C:/Users/user/Desktop/CNN_CVPR/__DATAAn__/random_work_test/node_importance_v2.tsv"
    out_path = r"C:/Users/user/Desktop/CNN_CVPR/__DATAAn__/random_work_test/node_importance_v2_annotated.tsv"

    df = pd.read_csv(tsv_path, sep="\t")

    # annotation 컬럼 추가
    df["prad_annotation"] = df["node"].apply(
        lambda g: build_annotation(g, PRAD_ANNOTATIONS)
    )

    # 저장
    df.to_csv(out_path, sep="\t", index=False)

    # 통계 출력
    annotated = df[df["prad_annotation"] != ""]
    erg_specific = df[df["prad_annotation"].str.startswith("(ergPRAD)")]

    print(f"총 노드 수        : {len(df):,}")
    print(f"주석 추가 유전자   : {len(annotated):,}")
    print(f"  ├ (ergPRAD) 특화 : {len(erg_specific):,}")
    print(f"  └ [PRAD target]  : {len(annotated) - len(erg_specific):,}")
    print()
    print("=" * 90)
    print(f"{'rank':>5}  {'node':<12} {'type':<6} {'visit_prob':>10}  prad_annotation[:60]")
    print("=" * 90)
    for _, row in annotated.sort_values("rank").iterrows():
        tag = "(ergPRAD)" if row["prad_annotation"].startswith("(ergPRAD)") else "[PRAD]  "
        print(f"{row['rank']:>5}  {row['node']:<12} {row['node_type']:<6} "
              f"{row['visit_prob']:>10.4%}  {tag}  {row['node']}")
    print(f"\n저장 완료: {out_path}")


if __name__ == "__main__":
    main()
