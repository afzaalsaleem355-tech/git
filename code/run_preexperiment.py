# -*- coding: utf-8 -*-
"""
最小预实验：面向 PET 的阻燃分子生成式设计（代理标签）
完整闭环：数据 -> 性质预测器(QSPR) -> 组合生成 -> 筛选 -> 输出候选
所有结果写入 output/ 目录。

诚实声明：
  - 标签是"专家规则代理分数"（模拟阻燃性），正式研究需替换为真实成炭率/LOI 数据；
  - 生成器采用"芳香骨架 × 阻燃官能团"组合枚举（化学上真实可行），
    作为生成式设计的基线；更高级的 VAE/遗传算法为后续工作。
"""
import os, math, itertools, warnings
import numpy as np
warnings.filterwarnings("ignore")

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache_pre")

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem, DataStructs, Draw
RDLogger.DisableLog("rdApp.*")

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from rdkit.Contrib.SA_Score import sascorer  # type: ignore
    HAS_SA = True
except Exception:
    HAS_SA = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------
# 1. 数据：手工整理的阻燃相关分子库（真实 SMILES）
# ------------------------------------------------------------------
LIBRARY = [
    "O=P1Oc2ccccc2-c2ccccc21", "O=P(Oc1ccccc1)(Oc1ccccc1)Oc1ccccc1",
    "O=P(c1ccccc1)(c1ccccc1)c1ccccc1", "O=P(O)(Oc1ccccc1)Oc1ccccc1",
    "O=P(O)(O)c1ccccc1", "O=P(OC)(OC)c1ccccc1", "O=P(OCC)(OCC)c1ccccc1",
    "COP(C)(=O)OC", "CCOP(=O)(OCC)OCC", "COP(=O)(OC)OC", "CCOP(=O)(CC)OCC",
    "OP(=O)(O)O", "COP(=O)(O)OC", "CP(=O)(O)O", "CCP(=O)(O)O",
    "O=P(N)(N)N", "O=P(Nc1ccccc1)(Nc1ccccc1)Nc1ccccc1",
    "ClP1(Cl)=NP(Cl)(Cl)=NP(Cl)(Cl)=N1",
    "Nc1nc(N)nc(N)n1", "O=c1[nH]c(=O)[nH]c(=O)[nH]1", "Nc1ccccc1",
    "Nc1ccc(N)cc1", "N(c1ccccc1)(c1ccccc1)c1ccccc1", "Nc1ccc(-c2ccccc2)cc1",
    "N(c1ccccc1)c1ccccc1", "c1ccncc1", "c1ccc2ncccc2c1", "c1ccc2[nH]cnc2c1",
    "O=C1NC(=O)c2ccccc21", "NC(N)=O", "c1nccnc1", "c1ccc2ncsc2c1",
    "c1ccccc1", "c1ccc(-c2ccccc2)cc1", "c1ccc2ccccc2c1", "c1ccc2cc3ccccc3cc2c1",
    "O(c1ccccc1)c1ccccc1", "O=S(=O)(c1ccccc1)c1ccccc1", "S(c1ccccc1)c1ccccc1",
    "O=C(c1ccccc1)c1ccccc1", "Cc1ccc(C)cc1", "c1ccc(Cc2ccccc2)cc1",
    "O=C(O)c1ccc(C(=O)O)cc1", "OCCO", "O=C(OCCO)c1ccc(C(=O)OCCO)cc1",
    "O=C(OC)c1ccc(C(=O)OC)cc1", "O=C(OCC)c1ccc(C(=O)OCC)cc1",
    "O=C(O)c1ccccc1", "O=C(O)c1ccccc1C(=O)O", "CCOC(=O)c1ccccc1C(=O)OCC",
    "CCCCCC", "CCCCCCCCCC", "OCC(O)CO", "OCC(CO)(CO)CO", "OCCCCO",
    "OCCOCCO", "CCOC(C)=O", "CCO", "CCCCCO", "CC(=O)O",
    "Oc1c(Br)cc(C(C)(C)c2cc(Br)c(O)c(Br)c2)cc1Br", "Brc1ccc(Br)cc1", "Clc1ccccc1Cl",
]

# ------------------------------------------------------------------
# 2. 代理标签（透明专家规则，模拟"阻燃性"）
# ------------------------------------------------------------------
def count_arom_rings(mol):
    ri = mol.GetRingInfo()
    return sum(1 for r in ri.AtomRings()
               if all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in r))

def count_hetero_arom_rings(mol):
    ri = mol.GetRingInfo(); n = 0
    for r in ri.AtomRings():
        if all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in r):
            if any(mol.GetAtomWithIdx(a).GetAtomicNum() != 6 for a in r):
                n += 1
    return n

def element_count(mol, z):
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == z)

def flame_proxy(mol):
    n_arom = count_arom_rings(mol); n_het = count_hetero_arom_rings(mol)
    n_P = element_count(mol, 15); n_N = element_count(mol, 7); n_O = element_count(mol, 8)
    n_Cali = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6 and not a.GetIsAromatic())
    n_rings = len(mol.GetRingInfo().AtomRings())
    frac_arom = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()) / max(1, mol.GetNumAtoms())
    s = (0.25 * math.tanh(n_arom / 2.0) + 0.30 * math.tanh(n_P / 1.0)
         + 0.20 * math.tanh(n_N / 2.0) + 0.10 * math.tanh(n_het / 2.0)
         + 0.05 * min(n_rings, 4) / 4.0 + 0.05 * frac_arom
         + 0.05 * math.tanh(n_O / 4.0) - 0.15 * math.tanh(n_Cali / 8.0))
    return max(0.0, min(1.0, s))

def domain_descriptors(mol):
    n_hal = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in (9, 17, 35, 53))
    return np.array([
        element_count(mol, 15), element_count(mol, 7), element_count(mol, 8),
        element_count(mol, 16), n_hal, count_arom_rings(mol),
        count_hetero_arom_rings(mol), len(mol.GetRingInfo().AtomRings()),
        sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()) / max(1, mol.GetNumAtoms()),
        sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6 and not a.GetIsAromatic()),
        Descriptors.MolWt(mol),
    ], dtype=float)

def featurize(mol):
    fp = np.zeros(1024, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(
        AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024), fp)
    return np.concatenate([fp, domain_descriptors(mol)])

def tanimoto_max(mol, lib_fps):
    v = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    return max(DataStructs.TanimotoSimilarity(v, x) for x in lib_fps)

# ------------------------------------------------------------------
# 3. 训练性质预测器
# ------------------------------------------------------------------
print("=" * 72)
print("STEP 1  数据与性质预测器(QSPR)")
print("=" * 72)

data = []
for s in LIBRARY:
    m = Chem.MolFromSmiles(s)
    if m is None:
        continue
    data.append((s, m, flame_proxy(m)))

X = np.vstack([featurize(m) for _, m, _ in data])
y = np.array([sc for _, _, sc in data])
train_smiles = {Chem.MolToSmiles(m) for _, m, _ in data}

rf = RandomForestRegressor(n_estimators=500, random_state=42)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
pred_cv = cross_val_predict(rf, X, y, cv=cv)
r2rf = r2_score(y, pred_cv); maerf = mean_absolute_error(y, pred_cv)
rf.fit(X, y)

ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
pred_rd = cross_val_predict(ridge, X, y, cv=cv)
r2rd = r2_score(y, pred_rd)

print(f"  数据量 {len(data)} | 5折交叉验证")
print(f"  随机森林: R²={r2rf:.3f}  MAE={maerf:.3f}")
print(f"  Ridge基线: R²={r2rd:.3f}")
print(f"  标签分布: min={y.min():.3f} max={y.max():.3f} mean={y.mean():.3f}")

with open(os.path.join(OUT, "library.csv"), "w") as f:
    f.write("SMILES,proxy_score\n")
    for s, _, sc in sorted(data, key=lambda t: -t[2]):
        f.write(f"{s},{sc:.4f}\n")
with open(os.path.join(OUT, "predictor_metrics.txt"), "w") as f:
    f.write(f"n_samples={len(data)}\n")
    f.write(f"RandomForest R2={r2rf:.4f} MAE={maerf:.4f}\n")
    f.write(f"Ridge R2={r2rd:.4f}\n")

plt.figure(figsize=(5, 5))
plt.scatter(y, pred_cv, alpha=0.8, edgecolors="k", linewidths=0.4)
lims = [y.min() - 0.05, y.max() + 0.05]
plt.plot(lims, lims, "r--", lw=1)
plt.xlabel("True proxy score"); plt.ylabel("Predicted proxy score")
plt.title(f"QSPR predictor (RF, 5-fold CV)  R²={r2rf:.3f}")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "parity_plot.png"), dpi=130); plt.close()

# ------------------------------------------------------------------
# 4. 生成：芳香骨架 × 阻燃官能团 组合枚举
# ------------------------------------------------------------------
print()
print("=" * 72)
print("STEP 2  组合生成（骨架 × 官能团）")
print("=" * 72)

SCAFFOLDS = [
    "[1*]c1ccccc1",                    # 单取代苯
    "[1*]c1ccc([2*])cc1",              # 1,4-二取代苯
    "[1*]c1ccccc1[2*]",                # 1,2-二取代苯
    "[1*]c1cc([2*])cc([3*])c1",        # 1,3,5-三取代苯
    "[1*]c1nc([2*])nc([3*])n1",        # 三嗪
    "[1*]c1ccc(-c2ccccc2)cc1",         # 联苯
    "[1*]c1ccc(Oc2ccccc2)cc1",         # 二苯醚
    "[1*]c1ccc2ccccc2c1",              # 萘
    "[1*]c1ccc(S(=O)(=O)c2ccccc2)cc1", # 二苯砜
    "[1*]c1ccncc1",                    # 吡啶
]
SUBSTITUENTS = [
    "P(=O)(O)O",      # 膦酸
    "OP(=O)(O)O",     # 磷酸酯
    "P(=O)(OCC)OCC",  # 膦酸二乙酯
    "N",              # 氨基
    "O",              # 羟基
    "C#N",            # 氰基
    "S(=O)(=O)O",     # 磺酸
    "C(=O)O",         # 羧酸
    "c1ccccc1",       # 苯基
    "C",              # 甲基
]

def dummy_count(smi):
    m = Chem.MolFromSmiles(smi)
    return sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == 0)

def decorate(scaffold_smi, combo):
    mol = Chem.MolFromSmiles(scaffold_smi)
    for i, sub in enumerate(combo, start=1):
        q = Chem.MolFromSmiles(f"[{i}*]")
        submol = Chem.MolFromSmiles(sub)
        prods = AllChem.ReplaceSubstructs(mol, q, submol, replaceAll=True)
        if not prods:
            return None
        mol = prods[0]
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None

def acceptable(mol):
    mw = Descriptors.MolWt(mol)
    if not (120 <= mw <= 650):
        return False
    if any(a.GetAtomicNum() in (9, 17, 35, 53) for a in mol.GetAtoms()):
        return False
    if count_arom_rings(mol) < 1 and element_count(mol, 15) < 1 and element_count(mol, 7) < 1:
        return False
    if HAS_SA:
        try:
            if sascorer.calculateScore(mol) > 6.0:
                return False
        except Exception:
            pass
    return True

lib_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for _, m, _ in data]
generated = {}
for scaff in SCAFFOLDS:
    n = dummy_count(scaff)
    for combo in itertools.product(SUBSTITUENTS, repeat=n):
        mol = decorate(scaff, combo)
        if mol is None or not acceptable(mol):
            continue
        smi = Chem.MolToSmiles(mol)
        if smi in generated:
            continue
        pred = rf.predict([featurize(mol)])[0]
        nov = 1.0 - tanimoto_max(mol, lib_fps)
        generated[smi] = (mol, pred, nov)

print(f"  组合总数 {sum(len(SUBSTITUENTS)**dummy_count(s) for s in SCAFFOLDS)}")
print(f"  过滤后合法候选: {len(generated)}")

# ------------------------------------------------------------------
# 5. 结果汇总
# ------------------------------------------------------------------
print()
print("=" * 72)
print("STEP 3  候选筛选与输出")
print("=" * 72)

best_train = max(y)
novel = {s: v for s, v in generated.items() if v[2] > 0.5}   # 与训练库明显不同
novel_sorted = sorted(novel.items(), key=lambda kv: -kv[1][1])
print(f"  训练库最佳代理分数: {best_train:.3f}")
print(f"  生成候选(全部) {len(generated)} | 新颖候选(相似度<0.5) {len(novel)}")
print()
print(f"  新颖候选 Top-10（按预测代理分数）:")
print(f"  {'SMILES':<46} {'proxy':>7} {'novel':>6} {'MW':>7} {'SA':>5}")
top = []
for smi, (mol, pred, nov) in novel_sorted[:10]:
    mw = Descriptors.MolWt(mol)
    sa = sascorer.calculateScore(mol) if HAS_SA else float("nan")
    top.append((smi, mol, pred, nov, mw, sa))
    print(f"  {smi:<46} {pred:>7.3f} {nov:>6.3f} {mw:>7.1f} {sa:>5.2f}")

with open(os.path.join(OUT, "generated_library.csv"), "w") as f:
    f.write("SMILES,predicted_proxy,novelty,MW\n")
    for smi, (mol, pred, nov) in sorted(generated.items(), key=lambda kv: -kv[1][1]):
        f.write(f"{smi},{pred:.4f},{nov:.4f},{Descriptors.MolWt(mol):.2f}\n")

with open(os.path.join(OUT, "top_candidates.csv"), "w") as f:
    f.write("rank,SMILES,predicted_proxy_score,novelty,MW,SAscore\n")
    for i, (smi, mol, pred, nov, mw, sa) in enumerate(top, 1):
        f.write(f"{i},{smi},{pred:.4f},{nov:.4f},{mw:.2f},{sa:.3f}\n")

mols = [m for _, m, *_ in top]
try:
    img = Draw.MolsToGridImage(
        mols, molsPerRow=5, subImgSize=(340, 340),
        legends=[f"#{i} proxy={p:.2f}" for i, (_, _, p, *_ ) in enumerate(top, 1)])
    if hasattr(img, "save"):
        img.save(os.path.join(OUT, "top_candidates.png"))
except Exception as e:
    print("  结构图生成失败:", e)

print()
print(f"  输出文件已保存到: {OUT}")
for fn in ["library.csv", "predictor_metrics.txt", "parity_plot.png",
           "generated_library.csv", "top_candidates.csv", "top_candidates.png"]:
    p = os.path.join(OUT, fn)
    print(f"    - {fn}  ({'存在' if os.path.exists(p) else '缺失'})")
print()
print("  说明: 本实验以代理标签验证流程可行性；正式研究替换为真实阻燃数据。")