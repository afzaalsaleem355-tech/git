# 面向 PET 的阻燃分子生成式设计（硕士开题项目）

本项目包含"阻燃分子生成式设计"的技术方案、开题报告初稿，以及一个端到端的最小预实验（Proof of Concept）。

## 目录结构

```
.
├── code/                    # 代码
│   ├── run_preexperiment.py   # 预实验主程序（数据→预测器→生成→筛选）
│   ├── md2docx.py             # Markdown 转 Word 工具
│   └── 预实验说明.md           # 预实验说明文档
├── data/                    # 数据
│   ├── library.csv            # 训练分子库与标签
│   ├── generated_library.csv  # 全部生成候选（574 个）
│   ├── top_candidates.csv     # 新颖候选 Top-10
│   └── predictor_metrics.txt  # 预测器评估指标
├── figures/                 # 图
│   ├── parity_plot.png        # 预测散点图（真实 vs 预测）
│   └── top_candidates.png     # 候选分子结构图
├── manuscript/              # 论文 / 文档
│   ├── 阻燃分子生成AI技术方案.md
│   ├── 开题报告_阻燃分子生成式设计_初稿.md
│   └── 开题报告_阻燃分子生成式设计_初稿.docx
├── requirements.txt        # Python 依赖
├── .gitignore
└── README.md
```

## 快速运行预实验

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python code/run_preexperiment.py
```

## 诚实声明

- 预实验使用**专家规则代理分数**模拟"阻燃性"，正式研究需替换为真实成炭率/LOI 数据；
- 生成器采用"芳香骨架 × 阻燃官能团"组合枚举作为基线，后续可升级为 VAE / 遗传算法 / 扩散模型。
