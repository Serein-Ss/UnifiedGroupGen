# UnifiedGroupGen 架构图来源与生成说明

本图描述提交 ebdc7ae 的实际架构，依据 src/unifiedgroupgen/{proposal,symmetry,flow,conditioning,training}.py 与联合模型配置。输出采用内置 image_gen 图像生成接口；工具未公开具体后端型号，不标称为 GPT Image 2。规范来源为用户指定的 ccf-visual-composer，图式采用 formal-roman 与局部几何表示。

## 科学口径

- 顶部为采样时的描述生成。训练连续模型时使用真实描述，而非每一步采样Transformer描述；下方注释明确训练机制。
- 轨道实例携带元素，规则编译为独立位置自由度及群相容晶格度量基底。
- 下方为共享连续模型，独立状态经确定性展开构成原子图；网络候选速度经合法映射得到可积分的独立参数速度。
- H 的平方根均为对称正定矩阵平方根；S为白化参考度量中的对数坐标。层群使用面内2×2矩阵，图中3×3矩阵仅示意三维情况。
- 原子图及输出晶体图为概念示意，不是真实生成样本或稳定材料证据。
- 为保持可读性，图中未展开每个条件编码、群编号embedding、仿射观测投影及CFG的分支细节；这些细节见代码与现有理论稿件。图不包含未来规则内容编码器或磁性模块。
- 当前成果是PNG位图，文字和连线不能作为原生矢量对象编辑。统计图另有可编辑SVG/PDF。

## 初始生成请求

Create a publication-quality scientific mechanism figure for the ACTUAL UnifiedGroupGen model, white background, landscape approximately 2:1, high-resolution, crisp readable English text with Times New Roman-like serif typography and compatible math. This is a paper computation graph, not a poster or dashboard. Restrained teal #167D8D for deterministic symmetry compilation, blue #3468A5 for learned neural modules, terracotta #C96C46 for the continuous transport trajectory, dark #243444 text. Content-fit compact layout, consistent alignment, at least 4% outer margins. No decorative title, logos, watermarks, made-up metrics or material identities. Scientific crystal drawings are clearly schematic.

Use TWO aligned horizontal regions, label them "(a) Structure rules" and "(b) Shared continuous generation". Give the group–orbit compiler strongest emphasis. Keep labels large and minimal, no explanatory paragraphs. Top region 36% height, bottom region 64%, with clean connector lanes and no crossing unrelated labels.

TOP REGION left to right:
- Small input representation labelled "Properties y" with adjacent optional input labelled "Composition c (optional)".
- Arrow into a compact learned stack labelled "Descriptor Transformer", with small exact secondary text "4 layers · 256 channels".
- Arrow into an actual short sequence of colored tokens labelled "a = (group, orbit–element list)". Tokens visually read "G", "(w₁,Z₁)", "(w₂,Z₂)", "STOP". This represents a variable-length list.
- Arrow into a prominent teal operator labelled "Group–orbit compiler". Beside it, two small computation insets showing the real outputs: "F = b + Bq" above an orbit illustration (one representative connected to symmetry-related atoms), and "H = H₀¹ᐟ² exp(Σᵢ kᵢEᵢ) H₀¹ᐟ²" above a symmetric matrix. Final small output label: "Allowed coordinates and lattice". The compiler uses group/Wyckoff tables; small label "Group tables". Connect group tables INTO compiler.
- Do NOT suggest that this computes thermodynamic validity. Do NOT include magnetic, spin, superspace or rule-content encoders; those are unimplemented future ideas.

BOTTOM REGION actual iterative generation, left to right:
1. Initial small distribution/scatter object labelled "Prior z₀", short sublabel "z = (k, q)".
2. Arrow to a state marker labelled "zₜ".
3. Arrow to an operator labelled "Expand Φₐ", then arrow to a small periodic atom graph representation.
4. Arrow into a learned stack labelled "OrbitFlow", small secondary line "6 layers · 512 channels"; nearby label "Shared for space and layer groups". Time t and property context y enter OrbitFlow with small side arrows.
5. Arrow labelled "Candidate velocity" into a teal operator labelled "Legal velocity map". Inset show compact formula "Ḟ = B q̇" and "Ṡ = Σᵢ k̇ᵢEᵢ", which describe the resulting legal motion. Do not show raw candidate vectors as already legal.
6. Arrow into a compact operator labelled "Heun integration".
7. Arrow to a terminal marker labelled "z₁", then arrow to "Expand Φₐ", then two small separate schematic outputs labelled "3D crystal" and "2D layer". Use a genuinely three-dimensional periodic cell and a flat sheet with finite out-of-plane thickness, respectively.

A clean curved feedback arrow from the integration operator back to zₜ, labelled "Next time step", stays below this row. A compiler output line connects into both Expand Φₐ and Legal velocity map using thin teal lines in the gap between regions. This is deterministic conditioning, not another learned network. The top property input also conditions OrbitFlow; can represent this by repeating the small "y" input at OrbitFlow rather than a long crossing line.

Small bottom caption-like line ONLY: "Training: true descriptors and flow-matching velocity targets". No training-data arrows into inference outputs. Use solid dark arrows for computations, thin teal arrows for compiled constraints. Keep exact formula symbols, correct spelling, no additional acronym expansions, no result charts, no claimed performance. All modules must be visible, legible and accurately connected.

## 针对性修正请求

Edit this UnifiedGroupGen scientific architecture figure while preserving its overall composition, all valid labels, font appearance, colors, two horizontal regions, crystal illustrations, and main computation paths. Correct ONLY these scientific errors:
1. In the upper right lattice inset, replace the malformed H0 exponent (currently looks like 12) with a completely unambiguous formula using matrix square-root notation: "H = √H₀ exp(S) √H₀". On a smaller second line place "S = Σᵢ kᵢ Eᵢ". Keep the symmetric matrix illustration below. The square roots are matrix square roots; do NOT write superscript 12.
2. The teal conditioning line feeding the lower Expand and Legal velocity map must originate at the OUTPUT of the Group–orbit compiler, not at the Group tables box. Keep the Group tables arrow pointing UP into the compiler. Draw an independent teal output connector leaving the lower LEFT edge of the compiler, routing in whitespace to both lower modules; it must not touch or pass through Group tables.
3. The bottom curved "Next time step" feedback arrow must start at the bottom of the Heun integration box and end at zₜ, NOT start from the final z₁. Keep the forward Heun → z₁ → Expand → 3D crystal / 2D layer path.
4. In the lower legal velocity formula ensure the time derivative on k is visible: "Ṡ = Σᵢ k̇ᵢ Eᵢ". Preserve "Ḟ = B q̇".
No other modules or labels added, no change to dataflow, no performance claim. Preserve sharp legibility, margins, existing white background and wide aspect ratio.

## 已完成核查

初稿存在H0平方根被误排为12次幂、编译条件线起点不清、反馈线从终态出发的问题。最终图已检查：sqrt(H0)公式正确，Group tables仅输入compiler，编译输出分别约束展开与合法速度映射，Heun反馈至中间态，终态展开为3D/2D输出。模块层数和宽度与联合配置一致。使用PNG栅格，未声称字体嵌入或精确矢量编辑性。

## 2026-10-18 结构显示风格统一

在原 PNG 上仅调整材料插图为 Crystal Toolkit 风格：白底、光照球体、VESTA 配色、细灰晶胞边框与分色显示键。已核对模块、参数量级标注、方程和 Heun 反馈路径。插图保持方法示意用途；原子坐标与晶格数值以可复现的 fig01_material_examples.png 为准。参考图仅使用本项目已有公开方法图和从已保存状态绘制的结构图，没有发送检查点、原始数据集、训练日志或完整文稿。

### 用户导出图确认后的样式

用户随后提供了实际 Crystal Toolkit 导出 PNG。最终风格据此调整为柔和哑光球体、半透明配位多面体、细灰边框和三色方向轴。独立数值结构图导出透明 PNG；方法图和视频保留白色排版背景。颜色、独立状态与真实轨迹保持原值，参考图只用于外观，不采用其材料坐标。原始参考 PNG 留在用户本地，未纳入仓库。
