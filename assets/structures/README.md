# UnifiedGroupGen 材料结构图的统一风格

以用户提供的 Materials Project Crystal Toolkit 导出 PNG 为样式参考，使用 VESTA 元素配色、0.5 Å 均一显示半径、柔和哑光球体、半透明配位多面体、分色键、细灰色晶胞边框和左下角三色方向轴。独立结构图使用透明背景 RGBA PNG；正文排版和视频使用白底。二维材料提供等轴、俯视和侧视视图，仅沿面内周期方向展开。

- Ti：RGB (120, 202, 255)；O：(254, 3, 0)。
- Mo：RGB (84, 181, 181)；S：(255, 250, 0)。
- 其他元素从 pymatgen 的 VESTA 颜色表读取。
- 周期边界上的原子可显示多个周期像；不增加独立原子或轨道数量。
- 结构几何来自已有编码状态或已保存 CIF；此过程没有训练模型或重新采样。

## 绘图来源与显示语义

共同绘图实现：`src/unifiedgroupgen/structure_rendering.py`。

本文采用可离线复现的正交投影绘图。显示键按当前几何的共价半径和距离判定：距离不超过共价半径之和的 1.15 倍，在同时含金属和非金属的体系中省略金属—金属显示连线。配位多面体由金属中心的这些显示邻居构造三维凸包；少于四个邻居或共面邻域不生成三维多面体。透明面与球面采用深度遮挡处理。连线与多面体仅用于可视化，不代表化学键判定、网络全部边或结构有效性检验。Crystal Toolkit 网页默认邻接算法为 CrystalNN，因此本地显示键与网页可能存在差别。

`mp-2657_*`、`1MoS2-1_*` 来自论文的真实编码例子。`fixed_*`、`full_*` 来自视频已保存的模型输出。完整输入哈希、周期方向与显示重复倍数见 [manifest.json](manifest.json)。

三色方向轴表示笛卡尔 x（红）、y（绿）、z（蓝）方向，与当前相机一致；二维图中的 z 方向轴不表示新增法向周期。

## 重建

```bash
export MKL_THREADING_LAYER=SEQUENTIAL
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python scripts/render_material_structures.py
python assets/paper/source/build_manuscript_figures.py --examples-only
python scripts/build_model_walkthrough.py --stage render
python scripts/build_model_walkthrough.py --stage verify
```

仅生成 PNG 和更新现有 MP4；训练数据、权重及轨迹不变。视频渲染还需 Pillow、imageio-ffmpeg 和可用中文字体，与模型训练依赖分开。

参考：[Crystal Toolkit 默认结构显示](https://github.com/materialsproject/crystaltoolkit/blob/main/crystal_toolkit/components/structure.py)、[颜色与半径接口](https://github.com/materialsproject/crystaltoolkit/blob/main/crystal_toolkit/core/legend.py)。
