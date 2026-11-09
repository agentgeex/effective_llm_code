#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ══════════════════════════════════════════════════════════════════
# ① 这是什么：微调数据管道 —— 模板合成问答对 → 规则清洗 → 字符级 TF-IDF 聚类三合一划分
# ② 对应章节：第 11 章 §11.3.1 / §11.3.2 / §11.3.3
# ③ 怎么跑：python ch11/cases/finetune/prepare_data.py
# ④ 依赖：scikit-learn（TfidfVectorizer）。无 API key、无模型权重
# ⑤ 实测：Python 3.10.12 + scikit-learn 1.7.2，rc=0（数据落 data/，统计落 outputs/）
# ⑥ 与书里的差异：书里用「大模型生成 + 裁判模型打分 + 句向量聚类」，本文件在没有 API key 的前提下
#    降级为「模板合成 + 规则审查 + 字符级 TF-IDF」，详见 README「已知限制」
# ══════════════════════════════════════════════════════════════════
"""prepare_data.py — ch11 §11.3 的第 1~3 步：采集 → 清洗 → 语义划分

对应第 11 章 §11.3.1 / §11.3.2 / §11.3.3。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ 先读这一段，否则会误读后面的所有数字
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
书里这三步的做法是：
    §11.3.1  用 GPT-4o 按「领域上下文注入」生成问答对
    §11.3.2  用 Claude 做裁判模型给每条打分
    §11.3.3  用 sentence-transformers 把问题嵌入 + DBSCAN 聚类切分

本案例**没有 API key**，也不允许下载新模型，所以：
    §11.3.1 的「生成」→ 改成**模板合成**：48 条人工撰写的领域事实 × 6 种问法
    §11.3.2 的「裁判」→ 改成**规则审查**：违禁译法 / 客服腔 / 过短 / 重复
    §11.3.3 的「嵌入」→ 改成**字符级 TF-IDF**（汉字无空格，字符 n-gram 比词级稳）

这不是「等价替代」，是**降级替代**。降级的代价写进 README 局限第 1 条：
模板合成的数据风格极其单一，训出来的模型也只会那一种语气。

另外：SEED_FACTS 里的 CT 维保内容是**教学用示意**，不是厂家维修手册原文摘录，
也没有逐条核对过任何一份真实手册。这一节讲的是数据管道，不是维保知识本身；
里面出现的参数与阈值不能当作操作规程。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

产物：
    data/train.jsonl  data/val.jsonl  data/test.jsonl   （git 不入库，见 .gitignore）
    outputs/dataset_stats.json
    outputs/01_prepare.log
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUT_DIR = HERE / "outputs"
SEED = 42

# ══════════════════════════════════════════════════════════════════
# 一、种子事实（48 条 × 4 类）—— 这一步对应书稿「知识审计」
# ══════════════════════════════════════════════════════════════════
# 字段：id / q（基准问法）/ a（答案）/ cat / diff / doc（来源标号）
# difficulty：easy=基础概念  medium=参数关系+常规程序  hard=故障分析

SEED_FACTS = [
    # ── basic_concept（12 条，easy）────────────────────────────────
    dict(id="b01", cat="basic_concept", diff="easy", doc="svc_manual_ch2",
         q="球管是什么",
         a="球管是 CT 的 X 射线发生部件，全称 X 射线管，由阴极灯丝和旋转阳极靶盘组成。阴极发射的电子在高压电场下加速轰击阳极靶面，动能绝大部分转为热量、约 1% 转为 X 射线。球管是整机唯一的耗材型核心部件，它的热容量和累计曝光秒数决定了更换周期。"),
    dict(id="b02", cat="basic_concept", diff="easy", doc="svc_manual_ch2",
         q="探测器是什么",
         a="探测器是把穿过人体的 X 射线转换成电信号的部件，位于扫描架内与球管相对的一侧。它由数百到数千个独立通道组成，每个通道对应重建图像里的一条投影路径。通道之间的增益必须一致，否则重建后会出现环状伪影。注意它不叫探头——探头是超声设备的部件，混用会让技师找错位置。"),
    dict(id="b03", cat="basic_concept", diff="easy", doc="qc_sop_1",
         q="空气校准是什么",
         a="空气校准是不放任何物体、只扫空气的一次归一化过程，用于把探测器各通道的响应拉到同一基准。它不是质控检测，也不产生剂量报告：质控检测是定期验证整机性能，空气校准是修正通道一致性。球管更换、探测器模块更换或图像出现环状伪影后，都要重新做一次空气校准。"),
    dict(id="b04", cat="basic_concept", diff="easy", doc="svc_manual_ch2",
         q="热容量是什么",
         a="热容量是球管阳极能够累积吸收的热量上限，单位 HU（Heat Unit）。一次曝光产生的热量与管电压、管电流和曝光时间相关，连续扫描时热量逐次累加。当累计热量接近额定热容量时，系统会强制等待冷却，表现为扫描中断或曝光按钮无响应。球管寿命按累计曝光秒数报，不按曝光次数报。"),
    dict(id="b05", cat="basic_concept", diff="easy", doc="svc_manual_ch5",
         q="环形伪影是什么",
         a="环形伪影是重建图像上以旋转中心为圆心的一圈或多圈环状条纹，本质是探测器某个通道（或几个相邻通道）的响应与相邻通道不一致。该通道的读数在全周扫描里始终偏低或偏高，投影重建后就被拉成了一个环。常见成因有三种：通道增益漂移、单个探测模块失效、球管更换后未重新做空气校准。"),
    dict(id="b06", cat="basic_concept", diff="easy", doc="svc_manual_ch1",
         q="CT 值是什么",
         a="CT 值是以水的线性衰减系数为基准的相对值，单位 HU，水的 CT 值定义为 0。空气约为 -1000，骨皮质可达 +1000 以上。CT 值不是绝对物理量，它依赖管电压、校准状态和重建条件；同一组织在不同设备或不同校准状态下测得的 CT 值会有几个 HU 的差异，这是正常的。"),
    dict(id="b07", cat="basic_concept", diff="easy", doc="dose_sop",
         q="CTDIvol 是什么",
         a="CTDIvol 是容积 CT 剂量指数，单位 mGy，表示一次扫描在标准模体上的平均剂量水平。它由 CTDIw 除以螺距得到，只描述设备输出，不描述患者实际吸收剂量。报告剂量时不能只写「剂量」，必须写清是 CTDIvol 还是 DLP，两者的量纲和含义都不同。"),
    dict(id="b08", cat="basic_concept", diff="easy", doc="dose_sop",
         q="DLP 是什么",
         a="DLP 是剂量长度乘积，单位 mGy·cm，等于 CTDIvol 乘以扫描长度。它把剂量水平和受照范围合在一起，是估算有效剂量的中间量。同一个 CTDIvol 下，扫描范围越长 DLP 越大。写剂量报告时 CTDIvol 和 DLP 要成对出现，只给一个数不构成完整记录。"),
    dict(id="b09", cat="basic_concept", diff="easy", doc="svc_manual_ch2",
         q="准直器的作用是什么",
         a="准直器位于球管出线侧，用可调铅门把 X 射线束限制在设定的层厚范围内。它决定了扫描的线束宽度，也直接影响剂量：线束越宽，探测器接收到的散射线越多，剂量效率越低。日常质控里检查准直器，主要看光野与扫描野是否对齐。"),
    dict(id="b10", cat="basic_concept", diff="easy", doc="svc_manual_ch2",
         q="滑环是什么",
         a="滑环是连接旋转部分与静止部分的电力和信号传输部件，取代了早期 CT 的拖缆。它让扫描架可以连续旋转，从而实现螺旋扫描。滑环接触面磨损或污染会导致信号噪声升高、偶发数据丢包，表现为图像上出现随机条纹而不是固定环状伪影。"),
    dict(id="b11", cat="basic_concept", diff="easy", doc="svc_manual_ch2",
         q="高压发生器的作用是什么",
         a="高压发生器把市电升压到 70 至 140 kV 的直流高压，供给球管产生 X 射线。它同时负责管电压和管电流的闭环调节，kV 与 mA 的稳定性直接决定 CT 值的可重复性。高压发生器故障的典型表现是曝光中断、kV 偏差报警，而不是图像质量问题。"),
    dict(id="b12", cat="basic_concept", diff="easy", doc="svc_manual_ch6",
         q="重建是什么",
         a="重建是把原始投影数据还原成断层图像的计算过程，常用滤波反投影或迭代重建。重建核（kernel）决定图像锐度与噪声的权衡：骨核锐度高、噪声大，软组织核平滑、噪声低。同一组原始数据可以用不同重建核生成不同观感的图像，所以调窗和换核不能混为一谈。"),

    # ── parameter_relation（12 条，medium）─────────────────────────
    dict(id="p01", cat="parameter_relation", diff="medium", doc="svc_manual_ch2",
         q="球管热容量和曝光秒数是什么关系",
         a="热容量是上限，曝光秒数是累计消耗。每次曝光把热量加进阳极，两次曝光之间的冷却速率由阳极热传导和散热方式决定。累计曝光秒数接近厂家给出的寿命值时，即使热容量还没到顶，球管输出也会因靶面磨损而衰减。评估球管剩余寿命要同时看这两个量，只看热容量会高估。"),
    dict(id="p02", cat="parameter_relation", diff="medium", doc="qc_sop_1",
         q="探测器通道增益漂移和环状伪影是什么关系",
         a="通道增益漂移是原因，环状伪影是结果，两者是单向因果关系。漂移量超过约 1% 时开始可见，超过 3% 时环带会明显影响诊断。判别的关键是做空气校准：校准后伪影消失属漂移，仍在则大概率是探测模块硬件故障。所以看到环状伪影的第一步不是换件，是校准。"),
    dict(id="p03", cat="parameter_relation", diff="medium", doc="svc_manual_ch2",
         q="管电压和 CT 值是什么关系",
         a="管电压升高会让高能光子比例增加，硬化效应减弱，同一组织的 CT 值会略微下降，骨与软组织之间的对比度也会变化。这就是为什么不同 kV 下测的水模 CT 值要分别记录。校准必须绑定管电压：在 120 kV 下做的空气校准，不能直接用于 100 kV 的协议。"),
    dict(id="p04", cat="parameter_relation", diff="medium", doc="dose_sop",
         q="管电流时间和剂量是什么关系",
         a="管电流时间乘积（mAs）与剂量近似成正比，这是剂量优化的第一根杠杆。但 mAs 翻倍只带来约 1.4 倍的图像信噪比提升（噪声按平方根下降），所以「加倍 mAs 换清晰度」的性价比很低。实际调参应优先改重建核和迭代强度，再考虑动 mAs。"),
    dict(id="p05", cat="parameter_relation", diff="medium", doc="dose_sop",
         q="螺距和剂量是什么关系",
         a="螺距等于进床距离除以线束宽度，它与 CTDIvol 成反比：螺距增大，单位长度上的曝光次数减少，剂量下降。但螺距过大会带来层面敏感度曲线变宽、小病灶漏检的风险。所以「加大螺距降剂量」不能无条件用，要看检查目的。"),
    dict(id="p06", cat="parameter_relation", diff="medium", doc="svc_manual_ch2",
         q="准直宽度和层厚是什么关系",
         a="准直宽度是硬件决定的线束宽度，重建层厚是从采集数据里插值出来的显示厚度，两者不必相等。用窄准直配薄层重建能得到各向同性的数据，代价是剂量效率降低。理解了这一点，就不会把「层厚 1 mm」误当成「线束只有 1 mm 宽」。"),
    dict(id="p07", cat="parameter_relation", diff="medium", doc="svc_manual_ch6",
         q="重建核和图像噪声是什么关系",
         a="重建核越锐，空间分辨率越高，噪声也越大；核越平滑，噪声越低，但细节被抹平。这是一条固定的权衡曲线，不是设备故障。同一患者同一次扫描换核，图像观感会明显不同，但剂量和原始数据完全没变。所以评价图像质量时必须先固定重建核。"),
    dict(id="p08", cat="parameter_relation", diff="medium", doc="qc_sop_1",
         q="水模 CT 值和校准状态是什么关系",
         a="水模中心区域的 CT 值应接近 0 HU，允许误差通常在 ±5 HU 量级（具体以厂家质控手册为准）。校准过期或探测器通道漂移时，水模 CT 值会整体偏移或出现中心与周边不一致。所以水模扫描是校准状态最直接的验证手段，比看图像「像不像」可靠得多。"),
    dict(id="p09", cat="parameter_relation", diff="medium", doc="svc_manual_ch2",
         q="球管输出衰减和校准周期是什么关系",
         a="球管输出随使用衰减，衰减后同样的 kV 与 mAs 产生的光子数变少，图像噪声上升。空气校准的作用之一就是把这个变化归一化掉。因此球管接近寿命末期时，校准周期要缩短，否则会出现「校准后图像尚可、一周后又变差」的反复。"),
    dict(id="p10", cat="parameter_relation", diff="medium", doc="svc_manual_ch3",
         q="滑环转速和时间分辨率是什么关系",
         a="机架旋转一周的时间越短，同一角度区间采到的数据越少，时间分辨率越高但单圈光子数下降。心脏扫描要求高时间分辨率，因此常用半扫描重建配合更快的转速，代价是必须提高 mAs 来补偿噪声。转速、mAs、时间分辨率三者是联动的，不能单独调一个。"),
    dict(id="p11", cat="parameter_relation", diff="medium", doc="qc_sop_1",
         q="探测器温度和通道漂移是什么关系",
         a="探测器对温度敏感，开机后需要预热到热平衡状态，通道响应才稳定。冷机直接扫描时，通道输出会有缓慢漂移，表现为图像噪声偏高或轻环状伪影。所以质控流程要求开机预热后再做空气校准，冷机校准的结果不可信。"),
    dict(id="p12", cat="parameter_relation", diff="medium", doc="dose_sop",
         q="扫描长度和 DLP 是什么关系",
         a="DLP 是 CTDIvol 与扫描长度的乘积，长度是线性因子：扫描范围加倍，DLP 加倍，而 CTDIvol 不变。这就是为什么定位像上把范围画得过长会显著抬高剂量。优化剂量的一个直接动作，就是把扫描范围收到临床需要的边界。"),

    # ── failure_analysis（12 条，hard）──────────────────────────────
    dict(id="f01", cat="failure_analysis", diff="hard", doc="svc_manual_ch5",
         q="图像出现同心圆环状条纹怎么查",
         a="先按三步走：一是看环的圈数和位置，单圈细环指向单个通道，多圈粗环指向多个相邻通道；二是做一次空气校准再扫水模，环消失说明是增益漂移，仍在说明是硬件；三是查探测器通道自检记录，定位具体模块。跳过校准直接换件，是这类故障最常见的过度维修。"),
    dict(id="f02", cat="failure_analysis", diff="hard", doc="svc_manual_ch5",
         q="图像出现随机条纹怎么查",
         a="随机条纹（不成环、位置每次不同）通常指向数据链路而不是探测器本身，重点查滑环接触面污染或磨损、光缆连接、重建计算机的内存与丢包计数。判别方法是连续扫三次水模：条纹位置随扫描变化且不重复，就基本排除固定通道问题，转向传输链路排查。"),
    dict(id="f03", cat="failure_analysis", diff="hard", doc="svc_manual_ch5",
         q="金属伪影怎么处理",
         a="金属伪影由高衰减物体的光子饥饿和射束硬化引起，表现为放射状亮暗条带。处置顺序是：先用迭代重建或金属伪影抑制算法；再调整扫描角度避开金属长轴；最后才考虑提高 kV 或增加 mAs。硬件层面没有能「消除」金属伪影的部件，这是物理限制不是故障。"),
    dict(id="f04", cat="failure_analysis", diff="hard", doc="svc_manual_ch4",
         q="曝光中断并报高压故障怎么处理",
         a="先读故障代码再看现象：高压发生器报过流或打火，多与球管内部真空度下降或高压电缆绝缘老化有关。处置顺序是记录代码、查球管累计曝光秒数、做高压电缆绝缘测试，不要直接更换高压发生器。球管打火往往在冷机首次曝光时最明显，这一点可作为辅助判据。"),
    dict(id="f05", cat="failure_analysis", diff="hard", doc="svc_manual_ch5",
         q="某个探测器模块失效的表现是什么",
         a="单个模块失效会在重建图像上留下固定位置的扇形或环状伪影，并且空气校准无法消除。进一步确认要看原始数据的通道响应曲线：失效模块对应的通道读数接近零或明显偏离邻道。更换模块后必须重做空气校准和水模验证，否则新模块与旧模块的增益差会引入新的伪影。"),
    dict(id="f06", cat="failure_analysis", diff="hard", doc="qc_sop_1",
         q="空气校准做不过去怎么办",
         a="先排查三类原因：球管输出不足（累计曝光秒数接近寿命）、探测器温度未达热平衡（开机时间不够）、扫描架内有余留异物或滤线栅污染。逐项排除后再看校准日志里的具体失败通道号。反复重试校准而不查原因，只会浪费时间并加重球管负担。"),
    dict(id="f07", cat="failure_analysis", diff="hard", doc="svc_manual_ch4",
         q="图像噪声突然升高怎么查",
         a="噪声升高要先分清是「全幅升高」还是「局部升高」。全幅升高多为球管输出衰减或 mAs 设置变动；局部升高多为对应区域的探测器通道异常。接着对比同协议的历史图像和当时的剂量报告，若 CTDIvol 没变而噪声升高，优先怀疑球管输出或校准状态。"),
    dict(id="f08", cat="failure_analysis", diff="hard", doc="qc_sop_1",
         q="水模 CT 值漂移超出允差怎么办",
         a="先重做空气校准再复测，多数漂移可在校准后回到允差内。若校准后仍超差，检查探测器温度是否稳定、球管是否接近寿命末期、模体本身是否放正。三项都正常而 CT 值仍持续偏移，才考虑高压发生器 kV 输出不准，需用 kV 表实测确认。"),
    dict(id="f09", cat="failure_analysis", diff="hard", doc="svc_manual_ch3",
         q="滑环接触不良怎么确认",
         a="典型证据是丢包计数随时间增长、图像偶发条纹、严重时机架旋转报错停机。确认方法是查系统日志里的数据传输错误计数，并在停机状态下检查滑环表面有无碳粉堆积或划痕。清洁滑环必须按厂家规程并使用专用清洁剂，用普通溶剂会损伤接触面。"),
    dict(id="f10", cat="failure_analysis", diff="hard", doc="svc_manual_ch6",
         q="重建报错或重建时间异常变长怎么办",
         a="先看是单序列还是全部序列：单序列报错多为原始数据缺失或投影数不足；全部变慢多为重建计算机资源耗尽（磁盘满、内存被占、散热不良）。处置顺序是清磁盘、重启重建服务、再查硬件。直接重装系统是最后手段，会丢掉校准与配置数据。"),
    dict(id="f11", cat="failure_analysis", diff="hard", doc="svc_manual_ch5",
         q="图像上出现固定位置的黑点或白点怎么查",
         a="固定位置的点状异常通常是单个探测器通道的死像素或饱和像素。判别依据是位置在多次扫描中完全一致，且空气校准后不消失。数量少时可软件标记补偿，数量增长则说明模块在劣化，应按模块更换而不是逐点修补。"),
    dict(id="f12", cat="failure_analysis", diff="hard", doc="svc_manual_ch4",
         q="机架不能旋转或旋转异响怎么处理",
         a="先区分电气与机械：不能旋转但无异响，多查驱动使能与安全联锁（门禁、急停、防夹）；有异响或不均匀阻力，多查轴承、皮带张力、滑环摩擦。任何情况下都不要在未断电挂牌时伸手进扫描架，这是维保作业的硬性安全前提。"),

    # ── procedure（12 条，medium/hard）──────────────────────────────
    dict(id="r01", cat="procedure", diff="medium", doc="qc_sop_1",
         q="空气校准的标准步骤是什么",
         a="第一步确认扫描架内无任何物体、床板退到校准位；第二步确认开机预热已达到热平衡（按厂家要求的时间）；第三步在服务界面选择空气校准并等待完成；第四步校准后立即扫水模验证 CT 值与噪声；第五步记录校准时间、球管累计曝光秒数、结果。缺第四步等于没做验证。"),
    dict(id="r02", cat="procedure", diff="medium", doc="qc_sop_1",
         q="日常质控要做哪些项",
         a="日检做水模 CT 值与噪声、图像有无明显伪影；周检做剂量输出一致性；月检做 CT 值线性、层厚准确性、定位光精度。每次都要留记录并签名。质控的目的不是「证明设备没问题」，而是尽早发现漂移趋势，所以趋势比单次数值更重要。"),
    dict(id="r03", cat="procedure", diff="hard", doc="svc_manual_ch4",
         q="更换球管后要做哪些校准",
         a="更换球管后至少要做：球管热身与老化训练、空气校准、水模 CT 值与噪声验证、剂量输出测量、以及一次完整的质控扫描。顺序不能颠倒——没做空气校准就做剂量测量，得到的数没有意义。全部完成后才可恢复临床使用，并在工单上记录新球管的序列号与初始曝光秒数。"),
    dict(id="r04", cat="procedure", diff="hard", doc="svc_manual_ch5",
         q="更换探测器模块后怎么验证",
         a="更换后先做空气校准，再扫水模看环状伪影是否消失、CT 值是否回到允差内，最后做一次人体等效模体的图像目视检查。三项全过才算修复。若环仍在，要确认新模块的固件版本与整机匹配——不同批次的模块增益曲线不同，未做匹配会引入新的环。"),
    dict(id="r05", cat="procedure", diff="medium", doc="svc_manual_ch4",
         q="球管寿命怎么评估",
         a="同时看三个量：累计曝光秒数与厂家寿命值之比、热容量使用率、以及最近一次空气校准后的图像噪声相对新球管的变化。三个量一致指向衰减，才建议更换。只凭「用了几年」判断是不专业的，因为使用强度差异极大。"),
    dict(id="r06", cat="procedure", diff="medium", doc="svc_manual_ch4",
         q="故障代码应该怎么用",
         a="故障代码是入口不是结论。正确用法是：先在服务手册里查到代码对应的子系统，再看同一时间窗内的关联代码，最后结合现象（能否曝光、有无异响、图像如何）定位。只看一个代码就换件，是误换件的主要来源。代码含义在不同软件版本间可能变化，必须以本机版本的册子为准。"),
    dict(id="r07", cat="procedure", diff="medium", doc="svc_manual_ch3",
         q="设备需要停机维修时怎么处置",
         a="先通知临床并登记停机时间，再在设备上挂「禁止使用」标识、断开患者登记入口，涉及高压或旋转部件的作业必须断电挂牌上锁。修复后按校准与质控流程验证，通过后才撤标识。停机期间的患者分流要提前和科室约定，不能等到设备停了才协调。"),
    dict(id="r08", cat="procedure", diff="medium", doc="svc_manual_ch3",
         q="维保工单要写清哪些内容",
         a="至少写清六项：故障现象与发生时间、读取到的故障代码、做过的检查与结论、更换的部件及序列号、做过的校准与质控结果、恢复使用的时间。不要把「已处理」当结论——三个月后同一台设备再出问题，这张工单就是唯一的线索。"),
    dict(id="r09", cat="procedure", diff="medium", doc="dose_sop",
         q="剂量记录要留哪些数据",
         a="每次扫描协议至少留 CTDIvol、DLP、扫描长度、kV 与 mAs。质控层面还要留剂量输出测量值与测量日期。这些数据用于横向比对和纵向趋势分析，缺项会让剂量审计无法追溯。记录时不要只存截图，要存可导出的结构化数据。"),
    dict(id="r10", cat="procedure", diff="hard", doc="svc_manual_ch4",
         q="更换高压电缆要注意什么",
         a="断电放电后按规程验电，拆装时避免过度弯折（弯曲半径不足会损伤绝缘），接头清洁后按扭矩要求紧固。装回后先做绝缘测试再上高压，首次曝光从小条件开始。跳过绝缘测试直接满负荷曝光，是引发高压打火和二次损坏的常见原因。"),
    dict(id="r11", cat="procedure", diff="medium", doc="qc_sop_1",
         q="质控不合格时的处理顺序是什么",
         a="先复核测量条件（模体位置、协议、温度），排除操作误差；能通过校准修正的先校准；校准无效再定位到部件；确认部件故障后走更换流程。每一步都要留记录。直接跳到「换件」会掩盖真实的漂移趋势，也无法防止同类问题复发。"),
    dict(id="r12", cat="procedure", diff="medium", doc="svc_manual_ch3",
         q="开机自检报错但能正常扫描怎么办",
         a="自检报错不能因为「能扫」就忽略。先记录报错项并对照手册确认属于哪一类：属于安全联锁类的必须停机处理；属于非关键传感器类的可登记观察，但要设定复查期限。把报错当噪音屏蔽，会让真正的问题在下一次故障里以更严重的形式出现。"),
]

# ── 故意掺入的违规样本（证明清洗规则真的在干活）──────────────────
BAD_SAMPLES = [
    dict(id="x01", cat="basic_concept", diff="easy", doc="gen_v1",
         q="球管是什么",
         a="这是一个产生 X 光的重要部件，在检查过程中非常重要哦，你可以试试多了解一下。"),
    dict(id="x02", cat="basic_concept", diff="easy", doc="gen_v1",
         q="探测器是什么",
         a="探测器就是机器里那个探头。"),
    dict(id="x03", cat="parameter_relation", diff="medium", doc="gen_v1",
         q="球管热容量多少",
         a="这是一个很好的问题！球管热容量通常是 8 MHU，希望对你有所帮助。"),
    dict(id="x04", cat="failure_analysis", diff="hard", doc="gen_v1",
         q="图像有环状伪影怎么办",
         a="请联系维修人员处理。"),
    dict(id="x05", cat="procedure", diff="medium", doc="gen_v1",
         q="空气校准怎么做",
         a="在服务界面点一下空气校准就可以了，机器会自己做完。"),
    dict(id="x06", cat="basic_concept", diff="easy", doc="gen_v1",
         q="CTDIvol 是什么",
         a="CTDIvol 是剂量当量指数，用来表示扫描的辐射量。"),
]

# ══════════════════════════════════════════════════════════════════
# 二、问法模板 —— 一条事实 × 6 种问法 = 「同题异问」
# ══════════════════════════════════════════════════════════════════
# §11.3.3 讲的就是这个东西：同一个知识点换 6 种问法，
# 切分时必须把 6 种问法分到不同数据集，测试集问的是「没见过的问法」。

QUESTION_TEMPLATES = [
    "{q}？",
    "请说明：{q}。",
    "{q}，请详细回答。",
    "作为 CT 设备维保工程师，{q}？",
    "我想知道{q}。",
    "{q}？请说明判断依据。",
]

# ══════════════════════════════════════════════════════════════════
# ---- 示例 1： 对应 §11.3.2（裁判模型评估问答对）----
# 三、清洗规则（对应 §11.3.2「裁判模型」的降级版）
# ══════════════════════════════════════════════════════════════════

# 书里明文禁止的错误译法（领域上下文里的「禁止」块）
BANNED_TRANSLATIONS = {
    "X 光管": "球管",
    "探头": "探测器",
    "球馆": "球管",
    "剂量当量指数": "CTDIvol",
}

# 客服腔 / 口水话（书里明确要求「不要『你可以试试』『这是一个很好的问题』」）
CHATTY_PATTERNS = [
    "你可以试试", "这是一个很好的问题", "希望对你有所帮助",
    "非常重要哦", "了解一下", "总的来说", "希望对你有帮助",
    "请联系维修人员处理",   # 空答案：不构成领域知识
]

MIN_ANSWER_CHARS = 40      # 短于此判为「信息不足」
MAX_ANSWER_CHARS = 600


def normalize(text: str) -> str:
    """全角转半角空格、压缩空白、去首尾。"""
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def clean_one(rec: dict) -> tuple[dict | None, str]:
    """返回 (清洗后记录, 丢弃原因)。清洗后记录为 None 表示被丢弃。"""
    q = normalize(rec["q"])
    a = normalize(rec["a"])

    # ① 违禁译法：出现即丢（硬约束，不自动改写，防止误伤）
    for bad, good in BANNED_TRANSLATIONS.items():
        if bad in a or bad in q:
            return None, f"违禁译法「{bad}」（应为「{good}」）"

    # ② 客服腔 / 空答案
    for pat in CHATTY_PATTERNS:
        if pat in a:
            return None, f"客服腔/空答案「{pat}」"

    # ③ 长度
    if len(a) < MIN_ANSWER_CHARS:
        return None, f"答案过短（{len(a)} < {MIN_ANSWER_CHARS} 字）"
    if len(a) > MAX_ANSWER_CHARS:
        return None, f"答案过长（{len(a)} > {MAX_ANSWER_CHARS} 字）"

    return dict(q=q, a=a), ""


def build_raw() -> list[dict]:
    """§11.3.1 的降级版：模板合成问答对。"""
    raw = []
    for fact in SEED_FACTS + BAD_SAMPLES:
        for tpl in QUESTION_TEMPLATES:
            raw.append(dict(
                q=tpl.format(q=fact["q"]),
                a=fact["a"],
                fact_id=fact["id"],
                cat=fact["cat"],
                diff=fact["diff"],
                doc=fact["doc"],
            ))
    return raw


def dedup(records: list[dict]) -> tuple[list[dict], int]:
    """按 (q, a) 去重，保留首次出现。"""
    seen, kept = set(), []
    for r in records:
        key = (r["q"], r["a"])
        if key in seen:
            continue
        seen.add(key)
        kept.append(r)
    return kept, len(records) - len(kept)


# ══════════════════════════════════════════════════════════════════
# ---- 示例 2： 对应 §11.3.3（语义去重 + 三合一划分）----
# 四、语义聚类切分（对应书稿 §11.3.3）
# ══════════════════════════════════════════════════════════════════
# 书稿：SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2') + DBSCAN
# 本案例：TfidfVectorizer(char 1~3gram) + DBSCAN
# 为什么换：那个嵌入模型本地没有，下载要 ~470MB，且书稿明确说「不许下载新模型」
# 代价见 README 局限第 2 条：字符 n-gram 只抓表面相似，聚类纯度实测 88%（50 簇 vs 真值 48 簇）。
#       也就是说有若干簇把两个知识点混在一起了 —— 书稿用真嵌入模型时这个数会更高。
#
# eps 是这套方案唯一的旋钮，必须**扫一遍再定**，不能拍脑袋（书稿写 eps=0.3，
# 那是 sentence-embedding 的余弦尺度；换到 TF-IDF 字符 n-gram 上同一个数字完全不通用）。

EPS_CANDIDATES = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
DEFAULT_EPS = 0.35          # 扫描结果：簇数最接近真值 48，同时纯度仍 ≥85%


def build_matrix(questions: list[str]):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=1)
    return vec.fit_transform(questions)


def cluster_with(X, eps: float) -> list[int]:
    from sklearn.cluster import DBSCAN
    return DBSCAN(eps=eps, min_samples=1, metric="cosine").fit_predict(X).tolist()


def cluster_questions(questions: list[str], eps: float = DEFAULT_EPS) -> list[int]:
    return cluster_with(build_matrix(questions), eps)


def sweep_eps(questions: list[str], fact_ids: list[str]) -> list[dict]:
    """扫 eps，报簇数与纯度。

    纯度 = 「簇内只含一个 fact_id」的簇占比。
    ⚠️ 只有在人造数据上才算得出来（我们知道每条属于哪个知识点）。
    真实场景没有这个真值，只能用眼看法或人工抽检 —— 这点在 README 里写明了。
    """
    X = build_matrix(questions)
    rows = []
    for eps in EPS_CANDIDATES:
        labels = cluster_with(X, eps)
        groups: dict[int, list[str]] = defaultdict(list)
        for fid, lab in zip(fact_ids, labels):
            groups[lab].append(fid)
        pure = sum(1 for v in groups.values() if len(set(v)) == 1)
        rows.append({
            "eps": eps,
            "clusters": len(groups),
            "pure_clusters": pure,
            "purity_pct": round(pure / max(1, len(groups)) * 100, 1),
        })
    return rows


def split_by_cluster(records: list[dict], labels: list[int],
                     rng: random.Random) -> tuple[list, list, list]:
    """书稿的分配规则：同一 cluster 的变体必须分散到不同数据集。"""
    by_cluster: dict[int, list[int]] = defaultdict(list)
    for idx, lab in enumerate(labels):
        by_cluster[lab].append(idx)

    train_i, val_i, test_i = [], [], []
    for lab, idxs in by_cluster.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n = len(idxs)
        if n == 1:
            train_i.extend(idxs)
        elif n == 2:
            train_i.append(idxs[0])
            val_i.append(idxs[1])
        else:
            n_train = max(1, int(n * 0.8))
            train_i.extend(idxs[:n_train])
            rest = idxs[n_train:]
            if rest:
                val_i.append(rest[0])
                test_i.extend(rest[1:])
    return ([records[i] for i in train_i],
            [records[i] for i in val_i],
            [records[i] for i in test_i])


# ══════════════════════════════════════════════════════════════════
# 五、输出
# ══════════════════════════════════════════════════════════════════

def to_jsonl(records: list[dict], path: Path) -> None:
    """写成训练用的对话格式（书稿 §11.3.2 的 messages 结构）。"""
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            obj = {
                "messages": [
                    {"role": "user", "content": r["q"]},
                    {"role": "assistant", "content": r["a"]},
                ],
                "metadata": {
                    "difficulty": r["diff"],
                    "category": r["cat"],
                    "source_quality": "verified" if r["doc"] != "gen_v1" else "generated",
                    "source_doc": r["doc"],
                    "fact_id": r["fact_id"],
                    "cluster": r["cluster"],
                },
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def pct(vals: list[int], p: float) -> int:
    if not vals:
        return 0
    s = sorted(vals)
    k = min(len(s) - 1, max(0, int(round((p / 100) * (len(s) - 1)))))
    return s[k]


def main() -> int:
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    log = []

    def say(s: str = "") -> None:
        print(s)
        log.append(s)

    say("=" * 62)
    say("ch11 §11.3 数据管道：采集 → 清洗 → 语义划分")
    say("=" * 62)

    # ── 1. 采集 ───────────────────────────────────────────────────
    raw = build_raw()
    say(f"\n[1] 采集（模板合成，种子事实 {len(SEED_FACTS)} 条 + 故意掺入违规 {len(BAD_SAMPLES)} 条）")
    say(f"    问法模板 {len(QUESTION_TEMPLATES)} 种 × 每条事实 → 原始 {len(raw)} 条")

    # ── 2. 清洗 ───────────────────────────────────────────────────
    say(f"\n[2] 清洗（规则版本，替代书稿的裁判模型）")
    dropped: list[tuple[str, str]] = []
    cleaned = []
    for r in raw:
        rec, why = clean_one(r)
        if rec is None:
            dropped.append((r["q"][:28], why))
        else:
            rec.update(fact_id=r["fact_id"], cat=r["cat"], diff=r["diff"], doc=r["doc"])
            cleaned.append(rec)

    drop_reasons = Counter(w for _, w in dropped)
    say(f"    规则剔除 {len(dropped)} 条：")
    for reason, n in drop_reasons.most_common():
        say(f"      - {reason}: {n} 条")

    cleaned, dup_n = dedup(cleaned)
    say(f"    精确去重（q+a 相同）再剔 {dup_n} 条")
    say(f"    清洗后 {len(cleaned)} 条")

    # ── 3. 语义聚类 + 切分 ────────────────────────────────────────
    say(f"\n[3] 语义聚类（TF-IDF char 1~3gram + DBSCAN，书稿用 SentenceTransformer）")
    qs = [r["q"] for r in cleaned]
    fids = [r["fact_id"] for r in cleaned]
    say(f"    真值: {len(set(fids))} 个知识点（人造数据才有真值，真实场景没有）")
    say(f"    eps 扫描（eps 越小簇越碎，越大越容易把两个知识点并一簇）:")
    sweep = sweep_eps(qs, fids)
    for row in sweep:
        mark = "  <-- 采用" if row["eps"] == DEFAULT_EPS else ""
        say(f"      eps={row['eps']:<5} 簇={row['clusters']:<3} 纯簇={row['pure_clusters']:<3} "
            f"纯度={row['purity_pct']:>5}%{mark}")
    labels = cluster_questions(qs, DEFAULT_EPS)
    n_clusters = len(set(labels))
    groups: dict[int, list[str]] = defaultdict(list)
    for fid, lab in zip(fids, labels):
        groups[lab].append(fid)
    pure = sum(1 for v in groups.values() if len(set(v)) == 1)
    say(f"    采用 eps={DEFAULT_EPS} → {n_clusters} 簇 vs 真值 {len(set(fids))} 个知识点，"
        f"纯度 {pure}/{n_clusters} = {pure / n_clusters * 100:.1f}%")
    sizes = Counter(labels)
    say(f"    min_samples=1（哪怕一条跟谁都不像也单独成簇，不丢素材）")
    say(f"    簇大小分布: " + ", ".join(
        f"{k}簇有{v}条" for k, v in sorted(Counter(sizes.values()).items())))

    for rec, lab in zip(cleaned, labels):
        rec["cluster"] = int(lab)

    train, val, test = split_by_cluster(cleaned, labels, rng)
    say(f"    切分: train {len(train)} / val {len(val)} / test {len(test)}")

    for name, part in (("train", train), ("val", val), ("test", test)):
        to_jsonl(part, DATA_DIR / f"{name}.jsonl")

    # ── 4. 防污染核验 ─────────────────────────────────────────────
    say(f"\n[4] 防污染核验")
    tr_facts = {r["fact_id"] for r in train}
    te_facts = {r["fact_id"] for r in test}
    va_facts = {r["fact_id"] for r in val}
    say(f"    train 覆盖知识点 {len(tr_facts)} 个 / test {len(te_facts)} / val {len(va_facts)}")
    say(f"    test 中「知识点在 train 出现过」的占比: "
        f"{len(te_facts & tr_facts) / max(1, len(te_facts)) * 100:.0f}%")
    say(f"    → 这正是书里要的口径：test 测的是**没见过的问法**，不是没见过的知识。")
    say(f"    ⚠️ 局限：知识级泛化本案例测不了（48 个知识点全都在 train 里出现过）")

    # cluster 是否真的跨集合
    cross = 0
    for lab in set(labels):
        parts = set()
        if any(r["cluster"] == lab for r in train):
            parts.add("train")
        if any(r["cluster"] == lab for r in val):
            parts.add("val")
        if any(r["cluster"] == lab for r in test):
            parts.add("test")
        if len(parts) > 1:
            cross += 1
    say(f"    跨集合的簇: {cross} / {n_clusters}（同题异问被拆开，符合设计要求）")

    # ── 5. 统计 ───────────────────────────────────────────────────
    lens = [len(r["a"]) for r in cleaned]
    cat_dist = Counter(r["cat"] for r in cleaned)
    diff_dist = Counter(r["diff"] for r in cleaned)

    def dist_of(part):
        return {
            "n": len(part),
            "category": dict(Counter(r["cat"] for r in part)),
            "difficulty": dict(Counter(r["diff"] for r in part)),
        }

    stats = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "book_section": "ch11 §11.3.1-11.3.3",
        "seed_facts": len(SEED_FACTS),
        "bad_samples_injected": len(BAD_SAMPLES),
        "question_templates": len(QUESTION_TEMPLATES),
        "raw": len(raw),
        "dropped_by_rules": len(dropped),
        "drop_reasons": dict(drop_reasons),
        "dropped_by_dedup": dup_n,
        "clean": len(cleaned),
        "answer_chars": {
            "min": min(lens), "p50": pct(lens, 50), "p90": pct(lens, 90),
            "max": max(lens), "mean": round(sum(lens) / len(lens), 1),
        },
        "category_dist": dict(cat_dist),
        "difficulty_dist": dict(diff_dist),
        "clusters": n_clusters,
        "eps": DEFAULT_EPS,
        "eps_sweep": sweep,
        "cluster_purity": {
            "pure_clusters": pure, "total_clusters": n_clusters,
            "pct": round(pure / n_clusters * 100, 1),
            "note": "簇内只含一个 fact_id 的比例；有真值才可算",
        },
        "cluster_sizes": dict(Counter(Counter(labels).values())),
        "clusters_spanning_splits": cross,
        "split": {"train": dist_of(train), "val": dist_of(val), "test": dist_of(test)},
        "contamination_check": {
            "test_facts_also_in_train_pct": round(
                len(te_facts & tr_facts) / max(1, len(te_facts)) * 100),
            "note": "test 测未见问法，不测未见知识",
        },
    }
    (OUT_DIR / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    say(f"\n[5] 数据画像")
    say(f"    答案长度: min {min(lens)} / p50 {pct(lens,50)} / p90 {pct(lens,90)} / max {max(lens)} 字")
    say(f"    类别: " + " ".join(f"{k}={v}" for k, v in cat_dist.most_common()))
    say(f"    难度: " + " ".join(f"{k}={v}" for k, v in diff_dist.most_common()))
    say(f"\n落盘:")
    for n in ("train", "val", "test"):
        p = DATA_DIR / f"{n}.jsonl"
        say(f"    {p.relative_to(HERE)}  {p.stat().st_size} 字节 / {sum(1 for _ in p.open())} 行")
    say(f"    outputs/dataset_stats.json")

    (OUT_DIR / "01_prepare.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    print(f"\n日志: outputs/01_prepare.log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
