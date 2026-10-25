# 四分钟讲解提纲

目标是让评委看清「IR 语义怎样变成 PTX」与「并行调度怎样保证正确」。
讲解围绕一条数据流展开，无需依次介绍每个文件。以下是提纲，由参赛者用自己的
语言和声音录制。

## 0:00–0:35：输入与输出

展示一份 Kernel IR 和旁边的 `meta.json`，再展示生成的 PTX。

- Emmy 负责把 JSON 解析成类型化的 IR 节点。
- 我们负责验证参数、维度和启动契约，然后直接生成 PTX。
- 参数顺序和 grid/block 来自 metadata；算子、索引和循环来自 IR。

屏幕上运行：

```sh
python compile.py .reference/problems/p00_scalar_add/k_add_pointwise.ir.json > /tmp/emmy-demo.ptx
```

## 0:35–1:25：先讲共同的 emitter

沿着 `Load → Assign → Write` 跟踪一个值：

1. `Load` 把有界的多维索引展平，计算字节地址，把数据放入 f32 寄存器。
2. `Assign` 根据算子的语义发出 PTX；中间结果保持显式 f32 运算顺序。
3. `Write` 存结果，另一个检查证明输出没有漏写或重复写入。

接着展示一个改 shape 或组合算子的本地回归。解释共用的寄存器分配、算术、
地址与循环逻辑，让同类新输入能复用相同的节点处理。

## 1:25–2:15：再讲并行调度的扩展

画一行数据被线程分段处理的示意图，解释 reduction：

- 每条 lane 执行 `tid + k × block_size` 对应的 Load/Assign/Accum。
- 没有数据的 lane 保留该 reduction 的 identity，仍参加 shuffle 和 barrier。
- warp 内先合并，每个 warp 的一个线程写 shared partial，再进行跨 warp 合并。
- 每个 CTA 对应一行，最后只有指定线程写输出。

共同 emitter 处理数学操作；调度层处理哪个线程执行、怎样通信和同步。
backend dispatch 检查这些 IR 调度节点，不根据题目文件名选择答案。

## 2:15–2:55：选择一个进阶例子

可从以下两项中选自己最熟悉的一项深入讲，避免在短视频里逐文件罗列。

- **Attention**：内层 dot-product state 每轮重置，外层在线 softmax state
  跨轮保留；`Cond` 用整数比较决定执行哪条分支，各分支有独立 SSA 作用域，
  同时可更新外部已初始化的 accumulator。guard 收窄索引边界，输出覆盖检查
  也跟随实际分支。展示 guarded 256-key 或较小矩形数值回归；完整 256×256、
  depth/output width 32 的新增用例目前只验证编译。
- **Matmul**：async copy 把数据搬进 shared memory；等待和 barrier 保证读取
  时数据可见；split-K 的 atomic output 依赖 metadata 的预先清零契约。

## 2:55–3:40：展示证据

先展示 [`RUNNER_RESULTS.md`](RUNNER_RESULTS.md) 的五类结果：公开题目 15/15
通过官方汇编、RTX 5090 执行和原始容差下的数组比较。

再展示本地测试命令和变形测试：改维度、改算子组合、改参数顺序，并保留非法 IR
必须报错的用例。CPU instruction model 用来检查数据流、地址、累加器生命周期和
同步；GPU receipts 记录实际汇编及执行结果。

可以用 m01 的一次汇编失败说明验证的价值：external shared declaration 原来位于
entry 内，修正到 module scope 后官方 runner 通过，失败和成功回执都保留。

## 3:40–4:00：收束

回到开头的数据流：类型化 IR、统一的 scalar emitter、调度扩展、PTX。
用一句话说明证据范围：15 个公开 fixture 有 GPU receipts，新增 shape/op 的结果
以本地回归记录为准。计时只有单次运行，不把它当性能基准。

录制前用当前 checkout 跑本地 suite，并选择已经通过的变形测试作演示；测试数量
和新增能力应以这次实际输出为准。
