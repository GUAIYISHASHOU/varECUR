#!/bin/bash
# 一键生成所有EuRoC序列的VIS训练数据
# 使用严格GT几何监督

EUROC_ROOT="F:/SLAMdata"
OUT_ROOT="F:/SLAMdata/_cache/vis_pairs"

# 确保输出目录存在
mkdir -p "$OUT_ROOT"

# 所有EuRoC序列
SEQUENCES=(
    "V1_01_easy"
    "V1_02_medium"
    "V1_03_difficult"
    "V2_01_easy"
    "V2_02_medium"
    "V2_03_difficult"
    "MH_01_easy"
    "MH_02_easy"
    "MH_03_medium"
    "MH_04_difficult"
    "MH_05_difficult"
)

echo "========================================="
echo "生成VIS训练数据 (严格GT版)"
echo "========================================="
echo "EuRoC Root: $EUROC_ROOT"
echo "Output Root: $OUT_ROOT"
echo "Sequences: ${#SEQUENCES[@]}"
echo "========================================="
echo

# 逐序列生成
for seq in "${SEQUENCES[@]}"; do
    echo "[$(date +%H:%M:%S)] Processing $seq..."
    
    python tools/gen_vis_pairs_euroc_strict.py \
        --euroc_root "$EUROC_ROOT" \
        --seq "$seq" \
        --delta 1 \
        --patch 32 \
        --max_pairs 60000 \
        --out_npz "$OUT_ROOT/${seq}.npz"
    
    if [ $? -eq 0 ]; then
        echo "  ✓ $seq completed"
    else
        echo "  ✗ $seq failed (continuing...)"
    fi
    echo
done

echo "========================================="
echo "所有序列处理完成！"
echo "========================================="
echo
echo "下一步："
echo "python tools/merge_vis_pairs_by_seq.py \\"
echo "  --pairs_root $OUT_ROOT \\"
echo "  --out_root F:/SLAMdata/_cache/vis_split \\"
echo "  --cap_per_seq 60000"

