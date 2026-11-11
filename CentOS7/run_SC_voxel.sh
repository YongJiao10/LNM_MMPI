#!/bin/bash
# ========== HPC job for DSI Studio SC lesion analysis (List Priority) ==========
#$ -N dsi_studio_list_check
#$ -pe smp 20
#$ -l mem_free=2G,h_vmem=100G
#$ -j y
#$ -q all.q,UI,PINC
#$ -o /Users/yojiao/logs_sc/$JOB_NAME.$JOB_ID.$TASK_ID.out
#$ -t 1-160
#$ -tc 50

# ===============================================================
# [环境加载]
# ===============================================================
module load gcc
module load fsl/6.0.4_gcc-5.4.0

OMP_NUM_THREADS=10

# ---------------- Paths ----------------
# [配置] 两个列表文件的路径
DIR_BASE="/Users/yojiao/SFC_project/LNM_voxel"
MISSING_FILE="${DIR_BASE}/Iowa_sLNM/lesions_missing.txt"
FULL_LIST_FILE="${DIR_BASE}/Iowa_Lesion_MMPI.txt"

# 其他路径
output_dir="${DIR_BASE}/Iowa_sLNM"
lesion_dir=/Shared/boeslab/Data/Lesion/Iowa/Registry/lesionMasks/Chronic/MNI152/2mm

atlas=/Users/yojiao/SFC_project/Schaefer2018_100Parcels_7Networks_order_Tian_Subcortex_S1_MNI152NLin6Asym_2mm.nii.gz
BRAIN_MASK=/Users/yojiao/SFC_project/MNI152_T1_2mm_brain_mask.nii
FIB=/Users/yojiao/SFC_project/HCP1065.1mm.fib.gz
TRACTS=/Users/yojiao/SFC_project/all_tracts.trk
: "${DSI_STUDIO_BIN:=/Shared/pinc/sharedopt/apps/DSI_Studio/Linux/x86_64/20200122/dsi_studio}"

# ---------------- Task ID Check ----------------
if [[ -z "${SGE_TASK_ID:-}" ]]; then
    echo "ERROR: SGE_TASK_ID not set. Submit with qsub -t 1-N" >&2
    exit 1
fi

# ==============================================================================
# [核心逻辑] 列表优先级选择
# ==============================================================================

if [[ -f "${MISSING_FILE}" ]]; then
    # ---------------------------------------------------------
    # 情况 A: 存在 missing 文件 -> 使用 missing 列表
    # ---------------------------------------------------------
    TARGET_LIST="${MISSING_FILE}"
    echo "[INFO] Found missing list: ${MISSING_FILE}"
    echo "[INFO] Running in RE-RUN mode for missing subjects."

elif [[ -f "${FULL_LIST_FILE}" ]]; then
    # ---------------------------------------------------------
    # 情况 B: 不存在 missing 文件 -> 使用 MMPI 总列表
    # ---------------------------------------------------------
    TARGET_LIST="${FULL_LIST_FILE}"
    echo "[INFO] Missing list not found."
    echo "[INFO] Running in NORMAL mode using full list: ${FULL_LIST_FILE}"

else
    # ---------------------------------------------------------
    # 情况 C: 两个列表都不存在 -> 报错退出 (不扫描文件夹)
    # ---------------------------------------------------------
    echo "ERROR: Neither '${MISSING_FILE}' nor '${FULL_LIST_FILE}' could be found." >&2
    echo "       Cannot determine subject list. Exiting." >&2
    exit 1
fi

# ==============================================================================
# [读取受试者 ID]
# ==============================================================================

# 从选定的列表中读取第 N 行
SUBJ_STEM=$(sed -n "${SGE_TASK_ID}p" "${TARGET_LIST}" | xargs)

# 检查是否读取到了 ID (处理空行或 ID 超出范围的情况)
if [[ -z "${SUBJ_STEM}" ]]; then
    echo "ERROR: Retrieved empty ID from line ${SGE_TASK_ID} of ${TARGET_LIST}." >&2
    echo "       Please check if the Task ID range matches the file line count." >&2
    exit 1
fi

# 寻找对应的病灶文件
if [[ -f "${lesion_dir}/${SUBJ_STEM}.nii.gz" ]]; then
    LESION_PATH="${lesion_dir}/${SUBJ_STEM}.nii.gz"
elif [[ -f "${lesion_dir}/${SUBJ_STEM}.nii" ]]; then
    LESION_PATH="${lesion_dir}/${SUBJ_STEM}.nii"
else
    echo "ERROR: Lesion file not found for ID: ${SUBJ_STEM}" >&2
    echo "       Searched in: ${lesion_dir}" >&2
    exit 1
fi

echo "[$(date)] HOST=$(hostname) JOB=${JOB_ID:-unknown} TASK=${SGE_TASK_ID} SUBJECT=${SUBJ_STEM}"
echo "Processing file: ${LESION_PATH}"

# ---------------- Sanity checks ----------------
for f in "${atlas}" "${BRAIN_MASK}" "${FIB}" "${TRACTS}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: required file missing: ${f}" >&2
        exit 1
    fi
done

# ---------------- Prepare output dir & run ----------------
OUTDIR="${output_dir}/${SUBJ_STEM}"
mkdir -p "${OUTDIR}"

# [DSI Studio] --roi=穿过病灶, --save_trk=0 不保存大文件
COMMON_OPTS=( --action=ana
              --source="${FIB}"
              --tract="${TRACTS}"
              --roi="${LESION_PATH}"
              --connectivity="${atlas}"
              --connectivity_type=pass
              --connectivity_threshold=0
              --output="${OUTDIR}/${SUBJ_STEM}_disconnection"
              --save_trk=0
              --export=tdi )

echo ">>> Running DSI Studio..."
"${DSI_STUDIO_BIN}" "${COMMON_OPTS[@]}"

# ==============================================================================
# [FSL Resampling] 1mm -> 2mm
# ==============================================================================
TDI_RAW="${OUTDIR}/${SUBJ_STEM}_disconnection.tdi.nii.gz"
TDI_2MM="${OUTDIR}/${SUBJ_STEM}_disconnection_2mm.nii.gz"

if [[ -f "${TDI_RAW}" ]]; then
    echo ">>> TDI generated. Resampling to 2mm using FSL..."
    
    if ! command -v flirt &> /dev/null; then
        echo "ERROR: 'flirt' not found. Check FSL module." >&2
        exit 1
    fi

    # 生成单位矩阵
    IDENT_MAT="${OUTDIR}/ident_${SUBJ_STEM}.mat"
    echo "1 0 0 0" > "${IDENT_MAT}"
    echo "0 1 0 0" >> "${IDENT_MAT}"
    echo "0 0 1 0" >> "${IDENT_MAT}"
    echo "0 0 0 1" >> "${IDENT_MAT}"

    # 运行 flirt
    flirt -in "${TDI_RAW}" \
          -ref "${BRAIN_MASK}" \
          -out "${TDI_2MM}" \
          -applyxfm -init "${IDENT_MAT}"
          
    rm "${IDENT_MAT}"

    if [[ -f "${TDI_2MM}" ]]; then
        echo ">>> Resampling finished: ${TDI_2MM}"
        rm "${TDI_RAW}" # 可选: 删除原始 1mm 文件
    else
        echo "ERROR: FSL resampling failed." >&2
        exit 1
    fi
else
    echo "ERROR: DSI Studio failed to generate TDI file." >&2
    exit 1
fi

echo "+++ [$(date)] ${SUBJ_STEM} finished."
exit 0