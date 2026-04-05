#!/usr/bin/env bash
# Continue the exact full-quality school run through the resilient Artiverse
# download/preparation entrypoint, then launch only the explicitly selected mode.

set -euo pipefail

REPO=/root/workspace/scenesmith-hts
RUN_META="${REPO}/outputs/preflight/full_quality_school_reference_20260710"
STATUS_FILE="${RUN_META}/full_quality_continuation.status"
LOCK_FILE="${RUN_META}/.full_quality_continuation.lock"
ARTIVERSE_STATUS_FILE="${RUN_META}/artiverse_download_prepare.status"
PIPELINE_STATUS_FILE="${REPO}/outputs/2026-07-10/full_quality_school_reference_sam3d_artvip_artiverse_20260710/pipeline.status"
MODE="${SCENESMITH_CONTINUATION_MODE:-full}"
POLL_SECONDS="${SCENESMITH_CONTINUATION_POLL_SECONDS:-30}"

case "${MODE}" in
    full|benchmark_classroom_01) ;;
    *)
        echo "Unsupported SCENESMITH_CONTINUATION_MODE=${MODE@Q}" >&2
        exit 2
        ;;
esac
if ! [[ "${POLL_SECONDS}" =~ ^[0-9]+$ ]] || (( POLL_SECONDS < 5 )); then
    echo "SCENESMITH_CONTINUATION_POLL_SECONDS must be an integer >= 5." >&2
    exit 2
fi

mkdir -p "${RUN_META}"
cd "${REPO}"
exec 8>"${LOCK_FILE}"
if ! flock -n 8; then
    echo "Another automatic continuation holds ${LOCK_FILE}." >&2
    exit 75
fi
phase=initializing
trap 'code=$?; if (( code != 0 )); then printf "failed phase=%s exit=%s %s\n" "${phase}" "${code}" "$(date -Is)" >"${STATUS_FILE}"; fi' EXIT

artiverse_complete() {
    [[ -s "${ARTIVERSE_STATUS_FILE}" ]] && \
        grep -Eq '^complete exit=0 ' "${ARTIVERSE_STATUS_FILE}"
}

artiverse_worker_running() {
    pgrep -f '[d]ownload_prepare_artiverse_sqz\.sh|[s]afe_extract_artiverse\.py|[p]repare_artiverse\.py' \
        >/dev/null
}

if artiverse_worker_running; then
    phase=waiting_for_existing_artiverse
    printf 'waiting_for_existing_artiverse mode=%s %s\n' \
        "${MODE}" "$(date -Is)" >"${STATUS_FILE}"
    while artiverse_worker_running; do
        sleep "${POLL_SECONDS}"
    done
    if ! artiverse_complete; then
        echo "Existing Artiverse worker ended without an exact complete exit=0 status." >&2
        exit 21
    fi
elif ! artiverse_complete; then
    phase=preparing_artiverse
    printf 'preparing_artiverse mode=%s %s\n' \
        "${MODE}" "$(date -Is)" >"${STATUS_FILE}"
    ARTIVERSE_APPROVE_FULL_DOWNLOAD=1 \
        bash remote_jobs/download_prepare_artiverse_sqz.sh
    if ! artiverse_complete; then
        echo "Artiverse preparation returned without an exact complete exit=0 status." >&2
        exit 22
    fi
fi

phase=waiting_for_idle_gpu
printf 'waiting_for_idle_gpu mode=%s %s\n' \
    "${MODE}" "$(date -Is)" >"${STATUS_FILE}"
while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    2>/dev/null | grep -Eq '^[[:space:]]*[0-9]+'; do
    sleep "${POLL_SECONDS}"
done

case "${MODE}" in
    benchmark_classroom_01)
        phase=benchmark_classroom_01
        printf 'starting_benchmark_classroom_01 %s\n' "$(date -Is)" >"${STATUS_FILE}"
        SCENESMITH_EXECUTION_MODE=benchmark_classroom_01 \
            bash remote_jobs/run_full_quality_school_sqz.sh
        if ! [[ -s "${PIPELINE_STATUS_FILE}" ]] || \
            ! grep -Eq '^benchmark_complete exit=0 ' "${PIPELINE_STATUS_FILE}"; then
            echo "Benchmark runner returned without benchmark_complete exit=0." >&2
            exit 23
        fi
        ;;
    full)
        phase=full_quality_pipeline
        printf 'starting_full_quality_pipeline %s\n' "$(date -Is)" >"${STATUS_FILE}"
        bash remote_jobs/run_full_quality_school_sqz.sh
        ;;
esac

printf 'pipeline_command_complete mode=%s %s\n' "${MODE}" "$(date -Is)" >"${STATUS_FILE}"
trap - EXIT
