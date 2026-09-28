import { useCallback, useState } from 'react';

export type VisionDecision = 'PASS' | 'FAIL' | 'REVIEW';

export type VisionLog = {
    id: number;
    request_id: string;
    stage: string;
    field_type: string;
    decision: VisionDecision;
    decision_reason: string;
    ocr_value: string | null;
    raw_ocr_text: string | null;
    expected_value: string | null;
    computed_value: string | null;
    vision_status: 'OK' | 'LOW_CONFIDENCE' | 'ERROR';
    confidence: number | null;
    format_valid: boolean | null;
    error_reason: string | null;
    image_url: string;
    created_at: string | null;
};

// JSON endpoint (not an Inertia visit), so the CSRF token is sent manually from Laravel's
// XSRF-TOKEN cookie, the same token Inertia/axios would send.
function xsrfToken(): string {
    const match = document.cookie.match(/(?:^|;\s*)XSRF-TOKEN=([^;]*)/);
    return match ? decodeURIComponent(match[1]) : '';
}

export function useVisionAnalyze(batchId: number) {
    const [processing, setProcessing] = useState(false);
    const [result, setResult] = useState<VisionLog | null>(null);
    const [error, setError] = useState<string | null>(null);

    const analyze = useCallback(
        async (photo: File, stage: string, fieldType: string): Promise<VisionLog | null> => {
            setProcessing(true);
            setError(null);
            setResult(null);

            const body = new FormData();
            body.append('photo', photo);
            body.append('stage', stage);
            body.append('field_type', fieldType);

            try {
                const response = await fetch(`/batches/${batchId}/vision/analyze`, {
                    method: 'POST',
                    body,
                    credentials: 'same-origin',
                    headers: { Accept: 'application/json', 'X-XSRF-TOKEN': xsrfToken() },
                });
                const data = await response.json();
                if (!response.ok) {
                    setError(data.message ?? `Gagal (HTTP ${response.status})`);
                    return null;
                }
                setResult(data as VisionLog);
                return data as VisionLog;
            } catch {
                setError('Tidak bisa menghubungi server. Coba lagi.');
                return null;
            } finally {
                setProcessing(false);
            }
        },
        [batchId],
    );

    return { analyze, processing, result, error };
}
