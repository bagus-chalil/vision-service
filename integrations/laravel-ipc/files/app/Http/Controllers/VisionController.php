<?php

namespace App\Http\Controllers;

use App\Actions\Vision\AnalyzeVisionField;
use App\Http\Requests\AnalyzeVisionRequest;
use App\Models\IpcBatch;
use App\Models\IpcLog;
use Illuminate\Http\JsonResponse;
use Illuminate\Support\Facades\Storage;

/**
 * JSON (not an Inertia redirect) because the page calls it with fetch and shows the result
 * inline next to the photo field — see resources/js/hooks/use-vision-analyze.ts.
 */
class VisionController extends Controller
{
    public function analyze(AnalyzeVisionRequest $request, IpcBatch $batch, AnalyzeVisionField $action): JsonResponse
    {
        // One OCR call is ~25s and queues behind other stations' calls, which can exceed PHP's
        // default max_execution_time (30s) under php-fpm/Apache.
        set_time_limit(config('vision.timeout') + 30);

        $log = $action->handle(
            $batch,
            $request->user(),
            $request->file('photo'),
            $request->validated('stage'),
            $request->validated('field_type'),
        );

        return response()->json($this->present($log));
    }

    public function index(IpcBatch $batch): JsonResponse
    {
        $logs = IpcLog::query()
            ->where('ipc_batch_id', $batch->id)
            ->latest('id')
            ->get()
            ->map(fn (IpcLog $log) => $this->present($log));

        return response()->json($logs);
    }

    private function present(IpcLog $log): array
    {
        return [
            'id' => $log->id,
            'request_id' => $log->request_id,
            'stage' => $log->stage,
            'field_type' => $log->field_type,
            'decision' => $log->decision,
            'decision_reason' => $log->decision_reason,
            'ocr_value' => $log->ocr_value,
            'raw_ocr_text' => $log->raw_ocr_text,
            'expected_value' => $log->expected_value,
            'computed_value' => $log->computed_value,
            'vision_status' => $log->vision_status,
            'confidence' => $log->confidence,
            'format_valid' => $log->format_valid,
            'error_reason' => $log->error_reason,
            'image_url' => Storage::disk('public')->url($log->image_path),
            'created_at' => $log->created_at?->toIso8601String(),
        ];
    }
}
